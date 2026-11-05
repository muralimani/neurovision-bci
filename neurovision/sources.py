"""EEG data sources. All yield (timestamps[n], samples[n, channels]) in microvolts.

The rest of the program only depends on this interface, which is what makes the
switch from synthetic data to the real PiEEG a one-word config change:

    synthetic  -> in-process generator (no network)
    pieeg      -> pieeg-server WebSocket (real board, `pieeg-server --mock`,
                  or our state-driven fake server: `python -m neurovision fake-server`)
    csv        -> replay a recording (pieeg-server CSV export or our synthetic CSVs)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from typing import AsyncIterator

import numpy as np

from .synthetic import SyntheticEEG

log = logging.getLogger(__name__)
Chunk = tuple[np.ndarray, np.ndarray]


def parse_scenario(text: str) -> list[tuple[str, float]]:
    """"relaxed:30,focused:20" -> [("relaxed", 30.0), ("focused", 20.0)]"""
    out = []
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, _, secs = item.partition(":")
        out.append((name.strip(), float(secs or 30)))
    return out


class SyntheticSource:
    def __init__(self, fs: int, labels: list[str], mains_hz: float, state: str = "relaxed",
                 scenario: str = "", seed: int | None = None, realtime: bool = True,
                 chunk: int = 25):
        self.gen = SyntheticEEG(fs, labels, mains_hz, seed)
        self.gen.set_state(state, immediate=True)
        self.scenario = parse_scenario(scenario) if scenario else []
        self.fs, self.chunk, self.realtime = fs, chunk, realtime
        self.n_channels = len(labels)
        self.current_truth = state

    async def stream(self) -> AsyncIterator[Chunk]:
        n = 0
        t0 = time.time()
        sched_idx, sched_left = 0, (self.scenario[0][1] if self.scenario else None)
        if self.scenario:
            self.gen.set_state(self.scenario[0][0], immediate=True)
            self.current_truth = self.scenario[0][0]
        while True:
            data = self.gen.next_chunk(self.chunk)
            ts = t0 + (n + np.arange(self.chunk)) / self.fs
            n += self.chunk
            yield ts, data
            if self.scenario:
                sched_left -= self.chunk / self.fs
                if sched_left <= 0:
                    sched_idx = (sched_idx + 1) % len(self.scenario)
                    st, sched_left = self.scenario[sched_idx]
                    self.gen.set_state(st)
                    self.current_truth = st
                    log.info("synthetic source -> %s", st)
            if self.realtime:
                delay = t0 + n / self.fs - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)


class PiEEGSource:
    """Client for pieeg-server's WebSocket API (default ws://<pi>:1616).

    Frame format (docs.pieeg.com/software/api/data-format):
        {"t": 1711234567.123, "n": 42, "channels": [12.34, -5.67, ...]}   # uV, 250 Hz
    The first message is a welcome dict with "sample_rate" and "channels".
    """

    def __init__(self, url: str, token: str = "", n_channels: int = 8, chunk: int = 25):
        self.url = url + (("&" if "?" in url else "?") + f"token={token}" if token else "")
        self.n_channels = n_channels
        self.chunk = chunk
        self.sample_rate = None
        self.last_n = None
        self.dropped = 0

    async def stream(self) -> AsyncIterator[Chunk]:
        import websockets

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url, max_size=2**22, ping_interval=20) as ws:
                    log.info("connected to %s", self.url)
                    backoff = 1.0
                    buf_t, buf_x = [], []
                    async for msg in ws:
                        obj = json.loads(msg)
                        frames = obj if isinstance(obj, list) else [obj]
                        for fr in frames:
                            if "channels" in fr and isinstance(fr["channels"], list) and "t" in fr:
                                ch = fr["channels"]
                                if self.last_n is not None and fr.get("n", 0) > self.last_n + 1:
                                    self.dropped += fr["n"] - self.last_n - 1
                                self.last_n = fr.get("n", self.last_n)
                                row = (ch + [0.0] * self.n_channels)[: self.n_channels]
                                buf_t.append(fr["t"])
                                buf_x.append(row)
                            elif fr.get("status") == "connected":
                                self.sample_rate = fr.get("sample_rate")
                                srv_ch = fr.get("channels")
                                log.info("pieeg-server: %s Hz, %s channels", self.sample_rate, srv_ch)
                                if srv_ch and srv_ch != self.n_channels:
                                    log.warning("Server reports %s channels but %s are configured; using the "
                                                "first %s. Start pieeg-server with the PiEEG-8 device option "
                                                "or edit channel_labels in config.toml.",
                                                srv_ch, self.n_channels, self.n_channels)
                        if len(buf_x) >= self.chunk:
                            yield np.asarray(buf_t), np.asarray(buf_x, dtype=float)
                            buf_t, buf_x = [], []
            except (OSError, asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
                if isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
                    raise
                log.warning("PiEEG connection problem (%s); retrying in %.0fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)


class CSVSource:
    """Replay a CSV recording. Uses columns named ch1..chN / Fp1.. if present,
    otherwise the last N numeric columns."""

    def __init__(self, path: str, fs: int, labels: list[str], realtime: bool = True,
                 chunk: int = 25, loop: bool = True):
        self.path, self.fs, self.labels = path, fs, labels
        self.realtime, self.chunk, self.loop = realtime, chunk, loop
        self.n_channels = len(labels)
        self.data = self._load()

    def _load(self) -> np.ndarray:
        with open(self.path, newline="") as f:
            rows = list(csv.reader(f))
        header = [h.strip() for h in rows[0]]
        body = rows[1:]
        lower = [h.lower() for h in header]
        wanted = [l.lower() for l in self.labels]
        if all(w in lower for w in wanted):
            idx = [lower.index(w) for w in wanted]
        elif all(f"ch{i+1}" in lower for i in range(self.n_channels)):
            idx = [lower.index(f"ch{i+1}") for i in range(self.n_channels)]
        else:
            idx = list(range(len(header) - self.n_channels, len(header)))
        return np.array([[float(r[i]) for i in idx] for r in body if r])

    async def stream(self) -> AsyncIterator[Chunk]:
        i, t0 = 0, time.time()
        while True:
            if i >= len(self.data):
                if not self.loop:
                    return
                i = 0
            x = self.data[i:i + self.chunk]
            ts = t0 + (np.arange(len(x)) + i) / self.fs
            i += len(x)
            yield ts, x
            await asyncio.sleep(len(x) / self.fs if self.realtime else 0)


def make_source(cfg, realtime: bool = True):
    s, e = cfg.source, cfg.eeg
    if s.kind == "synthetic":
        return SyntheticSource(e.sample_rate, e.channel_labels, e.mains_hz, s.synthetic_state,
                               s.synthetic_scenario, s.seed, realtime)
    if s.kind == "pieeg":
        return PiEEGSource(s.pieeg_url, s.pieeg_token, len(e.channel_labels))
    if s.kind == "csv":
        return CSVSource(s.csv_path, e.sample_rate, e.channel_labels, realtime)
    raise ValueError(f"Unknown source kind '{s.kind}'")
