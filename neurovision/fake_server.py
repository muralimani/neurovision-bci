"""A stand-in for pieeg-server that streams *state-driven* synthetic EEG.

It speaks the same WebSocket protocol as the real server (port 1616, a welcome
message, then {"t","n","channels"} frames at 250 Hz), so the app's `pieeg`
source code path is exercised end to end before the hardware arrives.

Extra command (not in the real server), handy for demos:
    {"cmd": "set_state", "state": "focused"}
Real-server commands such as set_filter are accepted and acknowledged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from .sources import parse_scenario
from .synthetic import STATE_PROFILES, SyntheticEEG

log = logging.getLogger(__name__)


class FakePiEEGServer:
    def __init__(self, fs=250, labels=None, mains_hz=50.0, state="relaxed", scenario="",
                 host="0.0.0.0", port=1616, seed=None):
        self.gen = SyntheticEEG(fs, labels, mains_hz, seed)
        self.gen.set_state(state, immediate=True)
        self.fs, self.host, self.port = fs, host, port
        self.scenario = parse_scenario(scenario) if scenario else []
        self.clients: set = set()
        self.n = 0

    def welcome(self):
        return {"status": "connected", "sample_rate": self.fs, "channels": self.gen.n_ch,
                "filter": False, "recording": False,
                "engine": {"native": False, "engine": "neurovision-fake", "version": None},
                "synthetic_state": self.gen.target}

    async def handler(self, ws):
        self.clients.add(ws)
        log.info("client connected (%d total)", len(self.clients))
        try:
            await ws.send(json.dumps(self.welcome()))
            async for msg in ws:
                try:
                    cmd = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if cmd.get("cmd") == "set_state" and cmd.get("state") in STATE_PROFILES:
                    self.gen.set_state(cmd["state"])
                    log.info("state -> %s", cmd["state"])
                await ws.send(json.dumps({"ack": cmd.get("cmd"), "synthetic_state": self.gen.target}))
        except Exception:  # noqa: BLE001 - client disconnected
            pass
        finally:
            self.clients.discard(ws)

    async def producer(self):
        block = 10  # samples per tick (40 ms at 250 Hz)
        t0 = time.time()
        sched_i, left = 0, (self.scenario[0][1] if self.scenario else 0)
        if self.scenario:
            self.gen.set_state(self.scenario[0][0], immediate=True)
        while True:
            x = self.gen.next_chunk(block)
            if self.clients:
                msgs = []
                for row in x:
                    msgs.append(json.dumps({"t": round(t0 + self.n / self.fs, 6), "n": self.n,
                                            "channels": [round(float(v), 3) for v in row]}))
                    self.n += 1
                for ws in list(self.clients):
                    try:
                        for m in msgs:
                            await ws.send(m)
                    except Exception:  # noqa: BLE001 - client went away
                        self.clients.discard(ws)
            else:
                self.n += block
            if self.scenario:
                left -= block / self.fs
                if left <= 0:
                    sched_i = (sched_i + 1) % len(self.scenario)
                    st, left = self.scenario[sched_i]
                    self.gen.set_state(st)
                    log.info("scenario -> %s", st)
            delay = t0 + self.n / self.fs - time.time()
            await asyncio.sleep(max(delay, 0))

    async def run(self):
        import websockets

        async with websockets.serve(self.handler, self.host, self.port):
            log.info("fake pieeg-server on ws://%s:%d (state=%s)", self.host, self.port, self.gen.target)
            await self.producer()
