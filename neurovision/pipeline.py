"""The live loop: source -> filter -> features -> state -> prompt -> Gemini."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import AppConfig
from .dsp import Preprocessor, extract_features
from .prompts import build_prompt, llm_prompt_request
from .sources import make_source
from .states import Prototypes, StateTracker

log = logging.getLogger(__name__)


class RingBuffer:
    def __init__(self, n: int, ch: int):
        self.buf = np.zeros((n, ch))
        self.filled = 0

    def push(self, x: np.ndarray):
        k = len(x)
        if k >= len(self.buf):
            self.buf[:] = x[-len(self.buf):]
        else:
            self.buf = np.roll(self.buf, -k, axis=0)
            self.buf[-k:] = x
        self.filled = min(len(self.buf), self.filled + k)

    @property
    def full(self):
        return self.filled >= len(self.buf)


class NeuroVision:
    def __init__(self, cfg: AppConfig, realtime: bool = True, max_generations: int | None = None):
        self.cfg = cfg
        self.realtime = realtime
        self.max_generations = max_generations
        self.source = make_source(cfg, realtime)
        e = cfg.eeg
        self.n_ch = len(e.channel_labels)
        self.pre = Preprocessor(e.sample_rate, self.n_ch, e.mains_hz)
        self.ring = RingBuffer(int(e.window_sec * e.sample_rate), self.n_ch)
        self.tracker = StateTracker(Prototypes.load(cfg.profiles_path), e.smoothing, e.stable_sec)
        log.info("state prototypes: %s", self.tracker.prototypes.source)
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.studio = None
        if not cfg.dry_run:
            if not cfg.api_key:
                raise SystemExit(f"Set {cfg.gemini.api_key_env} (or use --dry-run to only write prompts).")
            from .gemini import GeminiStudio
            g = cfg.gemini
            self.studio = GeminiStudio(cfg.api_key, g.image_model, g.video_model, g.text_model, g.aspect_ratio)
        self._busy = False
        self._last_gen = -1e9
        self.generations = 0
        self.history: list[dict] = []
        self._ema: dict[str, float] = {}

    # --------------------------------------------------------------- trigger
    def _should_generate(self, est, now) -> bool:
        t = self.cfg.trigger
        if self._busy or now - self._last_gen < t.cooldown_sec:
            return False
        if t.mode == "interval":
            return now - self._last_gen >= t.interval_sec
        if self.generations == 0 and est.confidence >= t.min_confidence:
            return True
        if est.changed and est.confidence >= t.min_confidence:
            return True
        return t.interval_sec > 0 and now - self._last_gen >= t.interval_sec

    async def _generate(self, state: str, features: dict):
        self._busy = True
        g = self.cfg.gemini
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = self.out_dir / f"{stamp}_{state}"
        record = {"time": stamp, "state": state, "features": features}
        try:
            if g.prompt_mode == "llm" and self.studio:
                prompt = await asyncio.to_thread(self.studio.write_prompt,
                                                 llm_prompt_request(state, features, g.style))
            else:
                prompt = build_prompt(state, features, g.style)
            record["image_prompt"] = prompt
            log.info("PROMPT [%s]: %s", state, prompt)
            if self.studio:
                img = await asyncio.to_thread(self.studio.generate_image, prompt, base.with_suffix(".png"))
                record["image"] = img.name
                log.info("image saved: %s", img)
                if g.make_video:
                    vprompt = build_prompt(state, features, g.style, for_video=True)
                    record["video_prompt"] = vprompt
                    vid = await asyncio.to_thread(self.studio.generate_video, vprompt,
                                                  base.with_suffix(".mp4"),
                                                  img if g.video_from_image else None)
                    record["video"] = vid.name
                    log.info("video saved: %s", vid)
        except Exception as e:  # noqa: BLE001 - keep the live loop alive
            record["error"] = str(e)
            log.error("generation failed: %s", e)
        finally:
            base.with_suffix(".json").write_text(json.dumps(record, indent=2))
            self.history.append(record)
            self.generations += 1
            self._busy = False

    # ------------------------------------------------------------------ loop
    async def run(self, duration: float | None = None):
        e = self.cfg.eeg
        step = int(e.update_sec * e.sample_rate)
        since = 0
        samples_seen = 0
        tasks = set()
        start = time.monotonic()
        async for ts, x in self.source.stream():
            self.ring.push(self.pre(x))
            since += len(x)
            samples_seen += len(x)
            # use sample time when not realtime so tests run fast but behave the same
            now = (time.monotonic() - start) if self.realtime else samples_seen / e.sample_rate
            if duration and now >= duration:
                break
            if not self.ring.full or since < step:
                continue
            since = 0
            feats = extract_features(self.ring.buf, e.sample_rate, e.channel_labels)
            # Asymmetry from two electrodes is noisy: smooth arousal / valence over time
            for key in ("arousal", "valence"):
                prev = self._ema.get(key, feats[key])
                self._ema[key] = 0.8 * prev + 0.2 * feats[key]
                feats[key] = round(self._ema[key], 3)
            est = self.tracker.update(feats, now)
            truth = getattr(self.source, "current_truth", None)
            r = feats["relative"]
            log.info("%-10s p=%.2f stable=%-10s | δ%.2f θ%.2f α%.2f β%.2f γ%.2f | arousal %.2f valence %+.2f%s",
                     est.state, est.confidence, est.stable_state, r["delta"], r["theta"], r["alpha"],
                     r["beta"], r["gamma"], feats["arousal"], feats["valence"],
                     f" | truth={truth}" if truth else "")
            if self._should_generate(est, now):
                self._last_gen = now
                task = asyncio.create_task(self._generate(est.stable_state, feats))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            if self.max_generations and self.generations >= self.max_generations and not self._busy:
                break
        if tasks:
            await asyncio.gather(*tasks)
