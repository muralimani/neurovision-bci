"""Mental-state estimation from band-power features.

Two layers:
  1. A nearest-centroid classifier over a small feature vector. Default centroids
     ship in default_prototypes.json (derived from the synthetic profiles). Real
     EEG varies a lot between people and electrode setups, so run
     ``python -m neurovision calibrate`` once the PiEEG arrives to record your own
     centroids; they are used automatically when present.
  2. Temporal smoothing (EMA) + hysteresis so the "current state" only changes
     when a new state has clearly won for a few seconds.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dsp import FEATURE_NAMES, feature_vector

DEFAULT_PROTOTYPES = Path(__file__).with_name("default_prototypes.json")


@dataclass
class Prototypes:
    centroids: dict[str, np.ndarray]
    scale: np.ndarray
    source: str = "default"

    @classmethod
    def load(cls, personal_path: str | Path | None = None) -> "Prototypes":
        for path, tag in ((personal_path, "personal"), (DEFAULT_PROTOTYPES, "default")):
            if path and Path(path).exists():
                data = json.loads(Path(path).read_text())
                cents = {k: np.array(v["mean"]) for k, v in data["states"].items() if v.get("mean")}
                if len(cents) >= 2:  # a personal file needs at least two calibrated states
                    return cls(cents, np.array(data["scale"]), tag)
        raise FileNotFoundError("No prototype file found; run `python -m neurovision build-prototypes`.")

    @staticmethod
    def build(samples: dict[str, list[np.ndarray]]) -> dict:
        """Build a prototype JSON dict from labelled feature vectors."""
        all_vecs = np.vstack([v for vs in samples.values() for v in vs])
        scale = np.maximum(all_vecs.std(axis=0), 0.05)
        return {
            "feature_names": FEATURE_NAMES,
            "scale": scale.round(5).tolist(),
            "states": {
                s: {"mean": np.mean(vs, axis=0).round(5).tolist(), "n": len(vs)}
                for s, vs in samples.items() if vs
            },
        }


@dataclass
class StateEstimate:
    state: str
    confidence: float
    probabilities: dict[str, float]
    stable_state: str
    changed: bool


@dataclass
class StateTracker:
    prototypes: Prototypes
    smoothing: float = 0.25
    stable_sec: float = 3.0
    temperature: float = 0.3  # lower = more decisive probabilities
    _probs: dict[str, float] = field(default_factory=dict)
    _stable: str | None = None
    _candidate: str | None = None
    _candidate_since: float = 0.0

    def classify(self, features: dict) -> dict[str, float]:
        x = feature_vector(features)
        d = {s: float(np.sqrt((((x - c) / self.prototypes.scale) ** 2).mean()))
             for s, c in self.prototypes.centroids.items()}
        logits = np.array([-v / self.temperature for v in d.values()])
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return dict(zip(d.keys(), p.round(4).tolist()))

    def update(self, features: dict, now: float | None = None) -> StateEstimate:
        now = time.monotonic() if now is None else now
        raw = self.classify(features)
        if not self._probs:
            self._probs = dict(raw)
        else:
            a = self.smoothing
            self._probs = {s: (1 - a) * self._probs.get(s, 0) + a * raw[s] for s in raw}
        top = max(self._probs, key=self._probs.get)
        changed = False
        if self._stable is None:
            self._stable, changed = top, True
        elif top != self._stable:
            if top != self._candidate:
                self._candidate, self._candidate_since = top, now
            elif now - self._candidate_since >= self.stable_sec:
                self._stable, self._candidate, changed = top, None, True
        else:
            self._candidate = None
        return StateEstimate(top, round(self._probs[top], 3),
                             {k: round(v, 3) for k, v in self._probs.items()},
                             self._stable, changed)
