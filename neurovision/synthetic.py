"""Synthetic multichannel EEG that mimics different mental states.

The model is deliberately physiological rather than just "add a sine wave":
  * every band is band-limited noise around a centre frequency (real rhythms
    drift and wax/wane), with a slow amplitude envelope (alpha "bursts");
  * each band has a scalp topography (alpha strongest at the back of the head,
    theta and beta at the front, etc.);
  * a 1/f "pink" background, mains hum and occasional eye-blink artefacts on the
    frontal channels are added, so the downstream DSP has to cope with the same
    junk real electrodes pick up;
  * frontal alpha asymmetry encodes valence (left-frontal activation = approach /
    positive, right = withdrawal / stress).

Amplitudes are in microvolts, matching what pieeg-server streams.
These are illustrative profiles for development, not a validated neuro model.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import BANDS, STATES

# Relative strength of each band per scalp region (frontal, central, parietal, occipital)
BAND_TOPOGRAPHY = {
    "delta": (1.0, 0.9, 0.8, 0.8),
    "theta": (1.0, 0.85, 0.7, 0.6),
    "alpha": (0.35, 0.6, 0.9, 1.0),
    "beta": (1.0, 1.0, 0.7, 0.55),
    "gamma": (1.0, 0.8, 0.6, 0.55),
}

# Per state: band -> (centre Hz, RMS microvolts at topography weight 1.0)
# plus frontal alpha asymmetry (+ = less LEFT alpha = left activation = positive valence)
# and blink rate (blinks per minute).
STATE_PROFILES: dict[str, dict] = {
    "relaxed": {  # calm, eyes closed / soft gaze: dominant posterior alpha
        "bands": {"delta": (2.0, 6), "theta": (6.0, 6), "alpha": (10.0, 28), "beta": (17.0, 5), "gamma": (40.0, 1.5)},
        "asymmetry": 0.1, "blinks_per_min": 2,
        "description": "Calm wakefulness, eyes closed: strong 10 Hz posterior alpha, little beta.",
    },
    "focused": {  # concentrated mental work: alpha suppressed, beta up, frontal theta up
        "bands": {"delta": (2.0, 6), "theta": (6.5, 10), "alpha": (11.0, 7), "beta": (20.0, 14), "gamma": (40.0, 3)},
        "asymmetry": 0.05, "blinks_per_min": 8,
        "description": "Active concentration: alpha desynchronised, 15-25 Hz beta and frontal-midline theta elevated.",
    },
    "meditative": {  # deep meditation: theta + slow alpha, very little beta
        "bands": {"delta": (2.5, 8), "theta": (6.0, 18), "alpha": (9.0, 20), "beta": (16.0, 4), "gamma": (40.0, 1.5)},
        "asymmetry": 0.15, "blinks_per_min": 1,
        "description": "Meditation: frontal theta and slowed alpha coexist, beta is quiet.",
    },
    "drowsy": {  # falling asleep: delta/theta take over, alpha slows and fades
        "bands": {"delta": (2.0, 28), "theta": (5.0, 18), "alpha": (8.5, 7), "beta": (15.0, 4), "gamma": (40.0, 1.0)},
        "asymmetry": 0.0, "blinks_per_min": 3,
        "description": "Drowsiness / sleep onset: large slow delta and theta, alpha dropping out.",
    },
    "stressed": {  # anxious / under pressure: high beta, low alpha, right-frontal activation
        "bands": {"delta": (2.0, 6), "theta": (6.0, 7), "alpha": (11.0, 6), "beta": (25.0, 18), "gamma": (40.0, 5)},
        "asymmetry": -0.35, "blinks_per_min": 20,
        "description": "Stress / anxiety: high-frequency beta (20-30 Hz), suppressed alpha, right-frontal activation.",
    },
    "creative": {  # daydreaming / ideation: alpha + theta with some beta
        "bands": {"delta": (2.0, 6), "theta": (6.5, 14), "alpha": (10.0, 18), "beta": (16.0, 9), "gamma": (40.0, 3)},
        "asymmetry": 0.15, "blinks_per_min": 6,
        "description": "Creative ideation / daydreaming: alpha and theta both strong, moderate beta.",
    },
    "excited": {  # joy / high positive arousal: beta + gamma, left-frontal activation
        "bands": {"delta": (2.0, 5), "theta": (6.0, 7), "alpha": (10.5, 9), "beta": (20.0, 15), "gamma": (40.0, 8)},
        "asymmetry": 0.35, "blinks_per_min": 15,
        "description": "Excitement / joy: beta and gamma elevated with left-frontal activation (positive valence).",
    },
}

REGION_INDEX = {"F": 0, "C": 1, "P": 2, "O": 3, "T": 1}


def channel_region(label: str) -> int:
    """Map a 10-20 label (Fp1, F3, Cz, P4, O2, T7...) to a region index."""
    head = label[0].upper()
    return REGION_INDEX.get(head, 1)


def channel_side(label: str) -> int:
    """-1 = left (odd number), +1 = right (even), 0 = midline (z)."""
    tail = label.rstrip().lower()[-1]
    if tail == "z":
        return 0
    if tail.isdigit():
        return -1 if int(tail) % 2 else 1
    return 0


# Paul Kellet's economy pink-noise filter (approximate 1/f)
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])


class SyntheticEEG:
    """Stateful, streamable generator. Call ``next_chunk(n)`` repeatedly."""

    def __init__(self, fs: int = 250, channel_labels: list[str] | None = None,
                 mains_hz: float = 50.0, seed: int | None = None,
                 transition_sec: float = 3.0, artifacts: bool = True):
        self.fs = fs
        self.labels = channel_labels or ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
        self.n_ch = len(self.labels)
        self.mains_hz = mains_hz
        self.rng = np.random.default_rng(seed)
        self.artifacts = artifacts
        self.t = 0  # sample counter
        self._alpha_per_sample = 1.0 / max(transition_sec * fs, 1)

        self._regions = np.array([channel_region(l) for l in self.labels])
        self._sides = np.array([channel_side(l) for l in self.labels])
        self._frontal = self._regions == 0

        # Per-band filter + state, one bank per band. Centre frequency comes from the
        # current state, so we design filters for every (state, band) pair up front.
        self._sos: dict[tuple[str, str], np.ndarray] = {}
        for st, prof in STATE_PROFILES.items():
            for band, (fc, _) in prof["bands"].items():
                bw = max(1.5, 0.25 * fc)
                lo, hi = max(0.3, fc - bw / 2), min(fs / 2 - 1, fc + bw / 2)
                self._sos[(st, band)] = signal.butter(2, [lo, hi], btype="band", fs=fs, output="sos")
        self._zi: dict[tuple[str, str], np.ndarray] = {
            k: np.zeros((sos.shape[0], 2, self.n_ch)) for k, sos in self._sos.items()
        }
        self._pink_zi = np.zeros((len(_PINK_A) - 1, self.n_ch))
        self._env_sos = signal.butter(1, 0.4, btype="low", fs=fs, output="sos")
        self._env_zi = np.zeros((self._env_sos.shape[0], 2, len(BANDS)))
        self._blink_left = 0
        self._blink_shape = np.array([])

        self.weights = {s: 0.0 for s in STATES}
        self.target = "relaxed"
        self.weights[self.target] = 1.0

    # ------------------------------------------------------------------ control
    def set_state(self, state: str, immediate: bool = False):
        if state not in STATE_PROFILES:
            raise ValueError(f"Unknown state '{state}'. Choose from {list(STATE_PROFILES)}")
        self.target = state
        if immediate:
            self.weights = {s: float(s == state) for s in STATES}

    # ---------------------------------------------------------------- generator
    def _unit_band(self, key, n):
        """Band-limited noise with unit RMS per channel (approximately)."""
        sos = self._sos[key]
        white = self.rng.standard_normal((n, self.n_ch))
        y, self._zi[key] = signal.sosfilt(sos, white, axis=0, zi=self._zi[key])
        return y

    def next_chunk(self, n: int) -> np.ndarray:
        """Return an (n, n_channels) array of microvolts."""
        # Cross-fade weights towards the target state
        w_hist = np.empty((n, len(STATES)))
        for i in range(n):
            for s in STATES:
                goal = 1.0 if s == self.target else 0.0
                self.weights[s] += (goal - self.weights[s]) * self._alpha_per_sample * 4
            w_hist[i] = [self.weights[s] for s in STATES]
        w_hist /= w_hist.sum(axis=1, keepdims=True)

        # Slow amplitude envelopes per band (waxing / waning rhythms)
        env_noise = self.rng.standard_normal((n, len(BANDS))) * 6.0
        env, self._env_zi = signal.sosfilt(self._env_sos, env_noise, axis=0, zi=self._env_zi)
        env = np.clip(1.0 + 0.35 * env, 0.2, 2.0)

        out = np.zeros((n, self.n_ch))
        for si, st in enumerate(STATES):
            ws = w_hist[:, si]
            if ws.max() < 1e-3:
                # keep filter states warm cheaply: skip, the cross-fade is slow anyway
                continue
            prof = STATE_PROFILES[st]
            asym = prof["asymmetry"]
            for bi, (band, (fc, rms)) in enumerate(prof["bands"].items()):
                y = self._unit_band((st, band), n)
                # normalise band noise to unit RMS
                bw = max(1.5, 0.25 * fc)
                y *= np.sqrt(self.fs / 2 / bw) * 1.1
                topo = np.array([BAND_TOPOGRAPHY[band][r] for r in self._regions])
                if band == "alpha" and asym:
                    # positive asym => left frontal alpha reduced (left activation)
                    side_gain = 1.0 - asym * (-self._sides) * self._frontal
                    topo = topo * np.clip(side_gain, 0.3, 1.7)
                out += y * topo * rms * env[:, [bi]] * ws[:, None]

        # 1/f background
        pink, self._pink_zi = signal.lfilter(_PINK_B, _PINK_A,
                                             self.rng.standard_normal((n, self.n_ch)),
                                             axis=0, zi=self._pink_zi)
        out += pink * 6.0

        # Mains hum (small, slightly different per channel) + sensor noise
        tt = (self.t + np.arange(n)) / self.fs
        out += 1.5 * np.sin(2 * np.pi * self.mains_hz * tt)[:, None] * (0.5 + self.rng.random(self.n_ch))
        out += self.rng.standard_normal((n, self.n_ch)) * 1.0

        if self.artifacts:
            out += self._blinks(n, w_hist)

        self.t += n
        return out

    def _blinks(self, n, w_hist):
        art = np.zeros((n, self.n_ch))
        rate = sum(w_hist[0, i] * STATE_PROFILES[s]["blinks_per_min"] for i, s in enumerate(STATES))
        p_start = rate / 60.0 / self.fs
        frontal_gain = np.where(self._regions == 0, 1.0, 0.08)
        i = 0
        while i < n:
            if self._blink_left > 0:
                k = min(self._blink_left, n - i)
                start = len(self._blink_shape) - self._blink_left
                art[i:i + k] += self._blink_shape[start:start + k, None] * frontal_gain
                self._blink_left -= k
                i += k
                continue
            if self.rng.random() < p_start * min(n - i, 50):
                dur = int(self.fs * self.rng.uniform(0.2, 0.4))
                amp = self.rng.uniform(80, 180)
                self._blink_shape = amp * np.hanning(dur)
                self._blink_left = dur
            i += min(n - i, 50)
        return art


def generate_recording(state_or_scenario, seconds: float | None = None, fs: int = 250,
                       channel_labels=None, mains_hz: float = 50.0, seed: int | None = 0):
    """Convenience: return (timestamps, data, state_per_sample) for a whole recording.

    ``state_or_scenario`` is either a state name or a list of (state, seconds).
    """
    if isinstance(state_or_scenario, str):
        scenario = [(state_or_scenario, seconds or 60)]
    else:
        scenario = list(state_or_scenario)
    gen = SyntheticEEG(fs, channel_labels, mains_hz, seed)
    gen.set_state(scenario[0][0], immediate=True)
    chunks, labels = [], []
    for st, dur in scenario:
        gen.set_state(st)
        n = int(dur * fs)
        chunks.append(gen.next_chunk(n))
        labels += [st] * n
    data = np.vstack(chunks)
    t = np.arange(len(data)) / fs
    return t, data, labels
