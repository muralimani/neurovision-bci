"""Signal processing: filtering and band-power feature extraction."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

from . import BANDS
from .synthetic import channel_region, channel_side


class Preprocessor:
    """Streaming band-pass (1-45 Hz) + mains notch, applied chunk by chunk."""

    def __init__(self, fs: int, n_ch: int, mains_hz: float = 50.0, band=(1.0, 45.0)):
        self.bp = signal.butter(4, band, btype="band", fs=fs, output="sos")
        b, a = signal.iirnotch(mains_hz, Q=30, fs=fs)
        self.notch = signal.tf2sos(b, a)
        self.sos = np.vstack([self.bp, self.notch])
        self.zi = np.zeros((self.sos.shape[0], 2, n_ch))

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, chunk, axis=0, zi=self.zi)
        return y


def band_powers(window: np.ndarray, fs: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Absolute band power per channel (uV^2) using Welch with median averaging.

    Median averaging makes the estimate robust to short artefacts such as blinks.
    Returns (powers_by_band, freqs, psd).
    """
    nper = min(len(window), fs)  # 1 s segments -> 1 Hz resolution
    freqs, psd = signal.welch(window, fs=fs, nperseg=nper, noverlap=nper // 2,
                              axis=0, average="median")
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = trapezoid(psd[m], freqs[m], axis=0) if m.any() else np.zeros(psd.shape[1])
    return out, freqs, psd


def extract_features(window: np.ndarray, fs: int, labels: list[str]) -> dict:
    """Interpretable features from a (samples, channels) window of filtered EEG."""
    bp, freqs, psd = band_powers(window, fs)
    total = sum(bp.values()) + 1e-12  # per channel
    rel = {b: bp[b] / total for b in bp}  # relative power per channel

    regions = np.array([channel_region(l) for l in labels])
    sides = np.array([channel_side(l) for l in labels])
    frontal = regions == 0
    posterior = regions >= 2

    def mean_where(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float(arr.mean())

    # high beta (20-30) vs low beta (13-20): a common stress / anxiety marker
    hb = (freqs >= 20) & (freqs < 30)
    lb = (freqs >= 13) & (freqs < 20)
    high_beta_ratio = float(trapezoid(psd[hb], freqs[hb], axis=0).mean()
                            / (trapezoid(psd[lb], freqs[lb], axis=0).mean() + 1e-12))

    # Frontal alpha asymmetry: ln(right alpha) - ln(left alpha).
    # Positive => relatively more LEFT activation (less left alpha) => approach / positive valence.
    left, right = frontal & (sides < 0), frontal & (sides > 0)
    if left.any() and right.any():
        faa = float(np.log(bp["alpha"][right].mean() + 1e-9) - np.log(bp["alpha"][left].mean() + 1e-9))
    else:
        faa = 0.0

    # Peak alpha frequency (posterior), useful for display and arousal
    am = (freqs >= 7) & (freqs <= 14)
    post_psd = psd[:, posterior].mean(axis=1) if posterior.any() else psd.mean(axis=1)
    paf = float(freqs[am][np.argmax(post_psd[am])]) if am.any() else 0.0

    rel_mean = {b: float(v.mean()) for b, v in rel.items()}
    features = {
        "relative": rel_mean,
        "absolute_uv2": {b: float(v.mean()) for b, v in bp.items()},
        "posterior_alpha_rel": mean_where(rel["alpha"], posterior),
        "frontal_theta_rel": mean_where(rel["theta"], frontal),
        "frontal_beta_rel": mean_where(rel["beta"], frontal),
        "high_beta_ratio": high_beta_ratio,
        "frontal_alpha_asymmetry": faa,
        "peak_alpha_hz": paf,
        "theta_beta_ratio": rel_mean["theta"] / (rel_mean["beta"] + 1e-9),
        "alpha_theta_ratio": rel_mean["alpha"] / (rel_mean["theta"] + 1e-9),
        "engagement_index": rel_mean["beta"] / (rel_mean["alpha"] + rel_mean["theta"] + 1e-9),
        "rms_uv": float(np.sqrt((window ** 2).mean())),
    }
    # Arousal / valence (0..1 / -1..1) for the prompt builder
    fast = rel_mean["beta"] + rel_mean["gamma"]
    slow = rel_mean["delta"] + rel_mean["theta"] + rel_mean["alpha"]
    features["arousal"] = float(np.clip(fast / (fast + slow + 1e-9) * 1.25, 0, 1))
    features["valence"] = float(np.clip(faa / 0.6, -1, 1))
    return features


def feature_vector(f: dict) -> np.ndarray:
    """Compact numeric vector used by the state classifier."""
    r = f["relative"]
    return np.array([
        np.log(r["delta"] + 1e-6),
        np.log(r["theta"] + 1e-6),
        np.log(r["alpha"] + 1e-6),
        np.log(r["beta"] + 1e-6),
        np.log(r["gamma"] + 1e-6),
        np.log(f["posterior_alpha_rel"] + 1e-6),
        np.log(f["high_beta_ratio"] + 1e-6),
        f["frontal_alpha_asymmetry"],
    ])


FEATURE_NAMES = ["log_rel_delta", "log_rel_theta", "log_rel_alpha", "log_rel_beta",
                 "log_rel_gamma", "log_posterior_alpha", "log_high_beta_ratio", "frontal_alpha_asym"]
