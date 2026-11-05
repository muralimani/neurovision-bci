"""NeuroVision: PiEEG brain-wave features -> Gemini image / video generation."""

__version__ = "0.1.0"

BANDS = {
    # Same band edges the PiEEG dashboard uses (docs.pieeg.com, Data Format)
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),  # capped at 45 Hz to stay clear of 50/60 Hz mains
}

STATES = ["relaxed", "focused", "meditative", "drowsy", "stressed", "creative", "excited"]
