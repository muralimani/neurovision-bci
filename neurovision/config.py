"""Configuration loading. Edit config.toml rather than this file."""

from __future__ import annotations

import os
try:
    import tomllib  # Python 3.11+ (Raspberry Pi OS Bookworm and later)
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # pip install tomli on older Pi OS
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EEGConfig:
    sample_rate: int = 250
    channel_labels: list[str] = field(
        default_factory=lambda: ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    )
    mains_hz: float = 50.0          # 50 in UK/EU/India, 60 in US
    window_sec: float = 4.0         # analysis window
    update_sec: float = 1.0         # how often features are recomputed
    smoothing: float = 0.25         # EMA factor for state probabilities (0..1, higher = faster)
    stable_sec: float = 3.0         # a state must win this long before it counts as a change


@dataclass
class SourceConfig:
    kind: str = "synthetic"         # synthetic | pieeg | csv
    pieeg_url: str = "ws://localhost:1616"
    pieeg_token: str = ""           # only if pieeg-server runs with --auth
    csv_path: str = ""
    synthetic_state: str = "relaxed"
    synthetic_scenario: str = ""    # e.g. "relaxed:30,focused:30,stressed:30" (loops)
    seed: int | None = None


@dataclass
class GeminiConfig:
    api_key_env: str = "GEMINI_API_KEY"
    image_model: str = "gemini-3.1-flash-image"
    video_model: str = "veo-3.1-fast-generate-preview"
    text_model: str = "gemini-flash-latest"   # used only when prompt_mode = "llm"
    prompt_mode: str = "template"   # template | llm
    aspect_ratio: str = "16:9"
    style: str = "luminous abstract digital painting"
    make_video: bool = False
    video_from_image: bool = True   # animate the generated image instead of text-only video


@dataclass
class TriggerConfig:
    mode: str = "on_change"         # on_change | interval
    cooldown_sec: float = 30.0      # minimum gap between generations
    interval_sec: float = 120.0     # used for mode=interval, and as a max gap for on_change
    min_confidence: float = 0.45


@dataclass
class AppConfig:
    eeg: EEGConfig = field(default_factory=EEGConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    output_dir: str = "output"
    profiles_path: str = "profiles/personal.json"
    dry_run: bool = False

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.gemini.api_key_env) or os.environ.get("GOOGLE_API_KEY")


def _apply(dc, values: dict):
    for k, v in values.items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    cfg = AppConfig()
    p = Path(path) if path else Path("config.toml")
    if p.exists():
        data = tomllib.loads(p.read_text())
        for section in ("eeg", "source", "gemini", "trigger"):
            _apply(getattr(cfg, section), data.get(section, {}))
        _apply(cfg, {k: v for k, v in data.items() if not isinstance(v, dict)})
    return cfg
