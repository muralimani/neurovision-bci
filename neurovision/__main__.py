"""Command-line entry point:  python -m neurovision <command> [options]

Commands
  run               live loop: EEG -> state -> Gemini image (and optional video)
  once              generate one image for a synthetic state (quick API test)
  fake-server       pieeg-server look-alike streaming synthetic states on :1616
  generate-data     write synthetic CSV datasets + feature summary + example prompts
  calibrate         record your own labelled states (use with the real PiEEG)
  build-prototypes  rebuild the default state prototypes from synthetic data
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path

import numpy as np

from . import STATES
from .config import load_config
from .dsp import Preprocessor, extract_features, feature_vector
from .prompts import build_prompt
from .states import DEFAULT_PROTOTYPES, Prototypes
from .synthetic import STATE_PROFILES, generate_recording


def _common(p):
    p.add_argument("-c", "--config", default="config.toml")
    p.add_argument("-v", "--verbose", action="store_true")


def _features_over(data, fs, labels, mains, window_sec=4.0, step_sec=1.0):
    """Run the same filter + windowing as the live loop over a whole array."""
    pre = Preprocessor(fs, data.shape[1], mains)
    y = pre(data)
    w, s = int(window_sec * fs), int(step_sec * fs)
    start = int(2 * fs)  # skip filter warm-up
    return [extract_features(y[i - w:i], fs, labels) for i in range(start + w, len(y) + 1, s)]


# ------------------------------------------------------------------ commands
def cmd_run(args, cfg):
    from .pipeline import NeuroVision

    if args.source:
        cfg.source.kind = args.source
    if args.state:
        cfg.source.synthetic_state = args.state
    if args.scenario:
        cfg.source.synthetic_scenario = args.scenario
    if args.url:
        cfg.source.pieeg_url = args.url
    if args.csv:
        cfg.source.kind, cfg.source.csv_path = "csv", args.csv
    if args.dry_run:
        cfg.dry_run = True
    if args.video:
        cfg.gemini.make_video = True
    if args.prompt_mode:
        cfg.gemini.prompt_mode = args.prompt_mode
    if args.cooldown is not None:
        cfg.trigger.cooldown_sec = args.cooldown
    app = NeuroVision(cfg, realtime=not args.fast, max_generations=args.max_images)
    try:
        asyncio.run(app.run(duration=args.duration))
    except KeyboardInterrupt:
        pass
    print(f"\nDone: {app.generations} generation(s) written to {cfg.output_dir}/")


def cmd_once(args, cfg):
    e = cfg.eeg
    _, data, _ = generate_recording(args.state, 12, e.sample_rate, e.channel_labels, e.mains_hz, seed=None)
    feats = _features_over(data, e.sample_rate, e.channel_labels, e.mains_hz)[-1]
    prompt = build_prompt(args.state, feats, cfg.gemini.style, for_video=args.video)
    print("PROMPT:\n" + prompt + "\n")
    if args.dry_run:
        return
    if not cfg.api_key:
        raise SystemExit(f"Set {cfg.gemini.api_key_env} first (export GEMINI_API_KEY=...).")
    from .gemini import GeminiStudio

    g = cfg.gemini
    studio = GeminiStudio(cfg.api_key, g.image_model, g.video_model, g.text_model, g.aspect_ratio)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = studio.generate_image(prompt, out / f"once_{args.state}.png")
    print("image:", img)
    if args.video:
        vid = studio.generate_video(build_prompt(args.state, feats, g.style, True),
                                    out / f"once_{args.state}.mp4", img)
        print("video:", vid)


def cmd_fake_server(args, cfg):
    from .fake_server import FakePiEEGServer

    e = cfg.eeg
    srv = FakePiEEGServer(e.sample_rate, e.channel_labels, e.mains_hz, args.state,
                          args.scenario, args.host, args.port)
    try:
        asyncio.run(srv.run())
    except KeyboardInterrupt:
        pass


def cmd_generate_data(args, cfg):
    e = cfg.eeg
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels, fs = e.channel_labels, e.sample_rate
    summary = {"sample_rate": fs, "units": "microvolts", "channels": labels, "states": {}}

    def write_csv(path, t, data, states):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "sample", *labels, "state"])
            for i in range(len(data)):
                w.writerow([f"{t[i]:.4f}", i, *[f"{v:.3f}" for v in data[i]], states[i]])

    for k, st in enumerate(STATES):
        t, data, st_lab = generate_recording(st, args.seconds, fs, labels, e.mains_hz, seed=100 + k)
        write_csv(out / f"{st}.csv", t, data, st_lab)
        feats = _features_over(data, fs, labels, e.mains_hz)
        mean_rel = {b: round(float(np.mean([f["relative"][b] for f in feats])), 4) for b in feats[0]["relative"]}
        mid = feats[len(feats) // 2]
        summary["states"][st] = {
            "file": f"{st}.csv",
            "description": STATE_PROFILES[st]["description"],
            "mean_relative_band_power": mean_rel,
            "arousal": round(float(np.mean([f["arousal"] for f in feats])), 3),
            "valence": round(float(np.mean([f["valence"] for f in feats])), 3),
            "peak_alpha_hz": round(float(np.median([f["peak_alpha_hz"] for f in feats])), 1),
            "example_image_prompt": build_prompt(st, mid, cfg.gemini.style),
            "example_video_prompt": build_prompt(st, mid, cfg.gemini.style, for_video=True),
        }
        print(f"  {st:<10} -> {out / (st + '.csv')}")

    journey = [("relaxed", 40), ("focused", 40), ("stressed", 30), ("meditative", 40),
               ("creative", 40), ("excited", 30), ("drowsy", 40)]
    t, data, st_lab = generate_recording(journey, None, fs, labels, e.mains_hz, seed=7)
    write_csv(out / "mood_journey.csv", t, data, st_lab)
    summary["mood_journey"] = {"file": "mood_journey.csv", "segments": journey}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  journey    -> {out / 'mood_journey.csv'}\n  summary    -> {out / 'summary.json'}")

    if not args.no_plot:
        _plot_psd(out, fs, labels, e.mains_hz, args.seconds)


def _plot_psd(out, fs, labels, mains, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = ["#2a9d8f", "#e76f51", "#8e7dbe", "#264653", "#d62828", "#f4a261", "#e9c46a"]
    for c, st in zip(colors, STATES):
        _, data, _ = generate_recording(st, min(seconds, 30), fs, labels, mains, seed=1)
        y = Preprocessor(fs, data.shape[1], mains)(data)[2 * fs:]
        f, p = signal.welch(y, fs=fs, nperseg=fs * 2, axis=0, average="median")
        axes[0].semilogy(f, p.mean(axis=1), color=c, label=st, lw=1.6)
    for lo, hi, name in [(0.5, 4, "δ"), (4, 8, "θ"), (8, 13, "α"), (13, 30, "β"), (30, 45, "γ")]:
        axes[0].axvspan(lo, hi, alpha=0.06, color="k" if name in "θβ" else "w")
        axes[0].text((lo + hi) / 2, 0.98, name, transform=axes[0].get_xaxis_transform(), ha="center", va="top")
    axes[0].set(xlim=(0.5, 45), ylim=(1e-2, 2e3), xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)",
                title="Synthetic spectra by mental state (channel average)")
    axes[0].legend(fontsize=8, ncol=2, frameon=False)

    _, data, _ = generate_recording("relaxed", 6, fs, labels, mains, seed=3)
    y = Preprocessor(fs, data.shape[1], mains)(data)[2 * fs:]
    t = np.arange(len(y)) / fs
    for i, lab in enumerate(labels):
        axes[1].plot(t, y[:, i] - i * 80, lw=0.6, color="#333")
        axes[1].text(-0.1, -i * 80, lab, ha="right", va="center", fontsize=8)
    axes[1].set(yticks=[], xlabel="Time (s)", title="'relaxed' – 8 channels, 1-45 Hz (note posterior alpha)")
    for s in ("top", "right", "left"):
        axes[1].spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "synthetic_overview.png", dpi=130)
    print(f"  plot       -> {out / 'synthetic_overview.png'}")


def cmd_calibrate(args, cfg):
    """Record labelled feature vectors from the configured source into profiles/personal.json."""
    from .sources import make_source

    if args.source:
        cfg.source.kind = args.source
    if args.source == "synthetic":
        cfg.source.synthetic_state = args.label
    e = cfg.eeg
    path = Path(cfg.profiles_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(path.read_text()) if path.exists() else {}
    raw = store.get("raw", {})

    async def record():
        src = make_source(cfg, realtime=True)
        pre = Preprocessor(e.sample_rate, len(e.channel_labels), e.mains_hz)
        from .pipeline import RingBuffer
        ring = RingBuffer(int(e.window_sec * e.sample_rate), len(e.channel_labels))
        vecs, seen, since = [], 0, 0
        print(f"Recording '{args.label}' for {args.seconds}s – hold that state now...")
        async for _, x in src.stream():
            ring.push(pre(x))
            seen += len(x)
            since += len(x)
            if seen < (2 + e.window_sec) * e.sample_rate or since < e.sample_rate:
                continue
            since = 0
            vecs.append(feature_vector(extract_features(ring.buf, e.sample_rate, e.channel_labels)).tolist())
            print(f"\r  {len(vecs)} windows", end="", flush=True)
            if seen >= (args.seconds + 2) * e.sample_rate:
                break
        print()
        return vecs

    vecs = asyncio.run(record())
    raw.setdefault(args.label, []).extend(vecs)
    built = Prototypes.build({k: [np.array(v) for v in vs] for k, vs in raw.items()})
    built["raw"] = raw
    path.write_text(json.dumps(built))
    print(f"Saved {len(vecs)} windows for '{args.label}' -> {path}. States so far: {sorted(raw)}")
    if len(raw) < 2:
        print("Record at least two states for the classifier to be meaningful.")


def cmd_build_prototypes(args, cfg):
    e = cfg.eeg
    samples = {}
    for k, st in enumerate(STATES):
        _, data, _ = generate_recording(st, 90, e.sample_rate, e.channel_labels, e.mains_hz, seed=1000 + k)
        feats = _features_over(data, e.sample_rate, e.channel_labels, e.mains_hz)
        samples[st] = [feature_vector(f) for f in feats]
    out = Prototypes.build(samples)
    out["note"] = "Derived from neurovision.synthetic profiles. Replace with your own via `calibrate`."
    DEFAULT_PROTOTYPES.write_text(json.dumps(out, indent=2))
    print("wrote", DEFAULT_PROTOTYPES)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neurovision", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="live loop")
    _common(p)
    p.add_argument("--source", choices=["synthetic", "pieeg", "csv"])
    p.add_argument("--state", choices=STATES, help="synthetic state")
    p.add_argument("--scenario", help='synthetic schedule, e.g. "relaxed:40,focused:40,stressed:30"')
    p.add_argument("--url", help="pieeg-server WebSocket URL, e.g. ws://raspberrypi.local:1616")
    p.add_argument("--csv", help="replay this CSV file")
    p.add_argument("--dry-run", action="store_true", help="write prompts only; no API calls")
    p.add_argument("--video", action="store_true", help="also make a Veo video per image")
    p.add_argument("--prompt-mode", choices=["template", "llm"])
    p.add_argument("--cooldown", type=float, help="seconds between generations")
    p.add_argument("--duration", type=float, help="stop after N seconds")
    p.add_argument("--max-images", type=int, help="stop after N generations")
    p.add_argument("--fast", action="store_true", help="process synthetic/CSV data faster than real time")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("once", help="one image from a synthetic state")
    _common(p)
    p.add_argument("--state", choices=STATES, default="relaxed")
    p.add_argument("--video", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_once)

    p = sub.add_parser("fake-server", help="pieeg-server look-alike")
    _common(p)
    p.add_argument("--state", choices=STATES, default="relaxed")
    p.add_argument("--scenario", default="")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=1616)
    p.set_defaults(func=cmd_fake_server)

    p = sub.add_parser("generate-data", help="write synthetic datasets")
    _common(p)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("calibrate", help="record your own labelled states")
    _common(p)
    p.add_argument("--label", required=True, help="state name, e.g. relaxed (any name works)")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--source", choices=["synthetic", "pieeg", "csv"])
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("build-prototypes", help="rebuild default prototypes")
    _common(p)
    p.set_defaults(func=cmd_build_prototypes)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("websockets", "httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args, load_config(args.config))


if __name__ == "__main__":
    main()
