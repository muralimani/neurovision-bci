# NeuroVision — PiEEG brain waves → Gemini images & video

Reads 8-channel EEG, extracts delta/theta/alpha/beta/gamma band power, estimates a
mental state (relaxed, focused, meditative, drowsy, stressed, creative, excited),
turns it into an art prompt and generates an image (Gemini "Nano Banana 2") and
optionally a short video (Veo) through the Gemini API.

Until the PiEEG arrives, everything runs on realistic **synthetic EEG**.

```
 EEG source ──► 1-45 Hz band-pass + 50 Hz notch ──► 4 s window, Welch PSD
 (synthetic │ pieeg │ csv)                              │
                                    band powers, ratios, alpha asymmetry
                                                         │
                          state classifier + smoothing ──► prompt builder
                                                         │
                                  Gemini image ──► (optional) Veo video
                                                         │
                                         output/<time>_<state>.png/.mp4/.json
```

## Why the switch to hardware is seamless

PiEEG's official software, `pieeg-server`, streams JSON frames over a WebSocket on
port 1616: `{"t": <unix time>, "n": <sample>, "channels": [µV × 8]}` at 250 Hz.
This project's `pieeg` source is a client for exactly that protocol, and the included
`fake-server` speaks the same protocol with state-driven synthetic data. So:

| Today | When the PiEEG arrives |
|---|---|
| `python -m neurovision fake-server --scenario "relaxed:40,focused:40"` | `pieeg-server --device pieeg8` |
| `python -m neurovision run --source pieeg` | `python -m neurovision run --source pieeg` (same command) |

Or skip the server entirely today with `--source synthetic` (in-process generator).

## Install on the Raspberry Pi

```bash
cd ~/neurovision
bash install_pi.sh                     # venv, dependencies, pieeg-server, synthetic data
source .venv/bin/activate
export GEMINI_API_KEY="..."            # https://aistudio.google.com/apikey (image generation needs a paid tier)
```

## Try it with synthetic data

```bash
# No API calls: prints live state + writes the prompts to output/*.json
python -m neurovision run --scenario "relaxed:40,focused:40,stressed:30,meditative:40" --dry-run

# One real image for a given state
python -m neurovision once --state creative
python -m neurovision once --state excited --video        # + Veo clip (slow, paid)

# Full live loop with real images (new image when the state changes, >=30 s apart)
python -m neurovision run --scenario "relaxed:60,excited:60,drowsy:60"

# End-to-end over the network protocol, two terminals:
python -m neurovision fake-server --scenario "meditative:45,stressed:45"
python -m neurovision run --source pieeg --url ws://localhost:1616
#   (you can also send {"cmd":"set_state","state":"focused"} to the fake server)

# Replay a CSV
python -m neurovision run --csv data/synthetic/mood_journey.csv
```

Useful flags: `--prompt-mode llm` (Gemini text model writes the prompt from the
numbers), `--cooldown 60`, `--max-images 5`, `--duration 300`, `--fast`.

## Synthetic datasets (`data/synthetic/`)

`generate-data` writes one 60 s CSV per state plus `mood_journey.csv` (≈4.5 min moving
through all states), `summary.json` (band powers, arousal/valence and the example
prompts that would be sent to Gemini) and `synthetic_overview.png`.

Columns: `timestamp, sample, Fp1, Fp2, C3, C4, P3, P4, O1, O2, state` — µV at 250 Hz.

| State | Signature in the data |
|---|---|
| relaxed | strong 10 Hz alpha, strongest over the back of the head (O1/O2) |
| focused | alpha suppressed, 15-25 Hz beta and frontal theta up |
| meditative | frontal theta + slower alpha together, very low beta |
| drowsy | large delta (≈2 Hz) and theta, alpha fading |
| stressed | high beta (20-30 Hz), low alpha, right-frontal activation |
| creative | alpha and theta both strong, moderate beta |
| excited | beta + 40 Hz gamma, left-frontal activation (positive valence) |

The generator adds 1/f background noise, 50 Hz mains hum, waxing/waning rhythms and
eye-blink artefacts on Fp1/Fp2, so the processing chain is tested against realistic junk.
These profiles are based on commonly reported EEG patterns but are simplifications —
real data is noisier and person-specific.

## When the PiEEG arrives

1. Follow PiEEG's safety guidance: battery power only, no mains connection to the Pi
   while electrodes are on your head.
2. `pieeg-server --device pieeg8` (first run enables SPI → `sudo reboot` once).
   Check signal quality in its dashboard at `http://raspberrypi.local:1617`.
3. Edit `channel_labels` in `config.toml` to match where you put the 8 electrodes.
   Alpha is easiest to see at O1/O2 with eyes closed.
4. In `config.toml` set `kind = "pieeg"` (or keep passing `--source pieeg`).
5. **Calibrate** — the default classifier is tuned on synthetic data. Record ~60 s of
   each state you care about; the personal profile is then used automatically:
   ```bash
   python -m neurovision calibrate --source pieeg --label relaxed --seconds 60    # eyes closed, calm
   python -m neurovision calibrate --source pieeg --label focused --seconds 60    # mental arithmetic
   python -m neurovision calibrate --source pieeg --label stressed --seconds 60   # timed hard puzzle
   ```
   Labels are free text, so you can define your own "thoughts" (e.g. `ocean`, `fire`),
   but keep expectations realistic: consumer EEG reliably separates broad states like
   calm vs. concentrating, not specific imagined pictures.
6. Optional autostart: see `neurovision.service`.

## Files

```
neurovision/
  synthetic.py     state-driven synthetic EEG generator (streaming)
  sources.py       synthetic / pieeg-server WebSocket / CSV sources
  fake_server.py   pieeg-server protocol look-alike
  dsp.py           filtering, band powers, ratios, asymmetry, arousal/valence
  states.py        nearest-centroid state classifier + smoothing/hysteresis
  prompts.py       features -> image/video prompt (template or LLM)
  gemini.py        google-genai image + Veo video calls
  pipeline.py      the live loop and generation trigger
config.toml        all settings (models, electrodes, trigger, cooldown)
```

## Cost control

Every image is a paid API call. The trigger only generates when the stable state
changes (with a minimum `cooldown_sec`, default 30 s) or after `interval_sec`. Veo
video is much more expensive and slower — leave `make_video = false` until you want it.
Model IDs change over time; update `image_model` / `video_model` in `config.toml` from
https://ai.google.dev/gemini-api/docs/models if a call reports an unknown model.
