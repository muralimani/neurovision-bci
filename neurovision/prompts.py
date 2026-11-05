"""Turn EEG features + estimated state into an image / video prompt.

The prompt is not just "draw a relaxed brain": it blends visual motifs in
proportion to the measured band powers, so two "relaxed" moments with different
spectra still produce different images. Arousal drives energy / motion and
frontal alpha asymmetry drives warmth (valence).
"""

from __future__ import annotations

import json

STATE_SCENES = {
    "relaxed": "a still lake at dusk reflecting soft auroras, gentle mist drifting over water",
    "focused": "a crystalline lattice of light converging to a single bright point, clean precise lines",
    "meditative": "an endless calm horizon where concentric ripples of light expand from a glowing centre",
    "drowsy": "a deep twilight ocean dissolving into dreams, slow luminous jellyfish fading into darkness",
    "stressed": "a storm of jagged electric filaments and shattered geometric shards under pressure",
    "creative": "a surreal garden where flowing ribbons of colour bloom into unexpected shapes and ideas",
    "excited": "a radiant burst of fireworks and sparkling particles spiralling upward in celebration",
}

BAND_MOTIFS = {
    "delta": ("deep, slow, oceanic shapes", "midnight indigo"),
    "theta": ("dreamlike, surreal, hazy imagery", "violet and magenta"),
    "alpha": ("soft, flowing waves of light", "sage green and sky blue"),
    "beta": ("crisp geometric structure and fine detail", "amber and warm white"),
    "gamma": ("tiny sparkling points and electric highlights", "electric crimson and gold"),
}


def describe_energy(arousal: float) -> str:
    if arousal < 0.2:
        return "very calm, almost motionless, wide open negative space"
    if arousal < 0.4:
        return "calm with gentle slow movement"
    if arousal < 0.6:
        return "balanced, steady flowing energy"
    if arousal < 0.8:
        return "energetic, dynamic movement and contrast"
    return "intense, high-energy, vivid contrast and rapid motion"


def describe_valence(valence: float) -> str:
    if valence > 0.3:
        return "warm, uplifting, optimistic light"
    if valence < -0.3:
        return "cool, tense, slightly ominous light"
    return "neutral, balanced lighting"


def build_prompt(state: str, features: dict, style: str, for_video: bool = False) -> str:
    rel = features["relative"]
    ranked = sorted(rel.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:3]
    motifs = "; ".join(f"{BAND_MOTIFS[b][0]} ({v*100:.0f}% {b})" for b, v in top)
    palette = ", ".join(BAND_MOTIFS[b][1] for b, _ in top)
    medium = "A cinematic, slowly evolving shot of" if for_video else "A"
    lines = [
        f"{medium} {style} visualising a human mind in {'an' if state[0] in 'aeiou' else 'a'} {state} state: {STATE_SCENES.get(state, '')}.",
        f"Visual emphasis follows the measured brain rhythms: {motifs}.",
        f"Colour palette: {palette}.",
        f"Energy: {describe_energy(features['arousal'])}. Mood: {describe_valence(features['valence'])}.",
    ]
    if for_video:
        lines.append(f"Motion pulses gently at about {features.get('peak_alpha_hz', 10):.0f} beats per second "
                     "in the light, camera drifts slowly. No cuts.")
    lines.append("No text, no letters, no logos, no anatomical brain diagrams, no people's faces.")
    return " ".join(lines)


LLM_INSTRUCTION = """You are an art director turning live EEG readings into a single image-generation prompt.
Here are the readings (relative band powers sum to 1; arousal 0-1; valence -1..1 from frontal alpha asymmetry):
{data}
Write ONE vivid prompt (max 90 words) for a {style}{video} that expresses this mental state.
Let the dominant rhythms decide imagery and colour (delta=deep/oceanic, theta=dreamlike, alpha=soft flowing light,
beta=crisp geometry, gamma=sparkling energy). No text, letters, logos, faces or anatomical brains.
Return only the prompt."""


def llm_prompt_request(state: str, features: dict, style: str, for_video: bool = False) -> str:
    data = {
        "estimated_state": state,
        "relative_band_power": {k: round(v, 3) for k, v in features["relative"].items()},
        "arousal": round(features["arousal"], 2),
        "valence": round(features["valence"], 2),
        "peak_alpha_hz": round(features.get("peak_alpha_hz", 0), 1),
    }
    return LLM_INSTRUCTION.format(data=json.dumps(data, indent=2), style=style,
                                  video=" 8-second video" if for_video else " still image")
