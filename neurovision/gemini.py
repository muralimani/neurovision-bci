"""Gemini API wrapper (google-genai SDK) for image and video generation."""

from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


class GeminiStudio:
    def __init__(self, api_key: str, image_model: str, video_model: str, text_model: str,
                 aspect_ratio: str = "16:9"):
        from google import genai
        from google.genai import types

        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.image_model = image_model
        self.video_model = video_model
        self.text_model = text_model
        self.aspect_ratio = aspect_ratio

    # ------------------------------------------------------------------ text
    def write_prompt(self, instruction: str) -> str:
        resp = self.client.models.generate_content(model=self.text_model, contents=instruction)
        return (resp.text or "").strip()

    # ----------------------------------------------------------------- image
    def generate_image(self, prompt: str, out_path: Path) -> Path:
        t = self.types
        resp = self.client.models.generate_content(
            model=self.image_model,
            contents=[prompt],
            config=t.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=t.ImageConfig(aspect_ratio=self.aspect_ratio),
            ),
        )
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content else []) or []:
                if part.inline_data and part.inline_data.data:
                    mime = part.inline_data.mime_type or "image/png"
                    ext = ".jpg" if "jpeg" in mime else ".png"
                    out_path = out_path.with_suffix(ext)
                    out_path.write_bytes(part.inline_data.data)
                    return out_path
        reason = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else "no candidates"
        raise RuntimeError(f"Gemini returned no image (finish_reason={reason}). Text: {resp.text!r}")

    # ----------------------------------------------------------------- video
    def generate_video(self, prompt: str, out_path: Path, image_path: Path | None = None,
                       poll_sec: float = 10.0, timeout_sec: float = 600.0) -> Path:
        t = self.types
        kwargs = {}
        if image_path and image_path.exists():
            mime = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            kwargs["image"] = t.Image(image_bytes=image_path.read_bytes(), mime_type=mime)
        op = self.client.models.generate_videos(
            model=self.video_model,
            prompt=prompt,
            config=t.GenerateVideosConfig(aspect_ratio=self.aspect_ratio, number_of_videos=1),
            **kwargs,
        )
        start = time.monotonic()
        while not op.done:
            if time.monotonic() - start > timeout_sec:
                raise TimeoutError("Video generation timed out")
            time.sleep(poll_sec)
            op = self.client.operations.get(op)
        if getattr(op, "error", None):
            raise RuntimeError(f"Video generation failed: {op.error}")
        vids = op.response.generated_videos if op.response else None
        if not vids:
            raise RuntimeError("Video generation returned no video (possibly filtered).")
        video = vids[0].video
        self.client.files.download(file=video)
        video.save(str(out_path))
        return out_path
