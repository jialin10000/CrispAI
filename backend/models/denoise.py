"""
Noise reduction — Topaz-style models.

Normal / Strong / Severe are AI-based (FFDNet). One model, sigma-controlled
strength — light, medium, heavy denoise respectively. The slider blends the
AI result with the original so 0 = no effect, 100 = full AI output.

Impulse (salt-and-pepper noise) uses median filter. AI denoisers trained on
Gaussian/real noise don't handle impulse well; median is the right tool.

If the AI model fails to load (no internet on first run, no GPU/CPU torch,
corrupted weights) we fall back to the classical OpenCV implementation so
the app keeps working.
"""

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# AI model targets (KAIR sigma scale, 0..255).
# Matches Topaz Photo AI's Denoise sub-models: Normal / Strong / Extreme.
_SIGMA_BY_MODEL = {
    "normal":  15.0,
    "strong":  25.0,
    "extreme": 50.0,
    # 'severe' kept as alias for backwards compat with earlier param payloads
    "severe":  50.0,
}


# ── Classical fallbacks (used if AI fails) ──────────────────────

def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _classical_denoise(image: Image.Image, strength: float, sigma: float) -> Image.Image:
    """OpenCV NLM fallback when AI is unavailable.
    sigma 15/25/50 maps to NLM h ~ 5/9/16."""
    h = sigma * 0.32
    bgr = _to_bgr(image)
    out = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=h, hColor=h,
        templateWindowSize=7, searchWindowSize=21,
    )
    return _to_pil(out)


def _denoise_impulse(image: Image.Image, strength: float) -> Image.Image:
    """Median filter for salt-and-pepper / dead-pixel artifacts.
    AI denoisers don't help with impulse noise — different statistics."""
    k = 3 if strength < 0.33 else (5 if strength < 0.66 else 7)
    bgr = _to_bgr(image)
    return _to_pil(cv2.medianBlur(bgr, k))


# ── Public model ────────────────────────────────────────────────

class DenoiseModel:
    def __init__(self):
        self._ai = None
        self._ai_load_attempted = False
        self._ai_failed_reason = None

    def _get_ai(self):
        """Lazy-load the AI denoiser. Returns None if loading fails (and stays
        None — we don't retry every call)."""
        if self._ai is not None:
            return self._ai
        if self._ai_load_attempted:
            return None
        self._ai_load_attempted = True
        try:
            from .ai_denoise import AIDenoiser
            self._ai = AIDenoiser.get()
            logger.info("AI denoiser loaded")
        except Exception as e:
            self._ai_failed_reason = str(e)
            logger.warning(f"AI denoiser unavailable, falling back to OpenCV NLM: {e}")
        return self._ai

    def process(self, image: Image.Image, strength: float = 0.5,
                model: str = "normal") -> Image.Image:
        if strength <= 0.01:
            return image

        if model == "impulse":
            return _denoise_impulse(image, strength)

        sigma = _SIGMA_BY_MODEL.get(model, _SIGMA_BY_MODEL["normal"])

        ai = self._get_ai()
        if ai is None:
            return _classical_denoise(image, strength, sigma)

        try:
            denoised = ai.denoise(image, sigma)
        except Exception as e:
            logger.error(f"AI inference failed, falling back: {e}")
            return _classical_denoise(image, strength, sigma)

        # Blend with original by strength so the slider does something:
        # strength=1.0 -> full AI; strength=0.5 -> halfway between source and AI
        if strength >= 0.99:
            return denoised
        return Image.blend(image.convert("RGB"), denoised, strength)
