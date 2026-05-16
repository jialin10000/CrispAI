"""
Denoise — NAFNet-SIDD (Topaz-grade AI denoising).

Models map to NAFNet inference with strength-controlled blending:
  Normal   - 50% AI strength
  Strong   - 80% AI strength
  Extreme  - 100% AI strength (full NAFNet output)
  Impulse  - kept on median filter (AI doesn't help salt-and-pepper noise)

NAFNet-SIDD is trained on the SIDD real-photo dataset and is genuinely
state-of-the-art for camera sensor noise. ~70 MB weights, auto-downloaded
from the official megvii Google Drive on first use.

Cascading fallbacks if anything goes wrong:
  NAFNet -> FFDNet (smaller AI) -> OpenCV fastNlMeansDenoisingColored
"""

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Strength-vs-AI blend per model. Topaz also uses a model-specific intensity.
_BLEND_BY_MODEL = {
    "normal":  0.55,
    "strong":  0.80,
    "extreme": 1.00,
    "severe":  1.00,    # legacy alias
}


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _classical_fallback(image: Image.Image, sigma: float) -> Image.Image:
    h = sigma * 0.32
    bgr = _to_bgr(image)
    out = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=h, hColor=h,
        templateWindowSize=7, searchWindowSize=21,
    )
    return _to_pil(out)


def _denoise_impulse(image: Image.Image, strength: float) -> Image.Image:
    k = 3 if strength < 0.33 else (5 if strength < 0.66 else 7)
    bgr = _to_bgr(image)
    return _to_pil(cv2.medianBlur(bgr, k))


class DenoiseModel:
    """Lazy-loads NAFNet on first call. Holds the runner in memory after that."""

    def __init__(self):
        self._nafnet = None
        self._ffdnet = None
        self._nafnet_attempted = False
        self._ffdnet_attempted = False

    def _get_nafnet(self):
        if self._nafnet is not None:
            return self._nafnet
        if self._nafnet_attempted:
            return None
        self._nafnet_attempted = True
        try:
            from .nafnet_runner import NAFNetRunner
            self._nafnet = NAFNetRunner.get("denoise")
            logger.info("NAFNet-SIDD denoiser loaded")
        except Exception as e:
            logger.warning(f"NAFNet denoise unavailable, will try FFDNet fallback: {e}")
        return self._nafnet

    def _get_ffdnet(self):
        if self._ffdnet is not None:
            return self._ffdnet
        if self._ffdnet_attempted:
            return None
        self._ffdnet_attempted = True
        try:
            from .ai_denoise import AIDenoiser
            self._ffdnet = AIDenoiser.get()
            logger.info("FFDNet fallback denoiser loaded")
        except Exception as e:
            logger.warning(f"FFDNet fallback also unavailable: {e}")
        return self._ffdnet

    def process(self, image: Image.Image, strength: float = 0.5,
                model: str = "normal") -> Image.Image:
        if strength <= 0.01:
            return image
        if model == "impulse":
            return _denoise_impulse(image, strength)

        base_image = image.convert("RGB")

        # Stage 1: NAFNet (best quality)
        runner = self._get_nafnet()
        if runner is not None:
            try:
                denoised = runner.run(base_image)
                # Per-model AI intensity * user strength
                model_blend = _BLEND_BY_MODEL.get(model, 0.55)
                alpha = max(0.0, min(1.0, model_blend * strength))
                if alpha >= 0.99:
                    return denoised
                return Image.blend(base_image, denoised, alpha)
            except Exception as e:
                logger.error(f"NAFNet inference failed: {e}")

        # Stage 2: FFDNet fallback (older AI, smaller)
        ffd = self._get_ffdnet()
        if ffd is not None:
            try:
                sigma_map = {"normal": 15.0, "strong": 25.0, "extreme": 50.0, "severe": 50.0}
                sigma = sigma_map.get(model, 15.0)
                denoised = ffd.denoise(base_image, sigma)
                if strength >= 0.99:
                    return denoised
                return Image.blend(base_image, denoised, strength)
            except Exception as e:
                logger.error(f"FFDNet inference failed: {e}")

        # Stage 3: classical OpenCV NLM
        sigma_map = {"normal": 15.0, "strong": 25.0, "extreme": 50.0, "severe": 50.0}
        return _classical_fallback(image, sigma_map.get(model, 15.0))
