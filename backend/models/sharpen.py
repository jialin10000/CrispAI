"""
Sharpen — Topaz Photo AI-style options.

  Standard     classical multi-scale unsharp — fine-detail edge boost.
  Strong       classical larger-radius unsharp — coarse-feature contrast.
  Lens Blur    NAFNet-GoPro AI deblur (mild blend) — wide-aperture softness.
  Motion Blur  NAFNet-GoPro AI deblur (medium-to-full blend) — camera shake.
  Refocus      NAFNet-GoPro AI deblur (full) + light unsharp — soft images.

NAFNet-GoPro is trained on real motion-blurred photos and estimates the
deblur implicitly — no need to specify a PSF or angle. This is the AI that
replaces our earlier classical Richardson-Lucy deconvolution.

If the AI model fails to load (no internet on first run, no GPU/CPU torch,
corrupted weights), Lens/Motion/Refocus fall back to the old RL deconv so
the app still works.
"""

import logging
import cv2
import numpy as np
from PIL import Image

from .deblur import disk_psf, motion_psf, richardson_lucy

logger = logging.getLogger(__name__)


# ── pure sharpen kernels ─────────────────────────────────────────────────

def _unsharp(rgb: np.ndarray, radius: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=radius, sigmaY=radius)
    out = cv2.addWeighted(rgb, 1 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _multi_scale_sharpen(rgb: np.ndarray, strength: float,
                         fine_range=(0.4, 2.4), mid_range=(0.2, 1.0)) -> np.ndarray:
    f = rgb.astype(np.float32)
    fine_blur   = cv2.GaussianBlur(f, (0, 0), sigmaX=0.6)
    medium_blur = cv2.GaussianBlur(f, (0, 0), sigmaX=2.0)
    fine_detail   = f - fine_blur
    medium_detail = fine_blur - medium_blur
    fine_amt = fine_range[0] + (fine_range[1] - fine_range[0]) * strength
    mid_amt  = mid_range[0]  + (mid_range[1]  - mid_range[0])  * strength
    out = f + fine_amt * fine_detail + mid_amt * medium_detail
    return np.clip(out, 0, 255).astype(np.uint8)


# ── AI deblur with classical fallback ────────────────────────────────────

class _DeblurAI:
    """Singleton wrapper for NAFNet-GoPro."""
    _runner = None
    _attempted = False

    @classmethod
    def get(cls):
        if cls._runner is not None:
            return cls._runner
        if cls._attempted:
            return None
        cls._attempted = True
        try:
            from .nafnet_runner import NAFNetRunner
            cls._runner = NAFNetRunner.get("deblur")
            logger.info("NAFNet-GoPro deblur loaded")
        except Exception as e:
            logger.warning(f"NAFNet deblur unavailable, RL fallback will be used: {e}")
        return cls._runner


def _ai_deblur_with_blend(image: Image.Image, strength: float,
                          model_strength: float,
                          fallback_psf, fallback_iters_max: int) -> Image.Image:
    """Run NAFNet-GoPro and blend with original by strength*model_strength.
    Falls back to Richardson-Lucy with the supplied PSF on failure."""
    base = image.convert("RGB")
    ai = _DeblurAI.get()
    if ai is not None:
        try:
            deblurred = ai.run(base)
            alpha = max(0.0, min(1.0, strength * model_strength))
            if alpha >= 0.99:
                return deblurred
            return Image.blend(base, deblurred, alpha)
        except Exception as e:
            logger.error(f"NAFNet deblur inference failed: {e}")

    # Fallback: classical RL deconv (the old behaviour)
    iters = int(round(4 + (fallback_iters_max - 4) * strength))
    arr = np.array(base)
    out = richardson_lucy(arr, fallback_psf, iters=iters)
    return Image.fromarray(out)


# ── Model functions ──────────────────────────────────────────────────────

def _model_standard(image: Image.Image, strength: float, _angle: float) -> Image.Image:
    return Image.fromarray(_multi_scale_sharpen(np.array(image.convert("RGB")), strength))


def _model_strong(image: Image.Image, strength: float, _angle: float) -> Image.Image:
    radius = 2.0 + 3.0 * strength
    amount = 0.6 + 1.7 * strength
    return Image.fromarray(_unsharp(np.array(image.convert("RGB")), radius, amount))


def _model_lens_blur(image: Image.Image, strength: float, _angle: float) -> Image.Image:
    """Mild deblur for slightly-soft images (wide-aperture lens softness)."""
    psf = disk_psf(1.0 + 3.0 * strength)
    return _ai_deblur_with_blend(image, strength, model_strength=0.55,
                                 fallback_psf=psf, fallback_iters_max=14)


def _model_motion_blur(image: Image.Image, strength: float, angle: float) -> Image.Image:
    """Anti-shake. NAFNet handles arbitrary motion — angle is unused for AI
    but still drives the fallback PSF."""
    psf = motion_psf(int(round(3 + 12 * strength)), angle)
    return _ai_deblur_with_blend(image, strength, model_strength=0.85,
                                 fallback_psf=psf, fallback_iters_max=17)


def _model_refocus(image: Image.Image, strength: float, _angle: float) -> Image.Image:
    """Heavier recovery for genuinely out-of-focus shots. Full-strength
    NAFNet + a final fine-detail pass."""
    psf = disk_psf(2.0 + 5.0 * strength)
    deblurred = _ai_deblur_with_blend(image, strength, model_strength=1.0,
                                      fallback_psf=psf, fallback_iters_max=28)
    arr = np.array(deblurred)
    final = _multi_scale_sharpen(arr, strength * 0.35)
    return Image.fromarray(final)


_METHODS = {
    "standard":    _model_standard,
    "strong":      _model_strong,
    "lens_blur":   _model_lens_blur,
    "motion_blur": _model_motion_blur,
    "refocus":     _model_refocus,
}


class SharpenModel:
    def __init__(self):
        pass

    def process(self, image: Image.Image, strength: float = 0.5,
                model: str = "standard", motion_angle: float = 0.0) -> Image.Image:
        if strength <= 0.01:
            return image
        fn = _METHODS.get(model, _model_standard)
        return fn(image, float(strength), float(motion_angle))
