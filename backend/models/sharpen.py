"""
Sharpen — Topaz Photo AI-style models.

Five models matching Topaz's Sharpen sub-models exactly:

  Standard    - Multi-scale unsharp. General-purpose fine-detail enhancement.
  Strong      - Heavier unsharp. Pushes contrast on bigger features.
  Lens Blur   - Richardson-Lucy deconv with a disk PSF (recovers softness
                from a slightly defocused lens — e.g. shooting wide-open).
  Motion Blur - RL deconv with a linear PSF — recovers camera shake.
                Has a motion-angle parameter (Topaz auto-detects; we let
                the user dial it in until AI estimation is implemented).
  Refocus     - Stronger Lens-Blur recovery + final fine-detail pass.
                For genuinely out-of-focus shots, not just slightly soft.

Standard / Strong work on any image (they boost contrast).
Lens Blur / Motion Blur / Refocus assume the image is actually blurred —
applying them to a sharp image creates ringing artefacts.
"""

import cv2
import numpy as np
from PIL import Image

from .deblur import disk_psf, motion_psf, richardson_lucy


# ── pure sharpen kernels ────────────────────────────────────────

def _unsharp(rgb: np.ndarray, radius: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=radius, sigmaY=radius)
    out = cv2.addWeighted(rgb, 1 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _multi_scale_sharpen(rgb: np.ndarray, strength: float,
                         fine_amt_range=(0.4, 2.4),
                         mid_amt_range=(0.2, 1.0)) -> np.ndarray:
    """Multi-scale unsharp: subtract small + medium Gaussians, boost both
    detail layers independently. Gives fine-texture lift with small halos."""
    f = rgb.astype(np.float32)
    fine_blur   = cv2.GaussianBlur(f, (0, 0), sigmaX=0.6)
    medium_blur = cv2.GaussianBlur(f, (0, 0), sigmaX=2.0)
    fine_detail   = f - fine_blur
    medium_detail = fine_blur - medium_blur
    fine_amt = fine_amt_range[0] + (fine_amt_range[1] - fine_amt_range[0]) * strength
    mid_amt  = mid_amt_range[0]  + (mid_amt_range[1]  - mid_amt_range[0])  * strength
    out = f + fine_amt * fine_detail + mid_amt * medium_detail
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Topaz-equivalent model functions ────────────────────────────

def _model_standard(rgb: np.ndarray, strength: float, _angle: float) -> np.ndarray:
    return _multi_scale_sharpen(rgb, strength)


def _model_strong(rgb: np.ndarray, strength: float, _angle: float) -> np.ndarray:
    """Larger radius, bigger amount. For coarser features."""
    radius = 2.0 + 3.0 * strength    # 2..5
    amount = 0.6 + 1.7 * strength    # 0.6..2.3
    return _unsharp(rgb, radius, amount)


def _model_lens_blur(rgb: np.ndarray, strength: float, _angle: float) -> np.ndarray:
    """Mild disk-PSF deconv — for lens softness, wide-aperture haze."""
    radius = 1.0 + 3.0 * strength       # 1..4
    iters  = int(round(4 + 10 * strength))   # 4..14
    psf = disk_psf(radius)
    return richardson_lucy(rgb, psf, iters=iters)


def _model_motion_blur(rgb: np.ndarray, strength: float, angle: float) -> np.ndarray:
    """Line-PSF deconv along `angle` (degrees) — for camera shake."""
    length = int(round(3 + 12 * strength))    # 3..15 px
    iters  = int(round(5 + 12 * strength))    # 5..17
    psf = motion_psf(length, angle)
    return richardson_lucy(rgb, psf, iters=iters)


def _model_refocus(rgb: np.ndarray, strength: float, _angle: float) -> np.ndarray:
    """Heavier disk deconv + final fine-detail pass. For seriously soft images."""
    radius = 2.0 + 5.0 * strength       # 2..7  — bigger than Lens Blur
    iters  = int(round(8 + 20 * strength))   # 8..28 — more iterations
    psf = disk_psf(radius)
    deblurred = richardson_lucy(rgb, psf, iters=iters)
    # Top off with mild multi-scale detail boost
    return _multi_scale_sharpen(deblurred, strength * 0.4)


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
        arr = np.array(image.convert("RGB"))
        out = fn(arr, float(strength), float(motion_angle))
        return Image.fromarray(out)
