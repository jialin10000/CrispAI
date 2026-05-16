"""
Sharpening / deblurring.

Stage 1 (current): unsharp mask with mode-specific radius — fast, good for
  baseline quality. Three modes mirror Topaz's UI:
    - focus_blur : small radius (1.2-2.5)  -- recover soft focus / mild blur
    - motion_blur: large radius (4-10)     -- camera shake / motion smear
    - auto       : medium (2-4)            -- general "make it look crisper"
Stage 2 (later): Restormer / NAFNet for deblur tasks.
"""

import cv2
import numpy as np
from PIL import Image


def _unsharp_mask(rgb: np.ndarray, radius: float, amount: float) -> np.ndarray:
    """Standard unsharp mask. radius=sigma, amount=multiplier of high-pass."""
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=radius, sigmaY=radius)
    sharp = cv2.addWeighted(rgb, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


class SharpenModel:
    def __init__(self):
        pass

    def process(self, image: Image.Image, mode: str = "auto",
                strength: float = 0.5) -> Image.Image:
        """strength: 0.0 (no-op) -> 1.0 (strong)."""
        if strength <= 0.01:
            return image

        # Mode -> (radius_min, radius_max, amount_max)
        params = {
            "focus_blur":  (1.2, 2.5, 1.8),   # small radius, high amount
            "motion_blur": (4.0, 10.0, 2.2),  # large radius, very high amount
            "auto":        (2.0, 4.0, 1.5),
        }
        rmin, rmax, amax = params.get(mode, params["auto"])

        # strength 0..1 -> radius rmin..rmax, amount 0.3..amax
        radius = rmin + (rmax - rmin) * strength
        amount = 0.3 + (amax - 0.3) * strength

        arr = np.array(image)  # RGB uint8
        sharp = _unsharp_mask(arr, radius, amount)
        return Image.fromarray(sharp)
