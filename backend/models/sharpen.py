"""
Sharpen — multiple methods, Topaz-style.

Models:
  fine_detail - Multi-scale unsharp (small + medium radius). Best for fine
                texture enhancement without coarse halos. Default.
  standard    - Classic unsharp mask, small radius. General-purpose.
  strong      - Larger radius unsharp for big-feature contrast boost.
  edge_aware  - Sharpens only along strong edges (Sobel mask). Skips
                flat regions to avoid amplifying noise (sky, skin, etc).
  clarity     - CLAHE on LAB-L channel. Lightroom-style local contrast.
"""

import cv2
import numpy as np
from PIL import Image


def _unsharp(rgb: np.ndarray, radius: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=radius, sigmaY=radius)
    out = cv2.addWeighted(rgb, 1 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen_fine_detail(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Multi-scale: extract fine + medium detail layers, boost both.
    Gives Topaz-like fine-texture enhancement without big halos."""
    f = rgb.astype(np.float32)
    fine_blur   = cv2.GaussianBlur(f, (0, 0), sigmaX=0.6)
    medium_blur = cv2.GaussianBlur(f, (0, 0), sigmaX=2.0)
    fine_detail   = f - fine_blur
    medium_detail = fine_blur - medium_blur
    # Amounts scale with strength; fine gets more boost than medium
    fine_amt   = 0.4 + 2.0 * strength    # 0.4..2.4
    medium_amt = 0.2 + 0.8 * strength    # 0.2..1.0
    out = f + fine_amt * fine_detail + medium_amt * medium_detail
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen_standard(rgb: np.ndarray, strength: float) -> np.ndarray:
    radius = 0.8 + 1.0 * strength    # 0.8..1.8 — kept small
    amount = 0.3 + 1.2 * strength    # 0.3..1.5
    return _unsharp(rgb, radius, amount)


def _sharpen_strong(rgb: np.ndarray, strength: float) -> np.ndarray:
    radius = 2.0 + 3.0 * strength    # 2..5
    amount = 0.5 + 1.8 * strength    # 0.5..2.3
    return _unsharp(rgb, radius, amount)


def _sharpen_edge_aware(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Sharpen only along edges; flat areas (sky / skin) stay clean."""
    radius = 0.9 + 1.0 * strength
    amount = 0.4 + 1.6 * strength
    sharp = _unsharp(rgb, radius, amount)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=1.5)
    mag = np.clip(mag / max(mag.max(), 1e-6), 0, 1)
    mask = (mag ** 0.7).astype(np.float32)
    mask3 = np.dstack([mask] * 3)
    out = sharp.astype(np.float32) * mask3 + rgb.astype(np.float32) * (1 - mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen_clarity(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Mid-frequency local contrast via CLAHE on the L channel."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clip = 1.0 + 3.0 * strength    # 1..4
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    L_out = cv2.addWeighted(L, 1 - strength * 0.7, L2, strength * 0.7, 0)
    out = cv2.merge([L_out, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


_METHODS = {
    "fine_detail": _sharpen_fine_detail,
    "standard":    _sharpen_standard,
    "strong":      _sharpen_strong,
    "edge_aware":  _sharpen_edge_aware,
    "clarity":     _sharpen_clarity,
}


class SharpenModel:
    def __init__(self):
        pass

    def process(self, image: Image.Image, strength: float = 0.5,
                model: str = "fine_detail") -> Image.Image:
        if strength <= 0.01:
            return image
        fn = _METHODS.get(model, _sharpen_fine_detail)
        arr = np.array(image)
        out = fn(arr, float(strength))
        return Image.fromarray(out)
