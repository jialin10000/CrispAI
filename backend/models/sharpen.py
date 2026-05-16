"""
Sharpen — multiple methods, Topaz-style.

Models:
  standard    - Unsharp mask. Good general sharpening.
  strong      - High-pass detail boost. More aggressive than standard.
  edge_aware  - Sharpens only along strong edges, leaves flat regions alone
                (avoids amplifying noise in skies, skin, etc).
  clarity     - Local contrast enhancement via CLAHE on L channel
                (mid-frequency detail, Lightroom-style "Clarity").
"""

import cv2
import numpy as np
from PIL import Image


def _unsharp(rgb: np.ndarray, radius: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=radius, sigmaY=radius)
    out = cv2.addWeighted(rgb, 1 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen_standard(rgb: np.ndarray, strength: float) -> np.ndarray:
    radius = 1.5 + 2.0 * strength    # 1.5..3.5
    amount = 0.3 + 1.5 * strength    # 0.3..1.8
    return _unsharp(rgb, radius, amount)


def _sharpen_strong(rgb: np.ndarray, strength: float) -> np.ndarray:
    # High-pass: subtract a heavily blurred copy, add back scaled
    radius = 3.0 + 5.0 * strength   # 3..8
    amount = 0.5 + 2.0 * strength   # 0.5..2.5
    return _unsharp(rgb, radius, amount)


def _sharpen_edge_aware(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Only sharpen along edges. Build a soft edge mask from gradient
    magnitude, blend sharpened version with original using that mask."""
    radius = 1.2 + 1.8 * strength
    amount = 0.5 + 1.8 * strength
    sharp = _unsharp(rgb, radius, amount)

    # Edge mask from gradient magnitude (per-pixel)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # Normalize and soften — keep mostly the strongest edges
    mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=1.5)
    mag = np.clip(mag / max(mag.max(), 1e-6), 0, 1)
    mask = mag ** 0.7   # gamma to widen mid-edge range slightly
    mask = np.dstack([mask] * 3).astype(np.float32)

    out = sharp.astype(np.float32) * mask + rgb.astype(np.float32) * (1 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen_clarity(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Local-contrast / clarity — boost mid-frequency detail via CLAHE on L."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clip = 1.0 + 3.0 * strength    # 1..4
    tile = 8
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    L2 = clahe.apply(L)
    # Blend original L vs CLAHE-L using strength so low strength is subtle
    L_out = cv2.addWeighted(L, 1 - strength * 0.7, L2, strength * 0.7, 0)
    out = cv2.merge([L_out, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


_METHODS = {
    "standard":   _sharpen_standard,
    "strong":     _sharpen_strong,
    "edge_aware": _sharpen_edge_aware,
    "clarity":    _sharpen_clarity,
}


class SharpenModel:
    def __init__(self):
        pass

    def process(self, image: Image.Image, strength: float = 0.5,
                model: str = "standard") -> Image.Image:
        if strength <= 0.01:
            return image
        fn = _METHODS.get(model, _sharpen_standard)
        arr = np.array(image)
        out = fn(arr, float(strength))
        return Image.fromarray(out)
