"""
Noise reduction — multiple methods, Topaz-style model selection.

Models:
  normal   - Non-local means (cv2.fastNlMeansDenoisingColored). General purpose.
             Good for typical sensor noise (ISO 800-3200).
  strong   - NLM at high h + bilateral pass. For ISO 6400+ or heavy compression noise.
  severe   - Cascaded NLM + bilateral + light median. Last-resort for very dirty inputs.
  impulse  - Median filter. For salt-and-pepper noise (dead pixels, scan artifacts).

Stage 2 (later): swap any of these for NAFNet / SCUNet AI denoisers.
"""

import cv2
import numpy as np
from PIL import Image


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _denoise_normal(bgr: np.ndarray, strength: float) -> np.ndarray:
    h = 3.0 + strength * 12.0   # 3..15
    return cv2.fastNlMeansDenoisingColored(
        bgr, None, h=h, hColor=h,
        templateWindowSize=7, searchWindowSize=21,
    )


def _denoise_strong(bgr: np.ndarray, strength: float) -> np.ndarray:
    h = 10.0 + strength * 15.0   # 10..25
    out = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=h, hColor=h,
        templateWindowSize=7, searchWindowSize=25,
    )
    # Follow-up bilateral smooths residual chroma noise without killing edges
    out = cv2.bilateralFilter(out, d=5, sigmaColor=40, sigmaSpace=40)
    return out


def _denoise_severe(bgr: np.ndarray, strength: float) -> np.ndarray:
    # Cascade for very noisy input
    h1 = 8.0 + strength * 10.0
    pass1 = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=h1, hColor=h1,
        templateWindowSize=7, searchWindowSize=21,
    )
    pass2 = cv2.bilateralFilter(pass1, d=7, sigmaColor=60, sigmaSpace=60)
    # Light median catches any remaining speckle
    pass3 = cv2.medianBlur(pass2, 3)
    # Blend with pass2 so we don't over-soften
    return cv2.addWeighted(pass2, 0.6, pass3, 0.4, 0)


def _denoise_impulse(bgr: np.ndarray, strength: float) -> np.ndarray:
    # Median kernel size scales with strength: 3, 5, 7
    k = 3 if strength < 0.33 else (5 if strength < 0.66 else 7)
    return cv2.medianBlur(bgr, k)


_METHODS = {
    "normal":  _denoise_normal,
    "strong":  _denoise_strong,
    "severe":  _denoise_severe,
    "impulse": _denoise_impulse,
}


class DenoiseModel:
    def __init__(self):
        pass

    def process(self, image: Image.Image, strength: float = 0.5,
                model: str = "normal") -> Image.Image:
        if strength <= 0.01:
            return image
        fn = _METHODS.get(model, _denoise_normal)
        bgr = _to_bgr(image)
        out = fn(bgr, float(strength))
        return _to_pil(out)
