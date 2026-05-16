"""
Noise reduction.

Stage 1 (current): OpenCV fastNlMeansDenoisingColored — high quality,
  good detail preservation, no GPU needed, no model weights.
Stage 2 (later): swap in NAFNet / SCUNet via PyTorch when weights are bundled.
"""

import cv2
import numpy as np
from PIL import Image


class DenoiseModel:
    def __init__(self):
        pass

    def process(self, image: Image.Image, strength: float = 0.5) -> Image.Image:
        """strength: 0.0 (no-op) -> 1.0 (strong denoise).

        h controls luminance noise removal (0=no effect, ~10=moderate, ~20=strong).
        hColor controls chrominance noise (color speckle); we keep it close to h.
        templateWindowSize=7, searchWindowSize=21 are the OpenCV defaults — good
        balance of quality vs speed.
        """
        if strength <= 0.01:
            return image

        # Map 0..1 to h in [3..15] — 15 is already quite aggressive
        h = 3.0 + strength * 12.0

        arr = np.array(image)  # RGB uint8
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        denoised = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            h=h,
            hColor=h,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
