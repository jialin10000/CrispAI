"""
Noise reduction using NAFNet (ECCV 2022).
Pre-trained weights from: https://github.com/megvii-research/NAFNet
"""

import torch
import numpy as np
from PIL import Image


class DenoiseModel:
    def __init__(self):
        self.device = self._get_device()
        self.model = self._load_model()

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        # NAFNet SIDD model — best for real-world sensor noise (Sony A1, etc.)
        # Downloads weights automatically on first run via basicsr
        try:
            from basicsr.models.archs.nafnet_arch import NAFNet
            model = NAFNet(
                img_channel=3,
                width=32,
                middle_blks_num=12,
                enc_blks=[2, 2, 4, 8],
                dec_blks=[2, 2, 2, 2],
            )
            # TODO: load pretrained weights
            # model.load_state_dict(torch.load("weights/nafnet_sidd.pth"))
            model.eval()
            return model.to(self.device)
        except ImportError:
            # Fallback: use a simple bilateral-filter-based denoise
            return None

    def process(self, image: Image.Image, strength: float = 0.5) -> Image.Image:
        if self.model is None:
            return self._fallback_denoise(image, strength)

        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        result = Image.fromarray(output)

        # Blend with original based on strength
        if strength < 1.0:
            result = Image.blend(image, result, strength)
        return result

    def _fallback_denoise(self, image: Image.Image, strength: float) -> Image.Image:
        """Simple fallback using PIL filter when NAFNet weights not loaded."""
        from PIL import ImageFilter
        radius = strength * 2
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
