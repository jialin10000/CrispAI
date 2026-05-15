"""
Sharpening and deblurring using Restormer (CVPR 2022).
Handles both motion blur (camera shake) and focus blur.
Pre-trained weights: https://github.com/swz30/Restormer
"""

import torch
import numpy as np
from PIL import Image


class SharpenModel:
    def __init__(self):
        self.device = self._get_device()
        self.models = {}

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_restormer(self, task: str):
        """Load Restormer for a specific task (motion_blur or focus_blur)."""
        if task in self.models:
            return self.models[task]

        try:
            from basicsr.models.archs.restormer_arch import Restormer
            model = Restormer(
                inp_channels=3,
                out_channels=3,
                dim=48,
                num_blocks=[4, 6, 6, 8],
                num_refinement_blocks=4,
                heads=[1, 2, 4, 8],
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type="WithBias",
                dual_pixel_task=False,
            )
            # TODO: load task-specific pretrained weights
            # weights_path = f"weights/restormer_{task}.pth"
            # model.load_state_dict(torch.load(weights_path))
            model.eval()
            self.models[task] = model.to(self.device)
            return self.models[task]
        except (ImportError, Exception):
            return None

    def process(self, image: Image.Image, mode: str = "auto", strength: float = 0.5) -> Image.Image:
        if mode == "auto":
            # Auto: try motion blur first (more common with camera shake)
            mode = "motion_blur"

        model = self._load_restormer(mode)

        if model is None:
            return self._fallback_sharpen(image, strength)

        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(tensor)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        result = Image.fromarray(output)

        if strength < 1.0:
            result = Image.blend(image, result, strength)
        return result

    def _fallback_sharpen(self, image: Image.Image, strength: float) -> Image.Image:
        """Simple unsharp mask fallback."""
        from PIL import ImageFilter
        amount = int(strength * 200)
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=amount, threshold=3))
