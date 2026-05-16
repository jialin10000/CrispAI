"""
AI denoising wrapper around FFDNet.

Single neural net, sigma-controlled strength — perfect for Topaz-style
'Normal / Strong / Severe' levels:
    Normal  -> sigma ~15  (light denoise)
    Strong  -> sigma ~25  (medium)
    Severe  -> sigma ~50  (heavy)

The model auto-downloads on first use (~1.5MB). Inference runs on CUDA if
available, otherwise CPU (slower but works). If anything fails, callers
should catch and fall back to classical denoising.
"""

import os
import logging
import urllib.request

import numpy as np
import torch
from PIL import Image

from .ffdnet_arch import FFDNet

logger = logging.getLogger(__name__)

WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_color.pth"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "ffdnet_color.pth")


def _download_weights() -> str:
    """Download the FFDNet color weights if not already cached."""
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    logger.info(f"Downloading FFDNet weights -> {WEIGHTS_PATH}")
    tmp = WEIGHTS_PATH + ".part"
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        os.replace(tmp, WEIGHTS_PATH)
        logger.info(f"FFDNet weights ready ({os.path.getsize(WEIGHTS_PATH)} bytes)")
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return WEIGHTS_PATH


class AIDenoiser:
    """Singleton-ish FFDNet wrapper. Lazy-loads on first call."""

    _instance = None

    @classmethod
    def get(cls) -> "AIDenoiser":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"AI denoiser device: {self.device}")
        path = _download_weights()

        # KAIR's pretrained color model: nc=96, nb=12
        self.model = FFDNet(in_nc=3, out_nc=3, nc=96, nb=12)
        state = torch.load(path, map_location=self.device, weights_only=False)
        # Some checkpoints wrap state under 'params' / 'state_dict'
        if isinstance(state, dict) and "params" in state:
            state = state["params"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

        # Half-precision on CUDA = ~2x faster, negligible quality loss
        if self.device.type == "cuda":
            self.model = self.model.half()
            self._half = True
        else:
            self._half = False

    @torch.no_grad()
    def denoise(self, image: Image.Image, sigma: float) -> Image.Image:
        """sigma in 0..255 scale (same as KAIR convention).
        Returns a new PIL.Image, same size as input."""
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        # (H, W, 3) -> (1, 3, H, W)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        sigma_tensor = torch.tensor([[[[sigma / 255.0]]]], device=self.device)

        if self._half:
            x = x.half()
            sigma_tensor = sigma_tensor.half()

        out = self.model(x, sigma_tensor).float()
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)
        return Image.fromarray(out)
