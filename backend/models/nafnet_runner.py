"""
NAFNet runner: lazy-loads SIDD (denoise) or GoPro (deblur) weights, runs
fp16 GPU inference on PIL images. One singleton per task.

Weight files (~70 MB each) auto-download from Google Drive on first use
via gdown. They're cached under backend/models/weights/ and gitignored.
"""

import os
import logging
import math

import numpy as np
import torch
from PIL import Image

from .nafnet_arch import NAFNet_SIDD_width64, NAFNet_GoPro_width64

logger = logging.getLogger(__name__)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Google Drive file IDs from megvii-research/NAFNet README
_WEIGHTS = {
    "denoise": {
        "filename":    "NAFNet-SIDD-width64.pth",
        "gdrive_id":   "14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR",
        "human_name":  "NAFNet-SIDD (denoise)",
        "arch_factory": NAFNet_SIDD_width64,
    },
    "deblur": {
        "filename":    "NAFNet-GoPro-width64.pth",
        "gdrive_id":   "1S0PVRbyTakYY9a82kujgZLbMihfNBLfC",
        "human_name":  "NAFNet-GoPro (deblur)",
        "arch_factory": NAFNet_GoPro_width64,
    },
}


def _download_gdrive(file_id: str, dest_path: str, human_name: str) -> str:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {human_name} -> {dest_path}")
    import gdown
    tmp = dest_path + ".part"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, tmp, quiet=False)
        if not os.path.exists(tmp) or os.path.getsize(tmp) < 1024 * 1024:
            raise RuntimeError("download produced an unexpectedly small file")
        os.replace(tmp, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"{human_name} ready ({size_mb:.1f} MB)")
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dest_path


def _load_state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("params", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


class NAFNetRunner:
    """One instance per task ('denoise' or 'deblur'). Use NAFNetRunner.get(task)."""

    _instances: dict = {}

    @classmethod
    def get(cls, task: str) -> "NAFNetRunner":
        if task not in _WEIGHTS:
            raise ValueError(f"unknown NAFNet task: {task}")
        if task not in cls._instances:
            cls._instances[task] = NAFNetRunner(task)
        return cls._instances[task]

    def __init__(self, task: str):
        info = _WEIGHTS[task]
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"NAFNet {task!r} -> device={self.device}")

        path = _download_gdrive(
            file_id=info["gdrive_id"],
            dest_path=os.path.join(WEIGHTS_DIR, info["filename"]),
            human_name=info["human_name"],
        )

        self.model = info["arch_factory"]()
        state = _load_state_dict(path)
        # The official ckpt sometimes wraps keys with 'module.' (DDP). Strip if needed.
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

        # fp16 on CUDA halves memory + ~2x faster, negligible quality loss
        if self.device.type == "cuda":
            self.model = self.model.half()
            self._half = True
        else:
            self._half = False

        # Tiled inference to keep VRAM bounded on big photos.
        # 768 px tile + 32 px overlap fits comfortably on a 6-8 GB GPU.
        self.tile = 768
        self.overlap = 32

    # ── Tiled inference ───────────────────────────────────────────────

    @torch.no_grad()
    def _infer_tile(self, x: torch.Tensor) -> torch.Tensor:
        if self._half:
            x = x.half()
        return self.model(x).float()

    @torch.no_grad()
    def run(self, image: Image.Image) -> Image.Image:
        """Run inference on a full-resolution PIL image. Uses sliding tiles
        with cosine-weighted blending so seams are imperceptible."""
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        H, W, _ = arr.shape

        # Easy case: image small enough for a single forward pass
        if H <= self.tile and W <= self.tile:
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            out = self._infer_tile(x)
            out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            return Image.fromarray((out * 255).round().astype(np.uint8))

        # Tiled with overlap & cosine blend
        tile = self.tile
        ov = self.overlap
        stride = tile - ov
        # Pad image so tiles cover everything
        pad_h = (math.ceil((H - ov) / stride) * stride + ov) - H
        pad_w = (math.ceil((W - ov) / stride) * stride + ov) - W
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        Hp, Wp, _ = arr.shape

        # Cosine window for blending
        def _half_cosine(n: int) -> np.ndarray:
            t = np.linspace(0, math.pi, n)
            return 0.5 - 0.5 * np.cos(t)

        window = np.ones((tile, tile), dtype=np.float32)
        ramp = _half_cosine(ov)
        window[:ov, :]  *= ramp[:, None]
        window[-ov:, :] *= ramp[::-1][:, None]
        window[:, :ov]  *= ramp[None, :]
        window[:, -ov:] *= ramp[None, ::-1]
        window3 = window[..., None]

        out_acc  = np.zeros_like(arr)
        weight_acc = np.zeros_like(arr)

        for y in range(0, Hp - tile + 1, stride):
            for x in range(0, Wp - tile + 1, stride):
                patch = arr[y:y + tile, x:x + tile, :]
                inp = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
                out = self._infer_tile(inp).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                out_acc   [y:y + tile, x:x + tile, :] += out * window3
                weight_acc[y:y + tile, x:x + tile, :] += window3

        result = out_acc / np.maximum(weight_acc, 1e-6)
        result = result[:H, :W, :]
        result = np.clip(result, 0, 1)
        return Image.fromarray((result * 255).round().astype(np.uint8))
