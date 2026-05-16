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

        # NAFNet is deep enough that fp16 accumulates noticeable error and
        # can produce extreme colour-checkerboard artefacts on real photos.
        # We keep the model in fp32 — RTX-class GPUs still fit 50 MP tiles
        # comfortably and inference stays well under 1 s per 1 MP tile.
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
        with cosine-weighted blending across tile boundaries — but the
        image's OUTER edges keep window=1 so they don't get divided by a
        near-zero weight (which would amplify AI residual noise into the
        rainbow checkerboard artefact we saw on real photos)."""
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        H, W, _ = arr.shape

        # Easy case: image small enough for a single forward pass
        if H <= self.tile and W <= self.tile:
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            out = self._infer_tile(x)
            out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            return Image.fromarray((out * 255).round().astype(np.uint8))

        tile = self.tile
        ov = self.overlap
        stride = tile - ov

        # Pad up to a multiple that the tile grid covers
        pad_h = max(0, (math.ceil(max(0, H - ov) / stride) * stride + ov) - H)
        pad_w = max(0, (math.ceil(max(0, W - ov) / stride) * stride + ov) - W)
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        Hp, Wp, _ = arr.shape

        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, ov))).astype(np.float32)

        def tile_window(top_ramp: bool, bottom_ramp: bool,
                        left_ramp: bool, right_ramp: bool) -> np.ndarray:
            """Window that is 1 in the centre and ramps to 0 only on the
            edges that overlap a neighbour tile. Outer-image edges stay 1."""
            w = np.ones((tile, tile), dtype=np.float32)
            if top_ramp:
                w[:ov, :]  *= ramp[:, None]
            if bottom_ramp:
                w[-ov:, :] *= ramp[::-1][:, None]
            if left_ramp:
                w[:, :ov]  *= ramp[None, :]
            if right_ramp:
                w[:, -ov:] *= ramp[None, ::-1]
            return w

        y_positions = list(range(0, Hp - tile + 1, stride))
        x_positions = list(range(0, Wp - tile + 1, stride))
        # Ensure final tile reaches the bottom/right edge exactly
        if y_positions[-1] + tile < Hp:
            y_positions.append(Hp - tile)
        if x_positions[-1] + tile < Wp:
            x_positions.append(Wp - tile)

        out_acc    = np.zeros_like(arr)
        weight_acc = np.zeros_like(arr)

        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                top_r    = i != 0
                bottom_r = i != len(y_positions) - 1
                left_r   = j != 0
                right_r  = j != len(x_positions) - 1
                win = tile_window(top_r, bottom_r, left_r, right_r)[..., None]

                patch = arr[y:y + tile, x:x + tile, :]
                inp = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
                out = self._infer_tile(inp).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                out_acc   [y:y + tile, x:x + tile, :] += out * win
                weight_acc[y:y + tile, x:x + tile, :] += win

        # weight_acc is now strictly >= 1 across the whole image — safe to divide.
        result = out_acc / np.maximum(weight_acc, 1e-3)
        result = result[:H, :W, :]
        result = np.clip(result, 0, 1)
        return Image.fromarray((result * 255).round().astype(np.uint8))
