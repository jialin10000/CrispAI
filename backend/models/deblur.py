"""
Deblur — recover sharpness from actual blur (motion / defocus).

This is fundamentally an inverse problem: given a blurred image and an
assumed point-spread function (PSF), try to recover the original.

Stage 1 (current): Richardson-Lucy iterative deconvolution.
  Gentler than Wiener — converges towards a maximum-likelihood
  reconstruction without explicit noise regularisation. A few iterations
  give a mild lift; too many introduce ringing / amplify noise.
Stage 2 (later): AI deblur (Restormer / NAFNet) — auto-estimates PSF,
  hallucinates plausible detail. Far better on real shake/defocus.

Note: deconvolution on an already-sharp image (no real blur) will always
introduce artefacts because the algorithm tries to "uninvent" blur that
isn't there. Deblur is for genuinely-blurred input.
"""

import cv2
import numpy as np
from PIL import Image


# ── PSF generators ──────────────────────────────────────────────

def motion_psf(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    size = length
    psf = np.zeros((size, size), dtype=np.float32)
    cx = cy = size // 2
    angle = np.deg2rad(angle_deg)
    half = length // 2
    for t in np.linspace(-half, half, num=length * 2):
        x = int(round(cx + t * np.cos(angle)))
        y = int(round(cy + t * np.sin(angle)))
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1.0
    total = psf.sum()
    if total > 0:
        psf /= total
    else:
        psf[cy, cx] = 1.0
    return psf


def disk_psf(radius: float) -> np.ndarray:
    r = max(1.0, float(radius))
    size = int(np.ceil(r * 2)) + 1
    if size % 2 == 0:
        size += 1
    c = size // 2
    y, x = np.ogrid[-c:c + 1, -c:c + 1]
    psf = (x * x + y * y <= r * r).astype(np.float32)
    s = psf.sum()
    return psf / s if s > 0 else psf


# ── Richardson-Lucy iterative deconvolution ─────────────────────

def _rl_single_channel(img: np.ndarray, psf: np.ndarray, iters: int) -> np.ndarray:
    """Richardson-Lucy on one 2D channel (float32, range 0..1).
    img and psf-flipped are convolved using OpenCV's filter2D (fast)."""
    psf_flipped = np.flipud(np.fliplr(psf))
    est = np.full_like(img, img.mean())   # start from mean — softer than starting from img
    eps = 1e-7

    for _ in range(iters):
        reblur = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REPLICATE)
        ratio  = img / (reblur + eps)
        corr   = cv2.filter2D(ratio, -1, psf_flipped, borderType=cv2.BORDER_REPLICATE)
        est    = est * corr
        # Keep estimate bounded
        np.clip(est, 0, 1.0, out=est)
    return est


def richardson_lucy(img_rgb: np.ndarray, psf: np.ndarray, iters: int = 10) -> np.ndarray:
    """Apply RL deconv channel-wise. uint8 -> uint8."""
    f = img_rgb.astype(np.float32) / 255.0
    out = np.zeros_like(f)
    for c in range(f.shape[2]):
        out[..., c] = _rl_single_channel(f[..., c], psf, iters)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


# ── Public class ────────────────────────────────────────────────

class DeblurModel:
    def __init__(self):
        pass

    def process(self, image: Image.Image, mode: str = "motion_shake",
                strength: float = 0.5, motion_angle: float = 0.0) -> Image.Image:
        """Strength controls both PSF size AND iteration count.
        Mild defaults are tuned to give a noticeable but artefact-free lift
        on actually-blurred input; over-applying still introduces ringing."""
        if strength <= 0.01:
            return image

        if mode == "defocus":
            radius = 1.0 + 4.0 * strength      # 1..5  (was 1.5..8 — too aggressive)
            psf = disk_psf(radius)
        else:   # motion_shake
            length = int(round(3 + 12 * strength))   # 3..15  (was 5..25)
            psf = motion_psf(length, motion_angle)

        # 4..16 iterations — more = sharper but more ringing
        iters = int(round(4 + 12 * strength))

        arr = np.array(image)
        out = richardson_lucy(arr, psf, iters=iters)
        return Image.fromarray(out)
