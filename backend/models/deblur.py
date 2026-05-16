"""
Deblur — recover sharpness lost to actual blur (motion / defocus).

Unlike sharpening (which boosts edge contrast), deblur uses deconvolution
to mathematically invert the blur process. We use Wiener deconvolution with
a known PSF (point-spread function).

Stage 1 (current): classical Wiener deconv with user-set kernel.
  - motion_shake: linear motion PSF (length + angle)
  - defocus:      disk-shaped PSF (radius)
Stage 2 (later): AI deblur (Restormer / NAFNet) — auto-estimates PSF.

References:
  Wiener filter: https://en.wikipedia.org/wiki/Wiener_deconvolution
"""

import cv2
import numpy as np
from PIL import Image


# ── PSF generators ──────────────────────────────────────────────

def motion_psf(length: int, angle_deg: float) -> np.ndarray:
    """Linear motion blur kernel (1-pixel-wide line at given angle)."""
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
    """Defocus blur kernel: uniform disk of given radius."""
    r = max(1.0, float(radius))
    size = int(np.ceil(r * 2)) + 1
    if size % 2 == 0:
        size += 1
    c = size // 2
    y, x = np.ogrid[-c:c + 1, -c:c + 1]
    psf = (x * x + y * y <= r * r).astype(np.float32)
    s = psf.sum()
    return psf / s if s > 0 else psf


# ── Wiener deconvolution ────────────────────────────────────────

def _wiener_single_channel(img: np.ndarray, psf: np.ndarray, K: float) -> np.ndarray:
    """FFT-based Wiener deconv on a single 2D channel (float32, 0..1)."""
    H, W = img.shape
    psf_padded = np.zeros((H, W), dtype=np.float32)
    ph, pw = psf.shape
    psf_padded[:ph, :pw] = psf
    # Center PSF so it doesn't shift the image
    psf_padded = np.roll(psf_padded, (-ph // 2, -pw // 2), axis=(0, 1))

    G = np.fft.fft2(img)
    H_fft = np.fft.fft2(psf_padded)
    H_conj = np.conj(H_fft)
    # Wiener: F = G * H* / (|H|^2 + K)
    denom = H_fft * H_conj + K
    F = G * H_conj / denom
    result = np.real(np.fft.ifft2(F))
    return result


def wiener_deconv(img_rgb: np.ndarray, psf: np.ndarray, K: float = 0.01) -> np.ndarray:
    """Apply Wiener deconvolution channel-wise. img_rgb uint8 -> uint8."""
    f = img_rgb.astype(np.float32) / 255.0
    out = np.zeros_like(f)
    for c in range(f.shape[2]):
        out[..., c] = _wiener_single_channel(f[..., c], psf, K)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


# ── Public class ────────────────────────────────────────────────

class DeblurModel:
    def __init__(self):
        pass

    def process(self, image: Image.Image, mode: str = "motion_shake",
                strength: float = 0.5, motion_angle: float = 0.0) -> Image.Image:
        """strength 0..1 controls blur-kernel size.
        - motion_shake: kernel length grows from 5 to 25 px
        - defocus:      disk radius grows from 1.5 to 8 px
        K is held small to favor sharpness; for very noisy input, sharpen
        after deblur or denoise first."""
        if strength <= 0.01:
            return image

        arr = np.array(image)
        h, w = arr.shape[:2]

        if mode == "defocus":
            radius = 1.5 + (8.0 - 1.5) * strength
            psf = disk_psf(radius)
        else:   # motion_shake
            length = int(round(5 + (25 - 5) * strength))
            psf = motion_psf(length, motion_angle)

        # K: regularization. Smaller K = sharper but more ringing/noise.
        # We keep it modest so artifacts stay tolerable.
        K = 0.01 + (0.005) * (1.0 - strength)  # 0.01..0.015

        out = wiener_deconv(arr, psf, K=K)
        return Image.fromarray(out)
