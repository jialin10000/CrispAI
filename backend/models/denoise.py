"""
Denoise — NAFNet-SIDD + detail-aware chroma-favouring blending.

The naive approach (lerp original and full NAFNet output by strength) over-
smooths fine detail even at low strength, because NAFNet's output is
*always* smoothed — there's no "low-noise mode" baked into the model. Even
5 % of that smoothing visibly degrades hair, eyelashes, fabric texture.

The Topaz-style approach: process in YCbCr and treat the two halves
differently.
  - Luminance (Y) carries all the perceived detail. We blend the AI's
    Y only in FLAT regions (low gradient) — edges and textures keep the
    original Y untouched. Detail mask = clamp(|grad|/threshold, 0, 1).
  - Chrominance (Cb, Cr) is where the ugly colour blotches live and where
    our eyes barely see detail. Blend aggressively proportional to user
    strength — chroma noise gets cleaned regardless of edges.

Three models map to different aggression levels:
  Normal   - mild  : luma cap 0.20 * strength, chroma 0.60 * strength
  Strong   - mid   : luma cap 0.35 * strength, chroma 0.85 * strength
  Extreme  - heavy : luma cap 0.55 * strength, chroma 1.00 * strength
  Impulse  - kept on median filter (AI doesn't help salt-and-pepper noise)

Cascading fallbacks if NAFNet unavailable:
  NAFNet -> FFDNet -> OpenCV fastNlMeansDenoisingColored
"""

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# (luma_max, chroma_max). Multiplied by user strength (0..1) to get the
# per-pixel blend amount in flat regions / chroma. Edges in luma are
# always preserved via the detail mask.
_BLEND_BY_MODEL = {
    "normal":  (0.20, 0.60),
    "strong":  (0.35, 0.85),
    "extreme": (0.55, 1.00),
    "severe":  (0.55, 1.00),  # legacy alias
}


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _classical_fallback(image: Image.Image, sigma: float) -> Image.Image:
    h = sigma * 0.32
    bgr = _to_bgr(image)
    out = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=h, hColor=h,
        templateWindowSize=7, searchWindowSize=21,
    )
    return _to_pil(out)


def _denoise_impulse(image: Image.Image, strength: float) -> Image.Image:
    k = 3 if strength < 0.33 else (5 if strength < 0.66 else 7)
    bgr = _to_bgr(image)
    return _to_pil(cv2.medianBlur(bgr, k))


def _detail_aware_merge(original: Image.Image, denoised: Image.Image,
                        luma_max: float, chroma_max: float,
                        strength: float) -> Image.Image:
    """Merge the AI denoise into the original using a detail-aware mask
    on luma and full-strength blend on chroma.

    - luma_max  : max luma-blend amount in perfectly-flat regions
    - chroma_max: max chroma-blend amount (no edge masking)
    - strength  : user 0..1, scales both maxima linearly

    In flat regions both are at their max. In high-gradient regions luma
    decays to 0 (original Y preserved) while chroma stays high."""

    # Y, Cb, Cr — PIL's YCbCr matches BT.601, 0..255 range
    img_y, img_cb, img_cr = [np.array(c, dtype=np.float32)
                             for c in original.convert("YCbCr").split()]
    ai_y,  ai_cb,  ai_cr  = [np.array(c, dtype=np.float32)
                             for c in denoised.convert("YCbCr").split()]

    # Detail mask from Y gradient. 0 = flat (keep AI luma blend), 1 = edge (preserve).
    gx = cv2.Sobel(img_y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_y, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    # Light blur so single-pixel impulses don't punch holes in flat regions.
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.2)
    # Threshold tuned for natural photos: ~25 = strong edges, ~5 = fine texture.
    # Gamma 0.6 widens the "considered detail" zone so fine texture stays.
    detail = np.clip(edge / 22.0, 0.0, 1.0) ** 0.6

    # Per-pixel luma blend = (1 - detail) * luma_max * strength
    luma_alpha   = (1.0 - detail) * (luma_max   * strength)
    chroma_alpha = np.float32(chroma_max * strength)

    out_y  = img_y  * (1 - luma_alpha)   + ai_y  * luma_alpha
    out_cb = img_cb * (1 - chroma_alpha) + ai_cb * chroma_alpha
    out_cr = img_cr * (1 - chroma_alpha) + ai_cr * chroma_alpha

    merged = np.stack([np.clip(out_y,  0, 255),
                       np.clip(out_cb, 0, 255),
                       np.clip(out_cr, 0, 255)], axis=-1).astype(np.uint8)
    return Image.fromarray(merged, mode="YCbCr").convert("RGB")


class DenoiseModel:
    """Lazy-loads NAFNet on first call. Holds the runner in memory after that."""

    def __init__(self):
        self._nafnet = None
        self._ffdnet = None
        self._nafnet_attempted = False
        self._ffdnet_attempted = False

    def _get_nafnet(self):
        if self._nafnet is not None:
            return self._nafnet
        if self._nafnet_attempted:
            return None
        self._nafnet_attempted = True
        try:
            from .nafnet_runner import NAFNetRunner
            self._nafnet = NAFNetRunner.get("denoise")
            logger.info("NAFNet-SIDD denoiser loaded")
        except Exception as e:
            logger.warning(f"NAFNet denoise unavailable, will try FFDNet fallback: {e}")
        return self._nafnet

    def _get_ffdnet(self):
        if self._ffdnet is not None:
            return self._ffdnet
        if self._ffdnet_attempted:
            return None
        self._ffdnet_attempted = True
        try:
            from .ai_denoise import AIDenoiser
            self._ffdnet = AIDenoiser.get()
            logger.info("FFDNet fallback denoiser loaded")
        except Exception as e:
            logger.warning(f"FFDNet fallback also unavailable: {e}")
        return self._ffdnet

    def process(self, image: Image.Image, strength: float = 0.5,
                model: str = "normal") -> Image.Image:
        if strength <= 0.01:
            return image
        if model == "impulse":
            return _denoise_impulse(image, strength)

        base_image = image.convert("RGB")
        luma_max, chroma_max = _BLEND_BY_MODEL.get(model, (0.20, 0.60))

        # Stage 1: NAFNet (best quality)
        runner = self._get_nafnet()
        if runner is not None:
            try:
                denoised = runner.run(base_image)
                return _detail_aware_merge(base_image, denoised,
                                            luma_max, chroma_max, strength)
            except Exception as e:
                logger.error(f"NAFNet inference failed: {e}")

        # Stage 2: FFDNet fallback (older AI, smaller)
        ffd = self._get_ffdnet()
        if ffd is not None:
            try:
                sigma_map = {"normal": 15.0, "strong": 25.0, "extreme": 50.0, "severe": 50.0}
                sigma = sigma_map.get(model, 15.0)
                denoised = ffd.denoise(base_image, sigma)
                return _detail_aware_merge(base_image, denoised,
                                            luma_max, chroma_max, strength)
            except Exception as e:
                logger.error(f"FFDNet inference failed: {e}")

        # Stage 3: classical OpenCV NLM (still detail-merged for consistency)
        sigma_map = {"normal": 15.0, "strong": 25.0, "extreme": 50.0, "severe": 50.0}
        cl = _classical_fallback(image, sigma_map.get(model, 15.0))
        return _detail_aware_merge(base_image, cl, luma_max, chroma_max, strength)
