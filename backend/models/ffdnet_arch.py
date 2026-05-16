"""
FFDNet (Fast and Flexible Denoising Network) — self-contained architecture.

Reference:
    Zhang, K., Zuo, W., Zhang, L.
    "FFDNet: Toward a fast and flexible solution for CNN-based image denoising"
    IEEE TIP 2018.

This implementation is adapted from cszn/KAIR (MIT License) and stripped of
external dependencies so it loads cleanly without basicsr / KAIR utils.
The state-dict layout is preserved so official pretrained weights load directly.
"""

import torch
import torch.nn as nn


# ── pixel unshuffle (KAIR-compatible) ───────────────────────────

def _pixel_unshuffle(x: torch.Tensor, r: int) -> torch.Tensor:
    """Inverse of nn.PixelShuffle. (B, C, rH, rW) -> (B, C*r^2, H, W)."""
    B, C, H, W = x.shape
    out_h, out_w = H // r, W // r
    x = x.contiguous().view(B, C, out_h, r, out_w, r)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(B, C * r * r, out_h, out_w)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        return _pixel_unshuffle(x, self.r)


# ── conv helper matching KAIR's 'CR'/'C' mode strings ───────────

def _conv(in_c, out_c, mode="CR"):
    """KAIR-style: mode='C' = Conv only, 'CR' = Conv + ReLU."""
    layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True)]
    if "R" in mode:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


# ── FFDNet ──────────────────────────────────────────────────────

class FFDNet(nn.Module):
    """
    FFDNet for color denoising:
        in_nc=3, out_nc=3, nc=96 channels, nb=12 conv layers, ReLU activations.
    Input  : (B, 3, H, W) image + (B, 1, 1, 1) noise level (sigma).
    Output : (B, 3, H, W) denoised image (residual learned internally).
    """

    def __init__(self, in_nc: int = 3, out_nc: int = 3, nc: int = 96, nb: int = 12):
        super().__init__()
        sf = 2  # KAIR uses 2x reversible down/up sampling

        self.m_down = PixelUnShuffle(sf)

        # head: takes (in_nc*4 + 1 sigma channel) -> nc, with ReLU
        m_head = _conv(in_nc * sf * sf + 1, nc, mode="CR")
        # body: nb-2 conv+ReLU blocks
        m_body = [_conv(nc, nc, mode="CR") for _ in range(nb - 2)]
        # tail: conv only, output nc -> out_nc*4
        m_tail = _conv(nc, out_nc * sf * sf, mode="C")

        # Flat sequential to match KAIR's state-dict keys (`model.0`, `model.1` ...)
        flat = []
        for m in [m_head, *m_body, m_tail]:
            if isinstance(m, nn.Sequential):
                flat.extend(list(m.children()))
            else:
                flat.append(m)
        self.model = nn.Sequential(*flat)

        self.m_up = nn.PixelShuffle(sf)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Pad H, W to even
        h, w = x.shape[-2:]
        pad_h = (h + 1) // 2 * 2 - h
        pad_w = (w + 1) // 2 * 2 - w
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        # 2x reversible downsampling — preserves all info
        x = self.m_down(x)

        # Broadcast sigma to a per-pixel map at downsampled resolution
        sigma_map = sigma.repeat(1, 1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, sigma_map], dim=1)

        # Body
        x = self.model(x)

        # 2x upsampling back to original
        x = self.m_up(x)
        if pad_h or pad_w:
            x = x[..., :h, :w]
        return x
