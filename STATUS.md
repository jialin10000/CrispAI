# CrispAI — Project Status (Paused, May 2026)

Self-hosted, open-source alternative to Topaz Photo AI. Built on Sony A1
workflows (50 MP raw photos), focused on AI denoise + sharpen, no
subscription, runs entirely on local GPU.

**Status: PAUSED.** All UX / infrastructure is production-grade. The current
public AI models (NAFNet-SIDD, NAFNet-GoPro) are not good enough on
low-ISO professional camera photos to justify dropping Topaz. Project is
on hold pending either (a) better open-source pretrained models, or
(b) a decision to invest 2-3 weeks into training a Sony-A1-specific model.

## What works (all production-ready)

| Area | Status |
|------|--------|
| Desktop launcher (`CrispAI.bat` + desktop shortcut) | ✅ One-click start, auto-shutdown on window close |
| Backend (`backend/server.py`) | ✅ Flask + heartbeat lifecycle, full session API |
| Borderless app window (Chrome/Edge `--app` mode) | ✅ No browser chrome, looks native |
| File upload (drag-drop + picker) | ✅ |
| Topaz-style dark UI (`backend/static/ui.html`) | ✅ Compare view, sidebar controls, status bar |
| Compare slider (pointer drag, anywhere in image) | ✅ Original RIGHT, CrispAI LEFT (user preference) |
| Zoom (mouse wheel, +/- buttons, double-click toggle) | ✅ Anchored to cursor / viewport / clicked point |
| Full-resolution original display | ✅ A1's 8640 px shown 1:1 at 100% zoom |
| AI preview at 1600 px for slider responsiveness | ✅ ~1 s per slider tick on RTX 4070 |
| Full-resolution AI processing on demand | ✅ "Full-Quality Preview" button |
| Save As (PNG / JPG / TIFF) | ✅ Sensible filename `<orig>_crispai.<ext>` |
| Tiled AI inference with cosine-weighted blending | ✅ 768 px tiles, 32 px overlap, no seams |
| Skip-when-idle | ✅ At strength 0 / all stages off: shows original on both sides, no AI runs |

## AI algorithms in place

| Module | Implementation | Quality on A1 vs Topaz |
|--------|----------------|------------------------|
| Denoise: Normal/Strong/Extreme | NAFNet-SIDD-width64 + YCbCr detail-aware blend | ~40 % (training data mismatch) |
| Denoise: Impulse | OpenCV median filter | OK for salt-and-pepper |
| Sharpen: Standard/Strong | Multi-scale unsharp mask | ~70 % |
| Sharpen: Lens Blur/Motion Blur/Refocus | NAFNet-GoPro deconv + classical fallback | ~70 % |

**The denoise quality is the blocker.** NAFNet-SIDD was trained on smartphone
sensor noise (SIDD dataset). On low-ISO Sony A1 photos (which have very
little noise), it tries to clean noise that isn't there and over-smooths
real detail. Even at 10 % strength the result is visibly worse than the
input on a clean photo.

## Where things live

```
CrispAI/
├── CrispAI.bat              # Launcher: start server + open app window
├── STATUS.md                # this file
│
├── backend/
│   ├── server.py            # Flask API, sessions, lifecycle
│   ├── static/ui.html       # The entire web UI in one file (HTML+CSS+JS)
│   ├── crispai.log          # gitignored
│   ├── weights/             # gitignored, auto-downloaded
│   │   ├── NAFNet-SIDD-width64.pth        # 464 MB, denoise
│   │   ├── NAFNet-GoPro-width64.pth       # 271 MB, deblur
│   │   └── ffdnet_color.pth               # 3 MB, fallback denoise
│   └── models/
│       ├── nafnet_arch.py       # Self-contained NAFNet (megvii)
│       ├── nafnet_runner.py     # Lazy load, gdown download, tiled fp32 inference
│       ├── denoise.py           # YCbCr detail-aware merge, 3 levels (AI) + impulse
│       ├── sharpen.py           # 2 classical + 3 AI-deblur sub-models
│       ├── deblur.py            # PSF generators + classical Richardson-Lucy
│       ├── ai_denoise.py        # FFDNet fallback wrapper
│       ├── ffdnet_arch.py       # FFDNet architecture
│       └── weights/             # symlink target for cached models
│
└── plugin/                  # UXP Photoshop plugin (abandoned, see note below)
    └── ...                  # Replaced by standalone web app architecture
```

## How to resume

### Option A: try a different pretrained denoiser (half a day)

Slot in a different model behind the same `DenoiseModel.process()` interface:

- **SCUNet** — `cszn/SCUNet`, has variants for different noise levels
- **Restormer** — multi-task restoration, often gentler
- **DRUNet** — KAIR, classic CNN, easy to integrate
- **KBNet** (2023) — newer architecture

For each, just write a `<name>_arch.py` + `<name>_runner.py` mirroring `nafnet_*`,
then add a routing branch in `models/denoise.py`.

### Option B: train a custom Sony-A1 model (2-3 weeks)

1. Collect paired training data (200-500 pairs):
   - Same scene shot at ISO 100 AND ISO 12800 (tripod, same WB, identical framing)
   - Or use existing low-ISO photos and add synthetic A1-shaped noise
2. Fine-tune NAFNet-SIDD with low LR (1e-5) on this dataset for 10-20 epochs
3. Write a small training script — RTX 4070 can finish in 1-3 days
4. Save the resulting `.pth` next to the others, swap which one `nafnet_runner`
   loads for the `denoise` task

The architecture and inference code are already correct — only the
**weights** need replacement.

### Option C: keep watching for new open-source models

The AI image-restoration research field moves fast. Realistic candidates
that may surface in 2026-2027:

- Diffusion-based restoration (DiffBIR, ResShift) — slower but higher quality
- A NAFNet trained on professional-camera RAWs by some open project
- A multi-task universal model from a major lab

When a promising one appears, swap weights + arch file. The rest of the
project is camera/photo agnostic and ready.

## Decisions and rationale (for future reference)

- **UXP Photoshop plugin abandoned.** UXP's browser engine has too many
  missing Web APIs (OffscreenCanvas, ImageData, etc.). Switched to standalone
  web app architecture (Topaz-style). PS integration can be re-added later
  as a thin plugin that sends pixels to the same backend.

- **fp32 not fp16 for NAFNet inference.** fp16 produces visible rainbow
  checkerboard artefacts on real photos. NAFNet is deep enough that
  fp16 accumulates error through residual connections. The 2x speed loss
  is acceptable.

- **Tile blending: ramp only edges with neighbours.** Earlier bug: cosine
  window ramped to 0 on outer image edges too, then dividing by ~0 weight
  amplified residual noise into the checkerboard pattern. Fix: per-tile
  window with ramps only where a neighbour tile actually overlaps.

- **YCbCr detail-aware merge.** Naive `lerp(original, AI_output, alpha)`
  drags down luma detail proportionally to alpha. Topaz-style fix:
  process in YCbCr; only blend AI luma in flat regions (Sobel edge mask);
  always blend AI chroma (chroma noise is usually pure noise). Helps but
  not enough to overcome the underlying model's bias toward over-smoothing.

- **PREVIEW_MAX = 1600** for slider previews; **original always at full res**.
  The processed-side compromise is visible when zoomed past fit-to-window;
  user can hit "Full-Quality Preview" for accurate evaluation.

## Reflective summary

The infrastructure took ~2 days and is genuinely good. The AI quality
ceiling is set by **training data**, not by code. Open-source pretrained
models for image denoising are trained on smartphone / web noise datasets
which don't match what high-end mirrorless cameras produce. Closing this
gap requires either:

1. Waiting for a better pretrained model (out of our control), or
2. Training one ourselves (significant investment).

For now, Topaz's monthly subscription is the path of least pain. This
project is preserved as both a working tool (better than no tool when
offline) and a launchpad for whichever of the above options unlocks
quality parity later.
