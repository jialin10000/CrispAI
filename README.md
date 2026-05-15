# CrispAI

AI-powered noise reduction and sharpening plugin for Adobe Photoshop.

Combines state-of-the-art deep learning models (NAFNet, Restormer) into a simple one-click Photoshop panel — noise reduction and sharpening/deblur without switching apps.

## Features

- **Noise Reduction** — NAFNet deep learning model, rivals Topaz DeNoise AI
- **Sharpening & Deblur** — Restormer model handles motion blur and focus blur
- **Photoshop Panel** — process current layer, result returned as new layer
- **Local processing** — no cloud, no subscription, runs on your GPU

## Requirements

- Adobe Photoshop 2022+ (UXP plugin support)
- Python 3.10+
- NVIDIA GPU (CUDA) or Apple Silicon (MPS)

## Project Structure

```
CrispAI/
├── backend/          # Python AI processing server
│   ├── server.py     # Local HTTP server
│   ├── models/       # AI model wrappers
│   └── utils/        # Image I/O helpers
├── plugin/           # Photoshop UXP plugin
│   ├── manifest.json
│   └── src/          # Panel UI (HTML/JS)
└── docs/             # Setup and usage guides
```

## Status

Early development — proof of concept phase.
