"""
Quick test: run Real-ESRGAN denoise on a single image.
Usage: python test_denoise.py <input_image> [output_image]
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def denoise(input_path, output_path):
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # RealESRGAN x2 — good for denoise without over-upscaling
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)

    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        model=model,
        tile=512,        # tile size — prevents OOM on large images
        tile_pad=10,
        pre_pad=0,
        half=True if device == "cuda" else False,
    )

    img = np.array(Image.open(input_path).convert("RGB"))
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    print("Processing...")
    output, _ = upsampler.enhance(img, outscale=1)  # outscale=1 keeps original size

    Image.fromarray(output).save(output_path)
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_denoise.py <input> [output]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.splitext(input_path)[0] + "_crispai.jpg"

    denoise(input_path, output_path)
