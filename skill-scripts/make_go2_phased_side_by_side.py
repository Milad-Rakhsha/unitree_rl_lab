#!/usr/bin/env python3
"""Compose phased MuJoCo policy recordings: MJWarp left, DVI right."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjwarp", required=True, help="MJWarp phased MP4")
    parser.add_argument("--dvi", required=True, help="DVI phased MP4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subtitle", default="x → y → yaw → combined", help="Header text for the paired replay.")
    parser.add_argument("--left-label", default="MJWarp", help="Left-panel label.")
    parser.add_argument("--right-label", default="DVI", help="Right-panel label.")
    args = parser.parse_args()

    inputs = [(Path(args.mjwarp), args.left_label, (0, 114, 189)), (Path(args.dvi), args.right_label, (214, 39, 40))]
    for path, *_ in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    readers = [imageio.get_reader(path) for path, *_ in inputs]
    fps = readers[0].get_meta_data().get("fps", 25)
    if any(abs(reader.get_meta_data().get("fps", fps) - fps) > 1e-6 for reader in readers[1:]):
        raise ValueError("Input videos must have the same FPS")
    writer = imageio.get_writer(args.output, fps=fps, quality=8, macro_block_size=16)
    try:
        for frames in zip(*readers):
            panels = []
            for frame, (_, name, color) in zip(frames, inputs):
                image = Image.fromarray(frame).convert("RGB")
                panel = Image.new("RGB", (image.width, image.height + 42), (18, 18, 18))
                panel.paste(image, (0, 42))
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, image.width, 7), fill=color)
                draw.text((12, 12), name, font=font, fill="white")
                draw.text((410, 17), args.subtitle, font=small, fill=(220, 220, 220))
                panels.append(np.asarray(panel))
            divider = np.full((panels[0].shape[0], 8, 3), 18, dtype=np.uint8)
            writer.append_data(np.concatenate([panels[0], divider, panels[1]], axis=1))
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    print(args.output)


if __name__ == "__main__":
    main()
