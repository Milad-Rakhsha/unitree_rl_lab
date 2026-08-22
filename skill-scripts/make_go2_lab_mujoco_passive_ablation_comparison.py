#!/usr/bin/env python3
"""Compose native DVI and two MuJoCo passive-force implementations with telemetry."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


LABELS = ("DVI native Isaac Lab", "MuJoCo XML passive", "MuJoCo actuator-side passive")
COLORS = ("#e15759", "#4e79a7", "#59a14f")
STYLES = ("-", "--", ":")
JOINT_COLORS = ("#f28e2b", "#e15759", "#b07aa1", "#59a14f", "#4e79a7", "#76b7b2")
JOINT_NAMES = ("RL hip", "RL thigh", "RL calf", "RR hip", "RR thigh", "RR calf")


def frame(reader, index):
    return reader.get_data(min(index, reader.count_frames() - 1))


def plot_panel(telemetry, t, width, height):
    fig, (ax_joint, ax_contact) = plt.subplots(
        2, 1, figsize=(width / 100, height / 100), dpi=100,
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True,
    )
    for joint, (joint_color, name) in enumerate(zip(JOINT_COLORS, JOINT_NAMES)):
        for data, label, style in zip(telemetry, LABELS, STYLES):
            ax_joint.plot(
                data["time"], data["joint_pos"][:, joint], color=joint_color,
                ls=style, lw=1.45 if style == "-" else 1.1, alpha=.92,
                label=f"{label}: {name}",
            )
    ax_joint.axvline(t, color="white", lw=2, zorder=10)
    ax_joint.set(
        xlim=(0, 10), ylabel="joint position (rad)",
        title="Rear-leg motion — color: joint; solid: DVI; dashed: MuJoCo XML; dotted: actuator-side passive",
    )
    ax_joint.grid(alpha=.25)
    ax_joint.legend(loc="upper right", ncol=3, fontsize=5.3, framealpha=.85)

    rows = (1.32, .75, .18)
    for row, label, data, color in zip(rows, LABELS, telemetry, COLORS):
        times, contacts = data["time"], data["contact"]
        for foot in range(2):
            mask, start = contacts[:, foot].astype(bool), None
            for k, on in enumerate(np.r_[mask, False]):
                if on and start is None:
                    start = k
                elif not on and start is not None:
                    duration = times[k - 1] - times[start] + .02
                    ax_contact.broken_barh([(times[start], duration)], (row + foot * .23, .18), facecolors=color)
                    start = None
    ax_contact.axvline(t, color="black", lw=2)
    ax_contact.set(
        xlim=(0, 10), ylim=(0, 2.0), yticks=(.40, .97, 1.54),
        yticklabels=("actuator-side", "XML passive", "DVI"), xlabel="time (s)",
        title="Rear-foot ground-contact events (RL, RR bars)",
    )
    for x in (2.5, 5, 7.5):
        ax_contact.axvline(x, color="0.6", lw=.7)
    ax_contact.grid(axis="x", alpha=.25)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    result = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-video", required=True)
    parser.add_argument("--xml-video", required=True)
    parser.add_argument("--actuator-video", required=True)
    parser.add_argument("--lab-telemetry", required=True)
    parser.add_argument("--xml-telemetry", required=True)
    parser.add_argument("--actuator-telemetry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", default=None)
    args = parser.parse_args()

    telemetry = tuple(np.load(path) for path in (args.lab_telemetry, args.xml_telemetry, args.actuator_telemetry))
    for data in telemetry:
        if data["joint_pos"].ndim != 2 or data["joint_pos"].shape[1] != 6 or data["contact"].shape[1] != 2:
            raise ValueError("Unexpected telemetry shape")

    readers = tuple(imageio.get_reader(path) for path in (args.lab_video, args.xml_video, args.actuator_video))
    width, top_height, graph_height, fps, frames = 1920, 480, 390, 25, 250
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=fps, quality=8, macro_block_size=16)
    try:
        for i in range(frames):
            t = i / fps
            top = Image.new("RGB", (width, top_height), (18, 18, 18))
            draw = ImageDraw.Draw(top)
            for col, (reader, label, color) in enumerate(zip(readers, LABELS, COLORS)):
                source_fps = reader.get_meta_data().get("fps", 25)
                image = Image.fromarray(frame(reader, round(t * source_fps))).convert("RGB").resize((640, 480))
                x = 640 * col
                top.paste(image, (x, 0))
                draw.rectangle((x, 0, x + 640, 7), fill=color)
                draw.text((x + 12, 14), label, font=font, fill="white")
            if args.provenance:
                draw.rectangle((0, 448, width, 480), fill=(18, 18, 18))
                draw.text((14, 452), args.provenance, font=font, fill="white")
            writer.append_data(np.vstack((np.asarray(top), plot_panel(telemetry, t, width, graph_height))))
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    print(output)


if __name__ == "__main__":
    main()
