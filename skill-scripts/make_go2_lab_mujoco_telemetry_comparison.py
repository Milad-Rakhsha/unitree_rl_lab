#!/usr/bin/env python3
"""Compose synchronized DVI Isaac Lab vs MuJoCo transfer video with rear-leg telemetry."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def get_frame(reader, index):
    return reader.get_data(min(index, reader.count_frames() - 1))


GO2_REAR_LIMITS = {
    "RL_hip_joint": (-1.0472, 1.0472), "RR_hip_joint": (-1.0472, 1.0472),
    "RL_thigh_joint": (-0.5236, 4.5379), "RR_thigh_joint": (-0.5236, 4.5379),
    "RL_calf_joint": (-2.7227, -0.83776), "RR_calf_joint": (-2.7227, -0.83776),
}


def normalized_joint_positions(data):
    names = [str(name) for name in data["joint_names"]]
    limits = np.asarray([GO2_REAR_LIMITS[name] for name in names], dtype=float)
    return 2.0 * (data["joint_pos"] - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1.0


def plot_panel(lab, mj, t, duration, width, height, left_label, right_label, plot_title, normalize_joint_limits=False, phase_boundaries=()):
    fig, (axj, axc) = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=100,
                                   gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    colors = ("#f28e2b", "#e15759", "#b07aa1", "#59a14f", "#4e79a7", "#76b7b2")
    names = [str(name).replace("_joint", "").replace("_", " ") for name in lab["joint_names"]]
    lab_pos = normalized_joint_positions(lab) if normalize_joint_limits else lab["joint_pos"]
    mj_pos = normalized_joint_positions(mj) if normalize_joint_limits else mj["joint_pos"]
    for i, (color, name) in enumerate(zip(colors, names)):
        axj.plot(lab["time"], lab_pos[:, i], color=color, lw=1.4, label=f"DVI {name}")
        axj.plot(mj["time"], mj_pos[:, i], color=color, lw=1.0, ls="--", alpha=.9, label=f"MuJoCo {name}")
    axj.axvline(t, color="white", lw=2, zorder=10)
    ylabel = "normalized joint coordinate" if normalize_joint_limits else "joint position (rad)"
    axj.set(xlim=(0, duration), ylabel=ylabel, title=plot_title)
    if normalize_joint_limits:
        axj.axhline(-1.0, color="black", lw=1.2, ls=":")
        axj.axhline(1.0, color="black", lw=1.2, ls=":")
        axj.set_ylim(-1.15, 1.15)
    axj.grid(alpha=.25); axj.legend(loc="upper right", ncol=3, fontsize=6, framealpha=.8)
    for row, label, data, color in [(1.25, left_label, lab, "#e15759"), (.35, right_label, mj, "#4e79a7")]:
        times=data["time"]; contacts=data["contact"]
        for foot in range(2):
            mask=contacts[:, foot].astype(bool); start=None
            for k, on in enumerate(np.r_[mask, False]):
                if on and start is None: start=k
                elif not on and start is not None:
                    axc.broken_barh([(times[start], times[k-1]-times[start]+.02)], (row + foot*.34, .25), facecolors=color)
                    start=None
    axc.axvline(t, color="black", lw=2); axc.set(xlim=(0,duration), ylim=(0,2.0), yticks=[.52,1.42], yticklabels=[right_label, left_label], xlabel="time (s)", title="Rear-foot ground-contact events (RL, RR bars)")
    for x in phase_boundaries: axc.axvline(x, color="0.6", lw=.9)
    axc.grid(axis="x", alpha=.25)
    canvas=FigureCanvasAgg(fig); canvas.draw()
    rgba=np.asarray(canvas.buffer_rgba()); plt.close(fig)
    return rgba[:, :, :3]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--lab-video",required=True);p.add_argument("--mujoco-video",required=True)
    p.add_argument("--lab-telemetry",required=True);p.add_argument("--mujoco-telemetry",required=True);p.add_argument("--output",required=True)
    p.add_argument("--provenance", default=None, help="Checkpoint/run identifier rendered into the output header.")
    p.add_argument("--left-label", default="DVI native Isaac Lab")
    p.add_argument("--right-label", default="DVI transferred to MuJoCo")
    p.add_argument("--plot-title", default="DVI rear-leg motion  —  solid: Isaac Lab, dashed: MuJoCo transfer")
    p.add_argument("--duration", type=float, default=10.0, help="Output duration in seconds (default: 10).")
    p.add_argument("--normalize-joint-limits", action="store_true", help="Plot rear-joint coordinates normalized so hard limits are -1 and +1.")
    p.add_argument("--phase-boundaries", type=float, nargs="*", default=(), help="Command transition times drawn on the contact panel.")
    a=p.parse_args()
    lab=np.load(a.lab_telemetry); mj=np.load(a.mujoco_telemetry)
    for d in (lab,mj):
        if d["joint_pos"].ndim != 2 or d["joint_pos"].shape[1] != 6 or d["contact"].shape[1] != 2: raise ValueError("Unexpected telemetry shape")
    if a.duration <= 0.0:
        raise ValueError("--duration must be positive")
    lr=imageio.get_reader(a.lab_video); mr=imageio.get_reader(a.mujoco_video); fps=25
    n=round(a.duration*fps); top_h=480; top_w=1280; graph_h=390
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",22)
    out=imageio.get_writer(a.output,fps=fps,quality=8,macro_block_size=16)
    try:
      for i in range(n):
        t=i/fps
        left=Image.fromarray(get_frame(lr, round(t*lr.get_meta_data().get("fps",50)))).convert("RGB").resize((640,480))
        right=Image.fromarray(get_frame(mr, round(t*mr.get_meta_data().get("fps",25)))).convert("RGB").resize((640,480))
        top=Image.new("RGB",(top_w,top_h),(18,18,18));top.paste(left,(0,0));top.paste(right,(640,0));d=ImageDraw.Draw(top)
        d.rectangle((0,0,640,7),fill=(220,65,55));d.rectangle((640,0,1280,7),fill=(78,121,167));d.text((14,14),a.left_label,font=font,fill="white");d.text((654,14),a.right_label,font=font,fill="white")
        if a.provenance:
            d.rectangle((0,448,1280,480),fill=(18,18,18));d.text((14,452),a.provenance,font=font,fill="white")
        graph=Image.fromarray(plot_panel(lab,mj,t,a.duration,top_w,graph_h,a.left_label,a.right_label,a.plot_title,a.normalize_joint_limits,a.phase_boundaries));out.append_data(np.vstack((np.asarray(top),np.asarray(graph))))
    finally:
      out.close();lr.close();mr.close()
    print(a.output)
if __name__ == '__main__': main()
