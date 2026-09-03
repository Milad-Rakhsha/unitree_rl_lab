#!/usr/bin/env python3
"""Plot Go2 velocity-tracking errors, aligning each training run at its own start."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

COLORS = {"Completed DVI (coupling-2)": "#FF1700", "Current DVI": "#0000FF", "MJWarp": "#168A2E"}
ORDER = ("Current DVI", "Completed DVI (coupling-2)", "MJWarp")
ALPHA = 2.0 / 36.0
WINDOW = 35
MAX_RAW_STEP: int | None = None
TAGS = (("Metrics/base_velocity/error_vel_xy", r"Planar velocity error $\Vert v_{xy}^{cmd}-v_{xy}\Vert$ [m/s]"),
        ("Metrics/base_velocity/error_vel_yaw", r"Yaw-rate error $|\omega_z^{cmd}-\omega_z|$ [rad/s]"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dvi-run", type=Path, required=True)
    p.add_argument("--current-dvi-run", type=Path, required=True)
    p.add_argument("--mjwarp-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-stem", default="velocity_tracking_errors")
    p.add_argument("--max-raw-step", type=int, help="Exclude scalar points after this TensorBoard iteration.")
    return p.parse_args()


def load(run: Path) -> EventAccumulator:
    events = sorted(run.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError(f"Expected one event file in {run}, found {len(events)}")
    acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def series(acc: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    points = acc.Scalars(tag)
    if MAX_RAW_STEP is not None:
        points = [p for p in points if p.step <= MAX_RAW_STEP]
    if not points:
        raise RuntimeError(f"No {tag} points at or before max raw step {MAX_RAW_STEP}")
    return np.asarray([p.step for p in points]), np.asarray([p.value for p in points], dtype=float)


def ema(y: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = ALPHA * y[i] + (1.0 - ALPHA) * out[i - 1]
    return out


def rolling_std(y: np.ndarray) -> np.ndarray:
    half = WINDOW // 2
    return np.asarray([np.std(y[max(0, i-half):min(len(y), i+half+1)]) for i in range(len(y))])


def main():
    global MAX_RAW_STEP
    args = parse_args()
    MAX_RAW_STEP = args.max_raw_step
    runs = {"Completed DVI (coupling-2)": args.baseline_dvi_run, "Current DVI": args.current_dvi_run, "MJWarp": args.mjwarp_run}
    accs = {name: load(run) for name, run in runs.items()}
    fps = {}
    for name, acc in accs.items():
        x, y = series(acc, "Perf/total_fps")
        fps[name] = float(np.median(y[x >= max(100, int(0.1 * x[-1]))]))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.1), sharex=True, constrained_layout=True)
    results = []
    for ax, (tag, ylabel) in zip(axes, TAGS):
        for name in ORDER:
            raw_x, y = series(accs[name], tag)
            x = raw_x - raw_x[0]
            smooth, spread = ema(y), rolling_std(y)
            ax.fill_between(x, smooth - spread, smooth + spread, color=COLORS[name], alpha=0.13, linewidth=0)
            ax.plot(x, smooth, color=COLORS[name], lw=1.65, label=f"{name} — {fps[name]:,.0f} steps/s")
            results.append((name, tag, int(raw_x[-1]), int(x[-1]), float(y[-1]), float(smooth[-1])))
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#999999", alpha=0.30, linewidth=0.6)
    axes[0].set_title("Go2")
    axes[0].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="none")
    axes[-1].set_xlabel("PPO iterations since run start")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    for ext, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)
    summary = stem.with_name(f"{stem.name}_summary.txt")
    summary.write_text("\n".join("\t".join(map(str, row)) for row in results) + "\n")
    print(summary)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
