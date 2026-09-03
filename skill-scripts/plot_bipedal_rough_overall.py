#!/usr/bin/env python3
"""Plot overall reward for two DVI runs and one MJWarp run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

COLORS = {"Old DVI": "#FF1700", "Recent DVI": "#0000FF", "MJWarp": "#168A2E"}
ORDER = ("Recent DVI", "Old DVI", "MJWarp")
ALPHA = 2.0 / 36.0
WINDOW = 35
MAX_RAW_STEP: int | None = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dvi-run", type=Path, required=True)
    parser.add_argument("--recent-dvi-run", type=Path, required=True)
    parser.add_argument("--mjwarp-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="go2_bipedal_rough_overall_reward")
    parser.add_argument("--max-raw-step", type=int, help="Exclude scalar points after this TensorBoard iteration.")
    return parser.parse_args()


def load(path: Path):
    events = sorted(path.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError(f"Expected one event file in {path}, found {len(events)}")
    acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def series(acc, tag):
    pts = acc.Scalars(tag)
    if MAX_RAW_STEP is not None:
        pts = [p for p in pts if p.step <= MAX_RAW_STEP]
    if not pts:
        raise RuntimeError(f"No {tag} points at or before max raw step {MAX_RAW_STEP}")
    return np.asarray([p.step for p in pts]), np.asarray([p.value for p in pts], dtype=float)


def relative_iterations(x):
    """Align each run at zero, including resumed runs whose TensorBoard steps inherit a checkpoint offset."""
    return x - x[0]


def ema(y):
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = ALPHA * y[i] + (1.0 - ALPHA) * out[i - 1]
    return out


def rolling_std(y):
    half = WINDOW // 2
    return np.asarray([np.std(y[max(0, i-half):min(len(y), i+half+1)]) for i in range(len(y))])


def main():
    global MAX_RAW_STEP
    args = parse_args()
    MAX_RAW_STEP = args.max_raw_step
    runs = {"Old DVI": args.old_dvi_run, "Recent DVI": args.recent_dvi_run, "MJWarp": args.mjwarp_run}
    for name, path in runs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{name} run directory does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accs = {name: load(path) for name, path in runs.items()}
    rewards = {
        name: (relative_iterations(x), y)
        for name, acc in accs.items()
        for x, y in [series(acc, "Train/mean_reward")]
    }
    fps = {}
    for name, acc in accs.items():
        x, y = series(acc, "Perf/total_fps")
        fps[name] = float(np.median(y[x >= max(100, int(0.1 * x[-1]))]))

    stem = args.output_dir / args.output_stem
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "legend.fontsize": 8.8})
    fig, ax = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    for name in ORDER:
        x, y = rewards[name]
        smooth, spread = ema(y), rolling_std(y)
        ax.fill_between(x, smooth-spread, smooth+spread, color=COLORS[name], alpha=0.13, linewidth=0)
        ax.plot(x, smooth, color=COLORS[name], linewidth=1.65, label=f"{name} — {fps[name]:,.0f} steps/s")
    ax.set(title="Go2", xlabel="PPO iterations since run start", ylabel="Mean episode reward",
           xlim=(0, max(int(x[-1]) for x, _ in rewards.values())))
    ax.grid(True, color="#999999", alpha=0.30, linewidth=0.6)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="none")
    for ext, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)

    for name in ORDER:
        x, y = rewards[name]
        print(f"{name}: relative_iteration={int(x[-1])} raw={y[-1]:.3f} ema={ema(y)[-1]:.3f} fps={fps[name]:.0f}")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
