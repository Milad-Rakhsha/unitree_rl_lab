#!/usr/bin/env python3
"""Plot common episode-reward terms for two DVI runs and one MJWarp run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

COLORS = {"Old DVI": "#FF1700", "Recent DVI": "#0000FF", "MJWarp": "#168A2E"}
PLOT_ORDER = ("Recent DVI", "Old DVI", "MJWarp")
ALPHA = 2.0 / 36.0
WINDOW = 35
MAX_RAW_STEP: int | None = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dvi-run", type=Path, required=True)
    parser.add_argument("--recent-dvi-run", type=Path, required=True)
    parser.add_argument("--mjwarp-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="go2_bipedal_rough_reward_terms")
    parser.add_argument("--max-raw-step", type=int, help="Exclude scalar points after this TensorBoard iteration.")
    return parser.parse_args()


def load(run: Path):
    events = sorted(run.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError(f"Expected one event file in {run}, found {len(events)}")
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
    fps = {}
    for name, acc in accs.items():
        x, y = series(acc, "Perf/total_fps")
        fps[name] = float(np.median(y[x >= max(100, int(0.1 * x[-1]))]))

    term_sets = [{tag.removeprefix("Episode_Reward/") for tag in acc.Tags()["scalars"]
                  if tag.startswith("Episode_Reward/")} for acc in accs.values()]
    common = sorted(set.intersection(*term_sets))
    if not common:
        raise RuntimeError("No common Episode_Reward/* tags across supplied runs")
    omitted = sorted(set.union(*term_sets) - set(common))

    fig, axes = plt.subplots(6, 4, figsize=(14.4, 15.4), sharex=True, constrained_layout=True)
    for index, (ax, term) in enumerate(zip(axes.flat, common)):
        traces = {}
        for name in PLOT_ORDER:
            raw_x, y = series(accs[name], f"Episode_Reward/{term}")
            x = relative_iterations(raw_x)
            smooth, spread = ema(y), rolling_std(y)
            traces[name] = smooth
            ax.fill_between(x, smooth-spread, smooth+spread, color=COLORS[name], alpha=0.09, linewidth=0)
            ax.plot(x, smooth, color=COLORS[name], lw=1.05,
                    label=f"{name} — {fps[name]:,.0f} steps/s" if index == 0 else None)
        dvi_lo = min(traces["Old DVI"].min(), traces["Recent DVI"].min())
        dvi_hi = max(traces["Old DVI"].max(), traces["Recent DVI"].max())
        pad = 0.05 * (dvi_hi - dvi_lo) if dvi_hi > dvi_lo else max(abs(dvi_lo) * 0.05, 1.0e-6)
        ax.set(title=term.replace("_", " "), xlim=(0, max(relative_iterations(series(accs[name], f"Episode_Reward/{term}")[0])[-1] for name in PLOT_ORDER)),
               ylim=(dvi_lo - pad, dvi_hi + pad))
        ax.grid(True, color="#999999", alpha=0.28, lw=0.5)
        ax.tick_params(labelsize=7.8)
    for ax in axes.flat[len(common):]:
        ax.set_visible(False)
    fig.suptitle("Go2", fontsize=13)
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.968))
    fig.supxlabel("PPO iterations since run start", fontsize=10)
    fig.supylabel("Episode reward term", fontsize=10)
    stem = args.output_dir / args.output_stem
    for ext, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)

    summary = args.output_dir / f"{args.output_stem}_summary.txt"
    lines = [f"common_reward_terms={len(common)}", f"omitted_noncommon_terms={','.join(omitted) or 'none'}"]
    for name, path in runs.items():
        lines.extend((f"{name}.run={path}", f"{name}.median_fps={fps[name]:.9g}"))
    summary.write_text("\n".join(lines) + "\n")
    print(summary)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
