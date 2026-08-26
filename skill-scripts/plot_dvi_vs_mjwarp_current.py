#!/usr/bin/env python3
"""Compare a current DVI run with the reference MJWarp run on aligned TensorBoard axes."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Preferred comparison-color order: blue, then red (skills.md).
COLORS = {"DVI": "#0072BD", "MJWarp": "#D62728"}
ALPHA = 2.0 / 36.0
WINDOW = 35


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dvi-run", type=Path, required=True)
    p.add_argument("--mjwarp-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dvi-max-step", type=int, required=True,
                   help="Immutable DVI snapshot cutoff in raw TensorBoard iterations.")
    p.add_argument("--reward-config", type=Path, required=True,
                   help="Matching saved bipedal_env_cfg.py; used to order terms by literal configured weight.")
    return p.parse_args()


def load(path: Path) -> EventAccumulator:
    event_files = sorted(path.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(f"Expected exactly one event file in {path}, found {len(event_files)}")
    acc = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def series(acc: EventAccumulator, tag: str, cutoff: int | None = None):
    pts = acc.Scalars(tag)
    if cutoff is not None:
        pts = [p for p in pts if p.step <= cutoff]
    if not pts:
        raise RuntimeError(f"No scalar points for {tag}")
    x = np.asarray([p.step for p in pts])
    return x - x[0], np.asarray([p.value for p in pts], dtype=float), x


def ema(y):
    z = np.empty_like(y)
    z[0] = y[0]
    for i in range(1, len(y)):
        z[i] = ALPHA * y[i] + (1.0 - ALPHA) * z[i - 1]
    return z


def spread(y):
    h = WINDOW // 2
    return np.asarray([np.std(y[max(0, i - h):min(len(y), i + h + 1)]) for i in range(len(y))])


def plot_trace(ax, data, title, ylabel):
    for name, (x, y) in data.items():
        sm, sd = ema(y), spread(y)
        ax.fill_between(x, sm - sd, sm + sd, color=COLORS[name], alpha=.13, linewidth=0)
        ax.plot(x, sm, color=COLORS[name], lw=1.6, label=name)
    ax.set(title=title, ylabel=ylabel)
    ax.grid(True, alpha=.28)


def save(fig, out: Path, stem: str):
    for ext, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def reward_weights(config: Path) -> dict[str, float]:
    """Extract literal RewTerm weights from the immutable saved environment config."""
    tree = ast.parse(config.read_text(), filename=str(config))
    weights: dict[str, float] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        if not (isinstance(node.value.func, ast.Name) and node.value.func.id == "RewTerm"):
            continue
        for keyword in node.value.keywords:
            if keyword.arg == "weight":
                try:
                    weights[target.id] = float(ast.literal_eval(keyword.value))
                except (TypeError, ValueError):
                    pass
    if not weights:
        raise RuntimeError(f"No literal RewTerm weights found in {config}")
    return weights


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"DVI": (load(a.dvi_run), a.dvi_max_step), "MJWarp": (load(a.mjwarp_run), None)}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 10})

    rewards = {name: series(acc, "Train/mean_reward", cutoff)[:2] for name, (acc, cutoff) in runs.items()}
    fig, ax = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    plot_trace(ax, rewards, "Go2", "Mean episode reward")
    ax.set_xlabel("PPO iterations since run start")
    ax.set_xlim(0, max(x[-1] for x, _ in rewards.values()))
    ax.legend(loc="lower right", frameon=True)
    save(fig, a.output_dir, "overall_reward_comparison")

    term_sets = [{t.removeprefix("Episode_Reward/") for t in acc.Tags()["scalars"] if t.startswith("Episode_Reward/")}
                 for acc, _ in runs.values()]
    common_set = set.intersection(*term_sets)
    weights = reward_weights(a.reward_config)
    missing_weights = sorted(common_set - weights.keys())
    if missing_weights:
        raise RuntimeError(f"Reward tags without literal weights in {a.reward_config}: {missing_weights}")
    # Highest numerical weights first.  Equal-weight terms are grouped only
    # when there are at most two; larger groups are split to keep legends and
    # traces readable. Colour encodes policy; line style encodes term.
    groups: dict[float, list[str]] = {}
    for term in common_set:
        groups.setdefault(weights[term], []).append(term)
    ordered_groups: list[tuple[float, list[str]]] = []
    for weight, terms in groups.items():
        terms = sorted(terms)
        ordered_groups.extend((weight, terms[i:i + 2]) for i in range(0, len(terms), 2))
    ordered_groups.sort(key=lambda item: (-item[0], item[1]))
    styles = ("-", "--")
    ncols = 3
    nrows = int(np.ceil(len(ordered_groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.4, 3.0 * nrows), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (weight, terms) in zip(axes, ordered_groups):
        max_x = 0
        for term_index, term in enumerate(terms):
            style = styles[term_index % len(styles)]
            for name, (acc, cutoff) in runs.items():
                x, y = series(acc, f"Episode_Reward/{term}", cutoff)[:2]
                sm, sd = ema(y), spread(y)
                ax.fill_between(x, sm - sd, sm + sd, color=COLORS[name], alpha=.07, linewidth=0)
                ax.plot(x, sm, color=COLORS[name], ls=style, lw=1.35)
                max_x = max(max_x, x[-1])
        ax.set(title=f"weight {weight:+g}: " + ", ".join(t.replace("_", " ") for t in terms), ylabel="reward")
        ax.set_xlim(0, max_x)
        ax.grid(True, alpha=.28)
        ax.tick_params(labelsize=7.2)
        # Every subplot owns its compact legend: colour = policy and, when
        # applicable, line style = reward term. No figure-level legend.
        handles = [plt.Line2D([], [], color=COLORS[name], lw=1.5, label=name) for name in runs]
        if len(terms) > 1:
            handles.extend(plt.Line2D([], [], color="#333333", ls=styles[i], lw=1.25,
                                      label=term.replace("_", " "))
                           for i, term in enumerate(terms))
        ax.legend(handles=handles, loc="best", fontsize=6.4, frameon=True,
                  framealpha=.92, edgecolor="none", ncol=2)
    for ax in axes[len(ordered_groups):]:
        ax.set_visible(False)
    fig.suptitle("Go2 — reward terms grouped by configured weight (highest to lowest)", fontsize=13)
    fig.supxlabel("PPO iterations since run start")
    save(fig, a.output_dir, "individual_reward_terms")

    tags = [
        ("Metrics/base_velocity/error_vel_xy", "Planar velocity error [m/s]"),
        ("Metrics/base_velocity/error_vel_yaw", "Yaw-rate error [rad/s]"),
        ("Metrics/success_rate", "Success rate"),
        ("Episode_Termination/base_contact", "Base-contact termination"),
        ("Episode_Termination/hip_contact", "Hip-contact termination"),
        ("Episode_Termination/time_out", "Timeout termination"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 8.2), sharex=True, constrained_layout=True)
    valid = []
    for ax, (tag, label) in zip(axes.flat, tags):
        if all(tag in acc.Tags()["scalars"] for acc, _ in runs.values()):
            data = {name: series(acc, tag, cutoff)[:2] for name, (acc, cutoff) in runs.items()}
            plot_trace(ax, data, tag.removeprefix("Metrics/").removeprefix("Episode_Termination/").replace("_", " "), label)
            ax.set_xlim(0, max(x[-1] for x, _ in data.values()))
            valid.append(tag)
        else:
            ax.set_visible(False)
    fig.suptitle("Go2 — command tracking and failure metrics", fontsize=13)
    fig.legend([plt.Line2D([], [], color=COLORS[n]) for n in runs], list(runs), loc="upper center", ncol=2, frameon=False)
    fig.supxlabel("PPO iterations since run start")
    save(fig, a.output_dir, "diagnostic_metrics")

    lines = [f"dvi_run={a.dvi_run}", f"dvi_raw_cutoff={a.dvi_max_step}", f"mjwarp_run={a.mjwarp_run}",
             f"reward_config={a.reward_config}", f"common_reward_terms={len(common_set)}",
             "reward_weight_groups=" + ";".join(f"{weight:+g}:" + ",".join(terms) for weight, terms in ordered_groups),
             f"diagnostic_tags={','.join(valid)}"]
    for name, (acc, cutoff) in runs.items():
        for tag in ("Train/mean_reward", "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw", "Metrics/success_rate"):
            if tag in acc.Tags()["scalars"]:
                x, y, raw = series(acc, tag, cutoff)
                lines.append(f"{name}.{tag}.raw_step={raw[-1]}")
                lines.append(f"{name}.{tag}.raw={y[-1]:.9g}")
                lines.append(f"{name}.{tag}.ema={ema(y)[-1]:.9g}")
    (a.output_dir / "comparison_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
