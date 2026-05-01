"""Diff a deploy-side BipedalRear CSV against the IIL play_deploy_fsm CSV.

Both CSVs are produced with the same schema from
``scripts/rsl_rl/play_deploy_fsm.py`` (IIL) and ``State_RLBase.cpp`` (deploy).
Rows are aligned on ``intro_elapsed`` (time since BipedalRear.enter()).

Usage:
    python diff_deploy_vs_iil.py <deploy_csv> <iil_csv> [--stride N]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> tuple[list[str], list[list[float | str]]]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: list[list[float | str]] = []
        for r in reader:
            parsed: list[float | str] = []
            for v in r:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
            rows.append(parsed)
    return header, rows


def col(header: list[str], name: str) -> int:
    return header.index(name)


def nearest_row(rows: list[list[float | str]], intro_col: int, t: float) -> list[float | str] | None:
    best = None
    best_d = float("inf")
    for r in rows:
        v = r[intro_col]
        if not isinstance(v, float):
            continue
        d = abs(v - t)
        if d < best_d:
            best_d = d
            best = r
    return best


def fnum(x: float | str) -> str:
    return f"{x:+.4f}" if isinstance(x, float) else str(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy_csv", type=Path)
    parser.add_argument("iil_csv", type=Path)
    parser.add_argument(
        "--at",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0],
        help="intro_elapsed timestamps (s) at which to print a row-by-row diff.",
    )
    parser.add_argument(
        "--groups",
        choices=["all", "summary"],
        default="summary",
        help="'all' prints every column; 'summary' prints a curated subset.",
    )
    args = parser.parse_args()

    hd, rd = load(args.deploy_csv)
    hi, ri = load(args.iil_csv)

    if hd != hi:
        print("HEADER MISMATCH")
        print("  deploy:", hd)
        print("  iil:   ", hi)
        return

    intro = col(hd, "intro_elapsed")

    # Curated groups for summary mode.
    q_cols = [f"qpos_{i}" for i in range(12)]
    qv_cols = [f"qvel_{i}" for i in range(12)]
    orient = ["base_roll", "base_pitch", "base_yaw"]
    omega = ["base_wx_b", "base_wy_b", "base_wz_b"]
    act_cols = [f"action_{i}" for i in range(12)]
    cmd_cols = ["cmd_vx", "cmd_vy", "cmd_yaw"]

    # Friendly short labels for joints (FR/FL/RR/RL hip/thigh/calf).
    joint_labels = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
    ]

    def show_group(title: str, cols: list[str], row_d, row_i, labels: list[str] | None = None) -> None:
        print(f"  {title}")
        print(f"    {'col':<10} {'deploy':>10} {'iil':>10} {'diff':>10}")
        for idx, name in enumerate(cols):
            c = col(hd, name)
            dv = row_d[c] if row_d else None
            iv = row_i[c] if row_i else None
            if isinstance(dv, float) and isinstance(iv, float):
                lbl = labels[idx] if labels else name
                print(f"    {lbl:<10} {dv:>+10.4f} {iv:>+10.4f} {dv - iv:>+10.4f}")
            else:
                print(f"    {name:<10} (non-numeric)")

    for t in args.at:
        rd_t = nearest_row(rd, intro, t)
        ri_t = nearest_row(ri, intro, t)
        if rd_t is None or ri_t is None:
            print(f"[t~{t:.2f}] missing data")
            continue
        t_d = rd_t[intro]
        t_i = ri_t[intro]
        print(f"\n=== intro_elapsed ~ {t:.2f}s  (deploy={t_d:.3f}s, iil={t_i:.3f}s) ===")
        if args.groups == "all":
            for name in hd[1:]:
                c = col(hd, name)
                dv, iv = rd_t[c], ri_t[c]
                if isinstance(dv, float) and isinstance(iv, float):
                    print(f"  {name:<14} d={fnum(dv)}  i={fnum(iv)}  Δ={fnum(dv - iv)}")
                else:
                    print(f"  {name:<14} d={dv!r}  i={iv!r}")
        else:
            show_group("joint positions [rad]", q_cols, rd_t, ri_t, joint_labels)
            show_group("joint velocities [rad/s]", qv_cols, rd_t, ri_t, joint_labels)
            show_group("base orientation [rad]", orient, rd_t, ri_t)
            show_group("base ang vel (body) [rad/s]", omega, rd_t, ri_t)
            show_group("action (raw policy) [unit]", act_cols, rd_t, ri_t, joint_labels)
            show_group("velocity command", cmd_cols, rd_t, ri_t)


if __name__ == "__main__":
    main()
