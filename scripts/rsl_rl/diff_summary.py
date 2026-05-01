"""Print a compact side-by-side summary of deploy vs IIL for a range of
intro_elapsed values. Used after diff_deploy_vs_iil.py to spot the first
divergence in base orientation / base ang-vel / action magnitudes.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load(path: Path):
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) if i != 1 else v for i, v in enumerate(r)] for r in reader]
    return header, rows


def col(h, name):
    return h.index(name)


def nearest(rows, intro_col, t):
    best = None
    best_d = 1e9
    for r in rows:
        d = abs(r[intro_col] - t)
        if d < best_d:
            best_d = d
            best = r
    return best


def l2(vals):
    return math.sqrt(sum(v * v for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deploy_csv", type=Path)
    ap.add_argument("iil_csv", type=Path)
    ap.add_argument("--start", type=float, default=0.9)
    ap.add_argument("--stop", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    hd, rd = load(args.deploy_csv)
    hi, ri = load(args.iil_csv)

    ic = col(hd, "intro_elapsed")
    pitch_c = col(hd, "base_pitch")
    roll_c = col(hd, "base_roll")
    wx_c = col(hd, "base_wx_b")
    wy_c = col(hd, "base_wy_b")
    wz_c = col(hd, "base_wz_b")
    a_cols = [col(hd, f"action_{i}") for i in range(12)]
    qv_cols = [col(hd, f"qvel_{i}") for i in range(12)]

    header_fmt = (
        f"{'t':>5}  {'d_pitch':>8} {'i_pitch':>8}  "
        f"{'d_roll':>7} {'i_roll':>7}  "
        f"{'d_|w|':>7} {'i_|w|':>7}  "
        f"{'d_|a|':>7} {'i_|a|':>7}  "
        f"{'d_|qv|':>8} {'i_|qv|':>8}"
    )
    print(header_fmt)
    print("-" * len(header_fmt))

    t = args.start
    while t <= args.stop + 1e-9:
        d = nearest(rd, ic, t)
        i = nearest(ri, ic, t)
        if d is None or i is None:
            t += args.step
            continue
        d_w = l2([d[wx_c], d[wy_c], d[wz_c]])
        i_w = l2([i[wx_c], i[wy_c], i[wz_c]])
        d_a = l2([d[c] for c in a_cols])
        i_a = l2([i[c] for c in a_cols])
        d_qv = l2([d[c] for c in qv_cols])
        i_qv = l2([i[c] for c in qv_cols])
        print(
            f"{t:>5.2f}  "
            f"{d[pitch_c]:>+8.3f} {i[pitch_c]:>+8.3f}  "
            f"{d[roll_c]:>+7.3f} {i[roll_c]:>+7.3f}  "
            f"{d_w:>7.3f} {i_w:>7.3f}  "
            f"{d_a:>7.2f} {i_a:>7.2f}  "
            f"{d_qv:>8.2f} {i_qv:>8.2f}"
        )
        t += args.step


if __name__ == "__main__":
    main()
