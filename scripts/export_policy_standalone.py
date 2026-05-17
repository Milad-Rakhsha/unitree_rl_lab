#!/usr/bin/env python3
"""Standalone policy export script that converts a TorchScript .pt policy to ONNX.

The Isaac Lab `play.py` writes both `policy.pt` (TorchScript) and `policy.onnx`,
but on installs where the bundled torch.onnx exporter rejects ScriptModules,
the `.onnx` file ends up missing or incomplete. This script regenerates it
from the `.pt` using the legacy `torch.onnx.utils._export` path, which still
supports ScriptModules.
"""

from __future__ import annotations

import argparse
import os
import sys


def _infer_input_dim(model) -> int | None:
    """Return the input feature dimension by inspecting the first weight tensor.

    Works for any MLP-style policy whose first parameter is a linear weight of
    shape ``[hidden, in_features]``. Falls back to ``None`` if no candidate is
    found (e.g. recurrent policies, conv inputs).
    """
    for name, param in model.named_parameters():
        if param.dim() == 2:
            # First 2D weight -> in_features is dim 1
            return int(param.shape[1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TorchScript policy.pt to ONNX")
    parser.add_argument("--jit_path", type=str, required=True, help="Path to policy.pt (TorchScript) file")
    parser.add_argument(
        "--obs_dim",
        type=int,
        default=None,
        help="Observation dimension. If omitted, derived from the first 2D weight in the model.",
    )
    parser.add_argument(
        "--opset", type=int, default=11, help="ONNX opset version (default: 11, matches deploy)."
    )
    args = parser.parse_args()

    jit_path = os.path.abspath(args.jit_path)
    if not os.path.exists(jit_path):
        print(f"[ERROR] JIT file not found: {jit_path}")
        sys.exit(1)

    import torch
    from torch.onnx import utils as onnx_utils

    print(f"[INFO] Loading TorchScript model from: {jit_path}")
    model = torch.jit.load(jit_path, map_location="cpu")
    model.eval()

    # Determine observation dimension: explicit flag wins, otherwise inspect.
    if args.obs_dim is not None:
        obs_dim = args.obs_dim
        print(f"[INFO] Using user-supplied obs dim: {obs_dim}")
    else:
        obs_dim = _infer_input_dim(model)
        if obs_dim is None:
            print(
                "[ERROR] Could not infer obs dim from model parameters. "
                "Pass --obs_dim explicitly."
            )
            sys.exit(1)
        print(f"[INFO] Inferred obs dim from first 2D weight: {obs_dim}")

    dummy_input = torch.zeros(1, obs_dim)
    try:
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"[ERROR] Forward pass with obs_dim={obs_dim} failed: {e}")
        print("        Try passing --obs_dim with the correct value.")
        sys.exit(1)

    onnx_path = os.path.join(os.path.dirname(jit_path), "policy.onnx")
    print(f"[INFO] Exporting to: {onnx_path}")

    # _export bypasses the new dynamo-based exporter, which rejects ScriptModules.
    with torch.no_grad():
        onnx_utils._export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["obs"],
            output_names=["actions"],
            verbose=False,
        )

    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX file was not created at {onnx_path}")
        sys.exit(1)

    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"[INFO] OK: {onnx_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
