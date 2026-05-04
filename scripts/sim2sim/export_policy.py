#!/usr/bin/env python3
"""Export a checkpoint to JIT policy.pt for MuJoCo sim2sim testing.

Usage:
    conda run -n go2 python export_policy.py \
        --checkpoint logs/rsl_rl/unitree_go2_bipedal_walk/<run>/model_<iter>.pt \
        --output exported_policy.pt
"""
import argparse
import torch
from torch import nn


def export_policy(checkpoint_path, output_path, obs_dim=135, act_dim=12, hidden_sizes=[512, 256, 128]):
    """Export actor from rsl_rl checkpoint to JIT torchscript."""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "actor_state_dict" in ckpt:
        sd = ckpt["actor_state_dict"]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")
    
    # Build actor network
    layers = []
    in_dim = obs_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ELU())
        in_dim = h
    layers.append(nn.Linear(in_dim, act_dim))
    actor = nn.Sequential(*layers)
    
    # Load weights (mlp.0.weight -> 0.weight, mlp.2.weight -> 2.weight, etc.)
    new_sd = {}
    for key, value in sd.items():
        if key.startswith("mlp."):
            new_key = key[len("mlp."):]
            new_sd[new_key] = value
    
    result = actor.load_state_dict(new_sd, strict=False)
    if result.unexpected_keys:
        print(f"Warning: unexpected keys ignored: {result.unexpected_keys}")
    if result.missing_keys:
        print(f"Warning: missing keys: {result.missing_keys}")
    
    actor.eval()
    
    # Export to JIT
    example_input = torch.randn(1, obs_dim)
    with torch.no_grad():
        traced = torch.jit.trace(actor, example_input)
    
    traced.save(output_path)
    print(f"Exported JIT policy to: {output_path}")
    
    # Verify
    with torch.no_grad():
        out_orig = actor(example_input)
        out_jit = traced(example_input)
    diff = (out_orig - out_jit).abs().max().item()
    print(f"Verification: max diff = {diff:.2e}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default=None, help="Output path (default: same dir as checkpoint)")
    parser.add_argument("--obs-dim", type=int, default=135)
    parser.add_argument("--act-dim", type=int, default=12)
    args = parser.parse_args()
    
    if args.output is None:
        import os
        ckpt_dir = os.path.dirname(args.checkpoint)
        os.makedirs(os.path.join(ckpt_dir, "exported"), exist_ok=True)
        args.output = os.path.join(ckpt_dir, "exported", "policy.pt")
    
    export_policy(args.checkpoint, args.output, args.obs_dim, args.act_dim)
