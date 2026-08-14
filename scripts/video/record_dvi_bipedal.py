#!/usr/bin/env python3
"""Render a trained Go2 bipedal policy with a Newton physics preset."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import pathlib
import pkgutil
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--task", type=str, default="Unitree-Go2-Bipedal-Walk")
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-envs", type=int, default=1, help="Number of simultaneous environments to render.")
parser.add_argument("--focus-left", action="store_true", help="Zoom and focus on the left side of a multi-environment grid.")
parser.add_argument("--preset", choices=("newton_dvi", "newton_mjwarp"), default="newton_dvi")
parser.add_argument("--joint-limit-iterations", type=int)
parser.add_argument("--coupling-iterations", type=int)
parser.add_argument("--disable-post-stabilization", action="store_true")
parser.add_argument("--contact-regularization", type=float)
parser.add_argument("--contact-recovery-speed", type=float)
parser.add_argument("--contact-alpha", type=float)
parser.add_argument("--joint-alpha", type=float)
parser.add_argument("--joint-recovery-speed", type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks


def _import_task_packages() -> None:
    robots = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots")
    for info in pkgutil.walk_packages(robots.__path__, robots.__name__ + "."):
        importlib.import_module(info.name)


_import_task_packages()

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_visualizers.newton import NewtonVisualizerCfg


def main() -> None:
    task = args.task
    cfg = load_cfg_from_registry(task, "play_env_cfg_entry_point")
    cfg = resolve_presets(cfg, {args.preset})
    if args.joint_limit_iterations is not None:
        cfg.sim.physics.solver_cfg.joint_limit_max_iterations = args.joint_limit_iterations
    if args.coupling_iterations is not None:
        cfg.sim.physics.solver_cfg.coupling_iterations = args.coupling_iterations
    if args.disable_post_stabilization:
        cfg.sim.physics.solver_cfg.post_stabilize_joints = False
    if args.contact_regularization is not None:
        cfg.sim.physics.solver_cfg.contact_reg = args.contact_regularization
    if args.contact_recovery_speed is not None:
        cfg.sim.physics.solver_cfg.contact_recovery_speed = args.contact_recovery_speed
    if args.contact_alpha is not None:
        cfg.sim.physics.solver_cfg.contact_alpha = args.contact_alpha
    if args.joint_alpha is not None:
        cfg.sim.physics.solver_cfg.joint_alpha = args.joint_alpha
    if args.joint_recovery_speed is not None:
        cfg.sim.physics.solver_cfg.joint_recovery_speed = args.joint_recovery_speed
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.sim.enable_newton_rendering = True
    # One viewport that encompasses the replicated scene.  The 64-env case is
    # an 8x8 layout with 2.5 m spacing, so use an elevated wide shot; retain
    # the close single-robot framing for ordinary videos.
    if args.num_envs == 1:
        eye, lookat = (3.0, -3.0, 1.7), (0.0, 0.0, 0.55)
    else:
        # Isaac Lab packs replicated environments on a near-square grid.  The
        # default is a wide grid shot; --focus-left deliberately zooms into
        # the leftmost two columns for inspectable gait behavior.
        cols = int(np.ceil(np.sqrt(args.num_envs)))
        rows = int(np.ceil(args.num_envs / cols))
        extent_x = 2.5 * (cols - 1)
        extent_y = 2.5 * (rows - 1)
        if args.focus_left:
            eye = (2.5, -7.0, 5.5)
            lookat = (0.0, 2.5 * min(rows - 1, 3) / 2.0, 0.45)
        else:
            span = max(extent_x, extent_y)
            center_x, center_y = 0.5 * extent_x, 0.5 * extent_y
            eye = (center_x + 0.90 * span, -0.90 * span, 0.78 * span)
            lookat = (center_x, center_y, 0.45)
    cfg.sim.visualizer_cfgs = [
        NewtonVisualizerCfg(
            headless=True,
            window_width=1280,
            window_height=720,
            eye=eye,
            lookat=lookat,
            enable_shadows=True,
            enable_sky=True,
        )
    ]

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(task, cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    viewer = next(
        (v._viewer for v in env.unwrapped.sim._visualizers if hasattr(v, "_viewer") and hasattr(v._viewer, "get_frame")),
        None,
    )
    if viewer is None:
        raise RuntimeError("Newton framebuffer viewer was not initialized")

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    frames: list[np.ndarray] = []
    frame_stride = max(1, round(1.0 / (args.fps * env.unwrapped.step_dt)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", pixelformat="yuv420p", quality=8)
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            if step % frame_stride == 0:
                frame = viewer.get_frame()
                frame = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
                frame = np.asarray(frame)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                writer.append_data(frame.astype(np.uint8, copy=False))
                frames.append(frame)
    finally:
        writer.close()
        env.close()
    print(
        f"NEWTON_VIDEO_OK preset={args.preset} output={args.output} frames={len(frames)} fps={args.fps}",
        flush=True,
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        app.close()
    raise SystemExit(exit_code)
