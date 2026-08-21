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
parser.add_argument("--zero-command", action="store_true", help="Set every velocity command to zero.")
parser.add_argument("--flat-terrain", action="store_true", help="Replace the configured terrain with a flat plane.")
parser.add_argument("--joint-limit-iterations", type=int)
parser.add_argument("--coupling-iterations", type=int)
parser.add_argument("--disable-post-stabilization", action="store_true")
parser.add_argument(
    "--contact-solver-type",
    choices=("sparse_jacobi", "sparse_apgd", "sparse_aspg", "sparse_pspg", "sparse_block_gs"),
)
parser.add_argument("--contact-max-iterations", type=int)
parser.add_argument("--contact-tolerance", type=float)
parser.add_argument("--contact-omega", type=float)
parser.add_argument("--contact-compliance", type=float)
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
    if args.flat_terrain:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    if args.zero_command:
        command_cfg = cfg.commands.base_velocity
        for ranges in (command_cfg.ranges, command_cfg.limit_ranges):
            ranges.lin_vel_x = (0.0, 0.0)
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
        command_cfg.rel_standing_envs = 1.0
        cfg.curriculum.lin_vel_cmd_levels = None
        cfg.curriculum.ang_vel_cmd_levels = None
    if args.joint_limit_iterations is not None:
        cfg.sim.physics.solver_cfg.joint_limit_max_iterations = args.joint_limit_iterations
    if args.coupling_iterations is not None:
        cfg.sim.physics.solver_cfg.coupling_iterations = args.coupling_iterations
    if args.disable_post_stabilization:
        cfg.sim.physics.solver_cfg.post_stabilize_joints = False
    if args.contact_solver_type is not None:
        cfg.sim.physics.solver_cfg.contact_solver_type = args.contact_solver_type
    if args.contact_max_iterations is not None:
        cfg.sim.physics.solver_cfg.contact_max_iterations = args.contact_max_iterations
    if args.contact_tolerance is not None:
        cfg.sim.physics.solver_cfg.contact_tolerance = args.contact_tolerance
    if args.contact_omega is not None:
        cfg.sim.physics.solver_cfg.contact_omega = args.contact_omega
    if args.contact_compliance is not None:
        cfg.sim.physics.solver_cfg.contact_compliance = args.contact_compliance
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
    # Fixed bipedal-rough playback framing, selected for gait inspection.
    # For explicit multi-environment overview renders, frame env_0 at the origin.
    if args.focus_left:
        eye, lookat, camera_fov = (-6.0, -6.0, 3.0), (0.0, 0.0, 0.7), 35.0
    else:
        eye, lookat, camera_fov = (-22.0, -3.0, 3.0), (-25.0, 2.0, 2.0), 25.0
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
    viewer.camera.fov = camera_fov

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
