#!/usr/bin/env python3
"""Render native Newton Go2 bipedal policy playback through x/y/yaw/combined commands."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import pathlib
import pkgutil
import sys

# This canonical copy lives directly under <repo>/skill-scripts/.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--task", type=str, default="Unitree-Go2-Bipedal-Walk")
parser.add_argument("--saved-env-cfg", type=pathlib.Path, help="Saved bipedal_env_cfg.py from the run; restores its native solver configuration.")
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--telemetry", type=pathlib.Path, help="Optional NPZ output: rear-leg joint positions and rear-foot contact events.")
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
parser.add_argument("--zero-command", action="store_true", help="Force zero velocity commands.")
parser.add_argument("--flat-terrain", action="store_true", help="Evaluate the rough-trained policy on Isaac Lab's infinite flat plane.")
parser.add_argument("--follow-robot", action="store_true", help="Track environment 0's robot with a fixed-offset camera.")
parser.add_argument("--front-view", action="store_true", help="Use the opposite fixed-world tracking offset, matching the MuJoCo recorder's front-facing view.")
parser.add_argument("--phased-commands", action="store_true", help="Play forward-x, lateral-y, yaw, then combined commands.")
parser.add_argument("--phase-duration", type=float, default=2.5, help="Seconds per command phase.")
parser.add_argument("--cmd-vx", type=float, default=0.7)
parser.add_argument("--cmd-vy", type=float, default=0.3)
parser.add_argument("--cmd-wz", type=float, default=0.7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    if args.saved_env_cfg:
        import importlib.util
        cfg_path = args.saved_env_cfg.resolve()
        spec = importlib.util.spec_from_file_location(
            "unitree_rl_lab.tasks.locomotion.robots.go2.saved_bipedal_env_cfg", cfg_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load saved environment config: {cfg_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cfg = module.RobotBipedalWalkRoughPlayEnvCfg()
        cfg = resolve_presets(cfg, {args.preset})
    else:
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
    if args.flat_terrain:
        # Retain the run's native solver configuration while replacing only the
        # evaluation geometry with the same infinite plane used by the base task.
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.max_init_terrain_level = None
    # One viewport that encompasses the replicated scene.  The 64-env case is
    # an 8x8 layout with 2.5 m spacing, so use an elevated wide shot; retain
    # the close single-robot framing for ordinary videos.
    if args.num_envs == 1:
        # Match the MuJoCo transfer recorder's fixed world-frame tracking view:
        # front-left looking toward the base.  This offset translates with the
        # base but never rotates/orbits with its yaw.
        eye, lookat = ((-3.0, -3.0, 1.7), (0.0, 0.0, 0.55)) if args.front_view else ((3.0, 3.0, 1.7), (0.0, 0.0, 0.55))
    else:
        # Isaac Lab packs replicated environments on a near-square grid.  The
        # default is a wide grid shot; --focus-left deliberately zooms into
        # the leftmost two columns for inspectable gait behavior.
        cols = int(np.ceil(np.sqrt(args.num_envs)))
        rows = int(np.ceil(args.num_envs / cols))
        extent_x = 2.5 * (cols - 1)
        extent_y = 2.5 * (rows - 1)
        if args.focus_left:
            eye = (-20.0, -3.0, 4.0)
            lookat = (-25.0, 2.0, 1.0)
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
            # Make the native DVI and MJWarp stacks visually comparable despite
            # their different default lighting/sky implementations.
            enable_shadows=False,
            enable_sky=False,
        )
    ]

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(task, cfg=cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    visualizer = next((v for v in env.unwrapped.sim._visualizers if hasattr(v, "_viewer")), None)
    if visualizer is None:
        raise RuntimeError(f"Newton visualizer was not initialized: {env.unwrapped.sim._visualizers!r}")
    viewer = visualizer._viewer

    # Camera is updated per frame below when --follow-robot is enabled.
    viewer.camera.fov = 35.0

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    if args.zero_command and args.phased_commands:
        raise ValueError("--zero-command and --phased-commands are mutually exclusive")
    cmd_mgr = env.unwrapped.command_manager
    phases = [
        ("forward x", (args.cmd_vx, 0.0, 0.0)),
        ("lateral y", (0.0, args.cmd_vy, 0.0)),
        ("yaw", (0.0, 0.0, args.cmd_wz)),
        ("combined", (args.cmd_vx, args.cmd_vy, args.cmd_wz)),
    ]
    phase_steps = max(1, round(args.phase_duration / env.unwrapped.step_dt))

    def force_command(command):
        command = torch.tensor(command, dtype=torch.float32, device=env.unwrapped.device)
        for term in cmd_mgr._terms.values():
            if term.command.shape[-1] >= 3:
                term.command[:, :3] = command

    if args.zero_command:
        force_command((0.0, 0.0, 0.0))
    elif args.phased_commands:
        force_command(phases[0][1])

    frames: list[np.ndarray] = []
    telemetry_time: list[float] = []
    telemetry_joint_pos: list[np.ndarray] = []
    telemetry_contact: list[np.ndarray] = []
    robot_data = env.unwrapped.scene["robot"].data
    rear_joint_indices = [
        i for i, name in enumerate(robot_data.joint_names)
        if name.startswith(("RL_", "RR_")) and any(part in name for part in ("hip", "thigh", "calf"))
    ]
    if len(rear_joint_indices) != 6:
        raise RuntimeError(f"Expected six rear-leg joints, got {[robot_data.joint_names[i] for i in rear_joint_indices]}")
    rear_joint_names = [robot_data.joint_names[i] for i in rear_joint_indices]
    rear_sensor = env.unwrapped.scene.sensors.get("contact_forces_rear_feet")
    if rear_sensor is None:
        raise RuntimeError(f"Missing DVI rear-foot contact sensor; available={list(env.unwrapped.scene.sensors)}")
    frame_stride = max(1, round(1.0 / (args.fps * env.unwrapped.step_dt)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", pixelformat="yuv420p", quality=8)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            phase_index = min(step // phase_steps, len(phases) - 1)
            if args.zero_command:
                force_command((0.0, 0.0, 0.0))
            elif args.phased_commands:
                force_command(phases[phase_index][1])
            if step % frame_stride == 0:
                if args.follow_robot:
                    root = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                    # MuJoCo-style tracking: translate the same world-frame
                    # camera with the base; never orbit/rotate around yaw.
                    camera_offset = (-3.0, -3.0, 1.7) if args.front_view else (3.0, 3.0, 1.7)
                    camera_pos = (float(root[0] + camera_offset[0]), float(root[1] + camera_offset[1]), float(root[2] + camera_offset[2]))
                    camera_target = (float(root[0]), float(root[1]), float(root[2] + 0.45))
                    viewer.camera.pos = type(viewer.camera.pos)(*camera_pos)
                    viewer.camera.look_at(camera_target)
                frame = visualizer.render_rgb_array() if hasattr(visualizer, "render_rgb_array") else viewer.get_frame()
                if frame is None:
                    continue
                frame = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
                frame = np.asarray(frame)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if args.phased_commands:
                    phase_name, command = phases[phase_index]
                    image = Image.fromarray(frame.astype(np.uint8, copy=False)).convert("RGB")
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 57), fill=(15, 15, 15))
                    draw.text((12, 7), f"Native {args.preset}: Phase {phase_index + 1}/4 — {phase_name}", font=font, fill="white")
                    draw.text((12, 32), f"cmd = [{command[0]:.1f}, {command[1]:.1f}, {command[2]:.1f}]  |  t = {step * env.unwrapped.step_dt:.1f}s", font=small, fill=(225, 225, 225))
                    frame = np.asarray(image)
                writer.append_data(frame)
                frames.append(frame)
            if args.telemetry:
                forces = rear_sensor.data.net_forces_w
                forces = forces.torch if hasattr(forces, "torch") else forces
                forces = forces[0].detach().cpu().numpy()
                telemetry_time.append(step * env.unwrapped.step_dt)
                telemetry_joint_pos.append(robot_data.joint_pos[0, rear_joint_indices].detach().cpu().numpy().copy())
                telemetry_contact.append((np.linalg.norm(forces, axis=-1) > 1.0).astype(np.uint8))
    finally:
        writer.close()
        env.close()
    if args.telemetry:
        args.telemetry.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.telemetry,
            time=np.asarray(telemetry_time), joint_pos=np.asarray(telemetry_joint_pos),
            contact=np.asarray(telemetry_contact), joint_names=np.asarray(rear_joint_names),
            contact_names=np.asarray(getattr(rear_sensor, "body_names", ["RL_foot", "RR_foot"])),
        )
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
