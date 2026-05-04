#!/usr/bin/env python3
"""Play bipedal policy with Newton visualizer for video capture."""
import argparse
import importlib.metadata as metadata
import sys, os

sys.path.insert(0, os.path.expanduser("~/Repos/GO2/unitree_rl_lab/scripts/rsl_rl"))

from packaging import version as pkg_version
from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Unitree-Go2-Bipedal-Walk")
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output", type=str, default="/home/horde/Repos/GO2/isaaclab_bipedal_policy.mp4")
parser.add_argument("--fps", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Use Newton visualizer
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import time
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa
import unitree_rl_lab.tasks  # noqa

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    # Enable Newton rendering in the env config
    env_cfg.sim.enable_newton_rendering = True
    
    # Configure Newton visualizer for headless EGL rendering
    from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg
    newton_cfg = NewtonVisualizerCfg(
        headless=True,
        window_width=1280,
        window_height=720,
        cam_source="cfg",
        eye=(5.0, -5.0, 3.0),
        lookat=(0.0, 0.0, 0.5),
    )
    env_cfg.sim.visualizer_cfgs = [newton_cfg]

    args_cli.resume = None
    args_cli.load_run = None
    args_cli.run_name = None
    args_cli.logger = None
    args_cli.log_project_name = None
    args_cli.experiment_name = None

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(f"[INFO] Loaded: {resume_path}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Running for {args_cli.num_steps} steps")

    # Get the Newton viewer from the sim's visualizers
    sim = env.unwrapped.sim
    viewer = None
    if hasattr(sim, '_visualizers'):
        for v in sim._visualizers:
            if hasattr(v, '_viewer') and hasattr(v._viewer, 'get_frame'):
                viewer = v._viewer
                break
    
    if viewer is None:
        print("[ERROR] Could not find Newton viewer!")
        print(f"  Sim visualizers: {getattr(sim, '_visualizers', 'N/A')}")
        env.close()
        return
    else:
        print(f"[INFO] Found Newton viewer: {type(viewer)}")

    # Run policy and capture frames
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    frames = []
    dt = env.unwrapped.step_dt
    frame_interval = max(1, int(1.0 / (args_cli.fps * dt)))

    print(f"[INFO] Capturing every {frame_interval} steps for {args_cli.fps} fps video")

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)

        if step % frame_interval == 0 and viewer is not None:
            frame = viewer.get_frame()
            if frame is not None:
                # Frame is a warp array, convert to numpy
                import warp as wp
                if hasattr(frame, 'numpy'):
                    frame_np = frame.numpy()
                else:
                    frame_np = np.array(frame)
                if frame_np.size > 0:
                    frames.append(frame_np.copy())

        if step % 100 == 0:
            robot = env.unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w[0].cpu().numpy()
            print(f"  Step {step}/{args_cli.num_steps}, z={root_pos[2]:.3f}, frames={len(frames)}")

    print(f"\nCaptured {len(frames)} frames")

    if frames:
        import mediapy
        mediapy.write_video(args_cli.output, frames, fps=args_cli.fps)
        print(f"Saved to {args_cli.output}")
    else:
        print("ERROR: No frames captured! Newton viewer may not have initialized.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
