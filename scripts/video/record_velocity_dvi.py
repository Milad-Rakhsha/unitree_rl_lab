#!/usr/bin/env python3
"""Record a Go2 velocity DVI checkpoint to MP4 (forces the newton_dvi preset).

play.py does not accept Hydra `presets=` overrides, and the play env cfg
defaults to the PhysX preset. This script loads the play env cfg, resolves the
requested physics preset (default: newton_dvi), loads the checkpoint, and records
a fixed-length rgb_array video.
"""
import argparse
import os

import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "rsl_rl"))
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity-Flat")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--preset", type=str, default="newton_dvi")
parser.add_argument("--output", type=str, default=None)
cli_args.add_rsl_rl_args(parser)  # provides --checkpoint, --resume, --load_run, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import resolve_presets
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import importlib.metadata as metadata


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    # Force the requested physics preset (default PhysX -> newton_dvi).
    resolve_presets(env_cfg, selected={args_cli.preset})

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    out_dir = os.path.join(log_dir, "videos", "play")
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    video_kwargs = {
        "video_folder": out_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.reset()
    steps = args_cli.video_length + 5
    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(obs)
            step_out = env.step(actions)
            obs = step_out[0]

    env.close()
    # Report the produced file.
    mp4s = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
    print("[RECORD_DONE] dir=%s files=%s" % (out_dir, mp4s))
    if args_cli.output and mp4s:
        import shutil
        src = os.path.join(out_dir, sorted(mp4s)[-1])
        shutil.copy(src, args_cli.output)
        print("[RECORD_COPY] %s -> %s" % (src, args_cli.output))
    simulation_app.close()


if __name__ == "__main__":
    main()
