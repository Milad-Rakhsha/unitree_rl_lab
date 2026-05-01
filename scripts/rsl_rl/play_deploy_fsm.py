# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-FSM replica for sim-to-sim debugging.

Runs a single Go2 in Isaac Lab under the same finite-state machine used by
the Mujoco deploy (see ``deploy/robots/go2/config/config.yaml``):

    Passive  --(S)-->  FixStand  --(B)-->  BipedalRear (1 s intro, then policy)
        ^                 ^                        |
        | (P)             | (S)                    | (S)
        +-----------------+------------------------+

Key bindings (in the Isaac Sim viewer):

    P   - Passive        (kp=0, kd=3 ; motors limp)
    S   - FixStand       (kp~75, kd~4 ; ramp to training default over 2 s)
    B   - BipedalRear    (intro kp=75/kd=4 for 1 s, then policy at kp=25/kd=0.5)
    R   - Reset env      (teleport robot to spawn pose, zero velocities)
    W/X - forward / backward velocity command
    A/D - strafe left / right velocity command
    Q/E - yaw left / right velocity command
    L   - zero all velocity commands
    C   - write per-step CSV log from this moment on (toggle)

Differences from the Mujoco deploy we intentionally preserve:

* The pre-policy open-loop PD control (FixStand, BipedalRear intro) is
  applied by directly writing the actuator's ``stiffness`` / ``damping``
  tensors, then commanding a joint-position target through the action
  manager's scale/offset. The policy itself is driven exactly the same
  way the Isaac Lab play script does it.
* On the ``FixStand -> BipedalRear`` transition we call
  ``observation_manager.reset()`` so the policy's first inference sees a
  clean history buffer, mirroring the post-intro refresh added in
  ``State_RLBase::policy_loop``.
* ``velocity_commands`` come from the keyboard (not from the command
  manager's uniform sampler). A dead stick reads as zero.

Usage:

    isaaclab.sh -p scripts/rsl_rl/play_deploy_fsm.py \
        --task Unitree-Go2-Bipedal-Walk \
        --checkpoint logs/rsl_rl/unitree_go2_bipedal_walk/2026-04-21_15-56-29/model_2000.pt
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Deploy-FSM replica for Go2 bipedal policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Bipedal-Walk", help="Task id to load the env cfg for.")
parser.add_argument("--csv", type=str, default=None, help="Optional CSV log path (overrides --log_dir default).")
parser.add_argument(
    "--start_state",
    type=str,
    default="FixStand",
    choices=["Passive", "FixStand", "BipedalRear"],
    help="Initial FSM state at startup (defaults to FixStand to match post-LT+A deploy state).",
)
parser.add_argument("--newton_visualizer", action="store_true", default=False, help="Enable Newton rendering.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
# Force the Kit viewer on: it owns ``omni.appwindow``, which is what our
# keyboard hook subscribes to. Without ``--visualizer kit`` the app runs
# headless and the keyboard subscription silently has no window to attach to.
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
args_cli.num_envs = 1  # debug rig is explicitly single-robot

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports depend on the sim app being alive."""

import csv
import os
import time
import weakref
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import carb
import omni  # noqa: F401 -- ``omni.appwindow`` is populated as an attribute once the Kit viewer is up.

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.timer import Timer
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

Timer.enable = False
Timer.enable_display_output = False


# ---------------------------------------------------------------------------
# FSM constants — mirror deploy/robots/go2/config/config.yaml
# ---------------------------------------------------------------------------

# FixStand's real deploy PD is asymmetric across hip / thigh / calf:
#     kp = [60, 80, 80] per leg ; kd = [5, 4, 4] per leg
# For a debug rig on ideal (non-motor-limited) gains we use a uniform
# kp~75, kd~4 that sits in the same ballpark. Making this per-joint would
# require matching SDK<->Isaac Lab joint order, which is a pure
# bookkeeping hazard for no real diagnostic value here.
FIXSTAND_KP = 75.0
FIXSTAND_KD = 4.0
FIXSTAND_RAMP_SECONDS = 2.0

PASSIVE_KP = 0.0
PASSIVE_KD = 3.0

# Policy-native gains from UNITREE_GO2_CFG.actuators["GO2HV"].
BIPEDAL_KP = 25.0
BIPEDAL_KD = 0.5
BIPEDAL_INTRO_SECONDS = 1.0

# During the intro, hold FixStand-level gains so the legs actually sit at the
# training-default pose instead of sagging ~0.2 rad past it. kp drops to
# BIPEDAL_KP at policy handover, matching the deploy YAML override
# (BipedalRear.intro.kp). Without this, the pose at handover is
# out-of-distribution for the policy and the robot flips on the first step.
BIPEDAL_INTRO_KP = FIXSTAND_KP
BIPEDAL_INTRO_KD = FIXSTAND_KD


@dataclass
class FSMFrame:
    """Output of ``DeployFSM.step()`` for a single control tick."""

    kp: float
    kd: float
    # If ``target_q`` is None, the caller feeds the policy's action through.
    # Otherwise the caller must compute ``action = (target_q - offset) / scale``
    # so the action manager produces exactly this target.
    target_q: torch.Tensor | None
    policy_active: bool
    state: str
    intro_elapsed: float  # seconds since entering the current state


class DeployFSM:
    """Tiny state machine that mirrors the Mujoco deploy transition graph."""

    PASSIVE = "Passive"
    FIXSTAND = "FixStand"
    BIPEDAL = "BipedalRear"

    def __init__(self, default_q: torch.Tensor, initial_q: torch.Tensor, start_state: str):
        self._default_q = default_q.clone()
        self._state = start_state
        self._q_at_entry = initial_q.clone()
        self._entry_time = time.monotonic()

    @property
    def state(self) -> str:
        return self._state

    def request(self, new_state: str, current_q: torch.Tensor) -> None:
        """Request a transition on the next control tick."""
        if new_state not in (self.PASSIVE, self.FIXSTAND, self.BIPEDAL):
            raise ValueError(f"Unknown FSM state: {new_state}")
        self._state = new_state
        self._q_at_entry = current_q.clone()
        self._entry_time = time.monotonic()

    def step(self, current_q: torch.Tensor) -> FSMFrame:
        elapsed = time.monotonic() - self._entry_time

        if self._state == self.PASSIVE:
            # Motors limp: damping-only. ``target_q = current_q`` would also
            # work since kp=0 ignores it, but we set target=default to keep
            # the action-manager's ``last_action`` deterministic.
            return FSMFrame(PASSIVE_KP, PASSIVE_KD, self._default_q, False, self._state, elapsed)

        if self._state == self.FIXSTAND:
            alpha = min(elapsed / FIXSTAND_RAMP_SECONDS, 1.0)
            target = self._q_at_entry + alpha * (self._default_q - self._q_at_entry)
            return FSMFrame(FIXSTAND_KP, FIXSTAND_KD, target, False, self._state, elapsed)

        # BipedalRear
        if elapsed < BIPEDAL_INTRO_SECONDS:
            alpha = elapsed / BIPEDAL_INTRO_SECONDS
            target = self._q_at_entry + alpha * (self._default_q - self._q_at_entry)
            return FSMFrame(BIPEDAL_INTRO_KP, BIPEDAL_INTRO_KD, target, False, self._state, elapsed)

        return FSMFrame(BIPEDAL_KP, BIPEDAL_KD, None, True, self._state, elapsed)


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------


class DeployKeyboard:
    """Captures key events in the Isaac Sim viewer and exposes an FSM request
    queue + a persistent velocity-command vector.
    """

    # Velocity-command increments per key press, in [vx, vy, yaw_rate].
    _VEL_BINDINGS: dict[str, tuple[int, float]] = {
        "W": (0, +0.5),
        "X": (0, -0.5),
        "A": (1, +0.3),
        "D": (1, -0.3),
        "Q": (2, +0.5),
        "E": (2, -0.5),
    }

    def __init__(self) -> None:
        self.pending_state: str | None = None
        self.pending_reset: bool = False
        self.pending_csv_toggle: bool = False
        self.vel_cmd = np.zeros(3, dtype=np.float32)

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

    def __del__(self) -> None:
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def _on_event(self, event, *_args, **_kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key == "P":
                self.pending_state = DeployFSM.PASSIVE
            elif key == "S":
                self.pending_state = DeployFSM.FIXSTAND
            elif key == "B":
                self.pending_state = DeployFSM.BIPEDAL
            elif key == "R":
                self.pending_reset = True
            elif key == "L":
                self.vel_cmd[:] = 0.0
            elif key == "C":
                self.pending_csv_toggle = True
            elif key in self._VEL_BINDINGS:
                i, v = self._VEL_BINDINGS[key]
                self.vel_cmd[i] += v
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input.name
            if key in self._VEL_BINDINGS:
                i, v = self._VEL_BINDINGS[key]
                self.vel_cmd[i] -= v
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tt(arr_or_tensor) -> torch.Tensor:
    """Return a torch tensor view of a Warp array or a torch tensor."""
    return arr_or_tensor if isinstance(arr_or_tensor, torch.Tensor) else wp.to_torch(arr_or_tensor)


def _override_event_params(env_cfg) -> None:
    """Disable the randomization pieces that we want deterministic for
    a debug session (so two back-to-back sessions replay identically)."""
    if getattr(env_cfg.events, "reset_robot_joints", None) is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    # Disable falling-out terminations so the user can watch a flipped
    # robot instead of being bounced back to spawn mid-experiment.
    for term_name in ("base_contact", "hip_contact", "bad_orientation"):
        if getattr(env_cfg.terminations, term_name, None) is not None:
            setattr(env_cfg.terminations, term_name, None)


def _print_legend(start_state: str) -> None:
    print("=" * 72)
    print("Deploy-FSM replica — key bindings")
    print("=" * 72)
    print("  P  Passive        (kp=0, kd=3 ; motors limp)")
    print("  S  FixStand       (kp~75, kd~4 ; 2 s ramp to training default)")
    print("  B  BipedalRear    (intro kp=75/kd=4 for 1 s, then policy at kp=25/kd=0.5)")
    print("  R  Reset env      (teleport to spawn)")
    print("  W/X/A/D/Q/E  velocity commands (forward/back/strafe/yaw)")
    print("  L  Zero velocity commands")
    print("  C  Toggle CSV logging")
    print("-" * 72)
    print(f"Starting FSM state: {start_state}")
    print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.sim.enable_newton_rendering = args_cli.newton_visualizer
    _override_event_params(env_cfg)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- pull the handles we'll poke each step ---
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action_term = unwrapped.action_manager.get_term("JointPositionAction")
    joint_ids = action_term._joint_ids  # slice or list of ints
    action_scale = action_term._scale  # float or tensor
    action_offset = action_term._offset  # tensor (num_envs, action_dim)
    device = unwrapped.device

    # ``actuator.stiffness`` and ``actuator.damping`` are explicit-actuator
    # tensors we can overwrite on every tick. The Go2 config wires all joints
    # through a single ``GO2HV`` actuator, but the key lookup below is generic.
    actuators = list(robot.actuators.values())
    assert len(actuators) == 1, "This script assumes a single actuator group for the Go2."
    actuator = actuators[0]

    default_q_full = _tt(robot.data.default_joint_pos)  # (num_envs, num_joints)
    default_q = default_q_full[:, joint_ids].clone()  # (num_envs, action_dim)

    # --- FSM + input ---
    current_q = _tt(robot.data.joint_pos)[:, joint_ids].clone()
    fsm = DeployFSM(default_q=default_q[0], initial_q=current_q[0], start_state=args_cli.start_state)
    keyboard = DeployKeyboard()
    _print_legend(args_cli.start_state)

    # --- optional CSV logging ---
    csv_path: str | None = args_cli.csv or os.path.join(os.path.dirname(resume_path), "play_deploy_fsm.csv")
    csv_writer = None
    csv_file = None
    csv_enabled = False

    def _open_csv() -> None:
        nonlocal csv_writer, csv_file, csv_enabled
        if csv_file is not None:
            return
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = ["t_wall", "state", "intro_elapsed"]
        header += [f"qpos_{i}" for i in range(default_q.shape[-1])]
        header += [f"qvel_{i}" for i in range(default_q.shape[-1])]
        header += ["base_roll", "base_pitch", "base_yaw"]
        header += ["base_vx_b", "base_vy_b", "base_vz_b"]
        header += ["base_wx_b", "base_wy_b", "base_wz_b"]
        header += [f"action_{i}" for i in range(default_q.shape[-1])]
        header += ["cmd_vx", "cmd_vy", "cmd_yaw"]
        csv_writer.writerow(header)
        csv_file.flush()
        csv_enabled = True
        print(f"[INFO] CSV log started: {csv_path}", flush=True)

    def _close_csv() -> None:
        nonlocal csv_writer, csv_file, csv_enabled
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
            csv_file = None
            csv_writer = None
        csv_enabled = False
        print("[INFO] CSV log stopped.", flush=True)

    # --- warm up: one get_observations so the obs history is primed ---
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # --- main loop ---
    step_dt = unwrapped.step_dt
    env_ids_all = torch.arange(1, device=device)

    try:
        while simulation_app.is_running():
            loop_t0 = time.time()

            # 1) read current joint state
            joint_pos = _tt(robot.data.joint_pos)[:, joint_ids]
            current_q_0 = joint_pos[0].clone()

            # 2) handle keyboard requests
            if keyboard.pending_state is not None:
                fsm.request(keyboard.pending_state, current_q_0)
                print(f"[FSM] -> {keyboard.pending_state}", flush=True)
                if keyboard.pending_state == DeployFSM.BIPEDAL:
                    # mirror ``State_RLBase::policy_loop`` obs refresh
                    # after intro — call it at entry so the first inference
                    # sees an all-FixStand-pose history instead of mixed.
                    unwrapped.observation_manager.reset(env_ids_all)
                    obs = env.get_observations()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                keyboard.pending_state = None

            if keyboard.pending_reset:
                env.reset()
                # get_observations() after reset may return a bare tensor
                obs = env.get_observations()
                if isinstance(obs, tuple):
                    obs = obs[0]
                current_q_0 = _tt(robot.data.joint_pos)[0, joint_ids].clone()
                fsm = DeployFSM(default_q=default_q[0], initial_q=current_q_0, start_state=args_cli.start_state)
                print("[FSM] env reset", flush=True)
                keyboard.pending_reset = False

            if keyboard.pending_csv_toggle:
                if csv_enabled:
                    _close_csv()
                else:
                    _open_csv()
                keyboard.pending_csv_toggle = False

            # 3) overwrite velocity command (kept user-driven, not sampled)
            cmd_term = unwrapped.command_manager.get_term("base_velocity")
            cmd_term.vel_command_b[:] = torch.as_tensor(keyboard.vel_cmd, device=device).unsqueeze(0)

            # 4) FSM tick
            frame = fsm.step(current_q_0)

            # 5) push kp/kd into the actuator every tick — cheap, avoids
            # "stale gains" bugs when a transition races the first poll.
            actuator.stiffness[:] = frame.kp
            actuator.damping[:] = frame.kd

            # 6) build the action
            if frame.policy_active:
                with torch.inference_mode():
                    action = policy(obs)
            else:
                # processed_target = offset + action * scale  =>  action = (target - offset) / scale
                target = frame.target_q.unsqueeze(0) if frame.target_q is not None else default_q
                action = (target - action_offset) / action_scale

            # 7) step the env
            with torch.inference_mode():
                obs, _, _, _ = env.step(action)

            # 8) optional CSV row
            if csv_enabled:
                root_quat = _tt(robot.data.root_quat_w)[0]
                root_lin_b = _tt(robot.data.root_lin_vel_b)[0]
                root_ang_b = _tt(robot.data.root_ang_vel_b)[0]
                qv = _tt(robot.data.joint_vel)[0, joint_ids]
                qp = _tt(robot.data.joint_pos)[0, joint_ids]
                # quat (w, x, y, z) -> roll/pitch/yaw
                w, x, y, z = (root_quat[0].item(), root_quat[1].item(), root_quat[2].item(), root_quat[3].item())
                roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
                pitch = np.arcsin(pitch_arg)
                yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                row: list[float | str] = [f"{time.time():.6f}", frame.state, f"{frame.intro_elapsed:.6f}"]
                row += [f"{v:.6f}" for v in qp.cpu().tolist()]
                row += [f"{v:.6f}" for v in qv.cpu().tolist()]
                row += [f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}"]
                row += [f"{v:.6f}" for v in root_lin_b.cpu().tolist()]
                row += [f"{v:.6f}" for v in root_ang_b.cpu().tolist()]
                row += [f"{v:.6f}" for v in action[0].detach().cpu().tolist()]
                row += [f"{v:.6f}" for v in keyboard.vel_cmd.tolist()]
                csv_writer.writerow(row)

            # 9) pace to real-time (so the Isaac Lab viewer and Mujoco sit
            # at the same wall-clock rate for side-by-side comparison).
            sleep_time = step_dt - (time.time() - loop_t0)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if csv_file is not None:
            csv_file.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
