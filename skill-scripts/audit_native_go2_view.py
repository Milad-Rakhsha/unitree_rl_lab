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
parser.add_argument("--telemetry", type=pathlib.Path, help="Optional NPZ output with policy inputs/actions, full robot state, actuator effort, feet, and contacts.")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-envs", type=int, default=1, help="Number of simultaneous environments to render.")
parser.add_argument("--focus-left", action="store_true", help="Zoom and focus on the left side of a multi-environment grid.")
parser.add_argument("--camera-eye", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Explicit fixed-world camera eye; overrides automatic framing.")
parser.add_argument("--camera-lookat", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Explicit fixed-world camera target; overrides automatic framing.")
parser.add_argument("--camera-fov", type=float, default=35.0, help="Camera vertical field of view in degrees.")
parser.add_argument("--preset", choices=("newton_dvi", "newton_mjwarp"), default="newton_dvi")
parser.add_argument("--joint-limit-iterations", type=int)
parser.add_argument("--coupling-iterations", type=int)
parser.add_argument("--disable-post-stabilization", action="store_true")
parser.add_argument("--contact-regularization", type=float)
parser.add_argument("--contact-recovery-speed", type=float)
parser.add_argument("--contact-max-iterations", type=int)
parser.add_argument("--contact-alpha", type=float)
parser.add_argument("--joint-alpha", type=float)
parser.add_argument("--joint-recovery-speed", type=float)
parser.add_argument("--zero-command", action="store_true", help="Force zero velocity commands.")
parser.add_argument("--deterministic-reset", action="store_true", help="Use identical nominal root and joint reset state (no reset perturbations).")
parser.add_argument("--initial-root-z", type=float, default=None, help="Diagnostic only: override the nominal root height before reset.")
parser.add_argument("--upright-root-lift", type=float, default=0.0, help="Additional fixed-root vertical clearance (m) for --upright-perturbation; keeps all robot geoms clear of the terrain.")
parser.add_argument("--joint-armature", type=float, default=None, help="Diagnostic only: override every actuated joint's armature.")
parser.add_argument("--disable-terminations", action="store_true", help="Diagnostic only: disable automatic episode terminations/resets.")
parser.add_argument("--single-command", action="store_true", help="Hold [--cmd-vx, --cmd-vy, --cmd-wz] throughout the recording.")
parser.add_argument("--flat-terrain", action="store_true", help="Evaluate the rough-trained policy on Isaac Lab's infinite flat plane.")
parser.add_argument("--follow-robot", action="store_true", help="Track environment 0's robot with a fixed-offset camera.")
parser.add_argument("--front-view", action="store_true", help="Use the opposite fixed-world tracking offset, matching the MuJoCo recorder's front-facing view.")
parser.add_argument("--phased-commands", action="store_true", help="Play forward-x, lateral-y, yaw, then combined commands.")
parser.add_argument("--zero-neg-pos", action="store_true", help="Play zero command, then negative-x, then positive-x.")
parser.add_argument("--phase-duration", type=float, default=2.5, help="Seconds per command phase.")
parser.add_argument("--zero-duration", type=float, default=10.0, help="Zero-command duration for --zero-neg-pos.")
parser.add_argument("--motion-duration", type=float, default=5.0, help="Duration of each signed-x phase for --zero-neg-pos.")
parser.add_argument("--cmd-vx", type=float, default=0.7)
parser.add_argument("--cmd-vy", type=float, default=0.3)
parser.add_argument("--cmd-wz", type=float, default=0.7)
parser.add_argument("--fixed-action-from-first", action="store_true", help="Diagnostic only: hold the first policy action for all control steps.")
parser.add_argument("--actuator-excitation", action="store_true", help="Phase-1 actuator test: fixed base, global gravity off, robot collisions off, and sequential open-loop joint target steps (no policy inference).")
parser.add_argument("--multibody-excitation", action="store_true", help="Phase-2 test: free root, collisions off, sequential open-loop joint target steps (no policy inference).")
parser.add_argument("--multibody-gravity", choices=("off", "on"), default="off", help="Gravity condition for --multibody-excitation.")
parser.add_argument("--excitation-amplitude", type=float, default=0.12, help="Joint-target amplitude in rad for --actuator-excitation.")
parser.add_argument("--excitation-rest", type=float, default=0.20, help="Seconds at zero target before/after each joint step in --actuator-excitation.")
parser.add_argument("--excitation-hold", type=float, default=0.40, help="Seconds at each signed target plateau in --actuator-excitation.")
parser.add_argument("--excitation-joints", type=int, default=12, help="Number of leading joints to excite sequentially; Phase-2 gravity-on uses a bounded one-joint window to avoid a long collision-free free fall.")
parser.add_argument("--upright-perturbation", action="store_true", help="Fixed-root Phase-1b test: drive all joints around the bipedal default posture with a deterministic zero-command-like then walking-like perturbation profile.")
parser.add_argument("--perturbation-zero-std", type=float, default=0.15, help="Per-joint target standard deviation (rad) in the first, zero-command-like half of --upright-perturbation.")
parser.add_argument("--perturbation-walk-std", type=float, default=0.17, help="Per-joint target standard deviation (rad) in the second, walking-command-like half of --upright-perturbation.")
parser.add_argument("--perturbation-half-duration", type=float, default=4.0, help="Duration (s) of each half of --upright-perturbation.")
# Low-velocity deterministic zero-command sample from the frozen bipedal
# rollout. The native replay convention has been visually validated: its
# recorded tuple is used directly by InitialStateCfg.rot. MuJoCo receives the
# corresponding wxyz tuple in its counterpart runner.
UPRIGHT_BIPEDAL_ROOT_POS = (0.0, 0.0, 0.541318)
UPRIGHT_BIPEDAL_ROOT_QUAT_XYZW = (0.059433, -0.738475, 0.043924, 0.670218)
UPRIGHT_BIPEDAL_TARGET = (0.101630, 0.734117, -1.453078, -0.090235, 0.764567, -1.446473, 0.137010, 2.307323, -1.453031, -0.235372, 2.290217, -1.482530)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import isaaclab.sim as sim_utils
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
        cfg_class_name = (
            "RobotBipedalWalkRoughPlayEnvCfg"
            if "Bipedal" in task
            else "RobotRoughPlayEnvCfg"
        )
        cfg_class = getattr(module, cfg_class_name)
        cfg = cfg_class()
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
    if args.contact_max_iterations is not None:
        cfg.sim.physics.solver_cfg.contact_max_iterations = args.contact_max_iterations
    if args.contact_alpha is not None:
        cfg.sim.physics.solver_cfg.contact_alpha = args.contact_alpha
    if args.joint_alpha is not None:
        cfg.sim.physics.solver_cfg.joint_alpha = args.joint_alpha
    if args.joint_recovery_speed is not None:
        cfg.sim.physics.solver_cfg.joint_recovery_speed = args.joint_recovery_speed
    solver_cfg = cfg.sim.physics.solver_cfg
    print(
        "EFFECTIVE_DVI_PARAMS "
        f"contact_solver_type={solver_cfg.contact_solver_type} "
        f"contact_early_exit={solver_cfg.contact_early_exit} "
        f"contact_tolerance={solver_cfg.contact_tolerance} "
        f"contact_residual_mode={solver_cfg.contact_residual_mode} "
        f"contact_max_iterations={solver_cfg.contact_max_iterations} "
        f"contact_omega={solver_cfg.contact_omega} "
        f"contact_relax={solver_cfg.contact_relax} "
        f"contact_reg={solver_cfg.contact_reg} "
        f"contact_compliance={solver_cfg.contact_compliance} "
        f"contact_alpha={solver_cfg.contact_alpha} "
        f"contact_recovery_speed={solver_cfg.contact_recovery_speed} "
        f"contact_friction_projection={solver_cfg.contact_friction_projection} "
        f"joint_solver_type={solver_cfg.joint_solver_type} "
        f"joint_reg={solver_cfg.joint_reg} "
        f"joint_alpha={solver_cfg.joint_alpha} "
        f"joint_recovery_speed={solver_cfg.joint_recovery_speed} "
        f"joint_iterative_refinement_steps={solver_cfg.joint_iterative_refinement_steps} "
        f"coupling_iterations={solver_cfg.coupling_iterations} "
        f"post_stabilize_joints={solver_cfg.post_stabilize_joints} "
        f"actuator_integration={solver_cfg.actuator_integration} "
        f"deterministic={solver_cfg.deterministic} "
        f"num_substeps={cfg.sim.physics.num_substeps}",
        flush=True,
    )
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.initial_root_z is not None:
        init_pos = list(cfg.scene.robot.init_state.pos)
        init_pos[2] = args.initial_root_z
        cfg.scene.robot.init_state.pos = tuple(init_pos)
    if args.joint_armature is not None:
        for actuator_cfg in cfg.scene.robot.actuators.values():
            actuator_cfg.armature = args.joint_armature
    if args.disable_terminations:
        for name in vars(cfg.terminations):
            if not name.startswith("_"):
                setattr(cfg.terminations, name, None)
    if args.actuator_excitation or args.upright_perturbation:
        # Phase 1 isolates the exact native actuator path from root, gravity,
        # terrain, and contact.  Use a real fixed root constraint rather than
        # overwriting base state after each integration step.
        cfg.sim.gravity = (0.0, 0.0, 0.0)
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    if args.multibody_excitation:
        # Phase 2 retains the floating base while removing contact constraints.
        # The same policy-disabled Go2HV target sequence then probes only free
        # multibody/integration response, with gravity as the sole switch.
        if args.multibody_gravity == "off":
            cfg.sim.gravity = (0.0, 0.0, 0.0)
            cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        else:
            cfg.sim.gravity = (0.0, 0.0, -9.81)
            cfg.scene.robot.spawn.rigid_props.disable_gravity = False
        cfg.scene.robot.spawn.articulation_props.fix_root_link = False
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    if args.upright_perturbation:
        # Start at the measured upright bipedal target itself, lifted only by
        # the explicit diagnostic clearance. This avoids terrain contact
        # while retaining its walking-root orientation.
        cfg.scene.robot.init_state.pos = (UPRIGHT_BIPEDAL_ROOT_POS[0], UPRIGHT_BIPEDAL_ROOT_POS[1], UPRIGHT_BIPEDAL_ROOT_POS[2] + args.upright_root_lift)
        cfg.scene.robot.init_state.rot = UPRIGHT_BIPEDAL_ROOT_QUAT_XYZW
        cfg.scene.robot.init_state.joint_pos = {
            name: float(value) for name, value in zip(
                ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
                UPRIGHT_BIPEDAL_TARGET,
            )
        }
    if args.deterministic_reset:
        cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.sim.enable_newton_rendering = True
    if args.flat_terrain:
        # Retain the run's native solver configuration while replacing only the
        # evaluation geometry with the same infinite plane used by the base task.
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.max_init_terrain_level = None
        if hasattr(cfg.curriculum, "terrain_levels"):
            cfg.curriculum.terrain_levels = None
    # One viewport that encompasses the replicated scene.  The 64-env case is
    # an 8x8 layout with 2.5 m spacing, so use an elevated wide shot; retain
    # the close single-robot framing for ordinary videos.
    if args.num_envs == 1:
        # Match the MuJoCo transfer recorder's fixed world-frame tracking view:
        # front-left looking toward the base.  This offset translates with the
        # base but never rotates/orbits with its yaw.
        eye, lookat = ((-3.2, -3.2, 2.7), (0.0, 0.0, 1.5)) if args.front_view else ((3.2, 3.2, 2.7), (0.0, 0.0, 1.5))
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
    if (args.camera_eye is None) != (args.camera_lookat is None):
        raise ValueError("--camera-eye and --camera-lookat must be supplied together")
    if args.camera_eye is not None:
        eye = tuple(args.camera_eye)
        lookat = tuple(args.camera_lookat)
    cfg.sim.visualizer_cfgs = [
        NewtonVisualizerCfg(
            headless=True,
            window_width=1280,
            window_height=720,
            eye=eye,
            lookat=lookat,
            # Use the normal Newton presentation rather than a black/flat
            # diagnostic background.
            enable_shadows=True,
            enable_sky=True,
        )
    ]

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(task, cfg=cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    robot_data_for_params = env.unwrapped.scene["robot"].data
    robot = env.unwrapped.scene["robot"]
    model = robot.root_view.model
    print("AUDIT_BODY_NAMES", list(enumerate(robot.body_names)), flush=True)
    print("AUDIT_VIEW_TYPE", type(robot.root_view).__name__, flush=True)
    print("AUDIT_VIEW_LINKS", list(enumerate(robot.root_view.link_names)), flush=True)
    print("AUDIT_MODEL_BODIES", list(enumerate(model.body_label)), flush=True)
    print("AUDIT_MODEL_BODY_WORLD", model.body_world.numpy().tolist(), flush=True)
    print("AUDIT_MODEL_BODY_Q", model.body_q.numpy().tolist(), flush=True)
    for key, layout in robot.root_view.frequency_layouts.items():
        print("AUDIT_LAYOUT", key, "offset", layout.offset, "between", layout.stride_between_worlds, "within", layout.stride_within_worlds, "value_count", layout.value_count, "slice", layout.slice, flush=True)
    print("AUDIT_VIEW_TRANSFORMS", robot.root_view.get_link_transforms(env.unwrapped.sim._state_0).numpy().tolist(), flush=True)
    env.close()
    return

    def _param_np(value):
        value = value.torch if hasattr(value, "torch") else value
        array = value.detach().cpu().numpy()
        return array[0] if array.ndim > 1 else array

    runtime_params = {
        name: _param_np(getattr(robot_data_for_params, name))
        for name in (
            "joint_armature",
            "joint_friction_coeff",
            "joint_stiffness",
            "joint_damping",
            "default_mass",
        )
        if hasattr(robot_data_for_params, name)
    }
    print(
        "NATIVE_MODEL_PARAMS "
        + " ".join(f"{name}={value.tolist()}" for name, value in runtime_params.items()),
        flush=True,
    )
    policy = None
    if not (args.actuator_excitation or args.multibody_excitation or args.upright_perturbation):
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    visualizer = next((v for v in env.unwrapped.sim._visualizers if hasattr(v, "_viewer")), None)
    if visualizer is None:
        raise RuntimeError(f"Newton visualizer was not initialized: {env.unwrapped.sim._visualizers!r}")
    viewer = visualizer._viewer

    # Camera is updated per frame below when --follow-robot is enabled.
    viewer.camera.fov = args.camera_fov

    if sum((args.zero_command, args.single_command, args.phased_commands, args.zero_neg_pos, args.actuator_excitation, args.multibody_excitation, args.upright_perturbation)) > 1:
        raise ValueError("command playback modes, --actuator-excitation, --multibody-excitation, and --upright-perturbation are mutually exclusive")
    cmd_mgr = env.unwrapped.command_manager
    phases = [
        ("forward x", (args.cmd_vx, 0.0, 0.0)),
        ("lateral y", (0.0, args.cmd_vy, 0.0)),
        ("yaw", (0.0, 0.0, args.cmd_wz)),
        ("combined", (args.cmd_vx, args.cmd_vy, args.cmd_wz)),
    ]
    phase_steps = max(1, round(args.phase_duration / env.unwrapped.step_dt))
    diagnostic_phases = [
        ("zero command", (0.0, 0.0, 0.0), max(1, round(args.zero_duration / env.unwrapped.step_dt))),
        ("negative x", (-abs(args.cmd_vx), 0.0, 0.0), max(1, round(args.motion_duration / env.unwrapped.step_dt))),
        ("positive x", (abs(args.cmd_vx), 0.0, 0.0), max(1, round(args.motion_duration / env.unwrapped.step_dt))),
    ]
    diagnostic_boundaries = np.cumsum([p[2] for p in diagnostic_phases])

    def force_command(command):
        command = torch.tensor(command, dtype=torch.float32, device=env.unwrapped.device)
        for term in cmd_mgr._terms.values():
            if term.command.shape[-1] >= 3:
                term.command[:, :3] = command

    if args.zero_command:
        force_command((0.0, 0.0, 0.0))
    elif args.single_command:
        force_command((args.cmd_vx, args.cmd_vy, args.cmd_wz))
    elif args.phased_commands:
        force_command(phases[0][1])
    elif args.zero_neg_pos:
        force_command(diagnostic_phases[0][1])

    # The first policy observation must contain the same forced command too.
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    frames: list[np.ndarray] = []
    root_start = None
    root_end = None
    telemetry_time: list[float] = []
    telemetry_joint_pos: list[np.ndarray] = []
    telemetry_contact: list[np.ndarray] = []
    telemetry_obs: list[np.ndarray] = []
    telemetry_action: list[np.ndarray] = []
    telemetry_all_joint_pos: list[np.ndarray] = []
    telemetry_body_pos: list[np.ndarray] = []
    telemetry_joint_vel: list[np.ndarray] = []
    telemetry_joint_target: list[np.ndarray] = []
    telemetry_computed_torque: list[np.ndarray] = []
    telemetry_applied_torque: list[np.ndarray] = []
    telemetry_root_pos: list[np.ndarray] = []
    telemetry_root_quat: list[np.ndarray] = []
    telemetry_root_lin_vel: list[np.ndarray] = []
    telemetry_root_ang_vel: list[np.ndarray] = []
    telemetry_projected_gravity: list[np.ndarray] = []
    telemetry_rear_foot_pos: list[np.ndarray] = []
    telemetry_rear_contact_force: list[np.ndarray] = []
    telemetry_done: list[np.ndarray] = []
    robot_data = env.unwrapped.scene["robot"].data
    rear_joint_indices = []
    rear_joint_names = []
    rear_sensor = None
    rear_foot_sensors = []
    if args.telemetry:
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
        rear_foot_sensors = [
            env.unwrapped.scene.sensors["rear_left_foot_kinematics"],
            env.unwrapped.scene.sensors["rear_right_foot_kinematics"],
        ]
    frame_stride = max(1, round(1.0 / (args.fps * env.unwrapped.step_dt)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", pixelformat="yuv420p", quality=8)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    fixed_action = None
    if not 1 <= args.excitation_joints <= 12:
        raise ValueError("--excitation-joints must be in [1, 12]")
    excitation_segment_steps = max(1, round((args.excitation_rest + 2.0 * args.excitation_hold + args.excitation_rest) / env.unwrapped.step_dt))
    try:
        for step in range(args.steps):
            if args.zero_neg_pos:
                phase_index = min(int(np.searchsorted(diagnostic_boundaries, step, side="right")), len(diagnostic_phases) - 1)
                command = diagnostic_phases[phase_index][1]
                force_command(command)
            with torch.inference_mode():
                policy_obs = obs.clone()
                if args.actuator_excitation or args.multibody_excitation:
                    # Raw action is target offset / 0.25, matching the frozen
                    # JointPositionAction configuration.  Each joint receives
                    # rest → +step → -step → rest while every other target is
                    # held at its exact default.
                    num_joints = len(robot_data.joint_names)
                    joint_index = min(step // excitation_segment_steps, args.excitation_joints - 1)
                    local = step % excitation_segment_steps
                    n_rest = max(1, round(args.excitation_rest / env.unwrapped.step_dt))
                    n_hold = max(1, round(args.excitation_hold / env.unwrapped.step_dt))
                    target_offset = 0.0 if local < n_rest or local >= n_rest + 2 * n_hold else (args.excitation_amplitude if local < n_rest + n_hold else -args.excitation_amplitude)
                    actions = torch.zeros((args.num_envs, num_joints), dtype=torch.float32, device=env.unwrapped.device)
                    actions[:, joint_index] = target_offset / 0.25
                elif args.upright_perturbation:
                    # Fixed, deterministic multi-joint targets around the
                    # bipedal default pose. The first half uses the measured
                    # zero-command target variation; the second uses the
                    # walking-phase magnitude.  Different joint phases expose
                    # coupling without policy feedback.
                    num_joints = len(robot_data.joint_names)
                    time_s = step * env.unwrapped.step_dt
                    std = args.perturbation_zero_std if time_s < args.perturbation_half_duration else args.perturbation_walk_std
                    peak = std * np.sqrt(2.0)
                    # Start exactly at the recorded bipedal pose, rather than
                    # giving different joints a nonzero phase-offset target at
                    # t=0. Alternating signs retain multi-joint coupling while
                    # the common smooth waveform provides the requested RMS.
                    signs = torch.tensor((1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1), device=env.unwrapped.device, dtype=torch.float32)
                    offset = peak * np.sin(2.0 * np.pi * 0.75 * time_s) * signs
                    upright = torch.tensor(UPRIGHT_BIPEDAL_TARGET, dtype=torch.float32, device=env.unwrapped.device)
                    default = robot_data.default_joint_pos[0]
                    actions = ((upright + offset - default) / 0.25).expand(args.num_envs, -1).clone()
                if args.zero_neg_pos:
                    # Patch only the inference input.  The command term is
                    # simultaneously forced above, so the next environment
                    # observation carries the same command natively.
                    history_length = (policy_obs["policy"].shape[-1] - 15) // 30
                    command_start = 6 * history_length
                    policy_obs["policy"][:, command_start:command_start + 3] = torch.as_tensor(
                        command, dtype=policy_obs["policy"].dtype, device=policy_obs["policy"].device
                    )
                if args.actuator_excitation or args.multibody_excitation or args.upright_perturbation:
                    pass
                elif fixed_action is None:
                    actions = policy(policy_obs)
                    if args.fixed_action_from_first:
                        fixed_action = actions.clone()
                elif args.fixed_action_from_first:
                    actions = fixed_action
                else:
                    actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if policy is not None:
                    policy.reset(dones)
            root_value = env.unwrapped.scene["robot"].data.root_pos_w[0]
            root_value = root_value.torch if hasattr(root_value, "torch") else root_value
            root_end = root_value.detach().cpu().numpy().copy()
            if root_start is None:
                root_start = root_end.copy()
            if not args.zero_neg_pos:
                phase_index = min(step // phase_steps, len(phases) - 1)
            if args.zero_command:
                force_command((0.0, 0.0, 0.0))
            elif args.single_command:
                force_command((args.cmd_vx, args.cmd_vy, args.cmd_wz))
            elif args.phased_commands:
                force_command(phases[phase_index][1])
            if step % frame_stride == 0:
                if args.follow_robot:
                    root = env.unwrapped.scene["robot"].data.root_pos_w[0]
                    root = root.torch if hasattr(root, "torch") else root
                    root = root.detach().cpu().numpy()
                    # MuJoCo-style tracking: translate the same world-frame
                    # camera with the base; never orbit/rotate around yaw.
                    camera_offset = (-3.0, -3.0, 1.9) if args.front_view else (3.0, 3.0, 1.9)
                    camera_pos = (float(root[0] + camera_offset[0]), float(root[1] + camera_offset[1]), float(root[2] + camera_offset[2]))
                    camera_target = (float(root[0]), float(root[1]), float(root[2] + 0.20))
                    viewer.camera.pos = type(viewer.camera.pos)(*camera_pos)
                    viewer.camera.look_at(camera_target)
                frame = visualizer.render_rgb_array() if hasattr(visualizer, "render_rgb_array") else viewer.get_frame()
                if frame is None:
                    continue
                frame = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
                frame = np.asarray(frame)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if args.phased_commands or args.zero_neg_pos:
                    phase_name, command = (phases[phase_index] if args.phased_commands else diagnostic_phases[phase_index][:2])
                    image = Image.fromarray(frame.astype(np.uint8, copy=False)).convert("RGB")
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 57), fill=(15, 15, 15))
                    phase_count = len(phases) if args.phased_commands else len(diagnostic_phases)
                    draw.text((12, 7), f"Native {args.preset}: Phase {phase_index + 1}/{phase_count} — {phase_name}", font=font, fill="white")
                    draw.text((12, 32), f"cmd = [{command[0]:.1f}, {command[1]:.1f}, {command[2]:.1f}]  |  t = {step * env.unwrapped.step_dt:.1f}s", font=small, fill=(225, 225, 225))
                    frame = np.asarray(image)
                writer.append_data(frame)
                frames.append(frame)
            if args.telemetry:
                forces = rear_sensor.data.net_forces_w
                forces = forces.torch if hasattr(forces, "torch") else forces
                forces = forces[0].detach().cpu().numpy()
                def np0(value):
                    value = value.torch if hasattr(value, "torch") else value
                    return value[0].detach().cpu().numpy().copy()
                telemetry_time.append(step * env.unwrapped.step_dt)
                telemetry_obs.append(policy_obs[0].detach().cpu().numpy().copy())
                telemetry_action.append(actions[0].detach().cpu().numpy().copy())
                telemetry_all_joint_pos.append(np0(robot_data.joint_pos))
                telemetry_body_pos.append(np0(robot_data.body_pos_w))
                telemetry_joint_pos.append(robot_data.joint_pos[0, rear_joint_indices].detach().cpu().numpy().copy())
                telemetry_contact.append((np.linalg.norm(forces, axis=-1) > 1.0).astype(np.uint8))
                telemetry_joint_vel.append(np0(robot_data.joint_vel))
                telemetry_joint_target.append(np0(robot_data.joint_pos_target))
                telemetry_computed_torque.append(np0(robot_data.computed_torque))
                telemetry_applied_torque.append(np0(robot_data.applied_torque))
                telemetry_root_pos.append(np0(robot_data.root_pos_w))
                telemetry_root_quat.append(np0(robot_data.root_quat_w))
                telemetry_root_lin_vel.append(np0(robot_data.root_lin_vel_w))
                telemetry_root_ang_vel.append(np0(robot_data.root_ang_vel_w))
                telemetry_projected_gravity.append(np0(robot_data.projected_gravity_b))
                telemetry_rear_foot_pos.append(np.asarray([np0(sensor.data.pos_w) for sensor in rear_foot_sensors]))
                telemetry_rear_contact_force.append(forces.copy())
                telemetry_done.append(dones.detach().cpu().numpy().copy())
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
            obs=np.asarray(telemetry_obs), action=np.asarray(telemetry_action),
            all_joint_pos=np.asarray(telemetry_all_joint_pos), body_names=np.asarray(robot_data.body_names),
            body_pos_w=np.asarray(telemetry_body_pos), joint_vel=np.asarray(telemetry_joint_vel),
            joint_pos_target=np.asarray(telemetry_joint_target), computed_torque=np.asarray(telemetry_computed_torque),
            applied_torque=np.asarray(telemetry_applied_torque), all_joint_names=np.asarray(robot_data.joint_names),
            root_pos=np.asarray(telemetry_root_pos), root_quat_xyzw=np.asarray(telemetry_root_quat),
            root_lin_vel_w=np.asarray(telemetry_root_lin_vel), root_ang_vel_w=np.asarray(telemetry_root_ang_vel),
            projected_gravity_b=np.asarray(telemetry_projected_gravity), rear_foot_pos=np.asarray(telemetry_rear_foot_pos),
            rear_contact_force_w=np.asarray(telemetry_rear_contact_force), done=np.asarray(telemetry_done),
        )
    root_delta = root_end - root_start
    print(
        f"ROOT_DELTA dx={root_delta[0]:.6f} dy={root_delta[1]:.6f} dz={root_delta[2]:.6f} "
        f"horizontal={np.linalg.norm(root_delta[:2]):.6f}",
        flush=True,
    )
    if args.actuator_excitation:
        print(
            f"ACTUATOR_EXCITATION native fixed_root=1 gravity=[0,0,0] collisions=0 "
            f"amplitude_rad={args.excitation_amplitude} rest_s={args.excitation_rest} hold_s={args.excitation_hold}",
            flush=True,
        )
    if args.upright_perturbation:
        print(
            f"UPRIGHT_PERTURBATION native fixed_root=1 gravity=[0,0,0] collisions=0 "
            f"zero_std_rad={args.perturbation_zero_std} walk_std_rad={args.perturbation_walk_std} "
            f"half_duration_s={args.perturbation_half_duration} root_lift_m={args.upright_root_lift}",
            flush=True,
        )
    if args.multibody_excitation:
        print(
            f"MULTIBODY_EXCITATION native fixed_root=0 gravity={args.multibody_gravity} collisions=0 "
            f"amplitude_rad={args.excitation_amplitude} rest_s={args.excitation_rest} hold_s={args.excitation_hold}",
            flush=True,
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
