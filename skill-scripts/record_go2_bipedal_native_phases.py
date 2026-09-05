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
parser.add_argument("--no-video", action="store_true", help="Skip frame rendering and video encoding for diagnostic sweeps.")
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
parser.add_argument("--joint-limit-alpha", type=float)
parser.add_argument("--joint-limit-recovery-speed", type=float)
parser.add_argument(
    "--joint-limit-tolerance",
    type=float,
    default=1.0e-5,
    help="Maximum accepted limit excursion in direct-torque diagnostics (rad).",
)
parser.add_argument("--joint-limit-omega", type=float)
parser.add_argument("--joint-limit-relax", type=float)
parser.add_argument("--joint-limit-regularization", type=float)
parser.add_argument("--physics-substeps", type=int, default=None, help="Override Newton substeps while preserving cfg.sim.dt; diagnostic only.")
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
parser.add_argument("--matched-actuator-initial-state", action="store_true", help="Phase-1 only: explicitly overwrite DVI q=default, qd=0 and root velocity=0 after reset, before the first control step; use with the matched MuJoCo recorder.")
parser.add_argument("--direct-torque-excitation", action="store_true", help="Pure dynamics diagnostic: bypass UnitreeActuator PD, saturation, and passive terms; apply a prescribed sequential joint-torque waveform directly.")
parser.add_argument("--direct-torque-amplitude", type=float, default=1.0, help="Per-joint torque amplitude (Nm) for --direct-torque-excitation.")
parser.add_argument("--direct-torque-free-root", action="store_true", help="Direct-torque only: leave the floating root unprojected after one-time matched initialization.")
parser.add_argument("--direct-torque-zero-input", action="store_true", help="Direct-torque only: apply zero torque throughout; asserts rest-state invariance.")
parser.add_argument("--direct-torque-settle-steps", type=int, default=0, help="Direct-torque free-root only: warm the constraint solver at zero input, then capture that consistent pose as the comparison initial state.")
parser.add_argument("--controlled-contact", choices=("off", "on"), help="Phase-3 isolated rear-foot landing with ground contact disabled or enabled.")
parser.add_argument("--controlled-contact-side", choices=("left", "right"), default="left")
parser.add_argument("--controlled-contact-downward-speed", type=float, default=0.5)
parser.add_argument("--contact-gap", type=float, help="Override Newton's per-shape contact gap in metres.")
parser.add_argument("--contact-margin", type=float, help="Override Newton's per-shape collision margin in metres.")
parser.add_argument("--require-training-dvi-config", action="store_true", help="Fail unless the loaded saved config is used unchanged for the DVI solver fields relevant to training/playback.")
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
CONTROLLED_CONTACT_ROOT = {
    "left": ((0.0, 0.0, 0.5735), (0.00079349, -0.73183698, 0.10811924, 0.67284785)),
    "right": ((0.0, 0.0, 0.5691), (0.11762024, -0.73949344, -0.02060549, 0.66248799)),
}
CONTROLLED_CONTACT_OFF_Z_OFFSET = 1.0
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
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
        # Select the saved task's matching play configuration.  In particular,
        # `Unitree-Go2-Bipedal-Walk` must stay on its infinite plane; only the
        # explicit rough task may select the rough-terrain play configuration.
        cfg_class_name = (
            "RobotBipedalWalkRoughPlayEnvCfg"
            if task == "Unitree-Go2-Bipedal-Walk-Rough"
            else "RobotBipedalWalkPlayEnvCfg"
            if task == "Unitree-Go2-Bipedal-Walk"
            else "RobotRoughPlayEnvCfg"
        )
        cfg_class = getattr(module, cfg_class_name)
        cfg = cfg_class()
        cfg = resolve_presets(cfg, {args.preset})
    else:
        cfg = load_cfg_from_registry(task, "play_env_cfg_entry_point")
        cfg = resolve_presets(cfg, {args.preset})
    if args.require_training_dvi_config and not args.saved_env_cfg:
        raise ValueError("--require-training-dvi-config requires --saved-env-cfg")
    if args.physics_substeps is not None:
        if args.physics_substeps < 1:
            raise ValueError("--physics-substeps must be positive")
        cfg.sim.physics.num_substeps = args.physics_substeps
    if args.joint_limit_iterations is not None:
        cfg.sim.physics.solver_cfg.joint_limit_max_iterations = args.joint_limit_iterations
    if args.joint_limit_alpha is not None:
        cfg.sim.physics.solver_cfg.joint_limit_alpha = args.joint_limit_alpha
    if args.joint_limit_recovery_speed is not None:
        cfg.sim.physics.solver_cfg.joint_limit_recovery_speed = args.joint_limit_recovery_speed
    if args.joint_limit_omega is not None:
        cfg.sim.physics.solver_cfg.joint_limit_omega = args.joint_limit_omega
    if args.joint_limit_relax is not None:
        cfg.sim.physics.solver_cfg.joint_limit_relax = args.joint_limit_relax
    if args.joint_limit_regularization is not None:
        cfg.sim.physics.solver_cfg.joint_limit_reg = args.joint_limit_regularization
    if args.coupling_iterations is not None:
        cfg.sim.physics.solver_cfg.coupling_iterations = args.coupling_iterations
    if args.contact_gap is not None:
        if args.contact_gap < 0.0:
            raise ValueError("--contact-gap must be nonnegative")
        cfg.sim.physics.default_shape_cfg.gap = args.contact_gap
    if args.contact_margin is not None:
        if args.contact_margin < 0.0:
            raise ValueError("--contact-margin must be nonnegative")
        cfg.sim.physics.default_shape_cfg.margin = args.contact_margin
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
    # Newton-DVI and MJWarp expose different solver configuration surfaces.
    # Keep the detailed DVI audit line, but do not dereference DVI-only fields
    # when recording the native MJWarp backend.
    if args.preset == "newton_mjwarp":
        mjwarp_fields = {
            name: getattr(solver_cfg, name)
            for name in dir(solver_cfg)
            if not name.startswith("_") and not callable(getattr(solver_cfg, name))
        }
        print(
            "EFFECTIVE_MJWARP_PARAMS "
            + " ".join(f"{name}={value}" for name, value in sorted(mjwarp_fields.items())),
            flush=True,
        )
    else:
        print(
        "EFFECTIVE_DVI_PARAMS "
        f"joint_limit_solver_type={solver_cfg.joint_limit_solver_type} "
        f"joint_limit_max_iterations={solver_cfg.joint_limit_max_iterations} "
        f"joint_limit_omega={solver_cfg.joint_limit_omega} "
        f"joint_limit_relax={solver_cfg.joint_limit_relax} "
        f"joint_limit_reg={solver_cfg.joint_limit_reg} "
        f"joint_limit_alpha={solver_cfg.joint_limit_alpha} "
        f"joint_limit_recovery_speed={solver_cfg.joint_limit_recovery_speed} "
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
    if args.joint_limit_tolerance < 0.0:
        raise ValueError("--joint-limit-tolerance must be nonnegative")
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
    if args.actuator_excitation or (args.direct_torque_excitation and not args.direct_torque_free_root) or args.upright_perturbation:
        # Phase 1 isolates the exact native actuator path from root, gravity,
        # terrain, and contact.  Preserve the training topology, then impose
        # the diagnostic fixed base by root-state projection after each step.
        cfg.sim.gravity = (0.0, 0.0, 0.0)
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        # NOTE: do NOT use `articulation_props.fix_root_link` here.  A fixed root
        # joint is itself a fixed joint, so the training-time
        # `collapse_fixed_joints=True` merges `base` into the world.  The
        # articulation then has no root body: all four hip joints become
        # parent == -1 roots, `ArticulationView` reports neither `is_fixed_base`
        # nor `is_floating_base`, and `get/set_root_transforms` silently aliases
        # `joint_X_p[joint 0]` == `FL_hip_joint`
        # (`newton/_src/utils/selection.py:1402-1425`).  IsaacLab's root-pose
        # write then overwrites the FL hip MOUNT with the root pose, displacing
        # only the FL chain and dropping `base` from body telemetry.
        #
        # Instead keep the exact training model (floating base + collapse) and
        # impose the fixed root kinematically by projecting the root state back
        # to its reference after every control step.  This is also precisely what
        # the MuJoCo Phase-1 counterpart does
        # (`run_go2_fixed_base_actuator_excitation_mujoco.py`), so the two sides
        # stay methodologically identical.
        cfg.scene.robot.spawn.articulation_props.fix_root_link = False
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    if args.direct_torque_excitation:
        # Free-root torque diagnostics still require the same zero-gravity,
        # collision-free world; only the post-step root projection is omitted.
        if args.direct_torque_free_root:
            cfg.sim.gravity = (0.0, 0.0, 0.0)
            cfg.scene.robot.spawn.rigid_props.disable_gravity = True
            cfg.scene.robot.spawn.articulation_props.fix_root_link = False
            cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
            cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        if args.actuator_excitation or args.upright_perturbation or args.multibody_excitation:
            raise ValueError("--direct-torque-excitation is mutually exclusive with the other diagnostic modes")
        if args.direct_torque_zero_input and not args.direct_torque_free_root:
            raise ValueError("--direct-torque-zero-input requires --direct-torque-free-root")
        # Feed-forward effort with every normal UnitreeActuator contribution
        # neutralized: DVI receives precisely the commanded joint torque.
        cfg.actions.JointPositionAction = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, clip={".*": (-100.0, 100.0)}
        )
        for actuator_cfg in cfg.scene.robot.actuators.values():
            actuator_cfg.stiffness = 0.0
            actuator_cfg.damping = 0.0
            actuator_cfg.friction = 0.0
            # Disable the Unitree torque-speed envelope too. Direct mode must
            # remain an exact force input even if the diagnostic drives a joint
            # beyond the Go2HV no-load speed.
            actuator_cfg.X1 = 1.0e9
            actuator_cfg.X2 = 2.0e9
            actuator_cfg.Y1 = 1.0e9
            actuator_cfg.Y2 = 1.0e9
            actuator_cfg.Fs = 0.0
            actuator_cfg.Fd = 0.0
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0
        # One environment step is one 5 ms physics sample.
        cfg.decimation = 1
    if args.controlled_contact:
        if args.controlled_contact_downward_speed <= 0.0:
            raise ValueError("--controlled-contact-downward-speed must be positive")
        cfg.sim.gravity = (0.0, 0.0, 0.0)
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        cfg.scene.robot.spawn.articulation_props.fix_root_link = False
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        if args.controlled_contact == "off":
            cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        cfg.actions.JointPositionAction = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, clip={".*": (-100.0, 100.0)}
        )
        for actuator_cfg in cfg.scene.robot.actuators.values():
            actuator_cfg.stiffness = 0.0
            actuator_cfg.damping = 0.0
            actuator_cfg.friction = 0.0
            actuator_cfg.X1 = 1.0e9
            actuator_cfg.X2 = 2.0e9
            actuator_cfg.Y1 = 1.0e9
            actuator_cfg.Y2 = 1.0e9
            actuator_cfg.Fs = 0.0
            actuator_cfg.Fd = 0.0
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0
        contact_pos, contact_quat = CONTROLLED_CONTACT_ROOT[args.controlled_contact_side]
        if args.controlled_contact == "off":
            contact_pos = (contact_pos[0], contact_pos[1], contact_pos[2] + CONTROLLED_CONTACT_OFF_Z_OFFSET)
        cfg.scene.robot.init_state.pos = contact_pos
        cfg.scene.robot.init_state.rot = contact_quat
        cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, -args.controlled_contact_downward_speed)
        cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
        cfg.scene.robot.init_state.joint_pos = {
            name: float(value) for name, value in zip(
                ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
                UPRIGHT_BIPEDAL_TARGET,
            )
        }
        cfg.decimation = 1
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
    cfg.sim.enable_newton_rendering = not args.no_video
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
    if args.controlled_contact:
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        shape_labels = list(model.shape_label)
        shape_gap = model.shape_gap.numpy()
        shape_margin = model.shape_margin.numpy()
        if args.contact_gap is not None:
            shape_gap.fill(args.contact_gap)
            model.shape_gap.assign(shape_gap)
        if args.contact_margin is not None:
            shape_margin.fill(args.contact_margin)
            model.shape_margin.assign(shape_margin)
        for shape_index, shape_label in enumerate(shape_labels):
            lower_label = shape_label.lower()
            if "ground" in lower_label or f"/r{args.controlled_contact_side[0]}_foot/" in lower_label:
                print(
                    f"CONTROLLED_CONTACT_SHAPE index={shape_index} label={shape_label!r} "
                    f"gap_m={shape_gap[shape_index]:.9g} margin_m={shape_margin[shape_index]:.9g}",
                    flush=True,
                )

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
    if not (args.actuator_excitation or args.direct_torque_excitation or args.controlled_contact or args.multibody_excitation or args.upright_perturbation):
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    visualizer = next((v for v in env.unwrapped.sim._visualizers if hasattr(v, "_viewer")), None)
    if not args.no_video and visualizer is None:
        raise RuntimeError(f"Newton visualizer was not initialized: {env.unwrapped.sim._visualizers!r}")
    viewer = visualizer._viewer if visualizer is not None else None

    # Camera is updated per frame below when --follow-robot is enabled.
    if viewer is not None:
        viewer.camera.fov = args.camera_fov

    if sum((args.zero_command, args.single_command, args.phased_commands, args.zero_neg_pos, args.actuator_excitation, args.direct_torque_excitation, bool(args.controlled_contact), args.multibody_excitation, args.upright_perturbation)) > 1:
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
    # Phase-1/1b keeps the training-time floating-base + collapsed-link model,
    # then imposes its diagnostic fixed base by restoring this state after every
    # environment step.  This matches the MuJoCo Phase-1 harness exactly.
    project_root = args.actuator_excitation or (args.direct_torque_excitation and not args.direct_torque_free_root) or args.upright_perturbation
    root_projection_state = None
    root_projection_max_pre_error = 0.0
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
    robot = env.unwrapped.scene["robot"]
    robot_data = robot.data
    # Audit imported limits before stepping. The data views are the exact DVI
    # model fields used by its joint-limit solver, in articulation-joint order.
    limit_lo = robot_data.joint_pos_limits[..., 0]
    limit_hi = robot_data.joint_pos_limits[..., 1]
    limit_lo = limit_lo.torch if hasattr(limit_lo, "torch") else limit_lo
    limit_hi = limit_hi.torch if hasattr(limit_hi, "torch") else limit_hi
    limit_lo_np, limit_hi_np = limit_lo[0].detach().cpu().numpy(), limit_hi[0].detach().cpu().numpy()
    if args.direct_torque_excitation and (not np.all(np.isfinite(limit_lo_np)) or not np.all(np.isfinite(limit_hi_np)) or np.any(limit_hi_np <= limit_lo_np)):
        raise RuntimeError(f"Invalid imported DVI joint limits: lower={limit_lo_np} upper={limit_hi_np}")
    print(f"DVI_JOINT_LIMITS names={list(robot_data.joint_names)} lower={limit_lo_np.tolist()} upper={limit_hi_np.tolist()}", flush=True)
    # A reset may contain one settling/integration interval.  For the solver
    # comparison, eliminate that hidden history explicitly: both DVI and
    # MuJoCo begin the first applied-control interval at q=default and qd=0.
    # Preserve DVI's instantiated world pose (each environment's origin), but
    # zero its velocity and use that same pose for the fixed-root projection.
    matched_initial_state_error = None
    if args.matched_actuator_initial_state:
        if not (args.actuator_excitation or args.direct_torque_excitation):
            raise ValueError("--matched-actuator-initial-state requires an actuator diagnostic")
        q0 = robot_data.default_joint_pos
        q0 = q0.torch if hasattr(q0, "torch") else q0
        q0 = q0.detach().clone()
        qd0 = torch.zeros_like(q0)
        root_pose0 = robot_data.root_link_pose_w
        root_pose0 = root_pose0.torch if hasattr(root_pose0, "torch") else root_pose0
        root_pose0 = root_pose0.detach().clone()
        root_velocity0 = torch.zeros((args.num_envs, 6), dtype=root_pose0.dtype, device=root_pose0.device)
        robot.write_joint_state_to_sim_index(position=q0, velocity=qd0)
        robot.write_root_link_pose_to_sim_index(root_pose=root_pose0)
        robot.write_root_link_velocity_to_sim_index(root_velocity=root_velocity0)
        q_check = robot_data.joint_pos
        q_check = q_check.torch if hasattr(q_check, "torch") else q_check
        qd_check = robot_data.joint_vel
        qd_check = qd_check.torch if hasattr(qd_check, "torch") else qd_check
        matched_initial_state_error = max(
            float(torch.max(torch.abs(q_check - q0)).item()),
            float(torch.max(torch.abs(qd_check - qd0)).item()),
        )
        if matched_initial_state_error > 1.0e-7:
            raise RuntimeError(f"Matched actuator initialization failed: max state error {matched_initial_state_error:.6g}")
    settled_initial_state_error = None
    if args.direct_torque_settle_steps:
        if not (args.direct_torque_excitation and args.direct_torque_free_root):
            raise ValueError("--direct-torque-settle-steps requires --direct-torque-excitation --direct-torque-free-root")
        if args.direct_torque_settle_steps < 1:
            raise ValueError("--direct-torque-settle-steps must be positive")
        zero_actions = torch.zeros((args.num_envs, len(robot_data.joint_names)), dtype=torch.float32, device=env.unwrapped.device)
        # Let the DVI bilateral constraints remove only their initial numerical
        # residual. This is not part of the logged experiment.
        for _ in range(args.direct_torque_settle_steps):
            obs, _, _, _ = env.step(zero_actions)
        q_settle = robot_data.joint_pos
        q_settle = q_settle.torch if hasattr(q_settle, "torch") else q_settle
        root_settle = robot_data.root_link_pose_w
        root_settle = root_settle.torch if hasattr(root_settle, "torch") else root_settle
        robot.write_joint_state_to_sim_index(position=q_settle.detach().clone(), velocity=torch.zeros_like(q_settle))
        robot.write_root_link_pose_to_sim_index(root_pose=root_settle.detach().clone())
        robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((args.num_envs, 6), dtype=root_settle.dtype, device=root_settle.device))
        q_post = robot_data.joint_pos; q_post = q_post.torch if hasattr(q_post, "torch") else q_post
        qd_post = robot_data.joint_vel; qd_post = qd_post.torch if hasattr(qd_post, "torch") else qd_post
        settled_initial_state_error = max(float(torch.max(torch.abs(q_post-q_settle)).item()), float(torch.max(torch.abs(qd_post)).item()))
        if settled_initial_state_error > 1e-7:
            raise RuntimeError(f"Settled-state initialization failed: {settled_initial_state_error:.6g}")
        print(f"DVI_SETTLED_INITIALIZATION warmup_steps={args.direct_torque_settle_steps} state_error={settled_initial_state_error:.6g}", flush=True)
    if project_root:
        root_pose = robot_data.root_link_pose_w
        root_pose = root_pose.torch if hasattr(root_pose, "torch") else root_pose
        root_projection_state = torch.cat(
            (root_pose.detach().clone(), torch.zeros((args.num_envs, 6), dtype=root_pose.dtype, device=root_pose.device)),
            dim=-1,
        )
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
    writer = None if args.no_video else imageio.get_writer(
        str(args.output), fps=args.fps, codec="libx264", pixelformat="yuv420p", quality=8
    )
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
                if args.controlled_contact:
                    actions = torch.zeros(
                        (args.num_envs, len(robot_data.joint_names)),
                        dtype=torch.float32,
                        device=env.unwrapped.device,
                    )
                elif args.actuator_excitation or args.direct_torque_excitation or args.multibody_excitation:
                    # Raw action is target offset / 0.25, matching the frozen
                    # JointPositionAction configuration.  Each joint receives
                    # rest → +step → -step → rest while every other target is
                    # held at its exact default.
                    num_joints = len(robot_data.joint_names)
                    joint_index = min(step // excitation_segment_steps, args.excitation_joints - 1)
                    local = step % excitation_segment_steps
                    n_rest = max(1, round(args.excitation_rest / env.unwrapped.step_dt))
                    n_hold = max(1, round(args.excitation_hold / env.unwrapped.step_dt))
                    amplitude = args.direct_torque_amplitude if args.direct_torque_excitation else args.excitation_amplitude
                    target_offset = 0.0 if (args.direct_torque_zero_input or local < n_rest or local >= n_rest + 2 * n_hold) else (amplitude if local < n_rest + n_hold else -amplitude)
                    actions = torch.zeros((args.num_envs, num_joints), dtype=torch.float32, device=env.unwrapped.device)
                    actions[:, joint_index] = target_offset if args.direct_torque_excitation else target_offset / 0.25
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
                if args.actuator_excitation or args.direct_torque_excitation or args.controlled_contact or args.multibody_excitation or args.upright_perturbation:
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
                if project_root:
                    current_pose = robot_data.root_link_pose_w
                    current_pose = current_pose.torch if hasattr(current_pose, "torch") else current_pose
                    root_projection_max_pre_error = max(
                        root_projection_max_pre_error,
                        float(torch.max(torch.abs(current_pose - root_projection_state[:, :7])).item()),
                    )
                    robot.write_root_link_pose_to_sim_index(root_pose=root_projection_state[:, :7])
                    robot.write_root_link_velocity_to_sim_index(root_velocity=root_projection_state[:, 7:])
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
            if not args.no_video and step % frame_stride == 0:
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
        if writer is not None:
            writer.close()
        env.close()
    if project_root:
        final_pose = robot_data.root_link_pose_w
        final_pose = final_pose.torch if hasattr(final_pose, "torch") else final_pose
        root_projection_post_error = float(
            torch.max(torch.abs(final_pose - root_projection_state[:, :7])).item()
        )
        if root_projection_post_error > 1.0e-5:
            raise RuntimeError(
                f"Phase-1 root projection post-write error exceeds tolerance: {root_projection_post_error:.6g}"
            )
    else:
        root_projection_post_error = None
    if args.telemetry:
        root_pos_array = np.asarray(telemetry_root_pos)
        body_pos_array = np.asarray(telemetry_body_pos)
        rear_foot_pos_array = np.asarray(telemetry_rear_foot_pos)
        if args.controlled_contact == "off":
            root_pos_array[..., 2] -= CONTROLLED_CONTACT_OFF_Z_OFFSET
            body_pos_array[..., 2] -= CONTROLLED_CONTACT_OFF_Z_OFFSET
            rear_foot_pos_array[..., 2] -= CONTROLLED_CONTACT_OFF_Z_OFFSET
        args.telemetry.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.telemetry,
            time=np.asarray(telemetry_time), joint_pos=np.asarray(telemetry_joint_pos),
            contact=np.asarray(telemetry_contact), joint_names=np.asarray(rear_joint_names),
            contact_names=np.asarray(getattr(rear_sensor, "body_names", ["RL_foot", "RR_foot"])),
            obs=np.asarray(telemetry_obs), action=np.asarray(telemetry_action),
            all_joint_pos=np.asarray(telemetry_all_joint_pos), body_names=np.asarray(robot_data.body_names),
            body_pos_w=body_pos_array, joint_vel=np.asarray(telemetry_joint_vel),
            joint_pos_target=np.asarray(telemetry_joint_target), computed_torque=np.asarray(telemetry_computed_torque),
            applied_torque=np.asarray(telemetry_applied_torque), all_joint_names=np.asarray(robot_data.joint_names),
            root_pos=root_pos_array, root_quat_xyzw=np.asarray(telemetry_root_quat),
            root_lin_vel_w=np.asarray(telemetry_root_lin_vel), root_ang_vel_w=np.asarray(telemetry_root_ang_vel),
            projected_gravity_b=np.asarray(telemetry_projected_gravity), rear_foot_pos=rear_foot_pos_array,
            rear_contact_force_w=np.asarray(telemetry_rear_contact_force), done=np.asarray(telemetry_done),
            root_projection_enabled=np.asarray(project_root),
            matched_actuator_initial_state=np.asarray(args.matched_actuator_initial_state),
            matched_actuator_initial_state_error=np.asarray(matched_initial_state_error),
            root_projection_max_pre_error=np.asarray(root_projection_max_pre_error),
            root_projection_post_error=np.asarray(root_projection_post_error),
            joint_limit_lower=limit_lo_np, joint_limit_upper=limit_hi_np,
        )
    root_delta = root_end - root_start
    print(
        f"ROOT_DELTA dx={root_delta[0]:.6f} dy={root_delta[1]:.6f} dz={root_delta[2]:.6f} "
        f"horizontal={np.linalg.norm(root_delta[:2]):.6f}",
        flush=True,
    )
    if project_root:
        root_delta_inf = float(np.max(np.abs(root_delta)))
        print(
            f"ROOT_PROJECTION mode=kinematic training_topology=1 "
            f"max_pre_projection_state_error={root_projection_max_pre_error:.6g} "
            f"post_projection_state_error={root_projection_post_error:.6g} "
            f"post_projection_root_delta_inf={root_delta_inf:.6g}",
            flush=True,
        )
        if root_delta_inf > 1.0e-5:
            raise RuntimeError(f"Phase-1 root projection drift exceeds tolerance: {root_delta_inf:.6g}")
    if args.actuator_excitation:
        print(
            f"ACTUATOR_EXCITATION native fixed_root=1 gravity=[0,0,0] collisions=0 "
            f"matched_initial_state={int(args.matched_actuator_initial_state)} "
            f"initial_state_error={matched_initial_state_error} "
            f"amplitude_rad={args.excitation_amplitude} rest_s={args.excitation_rest} hold_s={args.excitation_hold}",
            flush=True,
        )
    if args.direct_torque_excitation:
        applied = np.asarray(telemetry_applied_torque)
        command = np.asarray(telemetry_action)
        torque_error = float(np.max(np.abs(applied - command)))
        print(
            f"DIRECT_TORQUE_EXCITATION native root_projection={int(project_root)} gravity=[0,0,0] collisions=0 "
            f"physics_dt_s={env.unwrapped.step_dt} amplitude_Nm={args.direct_torque_amplitude} "
            f"max_applied_minus_command_Nm={torque_error:.6g}", flush=True,
        )
        if torque_error > 1.0e-5:
            raise RuntimeError(f"Direct effort was altered by the DVI actuator path: {torque_error:.6g} Nm")
        q_trace = np.asarray(telemetry_all_joint_pos)
        qd_trace = np.asarray(telemetry_joint_vel)
        limit_violation = float(max(np.max(q_trace - limit_hi_np), np.max(limit_lo_np - q_trace), 0.0))
        print(
            f"DVI_DIRECT_LIMIT_GATE max_violation_rad={limit_violation:.6g} "
            f"tolerance_rad={args.joint_limit_tolerance:.6g}",
            flush=True,
        )
        if limit_violation > args.joint_limit_tolerance:
            raise RuntimeError(
                f"DVI joint-limit gate failed: max violation {limit_violation:.6g} rad "
                f"> tolerance {args.joint_limit_tolerance:.6g} rad"
            )
        if args.direct_torque_zero_input:
            drift_q = float(np.max(np.abs(q_trace - q_trace[0])))
            drift_qd = float(np.max(np.abs(qd_trace)))
            print(f"DVI_ZERO_INPUT_GATE max_joint_displacement_rad={drift_q:.6g} max_joint_velocity_radps={drift_qd:.6g}", flush=True)
            if drift_q > 1.0e-6 or drift_qd > 1.0e-6:
                raise RuntimeError("DVI free-root zero-input rest gate failed")
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
    if args.controlled_contact:
        contact_trace = np.asarray(telemetry_contact)
        selected_index = 0 if args.controlled_contact_side == "left" else 1
        selected_contact = contact_trace[:, selected_index]
        first_contact = np.flatnonzero(selected_contact)
        print(
            f"CONTROLLED_CONTACT native mode={args.controlled_contact} side={args.controlled_contact_side} "
            f"gravity=off self_collisions=0 downward_speed_mps={args.controlled_contact_downward_speed} "
            f"shape_gap_m={cfg.sim.physics.default_shape_cfg.gap} "
            f"shape_margin_m={cfg.sim.physics.default_shape_cfg.margin} "
            f"first_contact_step={None if first_contact.size == 0 else int(first_contact[0])}",
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
