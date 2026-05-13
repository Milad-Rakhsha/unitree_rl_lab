"""Scenario-based reset functions for Go2 fall-recovery training.

Provides :func:`reset_fall_scenarios` — a structured reset that samples from
distinct fall categories so the policy gets practice on the specific states
it will encounter in deployment:

* **Bipedal forward fall** — tipping nose-down from a rear-stance
* **Bipedal backward fall** — tipping over backward from a rear-stance
* **Bipedal sideways fall** — rolling sideways from a rear-stance
* **Quadruped stumble** — large perturbation from a four-legged stance
* **Uniform random** — the original broad-coverage randomization

Each scenario sets a physically meaningful (root_pose, root_vel, joint_pos,
joint_vel) tuple that represents a *mid-fall* initial condition, not a
calm standing pose.
"""

from __future__ import annotations

import math
import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    sample_uniform,
)

_tt = wp.to_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Go2 joint reference poses
# ---------------------------------------------------------------------------
# Joint order (Isaac convention, 12 DOF):
#   FR_hip, FR_thigh, FR_calf,
#   FL_hip, FL_thigh, FL_calf,
#   RR_hip, RR_thigh, RR_calf,
#   RL_hip, RL_thigh, RL_calf
#
# Quadruped standing (from UNITREE_GO2_CFG.init_state):
#   hip: ∓0.1, F_thigh: 0.8, R_thigh: 1.0, calf: -1.5
#
# Bipedal rear-stance (robot pitched ~+90° about body-y, resting on rear feet):
#   Front legs tucked (hip ∓0.1, thigh ~0.8, calf ~-1.5)  — in the air
#   Rear legs extended (hip ∓0.1, thigh ~0.7, calf ~-1.0)  — supporting

QUADRUPED_JOINT_POS = [
    -0.1, 0.8, -1.5,   # FR
     0.1, 0.8, -1.5,   # FL
    -0.1, 1.0, -1.5,   # RR
     0.1, 1.0, -1.5,   # RL
]

BIPEDAL_REAR_JOINT_POS = [
    -0.1, 0.8, -1.5,   # FR (tucked, in the air)
     0.1, 0.8, -1.5,   # FL (tucked, in the air)
    -0.1, 0.7, -1.0,   # RR (extended, supporting)
     0.1, 0.7, -1.0,   # RL (extended, supporting)
]


def reset_fall_scenarios(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    # Scenario weights (must sum to 1.0)
    bipedal_forward_frac: float = 0.25,
    bipedal_backward_frac: float = 0.25,
    bipedal_sideways_frac: float = 0.20,
    quadruped_stumble_frac: float = 0.20,
    # remaining fraction → uniform random
    # Joint noise
    joint_pos_noise: float = 0.15,
    joint_vel_noise: float = 3.0,
    # Bipedal spawn height (world z when standing on rear legs)
    bipedal_base_z: float = 0.55,
    # Quadruped spawn height
    quadruped_base_z: float = 0.34,
    # Asset
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset envs to structured fall scenarios.

    Called as an Isaac Lab ``EventTerm`` function with ``mode="reset"``.
    Replaces both ``reset_base`` and ``reset_robot_joints`` — it sets
    the full (root_pose, root_vel, joint_pos, joint_vel) state.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    # Pre-allocate output tensors
    root_pos = torch.zeros(n, 3, device=device)
    root_quat = torch.zeros(n, 4, device=device)
    root_lin_vel = torch.zeros(n, 3, device=device)
    root_ang_vel = torch.zeros(n, 3, device=device)
    joint_pos = torch.zeros(n, asset.num_joints, device=device)
    joint_vel = torch.zeros(n, asset.num_joints, device=device)

    # Reference joint poses as tensors
    quad_ref = torch.tensor(QUADRUPED_JOINT_POS, device=device, dtype=torch.float32)
    biped_ref = torch.tensor(BIPEDAL_REAR_JOINT_POS, device=device, dtype=torch.float32)

    # Assign each env to a scenario
    fracs = [
        bipedal_forward_frac,
        bipedal_backward_frac,
        bipedal_sideways_frac,
        quadruped_stumble_frac,
    ]
    uniform_frac = max(0.0, 1.0 - sum(fracs))
    fracs.append(uniform_frac)
    cumulative = torch.tensor(fracs, device=device).cumsum(0)
    draws = torch.rand(n, device=device)
    scenario = torch.searchsorted(cumulative, draws).clamp(max=len(fracs) - 1)

    # Helper: random uniform tensor
    def _ru(lo, hi, shape):
        return sample_uniform(lo, hi, shape, device)

    # --- Scenario 0: Bipedal forward fall ---
    # Robot is in rear-stance (pitch ~+80-90°), tipping FORWARD (nose down).
    # Forward pitch velocity, slight drop.
    mask = scenario == 0
    cnt = mask.sum().item()
    if cnt > 0:
        idx = mask.nonzero(as_tuple=True)[0]
        root_pos[idx, 0] = _ru(-0.5, 0.5, (cnt,))  # x
        root_pos[idx, 1] = _ru(-0.5, 0.5, (cnt,))  # y
        root_pos[idx, 2] = bipedal_base_z + _ru(-0.15, 0.05, (cnt,))  # z — slightly below nominal
        # Pitch: 60-90° (tilted forward from bipedal, i.e. falling toward quadruped)
        roll = _ru(-0.2, 0.2, (cnt,))
        pitch = _ru(math.radians(40), math.radians(85), (cnt,))  # partially fallen forward
        yaw = _ru(-math.pi, math.pi, (cnt,))
        root_quat[idx] = quat_from_euler_xyz(roll, pitch, yaw)
        # Velocity: forward pitch rate (tipping toward ground)
        root_lin_vel[idx, 0] = _ru(-1.0, 1.0, (cnt,))
        root_lin_vel[idx, 1] = _ru(-0.5, 0.5, (cnt,))
        root_lin_vel[idx, 2] = _ru(-2.0, 0.0, (cnt,))  # falling
        root_ang_vel[idx, 0] = _ru(-1.0, 1.0, (cnt,))
        root_ang_vel[idx, 1] = _ru(-4.0, -1.0, (cnt,))  # negative pitch rate = tipping forward
        root_ang_vel[idx, 2] = _ru(-1.0, 1.0, (cnt,))
        # Joints: start from bipedal ref + noise
        joint_pos[idx] = biped_ref.unsqueeze(0) + _ru(-joint_pos_noise, joint_pos_noise, (cnt, asset.num_joints))
        joint_vel[idx] = _ru(-joint_vel_noise, joint_vel_noise, (cnt, asset.num_joints))

    # --- Scenario 1: Bipedal backward fall ---
    # Robot is in rear-stance, tipping BACKWARD (falling over its rear legs).
    # This is the hard case — the robot needs to catch itself going over backward.
    mask = scenario == 1
    cnt = mask.sum().item()
    if cnt > 0:
        idx = mask.nonzero(as_tuple=True)[0]
        root_pos[idx, 0] = _ru(-0.5, 0.5, (cnt,))
        root_pos[idx, 1] = _ru(-0.5, 0.5, (cnt,))
        root_pos[idx, 2] = bipedal_base_z + _ru(-0.15, 0.10, (cnt,))
        # Pitch: 95-140° (past vertical, tipping backward)
        roll = _ru(-0.2, 0.2, (cnt,))
        pitch = _ru(math.radians(95), math.radians(145), (cnt,))  # past bipedal, falling backward
        yaw = _ru(-math.pi, math.pi, (cnt,))
        root_quat[idx] = quat_from_euler_xyz(roll, pitch, yaw)
        # Velocity: positive pitch rate (continuing backward)
        root_lin_vel[idx, 0] = _ru(-1.0, 1.0, (cnt,))
        root_lin_vel[idx, 1] = _ru(-0.5, 0.5, (cnt,))
        root_lin_vel[idx, 2] = _ru(-2.0, 0.5, (cnt,))
        root_ang_vel[idx, 0] = _ru(-1.0, 1.0, (cnt,))
        root_ang_vel[idx, 1] = _ru(1.0, 5.0, (cnt,))  # positive pitch rate = tipping backward
        root_ang_vel[idx, 2] = _ru(-1.0, 1.0, (cnt,))
        # Joints: bipedal ref + noise
        joint_pos[idx] = biped_ref.unsqueeze(0) + _ru(-joint_pos_noise, joint_pos_noise, (cnt, asset.num_joints))
        joint_vel[idx] = _ru(-joint_vel_noise, joint_vel_noise, (cnt, asset.num_joints))

    # --- Scenario 2: Bipedal sideways fall ---
    # Robot is in rear-stance, rolling sideways.
    mask = scenario == 2
    cnt = mask.sum().item()
    if cnt > 0:
        idx = mask.nonzero(as_tuple=True)[0]
        root_pos[idx, 0] = _ru(-0.5, 0.5, (cnt,))
        root_pos[idx, 1] = _ru(-0.5, 0.5, (cnt,))
        root_pos[idx, 2] = bipedal_base_z + _ru(-0.15, 0.05, (cnt,))
        # Pitch near bipedal (~80-100°), but with significant roll (±30-70°)
        roll = _ru(math.radians(25), math.radians(70), (cnt,))
        # Randomize roll sign
        roll_sign = (torch.rand(cnt, device=device) > 0.5).float() * 2.0 - 1.0
        roll = roll * roll_sign
        pitch = _ru(math.radians(60), math.radians(100), (cnt,))
        yaw = _ru(-math.pi, math.pi, (cnt,))
        root_quat[idx] = quat_from_euler_xyz(roll, pitch, yaw)
        # Velocity: large roll rate
        root_lin_vel[idx, 0] = _ru(-1.0, 1.0, (cnt,))
        root_lin_vel[idx, 1] = _ru(-2.0, 2.0, (cnt,))  # lateral
        root_lin_vel[idx, 2] = _ru(-1.5, 0.0, (cnt,))
        root_ang_vel[idx, 0] = _ru(2.0, 5.0, (cnt,)) * roll_sign  # roll in the fall direction
        root_ang_vel[idx, 1] = _ru(-1.5, 1.5, (cnt,))
        root_ang_vel[idx, 2] = _ru(-1.5, 1.5, (cnt,))
        # Joints: bipedal ref + noise
        joint_pos[idx] = biped_ref.unsqueeze(0) + _ru(-joint_pos_noise, joint_pos_noise, (cnt, asset.num_joints))
        joint_vel[idx] = _ru(-joint_vel_noise, joint_vel_noise, (cnt, asset.num_joints))

    # --- Scenario 3: Quadruped stumble ---
    # Robot is near four-legged stance, gets a big perturbation (tripped, shoved).
    mask = scenario == 3
    cnt = mask.sum().item()
    if cnt > 0:
        idx = mask.nonzero(as_tuple=True)[0]
        root_pos[idx, 0] = _ru(-0.5, 0.5, (cnt,))
        root_pos[idx, 1] = _ru(-0.5, 0.5, (cnt,))
        root_pos[idx, 2] = quadruped_base_z + _ru(-0.05, 0.15, (cnt,))
        # Near-level orientation with moderate tilt (±25-55°) in any direction
        roll = _ru(math.radians(-55), math.radians(55), (cnt,))
        pitch = _ru(math.radians(-55), math.radians(55), (cnt,))
        yaw = _ru(-math.pi, math.pi, (cnt,))
        root_quat[idx] = quat_from_euler_xyz(roll, pitch, yaw)
        # Large linear and angular velocity (shoved hard)
        root_lin_vel[idx, 0] = _ru(-3.0, 3.0, (cnt,))
        root_lin_vel[idx, 1] = _ru(-3.0, 3.0, (cnt,))
        root_lin_vel[idx, 2] = _ru(-2.0, 1.0, (cnt,))
        root_ang_vel[idx, 0] = _ru(-5.0, 5.0, (cnt,))
        root_ang_vel[idx, 1] = _ru(-5.0, 5.0, (cnt,))
        root_ang_vel[idx, 2] = _ru(-5.0, 5.0, (cnt,))
        # Joints: quadruped ref + noise
        joint_pos[idx] = quad_ref.unsqueeze(0) + _ru(-joint_pos_noise, joint_pos_noise, (cnt, asset.num_joints))
        joint_vel[idx] = _ru(-joint_vel_noise, joint_vel_noise, (cnt, asset.num_joints))

    # --- Scenario 4: Uniform random (coverage) ---
    # Broad sampling similar to the original stabilization env.
    mask = scenario == 4
    cnt = mask.sum().item()
    if cnt > 0:
        idx = mask.nonzero(as_tuple=True)[0]
        root_pos[idx, 0] = _ru(-0.5, 0.5, (cnt,))
        root_pos[idx, 1] = _ru(-0.5, 0.5, (cnt,))
        root_pos[idx, 2] = _ru(0.25, 0.60, (cnt,))
        roll = _ru(-1.2, 1.2, (cnt,))
        pitch = _ru(-1.2, 1.2, (cnt,))
        yaw = _ru(-math.pi, math.pi, (cnt,))
        root_quat[idx] = quat_from_euler_xyz(roll, pitch, yaw)
        root_lin_vel[idx, 0] = _ru(-2.5, 2.5, (cnt,))
        root_lin_vel[idx, 1] = _ru(-2.5, 2.5, (cnt,))
        root_lin_vel[idx, 2] = _ru(-1.5, 1.5, (cnt,))
        root_ang_vel[idx, 0] = _ru(-4.0, 4.0, (cnt,))
        root_ang_vel[idx, 1] = _ru(-4.0, 4.0, (cnt,))
        root_ang_vel[idx, 2] = _ru(-4.0, 4.0, (cnt,))
        # Randomly choose between quadruped and bipedal joint refs
        use_biped = (torch.rand(cnt, device=device) > 0.5).unsqueeze(1)
        ref = torch.where(use_biped, biped_ref.unsqueeze(0), quad_ref.unsqueeze(0))
        joint_pos[idx] = ref + _ru(-0.25, 0.25, (cnt, asset.num_joints))
        joint_vel[idx] = _ru(-5.0, 5.0, (cnt, asset.num_joints))

    # Clamp joints to soft limits
    joint_pos_limits = _tt(asset.data.soft_joint_pos_limits)[env_ids]
    joint_pos = joint_pos.clamp(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = _tt(asset.data.soft_joint_vel_limits)[env_ids]
    joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

    # Build root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
    # Add the terrain origin offset for each env
    default_root_state = _tt(asset.data.default_root_state)[env_ids].clone()
    root_pos[:, :2] += default_root_state[:, :2]  # offset by env origin

    root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)

    # Write to sim
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=slice(None), env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=slice(None), env_ids=env_ids)
