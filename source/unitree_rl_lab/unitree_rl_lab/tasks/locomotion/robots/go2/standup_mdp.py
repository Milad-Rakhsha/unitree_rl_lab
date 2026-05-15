"""MDP helpers for Go2 bipedal stand-up environment.

Provides:
* :func:`reset_quadruped_prone` — reset to a flat quadruped stance
  (the starting pose for stand-up). Includes small randomization of
  position, orientation, and joint offsets for robustness.
* :func:`standing_success` — termination that fires when the robot has
  reached the target bipedal stance and held it calmly for a configurable
  number of consecutive steps.
* :func:`orientation_align_velocity_gated` — orientation alignment reward
  that is gated by body velocity. The robot only gets credit for being
  more upright when it is moving gently.
* :func:`rear_foot_contact_reward` — reward rear foot ground contact once
  the robot is mostly upright, encouraging feet-on-ground terminal pose.
* :func:`rear_calf_contact_penalty` — penalize rear calf ground contact
  once upright, discouraging calf-balancing in the terminal pose.
"""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


_tt = wp.to_torch

# ---------------------------------------------------------------------------
# Go2 joint reference poses (see stabilization_mdp.py for conventions)
# ---------------------------------------------------------------------------

QUADRUPED_JOINT_POS = [
    -0.1, 0.8, -1.5,   # FR
     0.1, 0.8, -1.5,   # FL
    -0.1, 1.0, -1.5,   # RR
     0.1, 1.0, -1.5,   # RL
]


# ---------------------------------------------------------------------------
# Reset: flat quadruped prone start
# ---------------------------------------------------------------------------

def reset_quadruped_prone(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    # Position randomization
    xy_range: float = 0.3,
    # Orientation randomization (small tilt to simulate imperfect ground)
    roll_range: tuple[float, float] = (-0.08, 0.08),
    pitch_range: tuple[float, float] = (-0.08, 0.08),
    # Joint noise
    joint_pos_noise: float = 0.05,
    joint_vel_noise: float = 0.2,
    # Spawn height (quadruped standing on 4 legs)
    base_z: float = 0.34,
    # Asset
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset envs to a flat quadruped stance — the starting pose for stand-up.

    The robot spawns in a calm four-legged standing pose with small
    randomization on position, yaw, tilt, and joint offsets. Velocity
    is near-zero — the episode starts from rest.

    Called as an Isaac Lab ``EventTerm`` with ``mode="reset"``.
    Replaces both ``reset_base`` and ``reset_robot_joints``.
    """
    from isaaclab.utils.math import quat_from_euler_xyz

    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    def _ru(lo, hi, shape):
        return sample_uniform(lo, hi, shape, device)

    # --- Root pose ---
    root_pos = torch.zeros(n, 3, device=device)
    root_pos[:, 0] = _ru(-xy_range, xy_range, (n,))
    root_pos[:, 1] = _ru(-xy_range, xy_range, (n,))
    root_pos[:, 2] = base_z + _ru(-0.02, 0.02, (n,))

    roll = _ru(roll_range[0], roll_range[1], (n,))
    pitch = _ru(pitch_range[0], pitch_range[1], (n,))
    yaw = _ru(-3.14159, 3.14159, (n,))
    root_quat = quat_from_euler_xyz(roll, pitch, yaw)

    # --- Root velocity (near-zero) ---
    root_lin_vel = torch.zeros(n, 3, device=device)
    root_lin_vel[:, 0] = _ru(-0.1, 0.1, (n,))
    root_lin_vel[:, 1] = _ru(-0.1, 0.1, (n,))
    root_ang_vel = torch.zeros(n, 3, device=device)

    # --- Joint state ---
    quad_ref = torch.tensor(QUADRUPED_JOINT_POS, device=device, dtype=torch.float32)
    joint_pos = quad_ref.unsqueeze(0).expand(n, -1) + _ru(
        -joint_pos_noise, joint_pos_noise, (n, asset.num_joints)
    )
    joint_vel = _ru(-joint_vel_noise, joint_vel_noise, (n, asset.num_joints))

    # Clamp to soft limits
    joint_pos_limits = _tt(asset.data.soft_joint_pos_limits)[env_ids]
    joint_pos = joint_pos.clamp(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = _tt(asset.data.soft_joint_vel_limits)[env_ids]
    joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

    # Offset by env origin
    default_root_state = _tt(asset.data.default_root_state)[env_ids].clone()
    root_pos[:, :2] += default_root_state[:, :2]

    root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_joint_position_to_sim_index(
        position=joint_pos, joint_ids=slice(None), env_ids=env_ids
    )
    asset.write_joint_velocity_to_sim_index(
        velocity=joint_vel, joint_ids=slice(None), env_ids=env_ids
    )


# ---------------------------------------------------------------------------
# Reward: velocity-gated orientation alignment
# ---------------------------------------------------------------------------


def orientation_align_velocity_gated(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    ang_vel_threshold: float = 1.5,
    lin_vel_threshold: float = 0.8,
    ang_vel_std: float = 1.0,
    lin_vel_std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Orientation alignment reward gated by body velocity — encourages slow stand-up.

    The raw orientation reward is ``max(0, dot(projected_gravity_b, desired_gravity))``
    (same monotonic signal as :func:`bipedal_mdp.orientation_align`), multiplied
    by a soft velocity gate:

    ::

        gate = exp(-max(0, |w| - t_w)^2 / s_w^2) * exp(-max(0, |v| - t_v)^2 / s_v^2)

    The gate is ~1.0 when angular and linear velocities are below the
    thresholds, and decays smoothly toward 0 as velocities increase.
    The ``max(0, ...)`` dead-zone means that velocities below the
    threshold have zero penalty — the robot isn't penalized for the
    small motions inherent in standing up, only for fast/violent ones.

    **Effect on learning:** the policy can only collect orientation reward
    (the dominant positive signal) when it is moving gently. A fast
    throw-to-vertical trajectory earns almost no orientation reward
    because the gate suppresses it during the high-velocity phase.
    The policy therefore learns to stand up slowly.

    Args:
        env: Environment instance.
        desired_gravity: Gravity direction in the target-pose body frame.
        ang_vel_threshold: Angular velocity dead-zone [rad/s]. Below this,
            the gate is 1.0. Default 1.5 rad/s.
        lin_vel_threshold: Linear velocity dead-zone [m/s]. Below this,
            the gate is 1.0. Default 0.8 m/s.
        ang_vel_std: Width of the angular velocity Gaussian gate beyond
            the threshold. Smaller = stricter. Default 1.0.
        lin_vel_std: Width of the linear velocity Gaussian gate beyond
            the threshold. Smaller = stricter. Default 0.5.
        asset_cfg: Robot asset configuration.

    Returns:
        Per-env reward in ``[0, 1]``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    # --- Orientation alignment (monotonic, same as bipedal_mdp) ---
    dot = torch.sum(_tt(asset.data.projected_gravity_b) * target, dim=-1)
    orient_reward = torch.clamp(dot, min=0.0)

    # --- Velocity gate ---
    # Angular velocity (world frame)
    ang_vel_norm = torch.linalg.norm(
        _tt(asset.data.root_ang_vel_w), dim=-1
    )
    # Linear velocity (world frame)
    lin_vel_norm = torch.linalg.norm(
        _tt(asset.data.root_lin_vel_w), dim=-1
    )

    # Soft Gaussian gate with dead-zone: 1.0 below threshold, decays above
    ang_excess = torch.clamp(ang_vel_norm - ang_vel_threshold, min=0.0)
    lin_excess = torch.clamp(lin_vel_norm - lin_vel_threshold, min=0.0)

    ang_gate = torch.exp(-(ang_excess**2) / (ang_vel_std**2))
    lin_gate = torch.exp(-(lin_excess**2) / (lin_vel_std**2))

    gate = ang_gate * lin_gate

    return orient_reward * gate


# ---------------------------------------------------------------------------
# Termination: standing success (positive — episode completed!)
# ---------------------------------------------------------------------------

def standing_success(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    min_cos_angle: float = 0.90,
    min_base_height: float = 0.45,
    max_ang_vel: float = 1.0,
    hold_steps: int = 50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate (success) when the robot reaches bipedal stance and holds it.

    Conditions checked every step:
    1. Orientation: ``cos(angle_to_target) >= min_cos_angle``
    2. Base height: ``root_pos_w.z >= min_base_height``
    3. Calm: ``||ang_vel_w|| < max_ang_vel``

    All three must be true for ``hold_steps`` consecutive steps.

    The step counter is stored as ``env._standup_hold_counter`` and reset
    to zero whenever any condition fails or an env resets.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    # Condition 1: orientation alignment
    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    orient_ok = cos_angle >= min_cos_angle

    # Condition 2: base height
    height_ok = _tt(asset.data.root_pos_w)[:, 2] >= min_base_height

    # Condition 3: calm angular velocity
    ang_vel_norm = torch.linalg.norm(
        _tt(asset.data.root_ang_vel_w), dim=-1
    )
    calm_ok = ang_vel_norm < max_ang_vel

    all_ok = orient_ok & height_ok & calm_ok

    # Manage hold counter
    if not hasattr(env, "_standup_hold_counter"):
        env._standup_hold_counter = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    counter = env._standup_hold_counter

    # Reset counter for envs that were just reset (episode_length_buf == 0
    # means the env was reset this step — the counter must not carry over
    # from the previous episode).
    just_reset = env.episode_length_buf == 0
    counter[just_reset] = 0

    # Increment where conditions met, reset where not
    counter[all_ok] += 1
    counter[~all_ok] = 0

    return counter >= hold_steps


# ---------------------------------------------------------------------------
# Reward: rear foot contact (encourage feet-on-ground terminal pose)
# ---------------------------------------------------------------------------

def rear_foot_contact_reward(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    onset_cos: float = 0.3,
    full_cos: float = 0.85,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward rear foot ground contact, soft-scaled by orientation.

    Uses a smooth linear ramp between ``onset_cos`` (where the reward
    starts) and ``full_cos`` (where it reaches full strength). This gives
    the policy a continuous gradient to learn foot placement as it tilts
    upright, avoiding the hard-gate problem where the policy fears
    crossing a threshold.

    Returns in [0, 1]: orientation_scale × (fraction of rear feet in contact).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    # Soft orientation scale: 0 below onset_cos, linear ramp, 1 above full_cos
    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    orient_scale = ((cos_angle - onset_cos) / (full_cos - onset_cos)).clamp(0.0, 1.0)

    # Contact detection
    net_forces = _tt(contact_sensor.data.net_forces_w)
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    foot_forces = net_forces[:, body_ids, :]
    force_mag = torch.linalg.norm(foot_forces, dim=-1)
    in_contact = (force_mag > 1.0).float()
    contact_score = in_contact.mean(dim=-1)  # 0, 0.5, or 1.0

    return orient_scale * contact_score


# ---------------------------------------------------------------------------
# Penalty: rear calf contact (discourage calf-balancing once upright)
# ---------------------------------------------------------------------------

def rear_calf_contact_penalty(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    onset_cos: float = 0.3,
    full_cos: float = 0.85,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="R[LR]_calf"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize rear calf ground contact, soft-scaled by orientation.

    Same smooth ramp as :func:`rear_foot_contact_reward`. Calf contact
    during early standup (low orientation) costs almost nothing; as the
    robot approaches upright the penalty ramps to full strength. This
    gives a smooth gradient to shift weight from calves to feet.

    Returns in [0, 1]: orientation_scale × (any calf in contact).
    Weight should be negative.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    # Soft orientation scale
    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    orient_scale = ((cos_angle - onset_cos) / (full_cos - onset_cos)).clamp(0.0, 1.0)

    # Calf contact detection
    net_forces = _tt(contact_sensor.data.net_forces_w)
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    calf_forces = net_forces[:, body_ids, :]
    force_mag = torch.linalg.norm(calf_forces, dim=-1)
    any_contact = (force_mag > 1.0).any(dim=-1).float()

    return orient_scale * any_contact
