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

# Several reference poses the robot might start from
QUADRUPED_JOINT_POS = [
    -0.1, 0.8, -1.5,   # FR
     0.1, 0.8, -1.5,   # FL
    -0.1, 1.0, -1.5,   # RR
     0.1, 1.0, -1.5,   # RL
]

# Crouched / low quadruped
QUADRUPED_CROUCHED_POS = [
    -0.1, 1.2, -2.0,   # FR
     0.1, 1.2, -2.0,   # FL
    -0.1, 1.4, -2.0,   # RR
     0.1, 1.4, -2.0,   # RL
]

# Splayed legs (wider stance)
QUADRUPED_SPLAYED_POS = [
    -0.3, 0.6, -1.2,   # FR
     0.3, 0.6, -1.2,   # FL
    -0.3, 0.8, -1.2,   # RR
     0.3, 0.8, -1.2,   # RL
]

_POSE_BANK = [QUADRUPED_JOINT_POS, QUADRUPED_CROUCHED_POS, QUADRUPED_SPLAYED_POS]


# ---------------------------------------------------------------------------
# Reset: diverse quadruped-like starts
# ---------------------------------------------------------------------------

def reset_quadruped_prone(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    # Position randomization
    xy_range: float = 0.3,
    # Orientation randomization — wide enough to include side-lying
    roll_range: tuple[float, float] = (-0.4, 0.4),
    pitch_range: tuple[float, float] = (-0.3, 0.3),
    # Joint noise — large so the robot sees many starting configs
    joint_pos_noise: float = 0.3,
    joint_vel_noise: float = 0.5,
    # Spawn height range (crouched to normal quadruped)
    base_z_range: tuple[float, float] = (0.20, 0.38),
    # Initial velocity range
    lin_vel_range: float = 0.3,
    ang_vel_range: float = 0.5,
    # Asset
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset envs to diverse quadruped-like poses for stand-up.

    The robot spawns in a variety of four-legged starting poses:
    - Normal quadruped, crouched, splayed (randomly selected per env)
    - Wide orientation DR (roll ±0.4, pitch ±0.3) — includes partial side-lying
    - Variable height (0.20–0.38m) — crouched to normal
    - Moderate initial velocity — not always starting from rest
    - Large joint noise (±0.3 rad) — many different leg configurations

    This ensures the stand-up policy generalizes to any quadruped-like
    starting condition, not just one specific pose.
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
    root_pos[:, 2] = _ru(base_z_range[0], base_z_range[1], (n,))

    roll = _ru(roll_range[0], roll_range[1], (n,))
    pitch = _ru(pitch_range[0], pitch_range[1], (n,))
    yaw = _ru(-3.14159, 3.14159, (n,))
    root_quat = quat_from_euler_xyz(roll, pitch, yaw)

    # --- Root velocity (moderate randomization) ---
    root_lin_vel = torch.zeros(n, 3, device=device)
    root_lin_vel[:, 0] = _ru(-lin_vel_range, lin_vel_range, (n,))
    root_lin_vel[:, 1] = _ru(-lin_vel_range, lin_vel_range, (n,))
    root_lin_vel[:, 2] = _ru(-0.1, 0.1, (n,))
    root_ang_vel = torch.zeros(n, 3, device=device)
    root_ang_vel[:, 0] = _ru(-ang_vel_range, ang_vel_range, (n,))
    root_ang_vel[:, 1] = _ru(-ang_vel_range, ang_vel_range, (n,))
    root_ang_vel[:, 2] = _ru(-ang_vel_range, ang_vel_range, (n,))

    # --- Joint state: randomly pick from pose bank per env ---
    pose_bank = torch.tensor(_POSE_BANK, device=device, dtype=torch.float32)
    pose_idx = torch.randint(0, len(_POSE_BANK), (n,), device=device)
    joint_ref = pose_bank[pose_idx]  # (n, 12)
    joint_pos = joint_ref + _ru(
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
# Reward: raw orientation alignment (no velocity gate)
# ---------------------------------------------------------------------------

def orientation_align_raw(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raw orientation alignment toward target pose.

    Returns ``max(0, dot(projected_gravity_b, desired_gravity))`` — a
    monotonic reward in [0, 1] that increases as the robot tilts toward
    the bipedal stance, regardless of velocity. This lets the robot
    actually commit to standing up rather than being penalized for moving.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )
    dot = torch.sum(_tt(asset.data.projected_gravity_b) * target, dim=-1)
    return torch.clamp(dot, min=0.0)


# ---------------------------------------------------------------------------
# Reward: success bonus (one-time reward when standup completes)
# ---------------------------------------------------------------------------

def success_bonus(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    min_cos_angle: float = 0.90,
    min_base_height: float = 0.45,
    max_ang_vel: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-step bonus when the robot is in the target upright pose.

    Unlike ``standing_success`` (which needs 50 consecutive steps),
    this gives immediate reward every step the robot holds the target
    pose. This provides strong gradient toward the goal without
    requiring the robot to hold perfectly still for a full second first.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    orient_ok = cos_angle >= min_cos_angle

    height_ok = _tt(asset.data.root_pos_w)[:, 2] >= min_base_height

    ang_vel_norm = torch.linalg.norm(
        _tt(asset.data.root_ang_vel_w), dim=-1
    )
    calm_ok = ang_vel_norm < max_ang_vel

    return (orient_ok & height_ok & calm_ok).float()


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
# Penalty: velocity scaled by orientation (slow down as you get upright)
# ---------------------------------------------------------------------------

def velocity_near_upright_penalty(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    onset_cos: float = 0.3,
    full_cos: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize velocity proportional to how upright the robot is.

    When flat (cos < onset_cos): no penalty — free to move.
    As the robot tilts upright: penalty ramps linearly.
    When near upright (cos > full_cos): full penalty on velocity.

    This lets the robot move freely during the standup transition but
    forces it to slow down and stabilize as it approaches the target
    pose. Combines angular + linear velocity into one term.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    orient_scale = ((cos_angle - onset_cos) / (full_cos - onset_cos)).clamp(0.0, 1.0)

    ang_vel_sq = torch.sum(torch.square(_tt(asset.data.root_ang_vel_w)), dim=-1)
    lin_vel_sq = torch.sum(torch.square(_tt(asset.data.root_lin_vel_w)), dim=-1)

    # Combined velocity magnitude (weighted: ang vel matters more for stability)
    vel_penalty = lin_vel_sq + 0.5 * ang_vel_sq

    return orient_scale * vel_penalty


# ---------------------------------------------------------------------------
# Penalty: horizontal drift (world-frame XY linear velocity)
# ---------------------------------------------------------------------------

def lin_vel_xy_world_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on horizontal (world XY) linear velocity.

    Penalizes backward/forward/sideways drift during stand-up.
    The robot should stand up in place, not slide around.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(_tt(asset.data.root_lin_vel_w)[:, :2]), dim=1)


# ---------------------------------------------------------------------------
# Penalty: yaw spin (world-frame Z angular velocity)
# ---------------------------------------------------------------------------

def ang_vel_z_world_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on yaw rate (world-Z angular velocity).

    Prevents the robot from spinning around the vertical axis
    during or after standing up.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(_tt(asset.data.root_ang_vel_w)[:, 2])


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


# ---------------------------------------------------------------------------
# Penalty: early standup (encourage taking longer to reach upright)
# ---------------------------------------------------------------------------

def early_standup_penalty(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    min_cos_angle: float = 0.85,
    min_steps: int = 100,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize reaching the upright pose too quickly.

    Returns a penalty (positive value to be used with negative weight)
    when the robot has cos_angle >= min_cos_angle AND the episode step
    count is less than min_steps. The penalty decays linearly as the
    step count approaches min_steps.

    This discourages the policy from snapping upright immediately and
    instead rewards a slow, controlled transition that takes at least
    ``min_steps`` steps (~2s at 50 Hz).

    Args:
        desired_gravity: Target gravity direction in body frame.
        min_cos_angle: Orientation threshold to consider "near upright".
        min_steps: Minimum step count before upright is penalty-free.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    is_upright = (cos_angle >= min_cos_angle).float()

    # Linear decay: penalty = 1.0 at step 0, 0.0 at step min_steps
    step_frac = (env.episode_length_buf.float() / min_steps).clamp(0.0, 1.0)
    time_penalty = 1.0 - step_frac

    return is_upright * time_penalty


# ---------------------------------------------------------------------------
# Reward: progressive orientation (reward orientation more as time passes)
# ---------------------------------------------------------------------------

def orientation_align_time_scaled(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    ramp_steps: int = 75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Orientation alignment scaled up over time.

    Like orientation_align_raw but the reward starts low and ramps up
    linearly over ``ramp_steps``. This means early steps get less
    reward for being upright, discouraging racing to stand immediately.
    At step >= ramp_steps, the reward is at full strength.

    This creates a "no rush" incentive: the same upright pose is worth
    more reward later in the episode than at the beginning.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    align = torch.clamp(cos_angle, min=0.0)

    # Time ramp: 0.2 at step 0, linearly to 1.0 at ramp_steps
    time_scale = (0.2 + 0.8 * (env.episode_length_buf.float() / ramp_steps)).clamp(0.2, 1.0)

    return align * time_scale


# ---------------------------------------------------------------------------
# Reward: late standup bonus (one-time, scales with episode progress)
# ---------------------------------------------------------------------------

def late_standup_bonus(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    min_cos_angle: float = 0.90,
    min_base_height: float = 0.45,
    max_ang_vel: float = 1.0,
    ideal_step: int = 400,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-time bonus when robot first reaches standing, scaled by when.

    The bonus fires exactly ONCE per episode — the first step all
    standing criteria are met. The reward magnitude depends on the
    episode step at which standing is achieved:

        reward = clamp(step / ideal_step, 0, 1)

    Standing at step 0 → ~0.0 reward (worthless).
    Standing at step ideal_step → 1.0 reward (full bonus).

    This makes the optimal strategy: take your time getting up,
    stand near step ideal_step.

    A per-env flag ``_late_standup_fired`` tracks whether the bonus
    has already been given this episode.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    orient_ok = cos_angle >= min_cos_angle

    height_ok = _tt(asset.data.root_pos_w)[:, 2] >= min_base_height

    ang_vel_norm = torch.linalg.norm(
        _tt(asset.data.root_ang_vel_w), dim=-1
    )
    calm_ok = ang_vel_norm < max_ang_vel

    all_ok = orient_ok & height_ok & calm_ok

    # Manage one-time flag
    if not hasattr(env, "_late_standup_fired"):
        env._late_standup_fired = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    fired = env._late_standup_fired

    # Reset flag for envs that just reset
    just_reset = env.episode_length_buf == 0
    fired[just_reset] = False

    # Find envs that meet criteria for the first time
    newly_standing = all_ok & (~fired)

    # Mark them as fired
    fired[newly_standing] = True

    # Time-scaled reward: later is better
    time_scale = (env.episode_length_buf.float() / ideal_step).clamp(0.0, 1.0)

    # Only reward the newly standing envs
    return newly_standing.float() * time_scale


# ---------------------------------------------------------------------------
# Penalty: excessive velocity during standup (hard cap)
# ---------------------------------------------------------------------------

def excessive_velocity_penalty(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    vel_threshold: float = 2.0,
    release_cos: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hard penalty when body velocity exceeds threshold during standup.

    Returns 1.0 (use with a large negative weight) when the combined
    velocity magnitude exceeds ``vel_threshold`` AND the robot has not
    yet reached ``release_cos`` orientation. Once the robot is nearly
    upright (cos >= release_cos), the penalty turns off.

    Combined velocity: ||lin_vel||^2 + ||ang_vel||^2

    The intent is to make fast standup motions unprofitable. The penalty
    should outweigh any per-step reward (e.g. success_bonus at +8.0),
    forcing the policy to stay under the speed limit.

    Args:
        desired_gravity: Target gravity direction in body frame.
        vel_threshold: Combined velocity threshold (squared norms).
        release_cos: Once cos_angle >= this, penalty is released.
        asset_cfg: Robot asset config.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(
        desired_gravity, device=env.device, dtype=torch.float32
    )

    # Orientation check — penalty only active while still getting up
    cos_angle = torch.sum(
        _tt(asset.data.projected_gravity_b) * target, dim=-1
    )
    still_rising = (cos_angle < release_cos).float()

    # Combined velocity magnitude
    lin_vel_sq = torch.sum(
        _tt(asset.data.root_lin_vel_w) ** 2, dim=-1
    )
    ang_vel_sq = torch.sum(
        _tt(asset.data.root_ang_vel_w) ** 2, dim=-1
    )
    vel_combined = lin_vel_sq + ang_vel_sq

    # Hard threshold — binary penalty
    too_fast = (vel_combined > vel_threshold).float()

    return still_rising * too_fast
