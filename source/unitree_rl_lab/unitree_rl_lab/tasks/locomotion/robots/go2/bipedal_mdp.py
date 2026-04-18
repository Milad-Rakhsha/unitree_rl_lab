"""Bipedal-specific MDP helpers for Go2 (stand/walk on hind or front legs).

Keeps the generic locomotion `mdp` package clean by scoping the bipedal-only
reward / termination / reset utilities to this robot-specific module.

Conventions for ``desired_gravity`` (gravity expressed in the body frame of a
robot that is in the target pose):

* Rear-legs stance  (Go2 "begging", body pitched nose-up): ``(-1.0, 0.0, 0.0)``
* Front-legs stance (Go2 "handstand", body pitched nose-down): ``(1.0, 0.0, 0.0)``

The nominal quadruped stance corresponds to ``(0.0, 0.0, -1.0)``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def target_orientation_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential shaped reward for aligning projected gravity with a target vector.

    Returns ``exp(-(1 - cos_sim)^2 / std)`` which is 1.0 at perfect alignment
    and decays smoothly as the body tilts away from the desired pose.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    cos_sim = torch.sum(asset.data.projected_gravity_b * target, dim=-1)
    err = torch.square(1.0 - cos_sim)
    return torch.exp(-err / std)


def target_orientation_l2(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on deviation of projected gravity from a target vector."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    return torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=-1)


def base_height_exp(
    env: "ManagerBasedRLEnv",
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential shaped reward for keeping the base near a target world-frame z."""
    asset: RigidObject = env.scene[asset_cfg.name]
    err = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-err / std)


def orientation_align(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Monotonic "tilt toward the target stance" reward.

    Returns ``max(0, dot(projected_gravity_b, desired_gravity))``, clamped to
    :math:`[0, 1]`:

    * 0 when the body is flat (quadruped / gravity aligned with ``-body_z``),
    * 1 when perfectly aligned with the bipedal stance,
    * 0 for any tilt away from the desired stance (e.g. nose-down when the
      target is nose-up) — with ``only_positive_rewards`` this just means
      "no reward", not a penalty.

    This is the Go2 counterpart of arclab-hku Go1's ``_reward_orientation``
    (``sum(projected_gravity_xy**2)``, positive-weighted). That version is
    non-directional (symmetric about the robot's y-axis); this one picks the
    correct stance (rear vs front) via ``desired_gravity``.

    Critically, the reward is *monotonic* from flat to the target pose, so
    the policy has a gradient everywhere on the manifold — unlike a narrow
    Gaussian that bottoms out to numerical zero and leaves the agent blind.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    dot = torch.sum(asset.data.projected_gravity_b * target, dim=-1)
    return torch.clamp(dot, min=0.0)


def orientation_commit(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    threshold: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sharp quadratic reward for the final approach to the target stance.

    Returns ``max(0, dot − threshold)²``:

    * 0 for ``dot ≤ threshold`` (no effect while the policy is exploring
      the lower half of the tilt manifold — the linear
      :func:`orientation_align` term provides the gradient there),
    * grows quadratically from ``threshold`` to ``1.0``, so the marginal
      reward per unit of extra tilt *increases* as the body approaches
      the target.

    With ``threshold = 0.7`` and a weight of ``10.0``:

    ======  ============  ================================
    ``dot`` bonus value   bonus at weight 10
    ======  ============  ================================
    0.70    0.00          +0.00
    0.80    0.01          +0.10
    0.85    0.0225        +0.22
    0.90    0.04          +0.40
    0.95    0.0625        +0.62
    1.00    0.09          +0.90
    ======  ============  ================================

    The quadratic grows fastest near the goal — exactly the opposite of
    the linear :func:`orientation_align` term, which has constant
    marginal reward and becomes too weak to punch through the
    nonlinearly-growing balance cost as the robot approaches vertical.

    Args:
        desired_gravity: Gravity direction in the target-pose body frame.
        threshold: Lower cutoff on ``dot`` below which the reward is 0.
            ``0.7`` corresponds to ~45° from the target stance.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    dot = torch.sum(asset.data.projected_gravity_b * target, dim=-1)
    shaped = torch.clamp(dot - threshold, min=0.0)
    return shaped * shaped


def track_lin_vel_xy_orient_gated_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``track_lin_vel_xy_exp`` multiplied by a **soft** orientation gate.

    The gate is ``max(0, dot(proj_grav_b, desired_gravity))`` (= the same
    quantity as :func:`orientation_align`). It varies smoothly from 0
    (flat quadruped) to 1 (target bipedal stance), so the reward is:

    * 0 at flat, regardless of tracking quality — prevents a
      hunched-quadruped local optimum from collecting tracking reward,
    * scaled linearly as the body pitches up,
    * full ``exp(-err)`` at the target stance.

    Crucially, this is *multiplicative*, so the policy cannot separate
    "walk well" from "stand up" — the only way to earn tracking reward
    is to be upright. This was the missing piece in our earlier FlatInit
    runs: tracking was pose-independent, so the policy walked as a
    slightly-leaned quadruped (dot ≈ 0.4) with near-max tracking and
    ignored the orientation reward.
    """
    from unitree_rl_lab.tasks.locomotion import mdp as _locomotion_mdp

    asset: RigidObject = env.scene[asset_cfg.name]
    reward = _locomotion_mdp.track_lin_vel_xy_exp(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    gate = torch.clamp(torch.sum(asset.data.projected_gravity_b * target, dim=-1), min=0.0)
    return reward * gate


def track_ang_vel_z_orient_gated_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Orientation-gated ``track_ang_vel_z_exp``.

    See :func:`track_lin_vel_xy_orient_gated_exp` for gate semantics.
    """
    from unitree_rl_lab.tasks.locomotion import mdp as _locomotion_mdp

    asset: RigidObject = env.scene[asset_cfg.name]
    reward = _locomotion_mdp.track_ang_vel_z_exp(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    gate = torch.clamp(torch.sum(asset.data.projected_gravity_b * target, dim=-1), min=0.0)
    return reward * gate


def base_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty ``(base_z - target_height)^2``.

    Use with a negative weight. Matches arclab-hku's ``_reward_base_height``.
    Prefer this over the Gaussian :func:`base_height_exp` for bipedal
    training from a flat spawn: the Gaussian bottoms out far from target
    and gives no gradient, whereas the quadratic pulls continuously toward
    the target height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def lin_vel_z_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on base vertical velocity. Use with negative weight.

    Standard Legged-Gym term; critical for bipedal stability (prevents
    bouncing / vertical oscillation modes). Not available in the
    ``unitree_rl_lab`` mdp re-exports, so provided here.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def joint_deviation_from_default_l1(
    env: "ManagerBasedRLEnv",
    joint_patterns: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 penalty on selected joints' deviation from their ``default_joint_pos``.

    Match the matching joint-name regexes once per config (cached), then
    compute ``sum |q_i - q_default_i|`` over the selected joints. Used for
    arclab-hku-style "motion penalties" (``hip_motion``, ``thigh_motion``,
    ``calf_motion``) that softly regularize toward the neutral quadruped
    pose. Combined with a monotonic orientation reward, this is still
    compatible with bipedal training — the orientation reward dominates
    and pulls the body up, while the motion penalty keeps the legs from
    going to absurd poses.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cache_attr = "_bipedal_joint_default_reward_cache"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, {})
    cache_dict: dict = getattr(env, cache_attr)
    cache_key = tuple(joint_patterns)
    ids = cache_dict.get(cache_key)
    if ids is None:
        joint_ids, _ = asset.find_joints(list(joint_patterns))
        ids = torch.tensor(sorted(set(int(j) for j in joint_ids)), dtype=torch.long, device=env.device)
        cache_dict[cache_key] = ids
    if ids.numel() == 0:
        return torch.zeros(env.num_envs, device=env.device)
    q = asset.data.joint_pos[:, ids]
    q0 = asset.data.default_joint_pos[:, ids]
    return torch.sum(torch.abs(q - q0), dim=-1)


def track_lin_vel_xy_upright_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    desired_gravity: list[float],
    cos_threshold: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``track_lin_vel_xy_exp``, gated on "is upright" for bipedal locomotion.

    Without the gate, a partially-collapsed robot can earn full tracking reward
    by matching the commanded velocity in a quadruped-ish stance — a competing
    local optimum that blocks bipedal emergence. The gate forces the policy to
    first align the body with the target stance before tracking pays anything.
    "Upright" is defined as ``dot(projected_gravity_b, desired_gravity) >
    cos_threshold`` (default 0.7 ≈ within 45° of the target stance).

    Idea adapted from Zhang et al. 2025, "Bipedalism for Quadrupedal Robots"
    (arXiv:2507.20382), Table I.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    cmd = env.command_manager.get_command(command_name)[:, :2]
    err = torch.sum(torch.square(cmd - asset.data.root_lin_vel_b[:, :2]), dim=-1)
    reward = torch.exp(-err / (std * std))
    upright = torch.sum(asset.data.projected_gravity_b * target, dim=-1) > cos_threshold
    return reward * upright.to(reward.dtype)


def track_ang_vel_z_upright_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    desired_gravity: list[float],
    cos_threshold: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``track_ang_vel_z_exp``, gated on "is upright" (see :func:`track_lin_vel_xy_upright_exp`)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    cmd = env.command_manager.get_command(command_name)[:, 2]
    err = torch.square(cmd - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-err / (std * std))
    upright = torch.sum(asset.data.projected_gravity_b * target, dim=-1) > cos_threshold
    return reward * upright.to(reward.dtype)


def joint_deviation_from_target_l1(
    env: "ManagerBasedRLEnv",
    joint_targets: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 penalty on joint deviation from configured per-joint target angles.

    Unlike :func:`isaaclab.envs.mdp.rewards.joint_deviation_l1`, which measures
    deviation from ``default_joint_pos`` (the flat quadruped stance), this term
    lets the caller specify a custom target pose. That matters for bipedal
    training: the "swing" legs have a tucked target (e.g. ``thigh=1.3,
    calf=-2.4``) that is very different from the quadruped default, so using
    ``default_joint_pos`` would actively pull the swing legs away from the pose
    we want.

    Args:
        joint_targets: Mapping from joint-name regex to target angle [rad].
            Joints matched by any pattern contribute a ``|q - q_target|`` term;
            unmatched joints are ignored.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    cache_attr = "_bipedal_joint_target_reward_cache"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, {})
    cache_dict: dict = getattr(env, cache_attr)
    cache_key = id(joint_targets)
    cache = cache_dict.get(cache_key)
    if cache is None:
        ids: list[int] = []
        targets: list[float] = []
        for pattern, target in joint_targets.items():
            joint_ids, _ = asset.find_joints(pattern)
            for jid in joint_ids:
                ids.append(int(jid))
                targets.append(float(target))
        idx_t = torch.tensor(ids, dtype=torch.long, device=env.device)
        tgt_t = torch.tensor(targets, dtype=torch.float, device=env.device)
        cache = (idx_t, tgt_t)
        cache_dict[cache_key] = cache
    joint_idx, targets_t = cache

    if joint_idx.numel() == 0:
        return torch.zeros(env.num_envs, device=env.device)

    deviation = asset.data.joint_pos[:, joint_idx] - targets_t.unsqueeze(0)
    return torch.sum(torch.abs(deviation), dim=-1)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def bad_target_orientation(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when body orientation deviates more than ``limit_angle`` (rad) from target.

    Angle is computed between the projected gravity vector in the body frame and
    the desired-gravity target (both assumed unit length).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    cos_sim = torch.sum(asset.data.projected_gravity_b * target, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos_sim) > limit_angle


# ---------------------------------------------------------------------------
# Reset events
# ---------------------------------------------------------------------------


def reset_joints_to_target(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    joint_targets: dict[str, tuple[float, float]],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot joints to configurable per-joint target positions with uniform noise.

    Joints matched by a regex key in ``joint_targets`` are sampled uniformly from
    ``(center - halfwidth, center + halfwidth)``. Joints not matched keep their
    default positions.

    Args:
        joint_targets: ``{joint_name_regex: (center, halfwidth)}``.
        velocity_range: ``(min, max)`` additive uniform noise on default velocities.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    cache_attr = "_bipedal_reset_cache"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, {})
    cache_dict: dict = getattr(env, cache_attr)
    cache_key = id(joint_targets)
    cache = cache_dict.get(cache_key)
    if cache is None:
        ids: list[int] = []
        centers: list[float] = []
        halfwidths: list[float] = []
        for pattern, (center, halfwidth) in joint_targets.items():
            joint_ids, _ = asset.find_joints(pattern)
            for jid in joint_ids:
                ids.append(int(jid))
                centers.append(float(center))
                halfwidths.append(float(halfwidth))
        idx_t = torch.tensor(ids, dtype=torch.long, device=env.device)
        ctr_t = torch.tensor(centers, dtype=torch.float, device=env.device)
        hw_t = torch.tensor(halfwidths, dtype=torch.float, device=env.device)
        cache = (idx_t, ctr_t, hw_t)
        cache_dict[cache_key] = cache
    joint_idx, centers_t, halfwidths_t = cache

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    if joint_idx.numel() > 0:
        n = len(env_ids)
        rand = torch.rand((n, joint_idx.numel()), device=env.device) * 2.0 - 1.0
        joint_pos[:, joint_idx] = centers_t.unsqueeze(0) + rand * halfwidths_t.unsqueeze(0)

    vlo, vhi = velocity_range
    joint_vel = joint_vel + torch.rand_like(joint_vel) * (vhi - vlo) + vlo

    pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(pos_limits[..., 0], pos_limits[..., 1])
    vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp(-vel_limits, vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
