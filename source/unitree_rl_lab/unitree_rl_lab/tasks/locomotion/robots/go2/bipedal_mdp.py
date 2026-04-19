"""Bipedal-specific MDP helpers for Go2.

Narrow set of rewards used by :mod:`bipedal_env_cfg`. The recipe is a Go2 port
of arclab-hku's Go1 baseline (``bipedal_dog_config_baseline.py``) / the npj
Robotics TumblerNet paper: a monotonic orientation reward, an L2 base-height
pull, ungated tracking, per-joint-group L1 regularizers toward the flat
quadruped default, and an L2 penalty on the body-frame gravity xy-components.

``desired_gravity`` convention (gravity in the target-pose body frame):

* Rear-legs stance  (nose-up, "begging"): ``(-1.0, 0.0, 0.0)``
* Front-legs stance (nose-down, "handstand"): ``(1.0, 0.0, 0.0)``
* Flat quadruped reference: ``(0.0, 0.0, -1.0)``
"""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# IsaacLab 3.0: ``asset.data.*`` properties return Warp arrays and do not
# transparently interop with torch operators. Every torch-side use of an
# ``asset.data.*`` property must be wrapped in ``wp.to_torch(...)``. Alias
# kept short for readability.
_tt = wp.to_torch


def orientation_align(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Monotonic "tilt toward the target stance" reward.

    Returns ``max(0, dot(projected_gravity_b, desired_gravity))``, clamped
    to ``[0, 1]``:

    * 0 when the body is flat (gravity aligned with ``-body_z``),
    * 1 when perfectly aligned with the bipedal stance,
    * 0 for any tilt away from the target (tilt in the wrong direction
      earns no reward; with ``only_positive_rewards`` the floor keeps it
      from becoming a penalty).

    Monotonic from flat to target, so the policy has a nonzero gradient
    everywhere on the tilt manifold — unlike a Gaussian that bottoms out
    far from target.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=torch.float32)
    dot = torch.sum(_tt(asset.data.projected_gravity_b) * target, dim=-1)
    return torch.clamp(dot, min=0.0)


def orientation_xy_sq(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on the body-frame gravity xy-components.

    Returns ``proj_grav_b.x^2 + proj_grav_b.y^2`` (``0`` for a flat body,
    ``1`` for a fully-pitched or fully-rolled body). Use with a small
    negative weight.

    In the rear-stance bipedal pose the x-component is large on purpose
    (``proj_grav_b.x ≈ -1``), so this term acts against the pitch-up we
    want. The intended role is therefore a *small* damping term that
    penalises lateral roll (``proj_grav_b.y``) and keeps the body from
    over-rotating past vertical — it is not a standalone orientation
    penalty. Matches arclab-hku's ``_reward_orientation`` weighted
    negatively.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pg = _tt(asset.data.projected_gravity_b)
    return torch.sum(pg[:, :2] * pg[:, :2], dim=-1)


def base_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty ``(base_z - target_height)^2``. Use with a negative weight.

    The quadratic has nonzero gradient everywhere, so it pulls the base
    continuously toward the target height — the Gaussian shape plateaus
    far from target and is dead weight from a flat-quadruped spawn.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(_tt(asset.data.root_pos_w)[:, 2] - target_height)


def lin_vel_z_world_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on vertical (world-Z) linear velocity.

    The stock :func:`isaaclab.envs.mdp.lin_vel_z_l2` squares ``root_lin_vel_b[:, 2]``
    (body-frame Z). For a bipedal rear-stance, body-Z points *horizontally*
    (perpendicular to gravity), so the body-frame version penalises forward /
    backward world motion instead of vertical bouncing — exactly the opposite
    of what is wanted. This term squares ``root_lin_vel_w[:, 2]`` so the
    penalty is always aligned with gravity regardless of body tilt.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(_tt(asset.data.root_lin_vel_w)[:, 2])


def ang_vel_xy_world_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on roll / pitch rates in world frame.

    The stock :func:`isaaclab.envs.mdp.ang_vel_xy_l2` penalises
    ``root_ang_vel_b[:, :2]`` (body-frame roll and pitch rates). At a ~90°
    bipedal tilt, body-X points along world-Z, so the body-X angular
    velocity is the world-Z yaw rate — which is the command the policy is
    supposed to track. This world-frame version penalises ``root_ang_vel_w[:, :2]``
    (world roll + pitch rates) and therefore never conflicts with the
    yaw-rate command.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(_tt(asset.data.root_ang_vel_w)[:, :2]), dim=1)


def joint_deviation_from_default_l1(
    env: "ManagerBasedRLEnv",
    joint_patterns: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 penalty on selected joints' deviation from their ``default_joint_pos``.

    Matches joint-name regexes once per config (cached), then returns
    ``sum |q_i - q_default_i|`` over the selected joints. Used for
    TumblerNet-style per-joint-group "motion" penalties (hip / thigh /
    calf), a soft pull toward the neutral quadruped pose that co-exists
    with the bipedal orientation reward: the orientation reward dominates
    and pulls the body up, while the motion penalty keeps the legs from
    drifting to unreasonable poses.
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
    q = _tt(asset.data.joint_pos)[:, ids]
    q0 = _tt(asset.data.default_joint_pos)[:, ids]
    return torch.sum(torch.abs(q - q0), dim=-1)
