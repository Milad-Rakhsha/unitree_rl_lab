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
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import normalize, quat_apply, quat_apply_inverse

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


def bipedal_yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Pitch-invariant yaw-only quaternion from the horizontal projection of body-Y.

    The upstream :func:`isaaclab.utils.math.yaw_quat` extracts yaw using the
    ZYX Euler formula ``atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))``,
    which hits its gimbal-lock singularity at a body pitch of
    :math:`\\pm \\pi/2` — exactly the bipedal stance target. At the
    singularity both numerator and denominator collapse to zero and the
    returned angle is determined by floating-point rounding; crossing the
    singularity (e.g. during overshoot or foot-strike perturbations)
    flips the extracted yaw by :math:`\\pi`, silently inverting the
    "forward" direction the yaw-frame velocity-tracking reward uses.

    This helper computes yaw from the horizontal projection of body-Y
    (the roll axis). Body-Y stays horizontal for any pitch rotation —
    it only degenerates at a :math:`\\pm \\pi/2` roll, which is outside
    the operating envelope of either the flat-quadruped or the bipedal
    task. The returned yaw matches standard :func:`yaw_quat` for flat
    poses (drop-in compatible) and is continuous through the bipedal
    singularity.

    Args:
        quat: Body orientation quaternion ``(x, y, z, w)``. Shape
            ``(..., 4)``.

    Returns:
        Pure-yaw quaternion ``(0, 0, sin(yaw/2), cos(yaw/2))`` of the
        same shape as ``quat``.
    """
    qx = quat[..., 0]
    qy = quat[..., 1]
    qz = quat[..., 2]
    qw = quat[..., 3]
    y_w_x = 2.0 * (qx * qy - qw * qz)
    y_w_y = 1.0 - 2.0 * (qx * qx + qz * qz)
    yaw = torch.atan2(-y_w_x, y_w_y)
    out = torch.zeros_like(quat)
    out[..., 2] = torch.sin(0.5 * yaw)
    out[..., 3] = torch.cos(0.5 * yaw)
    return normalize(out)


def track_lin_vel_xy_bipedal_frame_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear velocity tracking reward in a pitch-invariant walking frame.

    Functionally equivalent to
    :func:`~isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards.track_lin_vel_xy_yaw_frame_exp`
    but uses :func:`bipedal_yaw_quat` instead of the upstream
    :func:`~isaaclab.utils.math.yaw_quat`. The two helpers agree at flat
    quadruped poses; the bipedal variant additionally remains continuous
    across the :math:`\\pm \\pi/2`-pitch bipedal singularity, where
    ``yaw_quat`` flips sign and silently inverts the meaning of the
    ``lin_vel_x`` command.

    Args:
        env: Environment instance.
        std: Standard deviation of the exponential kernel.
        command_name: Name of the velocity command term.
        asset_cfg: Robot asset configuration.

    Returns:
        Per-env exponential-kernel reward in ``[0, 1]``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(
        bipedal_yaw_quat(_tt(asset.data.root_quat_w)),
        _tt(asset.data.root_lin_vel_w)[:, :3],
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


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


def _foot_point_kinematics(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    kinematics_sensor_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame position and velocity of the original foot origins.

    Without fixed-joint collapse, the original MJWarp MDP reads the foot rigid
    bodies directly. Under DVI collapse those bodies no longer exist, so one
    PVA site is placed at each former foot origin on its parent calf. PVA
    accounts for the rigid-body transport term ``omega x r``; rotating its
    body-frame velocity back to world gives the same kinematic quantity that
    ``body_lin_vel_w`` exposed before collapse.
    """
    if not kinematics_sensor_names:
        asset: RigidObject = env.scene[asset_cfg.name]
        return (
            _tt(asset.data.body_pos_w)[:, asset_cfg.body_ids, :],
            _tt(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, :],
        )

    positions = []
    velocities = []
    for sensor_name in kinematics_sensor_names:
        sensor = env.scene.sensors[sensor_name]
        quat_w = sensor.data.quat_w.torch
        positions.append(sensor.data.pos_w.torch)
        velocities.append(quat_apply(quat_w, sensor.data.lin_vel_b.torch))
    return torch.stack(positions, dim=1), torch.stack(velocities, dim=1)


def feet_clearance_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str,
    min_command_magnitude: float = 0.1,
    kinematics_sensor_names: list[str] | None = None,
) -> torch.Tensor:
    """Reward swinging feet for clearing a target height off the ground.

    For every body in ``asset_cfg.body_ids`` the per-foot contribution is
    the squared deviation of the foot's world-Z from ``target_height``
    multiplied by ``tanh(tanh_mult * |v_horiz|)`` of that foot's world-frame
    horizontal speed. The tanh gate is the key: a stationary stance foot
    has ``|v_horiz| ~= 0`` so its contribution is ~0 regardless of its
    z-position, while a swinging foot (moving horizontally) pays the full
    quadratic penalty against ``target_height``. Per-foot contributions are
    summed and passed through ``exp(-sum / std)`` so the term is a positive
    reward, maximal when every moving foot sits at ``target_height``.

    Gated on non-trivial linear command magnitude: when the commanded
    planar velocity has norm below ``min_command_magnitude`` the robot
    is in a standing regime and the reward returns zero so the policy
    is not pushed into spurious foot motion from a held pose.

    Ported from ``spot/mdp/rewards.py::foot_clearance_reward`` with the
    addition of a command gate. Use with a small positive weight.
    """
    foot_pos_w, foot_lin_vel_w = _foot_point_kinematics(
        env, asset_cfg, kinematics_sensor_names
    )
    foot_z_target_error = torch.square(foot_pos_w[:, :, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(foot_lin_vel_w[:, :, :2], dim=2)
    )
    weighted = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(weighted, dim=1) / std)
    command = env.command_manager.get_command(command_name)[:, :2]
    active = (torch.linalg.norm(command, dim=1) > min_command_magnitude).float()
    return reward * active


def feet_slide_reward(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    kinematics_sensor_names: list[str] | None = None,
) -> torch.Tensor:
    """Original foot-slide penalty using collapse-safe foot-origin velocity."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        _tt(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    _, foot_lin_vel_w = _foot_point_kinematics(
        env, asset_cfg, kinematics_sensor_names
    )
    return torch.sum(foot_lin_vel_w[:, :, :2].norm(dim=-1) * contacts, dim=1)


def rear_feet_airborne_at_standstill(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    min_command_magnitude: float = 0.1,
) -> torch.Tensor:
    """Return the airborne fraction of selected rear feet at near-zero command.

    The value is zero whenever the full velocity-command vector has norm at
    least ``min_command_magnitude``.  At standstill, one lifted rear foot
    returns ``0.5`` and two lifted feet return ``1.0``.  Use a negative
    reward weight to explicitly discourage in-place marching without
    changing the active-walking gait rewards.  The full command vector is
    intentional: commanded yaw is active movement and must not be penalized
    as standing still.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = _tt(contact_sensor.data.current_contact_time)[:, sensor_cfg.body_ids]
    airborne_fraction = (contact_time <= 0.0).float().mean(dim=1)
    command = env.command_manager.get_command(command_name)
    standing = torch.linalg.norm(command, dim=1) < min_command_magnitude
    return airborne_fraction * standing.float()


def feet_both_airborne(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    min_command_magnitude: float = 0.1,
) -> torch.Tensor:
    """Indicator penalty for all selected feet being airborne simultaneously.

    Returns a per-env tensor with value ``1.0`` when none of the feet
    referenced by ``sensor_cfg.body_ids`` are in contact, and ``0.0``
    otherwise. The term is gated on linear command magnitude: when the
    commanded planar velocity has norm below ``min_command_magnitude`` the
    robot is in a standing regime and the penalty returns zero so the
    policy is not punished for standing stances.

    Use with a negative weight alongside the single-stance air-time reward:
    that reward gives ``+`` credit for exactly one foot airborne, this term
    gives ``-`` credit for two (or more) feet airborne. Together they
    break the jump/pronk local optimum — a gait that only exists between
    two airborne feet and two contacting feet phases.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = _tt(contact_sensor.data.current_contact_time)[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    all_airborne = torch.all(~in_contact, dim=1).float()
    command = env.command_manager.get_command(command_name)[:, :2]
    active = torch.linalg.norm(command, dim=1) > min_command_magnitude
    return all_airborne * active.float()


def bad_bipedal_orientation(
    env: "ManagerBasedRLEnv",
    desired_gravity: list[float],
    max_angle_to_target: float,
    falling_vel_threshold: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the body is falling away from the bipedal target pose.

    The standard :func:`isaaclab.envs.mdp.bad_orientation` measures the angle
    between body-Z and world-up, which is geometrically incompatible with a
    bipedal target where body-Z is intended to be horizontal: the target pose
    itself sits at angle :math:`\\pi/2` from world-up and would always
    terminate under any sensible ``limit_angle``.

    This term measures the angle between :attr:`Articulation.data.projected_gravity_b`
    and ``desired_gravity`` (gravity expressed in the *target* body frame),
    which is ``0`` at the target pose, :math:`\\pi/2` at the flat quadruped
    spawn, and :math:`\\pi`` for an upside-down body.

    When ``falling_vel_threshold > 0``, the termination is gated on the
    world-frame vertical velocity of the base: only robots whose CoM is
    moving **downward** faster than ``-falling_vel_threshold`` [m/s] are
    terminated. This distinguishes a robot that is actively falling
    (high-contact crash imminent) from one that is rising toward the
    bipedal target through the same angular region. A robot tilting
    upward has ``root_lin_vel_w.z >= 0`` and is spared.

    Args:
        env: Environment instance.
        desired_gravity: Gravity direction in the target-pose body frame
            (unit vector). See module docstring for the rear/front/flat
            conventions.
        max_angle_to_target: Termination threshold [rad]. The robot is
            terminated when the angle between ``projected_gravity_b`` and
            ``desired_gravity`` exceeds this value (subject to velocity
            gate).
        falling_vel_threshold: Minimum downward speed [m/s] to trigger
            termination.  ``0`` disables the velocity gate (any pose past
            the angle threshold terminates). Typical values: ``0.5``--``1.0``.
        asset_cfg: Robot asset configuration.

    Returns:
        Boolean per-env termination signal.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.as_tensor(desired_gravity, device=env.device, dtype=torch.float32)
    cos_threshold = float(torch.cos(torch.tensor(max_angle_to_target)))
    cos_angle = torch.sum(_tt(asset.data.projected_gravity_b) * target, dim=-1)
    past_angle = cos_angle < cos_threshold
    if falling_vel_threshold <= 0.0:
        return past_angle
    vel_z = _tt(asset.data.root_lin_vel_w)[:, 2]
    is_falling = vel_z < -falling_vel_threshold
    return past_angle & is_falling


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
