import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def _to_numpy(x):
    """Convert a torch.Tensor, warp.array, numpy.ndarray, or python scalar to numpy array."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):  # warp.array
        return x.numpy()
    return np.asarray(x)


def _actuator_joint_gains(asset) -> tuple[np.ndarray, np.ndarray]:
    """Return articulation-wide (stiffness, damping) in ``asset.data.joint_names`` order.

    Isaac Lab 3.0 deprecated ``asset.data.default_joint_stiffness``; with the Newton backend
    ``asset.data.joint_stiffness`` is not populated (actuator torque is applied externally),
    so reading it yields zeros. The ground-truth PD gains live on the actuator objects.
    """
    num_joints = len(asset.data.joint_names)
    stiffness = np.zeros(num_joints, dtype=np.float32)
    damping = np.zeros(num_joints, dtype=np.float32)
    for actuator in asset.actuators.values():
        joint_ids = actuator.joint_indices
        if isinstance(joint_ids, slice):
            joint_ids = list(range(*joint_ids.indices(num_joints)))
        kp = _to_numpy(actuator.stiffness)[0]
        kd = _to_numpy(actuator.damping)[0]
        stiffness[joint_ids] = kp
        damping[joint_ids] = kd
    return stiffness, damping


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    asset: Articulation = env.scene["robot"]
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation

    actuator_stiffness, actuator_damping = _actuator_joint_gains(asset)
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = actuator_stiffness
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = actuator_damping
    cfg["damping"] = damping.tolist()
    if not np.any(stiffness) or not np.any(damping):
        raise RuntimeError(
            "export_deploy_cfg: resolved actuator stiffness/damping are all zero. "
            "Check the robot's ActuatorCfg (stiffness/damping) before exporting."
        )

    cfg["default_joint_pos"] = _to_numpy(asset.data.default_joint_pos)[0].tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = _to_numpy(action_term._scale)[0].tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = _to_numpy(action_term._clip)[0].tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = _to_numpy(action_term._offset)[0].tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = _to_numpy(term_cfg.scale).tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][obs_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
