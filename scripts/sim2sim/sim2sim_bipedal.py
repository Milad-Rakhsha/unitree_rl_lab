#!/usr/bin/env python3
"""Standalone MuJoCo sim2sim transfer test for Go2 bipedal walking policy.

Runs the trained Isaac Lab policy in MuJoCo without any DDS/SDK dependency.
Implements the same FSM as the real deploy: FixStand → BipedalRear (intro → policy).

Usage:
    conda run -n go2 python sim2sim_bipedal.py \
        --checkpoint ~/Repos/GO2/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk/<run>/model_<iter>.pt \
        [--deploy-yaml ~/Repos/GO2/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk/<run>/params/deploy.yaml] \
        [--scene ~/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene.xml] \
        [--duration 10.0] \
        [--render] \
        [--csv output.csv]
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Actuator model (matches Isaac Lab UnitreeActuatorCfg_Go2HV)
# ---------------------------------------------------------------------------

# Torque-speed curve parameters for Go2
ACTUATOR_Y1 = 20.2   # Peak torque (same direction) N·m
ACTUATOR_Y2 = 23.4   # Peak torque (opposing direction) N·m
ACTUATOR_X1 = 13.5   # Max speed at full torque rad/s
ACTUATOR_X2 = 30.0   # No-load speed rad/s
ACTUATOR_FRICTION = 0.01  # Friction coefficient from config


def clip_torque_speed_curve(tau, joint_vel):
    """Apply Go2 torque-speed curve limiting (per-joint).
    
    Matches UnitreeActuator._clip_effort in Isaac Lab.
    """
    same_direction = (joint_vel * tau) > 0
    max_effort = np.where(same_direction, ACTUATOR_Y1, ACTUATOR_Y2)

    # Linear ramp-down from X1 to X2
    abs_vel = np.abs(joint_vel)
    above_x1 = abs_vel > ACTUATOR_X1
    if np.any(above_x1):
        k = -max_effort / (ACTUATOR_X2 - ACTUATOR_X1)
        limit = k * (abs_vel - ACTUATOR_X1) + max_effort
        limit = np.maximum(limit, 0.0)
        max_effort = np.where(above_x1, limit, max_effort)

    # Apply friction
    friction_torque = ACTUATOR_FRICTION * joint_vel
    tau_with_friction = tau - friction_torque

    return np.clip(tau_with_friction, -max_effort, max_effort)


# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------


# Isaac Lab joint order (training order)
ISAAC_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

# MuJoCo joint order (from go2.xml)
MUJOCO_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Default joint positions (Isaac Lab order)
DEFAULT_JOINT_POS_ISAAC = np.array([
    0.1, -0.1, 0.1, -0.1,    # hips: FL, FR, RL, RR
    0.8, 0.8, 1.0, 1.0,      # thighs: FL, FR, RL, RR
    -1.5, -1.5, -1.5, -1.5,  # calves: FL, FR, RL, RR
])


def build_isaac_to_mujoco_map(mj_model):
    """Build mapping from Isaac Lab joint index to MuJoCo actuator index.
    
    Isaac Lab order: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf
    MuJoCo actuator order (from XML): FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
    """
    mj_actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
    
    # Build name->index map for actuators
    act_name_to_idx = {}
    for i, name in enumerate(mj_actuator_names):
        act_name_to_idx[name] = i
    
    # Isaac joint name -> MuJoCo actuator name mapping
    isaac_to_mj_actuator_name = {
        "FL_hip_joint": "FL_hip",
        "FR_hip_joint": "FR_hip",
        "RL_hip_joint": "RL_hip",
        "RR_hip_joint": "RR_hip",
        "FL_thigh_joint": "FL_thigh",
        "FR_thigh_joint": "FR_thigh",
        "RL_thigh_joint": "RL_thigh",
        "RR_thigh_joint": "RR_thigh",
        "FL_calf_joint": "FL_calf",
        "FR_calf_joint": "FR_calf",
        "RL_calf_joint": "RL_calf",
        "RR_calf_joint": "RR_calf",
    }
    
    isaac_to_actuator = []
    for isaac_name in ISAAC_JOINT_NAMES:
        mj_name = isaac_to_mj_actuator_name[isaac_name]
        isaac_to_actuator.append(act_name_to_idx[mj_name])
    
    return np.array(isaac_to_actuator)


def build_joint_pos_mapping(mj_model):
    """Build mapping from Isaac Lab joint index to MuJoCo qpos index (skipping free joint 7 dof)."""
    # Free joint takes qpos[0:7] (3 pos + 4 quat)
    # Then each hinge joint takes 1 qpos entry
    mj_joint_names = []
    mj_joint_qpos_idx = []
    for i in range(mj_model.njnt):
        j = mj_model.joint(i)
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        mj_joint_names.append(j.name)
        mj_joint_qpos_idx.append(j.qposadr)

    # Map Isaac index -> qpos index
    isaac_to_qpos = []
    isaac_to_qvel = []
    for isaac_name in ISAAC_JOINT_NAMES:
        found = False
        for idx, mj_name in enumerate(mj_joint_names):
            if mj_name == isaac_name:
                isaac_to_qpos.append(mj_joint_qpos_idx[idx])
                # qvel for free joint is 6 dof, then 1 per hinge
                isaac_to_qvel.append(6 + idx)
                found = True
                break
        if not found:
            raise ValueError(f"Could not find MuJoCo joint qpos for: {isaac_name}")

    return np.array(isaac_to_qpos).flatten(), np.array(isaac_to_qvel).flatten()


# ---------------------------------------------------------------------------
# Observation pipeline
# ---------------------------------------------------------------------------

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q (wxyz format)."""
    # q = [w, x, y, z]
    w, x, y, z = q[0], q[1], q[2], q[3]
    # Compute q_conj * v * q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v - w * t + np.cross(np.array([x, y, z]), t)


def get_projected_gravity(quat_wxyz):
    """Get gravity vector in body frame. MuJoCo uses wxyz quaternion."""
    # Gravity in world frame: [0, 0, -1] (MuJoCo convention)
    gravity_world = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat_wxyz, gravity_world)


def get_base_ang_vel(quat_wxyz, gyro_data):
    """Get angular velocity in body frame. 
    In MuJoCo, the IMU sensor gives angular velocity in body frame already."""
    return gyro_data.copy()


def bipedal_yaw_quat(quat_wxyz):
    """Extract pitch-invariant yaw from body-Y horizontal projection.
    
    At the bipedal stance (~90° pitch), the standard ZYX Euler yaw extraction
    is singular. This extracts yaw from the horizontal projection of body-Y axis,
    which stays continuous through the bipedal singularity.
    """
    # Rotate body-Y [0,1,0] to world frame
    w, x, y, z = quat_wxyz
    # body_y in world = R * [0,1,0]
    body_y_world = np.array([
        2.0 * (x*y + w*z),
        1.0 - 2.0 * (x*x + z*z),
        2.0 * (y*z - w*x),
    ])
    # Yaw from horizontal projection of body-Y
    yaw = math.atan2(body_y_world[0], body_y_world[1])
    # Convert yaw to quaternion [w, x, y, z]
    return np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])


class ObservationBuffer:
    """Manages observation history for the policy."""

    def __init__(self, num_joints=12, history_length=4, device="cpu"):
        self.num_joints = num_joints
        self.history_length = history_length
        self.device = device

        # History buffers: oldest at index 0, newest at index (history_length-1)
        # Matches C++ deploy's deque: push_back (append) new, pop_front old
        self.ang_vel_history = deque(maxlen=history_length)
        self.proj_grav_history = deque(maxlen=history_length)
        self.joint_pos_rel_history = deque(maxlen=history_length)
        self.joint_vel_rel_history = deque(maxlen=history_length)

        self.reset()

    def reset(self):
        """Clear history buffers with zeros."""
        for _ in range(self.history_length):
            self.ang_vel_history.append(np.zeros(3))
            self.proj_grav_history.append(np.zeros(3))
            self.joint_pos_rel_history.append(np.zeros(self.num_joints))
            self.joint_vel_rel_history.append(np.zeros(self.num_joints))

    def reset_with_obs(self, ang_vel, proj_grav, joint_pos_rel, joint_vel_rel):
        """Reset history by replicating current observation across all slots.
        
        Matches C++ deploy's observation_manager.reset() which calls add(obs)
        history_length times, and IsaacLab's CircularBuffer first-push behavior
        which fills all buffer slots with the first appended observation.
        """
        for _ in range(self.history_length):
            self.ang_vel_history.append(ang_vel.copy())
            self.proj_grav_history.append(proj_grav.copy())
            self.joint_pos_rel_history.append(joint_pos_rel.copy())
            self.joint_vel_rel_history.append(joint_vel_rel.copy())

    def update(self, ang_vel, proj_grav, joint_pos_rel, joint_vel_rel):
        """Push new observation into history (append = push_back = newest at end)."""
        self.ang_vel_history.append(ang_vel.copy())
        self.proj_grav_history.append(proj_grav.copy())
        self.joint_pos_rel_history.append(joint_pos_rel.copy())
        self.joint_vel_rel_history.append(joint_vel_rel.copy())

    def get_policy_obs(self, velocity_commands, last_action):
        """Build the full policy observation tensor.
        
        Policy obs layout (135 dims):
          base_ang_vel * 0.2: [4 * 3] = 12
          projected_gravity:  [4 * 3] = 12
          velocity_commands:  [3]     = 3
          joint_pos_rel:      [4 * 12] = 48
          joint_vel_rel * 0.05: [4 * 12] = 48
          last_action:        [12]    = 12
          Total: 135
        """
        obs_parts = []

        # base_ang_vel (history, scaled by 0.2)
        for i in range(self.history_length):
            obs_parts.append(self.ang_vel_history[i] * 0.2)

        # projected_gravity (history)
        for i in range(self.history_length):
            obs_parts.append(self.proj_grav_history[i])

        # velocity_commands (no history)
        obs_parts.append(velocity_commands)

        # joint_pos_rel (history)
        for i in range(self.history_length):
            obs_parts.append(self.joint_pos_rel_history[i])

        # joint_vel_rel (history, scaled by 0.05)
        for i in range(self.history_length):
            obs_parts.append(self.joint_vel_rel_history[i] * 0.05)

        # last_action (no history)
        obs_parts.append(last_action)

        obs = np.concatenate(obs_parts)
        obs = np.clip(obs, -100, 100)
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def load_policy(checkpoint_path, device="cpu"):
    """Load the trained policy from a checkpoint.
    
    The checkpoint contains the full runner state. We extract just the actor.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # rsl_rl checkpoint format: {'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
    # or the OnPolicyRunner saves {'actor_state_dict', 'critic_state_dict', ...}
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "actor_state_dict" in checkpoint:
        state_dict = checkpoint["actor_state_dict"]
    else:
        # Try to find the actor weights
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        raise ValueError("Cannot find actor state dict in checkpoint")

    # Build the actor network (MLP: 135 -> 512 -> 256 -> 128 -> 12)
    from torch import nn
    actor = nn.Sequential(
        nn.Linear(135, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 12),
    )

    # Map state dict keys to our sequential model
    # rsl_rl MLPModel state dict has keys like:
    #   mlp.0.weight, mlp.0.bias, mlp.2.weight, etc.
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'actor.' prefix if present
        clean_key = key.replace("actor.", "").replace("mlp.", "")
        # Map to sequential indices
        new_state_dict[clean_key] = value

    # Try direct load first
    try:
        actor.load_state_dict(new_state_dict, strict=False)
    except Exception:
        # Try with mlp prefix (sequential layers at 0,2,4,6)
        new_state_dict2 = {}
        for key, value in state_dict.items():
            if "mlp" in key:
                # e.g. "mlp.0.weight" -> "0.weight"
                new_key = key.split("mlp.")[-1]
                new_state_dict2[new_key] = value
        actor.load_state_dict(new_state_dict2, strict=False)

    actor.eval()
    actor.to(device)
    return actor


def load_exported_policy(policy_pt_path, device="cpu"):
    """Load the exported JIT policy (policy.pt from exported/ dir)."""
    policy = torch.jit.load(policy_pt_path, map_location=device)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------

class DeployFSM:
    """Finite state machine mirroring the real Go2 deploy."""

    PASSIVE = "Passive"
    FIXSTAND = "FixStand"
    BIPEDAL_INTRO = "BipedalIntro"
    BIPEDAL_POLICY = "BipedalPolicy"

    def __init__(self, default_joint_pos, fixstand_duration=2.0, intro_duration=1.0):
        self.default_joint_pos = default_joint_pos.copy()
        self.fixstand_duration = fixstand_duration
        self.intro_duration = intro_duration
        self.state = self.FIXSTAND
        self.state_start_time = 0.0
        self.intro_start_q = None

        # FixStand: high PD gains to hold position (per-joint, Isaac order)
        # Matches deploy config: kp=[60,80,80] per leg, kd=[5,4,4] per leg
        # Isaac order: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, ...
        self.fixstand_kp = np.array([60, 60, 60, 60, 80, 80, 80, 80, 80, 80, 80, 80], dtype=np.float64)
        self.fixstand_kd = np.array([5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.float64)

        # Intro: uniform kp=60, kd=5 (matches validated working configuration)
        self.intro_kp = np.full(12, 60.0, dtype=np.float64)
        self.intro_kd = np.full(12, 5.0, dtype=np.float64)

        # Policy: training-native gains
        self.policy_kp = 25.0
        self.policy_kd = 0.5

    def get_control(self, sim_time, current_q_isaac):
        """Returns (target_q_isaac, kp, kd, policy_active, obs_reset_needed)."""
        elapsed = sim_time - self.state_start_time

        if self.state == self.FIXSTAND:
            if elapsed >= self.fixstand_duration:
                self.state = self.BIPEDAL_INTRO
                self.state_start_time = sim_time
                self.intro_start_q = current_q_isaac.copy()
                return self.get_control(sim_time, current_q_isaac)

            # Simply hold the default joint pose with high gains
            return self.default_joint_pos.copy(), self.fixstand_kp, self.fixstand_kd, False, False

        elif self.state == self.BIPEDAL_INTRO:
            if elapsed >= self.intro_duration:
                self.state = self.BIPEDAL_POLICY
                self.state_start_time = sim_time
                # Signal that obs buffer should be reset
                return None, self.policy_kp, self.policy_kd, True, True

            # Hold at default with high gains (simpler than interpolation, matches working inline test)
            return self.default_joint_pos.copy(), self.intro_kp, self.intro_kd, False, False

        elif self.state == self.BIPEDAL_POLICY:
            return None, self.policy_kp, self.policy_kd, True, False

        return self.default_joint_pos.copy(), self.fixstand_kp, self.fixstand_kd, False, False


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_sim2sim(args):
    device = "cpu"  # Policy runs on CPU for MuJoCo sim

    # Load MuJoCo model
    scene_path = args.scene
    print(f"Loading MuJoCo scene: {scene_path}")
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)

    # Set timestep
    sim_dt = 0.005  # Match training physics dt
    mj_model.opt.timestep = sim_dt
    control_dt = 0.02  # 50 Hz policy (decimation=4)
    substeps = int(control_dt / sim_dt)

    print(f"  sim_dt={sim_dt}, control_dt={control_dt}, substeps={substeps}")

    # Build joint mappings
    isaac_to_qpos, isaac_to_qvel = build_joint_pos_mapping(mj_model)
    isaac_to_actuator = build_isaac_to_mujoco_map(mj_model)
    print(f"  Isaac->MuJoCo qpos mapping: {isaac_to_qpos}")
    print(f"  Isaac->MuJoCo qvel mapping: {isaac_to_qvel}")
    print(f"  Isaac->MuJoCo actuator mapping: {isaac_to_actuator}")

    # Initialize in the standing pose with proper height (start high, will settle)
    mj_data.qpos[0:3] = [0, 0, 0.31]  # equilibrium standing height
    mj_data.qpos[3:7] = [1, 0, 0, 0]  # identity quaternion (wxyz)
    for i, qpos_idx in enumerate(isaac_to_qpos):
        mj_data.qpos[qpos_idx] = DEFAULT_JOINT_POS_ISAAC[i]
    mujoco.mj_forward(mj_model, mj_data)

    # Load policy
    print(f"Loading policy from: {args.checkpoint}")
    if args.checkpoint.endswith(".pt") and "exported" in args.checkpoint:
        # JIT exported policy
        policy = load_exported_policy(args.checkpoint, device)
        use_jit = True
    else:
        policy = load_policy(args.checkpoint, device)
        use_jit = False
    print(f"  Policy loaded ({'JIT' if use_jit else 'state_dict'})")

    # Load deploy config if available
    deploy_cfg = None
    if args.deploy_yaml and os.path.exists(args.deploy_yaml):
        with open(args.deploy_yaml) as f:
            deploy_cfg = yaml.safe_load(f)
        print(f"  Deploy config loaded: {args.deploy_yaml}")

    # Use deploy config PD gains if available (per-joint, Isaac order)
    if deploy_cfg and "stiffness" in deploy_cfg:
        policy_kp = np.array(deploy_cfg["stiffness"])
        policy_kd = np.array(deploy_cfg["damping"])
        default_joint_pos = np.array(deploy_cfg["default_joint_pos"])
        print(f"  Using deploy.yaml PD gains: kp_mean={policy_kp.mean():.1f}, kd_mean={policy_kd.mean():.3f}")
    else:
        policy_kp = np.full(12, 25.0)
        policy_kd = np.full(12, 0.5)
        default_joint_pos = DEFAULT_JOINT_POS_ISAAC.copy()

    # Initialize FSM
    fsm = DeployFSM(default_joint_pos, fixstand_duration=2.0, intro_duration=1.0)

    # Initialize observation buffer
    obs_buffer = ObservationBuffer(num_joints=12, history_length=4, device=device)

    # State
    last_action = np.zeros(12)
    velocity_commands = np.array([args.cmd_vx, 0.0, args.cmd_yaw])  # [vx, vy, yaw]

    # CSV logging
    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "t_sim", "state", "base_x", "base_y", "base_z",
            "base_roll", "base_pitch", "base_yaw",
            *[f"qpos_{i}" for i in range(12)],
            *[f"action_{i}" for i in range(12)],
            "cmd_vx", "cmd_vy", "cmd_yaw",
        ])

    # Viewer
    viewer = None
    if args.render:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    print(f"\nStarting simulation (duration={args.duration}s, cmd_vx={args.cmd_vx}, cmd_yaw={args.cmd_yaw})")
    print(f"  FSM: FixStand(2s) → Intro(1s) → Policy")
    print("-" * 60)

    sim_time = 0.0
    control_step = 0
    max_steps = int(args.duration / control_dt)

    try:
        for step in range(max_steps):
            sim_time = step * control_dt

            # Read joint state (Isaac Lab order)
            joint_pos_isaac = np.array([mj_data.qpos[idx] for idx in isaac_to_qpos])
            joint_vel_isaac = np.array([mj_data.qvel[idx] for idx in isaac_to_qvel])

            # Read base state
            base_quat_wxyz = mj_data.qpos[3:7].copy()  # MuJoCo stores wxyz
            base_pos = mj_data.qpos[0:3].copy()

            # Get projected gravity and angular velocity in body frame
            proj_grav = get_projected_gravity(base_quat_wxyz)

            # Angular velocity — MuJoCo qvel[3:6] is ALREADY in body frame for free joints
            base_ang_vel_body = mj_data.qvel[3:6].copy()

            # FSM control
            target_q, kp, kd, policy_active, obs_reset_needed = fsm.get_control(sim_time, joint_pos_isaac)

            # Reset observation buffer at policy handover
            if obs_reset_needed:
                # Fill history with "clean startup" values:
                # - ang_vel: zeros (robot is stationary at handover)
                # - proj_grav: current value (reflects actual orientation)
                # - joint_pos/vel: zeros (policy expects near-default at startup)
                # This matches the working sim2sim configuration.
                obs_buffer.reset_with_obs(
                    np.zeros(3),  # ang_vel = 0 (stationary)
                    proj_grav,    # actual gravity orientation
                    np.zeros(12), # joint pos at default
                    np.zeros(12)  # joint vel = 0
                )
                last_action = np.zeros(12)

            if policy_active:
                # Update observation buffer
                joint_pos_rel = joint_pos_isaac - default_joint_pos
                joint_vel_rel = joint_vel_isaac  # relative to 0 (default vel)
                obs_buffer.update(base_ang_vel_body, proj_grav, joint_pos_rel, joint_vel_rel)

                # Get policy observation
                obs = obs_buffer.get_policy_obs(velocity_commands, last_action)

                # Run policy
                with torch.no_grad():
                    if use_jit:
                        action = policy(obs).squeeze(0).numpy()
                    else:
                        action = policy(obs).squeeze(0).detach().numpy()

                # Action is scaled: target_q = action * scale + offset
                action_scale = 0.25
                target_q = action * action_scale + default_joint_pos
                last_action = action.copy()

                # Debug: print first 3 policy steps
                policy_step_count = getattr(run_sim2sim, '_policy_steps', 0)
                if policy_step_count < 3:
                    print(f"  DEBUG policy_step={policy_step_count}: t={sim_time:.3f} z={base_pos[2]:.4f}")
                    print(f"    ang_vel_body={base_ang_vel_body}")
                    print(f"    proj_grav={proj_grav}")
                    print(f"    jpos_rel={joint_pos_rel}")
                    print(f"    obs[:12]={obs[0,:12].numpy()}")
                    print(f"    obs[12:24]={obs[0,12:24].numpy()}")
                    print(f"    obs[24:27]={obs[0,24:27].numpy()}")
                    print(f"    action={action}")
                    print(f"    target_q={target_q}")
                    print(f"    kp_arr={kp_arr}")
                    run_sim2sim._policy_steps = policy_step_count + 1

                # Use policy-native PD gains
                kp_arr = policy_kp if isinstance(policy_kp, np.ndarray) else np.full(12, policy_kp)
                kd_arr = policy_kd if isinstance(policy_kd, np.ndarray) else np.full(12, policy_kd)
            else:
                kp_arr = kp if isinstance(kp, np.ndarray) else np.full(12, kp)
                kd_arr = kd if isinstance(kd, np.ndarray) else np.full(12, kd)

            # Step physics with PD recomputed at physics rate
            for _ in range(substeps):
                # Compute PD torques for all joints
                tau_arr = np.zeros(12)
                vel_arr = np.zeros(12)
                for i in range(12):
                    qpos_idx = int(isaac_to_qpos[i])
                    qvel_idx = int(isaac_to_qvel[i])
                    q_actual = mj_data.qpos[qpos_idx]
                    dq_actual = mj_data.qvel[qvel_idx]
                    tau_arr[i] = kp_arr[i] * (target_q[i] - q_actual) + kd_arr[i] * (0 - dq_actual)
                    vel_arr[i] = dq_actual

                # Apply torque-speed curve (matches Isaac Lab UnitreeActuator)
                if policy_active:
                    tau_arr = clip_torque_speed_curve(tau_arr, vel_arr)

                # Write to MuJoCo ctrl
                for i in range(12):
                    act_idx = int(isaac_to_actuator[i])
                    mj_data.ctrl[act_idx] = tau_arr[i]

                mujoco.mj_step(mj_model, mj_data)

            # Log
            if csv_writer:
                # Extract euler angles
                w, x, y, z = base_quat_wxyz
                # Roll, pitch, yaw from quaternion
                sinr_cosp = 2 * (w * x + y * z)
                cosr_cosp = 1 - 2 * (x * x + y * y)
                roll = math.atan2(sinr_cosp, cosr_cosp)
                sinp = 2 * (w * y - z * x)
                pitch = math.asin(max(-1, min(1, sinp)))
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                csv_writer.writerow([
                    f"{sim_time:.4f}", fsm.state,
                    f"{base_pos[0]:.6f}", f"{base_pos[1]:.6f}", f"{base_pos[2]:.6f}",
                    f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}",
                    *[f"{joint_pos_isaac[i]:.6f}" for i in range(12)],
                    *[f"{last_action[i]:.6f}" for i in range(12)],
                    f"{velocity_commands[0]:.3f}", f"{velocity_commands[1]:.3f}", f"{velocity_commands[2]:.3f}",
                ])

            # Print progress
            if step % 50 == 0:  # Every 1 second
                w, x, y, z = base_quat_wxyz
                sinp = 2 * (w * y - z * x)
                pitch = math.asin(max(-1, min(1, sinp)))
                print(f"  t={sim_time:.2f}s state={fsm.state} "
                      f"z={base_pos[2]:.3f} pitch={math.degrees(pitch):.1f}°")

            # Viewer
            if viewer and viewer.is_running():
                viewer.sync()
            elif viewer and not viewer.is_running():
                break

            # Check for fall
            if policy_active and base_pos[2] < 0.15:
                print(f"  FELL at t={sim_time:.2f}s (base_z={base_pos[2]:.3f})")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        if csv_file:
            csv_file.close()
            print(f"\nCSV saved to: {args.csv}")
        if viewer:
            viewer.close()

    print(f"\nSimulation ended at t={sim_time:.2f}s")
    if policy_active:
        print(f"  Final base height: {base_pos[2]:.3f}m")
        print(f"  Final base pos: x={base_pos[0]:.3f}, y={base_pos[1]:.3f}")
    return sim_time, base_pos[2] if policy_active else 0.0


def main():
    parser = argparse.ArgumentParser(description="Go2 bipedal sim2sim transfer test")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--deploy-yaml", type=str, default=None,
                        help="Path to deploy.yaml for PD gains and joint config")
    parser.add_argument("--scene", type=str,
                        default=os.path.expanduser("~/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene.xml"),
                        help="MuJoCo scene XML path")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--render", action="store_true",
                        help="Launch MuJoCo viewer")
    parser.add_argument("--csv", type=str, default=None,
                        help="Output CSV log path")
    parser.add_argument("--cmd-vx", type=float, default=0.0,
                        help="Forward velocity command")
    parser.add_argument("--cmd-yaw", type=float, default=0.0,
                        help="Yaw rate command")
    args = parser.parse_args()

    # Auto-find deploy.yaml
    if args.deploy_yaml is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        deploy_path = os.path.join(checkpoint_dir, "params", "deploy.yaml")
        if os.path.exists(deploy_path):
            args.deploy_yaml = deploy_path
        else:
            # Try parent dir
            deploy_path = os.path.join(os.path.dirname(checkpoint_dir), "params", "deploy.yaml")
            if os.path.exists(deploy_path):
                args.deploy_yaml = deploy_path

    run_sim2sim(args)


if __name__ == "__main__":
    main()
