#!/usr/bin/env python3
"""Render a velocity-policy checkpoint in Unitree MuJoCo through four command phases.

Phases are forward-x, lateral-y, yaw, and all three together.  The observation,
PD deployment path, Newton joint mapping, scene, and 50 Hz control match the
single-command transfer recorder.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

SIM2SIM = Path(__file__).resolve().parents[1] / "scripts" / "sim2sim"
sys.path.insert(0, str(SIM2SIM))
from sim2sim_v2 import apply_dvi_go2hv_passive_torque, clip_torque, detect_joint_ordering, get_mappings, load_policy, qri


def yaw_from_quat(quat: np.ndarray) -> float:
    # MuJoCo free-joint quaternion convention: [w, x, y, z].
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def bipedal_yaw_from_quat(quat: np.ndarray) -> float:
    """Pitch-invariant heading from horizontal body-Y, matching training."""
    w, x, y, z = quat
    return float(np.arctan2(-2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deploy-yaml", default=None, help="Deployment YAML override, useful for exported policies.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--telemetry", help="Optional NPZ output: rear-leg joint positions and rear-foot ground-contact events.")
    parser.add_argument("--scene", default="/home/horde/go2/unitree_mujoco/unitree_robots/go2/scene_flat.xml")
    parser.add_argument("--cmd-vx", type=float, default=0.7)
    parser.add_argument("--cmd-vy", type=float, default=0.3)
    parser.add_argument("--cmd-wz", type=float, default=0.7)
    parser.add_argument("--phase-duration", type=float, default=2.5)
    parser.add_argument("--zero-command", action="store_true", help="Hold [0, 0, 0] for the full recording instead of command phases.")
    parser.add_argument("--single-command", action="store_true", help="Hold [--cmd-vx, --cmd-vy, --cmd-wz] for the full recording instead of command phases.")
    parser.add_argument("--zero-neg-pos", action="store_true", help="Play zero command, then negative-x, then positive-x.")
    parser.add_argument("--single-duration", type=float, default=10.0, help="Duration in seconds for zero/single-command recordings (default: 10).")
    parser.add_argument("--zero-duration", type=float, default=10.0, help="Zero-command duration for --zero-neg-pos.")
    parser.add_argument("--motion-duration", type=float, default=5.0, help="Duration of each signed-x phase for --zero-neg-pos.")
    parser.add_argument("--pd-gains", choices=("deployment", "native-nominal"), default="deployment",
                        help="Use checkpoint deploy.yaml gains (default) or native Isaac Lab nominal Kp=25, Kd=0.5.")
    parser.add_argument("--settle-duration", type=float, default=0.8,
                        help="Initial MuJoCo high-gain settling duration in seconds; use 0 for a controlled no-settle test.")
    parser.add_argument("--initial-root-z", type=float, default=0.31,
                        help="Initial root height before optional settling (default: 0.31 m).")
    parser.add_argument("--sim-dt", type=float, default=None,
                        help="Override MuJoCo physics timestep while retaining a 20-ms policy interval.")
    parser.add_argument("--integrator", choices=("euler", "implicitfast", "implicit"), default=None,
                        help="Diagnostic only: override the MuJoCo integrator.")
    parser.add_argument("--joint-armature", type=float, default=None,
                        help="Diagnostic only: override all 12 actuated-joint armatures.")
    parser.add_argument("--fixed-action-from-first", action="store_true",
                        help="Diagnostic only: hold the first policy action for all control steps.")
    parser.add_argument("--match-dvi-calf-inertials", action="store_true",
                        help="Diagnostic only: replace each MuJoCo calf inertial with the collapsed USD calf+foot inertial.")
    parser.add_argument("--rear-foot-margin", type=float, default=None,
                        help="Override RL/RR MuJoCo foot geom contact margin in meters for contact-geometry sweeps.")
    parser.add_argument("--ground-margin", type=float, default=None,
                        help="Override MuJoCo floor contact margin for contact-geometry sweeps.")
    parser.add_argument("--rear-foot-radius", type=float, default=None,
                        help="Override RL/RR MuJoCo spherical foot-contact radius in meters for contact-geometry sweeps.")
    parser.add_argument("--rear-foot-friction", type=float, default=None,
                        help="Override RL/RR MuJoCo sliding friction coefficient for contact-parameter sweeps.")
    passive_group = parser.add_mutually_exclusive_group()
    passive_group.add_argument("--no-joint-passive-forces", action="store_true",
                               help="Diagnostic only: set all 12 actuated-joint MuJoCo dof_damping and dof_frictionloss to zero.")
    passive_group.add_argument("--dvi-actuator-passive-forces", action="store_true",
                               help="Disable MuJoCo passive joints and apply exact Go2HV Fs*tanh(qd/Va)+Fd*qd in the replay torque path.")
    parser.add_argument("--hide-overlay", action="store_true", help="Do not render the per-frame command/title overlay; useful for composite videos.")
    parser.add_argument("--no-video", action="store_true", help="Compute phase metrics without rendering an MP4.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    deploy = Path(args.deploy_yaml).resolve() if args.deploy_yaml else checkpoint.parent / "params" / "deploy.yaml"
    if not deploy.exists():
        raise FileNotFoundError(f"Deployment YAML not found: {deploy}; pass --deploy-yaml explicitly.")
    with deploy.open() as f:
        cfg = yaml.safe_load(f)

    model = mujoco.MjModel.from_xml_path(args.scene)
    if args.ground_margin is not None:
        model.geom_margin[model.geom("floor").id] = args.ground_margin
    if args.sim_dt is not None:
        model.opt.timestep = args.sim_dt
    if args.integrator is not None:
        model.opt.integrator = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        }[args.integrator]
    for foot_name in ("RL", "RR"):
        geom_id = model.geom(foot_name).id
        if args.rear_foot_margin is not None:
            model.geom_margin[geom_id] = args.rear_foot_margin
        if args.rear_foot_radius is not None:
            model.geom_size[geom_id, 0] = args.rear_foot_radius
        if args.rear_foot_friction is not None:
            model.geom_friction[geom_id, 0] = args.rear_foot_friction
    ordering = detect_joint_ordering(str(deploy))
    iqpos, iqvel, iact, dpos, _, _ = get_mappings(ordering)
    actuated_dofs = np.asarray(iqvel, dtype=int)
    if args.joint_armature is not None:
        model.dof_armature[actuated_dofs] = args.joint_armature
    if args.match_dvi_calf_inertials:
        # Exact parallel-axis collapse of the stock USD calf (0.154 kg) and
        # fixed foot (0.040 kg at z=-0.213 m), expressed in the calf frame.
        calf_mass = 0.154
        foot_mass = 0.040
        calf_com_template = np.array([0.00548, -0.000975, -0.115])
        calf_diag = np.array([0.001080271, 0.0011000754, 0.000032553424])
        calf_quat_template = np.array([0.999887, 0.003973435, -0.008161128, -0.011986799])
        foot_com = np.array([0.0, 0.0, -0.213])
        foot_inertia = np.diag([9.6e-6, 9.6e-6, 9.6e-6])
        for side in ("FL", "FR", "RL", "RR"):
            body_id = model.body(f"{side}_calf").id
            calf_com = calf_com_template.copy()
            calf_quat = calf_quat_template.copy()
            if side in ("FR", "RR"):
                calf_com[1] *= -1.0
                calf_quat[[1, 3]] *= -1.0
            calf_rot = np.empty(9)
            mujoco.mju_quat2Mat(calf_rot, calf_quat)
            calf_rot = calf_rot.reshape(3, 3)
            calf_inertia = calf_rot @ np.diag(calf_diag) @ calf_rot.T
            total_mass = calf_mass + foot_mass
            total_com = (calf_mass * calf_com + foot_mass * foot_com) / total_mass
            total_inertia = np.zeros((3, 3))
            for mass, com, inertia in (
                (calf_mass, calf_com, calf_inertia),
                (foot_mass, foot_com, foot_inertia),
            ):
                offset = com - total_com
                total_inertia += inertia + mass * (
                    np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
                )
            eigvals, eigvecs = np.linalg.eigh(total_inertia)
            if np.linalg.det(eigvecs) < 0.0:
                eigvecs[:, 0] *= -1.0
            inertial_quat = np.empty(4)
            mujoco.mju_mat2Quat(inertial_quat, eigvecs.reshape(-1))
            model.body_mass[body_id] = total_mass
            model.body_ipos[body_id] = total_com
            model.body_inertia[body_id] = eigvals
            model.body_iquat[body_id] = inertial_quat
    if args.no_joint_passive_forces or args.dvi_actuator_passive_forces:
        for joint_dof in np.asarray(iqvel, dtype=int):
            model.dof_damping[joint_dof] = 0.0
            model.dof_frictionloss[joint_dof] = 0.0
    data = mujoco.MjData(model)
    if args.match_dvi_calf_inertials:
        mujoco.mj_setConst(model, data)
    dt = model.opt.timestep
    substeps = round(0.02 / dt)
    assert abs(substeps * dt - 0.02) < 1e-6
    dpos = np.asarray(cfg["default_joint_pos"], dtype=float)
    kp = np.asarray(cfg["stiffness"], dtype=float)
    kd = np.asarray(cfg["damping"], dtype=float)
    if args.pd_gains == "native-nominal":
        kp = np.full(12, 25.0)
        kd = np.full(12, 0.5)

    policy, is_jit = load_policy(str(checkpoint))
    if is_jit and hasattr(policy, "obs_dim"):
        actor_obs_dim = policy.obs_dim
    else:
        actor_obs_dim = next(policy.parameters()).shape[1] if is_jit else policy[0].in_features
    if (actor_obs_dim - 15) % 30:
        raise ValueError(f"Unsupported actor obs dimension {actor_obs_dim}")
    hist = (actor_obs_dim - 15) // 30
    if sum((args.zero_command, args.single_command, args.zero_neg_pos)) > 1:
        raise ValueError("--zero-command, --single-command, and --zero-neg-pos are mutually exclusive")
    phases = ([
        ("zero command", np.array([0.0, 0.0, 0.0])),
    ] if args.zero_command else [
        ("single command", np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz])),
    ] if args.single_command else [
        ("zero command", np.array([0.0, 0.0, 0.0])),
        ("negative x", np.array([-abs(args.cmd_vx), 0.0, 0.0])),
        ("positive x", np.array([abs(args.cmd_vx), 0.0, 0.0])),
    ] if args.zero_neg_pos else [
        ("forward x", np.array([args.cmd_vx, 0.0, 0.0])),
        ("lateral y", np.array([0.0, args.cmd_vy, 0.0])),
        ("yaw", np.array([0.0, 0.0, args.cmd_wz])),
        ("combined", np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz])),
    ])
    joint_damping = np.asarray(model.dof_damping[np.asarray(iqvel, dtype=int)])
    joint_frictionloss = np.asarray(model.dof_frictionloss[np.asarray(iqvel, dtype=int)])
    passive_mode = "dvi-actuator" if args.dvi_actuator_passive_forces else "off" if args.no_joint_passive_forces else "mujoco-model"
    armature = np.asarray(model.dof_armature[actuated_dofs])
    calf_masses = [model.body_mass[model.body(f"{side}_calf").id] for side in ("FL", "FR", "RL", "RR")]
    print(f"checkpoint={checkpoint.name}; ordering={ordering}; obs={actor_obs_dim}; history={hist}; ctrl=50Hz; pd={args.pd_gains}; settle={args.settle_duration:.3f}s; integrator={mujoco.mjtIntegrator(model.opt.integrator).name}; armature={armature.min():.6g}..{armature.max():.6g}; calf_mass={min(calf_masses):.6g}..{max(calf_masses):.6g}; passive_joint_forces={passive_mode} (damping={joint_damping.min():.3f}..{joint_damping.max():.3f}, frictionloss={joint_frictionloss.min():.3f}..{joint_frictionloss.max():.3f}); foot=(r={model.geom_size[model.geom('RL').id, 0]:.3f}, margin={model.geom_margin[model.geom('RL').id]:.3f}, mu={model.geom_friction[model.geom('RL').id, 0]:.2f})")
    print("phases=" + "; ".join(f"{n}: {c.tolist()}" for n, c in phases))

    data.qpos[:3] = [0, 0, args.initial_root_z]
    data.qpos[3:7] = [1, 0, 0, 0]
    for i in range(12):
        data.qpos[int(iqpos[i])] = dpos[i]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(round(args.settle_duration / dt)):
        for i in range(12):
            q, dq = data.qpos[int(iqpos[i])], data.qvel[int(iqvel[i])]
            data.ctrl[int(iact[i])] = 60.0 * (dpos[i] - q) - 5.0 * dq
        mujoco.mj_step(model, data)

    ang, grav, jpos, jvel = (deque(maxlen=hist) for _ in range(4))
    pg = qri(data.qpos[3:7], np.array([0.0, 0.0, -1.0]))
    av = data.qvel[3:6].copy()
    for _ in range(hist):
        ang.append(av * 0.2); grav.append(pg.copy()); jpos.append(np.zeros(12)); jvel.append(np.zeros(12))
    last_action = np.zeros(12)

    renderer = None if args.no_video else mujoco.Renderer(model, height=480, width=640)
    writer = None if args.no_video else imageio.get_writer(args.output, fps=25, quality=8)
    rear_joint_indices = [i for i, name in enumerate(model.joint(i).name for i in range(model.njnt)) if name in {"RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"}]
    rear_joint_names = [model.joint(i).name for i in rear_joint_indices]
    if len(rear_joint_indices) != 6:
        raise RuntimeError(f"Expected six rear-leg joints, got {rear_joint_names}")
    rear_geom_ids = {model.geom("RL").id: 0, model.geom("RR").id: 1}
    rear_foot_body_ids = [int(model.geom_bodyid[model.geom(name).id]) for name in ("RL", "RR")]
    telemetry_time, telemetry_joint_pos, telemetry_contact = [], [], []
    telemetry_obs, telemetry_action, telemetry_all_joint_pos, telemetry_joint_vel = [], [], [], []
    telemetry_joint_target, telemetry_computed_torque, telemetry_applied_torque = [], [], []
    telemetry_root_pos, telemetry_root_quat, telemetry_projected_gravity = [], [], []
    telemetry_root_lin_vel, telemetry_root_ang_vel = [], []
    telemetry_rear_foot_pos, telemetry_rear_foot_quat, telemetry_rear_contact_force = [], [], []
    track_body_id = model.body("base_link").id
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    if args.single_duration <= 0.0:
        raise ValueError("--single-duration must be positive")
    if args.zero_neg_pos:
        phase_step_counts = [round(args.zero_duration / 0.02), round(args.motion_duration / 0.02), round(args.motion_duration / 0.02)]
    else:
        common_steps = round(args.phase_duration / 0.02) if not (args.zero_command or args.single_command) else round(args.single_duration / 0.02)
        phase_step_counts = [common_steps] * len(phases)
    phase_boundaries = np.cumsum(phase_step_counts)
    phase_start = None
    phase_velocity_samples = []
    fixed_action = None
    try:
        for step in range(int(phase_boundaries[-1])):
            phase_idx = min(int(np.searchsorted(phase_boundaries, step, side="right")), len(phases) - 1)
            phase_name, cmd = phases[phase_idx]
            phase_start_step = 0 if phase_idx == 0 else int(phase_boundaries[phase_idx - 1])
            if step == phase_start_step:
                phase_start = (data.qpos[0], data.qpos[1], bipedal_yaw_from_quat(data.qpos[3:7]))
                phase_velocity_samples = []
                print(f"PHASE_START {phase_idx + 1}/4 {phase_name} t={step*.02:.2f}s cmd={cmd.tolist()}")
            quat = data.qpos[3:7]
            pg = qri(quat, np.array([0.0, 0.0, -1.0]))
            av = data.qvel[3:6].copy()
            # MuJoCo free-joint linear velocity is in world coordinates. Rotate it
            # into the current base yaw frame for command-tracking statistics.
            yaw = bipedal_yaw_from_quat(quat)
            c, s = np.cos(yaw), np.sin(yaw)
            vx_world, vy_world = data.qvel[0], data.qvel[1]
            phase_velocity_samples.append((c * vx_world + s * vy_world, -s * vx_world + c * vy_world, data.qvel[5]))
            jp = np.array([data.qpos[int(iqpos[i])] for i in range(12)])
            jv = np.array([data.qvel[int(iqvel[i])] for i in range(12)])
            ang.append(av * 0.2); grav.append(pg); jpos.append(jp - dpos); jvel.append(jv * 0.05)
            obs = np.concatenate([*ang, *grav, cmd, *jpos, *jvel, last_action]).clip(-100, 100)
            if fixed_action is None:
                with torch.no_grad():
                    out = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
                action = out.numpy() if is_jit else out.detach().numpy()
                if args.fixed_action_from_first:
                    fixed_action = action.copy()
            elif args.fixed_action_from_first:
                action = fixed_action.copy()
            else:
                with torch.no_grad():
                    out = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
                action = out.numpy() if is_jit else out.detach().numpy()
            last_action = action.copy()
            target = dpos + action * 0.25
            for _ in range(substeps):
                tau = np.empty(12); vel = np.empty(12)
                for i in range(12):
                    q, dq = data.qpos[int(iqpos[i])], data.qvel[int(iqvel[i])]
                    tau[i] = kp[i] * (target[i] - q) - kd[i] * dq
                    vel[i] = dq
                tau = clip_torque(tau, vel)
                if args.dvi_actuator_passive_forces:
                    tau = apply_dvi_go2hv_passive_torque(tau, vel)
                for i in range(12): data.ctrl[int(iact[i])] = tau[i]
                mujoco.mj_step(model, data)
            if step + 1 == int(phase_boundaries[phase_idx]):
                x0, y0, yaw0 = phase_start
                mean_vel = np.mean(phase_velocity_samples, axis=0)
                mae = np.abs(mean_vel - cmd)
                print(f"PHASE_END {phase_idx + 1}/4 {phase_name}: dx={data.qpos[0]-x0:.3f} dy={data.qpos[1]-y0:.3f} dyaw={bipedal_yaw_from_quat(data.qpos[3:7])-yaw0:.3f} z={data.qpos[2]:.3f}")
                print(f"TRACKING {phase_name}: mean_body=[{mean_vel[0]:.3f}, {mean_vel[1]:.3f}, {mean_vel[2]:.3f}] cmd=[{cmd[0]:.3f}, {cmd[1]:.3f}, {cmd[2]:.3f}] abs_error=[{mae[0]:.3f}, {mae[1]:.3f}, {mae[2]:.3f}]")
            if args.telemetry:
                contact = np.zeros(2, dtype=np.uint8)
                contact_force = np.zeros((2, 6), dtype=float)
                for ci in range(data.ncon):
                    con = data.contact[ci]
                    for geom_id, foot_idx in rear_geom_ids.items():
                        other = con.geom2 if con.geom1 == geom_id else con.geom1 if con.geom2 == geom_id else None
                        if other is not None and model.geom_bodyid[other] == 0:
                            contact[foot_idx] = 1
                            wrench = np.zeros(6, dtype=float)
                            mujoco.mj_contactForce(model, data, ci, wrench)
                            contact_force[foot_idx] += wrench
                telemetry_time.append(step * 0.02)
                telemetry_obs.append(obs.copy())
                telemetry_action.append(action.copy())
                telemetry_all_joint_pos.append(np.asarray([data.qpos[int(iqpos[i])] for i in range(12)]))
                telemetry_joint_vel.append(np.asarray([data.qvel[int(iqvel[i])] for i in range(12)]))
                telemetry_joint_target.append(target.copy())
                telemetry_computed_torque.append((kp * (target - jp) - kd * jv).copy())
                telemetry_applied_torque.append(tau.copy())
                telemetry_joint_pos.append(np.asarray([float(data.qpos[model.joint(joint_id).qposadr]) for joint_id in rear_joint_indices]))
                telemetry_contact.append(contact)
                telemetry_root_pos.append(data.qpos[:3].copy())
                telemetry_root_quat.append(data.qpos[3:7].copy())
                telemetry_projected_gravity.append(qri(data.qpos[3:7], np.array([0.0, 0.0, -1.0])))
                telemetry_root_lin_vel.append(data.qvel[:3].copy())
                telemetry_root_ang_vel.append(data.qvel[3:6].copy())
                telemetry_rear_foot_pos.append(np.asarray([data.xpos[body_id].copy() for body_id in rear_foot_body_ids]))
                telemetry_rear_foot_quat.append(np.asarray([data.xquat[body_id].copy() for body_id in rear_foot_body_ids]))
                telemetry_rear_contact_force.append(contact_force)
            if not args.no_video and step % 2 == 0:
                cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = track_body_id; cam.distance = 1.8; cam.azimuth = 135; cam.elevation = -20
                renderer.update_scene(data, camera=cam)
                frame = Image.fromarray(renderer.render().copy()).convert("RGB")
                if not args.hide_overlay:
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, frame.width, 57), fill=(15, 15, 15))
                    label = "Zero command" if args.zero_command else ("Single command" if args.single_command else f"Phase {phase_idx + 1}/{len(phases)}: {phase_name}")
                    draw.text((12, 7), label, font=font, fill="white")
                    draw.text((12, 32), f"cmd = [{cmd[0]:.1f}, {cmd[1]:.1f}, {cmd[2]:.1f}]  |  t = {step*.02:.1f}s", font=small, fill=(225, 225, 225))
                writer.append_data(np.asarray(frame))
            if data.qpos[2] < 0.15:
                print(f"FELL t={step*.02:.2f}s z={data.qpos[2]:.3f}")
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    if args.telemetry:
        telemetry_path = Path(args.telemetry)
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            telemetry_path, time=np.asarray(telemetry_time), joint_pos=np.asarray(telemetry_joint_pos),
            contact=np.asarray(telemetry_contact), joint_names=np.asarray(rear_joint_names),
            contact_names=np.asarray(["RL_foot", "RR_foot"]),
            root_pos=np.asarray(telemetry_root_pos), root_quat_wxyz=np.asarray(telemetry_root_quat),
            projected_gravity_b=np.asarray(telemetry_projected_gravity),
            rear_foot_pos=np.asarray(telemetry_rear_foot_pos),
            rear_foot_quat_wxyz=np.asarray(telemetry_rear_foot_quat),
            rear_contact_wrench=np.asarray(telemetry_rear_contact_force),
            rear_foot_orientation_body_names=np.asarray([model.body(i).name for i in rear_foot_body_ids]),
            obs=np.asarray(telemetry_obs), action=np.asarray(telemetry_action),
            all_joint_pos=np.asarray(telemetry_all_joint_pos), joint_vel=np.asarray(telemetry_joint_vel),
            joint_pos_target=np.asarray(telemetry_joint_target), computed_torque=np.asarray(telemetry_computed_torque),
            applied_torque=np.asarray(telemetry_applied_torque),
            all_joint_names=np.asarray([model.joint(int(model.dof_jntid[int(iqvel[i])])).name for i in range(12)]),
            root_lin_vel_w=np.asarray(telemetry_root_lin_vel), root_ang_vel_w=np.asarray(telemetry_root_ang_vel),
        )
    print(f"final x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} yaw={bipedal_yaw_from_quat(data.qpos[3:7]):.3f} z={data.qpos[2]:.3f}; video={args.output}")

if __name__ == "__main__":
    main()
