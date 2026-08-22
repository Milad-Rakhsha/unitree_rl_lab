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
    parser.add_argument("--pd-gains", choices=("deployment", "native-nominal"), default="deployment",
                        help="Use checkpoint deploy.yaml gains (default) or native Isaac Lab nominal Kp=25, Kd=0.5.")
    parser.add_argument("--settle-duration", type=float, default=0.8,
                        help="Initial MuJoCo high-gain settling duration in seconds; use 0 for a controlled no-settle test.")
    parser.add_argument("--rear-foot-margin", type=float, default=None,
                        help="Override RL/RR MuJoCo foot geom contact margin in meters for contact-geometry sweeps.")
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
    if args.no_joint_passive_forces or args.dvi_actuator_passive_forces:
        for joint_dof in np.asarray(iqvel, dtype=int):
            model.dof_damping[joint_dof] = 0.0
            model.dof_frictionloss[joint_dof] = 0.0
    data = mujoco.MjData(model)
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
    phases = ([
        ("zero command", np.array([0.0, 0.0, 0.0])),
    ] if args.zero_command else [
        ("forward x", np.array([args.cmd_vx, 0.0, 0.0])),
        ("lateral y", np.array([0.0, args.cmd_vy, 0.0])),
        ("yaw", np.array([0.0, 0.0, args.cmd_wz])),
        ("combined", np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz])),
    ])
    joint_damping = np.asarray(model.dof_damping[np.asarray(iqvel, dtype=int)])
    joint_frictionloss = np.asarray(model.dof_frictionloss[np.asarray(iqvel, dtype=int)])
    passive_mode = "dvi-actuator" if args.dvi_actuator_passive_forces else "off" if args.no_joint_passive_forces else "mujoco-model"
    print(f"checkpoint={checkpoint.name}; ordering={ordering}; obs={actor_obs_dim}; history={hist}; ctrl=50Hz; pd={args.pd_gains}; settle={args.settle_duration:.3f}s; passive_joint_forces={passive_mode} (damping={joint_damping.min():.3f}..{joint_damping.max():.3f}, frictionloss={joint_frictionloss.min():.3f}..{joint_frictionloss.max():.3f}); foot=(r={model.geom_size[model.geom('RL').id, 0]:.3f}, margin={model.geom_margin[model.geom('RL').id]:.3f}, mu={model.geom_friction[model.geom('RL').id, 0]:.2f})")
    print("phases=" + "; ".join(f"{n}: {c.tolist()}" for n, c in phases))

    data.qpos[:3] = [0, 0, 0.31]
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
    telemetry_time, telemetry_joint_pos, telemetry_contact = [], [], []
    track_body_id = model.body("base_link").id
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    phase_steps = (round(args.phase_duration / 0.02) if not args.zero_command else round(10.0 / 0.02))
    phase_start = None
    phase_velocity_samples = []
    try:
        for step in range(phase_steps * len(phases)):
            phase_idx = min(step // phase_steps, len(phases) - 1)
            phase_name, cmd = phases[phase_idx]
            if step % phase_steps == 0:
                phase_start = (data.qpos[0], data.qpos[1], yaw_from_quat(data.qpos[3:7]))
                phase_velocity_samples = []
                print(f"PHASE_START {phase_idx + 1}/4 {phase_name} t={step*.02:.2f}s cmd={cmd.tolist()}")
            quat = data.qpos[3:7]
            pg = qri(quat, np.array([0.0, 0.0, -1.0]))
            av = data.qvel[3:6].copy()
            # MuJoCo free-joint linear velocity is in world coordinates. Rotate it
            # into the current base yaw frame for command-tracking statistics.
            yaw = yaw_from_quat(quat)
            c, s = np.cos(yaw), np.sin(yaw)
            vx_world, vy_world = data.qvel[0], data.qvel[1]
            phase_velocity_samples.append((c * vx_world + s * vy_world, -s * vx_world + c * vy_world, data.qvel[5]))
            jp = np.array([data.qpos[int(iqpos[i])] for i in range(12)])
            jv = np.array([data.qvel[int(iqvel[i])] for i in range(12)])
            ang.append(av * 0.2); grav.append(pg); jpos.append(jp - dpos); jvel.append(jv * 0.05)
            obs = np.concatenate([*ang, *grav, cmd, *jpos, *jvel, last_action]).clip(-100, 100)
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
            if (step + 1) % phase_steps == 0:
                x0, y0, yaw0 = phase_start
                mean_vel = np.mean(phase_velocity_samples, axis=0)
                mae = np.abs(mean_vel - cmd)
                print(f"PHASE_END {phase_idx + 1}/4 {phase_name}: dx={data.qpos[0]-x0:.3f} dy={data.qpos[1]-y0:.3f} dyaw={yaw_from_quat(data.qpos[3:7])-yaw0:.3f} z={data.qpos[2]:.3f}")
                print(f"TRACKING {phase_name}: mean_body=[{mean_vel[0]:.3f}, {mean_vel[1]:.3f}, {mean_vel[2]:.3f}] cmd=[{cmd[0]:.3f}, {cmd[1]:.3f}, {cmd[2]:.3f}] abs_error=[{mae[0]:.3f}, {mae[1]:.3f}, {mae[2]:.3f}]")
            if args.telemetry:
                contact = np.zeros(2, dtype=np.uint8)
                for ci in range(data.ncon):
                    con = data.contact[ci]
                    for geom_id, foot_idx in rear_geom_ids.items():
                        other = con.geom2 if con.geom1 == geom_id else con.geom1 if con.geom2 == geom_id else None
                        if other is not None and model.geom_bodyid[other] == 0:
                            contact[foot_idx] = 1
                telemetry_time.append(step * 0.02)
                telemetry_joint_pos.append(np.asarray([float(data.qpos[model.joint(joint_id).qposadr]) for joint_id in rear_joint_indices]))
                telemetry_contact.append(contact)
            if not args.no_video and step % 2 == 0:
                cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = track_body_id; cam.distance = 1.8; cam.azimuth = 135; cam.elevation = -20
                renderer.update_scene(data, camera=cam)
                frame = Image.fromarray(renderer.render().copy()).convert("RGB")
                if not args.hide_overlay:
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, frame.width, 57), fill=(15, 15, 15))
                    draw.text((12, 7), ("Zero command" if args.zero_command else f"Phase {phase_idx + 1}/4: {phase_name}"), font=font, fill="white")
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
        )
    print(f"final x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} yaw={yaw_from_quat(data.qpos[3:7]):.3f} z={data.qpos[2]:.3f}; video={args.output}")

if __name__ == "__main__":
    main()
