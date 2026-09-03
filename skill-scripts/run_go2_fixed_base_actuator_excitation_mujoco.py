#!/usr/bin/env python3
"""Phase-1 fixed-base, zero-gravity, collision-free Go2 actuator excitation in MuJoCo.

This deliberately loads the robot XML without its free joint, so the base is
structurally fixed rather than numerically re-clamped after each step.
"""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
import sys
import mujoco
import numpy as np
import yaml

SIM2SIM = Path(__file__).resolve().parents[1] / "scripts" / "sim2sim"
sys.path.insert(0, str(SIM2SIM))
from sim2sim_v2 import apply_dvi_go2hv_passive_torque, clip_torque, detect_joint_ordering, get_mappings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--deploy-yaml", type=Path, required=True)
    p.add_argument("--robot-xml", type=Path, default=Path("/home/horde/go2/unitree_mujoco/unitree_robots/go2/go2.xml"))
    p.add_argument("--telemetry", type=Path, required=True)
    p.add_argument("--amplitude", type=float, default=0.12)
    p.add_argument("--rest", type=float, default=0.20)
    p.add_argument("--hold", type=float, default=0.40)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--control-dt", type=float, default=0.02, help="Held-target interval; must match native environment control timing.")
    p.add_argument("--joint-armature", type=float, default=0.0)
    p.add_argument("--free-root", action="store_true", help="Phase-2 diagnostic: retain floating-base dynamics instead of root projection.")
    p.add_argument("--gravity", choices=("off", "on"), default="off", help="Gravity condition; collisions remain disabled.")
    p.add_argument("--excitation-joints", type=int, default=12, help="Number of leading joints to excite sequentially.")
    p.add_argument("--initial-root-z", type=float, default=0.445, help="Initial free-root height; must match the native diagnostic.")
    p.add_argument("--initial-root-quat-wxyz", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), help="Initial root orientation in MuJoCo wxyz convention.")
    p.add_argument("--camera-distance", type=float, default=2.75, help="Renderer camera distance is recorded for reproducibility only.")
    p.add_argument("--log-post-control-state", action="store_true", help="Log q, qd, and recomputed torque after each held control interval. Use with DVI, whose environment telemetry is sampled after env.step().")
    p.add_argument("--direct-torque-excitation", action="store_true", help="Bypass Go2HV PD, saturation, and passive terms; inject the prescribed joint torque waveform directly.")
    p.add_argument("--direct-torque-amplitude", type=float, default=1.0)
    p.add_argument("--controlled-contact", choices=("off", "on"), help="Phase-3 isolated rear-foot landing with ground contact disabled or enabled.")
    p.add_argument("--controlled-contact-side", choices=("left", "right"), default="left")
    p.add_argument("--controlled-contact-downward-speed", type=float, default=0.5)
    p.add_argument("--controlled-contact-steps", type=int, default=80)
    p.add_argument("--upright-perturbation", action="store_true", help="Fixed-root deterministic upright bipedal perturbation profile matching the native Phase-1b diagnostic.")
    p.add_argument("--perturbation-zero-std", type=float, default=0.15)
    p.add_argument("--perturbation-walk-std", type=float, default=0.17)
    p.add_argument("--perturbation-half-duration", type=float, default=4.0)
    args = p.parse_args()
    if min(args.amplitude, args.direct_torque_amplitude, args.rest, args.hold, args.dt, args.control_dt) <= 0:
        raise ValueError("amplitudes, rest, hold, dt, and control_dt must be positive")
    if not 1 <= args.excitation_joints <= 12:
        raise ValueError("--excitation-joints must be in [1, 12]")
    if args.controlled_contact and not args.free_root:
        raise ValueError("--controlled-contact requires --free-root")
    if args.controlled_contact_downward_speed <= 0.0 or args.controlled_contact_steps < 1:
        raise ValueError("controlled-contact speed and steps must be positive")
    direct_mode = args.direct_torque_excitation or bool(args.controlled_contact)
    substeps = round(args.control_dt / args.dt)
    if abs(substeps * args.dt - args.control_dt) > 1e-10:
        raise ValueError("control-dt must be an integer multiple of dt")

    # Keep the production floating-base XML and all production coordinate,
    # inertia, and actuator semantics intact.  We project the root state back
    # after each physics step; this is deliberately an actuator diagnostic,
    # not an equality-constraint or floating-base dynamics experiment.
    args.robot_xml = args.robot_xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    if model.nq != 19 or model.nv != 18 or model.nu != 12:
        raise RuntimeError(f"Unexpected production model dimensions nq={model.nq} nv={model.nv} nu={model.nu}")
    model.opt.gravity[:] = (0.0, 0.0, -9.81) if args.gravity == "on" else (0.0, 0.0, 0.0)
    model.opt.timestep = args.dt
    model.dof_armature[6:] = args.joint_armature
    # The training UnitreeActuatorCfg_Go2HV supplies Fs/Fd itself as the
    # smooth effort term -0.2*tanh(qd/0.01)-0.1*qd.  Disable MuJoCo's XML
    # Coulomb/viscous joint forces so this exact actuator contribution is
    # applied once, rather than duplicated by dof_frictionloss/dof_damping.
    model.dof_damping[6:] = 0.0
    model.dof_frictionloss[6:] = 0.0
    if args.controlled_contact == "on":
        robot_geom = model.geom_bodyid != 0
        world_geom = ~robot_geom
        model.geom_contype[robot_geom] = 1
        model.geom_conaffinity[robot_geom] = 0
        model.geom_contype[world_geom] = 0
        model.geom_conaffinity[world_geom] = 1
    else:
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)

    cfg = yaml.safe_load(args.deploy_yaml.read_text())
    ordering = detect_joint_ordering(str(args.deploy_yaml))
    iqpos, iqvel, iact, dpos, _, _ = get_mappings(ordering)
    dpos = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
    # These are the per-joint gains frozen from the training Go2HV
    # UnitreeActuator configuration after its deterministic startup draw.
    # The controller below exactly mirrors UnitreeActuator.compute(): PD,
    # Go2HV torque-speed clipping, then the smooth Fs/Fd effort term.
    kp = np.asarray(cfg["stiffness"], dtype=np.float64)
    kd = np.asarray(cfg["damping"], dtype=np.float64)
    if kp.shape != (12,) or kd.shape != (12,):
        raise RuntimeError(f"Expected 12 Go2HV gains, got kp={kp.shape}, kd={kd.shape}")
    # Match the production runner's root initialization and preserve it by
    # exact projection after every physics step.
    contact_root = {
        "left": ((0.0, 0.0, 0.5735), (0.67284785, 0.00079349, -0.73183698, 0.10811924)),
        "right": ((0.0, 0.0, 0.5691), (0.66248799, 0.11762024, -0.73949344, -0.02060549)),
    }
    if args.controlled_contact:
        root_pos, root_quat = contact_root[args.controlled_contact_side]
        data.qpos[:3] = root_pos
        data.qpos[3:7] = root_quat
    else:
        data.qpos[:3] = [0.0, 0.0, args.initial_root_z]
        data.qpos[3:7] = np.asarray(args.initial_root_quat_wxyz, dtype=np.float64)
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    initial_joint_q = (np.asarray([0.101630, 0.734117, -1.453078, -0.090235, 0.764567, -1.446473, 0.137010, 2.307323, -1.453031, -0.235372, 2.290217, -1.482530], dtype=np.float64)
                       if args.upright_perturbation or args.controlled_contact else dpos)
    for i in range(12):
        data.qpos[int(iqpos[i])] = initial_joint_q[i]
    data.qvel[:] = 0.0
    if args.controlled_contact:
        data.qvel[2] = -args.controlled_contact_downward_speed
    mujoco.mj_forward(model, data)
    root_qpos0 = data.qpos[:7].copy()
    root_qvel0 = np.zeros(6, dtype=np.float64)

    n_rest = round(args.rest / args.control_dt)
    n_hold = round(args.hold / args.control_dt)
    if n_rest < 1 or n_hold < 1:
        raise ValueError("rest and hold must span at least one physics step")
    segment = 2 * n_rest + 2 * n_hold
    total = args.excitation_joints * segment
    if args.upright_perturbation:
        total = round(2.0 * args.perturbation_half_duration / args.control_dt)
        if total < 2:
            raise ValueError("upright perturbation must span at least two control intervals")
        upright = np.asarray([0.101630, 0.734117, -1.453078, -0.090235, 0.764567, -1.446473, 0.137010, 2.307323, -1.453031, -0.235372, 2.290217, -1.482530], dtype=np.float64)
        perturbation_signs = np.asarray([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1], dtype=np.float64)
    elif args.controlled_contact:
        total = args.controlled_contact_steps
    t, target_log, q_log, qd_log, raw_tau_log, tau_log, active, root_log, root_vel_log = [], [], [], [], [], [], [], [], []
    contact_log, rear_contact_force_log, rear_foot_pos_log = [], [], []
    rear_geom_ids = {model.geom("RL").id: 0, model.geom("RR").id: 1}
    for step in range(total):
        if args.controlled_contact:
            target = np.zeros(12, dtype=np.float64)
            joint = -1
        elif args.upright_perturbation:
            time_s = step * args.control_dt
            std = args.perturbation_zero_std if time_s < args.perturbation_half_duration else args.perturbation_walk_std
            # Same zero-at-start multi-joint waveform as native DVI. This
            # prevents an artificial t=0 pose jump that could dislocate a
            # shoulder/hip visually before the comparison even begins.
            target = upright + std * np.sqrt(2.0) * np.sin(2.0 * np.pi * 0.75 * time_s) * perturbation_signs
            joint = -1
        else:
            joint = step // segment
            local = step % segment
            offset = 0.0 if local < n_rest or local >= n_rest + 2 * n_hold else ((args.direct_torque_amplitude if direct_mode else args.amplitude) if local < n_rest + n_hold else -(args.direct_torque_amplitude if direct_mode else args.amplitude))
            target = np.zeros(12, dtype=np.float64) if direct_mode else dpos.copy()
            target[joint] += offset
        q = np.asarray([data.qpos[int(iqpos[i])] for i in range(12)])
        qd = np.asarray([data.qvel[int(iqvel[i])] for i in range(12)])
        raw_tau = target.copy() if direct_mode else kp * (target - q) - kd * qd
        tau = target.copy() if direct_mode else apply_dvi_go2hv_passive_torque(clip_torque(raw_tau, qd), qd)
        # Default preserves the original pre-interval trace convention.  DVI
        # environment telemetry is necessarily sampled after env.step(), so a
        # matched solver comparison uses --log-post-control-state below.
        if not args.log_post_control_state:
            t.append(step * args.control_dt); target_log.append(target); q_log.append(q); qd_log.append(qd)
            raw_tau_log.append(raw_tau); tau_log.append(tau); active.append(joint); root_log.append(data.qpos[:7].copy()); root_vel_log.append(data.qvel[:6].copy())
        # Recompute PD/torque-speed/passive effort at every physics step,
        # exactly as the validated production sim2sim runner does. Holding a
        # torque computed 20 ms earlier is a different controller.
        for _ in range(substeps):
            q_sub = np.asarray([data.qpos[int(iqpos[i])] for i in range(12)])
            qd_sub = np.asarray([data.qvel[int(iqvel[i])] for i in range(12)])
            raw_sub = target if direct_mode else kp * (target - q_sub) - kd * qd_sub
            tau_sub = target if direct_mode else apply_dvi_go2hv_passive_torque(clip_torque(raw_sub, qd_sub), qd_sub)
            for i in range(12):
                data.ctrl[int(iact[i])] = tau_sub[i]
            mujoco.mj_step(model, data)
            if not args.free_root:
                data.qpos[:7] = root_qpos0
                data.qvel[:6] = root_qvel0
                mujoco.mj_forward(model, data)
        if args.log_post_control_state:
            q = np.asarray([data.qpos[int(iqpos[i])] for i in range(12)])
            qd = np.asarray([data.qvel[int(iqvel[i])] for i in range(12)])
            raw_tau = target.copy() if direct_mode else kp * (target - q) - kd * qd
            tau = target.copy() if direct_mode else apply_dvi_go2hv_passive_torque(clip_torque(raw_tau, qd), qd)
            t.append(step * args.control_dt); target_log.append(target); q_log.append(q); qd_log.append(qd)
            raw_tau_log.append(raw_tau); tau_log.append(tau); active.append(joint); root_log.append(data.qpos[:7].copy()); root_vel_log.append(data.qvel[:6].copy())
        contact = np.zeros(2, dtype=np.uint8)
        rear_contact_force = np.zeros((2, 3), dtype=np.float64)
        for contact_index in range(data.ncon):
            con = data.contact[contact_index]
            for geom_id, foot_index in rear_geom_ids.items():
                other = con.geom2 if con.geom1 == geom_id else con.geom1 if con.geom2 == geom_id else -1
                if other >= 0 and model.geom_bodyid[other] == 0:
                    wrench = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, data, contact_index, wrench)
                    frame = np.asarray(con.frame).reshape(3, 3)
                    force = frame.T @ wrench[:3]
                    if con.geom1 == geom_id:
                        force = -force
                    contact[foot_index] = 1
                    rear_contact_force[foot_index] += force
        contact_log.append(contact)
        rear_contact_force_log.append(rear_contact_force)
        rear_foot_pos_log.append(np.asarray([data.geom_xpos[geom_id].copy() for geom_id in rear_geom_ids]))

    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    joint_names = [model.joint(int(model.dof_jntid[int(iqvel[i])])).name for i in range(12)]
    np.savez_compressed(args.telemetry, time=np.asarray(t), target=np.asarray(target_log), q=np.asarray(q_log), qd=np.asarray(qd_log), raw_tau=np.asarray(raw_tau_log), applied_tau=np.asarray(tau_log), active_joint=np.asarray(active), root_qpos=np.asarray(root_log), root_qvel=np.asarray(root_vel_log), joint_names=np.asarray(joint_names), kp=kp, kd=kd, default_q=dpos, dt=np.asarray(args.dt), control_dt=np.asarray(args.control_dt), amplitude=np.asarray(args.amplitude), rest=np.asarray(args.rest), hold=np.asarray(args.hold), free_root=np.asarray(args.free_root), gravity=np.asarray(args.gravity), contact=np.asarray(contact_log), rear_contact_force_w=np.asarray(rear_contact_force_log), rear_foot_pos=np.asarray(rear_foot_pos_log), controlled_contact=np.asarray(args.controlled_contact or ""), controlled_contact_side=np.asarray(args.controlled_contact_side))
    root_drift=float(np.max(np.abs(np.asarray(root_log)-np.asarray(root_log)[0])))
    if not args.free_root and root_drift > 1e-5:
        raise RuntimeError(f"Fixed-root weld drift exceeds tolerance: {root_drift}")
    label = "PHASE2_MUJOCO_OK" if args.free_root else "PHASE1_MUJOCO_OK"
    mode = "controlled_contact" if args.controlled_contact else "upright_perturbation" if args.upright_perturbation else "sequential_steps"
    initial_root_pos, initial_root_quat = contact_root[args.controlled_contact_side] if args.controlled_contact else ((0.0, 0.0, args.initial_root_z), args.initial_root_quat_wxyz)
    print(f"{label} mode={mode} actuator=UnitreeActuatorCfg_Go2HV pd=deployment Go2HV=(X1=13.5,X2=30,Y1=20.2,Y2=23.4,Fs=0.2,Fd=0.1,Va=0.01) fixed_root={int(not args.free_root)} gravity={args.gravity} collisions={int(args.controlled_contact == 'on')} armature={args.joint_armature} initial_root_z={initial_root_pos[2]} initial_root_quat_wxyz={np.asarray(initial_root_quat).round(6).tolist()} log_post_control_state={int(args.log_post_control_state)} excited_joints={args.excitation_joints} controls={total} physics_dt={args.dt} control_dt={args.control_dt} direct_torque={int(direct_mode)} root_span={root_drift:.6g} telemetry={args.telemetry}")

if __name__ == "__main__":
    main()
