#!/usr/bin/env python3
"""MuJoCo sim2sim for Go2 bipedal walking — validated configuration.

Direct translation of the working inline test with proper CLI interface.
Uses 125 Hz control rate (dt=0.002, 4 substeps) + FixStand-level intro gains.

Usage:
    conda run -n go2 python sim2sim_v2.py \
        --checkpoint ~/Repos/GO2/checkpoints/2026-04-21_15-56-29/exported/policy.pt \
        [--deploy-yaml /tmp/nominal_deploy.yaml] \
        [--cmd-vx 0.6] [--duration 30.0] [--render] [--csv output.csv]
"""
import argparse, csv, math, os
from collections import deque
import mujoco, numpy as np, torch, yaml

# ── Constants ────────────────────────────────────────────────────────
Y1, Y2, X1, X2 = 20.2, 23.4, 13.5, 30.0
IQPOS = np.array([7, 10, 13, 16, 8, 11, 14, 17, 9, 12, 15, 18])
IQVEL = np.array([6, 9, 12, 15, 7, 10, 13, 16, 8, 11, 14, 17])
IACT  = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
DPOS  = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
KPF   = np.array([60., 60, 60, 60, 80, 80, 80, 80, 80, 80, 80, 80])
KDF   = np.array([5., 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4])

def clip_torque(tau, vel):
    sd = (vel * tau) > 0; me = np.where(sd, Y1, Y2)
    av = np.abs(vel); ab = av > X1
    k = -me / (X2 - X1); lim = k * (av - X1) + me; lim = np.maximum(lim, 0.0)
    me = np.where(ab, lim, me)
    return np.clip(tau - 0.01 * vel, -me, me)

def qri(q, v):
    w, x, y, z = q; u = np.array([x, y, z]); t = 2.0 * np.cross(u, v)
    return v - w * t + np.cross(u, t)

def load_policy(path, device="cpu"):
    try:
        p = torch.jit.load(path, map_location=device); p.eval(); return p, True
    except Exception: pass
    from torch import nn
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("actor_state_dict", ck.get("model_state_dict"))
    a = nn.Sequential(nn.Linear(135,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                       nn.Linear(256,128),nn.ELU(),nn.Linear(128,12))
    a.load_state_dict({k.replace("mlp.",""):v for k,v in sd.items() if "mlp" in k}, strict=True)
    a.eval(); return a, False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--deploy-yaml", default=None)
    p.add_argument("--scene", default=os.path.expanduser("~/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene.xml"))
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--csv", default=None)
    p.add_argument("--cmd-vx", type=float, default=0.0)
    p.add_argument("--cmd-yaw", type=float, default=0.0)
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    dt = model.opt.timestep  # 0.002
    substeps = 4             # 125 Hz control
    ctrl_dt = substeps * dt  # 0.008

    kp = np.full(12, 25.0); kd = np.full(12, 0.5)
    dpos = DPOS.copy()
    if args.deploy_yaml and os.path.exists(args.deploy_yaml):
        with open(args.deploy_yaml) as f: cfg = yaml.safe_load(f)
        kp = np.array(cfg["stiffness"]); kd = np.array(cfg["damping"])
        dpos = np.array(cfg["default_joint_pos"])

    policy, is_jit = load_policy(args.checkpoint)
    print(f"MuJoCo dt={dt}, ctrl_dt={ctrl_dt} ({1/ctrl_dt:.0f} Hz), substeps={substeps}")
    print(f"Policy: {'JIT' if is_jit else 'sd'}, kp={kp.mean():.1f}, kd={kd.mean():.3f}")

    # ─── Init ────────────────────────────────────────────────────────
    data.qpos[:3] = [0, 0, 0.31]; data.qpos[3:7] = [1, 0, 0, 0]
    for i in range(12): data.qpos[int(IQPOS[i])] = dpos[i]
    data.qvel[:] = 0; mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data) if args.render else None
    csv_f, csv_w = (None, None)
    if args.csv:
        csv_f = open(args.csv, "w", newline=""); csv_w = csv.writer(csv_f)
        csv_w.writerow(["t","z","pitch","x","y",*[f"a{i}" for i in range(12)]])

    # ─── Warmup: 400 fixstand (FixStand gains) + 200 intro (uniform 60/5) ──
    for _ in range(400):
        for i in range(12):
            qi = data.qpos[int(IQPOS[i])]; dqi = data.qvel[int(IQVEL[i])]
            data.ctrl[int(IACT[i])] = KPF[i] * (dpos[i] - qi) + KDF[i] * (0 - dqi)
        mujoco.mj_step(model, data)
    for _ in range(200):
        for i in range(12):
            qi = data.qpos[int(IQPOS[i])]; dqi = data.qvel[int(IQVEL[i])]
            data.ctrl[int(IACT[i])] = 60.0 * (dpos[i] - qi) + 5.0 * (0 - dqi)
        mujoco.mj_step(model, data)
    print(f"  Warmup done (1.2s): z={data.qpos[2]:.4f}")

    # ─── Policy ──────────────────────────────────────────────────────
    hist = 4
    ang_buf = deque(maxlen=hist); grav_buf = deque(maxlen=hist)
    jpos_buf = deque(maxlen=hist); jvel_buf = deque(maxlen=hist)
    quat = data.qpos[3:7]; pg = qri(quat, np.array([0., 0., -1.])); av = data.qvel[3:6].copy()
    for _ in range(hist):
        ang_buf.append(av * 0.2); grav_buf.append(pg.copy())
        jpos_buf.append(np.zeros(12)); jvel_buf.append(np.zeros(12))
    last_action = np.zeros(12)
    cmd = np.array([args.cmd_vx, 0.0, args.cmd_yaw])
    t_start = data.time

    print(f"  Running policy for {args.duration}s (cmd_vx={args.cmd_vx}, cmd_yaw={args.cmd_yaw})")
    print("-" * 60)

    n_steps = int(args.duration / ctrl_dt)
    for step in range(n_steps):
        t = step * ctrl_dt
        quat = data.qpos[3:7]; pg = qri(quat, np.array([0., 0., -1.]))
        av = data.qvel[3:6].copy()
        jp = np.array([data.qpos[int(IQPOS[i])] for i in range(12)])
        jv = np.array([data.qvel[int(IQVEL[i])] for i in range(12)])
        jpr = jp - dpos

        ang_buf.append(av * 0.2); grav_buf.append(pg.copy())
        jpos_buf.append(jpr.copy()); jvel_buf.append(jv * 0.05)

        obs = []
        for h in range(hist): obs.extend(ang_buf[h])
        for h in range(hist): obs.extend(grav_buf[h])
        obs.extend(cmd)
        for h in range(hist): obs.extend(jpos_buf[h])
        for h in range(hist): obs.extend(jvel_buf[h])
        obs.extend(last_action)
        obs = np.clip(np.array(obs), -100, 100)

        with torch.no_grad():
            act = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = act[0].numpy() if is_jit else act[0].detach().numpy()
        target_q = dpos + action * 0.25; last_action = action.copy()

        for _ in range(substeps):
            tau = np.zeros(12); vel = np.zeros(12)
            for i in range(12):
                qi = data.qpos[int(IQPOS[i])]; dqi = data.qvel[int(IQVEL[i])]
                tau[i] = kp[i] * (target_q[i] - qi) + kd[i] * (0 - dqi); vel[i] = dqi
            tau = clip_torque(tau, vel)
            for i in range(12): data.ctrl[int(IACT[i])] = tau[i]
            mujoco.mj_step(model, data)

        sinp = 2 * (quat[0]*quat[2] - quat[3]*quat[1])
        pitch = math.degrees(math.asin(max(-1, min(1, sinp))))

        if csv_w:
            csv_w.writerow([f"{t:.4f}",f"{data.qpos[2]:.4f}",f"{pitch:.2f}",
                            f"{data.qpos[0]:.4f}",f"{data.qpos[1]:.4f}",
                            *[f"{action[i]:.4f}" for i in range(12)]])

        report = max(1, int(1.0 / ctrl_dt))
        if step % report == 0:
            print(f"  t={t:6.2f}s z={data.qpos[2]:.3f} pitch={pitch:+.1f}° x={data.qpos[0]:.2f}")

        if viewer:
            if viewer.is_running(): viewer.sync()
            else: break

        if data.qpos[2] < 0.15:
            print(f"  FELL at t={t:.2f}s (z={data.qpos[2]:.3f})")
            break
    else:
        fp = math.degrees(math.asin(max(-1,min(1,2*(data.qpos[3]*data.qpos[5]-data.qpos[6]*data.qpos[4])))))
        print(f"\n  ✓ SURVIVED {args.duration}s! z={data.qpos[2]:.3f} pitch={fp:.1f}°")

    if csv_f: csv_f.close(); print(f"  CSV: {args.csv}")
    print(f"\n  Final: x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} z={data.qpos[2]:.3f}")

if __name__ == "__main__":
    main()
