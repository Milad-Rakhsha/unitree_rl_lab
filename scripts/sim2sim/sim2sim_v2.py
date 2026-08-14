#!/usr/bin/env python3
"""MuJoCo sim2sim for Go2 bipedal walking — validated configuration.

MuJoCo sim2sim runner for Go2 checkpoints.
Uses the policy's 50 Hz control interval (dt=0.002, 10 substeps) + FixStand-level intro gains.

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

# Default mappings for PhysX policy order:
# Policy order: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, ...
IQPOS_PHYSX = np.array([7, 10, 13, 16, 8, 11, 14, 17, 9, 12, 15, 18])
IQVEL_PHYSX = np.array([6, 9, 12, 15, 7, 10, 13, 16, 8, 11, 14, 17])
IACT_PHYSX  = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
DPOS_PHYSX  = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
KPF_PHYSX   = np.array([60., 60, 60, 60, 80, 80, 80, 80, 80, 80, 80, 80])
KDF_PHYSX   = np.array([5., 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4])

# Newton (MuJoCo Warp) policy order:
# Policy order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, ...
IQPOS_NEWTON = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
IQVEL_NEWTON = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
IACT_NEWTON  = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
DPOS_NEWTON  = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
KPF_NEWTON   = np.array([60., 80, 80, 60, 80, 80, 60, 80, 80, 60, 80, 80])
KDF_NEWTON   = np.array([5., 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4])

# Legacy aliases (default to PhysX for backward compat)
IQPOS = IQPOS_PHYSX
IQVEL = IQVEL_PHYSX
IACT  = IACT_PHYSX
DPOS  = DPOS_PHYSX
KPF   = KPF_PHYSX
KDF   = KDF_PHYSX


def detect_joint_ordering(deploy_yaml_path):
    """Detect PhysX vs Newton joint ordering from deploy.yaml joint_ids_map.

    PhysX joint_ids_map: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
    Newton joint_ids_map: [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

    Returns 'newton' or 'physx'.
    """
    if not deploy_yaml_path or not os.path.exists(deploy_yaml_path):
        return 'physx'  # default
    with open(deploy_yaml_path) as f:
        cfg = yaml.safe_load(f)
    jmap = cfg.get('joint_ids_map')
    if jmap is None:
        return 'physx'
    # Newton has per-leg grouped: [3,4,5,0,1,2,9,10,11,6,7,8]
    if jmap == [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]:
        return 'newton'
    return 'physx'


def get_mappings(ordering):
    """Return (IQPOS, IQVEL, IACT, DPOS, KPF, KDF) for the given ordering."""
    if ordering == 'newton':
        return IQPOS_NEWTON, IQVEL_NEWTON, IACT_NEWTON, DPOS_NEWTON, KPF_NEWTON, KDF_NEWTON
    return IQPOS_PHYSX, IQVEL_PHYSX, IACT_PHYSX, DPOS_PHYSX, KPF_PHYSX, KDF_PHYSX

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
    if str(path).lower().endswith(".onnx"):
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        class OnnxPolicy:
            obs_dim = int(input_shape[1])
            def __call__(self, obs):
                array = obs.detach().cpu().numpy().astype(np.float32, copy=False)
                return torch.from_numpy(session.run(None, {input_name: array})[0])
            def parameters(self):
                return iter(())
        return OnnxPolicy(), True
    try:
        p = torch.jit.load(path, map_location=device); p.eval(); return p, True
    except Exception:
        pass
    from torch import nn
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("actor_state_dict", ck.get("model_state_dict"))
    first_weight = sd.get("mlp.0.weight")
    if first_weight is None:
        raise ValueError("Checkpoint does not contain actor mlp.0.weight.")
    obs_dim = first_weight.shape[1]
    act_dim = sd["mlp.6.weight"].shape[0]
    a = nn.Sequential(nn.Linear(obs_dim,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                       nn.Linear(256,128),nn.ELU(),nn.Linear(128,act_dim))
    a.load_state_dict({k.replace("mlp.",""):v for k,v in sd.items() if k.startswith("mlp.")}, strict=True)
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
    p.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Observation-history length; default infers one frame from the checkpoint actor.",
    )
    p.add_argument("--sim-dt", type=float, default=None, help="Override MuJoCo sim timestep (smaller=more stable)")
    p.add_argument("--backend", choices=["physx", "newton", "auto"], default="auto",
                   help="Joint ordering: physx, newton, or auto-detect from deploy.yaml")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    # Allow overriding sim dt for stability (smaller = more accurate explicit integration)
    if hasattr(args, 'sim_dt') and args.sim_dt:
        model.opt.timestep = args.sim_dt
    dt = model.opt.timestep
    # Keep control at 50 Hz regardless of sim dt
    substeps = round(0.02 / dt)  # ctrl_dt = 0.02s = 50 Hz
    ctrl_dt = substeps * dt
    assert abs(ctrl_dt - 0.02) < 1e-6, f"ctrl_dt={ctrl_dt} != 0.02"

    # Auto-find deploy.yaml from checkpoint dir
    deploy_yaml = args.deploy_yaml
    if deploy_yaml is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        for candidate in [
            os.path.join(checkpoint_dir, "params", "deploy.yaml"),
            os.path.join(os.path.dirname(checkpoint_dir), "params", "deploy.yaml"),
        ]:
            if os.path.exists(candidate):
                deploy_yaml = candidate
                break

    # Detect joint ordering
    if args.backend == "auto":
        ordering = detect_joint_ordering(deploy_yaml)
    else:
        ordering = args.backend
    iqpos, iqvel, iact, dpos, kpf, kdf = get_mappings(ordering)
    print(f"Joint ordering: {ordering} (deploy_yaml: {deploy_yaml})")

    kp = np.full(12, 25.0); kd = np.full(12, 0.5)
    if deploy_yaml and os.path.exists(deploy_yaml):
        with open(deploy_yaml) as f: cfg = yaml.safe_load(f)
        kp = np.array(cfg["stiffness"]); kd = np.array(cfg["damping"])
        dpos = np.array(cfg["default_joint_pos"])

    policy, is_jit = load_policy(args.checkpoint)
    if is_jit:
        try:
            actor_obs_dim = next(policy.parameters()).shape[1]
        except StopIteration as exc:
            raise ValueError("Cannot infer actor observation dimension from TorchScript policy.") from exc
    else:
        actor_obs_dim = policy[0].in_features
    if args.history_length is None:
        # Observation layout: 3H angular velocity + 3H gravity + 3 command
        # + 12H joint position + 12H joint velocity + 12 last action = 15 + 30H.
        if (actor_obs_dim - 15) % 30:
            raise ValueError(f"Unsupported actor observation dimension {actor_obs_dim}; expected 15 + 30H.")
        hist = (actor_obs_dim - 15) // 30
    else:
        hist = args.history_length
        expected_obs_dim = 15 + 30 * hist
        if actor_obs_dim != expected_obs_dim:
            raise ValueError(
                f"history_length={hist} requires {expected_obs_dim} actor inputs, but policy expects {actor_obs_dim}."
            )
    print(f"MuJoCo dt={dt}, ctrl_dt={ctrl_dt} ({1/ctrl_dt:.0f} Hz), substeps={substeps}")
    print(
        f"Policy: {'JIT' if is_jit else 'sd'}, obs_dim={actor_obs_dim}, history={hist}, "
        f"kp={kp.mean():.1f}, kd={kd.mean():.3f}"
    )

    # ─── Init ────────────────────────────────────────────────────────
    data.qpos[:3] = [0, 0, 0.31]; data.qpos[3:7] = [1, 0, 0, 0]
    for i in range(12): data.qpos[int(iqpos[i])] = dpos[i]
    data.qvel[:] = 0; mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data) if args.render else None
    csv_f, csv_w = (None, None)
    if args.csv:
        csv_f = open(args.csv, "w", newline=""); csv_w = csv.writer(csv_f)
        csv_w.writerow(["t","z","pitch","x","y",*[f"a{i}" for i in range(12)]])

    # ─── Warmup: 400 fixstand (FixStand gains) + 200 intro (uniform 60/5) ──
    for _ in range(400):
        for i in range(12):
            qi = data.qpos[int(iqpos[i])]; dqi = data.qvel[int(iqvel[i])]
            data.ctrl[int(iact[i])] = kpf[i] * (dpos[i] - qi) + kdf[i] * (0 - dqi)
        mujoco.mj_step(model, data)
    for _ in range(200):
        for i in range(12):
            qi = data.qpos[int(iqpos[i])]; dqi = data.qvel[int(iqvel[i])]
            data.ctrl[int(iact[i])] = 60.0 * (dpos[i] - qi) + 5.0 * (0 - dqi)
        mujoco.mj_step(model, data)
    print(f"  Warmup done (1.2s): z={data.qpos[2]:.4f}")

    # ─── Policy ──────────────────────────────────────────────────────
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
        jp = np.array([data.qpos[int(iqpos[i])] for i in range(12)])
        jv = np.array([data.qvel[int(iqvel[i])] for i in range(12)])
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
                qi = data.qpos[int(iqpos[i])]; dqi = data.qvel[int(iqvel[i])]
                tau[i] = kp[i] * (target_q[i] - qi) + kd[i] * (0 - dqi); vel[i] = dqi
            tau = clip_torque(tau, vel)
            for i in range(12): data.ctrl[int(iact[i])] = tau[i]
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
