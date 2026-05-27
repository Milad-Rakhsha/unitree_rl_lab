#!/usr/bin/env python3
"""
Sim2Sim FSM video: replicate the C++ deploy FSM in pure Python + MuJoCo.

Chains policies through the FSM scenario:
  Passive → FixStand → BipedalStandUp (2s) → BipedalRear (walk 15s)
  → FixStand (3s) → Stabilize (5s) → FixStand (3s)
  → VelocityGuarded (quad walk 10s) → FixStand → Passive

Each policy loaded from 0_best with deploy.yaml gains + obs spec.
"""
import sys, os, math, time
from collections import deque
import mujoco, numpy as np, torch, yaml, imageio

sys.path.insert(0, os.path.expanduser("~/Repos/GO2/unitree_rl_lab/scripts/sim2sim"))
from sim2sim_v2 import load_policy, clip_torque, qri, detect_joint_ordering, get_mappings

SCENE = os.path.expanduser("~/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene_flat.xml")
LOGS = os.path.expanduser("~/Repos/GO2/unitree_rl_lab/logs/rsl_rl")
OUT = os.path.expanduser("~/Repos/GO2/policy_videos/fsm_full_test.mp4")

# FixStand config from config.yaml
FIXSTAND_KP = np.array([60, 80, 80, 60, 80, 80, 60, 80, 80, 60, 80, 80], dtype=float)
FIXSTAND_KD = np.array([5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4], dtype=float)
FIXSTAND_STANDING = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])
FIXSTAND_BLEND_DURATION = 1.0  # seconds to blend from current to standing

# Newton joint ordering (all 0_best policies use newton)
IQPOS = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
IQVEL = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
IACT  = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


def load_policy_bundle(name):
    """Load policy + deploy config from 0_best directory."""
    d = os.path.join(LOGS, name, "0_best")
    deploy_path = os.path.join(d, "params", "deploy.yaml")
    checkpoint = os.path.join(d, "exported", "policy.pt")
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(d, "best.pt")
    
    with open(deploy_path) as f:
        deploy_cfg = yaml.safe_load(f)
    
    policy, is_jit = load_policy(checkpoint)
    kp = np.array(deploy_cfg["stiffness"])
    kd = np.array(deploy_cfg["damping"])
    dpos = np.array(deploy_cfg["default_joint_pos"])
    obs_spec = deploy_cfg["observations"]
    act_cfg = deploy_cfg.get("actions", {})
    act_key = list(act_cfg.keys())[0] if act_cfg else None
    action_scale = 0.25
    if act_key and "scale" in act_cfg[act_key]:
        s = act_cfg[act_key]["scale"]
        action_scale = s[0] if isinstance(s, list) else s
    
    max_hist = max(spec.get("history_length", 1) for spec in obs_spec.values())
    
    return {
        "policy": policy, "is_jit": is_jit,
        "kp": kp, "kd": kd, "dpos": dpos,
        "obs_spec": obs_spec, "action_scale": action_scale,
        "max_hist": max_hist, "name": name,
    }


class ObsBuilder:
    """Builds observation vectors from deploy.yaml obs spec."""
    def __init__(self, max_hist=4):
        self.buf_len = max_hist + 2
        self.reset()
    
    def reset(self):
        self.ang_vel_buf = deque(maxlen=self.buf_len)
        self.grav_buf = deque(maxlen=self.buf_len)
        self.jpos_buf = deque(maxlen=self.buf_len)
        self.jvel_buf = deque(maxlen=self.buf_len)
        self.last_action = np.zeros(12)
        for _ in range(self.buf_len):
            self.ang_vel_buf.append(np.zeros(3))
            self.grav_buf.append(np.array([0., 0., -1.]))
            self.jpos_buf.append(np.zeros(12))
            self.jvel_buf.append(np.zeros(12))
    
    def update(self, data, dpos):
        quat = data.qpos[3:7]
        pg = qri(quat, np.array([0., 0., -1.]))
        av = data.qvel[3:6].copy()
        jp = np.array([data.qpos[int(IQPOS[i])] for i in range(12)])
        jv = np.array([data.qvel[int(IQVEL[i])] for i in range(12)])
        
        self.ang_vel_buf.append(av.copy())
        self.grav_buf.append(pg.copy())
        self.jpos_buf.append((jp - dpos).copy())
        self.jvel_buf.append(jv.copy())  # raw; scale applied by obs spec
    
    def build(self, obs_spec, cmd):
        obs = []
        for obs_name, spec in obs_spec.items():
            hist = spec.get("history_length", 1)
            scale = np.array(spec.get("scale", [1.0]))
            clip_lo, clip_hi = spec.get("clip", [-100, 100])
            
            if obs_name == "base_ang_vel":
                buf = list(self.ang_vel_buf)[-hist:]
                for e in buf:
                    obs.extend(np.clip(e * scale, clip_lo, clip_hi))
            elif obs_name == "projected_gravity":
                buf = list(self.grav_buf)[-hist:]
                for e in buf:
                    obs.extend(np.clip(e * scale, clip_lo, clip_hi))
            elif obs_name == "velocity_commands":
                obs.extend(np.clip(cmd * scale, clip_lo, clip_hi))
            elif obs_name == "joint_pos_rel":
                buf = list(self.jpos_buf)[-hist:]
                for e in buf:
                    obs.extend(np.clip(e * scale, clip_lo, clip_hi))
            elif obs_name == "joint_vel_rel":
                buf = list(self.jvel_buf)[-hist:]
                for e in buf:
                    obs.extend(np.clip(e * scale, clip_lo, clip_hi))
            elif obs_name == "last_action":
                obs.extend(np.clip(self.last_action * scale, clip_lo, clip_hi))
        
        return np.array(obs, dtype=np.float64)


def apply_pd(data, target_q, kp, kd, substeps):
    """Apply PD control for substeps."""
    for _ in range(substeps):
        tau = np.zeros(12)
        vel = np.zeros(12)
        for i in range(12):
            qi = data.qpos[int(IQPOS[i])]
            dqi = data.qvel[int(IQVEL[i])]
            tau[i] = kp[i] * (target_q[i] - qi) + kd[i] * (0 - dqi)
            vel[i] = dqi
        tau = clip_torque(tau, vel)
        for i in range(12):
            data.ctrl[int(IACT[i])] = tau[i]
        mujoco.mj_step(model, data)


def run_fixstand(data, duration_s, substeps, ctrl_dt, label="FixStand"):
    """Run FixStand: blend from current joint angles to standing pose."""
    n = int(duration_s / ctrl_dt)
    # Capture entry pose
    entry_pose = data.qpos[IQPOS].copy()
    for step in range(n):
        t = step * ctrl_dt
        if t < FIXSTAND_BLEND_DURATION:
            alpha = t / FIXSTAND_BLEND_DURATION
            target = (1 - alpha) * entry_pose + alpha * FIXSTAND_STANDING
        else:
            target = FIXSTAND_STANDING.copy()

        apply_pd(data, target, FIXSTAND_KP, FIXSTAND_KD, substeps)
        yield step, t, label


def run_policy_state(data, bundle, obs_builder, cmd, duration_s, substeps, ctrl_dt, label=None):
    """Run an RL policy for duration_s."""
    policy = bundle["policy"]
    is_jit = bundle["is_jit"]
    kp = bundle["kp"]
    kd = bundle["kd"]
    dpos = bundle["dpos"]
    obs_spec = bundle["obs_spec"]
    action_scale = bundle["action_scale"]
    
    if label is None:
        label = bundle["name"]
    
    n = int(duration_s / ctrl_dt)
    for step in range(n):
        t = step * ctrl_dt
        obs_builder.update(data, dpos)
        obs = obs_builder.build(obs_spec, cmd)
        
        with torch.no_grad():
            act = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        action = act[0].numpy() if is_jit else act[0].detach().numpy()
        obs_builder.last_action = action.copy()
        target_q = dpos + action * action_scale
        
        apply_pd(data, target_q, kp, kd, substeps)
        yield step, t, label
        
        if data.qpos[2] < 0.10:
            print(f"  ✗ FELL during {label} at t={t:.2f}s")
            return


# ─── Main ────────────────────────────────────────────────────────────────

print("Loading policies...")
bipedal_standup = load_policy_bundle("unitree_go2_bipedal_standup")
bipedal_walk = load_policy_bundle("unitree_go2_bipedal_walk_rough")
stabilize = load_policy_bundle("unitree_go2_stabilize")
velocity = load_policy_bundle("unitree_go2_velocity_rough")

print("Setting up MuJoCo...")
model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
dt = model.opt.timestep
substeps = round(0.02 / dt)
ctrl_dt = substeps * dt

# Init
data.qpos[:3] = [0, 0, 0.31]
data.qpos[3:7] = [1, 0, 0, 0]
for i in range(12):
    data.qpos[int(IQPOS[i])] = FIXSTAND_STANDING[i]
data.qvel[:] = 0
mujoco.mj_forward(model, data)

# Renderer + video
renderer = mujoco.Renderer(model, height=480, width=640)
fps = 25
frame_every = max(1, int((1.0 / fps) / ctrl_dt))
writer = imageio.get_writer(OUT, fps=fps, quality=8)
obs_builder = ObsBuilder(max_hist=4)

total_step = 0
total_time = 0.0

def capture_frame(label):
    global total_step
    if total_step % frame_every == 0:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 0
        cam.distance = 2.0
        cam.azimuth = 135
        cam.elevation = -20
        renderer.update_scene(data, camera=cam)
        frame = renderer.render().copy()
        # Add label text overlay (simple: draw on top-left)
        # No PIL available, just capture raw frame
        writer.append_data(frame)

def run_phase(gen, phase_name):
    global total_step, total_time
    t0 = time.time()
    print(f"\n  Phase: {phase_name}")
    for step, t, label in gen:
        capture_frame(label)
        total_step += 1
        total_time += ctrl_dt
        if step % max(1, int(2.0 / ctrl_dt)) == 0 and step > 0:
            quat = data.qpos[3:7]
            sinp = 2 * (quat[0]*quat[2] - quat[3]*quat[1])
            pitch = math.degrees(math.asin(max(-1, min(1, sinp))))
            print(f"    [{label}] t={total_time:.1f}s z={data.qpos[2]:.3f} pitch={pitch:+.1f}° x={data.qpos[0]:.2f}")
    elapsed = time.time() - t0
    print(f"  Phase done ({elapsed:.1f}s wall)")


# ─── FSM Scenario ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FSM SIM2SIM TEST")
print("="*60)

# 1. FixStand (2s) — settle into default pose
run_phase(run_fixstand(data, 2.0, substeps, ctrl_dt, "FixStand (init)"), "FixStand (init)")

# 2. BipedalStandUp (2s) — stand up on rear legs
obs_builder.reset()
cmd_zero = np.array([0.0, 0.0, 0.0])
run_phase(run_policy_state(data, bipedal_standup, obs_builder, cmd_zero, 2.0, substeps, ctrl_dt, "BipedalStandUp"), "BipedalStandUp")

# 3. BipedalRear — walk forward (6s), backward (4s), turn (4s)
obs_builder.reset()
cmd_fwd = np.array([0.3, 0.0, 0.0])
run_phase(run_policy_state(data, bipedal_walk, obs_builder, cmd_fwd, 6.0, substeps, ctrl_dt, "BipedalRear (fwd)"), "BipedalRear forward")

cmd_back = np.array([-0.2, 0.0, 0.0])
run_phase(run_policy_state(data, bipedal_walk, obs_builder, cmd_back, 4.0, substeps, ctrl_dt, "BipedalRear (back)"), "BipedalRear backward")

cmd_turn = np.array([0.2, 0.0, 0.3])
run_phase(run_policy_state(data, bipedal_walk, obs_builder, cmd_turn, 4.0, substeps, ctrl_dt, "BipedalRear (turn)"), "BipedalRear turn")

# 4. FixStand (2s) — settle
run_phase(run_fixstand(data, 2.0, substeps, ctrl_dt, "FixStand"), "FixStand")

# 5. Stabilize (3s) — recovery from current pose
obs_builder.reset()
run_phase(run_policy_state(data, stabilize, obs_builder, cmd_zero, 3.0, substeps, ctrl_dt, "Stabilize"), "Stabilize")

# 6. FixStand (2s) — settle
run_phase(run_fixstand(data, 2.0, substeps, ctrl_dt, "FixStand"), "FixStand")

# 7. Velocity — quad walk forward (6s), turn (4s)
obs_builder.reset()
cmd_fwd_q = np.array([0.5, 0.0, 0.0])
run_phase(run_policy_state(data, velocity, obs_builder, cmd_fwd_q, 6.0, substeps, ctrl_dt, "Velocity (fwd)"), "Velocity forward")

cmd_turn_q = np.array([0.3, 0.0, 0.5])
run_phase(run_policy_state(data, velocity, obs_builder, cmd_turn_q, 4.0, substeps, ctrl_dt, "Velocity (turn)"), "Velocity turn")

# 8. FixStand (1s) — final settle
run_phase(run_fixstand(data, 1.0, substeps, ctrl_dt, "FixStand (final)"), "FixStand final")

writer.close()
sz = os.path.getsize(OUT)
print(f"\n{'='*60}")
print(f"DONE — {total_time:.1f}s sim time, video: {OUT} ({sz/1024/1024:.1f} MB)")
print(f"{'='*60}")
