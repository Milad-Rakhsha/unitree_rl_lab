#!/bin/bash
# Record MuJoCo sim2sim video of a trained checkpoint.
# Useful for quick visual validation of policy transfer.
#
# Usage:
#   ./record_sim2sim.sh <checkpoint_path> [output.mp4] [--cmd-vx 0.3]
#
# Examples:
#   ./record_sim2sim.sh logs/.../model_3000.pt sim2sim_newton_3k.mp4
#   ./record_sim2sim.sh logs/.../model_3000.pt --cmd-vx 0.0  # standing test
#
# Output: 640x480 @ 25fps MP4 with tracking camera.
# Requires: conda env 'go2', MuJoCo with EGL rendering, mediapy.

set -euo pipefail

CONDA_EXE="${CONDA_EXE:-$HOME/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dvi}"
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate "$CONDA_ENV"
export DISPLAY=${DISPLAY:-:99}
export MUJOCO_GL=${MUJOCO_GL:-egl}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM2SIM_DIR="$SCRIPT_DIR/../sim2sim"
SCENE="${SCENE:-$(realpath "$SCRIPT_DIR/../../unitree_mujoco/unitree_robots/go2/scene_flat.xml" 2>/dev/null || echo "$HOME/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene_flat.xml")}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [output.mp4] [--cmd-vx 0.3] [--duration 15]"
    exit 1
fi

CHECKPOINT="$1"
shift

# Check if second arg is an output path (ends in .mp4)
OUTPUT="sim2sim_recording.mp4"
if [ $# -gt 0 ] && [[ "$1" == *.mp4 ]]; then
    OUTPUT="$1"
    shift
fi

CMD_VX="${CMD_VX:-0.3}"
DURATION="${DURATION:-15}"

# Parse remaining args for overrides
while [ $# -gt 0 ]; do
    case "$1" in
        --cmd-vx) CMD_VX="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$SIM2SIM_DIR"

python -c "
import sys, os, math
sys.path.insert(0, '.')
from sim2sim_v2 import load_policy, clip_torque, qri, detect_joint_ordering, get_mappings
import imageio.v2 as imageio
import mujoco, numpy as np, torch, yaml
from collections import deque

SCENE = '$SCENE'
CHECKPOINT = '$CHECKPOINT'
OUTPUT = '$OUTPUT'
CMD_VX = $CMD_VX
DURATION = $DURATION

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
dt = model.opt.timestep
substeps = round(0.02 / dt)
ctrl_dt = substeps * dt

# Auto-detect joint ordering
checkpoint_dir = os.path.dirname(os.path.abspath(CHECKPOINT))
deploy_yaml = None
for candidate in [
    os.path.join(checkpoint_dir, 'params', 'deploy.yaml'),
    os.path.join(os.path.dirname(checkpoint_dir), 'params', 'deploy.yaml'),
]:
    if os.path.exists(candidate):
        deploy_yaml = candidate
        break
ordering = detect_joint_ordering(deploy_yaml)
iqpos, iqvel, iact, dpos, kpf, kdf = get_mappings(ordering)
print(f'Joint ordering: {ordering}')

kp = np.full(12, 25.0); kd = np.full(12, 0.5)
if deploy_yaml and os.path.exists(deploy_yaml):
    with open(deploy_yaml) as f: cfg = yaml.safe_load(f)
    kp = np.array(cfg['stiffness']); kd = np.array(cfg['damping'])
    dpos = np.array(cfg['default_joint_pos'])

policy, is_jit = load_policy(CHECKPOINT)
print(f'ctrl={1/ctrl_dt:.0f}Hz, policy={\"JIT\" if is_jit else \"ckpt\"}')

# Init
data.qpos[:3] = [0, 0, 0.31]; data.qpos[3:7] = [1, 0, 0, 0]
for i in range(12): data.qpos[int(iqpos[i])] = dpos[i]
data.qvel[:] = 0; mujoco.mj_forward(model, data)

# Warmup: 400 fixstand + 200 intro
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
print(f'Warmup done: z={data.qpos[2]:.4f}')

# Renderer
renderer = mujoco.Renderer(model, height=480, width=640)
frames = []

# History buffers
hist = 4
ang_buf = deque(maxlen=hist); grav_buf = deque(maxlen=hist)
jpos_buf = deque(maxlen=hist); jvel_buf = deque(maxlen=hist)
quat = data.qpos[3:7]
pg = qri(quat, np.array([0.,0.,-1.])); av = data.qvel[3:6].copy()
for _ in range(hist):
    ang_buf.append(av * 0.2); grav_buf.append(pg.copy())
    jpos_buf.append(np.zeros(12)); jvel_buf.append(np.zeros(12))
last_action = np.zeros(12)
cmd = np.array([CMD_VX, 0.0, 0.0])

n_steps = int(DURATION / ctrl_dt)
frame_every = max(1, int(0.04 / ctrl_dt))  # 25fps

for step in range(n_steps):
    t = step * ctrl_dt
    quat = data.qpos[3:7]; pg = qri(quat, np.array([0.,0.,-1.]))
    av = data.qvel[3:6].copy()
    jp = np.array([data.qpos[int(iqpos[i])] for i in range(12)])
    jv = np.array([data.qvel[int(iqvel[i])] for i in range(12)])
    ang_buf.append(av * 0.2); grav_buf.append(pg.copy())
    jpos_buf.append((jp - dpos).copy()); jvel_buf.append((jv * 0.05).copy())

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
    last_action = action.copy()
    target_q = dpos + action * 0.25

    for _ in range(substeps):
        tau = np.zeros(12); vel = np.zeros(12)
        for i in range(12):
            qi = data.qpos[int(iqpos[i])]; dqi = data.qvel[int(iqvel[i])]
            tau[i] = kp[i] * (target_q[i] - qi) + kd[i] * (0 - dqi); vel[i] = dqi
        tau = clip_torque(tau, vel)
        for i in range(12): data.ctrl[int(iact[i])] = tau[i]
        mujoco.mj_step(model, data)

    if step % frame_every == 0:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 0
        cam.distance = 1.8
        cam.azimuth = 135
        cam.elevation = -20
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    if data.qpos[2] < 0.15:
        print(f'FELL at t={t:.2f}s')
        break
    if step % 100 == 0 and step > 0:
        print(f'  t={t:.1f}s z={data.qpos[2]:.3f} x={data.qpos[0]:.2f}')
else:
    print(f'SURVIVED {DURATION}s! z={data.qpos[2]:.3f} x={data.qpos[0]:.2f}')

print(f'Saving {len(frames)} frames -> {OUTPUT}')
imageio.mimwrite(OUTPUT, frames, fps=25, codec='libx264', quality=8)
print(f'Done: {OUTPUT}')
"
