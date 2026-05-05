# Go2 Bipedal Walking Scripts

Convenience scripts for training, evaluating, and recording Go2 bipedal walking policies.

## Quick Start

```bash
# 1. Train a Newton policy (recommended for sim2sim/sim2real)
./train_newton.sh

# 2. Test sim2sim transfer in MuJoCo
./sim2sim_test.sh logs/rsl_rl/unitree_go2_bipedal_walk_newton/<run>/model_3000.pt --cmd-vx 0.3

# 3. Record a video of the transfer
./record_sim2sim.sh logs/rsl_rl/unitree_go2_bipedal_walk_newton/<run>/model_3000.pt result.mp4

# 4. Export checkpoint for deployment
./export_policy.sh logs/.../model_3000.pt deployed_policy.pt
```

## Scripts

| Script | Purpose |
|--------|---------|
| `train_newton.sh` | Train bipedal policy with Newton (MuJoCo Warp) backend. **Use this for sim2sim/sim2real.** |
| `train_physx.sh` | Train with PhysX backend. Fast prototyping only — does NOT transfer to MuJoCo. |
| `sim2sim_test.sh` | Run MuJoCo sim2sim evaluation (terminal output, optional CSV logging). |
| `record_sim2sim.sh` | Record MuJoCo sim2sim video (480p @ 25fps MP4). |
| `record_isaaclab.sh` | Record Isaac Lab playback video using Newton headless renderer. |
| `export_policy.sh` | Export checkpoint to JIT torchscript for deployment. |

## Backend Choice: Newton vs PhysX

| | Newton | PhysX |
|---|--------|-------|
| **Sim2sim transfer** | ✅ 15s+ stable | ❌ ~3s max |
| **Training speed** | Slightly slower | Faster |
| **Use case** | Deployment, sim2real | Fast reward/behavior iteration |

The sim2sim gap is caused by fundamental contact dynamics differences between PhysX (soft PGS solver) and MuJoCo (stiff Newton solver). Newton uses MuJoCo Warp internally, so policies transfer directly.

## Key Parameters

- **Control frequency**: 50 Hz (dt=0.005, decimation=4 in Isaac Lab; dt=0.002, substeps=10 in MuJoCo)
- **Observation**: 135-dim (ang_vel×4, gravity×4, cmd×1, jpos×4, jvel×4, last_action×1)
- **Action**: 12-dim joint position offsets, scaled by 0.25
- **Actuator**: Explicit PD (kp≈25, kd≈0.5) + torque-speed clipping + friction

## Environment Variables

All scripts accept overrides via environment variables:

```bash
NUM_ENVS=2048 SEED=123 MAX_ITER=5000 ./train_newton.sh
SCENE=/path/to/scene.xml ./sim2sim_test.sh checkpoint.pt
```

## Dependencies

- Conda environment: `go2`
- Isaac Lab 3.0 + `isaaclab-newton` + `isaaclab-visualizers`
- MuJoCo (for sim2sim): `mujoco`, `mediapy`, `torch`
- Go2 MuJoCo model: `unitree_mujoco/unitree_robots/go2/scene_flat.xml`
