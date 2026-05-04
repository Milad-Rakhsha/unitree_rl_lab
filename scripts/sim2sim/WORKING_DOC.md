# Go2 Bipedal Sim2Sim Transfer - Working Configuration

## ✅ VALIDATED: Static Bipedal Standing

The old policy (Milad's 10k-iter checkpoint, trained dt=0.005) successfully transfers to MuJoCo with:

### Configuration
```
Scene:       ~/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene.xml
MuJoCo dt:   0.002 (native, do NOT override)
Control rate: 125 Hz (control_dt=0.008, 4 substeps)
Policy:      checkpoints/2026-04-21_15-56-29/exported/policy.pt
PD gains:    kp=25, kd=0.5 (nominal)
Torque curve: ENABLED (Go2 HV: Y1=20.2, Y2=23.4, X1=13.5, X2=30.0)
```

### FSM Phases
1. **FixStand** (0.8s = 400 physics steps): kp=[60,60,60,60,80,80,80,80,80,80,80,80], kd=[5,5,5,5,4,4,4,4,4,4,4,4]
2. **Intro** (0.4s = 200 physics steps): uniform kp=60, kd=5
3. **Policy**: 125 Hz inference, torque curve applied

### Observation Reset at Policy Start
- ang_vel history: fill with current value × 0.2
- proj_gravity history: fill with current value
- joint_pos/vel history: fill with ZEROS (not actual values)
- last_action: zeros

### Result
- **30+ seconds** of stable bipedal stance
- z ≈ 0.59m, pitch ≈ -86° (nearly vertical on rear legs)
- Minimal drift (< 1m lateral in 30s)

## ⚠️ Known Limitations
1. Walking (cmd_vx > 0) does NOT transfer — robot falls
2. Only works at 125 Hz (50 Hz = native training rate does NOT work in MuJoCo)
3. Sensitive to warmup duration (full 2s+1s warmup fails, short 0.8s+0.4s works)

## Root Cause of Sim Gap
- PhysX PGS solver: soft/compliant contacts at dt=0.005
- MuJoCo Newton solver: stiff/accurate contacts at dt=0.002
- The 125 Hz control rate compensates by providing smoother torque application
- Unitree's official MuJoCo sim uses `SIMULATE_DT=0.005` (overrides native dt)

## How to Run
```bash
conda run -n go2 python sim2sim_v2.py \
    --checkpoint checkpoints/2026-04-21_15-56-29/exported/policy.pt \
    --deploy-yaml /tmp/nominal_deploy.yaml \
    --duration 30.0
```

## Scripts
- `sim2sim_v2.py` — validated sim2sim runner
- `sim2sim_bipedal.py` — old script (has correct 50 Hz logic, but fails due to sim gap)
- `export_policy.py` — export JIT from rsl_rl checkpoint
- `test_checkpoints.sh` — batch test script

## Ongoing Training (dt=0.002)
- Run: `2026-05-04_07-57-36`
- ETA: ~11:30 UTC
- Goal: train policy that might transfer at native 50 Hz (matches MuJoCo dt)
- Early results: iters 100-500 survive at 125 Hz in partial rear-up
