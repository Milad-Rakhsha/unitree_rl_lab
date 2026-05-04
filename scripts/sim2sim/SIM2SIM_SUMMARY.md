# Go2 Bipedal Sim2Sim Transfer — Summary

**Date**: 2026-05-04  
**Author**: AI Assistant (autonomous session)  
**Status**: ✅ Sim2sim validated for static bipedal standing

## Key Results

| Configuration | Policy | Control Rate | Duration | Result |
|---|---|---|---|---|
| Native dt=0.002, 4 substeps | Old (Milad 10k) | 125 Hz | 30s | ✅ z=0.59, pitch=-86° |
| Native dt=0.002, 10 substeps | Old (Milad 10k) | 50 Hz | 5.2s | ❌ Falls |
| dt=0.005 override, 4 substeps | Old (Milad 10k) | 50 Hz | 5.3s | ❌ Falls |
| Native dt=0.002, 4 substeps | DR trained (dt=0.005) | 125 Hz | 2-3s | ❌ Falls |
| Native dt=0.002, 4 substeps | dt=0.002 trained (iter 100-500) | 125 Hz | 15s+ | ✅ Partial (pitch -30 to -64°) |
| Walking (cmd_vx=0.5) | Old (Milad 10k) | 125 Hz | 5.8s | ❌ Falls |

## Root Cause Analysis

The sim gap comes from **fundamentally different contact solvers**:
- **PhysX (Isaac Lab)**: PGS solver → soft/compliant contacts, naturally smooths dynamics
- **MuJoCo**: Newton solver → stiff/accurate contacts, requires smoother control

Running the policy at **125 Hz** (2.5× the training rate) compensates by providing smoother torque application. This is analogous to how PhysX's soft contacts naturally smooth out the 50 Hz control signal.

**Note**: Unitree's official MuJoCo sim (`config.py`) overrides `SIMULATE_DT = 0.005`, but this alone doesn't fix transfer because the contact model remains different.

## Bugs Fixed Along the Way

1. **Angular velocity frame** — MuJoCo `qvel[3:6]` is already body-frame; was double-rotating
2. **Observation history ordering** — Changed to oldest-first (matching IsaacLab CircularBuffer)
3. **Torque-speed curve** — Must be enabled during policy phase (Go2 HV motors)
4. **History initialization** — Zeros for jpos/jvel at policy handover
5. **Physics timestep awareness** — Understanding dt=0.002 vs 0.005 mismatch

## Files

| File | Purpose |
|---|---|
| `sim2sim_v2.py` | **Validated sim2sim script** (125 Hz, torque curve) |
| `sim2sim_bipedal.py` | Original script (50 Hz, has FSM — currently doesn't transfer) |
| `export_policy.py` | JIT export from rsl_rl checkpoint |
| `test_checkpoints.sh` | Batch testing script |
| `WORKING_DOC.md` | Quick reference for working configuration |
| `setup_env.sh` | Environment setup for training |

## How to Run

```bash
# Sim2sim (static standing)
conda run -n go2 python sim2sim_v2.py \
    --checkpoint checkpoints/2026-04-21_15-56-29/exported/policy.pt \
    --deploy-yaml /tmp/nominal_deploy.yaml \
    --duration 30.0

# With rendering
conda run -n go2 python sim2sim_v2.py \
    --checkpoint checkpoints/2026-04-21_15-56-29/exported/policy.pt \
    --deploy-yaml /tmp/nominal_deploy.yaml \
    --duration 30.0 --render
```

## Paths to Improve Transfer

1. **Train with MuJoCo in the loop** — Use MuJoCo as a DR variant during training
2. **Add control rate randomization** — Train at 50-150 Hz randomly
3. **Contact model matching** — Tune MuJoCo contact parameters to approximate PhysX
4. **Action smoothing in training** — Add stronger action rate penalties
5. **Explicit decimation DR** — Randomize the decimation factor (2-6 instead of fixed 4)

## Active Training Run

- **Config**: `sim.dt=0.002, decimation=10` (50 Hz control, matching MuJoCo physics dt)
- **Run dir**: `2026-05-04_07-57-36`
- **Status**: Converging well (reward ~69, ep_len=1000 at iter 1100)
- **ETA**: ~11:30 UTC (10k iterations)
- **Result so far**: Early checkpoints (100-500) transfer to MuJoCo at 125 Hz with partial rear-up
