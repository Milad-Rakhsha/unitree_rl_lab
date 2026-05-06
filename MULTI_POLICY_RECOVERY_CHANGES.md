# Multi-Policy Recovery: Implementation Summary

## Overview

Enhanced the Go2 stabilization policy to recover from both bipedal and velocity (quadrupedal) walking falls. The implementation includes:

1. Added rough terrain to bipedal policy training
2. Expanded stabilization policy's initial state randomization

## Changes Made

### 1. Bipedal Policy (`bipedal_env_cfg.py`)

**Added rough terrain variants (backward compatible):**

- **Base config unchanged:** `RobotBipedalWalkEnvCfg` still uses flat plane - old policies work unchanged
- **New terrain config:** Created `GO2_BIPEDAL_TERRAIN_CFG` (30% flat, 35% random rough, 25% pyramid slopes, 10% boxes)
- **New rough terrain variants:**
  - `RobotBipedalWalkRoughEnvCfg` (PhysX + rough terrain)
  - `RobotBipedalWalkRoughPlayEnvCfg` (PhysX + rough terrain, eval)
  - `RobotBipedalWalkRoughNewtonEnvCfg` (Newton + rough terrain)
  - `RobotBipedalWalkRoughNewtonPlayEnvCfg` (Newton + rough terrain, eval)
- **Rough terrain changes:**
  - Terrain type: `"generator"` with `GO2_BIPEDAL_TERRAIN_CFG`
  - Initial spawn tilt: expanded from ±0.05 to ±0.14 rad for roll/pitch
  - Terrain curriculum disabled for consistent training difficulty
  - `max_init_terrain_level=0` to ease initial training

**Rationale:** Bipedal walking on rough terrain is more challenging than quadrupedal due to smaller support polygon and higher CoM. The separate configs maintain backward compatibility while enabling robust outdoor training.

### 2. Stabilization Policy (`stabilization_env_cfg.py`)

**Expanded initial state randomization to cover both bipedal and velocity policy failure modes:**

**Pose Range Updates (`reset_base`):**
- `z`: `(-0.08, 0.25)` → `(-0.08, 0.60)` m
  - Now covers bipedal height (~0.55 m) + margin
- `roll`: `(-0.85, 0.85)` → `(-1.2, 1.2)` rad (±49° → ±69°)
  - Broader side-tip coverage for both policies
- `pitch`: `(-0.85, 0.85)` → `(-1.2, 1.2)` rad (±49° → ±69°)
  - Closer to bipedal singularity (π/2 ≈ 1.57 rad)

**Velocity Range Updates (`reset_base`):**
- `x`: `(-2.0, 2.0)` → `(-2.5, 2.5)` m/s
- `y`: `(-2.0, 2.0)` → `(-2.5, 2.5)` m/s
- `z`: `(-1.2, 1.2)` → `(-1.5, 1.5)` m/s
- `roll`: `(-2.5, 2.5)` → `(-3.5, 3.5)` rad/s
- `pitch`: `(-2.5, 2.5)` → `(-4.0, 4.0)` rad/s
- `yaw`: `(-3.5, 3.5)` → `(-4.0, 4.0)` rad/s

**Joint Range Updates (`reset_robot_joints`):**
- `position_range`: `(0.45, 1.55)` → `(0.35, 1.75)` (scale factor)
  - Covers more extreme joint configurations (bipedal extended, velocity tucked)
- `velocity_range`: `(-3.5, 3.5)` → `(-5.0, 5.0)` rad/s
  - Handles fast swing/retraction during falls

**Rationale:** The old stabilization ranges (z max 0.25 m, pitch max ±0.85 rad) did not overlap with bipedal operating regime (z ~0.55 m, pitch ~+π/2 rad). The expanded ranges bridge this gap while maintaining coverage of velocity policy failures.

## Testing Instructions

### Phase 1: Train Bipedal Policy with Rough Terrain

```bash
cd /home/milad/Documents/Repos/IsaacLab

# OLD: Train on flat terrain (backward compatible, for old policies)
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Unitree-Go2-Bipedal-Walk \
  --num_envs 4096 \
  --headless

# NEW: Train bipedal policy with rough terrain (PhysX)
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Unitree-Go2-Bipedal-Walk-Rough \
  --num_envs 4096 \
  --headless

# OR train with rough terrain + Newton (MuJoCo Warp) physics
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Unitree-Go2-Bipedal-Walk-Rough-Newton \
  --num_envs 4096 \
  --headless

# Monitor for:
# - Convergence rate (may be slower due to terrain complexity)
# - Policy robustness on bumps/slopes during evaluation
# - Fall rate compared to flat-only policy
```

### Phase 2: Train Stabilization Policy with Expanded DR

```bash
# Train stabilization policy with expanded state ranges (PhysX)
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Unitree-Go2-Stabilize \
  --num_envs 4096 \
  --headless

# OR train with Newton (MuJoCo Warp) physics
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Unitree-Go2-Stabilize-Newton \
  --num_envs 4096 \
  --headless

# Monitor for:
# - Initial training may take longer due to expanded state space
# - Check if policy learns to recover from high-z drops (bipedal-like states)
# - Check if policy handles large pitch angles (near π/2)
```

### Phase 3: Evaluate Recovery from Both Policy Types

**Test recovery from bipedal falls:**

```bash
# Play bipedal policy and observe fall scenarios
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
  --task Unitree-Go2-Bipedal-Walk-Play \
  --num_envs 32 \
  --load_run <bipedal_run_dir> \
  --checkpoint model_*.pt

# Note typical fall states (z height, pitch angle, velocities)
# Then test stabilization policy starting from those states
```

**Test recovery from velocity falls:**

```bash
# Play velocity policy and observe fall scenarios
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
  --task Unitree-Go2-Velocity-Play \
  --num_envs 32 \
  --load_run <velocity_run_dir> \
  --checkpoint model_*.pt
```

### Phase 4: Integrated FSM Testing (Deploy)

```bash
# Update deploy config with new policy checkpoints
# config.yaml paths:
# - FSM.Velocity.policy_path: <new_velocity_or_existing>
# - FSM.Stabilize.policy_path: <new_stabilization>

# Build and deploy
cd /home/milad/Documents/Repos/unitree_rl_lab/deploy/robots/go2
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# On robot:
./go2_ctrl -n wlan0

# Test scenarios:
# 1. Run velocity policy, induce fall (push robot), verify stabilization recovers
# 2. Transition to bipedal (if mixed FSM supports it), induce fall, verify recovery
# 3. Monitor transition success rate between walking policies and stabilization
```

## Expected Outcomes

### Bipedal Policy
- **More robust** to uneven terrain (bumps, small slopes)
- **Slightly slower training** due to terrain generation overhead
- **Better sim-to-real transfer** when deployed on outdoor surfaces

### Stabilization Policy
- **Universal recovery:** Can stabilize from both bipedal and velocity policy failures
- **Higher initial state coverage:** Samples from wider range of fall scenarios
- **May require more training iterations** due to expanded state space (monitor early reward)

### Integrated System
- **Seamless transitions:** Walking policies (bipedal, velocity) → Stabilization → back to walking
- **Robust outdoor operation:** Combined terrain robustness and fall recovery

## Risk Mitigation

**If bipedal policy struggles to converge:**
- Reduce terrain difficulty: increase flat % in `GO2_VELOCITY_TERRAIN_CFG`
- Consider terrain curriculum: start flat, gradually add roughness after 500-1000 iterations

**If stabilization policy doesn't converge or takes too long:**
- Train in stages: expand z/pitch first, then velocities
- Monitor reward: if near zero after 200 iterations, state space may be too hard
- Consider biased sampling (add explicit bipedal-fall distribution) if uniform is too sparse

**If stabilization fails on one mode (e.g., bipedal but not velocity):**
- Analyze failure modes: which initial states have low success rate?
- Add targeted curriculum or biased sampling for the weak mode
- Fallback: separate stabilization policies per mode (not ideal, but viable)

## Files Modified

1. `/home/milad/Documents/Repos/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/bipedal_env_cfg.py`
   - Lines ~173, 196-217, 352-372, 975-988
   - Newton variants (`RobotBipedalWalkNewtonEnvCfg`, `RobotBipedalWalkNewtonPlayEnvCfg`) automatically inherit all changes

2. `/home/milad/Documents/Repos/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/stabilization_env_cfg.py`
   - Lines 131-161 (pose/velocity/joint ranges)
   - Added Newton variants: `RobotNewtonEnvCfg`, `RobotNewtonPlayEnvCfg` (lines 303-336)

3. `/home/milad/Documents/Repos/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py`
   - Registered new tasks: `Unitree-Go2-Stabilize-Newton`, `Unitree-Go2-Bipedal-Walk-Rough`, `Unitree-Go2-Bipedal-Walk-Rough-Newton`

## Available Tasks

**Bipedal Walking (Flat Terrain - Backward Compatible):**
- `Unitree-Go2-Bipedal-Walk` — PhysX, flat plane (original, works with old policies)
- `Unitree-Go2-Bipedal-Walk-Newton` — Newton, flat plane

**Bipedal Walking (Rough Terrain - NEW):**
- `Unitree-Go2-Bipedal-Walk-Rough` — PhysX, rough terrain (30% flat, 35% random rough, 25% slopes, 10% boxes)
- `Unitree-Go2-Bipedal-Walk-Rough-Newton` — Newton, rough terrain

**Stabilization/Recovery:**
- `Unitree-Go2-Stabilize` — PhysX, expanded DR (covers bipedal + velocity falls)
- `Unitree-Go2-Stabilize-Newton` — Newton, expanded DR

## Next Steps

1. Train new bipedal policy with terrain-enabled config
2. Train new stabilization policy with expanded DR
3. Evaluate both policies in simulation
4. Deploy to robot and test integrated FSM recovery
5. Iterate based on observed failure modes

## Notes

- The changes maintain backward compatibility for play configs (no DR in eval)
- **Newton variants:** Both bipedal and stabilization now have Newton (MuJoCo Warp) variants
  - Bipedal: `Unitree-Go2-Bipedal-Walk-Newton` (already existed, inherits terrain changes automatically)
  - Stabilization: `Unitree-Go2-Stabilize-Newton` (newly added with expanded DR ranges)
- Height scanner remains disabled for bipedal (not used by policy)
- Terrain material DR for ground is handled via `physics_material` in `TerrainImporterCfg`
- Robot body material DR is already active in bipedal policy (PhysX only, disabled for Newton)
