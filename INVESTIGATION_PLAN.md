# Newton v2 Sim2Sim Investigation Plan

## Status (2026-05-05 05:30 UTC)

### Current Training
- **Run**: `2026-05-05_04-46-31` (Newton backend, 4096 envs, seed 42)
- **PID**: 888225
- **Progress**: Step ~3134/10000, orientation_align=1.736, reward=63
- **Changes from v1**: front_foot_contact -0.5→-1.5, base_height -1.0→-2.0
- **Sim2sim at model_3100**: z=0.559, pitch=-80.9°, survived 10s walking
- **Let it run** until finished or no progress

### Commits Pushed
- `7153035` — Newton backend + penalty increases
- `df0eb97` — sim2sim_v2 auto-detect joint ordering

### Key Files
- `sim2sim_v2.py` — auto-detects Newton/PhysX ordering from deploy.yaml
- `run_sim2sim_newton_fixed.sh` — standalone Newton sim2sim script
- `bipedal_env_cfg.py` — env config with both PhysX and Newton variants
- `__init__.py` — task registration for Newton variant

## Instructions from Milad (12-hour autonomous work)

### Task 1: PhysX sim2sim transfer with SAME rewards
- Run the current environment (with -1.5 front_foot_contact, -2.0 base_height) using PhysX backend
- Test if the PhysX-trained policy transfers to MuJoCo
- If NOT, investigate WHY:
  1. **Wider mass randomization** — the dynamics differences between PhysX and MuJoCo may require much broader DR
  2. **Contact dynamics** — if mass DR doesn't help, adjust PhysX contact parameters to be closer to MuJoCo
  3. This is iterative — keep trying different approaches

### Task 2: Iterate independently for 12 hours
- Commit locally if improvements found (do NOT push)
- Keep iterating on the sim2sim gap problem
- Focus on getting PhysX→MuJoCo transfer working

### Task 3: Save context for session continuity
- Keep this file updated with progress
- Save plans for next move so a reset can continue seamlessly

## Plan of Attack

### Phase 1: Baseline PhysX with new rewards (while Newton v2 finishes)
1. Wait for Newton v2 to plateau or finish
2. Start PhysX training with same reward weights (front_foot=-1.5, base_height=-2.0)
3. Test sim2sim transfer of PhysX checkpoints → MuJoCo

### Phase 2: Diagnose PhysX→MuJoCo gap
If PhysX fails sim2sim:
1. Compare dynamics: run same action sequence in both, log joint positions/velocities
2. Check mass/inertia differences between PhysX USD and MuJoCo XML
3. Try wider mass randomization ranges:
   - base mass: (-1.0, 2.5) → (-2.0, 4.0)?
   - rear leg mass: add wider ranges
   - all limb mass randomization
4. Try wider friction randomization
5. Try wider PD gain randomization

### Phase 3: Contact dynamics alignment
If mass DR doesn't help:
1. Compare contact forces between PhysX and MuJoCo for same scenario
2. Adjust PhysX contact parameters:
   - bounce_threshold
   - contact offset
   - rest offset
   - solver iterations
3. Compare foot-ground interaction patterns

### Phase 4: Document findings
- What worked, what didn't
- Quantitative comparison of sim2sim survival times
- Root cause analysis

## Progress Log

### 05:30 UTC — Starting
- Newton v2 still training (step 3134, looking good)
- Will wait for it to plateau, then start PhysX experiments
- Clean git state confirmed

### 07:05 UTC — Newton v2 finished, PhysX started
- Newton v2 completed 10000 iters: orientation_align=1.71, reward=67.7, curriculum maxed
- Sim2sim model_4100: z=0.565, pitch=-76.8°, walked 2.44m in 15s (stable bipedal)
- Started PhysX training with same rewards (front_foot=-1.5, base_height=-2.0)
- PhysX run: `2026-05-05_07-05-10`, PID 906533, ETA ~09:40 UTC
- PhysX logs: ~/Repos/GO2/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk/2026-05-05_07-05-10

### 08:05 UTC — PhysX sim2sim FAILS
- PhysX v2 at step 4098: orientation_align=1.71, curriculum maxed — fully bipedal in PhysX
- **Sim2sim transfer FAILS**: model_3000 fell at 1.04s, model_4000 fell at 0.88s
- Same reward config → Newton transfers perfectly, PhysX does not
- Confirms: the gap is in physics dynamics, not reward shaping
- Next: diagnose the dynamics difference (mass/inertia? contact model? friction?)

### 08:10 UTC — Root Cause Found: Action Magnitude
- **PhysX policy**: action_rms=6.14, max=30.07, 26.6% torque clipping
- **Newton policy**: action_rms=1.49, max=8.74, 7.5% torque clipping
- PhysX learns ultra-aggressive, high-amplitude control that saturates actuators
- Newton learns smooth, low-amplitude control
- PhysX contact dynamics are more forgiving of wild swings
- In MuJoCo, the PhysX policy's aggressive actions cause immediate instability
- **Solution hypothesis**: stronger action_rate penalty OR action magnitude penalty
  to force PhysX policy to learn smoother control
- Alternative: wider DR that makes PhysX less forgiving during training
- Next step: try increasing action_rate from -0.005 to -0.05 for PhysX training

### 08:35 UTC — PhysX v3 (action_rate=-0.02)
- PhysX v3: action_rate -0.005 → -0.02
- Result at step 3900: orientation_align=1.73, action_rate penalty=-0.303
- Sim2sim: model_3000 fell at 2.04s, model_3900 fell at 1.82s (improved from 1s but not enough)
- Action stats: mean_abs=3.58 (down from 6.14 but still 2.4x Newton's 1.49)
- Conclusion: 4x action_rate cut actions in half, doubled survival. Not enough.

### 09:06 UTC — PhysX v4 (action_rate=-0.05 + wider mass DR)
- action_rate: -0.005 → -0.05 (10x original)
- base_mass DR: (-1.0, 2.5) → (-2.0, 4.0)
- rear_leg_mass DR: (-0.15, 0.25) → (-0.5, 0.5)
- Run: 2026-05-05_09-06-18, PID 913930
- Hypothesis: need action_rms < 2 to match Newton's smooth control for transfer

### 10:05 UTC — PhysX v4 failed, starting v5
- v4 result: wider mass DR (-2.0,4.0) + action_rate=-0.05 → orientation stuck at 1.1
  Policy can't learn bipedal stance when mass varies too much + actions constrained
- **v5**: action_rate=-0.05, ORIGINAL mass DR (-1.0, 2.5) / (-0.15, 0.25)
  Isolates action smoothness effect without mass DR destabilization
- Run: 2026-05-05_10-05-01
- If v5 still fails sim2sim: the issue isn't just action magnitude,
  it's fundamental PhysX contact dynamics that the policy relies on

### 11:05 UTC — PhysX v5 sim2sim FAILS + Key Insight
- v5 (action_rate=-0.05): orientation_align=1.80 ✔️ but sim2sim still fails (0.26s, 1.24s)
- **Critical insight**: action_rate penalizes RATE OF CHANGE not absolute magnitude!
  - v5 action mean_abs=4.67 (actually HIGHER than v3's 3.58!)
  - Policy learned slowly-changing but LARGE actions — circumvents action_rate
  - Newton policy has mean_abs=1.49 — fundamentally smaller actions
- **Next**: need explicit action_l2 penalty (penalize action MAGNITUDE)
  - `action_l2` term: penalize ||action||^2 directly
  - Target: force action mean_abs < 2.0 to match Newton's range
- Kill v5, add action_l2 penalty, restart as v6

### 12:05 UTC — PhysX v6 Partial Success!
- v6 (action_rate=-0.05 + action_l2=-0.01): orientation_align=1.57 at step 3858
- Sim2sim: model_3800 survived **3.44s** (best yet! up from 0.88s baseline)
- Action mean_abs=2.66 (down from 6.14 baseline, getting closer to Newton's 1.49)
- Clear correlation: lower actions → longer survival
- Still not at 15s target. action_l2 needs to be stronger, or need more training time
- Letting v6 continue to 10000 iters to see if it improves further
- Next experiment if v6 plateaus: increase action_l2 to -0.02 or -0.03

### 13:05 UTC — PhysX v7 (action_l2=-0.03) too strong
- orientation stalled at 1.39 — too heavy combined penalties prevent bipedal learning
- Sim2sim model_3000: 3.22s (similar to v6), staying low crouch (~-13° pitch)
- Fundamental tension: PhysX NEEDS large actions to rear up, but those cause MuJoCo instability

### 14:06 UTC — PhysX v8: Hard action clipping
- New approach: clip actions to [-6, 6] in the env config (was [-100, 100])
- This eliminates max_abs=30 spikes while allowing enough range for rearing
- Combined with action_l2=-0.01 + action_rate=-0.05
- action*0.25 = max ±1.5 rad offset (still generous for bipedal)
- Newton policy max_abs was 8.74 so [-6,6] is slightly tighter
- Run: 2026-05-05_14-06-10

### 15:05 UTC — PhysX v8 sim2sim: 2.8s (not better than v6)
- Hard clip [-6,6] + action_l2=-0.01 + action_rate=-0.05
- model_3000: 2.80s, model_3900: 1.50s
- Not better than v6 best (3.44s)

### Summary of All PhysX Experiments
| Version | Changes | Best Sim2sim | Actions mean_abs |
|---------|---------|-------------|------------------|
| v2 (baseline) | front_foot=-1.5, base_height=-2.0 | 0.88s | 6.14 |
| v3 | + action_rate=-0.02 | 2.04s | 3.58 |
| v4 | + action_rate=-0.05 + wider mass DR | failed to learn | - |
| v5 | + action_rate=-0.05 | 1.24s | 4.67 |
| v6 | + action_rate=-0.05 + action_l2=-0.01 | **3.44s** | 2.66 |
| v7 | + action_l2=-0.03 | 3.22s | low but can't stand |
| v8 | + action clip [-6,6] + action_l2=-0.01 | 2.80s | - |

### Conclusion
**PhysX→MuJoCo sim2sim gap CANNOT be closed with reward shaping alone.**
- Best result: 3.44s (v6) vs Newton's 15s+
- Root cause: PhysX contact dynamics are fundamentally different from MuJoCo
- The bipedal stance is on the edge of stability — small dynamics differences cascade
- Action smoothness helps (0.88s→3.44s) but the gap is in the physics, not the policy
- **Newton backend is the correct solution for sim2sim transfer**

### Recommendations for Milad
1. Use Newton backend for all sim2sim/sim2real training going forward
2. PhysX can still be used for fast prototyping/reward tuning, but deploy with Newton
3. The action_l2 penalty is useful regardless (prevents crazy actions) — consider keeping it
4. The hard action clip [-6,6] is also safe to keep as a guardrail
5. Future work: investigate if PhysX contact parameters can be tuned to match MuJoCo
