---
name: go2-sim2sim
description: >
  Run sim2sim transfer tests for the Unitree Go2 robot using MuJoCo headless
  simulation with the go2_ctrl C++ deploy stack over DDS. Use when asked to:
  test the deploy FSM, run sim2sim, record a sim2sim video, debug joystick
  transitions, modify the FSM scenario, verify policy deployment, or
  troubleshoot go2_ctrl connectivity. Covers the full pipeline from building
  go2_ctrl to recording labeled videos of FSM state transitions.
---

# Go2 Sim2Sim FSM Test

Run the Go2 C++ deploy controller (`go2_ctrl`) against a headless MuJoCo
simulation with scripted virtual joystick input and video recording.

## Architecture

```
Python (sim2sim script)              C++ (go2_ctrl --network lo)
┌──────────────┐                     ┌──────────────┐
│ MuJoCo (EGL) │──physics──>┐        │ FSM (1kHz)   │
│ DDS Bridge   │ pub rt/lowstate ──> │ sub lowstate  │
│              │ sub rt/lowcmd   <── │ pub lowcmd    │
│ Joystick     │ scripted keys       │ ONNX policies │
│ Video Writer │ labeled frames      │ DSL joystick  │
└──────────────┘                     └──────────────┘
```

Both processes communicate via DDS on domain 0 over loopback (`lo`).

## Prerequisites

```bash
conda activate go2

# Verify these exist:
# - go2_ctrl binary:       deploy/robots/go2/build/go2_ctrl
# - ONNX runtime:          deploy/thirdparty/onnxruntime-linux-x64-*/lib/
# - MuJoCo scene:          $UNITREE_MUJOCO/unitree_robots/go2/scene_flat_offscreen.xml
# - unitree_sdk2py:        pip install unitree_sdk2py
# - Python packages:       mujoco imageio imageio-ffmpeg opencv-python-headless numpy
```

If `go2_ctrl` is not built, build it:
```bash
cd deploy/robots/go2/build
cmake .. && make -j$(nproc)
```

CMakeLists.txt must have unitree_sdk2 paths configured. If cmake fails to find
unitree_sdk2, add:
```cmake
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)
```

## Running

```bash
cd deploy/robots/go2
MUJOCO_GL=egl python3 sim2sim_fsm_test.py
```

Output: `/tmp/sim2sim_fsm_test.mp4`

The script is at `deploy/robots/go2/sim2sim_fsm_test.py`. Edit `FSM_SCENARIO`
inside to change phases, durations, or velocity commands.

## Scenario Format

Each entry in `FSM_SCENARIO`:
```python
(label, duration_s, keys_to_press, lx, ly, rx, ry)
```

- `keys_to_press`: button names from `KEY_MAP` (e.g. `["L2", "A"]`, `["start"]`)
- `ly`: forward/backward velocity (positive = forward)
- `lx`: lateral velocity (positive = robot moves left)
- `rx`: yaw rate (positive = turn left)

Axis mapping comes from `State_RLBase.cpp`:
```
cmd_vx = joystick->ly()     # forward
cmd_vy = -joystick->lx()    # lateral
cmd_wz = -joystick->rx()    # yaw
```

## FSM Transitions (from config.yaml)

| From | To | Command |
|------|----|---------|
| Passive | FixStand | LT + A |
| FixStand | Passive | LT + B |
| FixStand | Velocity | LT + down |
| FixStand | VelocityGuarded | LB + down |
| FixStand | BipedalStandUp | LT + up |
| FixStand | BipedalStandUpGuarded | LB + up |
| FixStand | Stabilize | start |
| BipedalStandUp | BipedalRear | auto (2s) |
| Any -> Passive | (on DDS timeout >1s) | |

All transitions use `.on_pressed` (edge-triggered, single tick).

## Critical: LT/RT Axis Smoothing

**LT and RT are `Axis` type, not `Button`.** They have exponential smoothing:

```cpp
// smooth=0.03, threshold=0.5
new_data = data_ * 0.97 + input * 0.03;
is_pressed = (new_data > 0.5);  // takes ~23 ticks to ramp
```

**All other buttons (A, B, L1, start, up, down, etc.) are instant.**

For any combo using L2 (LT) or R2 (RT), you MUST stage the press:

1. Press L2 alone for ~50-100ms (ramp the Axis past 0.5 threshold)
2. Then add the combo button (A, B, up, etc.) so its `on_pressed` fires
   while LT is already `pressed=true`

If you press both simultaneously, `A.on_pressed` fires at tick N but
`LT.pressed` is still false (hasn't ramped). By tick N+23, `A.on_pressed`
is consumed. **The transition never fires.**

The script handles this automatically in the key injection logic.

## Joystick Byte Encoding

The `wireless_remote` field in `rt/lowstate` is 40 bytes (`REMOTE_DATA_RX`):

```
Offset  Type      Field
0-1     uint8[2]  head (unused)
2-3     uint16    BtnUnion (little-endian)
4-7     float32   lx
8-11    float32   rx
12-15   float32   ry
16-19   float32   L2 (trigger axis)
20-23   float32   ly
24-39   padding
```

BtnUnion bit layout (uint16, LE):
```
Bit 0:R1  1:L1  2:Start  3:Select  4:R2  5:L2  6:f1  7:f2
Bit 8:A   9:B   10:X     11:Y      12:up 13:right 14:down 15:left
```

Example encoding L2+A: `btn_val = (1<<5)|(1<<8) = 0x0120`
→ `wireless_remote[2] = 0x20, wireless_remote[3] = 0x01`

## Timing Requirements

- **Warmup**: go2_ctrl needs ~7-10s to load ONNX policies. Publish lowstate
  continuously during warmup before starting the scenario.
- **Real-time pacing**: go2_ctrl FSM runs at 1kHz wall-clock. The MuJoCo sim
  must be paced to match (`time.sleep` when sim_time > wall_elapsed).
- **Continuous publishing**: If no lowstate arrives for >1s, `isTimeout()`
  triggers and ALL states transition to Passive.
- **Sim time offset**: Physics accumulates time during warmup. Record
  `d.time` after warmup and subtract it from scenario timing.

## Policy Configuration

Policy paths are in `deploy/robots/go2/config/config.yaml`. Each FSM state
has a `policy_dir` field pointing to a `0_best/` directory:

```yaml
BipedalRear:
  policy_dir: ../../../logs/rsl_rl/unitree_go2_bipedal_walk/0_best
BipedalStandUp:
  policy_dir: ../../../logs/rsl_rl/unitree_go2_bipedal_standup/0_best
```

Each `0_best/` must contain `exported/policy.onnx` and `params/deploy.yaml`.

To export a new policy: use `scripts/export_all_policies.sh` or the standalone
`scripts/export_policy_standalone.py --jit_path <path>`.

## Troubleshooting

**Robot stuck in Passive**: Check go2_ctrl log for `FSM: Start Passive`.
If no `Change state` messages, the joystick encoding or timing is wrong.
Add debug logging in `main.cpp` to print `joy.LT.pressed`, `joy.A.on_pressed`.

**go2_ctrl exits immediately**: Missing ONNX policies or onnxruntime lib.
Set `LD_LIBRARY_PATH` to include the onnxruntime lib directory.

**No video output / black frames**: `MUJOCO_GL=egl` must be set. The scene
XML needs `offwidth`/`offheight` attributes in `<visual>`.

**DDS connection timeout**: Both processes must use the same domain (0) and
interface (`lo`). Check for stale DDS processes from previous runs.

## Label Rendering

Use only ASCII in video labels — OpenCV `putText` does not support Unicode.
Use `->` instead of `→`.

## Files

| File | Purpose |
|------|---------|
| `deploy/robots/go2/sim2sim_fsm_test.py` | Main sim2sim script |
| `deploy/robots/go2/config/config.yaml` | FSM config + policy paths |
| `deploy/robots/go2/build/go2_ctrl` | C++ controller binary |
| `deploy/robots/go2/SIM2SIM.md` | Detailed technical reference |
