# Sim2Sim FSM Test Guide

End-to-end guide for running the Go2 deploy stack (go2_ctrl) in MuJoCo headless simulation with scripted joystick input and video capture.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Python (sim2sim_fsm_test.py)                   │
│                                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │
│  │ MuJoCo   │  │ DDS Bridge│  │ Virtual      │ │
│  │ Physics  │──│ pub: rt/  │──│ Joystick     │ │
│  │ (EGL)    │  │  lowstate │  │ (scripted)   │ │
│  │          │  │ sub: rt/  │  │              │ │
│  │          │  │  lowcmd   │  │              │ │
│  └──────────┘  └───────────┘  └──────────────┘ │
│       │                                         │
│  ┌──────────┐                                   │
│  │ Video    │  imageio + OpenCV labels           │
│  │ Writer   │                                   │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘
        │ DDS (domain 0, loopback)
        ▼
┌─────────────────────────────────────────────────┐
│  C++ (go2_ctrl --network lo)                    │
│                                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │
│  │ FSM      │  │ ONNX      │  │ Joystick     │ │
│  │ Engine   │──│ Policies  │  │ DSL Parser   │ │
│  │ (1kHz)   │  │ (standup, │  │ (from YAML)  │ │
│  │          │  │  walk,    │  │              │ │
│  │          │  │  etc.)    │  │              │ │
│  └──────────┘  └───────────┘  └──────────────┘ │
└─────────────────────────────────────────────────┘
```

## Prerequisites

### System
- Linux x86_64, NVIDIA GPU with EGL support
- Conda environment `go2` with Python 3.10

### Installed Dependencies
```bash
# unitree_sdk2 (C++ DDS library)
# Installed to /opt/unitree_robotics
# Headers: /opt/unitree_robotics/include/
# Libs: /opt/unitree_robotics/lib/

# MuJoCo 3.3.6
# Installed to ~/.mujoco/mujoco-3.3.6

# unitree_sdk2py (Python DDS bindings)
pip install unitree_sdk2py

# Python packages
pip install mujoco imageio imageio-ffmpeg opencv-python-headless numpy
```

### Built Artifacts
```bash
# go2_ctrl (C++ deploy controller)
cd deploy/robots/go2/build
cmake .. && make -j$(nproc)
# Binary: deploy/robots/go2/build/go2_ctrl

# ONNX Runtime (for go2_ctrl)
# At: deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib/
```

### Required Policy Files (ONNX exports in 0_best/exported/)
- `logs/rsl_rl/unitree_go2_bipedal_walk/0_best/` — bipedal walking
- `logs/rsl_rl/unitree_go2_bipedal_standup/0_best/` — bipedal standup
- `logs/rsl_rl/unitree_go2_velocity_rough/0_best/` — quadruped walking
- `logs/rsl_rl/unitree_go2_stabilize/0_best/` — recovery/stabilize

Each `0_best/` dir must contain:
- `exported/policy.onnx` — the ONNX model
- `params/deploy.yaml` — joint mappings, PD gains, etc.

### MuJoCo Scene
```
~/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene_flat_offscreen.xml
```
Must have `<visual>` with `offwidth="1280" offheight="720"` for EGL rendering.

## Running

```bash
conda activate go2
cd ~/Repos/GO2/unitree_rl_lab/deploy/robots/go2
MUJOCO_GL=egl python3 /tmp/sim2sim_fsm_test.py
```

Output: `/tmp/sim2sim_fsm_test.mp4`

## Critical Technical Details

### Joystick Encoding (wireless_remote)

The `rt/lowstate` DDS message has a 40-byte `wireless_remote` field encoding joystick state as `REMOTE_DATA_RX`:

```
Offset  Type      Field
0-1     uint8[2]  head (unused)
2-3     uint16    BtnUnion (little-endian bitfield)
4-7     float32   lx (lateral axis)
8-11    float32   rx (yaw axis)
12-15   float32   ry (unused)
16-19   float32   L2 (trigger axis value)
20-23   float32   ly (forward/backward axis)
24-39   padding
```

**BtnUnion bit layout** (uint16_t, little-endian):
| Bit | Button | Bit | Button |
|-----|--------|-----|--------|
| 0   | R1/RB  | 8   | A      |
| 1   | L1/LB  | 9   | B      |
| 2   | Start  | 10  | X      |
| 3   | Select | 11  | Y      |
| 4   | R2/RT  | 12  | up     |
| 5   | L2/LT  | 13  | right  |
| 6   | f1     | 14  | down   |
| 7   | f2     | 15  | left   |

### ⚠️ LT/RT Axis Smoothing Bug

**LT and RT are `Axis` type in UnitreeJoystick, NOT `Button`.**

The `Axis` class applies exponential smoothing:
```cpp
float new_data = data_ * 0.97 + input * 0.03;  // smooth=0.03
bool is_pressed = (new_data > 0.5);              // threshold=0.5
```

This means LT takes **~23 FSM ticks (23ms)** to ramp from 0 to the 0.5 threshold.

**Problem**: For combo presses like `LT + A.on_pressed`, if you press both simultaneously, A's `on_pressed` fires instantly (it's a `Button`) but LT hasn't ramped past threshold yet. By the time LT is ready, `A.on_pressed` has been consumed.

**Solution**: Stage the press — set L2 alone for ~100ms first, THEN add the combo button:
```python
# Phase A: L2 only (ramp the axis)
if 0.05 < elapsed < 0.10:
    keys = L2_only
# Phase B: L2 + combo key  
elif 0.10 <= elapsed < 0.50:
    keys = L2 + combo_key
```

All other buttons (A, B, L1, start, up, down, etc.) are `Button<int>` type and fire instantly.

### Axis-to-Velocity Mapping

From `State_RLBase.cpp`:
```cpp
cmd_vx = joystick->ly();     // ly = forward/backward
cmd_vy = -joystick->lx();    // lx = lateral  
cmd_wz = -joystick->rx();    // rx = yaw rate
```

### Real-Time Pacing

**go2_ctrl's FSM runs at 1kHz wall-clock time.** The Python sim must be paced to match:
```python
wall_elapsed = time.time() - scenario_start
if sim_time > wall_elapsed:
    time.sleep(sim_time - wall_elapsed)
```

Without this, physics runs faster than real-time and go2_ctrl's timing gets desynced.

### Warmup Phase

go2_ctrl takes ~7-10s to:
1. Connect to `rt/lowstate` DDS topic
2. Load all ONNX policies
3. Initialize FSM states

The sim must publish lowstate continuously during this warmup before starting the scenario.

### Timeout Protection

`SubscriptionBase::isTimeout()` (default 1000ms) is registered as a transition check for ALL FSM states. If go2_ctrl doesn't receive a lowstate message within 1s, every state transitions to Passive. The sim must publish continuously.

## FSM Transition Reference (from config.yaml)

| From State | To State | Joystick Command |
|------------|----------|------------------|
| Passive | FixStand | LT + A.on_pressed |
| FixStand | Passive | LT + B.on_pressed |
| FixStand | Velocity | LT + down.on_pressed |
| FixStand | VelocityGuarded | LB + down.on_pressed |
| FixStand | BipedalStandUp | LT + up.on_pressed |
| FixStand | BipedalStandUpGuarded | LB + up.on_pressed |
| FixStand | Stabilize | start.on_pressed |
| BipedalStandUp | BipedalRear | auto (2s) |
| BipedalRear | FixStand | LT + A.on_pressed |
| BipedalRear | Stabilize | start.on_pressed |
| VelocityGuarded | FixStand | LT + A.on_pressed |
| Stabilize | FixStand | LT + A.on_pressed |

## Modifying the Scenario

Edit `FSM_SCENARIO` in the script. Each entry:
```python
(label, duration_s, keys_to_press, lx, ly, rx, ry)
```

- `keys_to_press`: list of button names from KEY_MAP (e.g., `["L2", "A"]`)
- `lx`: lateral velocity (positive = robot moves left)
- `ly`: forward/backward (positive = forward)
- `rx`: yaw rate (positive = turn left)
- `ry`: unused

## File Locations

| File | Purpose |
|------|---------|
| `/tmp/sim2sim_fsm_test.py` | Main sim2sim script |
| `deploy/robots/go2/config/config.yaml` | FSM config, policy paths, transitions |
| `deploy/robots/go2/build/go2_ctrl` | C++ deploy controller binary |
| `deploy/include/FSM/CtrlFSM.h` | FSM engine |
| `deploy/include/FSM/FSMState.h` | State base with joystick DSL transitions |
| `deploy/include/unitree_joystick_dsl.hpp` | Joystick condition DSL compiler |
| `/opt/unitree_robotics/include/unitree/dds_wrapper/common/unitree_joystick.hpp` | UnitreeJoystick, Axis, Button, REMOTE_DATA_RX |
| `~/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene_flat_offscreen.xml` | MuJoCo scene |
