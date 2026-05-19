#!/usr/bin/env python3
import os
os.environ['MUJOCO_GL'] = 'egl'
"""
Sim2Sim FSM test: headless MuJoCo + DDS bridge + virtual joystick + video.

Runs Go2 in MuJoCo with the go2_ctrl control stack over DDS, injecting
scripted joystick commands that walk through the FSM scenario:

  Passive → FixStand → VelocityGuarded (quad walk) → Stabilize → 
  BipedalStandUp → BipedalRear (bipedal walk) → FixStand → Passive

Each section is labeled in the output video for clarity.
"""

import os
import sys
import time
import struct
import threading
import numpy as np
import mujoco
import imageio
import subprocess
import signal

# --------------------------------------------------------------------------
# DDS imports (unitree_sdk2_python)
# --------------------------------------------------------------------------
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
    ChannelPublisher,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as Go2LowCmd
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as Go2LowState
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowState_ as LowState_default,
    unitree_go_msg_dds__SportModeState_ as SportModeState_default,
    unitree_go_msg_dds__WirelessController_ as WirelessController_default,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MUJOCO_SCENE = os.path.expanduser(
    "~/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene_flat_offscreen.xml"
)
DEPLOY_DIR = os.path.expanduser(
    "~/Repos/GO2/unitree_rl_lab/deploy/robots/go2"
)
GO2_CTRL = os.path.join(DEPLOY_DIR, "build", "go2_ctrl")
ONNX_RT_LIB = os.path.expanduser(
    "~/Repos/GO2/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib"
)

DOMAIN_ID = 0
INTERFACE = "lo"
SIM_DT = 0.001  # MuJoCo timestep
CTRL_DT = 0.001  # Bridge publish rate (matches C++ bridge at 1kHz)
RENDER_FPS = 30
RENDER_W = 1280
RENDER_H = 720
OUTPUT_VIDEO = "/tmp/sim2sim_fsm_test.mp4"

# --------------------------------------------------------------------------
# Joystick key bitmap (from unitree_sdk2py_bridge.py)
# --------------------------------------------------------------------------
KEY_MAP = {
    "R1": 0,   "L1": 1,   "start": 2, "select": 3,
    "R2": 4,   "L2": 5,   "F1": 6,    "F2": 7,
    "A": 8,    "B": 9,    "X": 10,    "Y": 11,
    "up": 12,  "right": 13, "down": 14, "left": 15,
}


def make_keys(*names):
    """Build a 16-bit key bitmap from named buttons."""
    val = 0
    for n in names:
        val |= 1 << KEY_MAP[n]
    return val


# --------------------------------------------------------------------------
# FSM Scenario definition
# --------------------------------------------------------------------------
# Each step: (label, duration_s, keys_to_press, lx, ly, rx, ry)
#
# Axis mapping (from State_RLBase.cpp):
#   cmd_vx = joystick->ly()    → ly = forward/backward
#   cmd_vy = -joystick->lx()   → lx = lateral (positive lx = move left)
#   cmd_wz = -joystick->rx()   → rx = yaw rate (positive rx = turn left)
#
# Scenario order: Bipedal first, then stabilize, then quadruped.
FSM_SCENARIO = [
    # --- Phase 1: Bipedal ---
    # 1. Passive -> FixStand (LT+A)
    ("Passive -> FixStand", 4.0, ["L2", "A"], 0, 0, 0, 0),
    # 2. FixStand -> BipedalStandUp (LT+up, auto->BipedalRear after 2s)
    ("FixStand -> Bipedal StandUp", 1.0, ["L2", "up"], 0, 0, 0, 0),
    # 3. Standup + auto-transition to walk
    ("Bipedal StandUp -> Walk", 4.0, [], 0, 0, 0, 0),
    # 4. Bipedal walk forward (ly=0.3, gentle)
    ("Bipedal Walk (forward)", 8.0, [], 0, 0.3, 0, 0),
    # 5. Bipedal walk backward (ly=-0.2, gentle)
    ("Bipedal Walk (backward)", 5.0, [], 0, -0.2, 0, 0),
    # 6. Bipedal walk turn (forward + yaw)
    ("Bipedal Walk (turn)", 6.0, [], 0, 0.2, 0.3, 0),
    # 7. Bipedal -> FixStand (LT+A)
    ("Bipedal Walk -> FixStand", 1.0, ["L2", "A"], 0, 0, 0, 0),
    # 8. Rest in FixStand
    ("FixStand (rest)", 3.0, [], 0, 0, 0, 0),

    # --- Phase 2: Stabilize (from FixStand) ---
    # 9. FixStand -> Stabilize (start)
    ("FixStand -> Stabilize", 1.0, ["start"], 0, 0, 0, 0),
    # 10. Stabilize recovery hold
    ("Stabilize (recovery)", 4.0, [], 0, 0, 0, 0),
    # 11. Stabilize -> FixStand (LT+A)
    ("Stabilize -> FixStand", 1.0, ["L2", "A"], 0, 0, 0, 0),
    # 12. Rest in FixStand
    ("FixStand (rest)", 2.0, [], 0, 0, 0, 0),

    # --- Phase 3: Quadruped ---
    # 13. FixStand -> VelocityGuarded (LB+down)
    ("FixStand -> Quad Walk", 1.0, ["L1", "down"], 0, 0, 0, 0),
    # 14. Quad walk forward (ly > 0)
    ("Quad Walk (forward)", 5.0, [], 0, 0.5, 0, 0),
    # 15. Quad walk backward (ly < 0)
    ("Quad Walk (backward)", 3.0, [], 0, -0.3, 0, 0),
    # 16. Quad walk turn (forward + yaw)
    ("Quad Walk (turn)", 3.0, [], 0, 0.3, 0.5, 0),
    # 17. Quad -> FixStand (LT+A)
    ("Quad Walk -> FixStand", 1.0, ["L2", "A"], 0, 0, 0, 0),
    # 18. Final FixStand
    ("FixStand (final)", 2.0, [], 0, 0, 0, 0),
    # 19. FixStand -> Passive (LT+B)
    ("FixStand -> Passive", 1.0, ["L2", "B"], 0, 0, 0, 0),
    # 20. End
    ("Passive (end)", 2.0, [], 0, 0, 0, 0),
]


class VirtualJoystick:
    """Manages virtual joystick state with timed key pulses."""

    def __init__(self):
        self.keys = 0
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self._pulse_until = 0.0

    def set_pulse(self, key_names, duration=0.15):
        """Press keys for a brief pulse."""
        self.keys = make_keys(*key_names) if key_names else 0
        self._pulse_until = time.time() + duration

    def set_axes(self, lx=0, ly=0, rx=0, ry=0):
        self.lx = float(lx)
        self.ly = float(ly)
        self.rx = float(rx)
        self.ry = float(ry)

    def update(self):
        """Call periodically — clears keys after pulse expires."""
        if time.time() > self._pulse_until:
            self.keys = 0

    def encode_wireless_remote(self):
        """Encode into 40-byte wireless_remote format for low_state.

        Layout (REMOTE_DATA_RX / BtnDataStruct):
          bytes 0-1:  head[2]  (unused)
          bytes 2-3:  BtnUnion (uint16_t little-endian bitfield)
                      bit 0: R1,  bit 1: L1,  bit 2: Start,  bit 3: Select,
                      bit 4: R2,  bit 5: L2,  bit 6: f1,     bit 7: f2,
                      bit 8: A,   bit 9: B,   bit10: X,      bit11: Y,
                      bit12: up,  bit13: right, bit14: down,  bit15: left
          bytes 4-7:  float lx
          bytes 8-11: float rx
          bytes 12-15: float ry
          bytes 16-19: float L2 (trigger value, not used here)
          bytes 20-23: float ly
          bytes 24-39: padding
        """
        buf = bytearray(40)

        # Build the 16-bit button field matching BtnUnion bit positions
        BTN_BITS = {
            "R1": 0, "L1": 1, "start": 2, "select": 3,
            "R2": 4, "L2": 5, "F1": 6, "F2": 7,
            "A": 8,  "B": 9,  "X": 10, "Y": 11,
            "up": 12, "right": 13, "down": 14, "left": 15,
        }
        btn_val = 0
        for name, bit in BTN_BITS.items():
            if self.keys & (1 << KEY_MAP[name]):
                btn_val |= (1 << bit)
        struct.pack_into("<H", buf, 2, btn_val)  # little-endian uint16

        # Axes as float32 (little-endian)
        struct.pack_into("<f", buf, 4, self.lx)
        struct.pack_into("<f", buf, 8, self.rx)
        struct.pack_into("<f", buf, 12, self.ry)
        # L2 trigger at bytes 16-19 — set to 1.0 when L2 is pressed
        l2_val = 1.0 if (self.keys & (1 << KEY_MAP["L2"])) else 0.0
        struct.pack_into("<f", buf, 16, l2_val)
        struct.pack_into("<f", buf, 20, self.ly)
        return list(buf)


class DDSBridge:
    """Minimal DDS bridge: receives lowcmd, publishes lowstate + joystick."""

    def __init__(self, mj_model, mj_data, joystick):
        self.m = mj_model
        self.d = mj_data
        self.joy = joystick
        self.num_motor = mj_model.nu
        self.lock = threading.Lock()
        self.running = True

        # State message
        self.low_state = LowState_default()

        # Publishers
        self.state_pub = ChannelPublisher("rt/lowstate", Go2LowState)
        self.state_pub.Init()

        self.high_state = SportModeState_default()
        self.high_pub = ChannelPublisher("rt/sportmodestate", SportModeState_)
        self.high_pub.Init()

        self.wc = WirelessController_default()
        self.wc_pub = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
        self.wc_pub.Init()

        # Subscriber
        self.cmd_sub = ChannelSubscriber("rt/lowcmd", Go2LowCmd)
        self.cmd_sub.Init(self._on_cmd, 10)

        # Find sensor addresses
        self.dim_motor_sensor = 3 * self.num_motor
        self._find_imu_sensors()

    def _find_imu_sensors(self):
        self.imu_quat_adr = -1
        self.imu_gyro_adr = -1
        self.imu_acc_adr = -1
        self.frame_pos_adr = -1
        self.frame_vel_adr = -1
        for i in range(self.m.nsensor):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = self.m.sensor_adr[i]
            if name == "imu_quat":
                self.imu_quat_adr = adr
            elif name == "imu_gyro":
                self.imu_gyro_adr = adr
            elif name == "imu_acc":
                self.imu_acc_adr = adr
            elif name == "frame_pos":
                self.frame_pos_adr = adr
            elif name == "frame_vel":
                self.frame_vel_adr = adr

    def _on_cmd(self, msg):
        with self.lock:
            for i in range(self.num_motor):
                mc = msg.motor_cmd[i]
                self.d.ctrl[i] = (
                    mc.tau
                    + mc.kp * (mc.q - self.d.sensordata[i])
                    + mc.kd * (mc.dq - self.d.sensordata[i + self.num_motor])
                )

    def publish_state(self):
        """Publish lowstate + wireless controller. Call at ~1kHz."""
        sd = self.d.sensordata
        nm = self.num_motor

        for i in range(nm):
            self.low_state.motor_state[i].q = sd[i]
            self.low_state.motor_state[i].dq = sd[i + nm]
            self.low_state.motor_state[i].tau_est = sd[i + 2 * nm]

        if self.imu_quat_adr >= 0:
            q = self.low_state.imu_state.quaternion
            q[0] = sd[self.imu_quat_adr]
            q[1] = sd[self.imu_quat_adr + 1]
            q[2] = sd[self.imu_quat_adr + 2]
            q[3] = sd[self.imu_quat_adr + 3]
            w, x, y, z = q[0], q[1], q[2], q[3]
            self.low_state.imu_state.rpy[0] = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
            self.low_state.imu_state.rpy[1] = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
            self.low_state.imu_state.rpy[2] = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))

        if self.imu_gyro_adr >= 0:
            g = self.low_state.imu_state.gyroscope
            g[0] = sd[self.imu_gyro_adr]
            g[1] = sd[self.imu_gyro_adr + 1]
            g[2] = sd[self.imu_gyro_adr + 2]

        if self.imu_acc_adr >= 0:
            a = self.low_state.imu_state.accelerometer
            a[0] = sd[self.imu_acc_adr]
            a[1] = sd[self.imu_acc_adr + 1]
            a[2] = sd[self.imu_acc_adr + 2]

        self.low_state.tick = int(self.d.time / 1e-3)

        # Joystick into wireless_remote
        wr = self.joy.encode_wireless_remote()
        for i in range(min(len(wr), 40)):
            self.low_state.wireless_remote[i] = wr[i]

        self.state_pub.Write(self.low_state)

        # High state
        if self.frame_pos_adr >= 0:
            self.high_state.position[0] = sd[self.frame_pos_adr]
            self.high_state.position[1] = sd[self.frame_pos_adr + 1]
            self.high_state.position[2] = sd[self.frame_pos_adr + 2]
        if self.frame_vel_adr >= 0:
            self.high_state.velocity[0] = sd[self.frame_vel_adr]
            self.high_state.velocity[1] = sd[self.frame_vel_adr + 1]
            self.high_state.velocity[2] = sd[self.frame_vel_adr + 2]
        self.high_pub.Write(self.high_state)

        # Wireless controller
        self.joy.update()
        self.wc.keys = self.joy.keys
        self.wc.lx = self.joy.lx
        self.wc.ly = self.joy.ly
        self.wc.rx = self.joy.rx
        self.wc.ry = self.joy.ry
        self.wc_pub.Write(self.wc)


def add_label(frame, text, y=40, font_scale=1.2, color=(255, 255, 255)):
    """Add text label to a frame using OpenCV."""
    import cv2
    # Black background bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], y + 20), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2, cv2.LINE_AA)
    return frame


def render_frame(m, d, renderer, camera):
    """Render a frame using MuJoCo offscreen renderer."""
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None,
                           camera, mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, RENDER_W, RENDER_H),
                      renderer.scene, renderer.con)
    pixels = np.empty((RENDER_H, RENDER_W, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(pixels, None,
                          mujoco.MjrRect(0, 0, RENDER_W, RENDER_H), renderer.con)
    return np.flipud(pixels)


def main():
    print(f"Loading MuJoCo scene: {MUJOCO_SCENE}")
    m = mujoco.MjModel.from_xml_path(MUJOCO_SCENE)
    d = mujoco.MjData(m)
    m.opt.timestep = SIM_DT

    # Set up offscreen renderer
    renderer = mujoco.Renderer(m, RENDER_H, RENDER_W)

    # Camera setup — tracking the robot
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    cam.distance = 3.0
    cam.azimuth = 135
    cam.elevation = -25
    cam.lookat[:] = [0, 0, 0.3]

    # Initialize DDS
    print("Initializing DDS...")
    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)
    time.sleep(0.5)

    joy = VirtualJoystick()
    bridge = DDSBridge(m, d, joy)
    print("DDS bridge ready.")

    # Start go2_ctrl
    print(f"Starting go2_ctrl: {GO2_CTRL}")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ONNX_RT_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    ctrl_log_path = "/tmp/go2_ctrl_output.log"
    ctrl_log = open(ctrl_log_path, "w")
    ctrl_proc = subprocess.Popen(
        [GO2_CTRL, "--network", "lo"],
        cwd=DEPLOY_DIR,
        env=env,
        stdout=ctrl_log,
        stderr=subprocess.STDOUT,
    )

    # Warm-up: run physics + publish lowstate so go2_ctrl can connect.
    # go2_ctrl waits for rt/lowstate before initializing the FSM.
    print("Warming up (publishing lowstate so go2_ctrl can connect)...")
    warmup_start = time.time()
    warmup_duration = 10.0  # go2_ctrl needs ~7s to load all ONNX policies
    while time.time() - warmup_start < warmup_duration:
        with bridge.lock:
            mujoco.mj_step(m, d)
        bridge.publish_state()
        # Check if go2_ctrl died
        if ctrl_proc.poll() is not None:
            ctrl_log.close()
            print("ERROR: go2_ctrl exited during warmup!")
            with open(ctrl_log_path) as f:
                print(f.read()[:3000])
            return

    # Verify FSM started by checking log
    ctrl_log.flush()
    with open(ctrl_log_path) as f:
        log_text = f.read()
    if "FSM: Start" not in log_text:
        print("WARNING: go2_ctrl may not have started FSM yet. Continuing anyway...")
    else:
        print("go2_ctrl FSM started.")

    print("Beginning FSM scenario...")

    # Video writer
    writer = imageio.get_writer(OUTPUT_VIDEO, fps=RENDER_FPS, codec="libx264", quality=8)
    render_interval = 1.0 / RENDER_FPS
    last_render = 0.0

    total_time = sum(s[1] for s in FSM_SCENARIO)
    scenario_start = time.time()
    sim_time_offset = d.time  # offset to account for warmup physics time
    step_idx = 0
    step_start = 0.0
    current_label = ""

    # Advance scenario timing
    step_times = []
    t = 0.0
    for label, dur, keys, lx, ly, rx, ry in FSM_SCENARIO:
        step_times.append((t, t + dur, label, keys, lx, ly, rx, ry))
        t += dur

    print(f"Total scenario duration: {total_time:.1f}s")
    print(f"Output: {OUTPUT_VIDEO}")
    print()

    sim_time = 0.0
    frame_count = 0

    try:
        while sim_time < total_time + 1.0:
            # Find current scenario step
            current_step = None
            for start_t, end_t, label, keys, lx, ly, rx, ry in step_times:
                if start_t <= sim_time < end_t:
                    current_step = (label, keys, lx, ly, rx, ry, start_t)
                    break

            if current_step:
                label, keys, lx, ly, rx, ry, start_t = current_step
                elapsed_in_step = sim_time - start_t
                if keys:
                    has_l2 = "L2" in keys
                    other_keys = [k for k in keys if k != "L2"]
                    if has_l2:
                        # LT is an Axis with smooth=0.03, threshold=0.5.
                        # It takes ~23 ticks (23ms) to ramp from 0→0.5.
                        # Strategy: press L2 alone first (0.05-0.10s),
                        # then add the combo key so its on_pressed fires
                        # while LT.pressed is already true.
                        if 0.05 < elapsed_in_step < 0.10:
                            # Phase A: L2 only (ramp the axis)
                            joy.keys = make_keys("L2")
                        elif 0.10 <= elapsed_in_step < 0.50:
                            # Phase B: L2 + combo key
                            joy.keys = make_keys(*keys)
                        else:
                            joy.keys = 0
                    else:
                        # No L2 — simple pulse
                        if 0.05 < elapsed_in_step < 0.35:
                            joy.keys = make_keys(*keys)
                        else:
                            joy.keys = 0
                else:
                    joy.keys = 0
                # Set axes (only after key pulse settles)
                if elapsed_in_step > 0.55:
                    joy.set_axes(lx, ly, rx, ry)
                else:
                    joy.set_axes(0, 0, 0, 0)
                current_label = label
            else:
                joy.set_axes(0, 0, 0, 0)
                joy.keys = 0

            # Physics step
            with bridge.lock:
                mujoco.mj_step(m, d)

            # Publish DDS state at ~1kHz (every step)
            bridge.publish_state()

            sim_time = d.time - sim_time_offset

            # Render at video FPS
            if sim_time - last_render >= render_interval:
                renderer.update_scene(d, cam)
                pixels = renderer.render()
                # Add label
                frame = add_label(pixels.copy(), current_label)
                # Add time counter
                import cv2
                time_text = f"t={sim_time:.1f}s"
                cv2.putText(frame, time_text, (RENDER_W - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                writer.append_data(frame)
                frame_count += 1
                last_render = sim_time

                if frame_count % (RENDER_FPS * 5) == 0:
                    print(f"  t={sim_time:.1f}s | {current_label}")

            # Real-time pacing: keep sim time ~ wall-clock time
            # so go2_ctrl's 1kHz FSM loop stays in sync.
            wall_elapsed = time.time() - scenario_start
            if sim_time > wall_elapsed:
                time.sleep(sim_time - wall_elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        writer.close()
        ctrl_proc.send_signal(signal.SIGINT)
        time.sleep(0.5)
        ctrl_proc.kill()
        ctrl_proc.wait()
        ctrl_log.close()

    size_mb = os.path.getsize(OUTPUT_VIDEO) / 1024 / 1024
    print(f"\nDone! Saved {OUTPUT_VIDEO} ({size_mb:.1f} MB, {frame_count} frames)")
    print("\n--- go2_ctrl output ---")
    with open("/tmp/go2_ctrl_output.log") as f:
        print(f.read()[-3000:])


if __name__ == "__main__":
    main()
