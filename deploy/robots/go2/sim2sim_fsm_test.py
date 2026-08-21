#!/usr/bin/env python3
import os
os.environ['MUJOCO_GL'] = 'egl'
"""
Sim2Sim FSM test: headless MuJoCo + DDS bridge + virtual joystick + video.

Runs Go2 in MuJoCo with the go2_ctrl control stack over DDS, injecting
scripted joystick commands that walk through the current FSM scenario:

  Passive → FixStand → BipedalVelocityDVI → FixStand → Passive

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
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MUJOCO_SCENE = os.path.join(
    os.path.dirname(REPO_ROOT), "unitree_mujoco", "unitree_robots", "go2", "scene_flat.xml"
)
DEPLOY_DIR = os.path.join(REPO_ROOT, "deploy", "robots", "go2")
GO2_CTRL = os.environ.get("GO2_CTRL", os.path.join(DEPLOY_DIR, "build", "go2_ctrl"))
ONNX_RT_LIB = os.path.join(
    REPO_ROOT, "deploy", "thirdparty", "onnxruntime-linux-x64-1.22.0", "lib"
)

DOMAIN_ID = 0
INTERFACE = "lo"
# Match the successful standalone transfer's scene physics exactly.
SIM_DT = 0.002
CTRL_DT = 0.002
# The C++ controller runs at 1 kHz but only needs fresh robot state at 100 Hz.
# Publishing every physics step previously accumulated DDS samples and delayed
# each virtual joystick edge until after the visual scenario had completed.
STATE_PUBLISH_DT = 0.01
RENDER_FPS = 30
# scene_flat.xml declares a 640×480 offscreen framebuffer.
RENDER_W = 640
RENDER_H = 480
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
# Select the policy path without rewriting the test.  Default remains the
# bipedal regression; SIM2SIM_POLICY=velocity exercises the quadrupedal DVI.
TEST_POLICY = os.environ.get("SIM2SIM_POLICY", "bipedal").lower()
if TEST_POLICY not in {"bipedal", "velocity"}:
    raise ValueError("SIM2SIM_POLICY must be 'bipedal' or 'velocity'")
POLICY_STATE = "BipedalVelocityDVI" if TEST_POLICY == "bipedal" else "VelocityDVI"
POLICY_BUTTON = "up" if TEST_POLICY == "bipedal" else "down"

# Scenario order: chosen policy first, then stabilize, then quadruped.
FSM_SCENARIO = [
    ("Passive -> FixStand", 4.0, ["L2", "A"], 0, 0, 0, 0),
    ("FixStand settle", 3.0, [], 0, 0, 0, 0),
    (f"FixStand -> {POLICY_STATE}", 3.0, ["L2", POLICY_BUTTON], 0, 0, 0, 0),
    (f"{POLICY_STATE} settle", 4.0, [], 0, 0, 0, 0),
    (f"{POLICY_STATE} forward", 5.0, [], 0, 0.3, 0, 0),
    (f"{POLICY_STATE} lateral", 4.0, [], -0.2, 0, 0, 0),
    (f"{POLICY_STATE} yaw", 4.0, [], 0, 0, 0.25, 0),
    (f"{POLICY_STATE} combined", 5.0, [], -0.15, 0.2, 0.2, 0),
    (f"{POLICY_STATE} -> FixStand", 1.0, ["L2", "A"], 0, 0, 0, 0),
    ("FixStand -> Passive", 1.0, ["L2", "B"], 0, 0, 0, 0),
    ("Passive", 2.0, [], 0, 0, 0, 0),
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
        # Keep the robot in the verified nominal pose until the FSM has entered
        # FixStand.  Otherwise the 10 s controller warm-up occurs in Passive,
        # whose zero-Kp command lets the model collapse before the test starts.
        self.force_nominal_hold = True

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

    def initialize_nominal_pose(self, q_nominal):
        """Put the free base and motors at the policy's saved nominal state."""
        self.d.qpos[:3] = [0.0, 0.0, 0.31]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[:] = 0.0
        for motor_id, q in enumerate(q_nominal):
            joint_id = self.m.actuator_trnid[motor_id, 0]
            self.d.qpos[self.m.jnt_qposadr[joint_id]] = q
        mujoco.mj_forward(self.m, self.d)

    def apply_nominal_hold(self, q_nominal):
        """Match the recorder's high-gain pre-policy settling controller."""
        for motor_id, target in enumerate(q_nominal):
            q = self.d.sensordata[motor_id]
            dq = self.d.sensordata[motor_id + self.num_motor]
            tau = 60.0 * (target - q) - 5.0 * dq
            peak = 20.2 if dq * tau > 0.0 else 23.4
            speed = abs(dq)
            limit = peak if speed < 13.5 else max(0.0, peak * (30.0 - speed) / (30.0 - 13.5))
            self.d.ctrl[motor_id] = np.clip(tau, -limit, limit)

    def _on_cmd(self, msg):
        if self.force_nominal_hold:
            return
        with self.lock:
            for i in range(self.num_motor):
                mc = msg.motor_cmd[i]
                tau = (
                    mc.tau
                    + mc.kp * (mc.q - self.d.sensordata[i])
                    + mc.kd * (mc.dq - self.d.sensordata[i + self.num_motor])
                )
                # Match UnitreeActuator/standalone transfer: Go2HV speed envelope.
                qvel = self.d.sensordata[i + self.num_motor]
                peak = 20.2 if qvel * tau > 0.0 else 23.4
                speed = abs(qvel)
                limit = peak if speed < 13.5 else max(0.0, peak * (30.0 - speed) / (30.0 - 13.5))
                self.d.ctrl[i] = np.clip(tau, -limit, limit)

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

        # Wireless controller. The scenario loop owns key lifetimes; calling
        # VirtualJoystick.update() here would clear every scripted transition
        # because no set_pulse() deadline is installed.
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
    # SDK/MuJoCo motor order, obtained by permuting deploy.yaml's policy order
    # through joint_ids_map.  This is the same state held by FixStand.
    nominal_motor_q = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5,
                                 -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])
    bridge.initialize_nominal_pose(nominal_motor_q)
    print("DDS bridge ready; initialized policy nominal pose.")

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
    warmup_sim_start = d.time
    last_state_publish = -float("inf")
    while time.time() - warmup_start < warmup_duration:
        with bridge.lock:
            bridge.apply_nominal_hold(nominal_motor_q)
            mujoco.mj_step(m, d)
        if d.time - last_state_publish >= STATE_PUBLISH_DT:
            bridge.publish_state()
            last_state_publish = d.time
        # Pace warm-up too. An unpaced loop flooded DDS with tens of thousands
        # of stale zero-joystick messages, so the controller saw the scenario
        # one full run late and never made its intended transitions.
        warmup_sim_elapsed = d.time - warmup_sim_start
        warmup_wall_elapsed = time.time() - warmup_start
        if warmup_sim_elapsed > warmup_wall_elapsed:
            time.sleep(warmup_sim_elapsed - warmup_wall_elapsed)
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
    last_state_publish = d.time

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
                # Keep LT continuously asserted throughout the test. The
                # controller's LT is a smoothed Axis, whereas every other
                # transition control is edge-triggered. Dropping LT between
                # scripted phases made queued zero-wireless packets erase LT
                # before A/up's edge was evaluated.
                joy.keys = make_keys("L2")
                if keys:
                    other_keys = [k for k in keys if k != "L2"]
                    # Inject a short, single transition edge only after LT has
                    # been asserted for long enough to cross its 0.5 threshold.
                    if other_keys and 0.75 <= elapsed_in_step < 1.10:
                        joy.keys = make_keys("L2", *other_keys)
                # Set axes (only after key pulse settles)
                if elapsed_in_step > 0.55:
                    joy.set_axes(lx, ly, rx, ry)
                else:
                    joy.set_axes(0, 0, 0, 0)
                current_label = label
            else:
                joy.set_axes(0, 0, 0, 0)
                joy.keys = 0

            # Keep the same nominal hold until FixStand has taken over.  Without
            # it, the 10 s DDS/controller warm-up leaves Passive torque-free and
            # the robot reaches the policy already collapsed/inverted.
            with bridge.lock:
                if bridge.force_nominal_hold:
                    bridge.apply_nominal_hold(nominal_motor_q)
                # The LT+A transition has been held for four seconds by here;
                # release command ownership only after FixStand is established.
                if sim_time >= 4.5:
                    bridge.force_nominal_hold = False
                mujoco.mj_step(m, d)

            # Publish state at 100 Hz, enough for the 1 kHz controller while
            # preventing stale virtual-joystick samples from queuing in DDS.
            if d.time - last_state_publish >= STATE_PUBLISH_DT:
                bridge.publish_state()
                last_state_publish = d.time

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
