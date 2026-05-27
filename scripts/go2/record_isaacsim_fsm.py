"""Go2 FSM recording — better camera, lighting, zoomed out."""
import sys, os, math, glob, time
import numpy as np

def p(msg):
    sys.stderr.write(msg + "\n"); sys.stderr.flush()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.robot.policy.examples.controllers.fsm_controller import FSMState
from isaacsim.robot.policy.examples.robots import Go2FSMPolicy
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdLux, Sdf
import omni.usd
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from scipy.spatial.transform import Rotation as R

assets = get_assets_root_path()
my_world = World(stage_units_in_meters=1.0, physics_dt=1/200, rendering_dt=1/50)
ground = define_prim("/World/Ground", "Xform")
ground.GetReferences().AddReference(assets + "/Isaac/Environments/Grid/default_environment.usd")

go2 = Go2FSMPolicy.from_local_policies(
    prim_path="/World/Go2",
    logs_dir="/home/horde/Repos/GO2/unitree_rl_lab/logs/rsl_rl",
    name="Go2",
    position=np.array([0, 0, 0.4]),
)

stage = omni.usd.get_context().get_stage()

# --- Lighting: dome light + key light ---
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.GetIntensityAttr().Set(500.0)
dome.GetColorAttr().Set(Gf.Vec3f(0.9, 0.95, 1.0))

key_light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key_light.GetIntensityAttr().Set(3000.0)
key_light.GetAngleAttr().Set(1.0)
key_xf = UsdGeom.Xformable(key_light.GetPrim())
key_rot = key_xf.AddRotateXYZOp()
key_rot.Set(Gf.Vec3f(-45, 30, 0))

fill_light = UsdLux.DistantLight.Define(stage, "/World/FillLight")
fill_light.GetIntensityAttr().Set(1500.0)
fill_light.GetAngleAttr().Set(2.0)
fill_xf = UsdGeom.Xformable(fill_light.GetPrim())
fill_rot = fill_xf.AddRotateXYZOp()
fill_rot.Set(Gf.Vec3f(-30, -45, 0))

# --- Camera ---
cam_path = "/World/RecordCam"
cam_prim = stage.DefinePrim(cam_path, "Camera")
UsdGeom.Camera(cam_prim).GetFocalLengthAttr().Set(18.0)  # wider lens
UsdGeom.Camera(cam_prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
cam_xformable = UsdGeom.Xformable(cam_prim)
cam_translate = cam_xformable.AddTranslateOp()
cam_orient = cam_xformable.AddOrientOp()

def set_camera(ex, ey, ez, tx, ty, tz):
    eye = np.array([ex, ey, ez])
    target = np.array([tx, ty, tz])
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up); rn = np.linalg.norm(right)
    right /= rn
    up = np.cross(right, fwd)
    rot_mat = np.column_stack([right, up, -fwd])
    r = R.from_matrix(rot_mat)
    q = r.as_quat()
    cam_translate.Set(Gf.Vec3d(ex, ey, ez))
    cam_orient.Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))

viewport = get_active_viewport()
viewport.set_active_camera(cam_path)

my_world.reset()
go2.initialize()

output_dir = os.path.expanduser("~/Repos/GO2/policy_videos")
frames_dir = os.path.join(output_dir, "isaacsim_frames")
for f in glob.glob(os.path.join(frames_dir, "frame_*.png")):
    os.remove(f)
os.makedirs(frames_dir, exist_ok=True)

# Zoomed-out camera offset
CAM_BACK = 1.8
CAM_SIDE = 2.5
CAM_HEIGHT = 1.2

set_camera(-CAM_BACK, -CAM_SIDE, CAM_HEIGHT, 0, 0, 0.2)

SEQUENCE = [
    (FSMState.FIXSTAND,        3.0, [0, 0, 0],       "FixStand"),
    (FSMState.VELOCITY,        4.0, [0.5, 0, 0],     "Velocity Fwd"),
    (FSMState.VELOCITY,        2.0, [0.3, 0, 0.5],   "Velocity Turn"),
    (FSMState.FIXSTAND,        3.0, [0, 0, 0],       "FixStand"),
    (FSMState.BIPEDAL_STANDUP, 4.0, [0.2, 0, 0],     "Bipedal"),
    (FSMState.FIXSTAND,        3.0, [0, 0, 0],       "FixStand Recover"),
]

physics_dt = 1.0 / 200
render_every = 4
capture_every_render = 2
frame_idx = 0
render_idx = 0

for _ in range(30):
    my_world.step(render=True)

p("Recording...")
for state, duration, cmd, label in SEQUENCE:
    go2.force_transition(state)
    cmd_np = np.array(cmd, dtype=np.float64)
    steps = int(duration / physics_dt)
    p(f"  {label}: {duration}s")

    for step in range(steps):
        go2.forward(physics_dt, cmd_np)
        do_render = (step % render_every == 0)
        my_world.step(render=do_render)

        if do_render:
            render_idx += 1
            if render_idx % capture_every_render == 0:
                pos, _ = go2.robot.get_world_pose()
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                # Camera tracks robot, zoomed out
                set_camera(px - CAM_BACK, py - CAM_SIDE, CAM_HEIGHT,
                          px, py, pz + 0.1)
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
                capture_viewport_to_file(viewport, frame_path)
                frame_idx += 1

    pos, _ = go2.robot.get_world_pose()
    p(f"    end: z={float(pos[2]):.3f} x={float(pos[0]):.3f}")

for _ in range(20):
    my_world.step(render=True)
    time.sleep(0.05)

p(f"Captured {frame_idx} frames")
frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
p(f"On disk: {len(frame_files)}")

if frame_files:
    import imageio.v2 as imageio
    output_path = os.path.join(output_dir, "go2_fsm_isaacsim.mp4")
    writer = imageio.get_writer(output_path, fps=25, codec="libx264",
                                output_params=["-crf", "23", "-preset", "medium"])
    for ff in frame_files:
        try:
            img = imageio.imread(ff)
            if img.shape[0] > 1:
                writer.append_data(img[:,:,:3])
        except:
            pass
    writer.close()
    sz = os.path.getsize(output_path)
    p(f"Video: {output_path} ({sz/1024/1024:.1f} MB)")

simulation_app.close()
