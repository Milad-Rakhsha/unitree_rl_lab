#!/usr/bin/env python3
"""Render Phase-1 fixed-base MuJoCo telemetry exactly as recorded."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--telemetry',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--robot-xml',type=Path,default=Path('/home/horde/go2/unitree_mujoco/unitree_robots/go2/go2.xml')); p.add_argument('--fps',type=int,default=50); p.add_argument('--frame-stride',type=int,default=1); p.add_argument('--camera-distance',type=float,default=3.5); p.add_argument('--camera-lookat-z',type=float,default=1.5); a=p.parse_args()
    xml_path=a.robot_xml.resolve(); xml=xml_path.read_text()
    if "<include " in xml:
      model=mujoco.MjModel.from_xml_path(str(xml_path))
    else:
      xml=xml.replace('<option ', '<visual><global offwidth="960" offheight="720"/></visual>\n  <option ', 1)
      assets={f'assets/{f.name}':f.read_bytes() for f in (xml_path.parent/'assets').glob('*') if f.is_file()}
      model=mujoco.MjModel.from_xml_string(xml,assets=assets)
    model.opt.gravity[:]=0; model.geom_contype[:]=0; model.geom_conaffinity[:]=0
    model.vis.global_.offwidth=960; model.vis.global_.offheight=720
    data=mujoco.MjData(model); z=np.load(a.telemetry); q=np.asarray(z['q']); root=np.asarray(z['root_qpos'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    renderer=mujoco.Renderer(model,height=720,width=960)
    camera=mujoco.MjvCamera(); camera.type=mujoco.mjtCamera.mjCAMERA_FREE; camera.lookat[:]=[0,0,a.camera_lookat_z]; camera.distance=a.camera_distance; camera.azimuth=140; camera.elevation=-5
    writer=imageio.get_writer(a.output,fps=a.fps,codec='libx264',pixelformat='yuv420p')
    try:
      for qi, ri in zip(q[::a.frame_stride], root[::a.frame_stride]):
        data.qpos[:7]=ri; data.qpos[7:]=qi; data.qvel[:]=0; mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=camera); writer.append_data(renderer.render())
    finally:
      writer.close(); renderer.close()
    print(f'MUJOCO_PHASE1_VIDEO_OK frames={len(q[::a.frame_stride])} fps={a.fps} output={a.output}')
if __name__=='__main__': main()
