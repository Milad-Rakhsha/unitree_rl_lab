#!/usr/bin/env python3
"""Compose baseline/updated DVI-solver policy videos as a synchronized 2x2 grid."""
import argparse
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def frame(reader, i):
    return reader.get_data(min(i, reader.count_frames() - 1))


def main():
    p=argparse.ArgumentParser()
    for x in ("baseline-mjwarp-video","baseline-dvi-video","updated-mjwarp-video","updated-dvi-video"):
        p.add_argument("--"+x, required=True)
    # Retained for compatibility with existing sweep scripts; telemetry is not plotted.
    for x in ("baseline-mjwarp-telemetry","baseline-dvi-telemetry","updated-mjwarp-telemetry","updated-dvi-telemetry"):
        p.add_argument("--"+x)
    p.add_argument("--output",required=True);p.add_argument("--updated-label",required=True);p.add_argument("--provenance",required=True)
    a=p.parse_args()
    readers=[imageio.get_reader(getattr(a,x.replace('-','_'))) for x in ("baseline-mjwarp-video","baseline-dvi-video","updated-mjwarp-video","updated-dvi-video")]
    W,H=1280,720; panel=(640,360); fps=25; n=250
    bold=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",20)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    labels=["Baseline — MJWarp policy","Baseline — DVI policy",f"{a.updated_label} — MJWarp policy",f"{a.updated_label} — DVI policy"]
    out=imageio.get_writer(a.output,fps=fps,quality=8,macro_block_size=16)
    try:
        for i in range(n):
            canvas=Image.new("RGB",(W,H),(18,18,18)); draw=ImageDraw.Draw(canvas)
            for k,(reader,label) in enumerate(zip(readers,labels)):
                source_fps=reader.get_meta_data().get("fps",50)
                im=Image.fromarray(frame(reader,round(i/fps*source_fps))).convert("RGB").resize(panel)
                x=(k%2)*640;y=(k//2)*360;canvas.paste(im,(x,y))
                draw.rectangle((x,y,x+640,y+6),fill=(220,65,55) if k%2==0 else (78,121,167))
                draw.rectangle((x,y+6,x+640,y+38),fill=(16,16,16));draw.text((x+12,y+10),label,font=bold,fill="white")
            draw.rectangle((0,H-28,W,H),fill=(18,18,18));draw.text((10,H-24),a.provenance,font=small,fill="white")
            out.append_data(np.asarray(canvas))
    finally:
        out.close()
        for r in readers:r.close()
    print(a.output)
if __name__=="__main__":main()
