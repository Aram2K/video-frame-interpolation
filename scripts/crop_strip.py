#!/usr/bin/env python
"""Full-resolution crop strip across timesteps: python crop_strip.py RUN_DIR OUT.png --indices 0,3,5,7,10 --box x,y,w,h [--view rec709]"""
import argparse
from pathlib import Path
import cv2, numpy as np
from frameio import read_image

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--indices", default="0,3,5,7,10")
ap.add_argument("--box", default="800,330,600,600")
ap.add_argument("--view", default="rec709")
ap.add_argument("--sf", type=int, default=10)
a = ap.parse_args()
x, y, w, h = [int(v) for v in a.box.split(",")]
tiles = []
for k in [int(v) for v in a.indices.split(",")]:
    img = read_image(Path(a.run) / f"frame_{k:06d}.tif").astype(np.float32) / 65535.0
    crop = img[y:y+h, x:x+w]
    if a.view == "rec709":
        import colorxf
        crop = colorxf.gen5_to_rec709_display(crop)
    t = cv2.cvtColor((np.clip(crop, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    kind = "GENUINE (original pixels)" if k % a.sf == 0 else f"generated t={(k % a.sf)/a.sf:.1f}"
    cv2.rectangle(t, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(t, f"{k}: {kind}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (60, 220, 60) if k % a.sf == 0 else (0, 150, 255), 2, cv2.LINE_AA)
    tiles.append(t)
cv2.imwrite(a.out, np.concatenate(tiles, axis=1))
print("wrote", a.out, np.concatenate(tiles, axis=1).shape)
