#!/usr/bin/env python
"""Test B input: technically normalised Rec.709 copy of a Gen5-log TIFF sequence (16-bit TIFF out).

Usage: python make_rec709.py SRC_DIR DST_DIR [--workers 16]
Never touches SRC_DIR. See colorxf.py for the exact transform.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

import colorxf
from frameio import TIFF_EXTS, natural_key, read_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in TIFF_EXTS], key=lambda p: natural_key(p.name))
    dst.mkdir(parents=True, exist_ok=False)

    def conv(p):
        x = read_image(p)
        if x.dtype != np.uint16:
            raise TypeError(f"{p}: expected uint16, got {x.dtype}")
        y, over, under = colorxf.gen5_to_rec709_display(x.astype(np.float32) / 65535.0, return_clip_fraction=True)
        tifffile.imwrite(dst / p.name, np.round(y * 65535.0).astype(np.uint16),
                         photometric="rgb", compression=None, metadata=None)
        return {"file": p.name, "frac_over_1": over, "frac_below_0": under}

    with ThreadPoolExecutor(a.workers) as ex:
        stats = list(ex.map(conv, files))
    info = {"source_dir": str(src.resolve()), "frames": len(files), "transform": colorxf.describe(),
            "max_frac_clipped_high": max(s["frac_over_1"] for s in stats),
            "mean_frac_clipped_high": float(np.mean([s["frac_over_1"] for s in stats])),
            "per_frame": stats}
    (dst.parent / f"{dst.name}_transform.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("frames", "max_frac_clipped_high", "mean_frac_clipped_high")}, indent=1))


if __name__ == "__main__":
    main()
