#!/usr/bin/env python
"""Sanity checks for an EMA-VFI harness run (used by smoke_emavfi.sbatch; no torch import).

    python check_emavfi_output.py OUT_DIR INPUT_DIR SF N_INPUT [--ref OTHER_OUT_DIR]

Fails (exit 1) unless:
  * (N_INPUT-1)*SF+1 frame_*.tif files, all uint16 (H, W, 3) like the source
  * genuine positions are bit-exact copies of the source frames
  * the adapter saw no non-finite float values and its raw float output lies in [0, 1] (run_report.json)
  * generated frames are not constant and are not copies of either neighbour
  * for every pair: t=1/SF is closer (mean abs diff) to img0 than to img1, and t=(SF-1)/SF the reverse
Prints per-frame distances to both neighbours. With --ref, also prints the difference to another run's
frames with the same indices (e.g. TF32 on vs off, down_scale 0.5 vs 0.25).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frameio import IMG_EXTS, natural_key, read_image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("src")
ap.add_argument("sf", type=int)
ap.add_argument("n_in", type=int)
ap.add_argument("--ref", default=None)
a = ap.parse_args()
out, src, sf, n_in = Path(a.out), Path(a.src), a.sf, a.n_in
fails = []

frames = sorted(out.glob("frame_*.tif"))
expect = (n_in - 1) * sf + 1
print(f"[{out.name}] frames: {len(frames)} (expected {expect})")
if len(frames) != expect:
    fails.append(f"frame count {len(frames)} != {expect}")

rep = json.loads((out / "logs" / "run_report.json").read_text())
st = rep["model"].get("output_stats", {})
print(f"[{out.name}] adapter float stats: {st}")
print(f"[{out.name}] settings: {rep['model'].get('settings')}")
print(f"[{out.name}] timing: {rep['timing_s']}  gpu: {rep['gpu']}")
if rep["model"].get("repo_commit") != rep["model"].get("repo_commit_expected"):
    fails.append(f"repo commit {rep['model'].get('repo_commit')} != expected")
if st.get("nonfinite_values", 1) != 0:
    fails.append(f"non-finite values in model output: {st.get('nonfinite_values')}")
if not (-1e-5 <= st.get("raw_min", -1) and st.get("raw_max", 2) <= 1 + 1e-5):
    fails.append(f"raw float output outside [0,1]: {st.get('raw_min')}..{st.get('raw_max')}")
if st.get("frames") != (n_in - 1) * (sf - 1):
    fails.append(f"adapter produced {st.get('frames')} frames, expected {(n_in - 1) * (sf - 1)}")

srcs = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: natural_key(p.name))[:n_in]
S = [read_image(p) for p in srcs]
H, W = S[0].shape[:2]
Sf = [s.astype(np.float32) for s in S]


def mad(x, y):
    return float(np.abs(x - y).mean())


gmin, gmax = 65535, 0
for j in range(n_in - 1):
    A, B = Sf[j], Sf[j + 1]
    print(f"[{out.name}] pair {j}: mean|img0-img1| = {mad(A, B):.1f} (16-bit codes)")
    dA, dB = [], []
    for i in range(0, sf + 1):
        k = j * sf + i
        arr = tifffile.imread(out / f"frame_{k:06d}.tif")
        if arr.dtype != np.uint16 or arr.shape != (H, W, 3):
            fails.append(f"frame {k}: dtype/shape {arr.dtype} {arr.shape}")
            continue
        if i in (0, sf):
            ref = S[j] if i == 0 else S[j + 1]
            if not np.array_equal(arr, ref):
                fails.append(f"frame {k}: genuine frame is not bit-exact")
            continue
        f = arr.astype(np.float32)
        gmin, gmax = min(gmin, int(arr.min())), max(gmax, int(arr.max()))
        a_, b_ = mad(f, A), mad(f, B)
        dA.append(a_)
        dB.append(b_)
        line = f"   t={i / sf:.1f} frame {k:3d}: d(img0)={a_:8.2f} d(img1)={b_:8.2f} std={f.std():8.1f}"
        if a.ref:
            rp = Path(a.ref) / f"frame_{k:06d}.tif"
            if rp.exists():
                r = tifffile.imread(rp).astype(np.int32)
                d = np.abs(arr.astype(np.int32) - r)
                line += f" | vs ref: mean {d.mean():.3f} max {d.max()} codes"
        print(line)
        if f.std() < 1.0:
            fails.append(f"frame {k}: constant output")
        if a_ == 0.0 or b_ == 0.0:
            fails.append(f"frame {k}: identical to a neighbour")
    if dA and not dA[0] < dB[0]:
        fails.append(f"pair {j}: t={1 / sf:.1f} not closer to img0 ({dA[0]:.2f} vs {dB[0]:.2f})")
    if dA and not dB[-1] < dA[-1]:
        fails.append(f"pair {j}: t={(sf - 1) / sf:.1f} not closer to img1 ({dB[-1]:.2f} vs {dA[-1]:.2f})")
    inc = sum(y > x for x, y in zip(dA, dA[1:]))
    print(f"   d(img0) increases in {inc}/{len(dA) - 1} steps; d(img1) decreases in "
          f"{sum(y < x for x, y in zip(dB, dB[1:]))}/{len(dB) - 1} steps")
print(f"[{out.name}] generated uint16 range: {gmin}..{gmax}")
if fails:
    print(f"[{out.name}] FAIL:\n  " + "\n  ".join(fails))
    sys.exit(1)
print(f"[{out.name}] PASS")
