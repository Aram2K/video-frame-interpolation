#!/usr/bin/env python
"""Viewing aids for one interpolation run (the TIFF sequence stays the evaluation source).

Writes into RUN_DIR:
  preview.mp4                          final sequence, 25 fps, 1920 wide, 8-bit H.264 (viewing only)
  comparison/side_by_side.mp4          left: original source, each genuine frame HELD for sf frames
                                       (time-aligned); right: final sequence; labels GENUINE / AI t=x
  comparison/original_fast.mp4         the source exactly as recorded, at the source rate
  comparison/interval_XXX_contact.png  A, t=0.1 ... t=0.9, B for a few intervals
  comparison/anchor_error_XXX.png      |VAE-reconstructed genuine frame - original| x16
  comparison/temporal_profile.csv/png  mean |frame(k) - frame(k-1)| for final and reconstructed
Views: --view as_encoded (code values shown as-is; log looks flat) or rec709 (colorxf technical
transform, for viewing only). Nothing here is written back into any TIFF.
"""
import argparse
import csv
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from frameio import read_image

GREEN, ORANGE, WHITE = (60, 220, 60), (255, 150, 0), (255, 255, 255)


def load_view(path, view, width):
    a = read_image(path)
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    h = int(round(a.shape[0] * width / a.shape[1] / 2) * 2)
    x = cv2.resize(a.astype(np.float32) / scale, (width, h), interpolation=cv2.INTER_AREA)
    if view == "rec709":
        import colorxf
        x = colorxf.gen5_to_rec709_display(x)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def label(img, text, color):
    s = img.shape[1] / 1920.0
    cv2.rectangle(img, (0, 0), (int(620 * s), int(64 * s)), (0, 0, 0), -1)
    cv2.putText(img, text, (int(14 * s), int(46 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.3 * s, color,
                max(1, int(3 * s)), cv2.LINE_AA)
    return img


class Encoder:
    def __init__(self, path, w, h, fps, crf=16):
        ff = shutil.which("ffmpeg")
        if ff is None:
            raise RuntimeError("ffmpeg not found in the environment")
        self.p = subprocess.Popen(
            [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-colorspace", "bt709", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--source_dir", required=True, help="folder holding the genuine source frames")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--view", choices=["as_encoded", "rec709"], default="as_encoded")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--intervals", default="0,1,2", help="interval indices for contact sheets")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    run, src_dir = Path(a.run_dir), Path(a.source_dir)
    comp = run / "comparison"
    comp.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(run / "frame_map.csv")))
    rep = json.loads((run / "logs" / "run_report.json").read_text())
    sf, n = int(rep["temporal_sf"]), len(rows)
    suffix = "" if a.view == "as_encoded" else "_rec709"
    pool = ThreadPoolExecutor(a.workers)
    finals = [run / f"frame_{k:06d}.tif" for k in range(n)]
    recons = [run / "reconstructed" / f"frame_{k:06d}.tif" for k in range(n)]

    # Final sequence + reconstructed, loaded once at preview width.
    fin = list(pool.map(lambda p: load_view(p, a.view, a.width), finals))
    rec = list(pool.map(lambda p: load_view(p, a.view, a.width), recons))
    h, w = fin[0].shape[:2]

    enc = Encoder(run / f"preview{suffix}.mp4", w, h, a.fps)
    for f in fin:
        enc.write(f)
    enc.close()

    # Original source frames used by the run (prev_src at genuine rows), in order.
    genuine_rows = [r for r in rows if r["kind"] == "genuine"]
    src_files = [src_dir / r["prev_src"] for r in genuine_rows]
    src = list(pool.map(lambda p: load_view(p, a.view, a.width), src_files))

    enc = Encoder(comp / f"original_fast{suffix}.mp4", w, h, a.fps)
    for f in src:
        enc.write(f)
    enc.close()

    enc = Encoder(comp / f"side_by_side{suffix}.mp4", 2 * w, h, a.fps)
    for k, r in enumerate(rows):
        left = label(src[k // sf].copy(), f"ORIGINAL #{k // sf}  (held x{sf})", WHITE)
        if r["kind"] == "genuine":
            right = label(fin[k].copy(), f"GENUINE  out {k}", GREEN)
        else:
            right = label(fin[k].copy(), f"AI  t={float(r['t_frac']):.1f}  out {k}", ORANGE)
        enc.write(np.concatenate([left, right], axis=1))
    enc.close()

    # Contact sheets: one row per interval, A .. B (final sequence).
    for iv in [int(x) for x in a.intervals.split(",") if x != ""]:
        if (iv + 1) * sf >= n:
            continue
        tiles = []
        for k in range(iv * sf, (iv + 1) * sf + 1):
            t = cv2.resize(fin[k], (w // 3, h // 3), interpolation=cv2.INTER_AREA)
            tag = "GENUINE" if k % sf == 0 else f"t={(k % sf) / sf:.1f}"
            tiles.append(label(t, tag, GREEN if k % sf == 0 else ORANGE))
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)], axis=0)
        cv2.imwrite(str(comp / f"interval_{iv:03d}_contact{suffix}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # Anchor reconstruction error maps (full-resolution difference, shown at preview width).
    for j in sorted({0, len(genuine_rows) // 2, len(genuine_rows) - 1}):
        k = j * sf
        o = read_image(src_files[j]).astype(np.float32)
        v = read_image(recons[k]).astype(np.float32)
        scale = 65535.0 if read_image(src_files[j]).dtype == np.uint16 else 255.0
        d = np.abs(v / 65535.0 - o / scale).mean(axis=-1)
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
        heat = cv2.applyColorMap(np.clip(d * 16 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(comp / f"anchor_error_{k:06d}_x16.png"), heat)

    # Temporal profile: flicker / pulse at genuine positions shows as periodic spikes.
    prof = []
    for k in range(1, n):
        prof.append({"out_index": k, "kind": rows[k]["kind"],
                     "final_mad": float(np.abs(fin[k].astype(np.int16) - fin[k - 1]).mean()),
                     "recon_mad": float(np.abs(rec[k].astype(np.int16) - rec[k - 1]).mean())})
    with open(comp / f"temporal_profile{suffix}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(prof[0].keys()))
        wr.writeheader()
        wr.writerows(prof)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 4), dpi=110)
        ks = [p["out_index"] for p in prof]
        ax.plot(ks, [p["recon_mad"] for p in prof], lw=1, label="model output (VAE genuine frames)")
        ax.plot(ks, [p["final_mad"] for p in prof], lw=1, label="final (original pixels pasted)")
        for k in range(0, n, sf):
            ax.axvline(k, color="0.85", lw=0.6, zorder=0)
        ax.set_xlabel("output frame")
        ax.set_ylabel("mean |frame k - frame k-1| (8-bit view)")
        ax.legend(frameon=False)
        ax.set_title("Frame-to-frame change; grey lines = genuine frames")
        fig.tight_layout()
        fig.savefig(comp / f"temporal_profile{suffix}.png")
    except ImportError:
        pass
    print(f"previews written to {run} ({a.view})")


if __name__ == "__main__":
    main()
