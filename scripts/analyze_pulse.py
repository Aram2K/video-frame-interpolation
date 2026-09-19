#!/usr/bin/env python
"""Objective per-model comparison: detail retention of generated frames and the resulting 2.5 Hz pulse.

    python analyze_pulse.py SHOT [--runs model/variant ...] [--stride 2]

For each finished run of SHOT it measures, on the real 16-bit frames (no preview encoding involved):
  sharpness   variance of the Laplacian per frame -> mean over genuine frames vs over generated frames.
              detail_ratio = generated / genuine; 1.0 = generated frames carry the same detail.
  noise       high-frequency energy (frame minus a 3x3 box blur), same split: grain retention.
  pulse       mean |frame k - frame k-1| for steps adjacent to a genuine frame vs steps between two
              generated frames. pulse_ratio = adjacent / between; 1.0 = no visible periodic modulation.
Everything is computed in the frames' own encoding on a centre crop (letterbox bars excluded).
Writes output/<SHOT>/_comparison/pulse_metrics.{csv,md}.
"""
import os
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("MVFI_ROOT", Path(__file__).resolve().parents[1])).resolve()


def frame_stats(path, box):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"cannot read {path}")
    if a.ndim == 3:
        a = a[..., ::-1]  # cv2 gives BGR
    y0, y1, x0, x1 = box
    g = a[y0:y1, x0:x1].astype(np.float32).mean(axis=2) / 65535.0
    lap = float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())
    hf = float(np.abs(g - cv2.blur(g, (3, 3))).mean())
    return g, lap, hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--max_frames", type=int, default=121, help="analyse the first N output frames")
    a = ap.parse_args()
    base = ROOT / "output" / a.shot
    out = base / "_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = sorted(base.glob("*/*/logs/run_report.json")) + sorted(base.glob("*/*/*/logs/run_report.json"))
    for rep_path in reports:
        run = rep_path.parent.parent
        name = str(run.relative_to(base))
        if a.runs and name not in a.runs:
            continue
        rep = json.loads(rep_path.read_text())
        sf = rep["temporal_sf"]
        n = min(a.max_frames, rep["output_frames"])
        h, w = rep["height"], rep["width"]
        box = (int(h * 0.15), int(h * 0.85), int(w * 0.1), int(w * 0.9))  # skip letterbox bars
        prev = None
        gen_lap, gen_hf, an_lap, an_hf, d_adj, d_between = [], [], [], [], [], []
        for k in range(n):
            g, lap, hf = frame_stats(run / f"frame_{k:06d}.tif", box)
            (an_lap if k % sf == 0 else gen_lap).append(lap)
            (an_hf if k % sf == 0 else gen_hf).append(hf)
            if prev is not None:
                d = float(np.abs(g - prev).mean())
                (d_adj if (k % sf == 0 or k % sf == 1) else d_between).append(d)
            prev = g
        rows.append({
            "run": name,
            "frames_analysed": n,
            "sharpness_genuine": round(float(np.mean(an_lap)), 6),
            "sharpness_generated": round(float(np.mean(gen_lap)), 6),
            "detail_ratio": round(float(np.mean(gen_lap) / np.mean(an_lap)), 3),
            "grain_genuine": round(float(np.mean(an_hf)), 6),
            "grain_generated": round(float(np.mean(gen_hf)), 6),
            "grain_ratio": round(float(np.mean(gen_hf) / np.mean(an_hf)), 3),
            "step_adjacent_to_genuine": round(float(np.mean(d_adj)), 5),
            "step_between_generated": round(float(np.mean(d_between)), 5),
            "pulse_ratio": round(float(np.mean(d_adj) / np.mean(d_between)), 3),
        })
        print(f"  {name}: detail {rows[-1]['detail_ratio']}, grain {rows[-1]['grain_ratio']}, pulse {rows[-1]['pulse_ratio']}", flush=True)
    if not rows:
        raise SystemExit("no runs found")
    rows.sort(key=lambda r: (-r["detail_ratio"], r["pulse_ratio"]))
    with open(out / "pulse_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md = ["Detail/grain ratio: generated vs genuine frames (1.0 = identical detail).",
          "Pulse ratio: frame-to-frame change next to a genuine frame vs between generated frames (1.0 = no pulse).", ""] + md
    (out / "pulse_metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
