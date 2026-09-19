#!/usr/bin/env python
"""Cross-model comparison for one shot: summary table, per-timestep grids, side-by-side video.

    python compare_models.py SHOT [--runs model/variant ...] [--view as_encoded|rec709] [--width 960]

Reads every output/<SHOT>/<model>/<variant>/ that has logs/run_report.json and writes into
output/<SHOT>/_comparison/: summary.md + summary.csv, grid_<index>.png (one tile per run at that
output frame), side_by_side.mp4 (up to 4 runs, time-aligned at the output rate).
Viewing aids only; the 16-bit TIFF sequences remain the evaluation source.
"""
import os
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from make_previews import Encoder, label, load_view

ROOT = Path(os.environ.get("MVFI_ROOT", Path(__file__).resolve().parents[1])).resolve()
WHITE, GREEN, ORANGE = (255, 255, 255), (60, 220, 60), (255, 150, 0)


def discover(shot, only):
    runs = []
    base = ROOT / "output" / shot
    for rep in sorted(base.glob("*/*/logs/run_report.json")) + sorted(base.glob("*/*/*/logs/run_report.json")):
        run = rep.parent.parent
        name = str(run.relative_to(base))
        if only and name not in only:
            continue
        try:
            runs.append({"name": name, "dir": run, "report": json.loads(rep.read_text())})
        except json.JSONDecodeError:
            print(f"  skipping unreadable report: {rep}")
    return runs


def summary_rows(runs):
    rows = []
    for r in runs:
        rep, t, gpu = r["report"], r["report"].get("timing_s", {}), r["report"].get("gpu") or {}
        hold = r["dir"] / "logs" / "holdout_metrics.csv"
        hold_psnr = rep.get("holdout_psnr_db_mean")
        anchor = r["dir"] / "logs" / "anchor_fidelity.csv"
        anchor_psnr = rep.get("anchor_psnr_db_mean")
        rows.append({
            "run": r["name"],
            "input": Path(rep.get("input", "")).name,
            "frames_in": rep.get("model_input_frames"),
            "sf": rep.get("temporal_sf"),
            "frames_out": rep.get("output_frames"),
            "minutes": round((t.get("generation_and_write") or 0) / 60, 1),
            "s_per_out_frame": round(t.get("sec_per_output_frame") or 0, 2),
            "gpu_peak_gib": round(gpu.get("max_memory_allocated_gib") or 0, 1),
            "holdout_psnr_db": round(hold_psnr, 2) if hold_psnr else ("see " + hold.name if hold.exists() else ""),
            "anchor_vae_psnr_db": round(anchor_psnr, 2) if anchor_psnr else ("see " + anchor.name if anchor.exists() else ""),
            "vae_dtype": (rep.get("settings") or {}).get("vae_dtype", ""),
            "adapter_args": json.dumps((rep.get("model") or {}).get("adapter_args", {})) if rep.get("model") else "",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("--runs", nargs="*", default=None, help="limit to these model/variant names")
    ap.add_argument("--view", choices=["as_encoded", "rec709"], default="as_encoded")
    ap.add_argument("--width", type=int, default=960, help="tile width in the grids")
    ap.add_argument("--indices", default=None, help="comma-separated output indices for the grids (default: t=0.5 of the first 3 intervals)")
    ap.add_argument("--video_runs", nargs="*", default=None, help="runs for side_by_side.mp4 (max 4)")
    ap.add_argument("--video_width", type=int, default=960)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args()

    out = ROOT / "output" / a.shot / "_comparison"
    out.mkdir(parents=True, exist_ok=True)
    runs = discover(a.shot, a.runs)
    if not runs:
        raise SystemExit(f"no finished runs under output/{a.shot}")
    print(f"runs: {[r['name'] for r in runs]}")

    rows = summary_rows(runs)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # Grids: same output frame from every run, side by side.
    sf = runs[0]["report"]["temporal_sf"]
    n_out = min(r["report"]["output_frames"] for r in runs)
    idx = ([int(x) for x in a.indices.split(",")] if a.indices
           else [i * sf + sf // 2 for i in range(3) if (i * sf + sf // 2) < n_out])
    suffix = "" if a.view == "as_encoded" else "_rec709"
    for k in idx:
        tiles = []
        for r in runs:
            p = r["dir"] / f"frame_{k:06d}.tif"
            if not p.exists():
                continue
            img = load_view(p, a.view, a.width)
            kind = "GENUINE" if k % r["report"]["temporal_sf"] == 0 else f"t={(k % r['report']['temporal_sf']) / r['report']['temporal_sf']:.1f}"
            tiles.append(label(img, f"{r['name']}  {kind}", GREEN if kind == "GENUINE" else ORANGE))
        if not tiles:
            continue
        cols_n = min(3, len(tiles))
        while len(tiles) % cols_n:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[i:i + cols_n], axis=1) for i in range(0, len(tiles), cols_n)], axis=0)
        cv2.imwrite(str(out / f"grid_{k:06d}{suffix}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  grid_{k:06d}{suffix}.png  ({len(tiles)} tiles)")

    # Side-by-side video of up to 4 runs.
    vruns = [r for r in runs if not a.video_runs or r["name"] in a.video_runs][:4]
    if len(vruns) >= 2:
        w = a.video_width
        first = load_view(vruns[0]["dir"] / "frame_000000.tif", a.view, w)
        h = first.shape[0]
        cols_n = 2 if len(vruns) <= 2 else 2
        rows_n = (len(vruns) + cols_n - 1) // cols_n
        enc = Encoder(out / f"side_by_side{suffix}.mp4", cols_n * w, rows_n * h, a.fps)
        for k in range(n_out):
            tiles = []
            for r in vruns:
                img = load_view(r["dir"] / f"frame_{k:06d}.tif", a.view, w)
                s = r["report"]["temporal_sf"]
                kind = "GENUINE" if k % s == 0 else f"AI t={(k % s) / s:.1f}"
                tiles.append(label(img, f"{r['name']}  {kind}", GREEN if k % s == 0 else ORANGE))
            while len(tiles) < cols_n * rows_n:
                tiles.append(np.zeros_like(tiles[0]))
            frame = np.concatenate([np.concatenate(tiles[i:i + cols_n], axis=1) for i in range(0, len(tiles), cols_n)], axis=0)
            enc.write(frame)
        enc.close()
        print(f"  side_by_side{suffix}.mp4  ({[r['name'] for r in vruns]})")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
