#!/usr/bin/env python
"""Score hold-out runs (input_stride=k, temporal_sf=k) against the real frames that were dropped.

All metrics are computed in ONE common display space (Rec.709 / BT.1886 via colorxf) so a
log-input run and a Rec.709-input run are comparable; native-space PSNR is reported too.

Usage: python eval_holdout.py --run NAME:RUN_DIR:SPACE ... --out OUT.json
  SPACE = gen5 (run input was Gen5 log; converted for scoring) or rec709 (already display space)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import colorxf
from frameio import natural_key, read_image


def to_display(a, space):
    x = a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)
    return colorxf.gen5_to_rec709_display(x) if space == "gen5" else x


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lpips_width", type=int, default=1920)
    a = ap.parse_args()
    import cv2
    import lpips
    import piq
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    results = {}
    for spec in a.run:
        name, run_dir, space = spec.split(":")
        run = Path(run_dir)
        rep = json.loads((run / "logs" / "run_report.json").read_text())
        src = Path(rep["input"])
        src_files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".png")],
                           key=lambda p: natural_key(p.name))
        rows = [r for r in csv.DictReader(open(run / "frame_map.csv")) if r["kind"] == "generated"]
        per = []
        for r in rows:
            k = int(r["out_index"])
            gt_raw = read_image(src_files[k])
            pr_raw = read_image(run / "reconstructed" / f"frame_{k:06d}.tif")
            gt, pr = to_display(gt_raw, space), to_display(pr_raw, space)
            native_psnr = psnr(pr_raw.astype(np.float32) / 65535.0,
                               gt_raw.astype(np.float64) / (65535.0 if gt_raw.dtype == np.uint16 else 255.0))
            tg = torch.from_numpy(gt).permute(2, 0, 1)[None].to(dev)
            tp = torch.from_numpy(pr).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                ssim = float(piq.ssim(tp.clamp(0, 1), tg.clamp(0, 1), data_range=1.0))
                h = int(round(gt.shape[0] * a.lpips_width / gt.shape[1]))
                sg = torch.from_numpy(cv2.resize(gt, (a.lpips_width, h), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].to(dev)
                sp = torch.from_numpy(cv2.resize(pr, (a.lpips_width, h), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].to(dev)
                lpv = float(lp(sp * 2 - 1, sg * 2 - 1))
            per.append({"out_index": k, "gt": src_files[k].name, "psnr_display_db": psnr(pr, gt),
                        "ssim_display": ssim, "lpips_display": lpv, "psnr_native_db": native_psnr})
        agg = {m: float(np.mean([p[m] for p in per])) for m in ("psnr_display_db", "ssim_display", "lpips_display", "psnr_native_db")}
        results[name] = {"run_dir": str(run), "input_space": space, "n_scored": len(per), **agg, "per_frame": per}
        print(name, json.dumps(agg))
    Path(a.out).write_text(json.dumps({"scoring_space": colorxf.describe(), "results": results}, indent=1))


if __name__ == "__main__":
    main()
