#!/usr/bin/env python
"""Restore letterbox matte rows/columns to exact black in generated frames.

Models do not know the black bars are a matte, so they warp picture pixels into them; the genuine
frames keep them at code 0, which makes the bar edge twinkle at the genuine-frame rate.

    python restore_bars.py RUN_DIR [--apply] [--source DIR]

The matte is detected from the GENUINE frames only: rows/columns that are exactly 0 in every one of
them. Picture area is never touched. Without --apply it reports what it would change (dry run).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from frameio import read_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--source", default=None, help="genuine frames (default: from run_report.json)")
    a = ap.parse_args()
    run = Path(a.run)
    rep = json.loads((run / "logs" / "run_report.json").read_text())
    sf = rep["temporal_sf"]
    rows = list(csv.DictReader(open(run / "frame_map.csv")))
    genuine = [int(r["out_index"]) for r in rows if r["kind"] == "genuine"]
    src = Path(a.source or rep["input"])

    # matte = rows/cols that are exactly 0 in every genuine frame
    row_mask = col_mask = None
    for k in genuine:
        g = read_image(run / f"frame_{k:06d}.tif")
        r = (g.max(axis=(1, 2)) == 0)
        c = (g.max(axis=(0, 2)) == 0)
        row_mask = r if row_mask is None else (row_mask & r)
        col_mask = c if col_mask is None else (col_mask & c)
    n_rows, n_cols = int(row_mask.sum()), int(col_mask.sum())
    print(f"matte detected from {len(genuine)} genuine frames: {n_rows} rows, {n_cols} columns")
    if n_rows == 0 and n_cols == 0:
        print("no matte: nothing to do")
        return

    changed, worst = 0, 0
    for r in rows:
        if r["kind"] == "genuine":
            continue
        k = int(r["out_index"])
        p = run / f"frame_{k:06d}.tif"
        img = read_image(p)
        m = max(int(img[row_mask].max()) if n_rows else 0, int(img[:, col_mask].max()) if n_cols else 0)
        if m == 0:
            continue
        worst = max(worst, m)
        changed += 1
        if a.apply:
            if n_rows:
                img[row_mask] = 0
            if n_cols:
                img[:, col_mask] = 0
            tifffile.imwrite(p, img, photometric="rgb", compression=None, metadata=None)
    verb = "cleaned" if a.apply else "would clean"
    print(f"{verb} {changed} generated frames; worst bar value was {worst} ({worst / 655.35:.2f}% of full scale)")
    if a.apply:
        note = run / "logs" / "post_process.txt"
        with open(note, "a") as f:
            f.write(f"restore_bars: matte {n_rows} rows / {n_cols} cols forced to 0 in {changed} generated frames "
                    f"(worst was {worst}); picture area untouched; genuine frames untouched\n")
        print(f"recorded in {note}")
    else:
        print("dry run: re-run with --apply to write the frames")


if __name__ == "__main__":
    main()
