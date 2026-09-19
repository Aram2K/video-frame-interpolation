#!/usr/bin/env python
"""Validate a TIFF frame sequence before interpolation and write its metadata.

Checks: enumeration, numbering/order, gaps, identical geometry, bit depth, channel
count, alpha, compression, pixel statistics (to recognise log vs display encoding),
and exact/near duplicate frames. Never writes into the input directory.

Usage: python inspect_input.py INPUT_DIR OUT_JSON [--fps 25] [--sf 10]
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

EXTS = {".tif", ".tiff"}


def frame_number(name):
    m = re.search(r"(\d+)(?!.*\d)", Path(name).stem)
    return int(m.group(1)) if m else None


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]


def _name(v):
    return None if v is None else getattr(v, "name", v)


def tiff_header(path):
    with tifffile.TiffFile(path) as tf:
        p = tf.pages[0]
        extras = p.extrasamples if p.extrasamples else ()
        tags = p.tags
        return {
            "n_pages": len(tf.pages),
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "bitspersample": p.bitspersample,
            "samplesperpixel": p.samplesperpixel,
            "photometric": _name(p.photometric),
            "planarconfig": _name(p.planarconfig),
            "compression": _name(p.compression),
            "sampleformat": _name(p.sampleformat),
            "extrasamples": [int(e) for e in extras],
            "has_icc_profile": "InterColorProfile" in tags,
            "software": tags["Software"].value if "Software" in tags else None,
            "orientation": int(tags["Orientation"].value) if "Orientation" in tags else None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("out_json")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sf", type=int, default=10)
    ap.add_argument("--cut-ratio", type=float, default=6.0,
                    help="flag a possible cut when a consecutive difference exceeds this multiple of the median")
    ap.add_argument("--dup-thresh", type=float, default=1e-4,
                    help="mean |diff| (0-1 scale) below which consecutive frames are flagged near-duplicate")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS],
                   key=lambda p: natural_key(p.name))
    ignored = sorted(p.name for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() not in EXTS)
    problems, warnings = [], []
    if not files:
        print(f"ERROR: no .tif/.tiff files in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Numbering / order / gaps.
    nums = [frame_number(p.name) for p in files]
    if any(n is None for n in nums):
        problems.append("some filenames carry no frame number")
    prefixes = {re.sub(r"(\d+)(?!.*\d)", "#", p.stem) for p in files}
    if len(prefixes) > 1:
        problems.append(f"more than one filename pattern: {sorted(prefixes)[:5]}")
    gaps, dup_numbers = [], []
    if all(n is not None for n in nums):
        if nums != sorted(nums):
            problems.append("natural filename order differs from numeric frame order")
        if len(set(nums)) != len(nums):
            dup_numbers = sorted({n for n in nums if nums.count(n) > 1})
            problems.append(f"duplicate frame numbers: {dup_numbers[:10]}")
        s = sorted(set(nums))
        gaps = [[a + 1, b - 1] for a, b in zip(s, s[1:]) if b - a > 1]
        if gaps:
            problems.append(f"missing frame numbers (ranges): {gaps[:10]}")

    # Headers.
    headers = {}
    for p in files:
        h = tiff_header(p)
        key = json.dumps({k: h[k] for k in h if k not in ("software",)}, sort_keys=True)
        headers.setdefault(key, []).append(p.name)
    if len(headers) > 1:
        problems.append(f"{len(headers)} distinct TIFF layouts (geometry/bit depth/channels differ)")
    h0 = tiff_header(files[0])
    if h0["n_pages"] != 1:
        warnings.append(f"first file has {h0['n_pages']} pages; only page 0 is used")
    if h0["samplesperpixel"] == 4 or h0["extrasamples"]:
        problems.append(f"alpha/extra channel present (samplesperpixel={h0['samplesperpixel']}, extrasamples={h0['extrasamples']})")
    if h0["samplesperpixel"] not in (3, 4):
        problems.append(f"unexpected channel count {h0['samplesperpixel']}")
    if h0["bitspersample"] != 16:
        warnings.append(f"bit depth is {h0['bitspersample']}, not 16")
    if h0["photometric"] != "RGB":
        problems.append(f"photometric is {h0['photometric']}, expected RGB")

    # Pixel pass: statistics, hashes, consecutive differences.
    per_frame = []
    prev_small = None
    hist = None
    maxval = None
    for i, p in enumerate(files):
        a = tifffile.imread(p, key=0)
        if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
            a = np.moveaxis(a, 0, -1)  # planar -> interleaved view
        maxval = float(np.iinfo(a.dtype).max) if np.issubdtype(a.dtype, np.integer) else 1.0
        rgb = a[..., :3]
        md5 = hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        small = rgb[::4, ::4].astype(np.float32) / maxval
        rec = {
            "index": i, "file": p.name, "frame_number": nums[i], "md5_rgb": md5,
            "min": int(rgb.min()), "max": int(rgb.max()),
            "mean": float(small.mean()),
            "frac_at_0": float((rgb == 0).mean()),
            "frac_at_max": float((rgb >= maxval).mean()),
        }
        if prev_small is not None:
            rec["mad_to_prev"] = float(np.abs(small - prev_small).mean())
        per_frame.append(rec)
        prev_small = small
        vals = (small * 1023).clip(0, 1023).astype(np.int32).ravel()
        h = np.bincount(vals, minlength=1024)
        hist = h if hist is None else hist + h

    exact_dups = [[per_frame[i - 1]["file"], per_frame[i]["file"]]
                  for i in range(1, len(per_frame)) if per_frame[i]["md5_rgb"] == per_frame[i - 1]["md5_rgb"]]
    near_dups = [[per_frame[i - 1]["file"], per_frame[i]["file"], per_frame[i]["mad_to_prev"]]
                 for i in range(1, len(per_frame))
                 if per_frame[i]["mad_to_prev"] < args.dup_thresh and per_frame[i]["md5_rgb"] != per_frame[i - 1]["md5_rgb"]]
    if exact_dups:
        problems.append(f"{len(exact_dups)} exact duplicate consecutive frames, e.g. {exact_dups[:3]}")
    if near_dups:
        warnings.append(f"{len(near_dups)} near-duplicate consecutive frames (mean|diff| < {args.dup_thresh})")

    # Possible editorial cuts: consecutive-difference spikes far above the clip's typical motion.
    mads = np.array([r["mad_to_prev"] for r in per_frame[1:]])
    cut_candidates = []
    if mads.size >= 3:
        med = float(np.median(mads))
        for r in per_frame[1:]:
            if r["mad_to_prev"] > max(args.cut_ratio * med, 1e-3):
                cut_candidates.append([per_frame[r["index"] - 1]["file"], r["file"], round(r["mad_to_prev"] / med, 2)])
    if cut_candidates:
        warnings.append(f"{len(cut_candidates)} possible cut(s) (difference > {args.cut_ratio}x median): {cut_candidates}")

    cdf = np.cumsum(hist) / hist.sum()
    pct = {f"p{q}": float(np.searchsorted(cdf, q / 100.0) / 1023.0) for q in (0.1, 1, 5, 50, 95, 99, 99.9)}
    distinct_levels = None
    a0 = tifffile.imread(files[0], key=0)
    if a0.ndim == 3 and a0.shape[0] in (3, 4) and a0.shape[-1] not in (3, 4):
        a0 = np.moveaxis(a0, 0, -1)
    distinct_levels = {c: int(np.unique(a0[..., k]).size) for k, c in enumerate("RGB")}
    uses_low_byte = bool(np.any(a0[..., :3] % 257 != 0)) if a0.dtype == np.uint16 else None

    N = len(files)
    sf = args.sf
    out_count = (N - 1) * sf + 1
    meta = {
        "input_dir": str(in_dir.resolve()),
        "n_frames": N,
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "frame_number_range": [min(nums), max(nums)] if all(n is not None for n in nums) else None,
        "filename_patterns": sorted(prefixes),
        "ignored_non_tiff_files": ignored,
        "fps_assumed": args.fps,
        "input_duration_s": N / args.fps,
        "tiff": h0,
        "layout_groups": {k: len(v) for k, v in headers.items()},
        "height": int(a0.shape[0]), "width": int(a0.shape[1]),
        "distinct_code_values_frame0": distinct_levels,
        "uses_full_16bit_precision_frame0": uses_low_byte,
        "value_percentiles_0to1": pct,
        "global_min": min(r["min"] for r in per_frame),
        "global_max": max(r["max"] for r in per_frame),
        "max_frac_at_0": max(r["frac_at_0"] for r in per_frame),
        "max_frac_at_max": max(r["frac_at_max"] for r in per_frame),
        "exact_duplicate_pairs": exact_dups,
        "near_duplicate_pairs": near_dups,
        "possible_cuts": cut_candidates,
        "temporal_sf": sf,
        "expected_output_frames": out_count,
        "expected_output_duration_s": out_count / args.fps,
        "expected_expansion_factor": out_count / N,
        "problems": problems,
        "warnings": warnings,
        "per_frame": per_frame,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1))

    lines = [
        f"input_dir            {meta['input_dir']}",
        f"frames N             {N}  ({files[0].name} .. {files[-1].name})",
        f"frame numbers        {meta['frame_number_range']}  gaps={gaps or 'none'}",
        f"geometry             {meta['width']}x{meta['height']}  channels={h0['samplesperpixel']}  extrasamples={h0['extrasamples']}",
        f"bit depth / dtype    {h0['bitspersample']} / {h0['dtype']}  sampleformat={h0['sampleformat']}  compression={h0['compression']}  planar={h0['planarconfig']}",
        f"16-bit precision     low byte used in frame 0: {uses_low_byte}; distinct levels frame 0: {distinct_levels}",
        f"value percentiles    {', '.join(f'{k}={v:.3f}' for k, v in pct.items())}",
        f"min/max code         {meta['global_min']} / {meta['global_max']}  (max frac at 0: {meta['max_frac_at_0']:.2e}, at max: {meta['max_frac_at_max']:.2e})",
        f"duplicates           exact={len(exact_dups)} near={len(near_dups)}",
        f"possible cuts        {cut_candidates or 'none'}",
        f"input duration       {N} / {args.fps:g} = {N / args.fps:.3f} s",
        f"expected output      (N-1)*{sf}+1 = {out_count} frames = {out_count / args.fps:.3f} s @ {args.fps:g} fps  (x{out_count / N:.3f})",
        f"PROBLEMS             {problems or 'none'}",
        f"WARNINGS             {warnings or 'none'}",
    ]
    txt = "\n".join(lines)
    out.with_suffix(".txt").write_text(txt + "\n")
    print(txt)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
