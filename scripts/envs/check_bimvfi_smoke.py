#!/usr/bin/env python
"""Post-run checks for a BiM-VFI harness run (used by smoke_bimvfi.sbatch).

    python check_bimvfi_smoke.py OUT_DIR INPUT_DIR [--sf 10] [--expect 31]

PASS criteria (exit 1 otherwise):
  * exactly --expect frame_*.tif, all uint16 (H, W, 3) at the source size
  * genuine positions are bit-exact copies of the source frames
  * the adapter saw no NaN/Inf (run_report.json model.raw_output_stats.nonfinite == 0), raw float output range
    sane (min > -0.5, max < 1.5, < 1 % of values clipped by the harness), written frames not constant
  * for every pair: the t=0.1 frame is closer (mean abs diff) to img0 than to img1, and t=0.9 the reverse
Informative only: distance profile over t, distance to a linear cross-fade, edge-strip vs interior deviation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frameio import IMG_EXTS, natural_key, read_image  # noqa: E402


def mad(a, b):
    return float(np.abs(a - b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("input_dir")
    ap.add_argument("--sf", type=int, default=10)
    ap.add_argument("--expect", type=int, default=31)
    a = ap.parse_args()
    out, sf = Path(a.out_dir), a.sf
    fails, res = [], {}

    frames = sorted(out.glob("frame_*.tif"))
    res["n_frames"] = len(frames)
    if len(frames) != a.expect:
        fails.append(f"expected {a.expect} frames, found {len(frames)}")
    rp = out / "logs" / "run_report.json"
    if not rp.exists() or len(frames) != (json.loads(rp.read_text())["output_frames"] if rp.exists() else -1):
        print("SMOKE FAIL:", fails or "", "run_report.json missing or frame count mismatch")
        sys.exit(1)
    report = json.loads(rp.read_text())
    n_in = report["model_input_frames"]
    src = sorted([p for p in Path(a.input_dir).iterdir() if p.suffix.lower() in IMG_EXTS],
                 key=lambda p: natural_key(p.name))[:n_in]
    H, W = report["height"], report["width"]

    st = report["model"].get("raw_output_stats", {})
    res["raw_output_stats"] = st
    if st.get("nonfinite", 1) != 0:
        fails.append(f"non-finite model output: {st.get('nonfinite')}")
    if not (st.get("min", -9) > -0.5 and st.get("max", 9) < 1.5):
        fails.append(f"raw output range suspicious: [{st.get('min')}, {st.get('max')}]")
    clipped = st.get("frac_below0", 1) + st.get("frac_above1", 1)
    if clipped > 0.01:
        fails.append(f"{clipped:.3%} of raw values outside [0,1]")

    def load(k):
        arr = tifffile.imread(frames[k])
        if arr.dtype != np.uint16 or arr.shape != (H, W, 3):
            fails.append(f"{frames[k].name}: {arr.dtype} {arr.shape}")
        return arr

    pairs, stats = [], []
    for j in range(n_in - 1):
        k0, k1 = j * sf, (j + 1) * sf
        f0, f1 = load(k0), load(k1)
        for k, arr, s in ((k0, f0, j), (k1, f1, j + 1)):
            if not np.array_equal(arr, read_image(src[s])):
                fails.append(f"genuine frame {k} differs from {src[s].name}")
        g0, g1 = f0.astype(np.float32) / 65535, f1.astype(np.float32) / 65535
        prof = []
        for i in range(1, sf):
            t = i / sf
            g = load(k0 + i).astype(np.float32) / 65535
            if g.max() == g.min():
                fails.append(f"frame {k0 + i} is constant")
            d0, d1 = mad(g, g0), mad(g, g1)
            blend = (1 - t) * g0 + t * g1
            e, dev = 16, np.abs(g - blend)
            strips = np.concatenate([dev[:e].ravel(), dev[-e:].ravel(), dev[:, :e].ravel(), dev[:, -e:].ravel()])
            prof.append({"t": t, "mad_to_img0": d0, "mad_to_img1": d1, "mad_to_linear_blend": float(dev.mean()),
                         "edge16_dev_from_blend": float(strips.mean()),
                         "interior_dev_from_blend": float(dev[e:-e, e:-e].mean()),
                         "min16": int(g.min() * 65535 + 0.5), "max16": int(g.max() * 65535 + 0.5)})
            stats.append((g.min(), g.max()))
        p01, p09 = prof[0], prof[-1]
        ok = p01["mad_to_img0"] < p01["mad_to_img1"] and p09["mad_to_img1"] < p09["mad_to_img0"]
        if not ok:
            fails.append(f"pair {j}: temporal order wrong (t=0.1: {p01['mad_to_img0']:.5f} vs {p01['mad_to_img1']:.5f};"
                         f" t=0.9: {p09['mad_to_img0']:.5f} vs {p09['mad_to_img1']:.5f})")
        pairs.append({"pair": j, "src": [src[j].name, src[j + 1].name], "mad_img0_img1": mad(g0, g1),
                      "order_ok": ok, "profile": prof})
        print(f"pair {j} ({src[j].name} -> {src[j + 1].name}): |img0-img1| = {mad(g0, g1):.5f}  order_ok={ok}")
        print("   t    d(img0)   d(img1)   d(blend)  edge16  interior   min..max (16-bit)")
        for p in prof:
            print(f"  {p['t']:.1f}  {p['mad_to_img0']:.5f}  {p['mad_to_img1']:.5f}  {p['mad_to_linear_blend']:.5f}"
                  f"  {p['edge16_dev_from_blend']:.5f}  {p['interior_dev_from_blend']:.5f}  {p['min16']}..{p['max16']}")
    res["pairs"] = pairs
    res["generated_min_max_float"] = [float(min(s[0] for s in stats)), float(max(s[1] for s in stats))] if stats else None
    res["timing_s"] = report["timing_s"]
    res["gpu"] = report["gpu"]
    res["fails"] = fails
    res["pass"] = not fails
    (out / "logs" / "smoke_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_frames", "raw_output_stats", "generated_min_max_float",
                                          "timing_s", "gpu", "fails", "pass")}, indent=1))
    print("SMOKE", "PASS" if not fails else "FAIL")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
