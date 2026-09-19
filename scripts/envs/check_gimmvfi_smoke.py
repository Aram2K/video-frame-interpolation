#!/usr/bin/env python
"""Checks for a GIMM-VFI harness smoke run (used by smoke_gimmvfi.sbatch). Exit code 1 on any failure.

  file checks : frame count (N-1)*sf+1, run_report.json, uint16 (H,W,3), genuine frames bit-exact,
                generated frames not constant, mean brightness plausible, t=0.1 closer (mean abs diff) to
                img0 than img1 and t=0.9 the reverse (every pair), sub-8-bit values present when the source is
                8-bit-in-16 (proves no 8-bit quantisation in the path)
  --direct    : reloads the adapter on the GPU and calls interpolate() on the first pair: float32, finite,
                range [0,1] BEFORE the harness clamp, same pixels as the written TIFFs, 1/65535 input
                sensitivity, NaN input -> ValueError, NaN injected inside the model -> RuntimeError
"""
import os
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(os.environ.get("MVFI_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
from frameio import IMG_EXTS, natural_key, read_image  # noqa: E402

FAILURES = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


def mad(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def to_u16(t):  # identical to vfi_harness.to_u16
    import torch
    t = t.detach().float()
    return (t.clamp(0, 1) * 65535.0).round().to(torch.int32).permute(1, 2, 0).cpu().numpy().astype(np.uint16)


def to_float(a):  # identical to vfi_harness.to_float for uint16 input
    import torch
    return torch.from_numpy(a.astype(np.float32) / 65535.0).permute(2, 0, 1).contiguous()


def file_checks(out, src_files, sf):
    N = len(src_files)
    n_out = (N - 1) * sf + 1
    frames = sorted(out.glob("frame_*.tif"))
    check(len(frames) == n_out, f"{len(frames)} frames written, expected {n_out}")
    rep_p = out / "logs" / "run_report.json"
    check(rep_p.is_file(), "logs/run_report.json present")
    rep = json.loads(rep_p.read_text()) if rep_p.is_file() else {}
    check(rep.get("output_frames") == n_out, f"run_report output_frames={rep.get('output_frames')}")
    print("  timing:", json.dumps(rep.get("timing_s")), "\n  gpu:", json.dumps(rep.get("gpu")))
    src = [read_image(p) for p in src_files]
    H, W = src[0].shape[:2]
    eight_in_16 = all(not (s % 257).any() for s in src)
    print(f"  source {W}x{H} {src[0].dtype}; values all multiples of 257 (8-bit content in 16-bit): {eight_in_16}")
    print("  k    t    mean     std    min    max   MAD->img0 MAD->img1  frac(v%257!=0)")
    fracs, bad_shape, rows = [], [], {}
    for k in range(n_out):
        p = out / f"frame_{k:06d}.tif"
        if not p.is_file():
            continue
        a = tifffile.imread(p)
        if a.dtype != np.uint16 or a.shape != (H, W, 3):
            bad_shape.append((k, a.dtype, a.shape))
            continue
        j, i = divmod(k, sf)
        if i == 0:
            check(np.array_equal(a, src[j]), f"frame {k}: genuine, bit-exact to {src_files[j].name}")
            continue
        t = i / sf
        d0, d1 = mad(a, src[j]), mad(a, src[j + 1])
        frac = float(((a % 257) != 0).mean())
        fracs.append(frac)
        rows[k] = (j, t, d0, d1)
        print(f"  {k:<4d} {t:.1f} {a.mean():8.1f} {a.std():7.1f} {a.min():6d} {a.max():6d} {d0:9.1f} {d1:9.1f}  {frac:.3f}")
        check(a.std() > 100, f"frame {k}: not constant (std {a.std():.1f})")
        expect = (1 - t) * src[j].mean() + t * src[j + 1].mean()
        check(abs(a.mean() - expect) < 0.05 * 65535, f"frame {k}: mean {a.mean():.0f} near lerp of sources {expect:.0f}")
    check(not bad_shape, f"all frames uint16 {(H, W, 3)} {bad_shape[:3]}")
    for j in range(N - 1):
        k1, k9 = j * sf + 1, j * sf + sf - 1
        if k1 in rows:
            _, _, d0, d1 = rows[k1]
            check(d0 < d1, f"pair {j}: t={1/sf:.1f} frame {k1} closer to img0 ({d0:.1f}) than img1 ({d1:.1f})")
        if k9 in rows:
            _, _, d0, d1 = rows[k9]
            check(d1 < d0, f"pair {j}: t={(sf-1)/sf:.1f} frame {k9} closer to img1 ({d1:.1f}) than img0 ({d0:.1f})")
        d0s = [rows[k][2] for k in range(j * sf + 1, (j + 1) * sf) if k in rows]
        mono = all(b >= a - 1e-6 for a, b in zip(d0s, d0s[1:]))
        print(f"  pair {j}: MAD->img0 over t non-decreasing: {mono} (informational)")
    if eight_in_16 and fracs:
        check(np.mean(fracs) > 0.3, f"generated frames have sub-8-bit values (mean fraction {np.mean(fracs):.3f})")
    return src


def direct_checks(src, adapter_path, variant, ds, sf):
    import torch
    spec = importlib.util.spec_from_file_location("gimmvfi_adapter", adapter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = torch.device("cuda")
    ad = mod.Adapter(dev, model_variant=variant, ds_factor=ds)
    info = ad.info()
    print("  adapter info:", json.dumps({k: info[k] for k in ("name", "repo_commit", "settings", "runtime")}))
    check(info["repo_commit"] == mod.REPO_COMMIT_EXPECTED, f"repo commit {info['repo_commit']}")
    ts = [k / sf for k in range(1, sf)]
    x0, x1 = to_float(src[0]).to(dev), to_float(src[1]).to(dev)
    torch.cuda.reset_peak_memory_stats()
    outs = ad.interpolate(x0, x1, ts)
    torch.cuda.synchronize()
    check(len(outs) == len(ts), f"{len(outs)} outputs for {len(ts)} ts")
    for t, o, k in zip(ts, outs, range(1, sf)):
        finite = bool(torch.isfinite(o).all())
        lo, hi = float(o.min()), float(o.max())
        check(o.dtype == torch.float32 and finite and lo >= 0.0 and hi <= 1.0,
              f"direct t={t:.1f}: {o.dtype}, finite={finite}, raw range [{lo:.6f}, {hi:.6f}]")
        disk = tifffile.imread(OUT / f"frame_{k:06d}.tif")
        diff = np.abs(to_u16(o).astype(np.int32) - disk.astype(np.int32))
        check(diff.mean() < 1.0, f"direct t={t:.1f} vs written frame {k}: mean|diff| {diff.mean():.4f}, max {diff.max()} codes")
    print(f"  direct call peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    # 16-bit sensitivity: +1 code on both inputs should move the output by ~1 code (0 if quantised to 8 bits)
    d = 1.0 / 65535.0
    o1 = ad.interpolate((x0 + d).clamp(0, 1), (x1 + d).clamp(0, 1), [0.5])[0]
    o0 = ad.interpolate(x0, x1, [0.5])[0]
    shift = float((o1 - o0).mean() * 65535.0)
    check(0.2 < shift < 5.0, f"+1/65535 on the inputs moves the t=0.5 output by {shift:.3f} codes on average")
    # NaN handling
    bad = x0.clone()
    bad[0, 10, 10] = float("nan")
    try:
        ad.interpolate(bad, x1, [0.5])
        check(False, "NaN input rejected")
    except ValueError as e:
        check(True, f"NaN input rejected: {e}")
    hook = ad.model.cnn_encoder.register_forward_hook(lambda m, i, o: o * float("nan"))
    try:
        ad.interpolate(x0, x1, [0.5])
        check(False, "NaN inside the model turned into an error")
    except RuntimeError as e:
        check("assert" in str(e) or "non-finite" in str(e), f"NaN inside the model -> RuntimeError: {str(e)[:120]}...")
    finally:
        hook.remove()
    again = ad.interpolate(x0, x1, [0.5])[0]
    check(bool(torch.equal(again, o0)) or float((again - o0).abs().max()) < 1e-4,
          f"adapter still usable after the error (max|diff| {float((again - o0).abs().max()):.2e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--sf", type=int, default=10)
    ap.add_argument("--max_input_frames", type=int, default=None)
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--adapter", default=str(ROOT / "scripts/adapters/gimmvfi.py"))
    ap.add_argument("--model_variant", default="R-P")
    ap.add_argument("--ds_factor", type=float, default=0.25)
    args = ap.parse_args()
    OUT = Path(args.out)
    files = sorted([p for p in Path(args.input).iterdir() if p.suffix.lower() in IMG_EXTS],
                   key=lambda p: natural_key(p.name))[: args.max_input_frames]
    print(f"== checks for {OUT}")
    src = file_checks(OUT, files, args.sf)
    if args.direct:
        print("== direct adapter checks")
        direct_checks(src, args.adapter, args.model_variant, args.ds_factor, args.sf)
    if FAILURES:
        print(f"CHECKS FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")
