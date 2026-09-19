#!/usr/bin/env python
"""Model-agnostic 16-bit harness for two-frame (pairwise) interpolation models.

    python vfi_harness.py --adapter adapters/<name>.py --input DIR --out_dir OUT --temporal_sf 10 --fps 25 \
        [--input_stride 2] [--max_input_frames N] [--adapter_arg key=value ...]

Output layout is identical to ldf_tiff_infer.py so make_previews.py / eval_holdout.py work unchanged:
  frame_XXXXXX.tif   final 16-bit sequence; genuine positions = ORIGINAL pixels (bit-exact)
  reconstructed/     same frames as hard links (pairwise models never regenerate genuine frames)
  original/          symlinks to the genuine source frames at their output index
  frame_map.csv, logs/run_report.json, logs/holdout_metrics.csv (hold-out mode)

Indexing: N inputs, factor s -> (N-1)*s + 1 outputs; output k is genuine input k/s when k % s == 0,
else generated at t = (k % s)/s between inputs k//s and k//s + 1 (same convention as LDF-VFI).

ADAPTER CONTRACT (a python file defining `class Adapter`):
    Adapter(device: torch.device, **adapter_args)
    .info() -> dict           repo URL, commit, weights files + sha256, settings, licence
    .interpolate(img0, img1, ts) -> list of tensors
        img0, img1: float32 tensors (3, H, W) on `device`, RGB, values in [0, 1] (full precision of the
        source; 16-bit sources arrive as value/65535). ts: list of floats in (0, 1).
        Return one (3, H, W) tensor per t (any float dtype, any device), values nominally in [0, 1].
        Padding/cropping, normalisation and batching are the adapter's job. Must not quantise.
    optional .interpolate_sequence(frames_reader, sf) -> iterator of (out_index, tensor)   (sequence models)
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frameio import IMG_EXTS, natural_key, read_image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_commit(repo):
    """Commit id without a git binary (compute nodes have none)."""
    repo = Path(repo)
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (repo / ".git" / ref).exists():
            return (repo / ".git" / ref).read_text().strip()
        for line in (repo / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("vfi_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Adapter


def as_uint16(a):
    if a.dtype == np.uint16:
        return a
    if a.dtype == np.uint8:
        return a.astype(np.uint16) * 257
    raise TypeError(f"unsupported source dtype {a.dtype}")


def to_float(a):
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return torch.from_numpy(a.astype(np.float32) / scale).permute(2, 0, 1).contiguous()


def to_u16(t):
    t = t.detach().float()
    return (t.clamp(0, 1) * 65535.0).round().to(torch.int32).permute(1, 2, 0).cpu().numpy().astype(np.uint16)


def psnr16(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(65535.0 ** 2 / mse))


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        if v in ("true", "True"):
            v = True
        elif v in ("false", "False"):
            v = False
        out[k] = v
    return out


def write_tif(path, arr):
    tifffile.imwrite(path, arr, photometric="rgb", compression=None, metadata=None)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--input", required=True, help="folder of TIFF/PNG frames")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--temporal_sf", type=int, default=10)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--input_stride", type=int, default=1)
    ap.add_argument("--max_input_frames", type=int, default=None)
    ap.add_argument("--adapter_arg", action="append", default=[], help="key=value passed to Adapter()")
    ap.add_argument("--write_workers", type=int, default=8)
    ap.add_argument("--cpu", action="store_true", help="run the adapter on CPU (tests only)")
    args = ap.parse_args()

    t_start = time.time()
    out = Path(args.out_dir)
    for sub in ("reconstructed", "original", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if any(out.glob("frame_*.tif")):
        sys.exit(f"ERROR: {out} already holds frames; use a new --out_dir")
    sf, stride = args.temporal_sf, args.input_stride
    if stride > 1 and stride != sf:
        sys.exit("hold-out mode needs temporal_sf == input_stride")

    src = Path(args.input)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: natural_key(p.name))
    if args.max_input_frames:
        files = files[: args.max_input_frames]
    used = files[::stride]
    N = len(used)
    n_out = (N - 1) * sf + 1
    a0 = read_image(used[0])
    H, W = a0.shape[:2]
    print(f"input: {len(files)} files {W}x{H} {a0.dtype}; model input N={N}; sf={sf}; output={n_out}", flush=True)

    device = torch.device("cpu" if args.cpu else "cuda")
    t0 = time.time()
    Adapter = load_adapter(args.adapter)
    adapter = Adapter(device, **parse_kv(args.adapter_arg))
    t_load = time.time() - t0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pool = ThreadPoolExecutor(args.write_workers)
    futures, rows, holdout = [], [], []

    def emit(k, arr16, genuine):
        fp = out / f"frame_{k:06d}.tif"
        write_tif(fp, arr16)
        os.link(fp, out / "reconstructed" / fp.name)
        j = k // sf
        row = {"out_index": k, "time_s": round(k / args.fps, 6), "kind": "genuine" if genuine else "generated",
               "t_frac": (k % sf) / sf, "prev_src": used[j].name, "next_src": used[min(j + 1, N - 1)].name,
               "final_pixels": "original source (bit-exact)" if genuine else "model"}
        if genuine:
            os.symlink(os.path.relpath(used[j], out / "original"), out / "original" / fp.name)
        elif stride > 1:
            gt = as_uint16(read_image(files[k]))
            holdout.append({"out_index": k, "gt_src": files[k].name, "psnr_db": psnr16(arr16, gt),
                            "mean_abs_diff_16bit": float(np.abs(arr16.astype(np.int32) - gt).mean())})
        rows.append(row)

    def flush(limit):
        nonlocal futures
        if len(futures) > limit:
            for f in futures:
                f.result()
            futures = []

    ts = [k / sf for k in range(1, sf)]
    pair_times = []
    t_gen = time.time()
    prev_native = as_uint16(a0)
    prev = to_float(a0).to(device)
    futures.append(pool.submit(emit, 0, prev_native, True))
    for j in range(1, N):
        nxt_native = read_image(used[j])
        nxt = to_float(nxt_native).to(device)
        tp = time.time()
        outs = adapter.interpolate(prev, nxt, ts)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pair_times.append(time.time() - tp)
        if len(outs) != len(ts):
            raise RuntimeError(f"adapter returned {len(outs)} frames for {len(ts)} timesteps")
        for i, o in enumerate(outs):
            if tuple(o.shape[-2:]) != (H, W):
                raise RuntimeError(f"adapter output {tuple(o.shape)} != {(3, H, W)}")
            futures.append(pool.submit(emit, (j - 1) * sf + i + 1, to_u16(o), False))
        futures.append(pool.submit(emit, j * sf, as_uint16(nxt_native), True))
        prev = nxt
        flush(4 * args.write_workers)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"  pair {j}/{N - 1}: {pair_times[-1]:.1f} s, gpu peak {peak:.1f} GiB", flush=True)
    flush(-1)
    pool.shutdown()
    t_gen = time.time() - t_gen

    rows.sort(key=lambda r: r["out_index"])
    assert [r["out_index"] for r in rows] == list(range(n_out)), "missing output indices"
    with open(out / "frame_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if holdout:
        holdout.sort(key=lambda r: r["out_index"])
        with open(out / "logs" / "holdout_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(holdout[0].keys()))
            w.writeheader()
            w.writerows(holdout)

    info = adapter.info() if hasattr(adapter, "info") else {}
    report = {
        "input": str(src.resolve()), "input_files_used": len(files), "input_stride": stride,
        "source_dtype": str(a0.dtype), "width": W, "height": H,
        "model_input_frames": N, "temporal_sf": sf, "output_frames": n_out,
        "input_duration_s_at_fps": N / args.fps, "output_duration_s_at_fps": n_out / args.fps,
        "expansion_factor": n_out / N, "fps": args.fps, "paste_originals": True,
        "holdout_psnr_db_mean": float(np.mean([h["psnr_db"] for h in holdout])) if holdout else None,
        "timing_s": {"model_load": t_load, "generation_and_write": t_gen, "total": time.time() - t_start,
                     "sec_per_pair_mean": float(np.mean(pair_times)) if pair_times else None,
                     "sec_per_output_frame": t_gen / n_out},
        "gpu": ({"name": torch.cuda.get_device_name(0),
                 "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
                if device.type == "cuda" else None),
        "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "torch_cuda": torch.version.cuda, "harness_sha256": sha256(Path(__file__))},
        "model": {"adapter": str(Path(args.adapter).resolve()), "adapter_sha256": sha256(args.adapter),
                  "adapter_args": parse_kv(args.adapter_arg), **info},
        "settings": vars(args),
        "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES")},
        "output_dir": str(out.resolve()),
    }
    (out / "logs" / "run_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("output_frames", "holdout_psnr_db_mean", "timing_s", "gpu")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
