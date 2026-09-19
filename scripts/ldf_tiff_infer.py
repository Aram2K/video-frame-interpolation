#!/usr/bin/env python
"""LDF-VFI temporal interpolation of a high-bit-depth TIFF sequence.

Upstream generate.py only reads video files / 8-bit PNG-JPEG folders (uint8) and
quantises every generated chunk to uint8 inside `sample_skip_concat`
(`.add(1).mul(127.5).clip(0, 255).byte()`), then writes an H.264 yuv420p MP4.

This wrapper keeps upstream code untouched and instead:
  * reads uint16 TIFFs straight to float32 (value/65535*255, i.e. the scale the upstream
    sampler expects before its `.div(127.5).sub(1)`), so no 8-bit step happens on input;
  * builds a float-output copy of the upstream sampler by exact, counted string
    replacements (the diff is saved next to the results);
  * optionally runs the VAE (encoder, conditional decoder) in float32 instead of bfloat16;
  * streams 16-bit TIFFs to disk: `reconstructed/` = everything the model produced,
    top level = final sequence where genuine positions carry the ORIGINAL source pixels.

Frame indexing (upstream convention, generate.py:188-189 and :549):
  N input frames, factor s -> output length (N-1)*s + 1; output k is genuine input k/s when
  k % s == 0, otherwise generated at fractional time (k % s)/s between inputs k//s and k//s+1.
"""
import argparse
import csv
import difflib
import hashlib
import inspect
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import torch

REPO = Path(os.environ.get("MVFI_REPO", str(Path(__file__).resolve().parents[1] / "src" / "LDF-VFI")))
sys.path.insert(0, str(REPO))
import generate as G  # noqa: E402  (upstream module, unmodified)

from frameio import IMG_EXTS, TIFF_EXTS, natural_key, read_image, to_hwc3  # noqa: E402,F401


# ---------------------------------------------------------------------------------------------
# Readers. Both index like the upstream readers: reader[i] -> (C,H,W), reader[a:b] -> (T,C,H,W),
# but return float32 on the 0..255 scale instead of uint8.

class FloatFrameReader:
    def __init__(self, files, cache_size=48):
        self.files = list(files)
        a0 = read_image(self.files[0])
        self.src_dtype = a0.dtype
        self.maxval = float(np.iinfo(a0.dtype).max) if np.issubdtype(a0.dtype, np.integer) else 1.0
        self.H, self.W = a0.shape[:2]
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.files)

    def native(self, idx):
        """Original array (H, W, 3), source dtype — used to paste genuine frames back bit-exactly."""
        return read_image(self.files[idx])

    def _load(self, idx):
        with self._lock:
            if idx in self._cache:
                self._cache.move_to_end(idx)
                return self._cache[idx]
        a = self.native(idx)
        if a.shape[:2] != (self.H, self.W):
            raise ValueError(f"{self.files[idx]}: size {a.shape[:2]} != {(self.H, self.W)}")
        t = torch.from_numpy(a.astype(np.float32) * (255.0 / self.maxval)).permute(2, 0, 1).contiguous()
        with self._lock:
            self._cache[idx] = t
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return t

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            ids = list(range(*idx.indices(len(self))))
        elif isinstance(idx, (list, tuple)):
            ids = [int(i) for i in idx]
        elif torch.is_tensor(idx) and idx.ndim > 0:
            ids = idx.tolist()
        else:
            i = int(idx)
            return self._load(i + len(self) if i < 0 else i)
        if not ids:
            return torch.empty((0, 3, self.H, self.W), dtype=torch.float32)
        return torch.stack([self._load(i + len(self) if i < 0 else i) for i in ids])


class SubsampledReader:
    """View of every k-th frame (hold-out evaluation)."""
    def __init__(self, base, stride):
        self.base, self.stride = base, stride
        self.idx = list(range(0, len(base), stride))

    def __len__(self):
        return len(self.idx)

    def native(self, i):
        return self.base.native(self.idx[i])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.base[[self.idx[i] for i in range(*idx.indices(len(self)))]]
        i = int(idx)
        return self.base[self.idx[i]]


def video_to_frames(video, out_dir):
    """Decode a video once to lossless 8-bit PNGs (for the bundled demo / hold-out tests)."""
    from torchcodec.decoders import VideoDecoder
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    dec = VideoDecoder(str(video), dimension_order="NHWC")
    for i in range(len(dec)):
        Image.fromarray(dec[i].numpy()).save(out_dir / f"src_{i:06d}.png", compress_level=1)
    return sorted(out_dir.glob("src_*.png"))


# ---------------------------------------------------------------------------------------------
# Float-output sampler, derived from the upstream one by counted replacements.

SAMPLER_PATCHES = [
    # name, old, new, expected count
    ("rename", "def sample_skip_concat(", "def sample_skip_concat_float(", 1),
    ("vae_dtype arg", "device='cuda', weight_dtype=torch.bfloat16,",
     "device='cuda', weight_dtype=torch.bfloat16, vae_dtype=torch.bfloat16,", 1),
    ("keep float output (no uint8)", ".add(1).mul(127.5).clip(0, 255).byte()", ".float().add(1).mul(127.5)", 4),
    ("LQ in VAE dtype", "lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=weight_dtype).div(127.5).sub(1)",
     "lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=vae_dtype).div(127.5).sub(1)", 4),
    ("DiT condition in DiT dtype", "y = model.lq_encoder.encode(lq, for_train=True)",
     "y = model.lq_encoder.encode(lq, for_train=True).to(weight_dtype)", 4),
    ("decode latents in VAE dtype", "vae_decode(model.vae, xt, ", "vae_decode(model.vae, xt.to(vae_dtype), ", 4),
]


def build_float_sampler(log_dir, keep_uint8_output=False):
    """keep_uint8_output=True applies every patch except the output one (used to prove the plumbing
    changes are bit-exact no-ops in bf16 mode)."""
    src = inspect.getsource(G.sample_skip_concat)
    patched = src
    patches = [p for p in SAMPLER_PATCHES if not (keep_uint8_output and p[0] == "keep float output (no uint8)")]
    if keep_uint8_output:
        patches = [("rename", "def sample_skip_concat(", "def sample_skip_concat_u8check(", 1) if p[0] == "rename" else p
                   for p in patches]
    for name, old, new, n in patches:
        c = patched.count(old)
        if c != n:
            raise RuntimeError(f"sampler patch '{name}': expected {n} occurrences, found {c} — upstream changed?")
        patched = patched.replace(old, new)
    fname = "sample_skip_concat_u8check" if keep_uint8_output else "sample_skip_concat_float"
    ns = G.__dict__
    exec(compile(patched, f"<{fname}>", "exec"), ns)
    diff = "".join(difflib.unified_diff(src.splitlines(True), patched.splitlines(True),
                                        "generate.py:sample_skip_concat", fname))
    (log_dir / f"{fname}.diff" if keep_uint8_output else log_dir / "sampler_patch.diff").write_text(diff)
    return ns[fname]


def build_model(args, device, vae_dtype):
    tiled_kwargs = dict(
        tile_sample_min_height=args.tile_min_h, tile_sample_min_width=args.tile_min_w,
        tile_sample_min_time=args.tile_min_t,
        tile_sample_stride_height=args.tile_stride_h, tile_sample_stride_width=args.tile_stride_w,
        spatial_compression_ratio=8, temporal_compression_ratio=4,
    )
    vae_cls = G.Wan2_1SpatialTiledConditionEncoder3Dv2  # --vae_type=wan2_1_cond_v2 (quick_start)
    vae = vae_cls(args.vae_path, args.vae_batch_size, **tiled_kwargs)
    vae.dtype = vae_dtype  # upstream hard-codes bfloat16 in __init__; init() uses this attribute
    vae.init(device)
    transformer = G.WanTransformer3DModel.from_pretrained(args.model_path, subfolder="transformer")
    transformer.set_attention_type(args.attention_type)
    lq_encoder = vae_cls(args.vae_path, args.vae_batch_size, **tiled_kwargs)
    lq_encoder.dtype = vae_dtype
    lq_encoder.init(device)
    msk_encoder = G.MaskSpatialTiledEncoder3D(**tiled_kwargs)
    model = G.Precond(transformer=transformer, vae=vae, lq_encoder=lq_encoder, msk_encoder=msk_encoder)
    model.to(device=device, dtype=torch.bfloat16).eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_vae = sum(p.numel() for p in vae.vae.parameters())
    return model, {"transformer_params": n_params, "vae_params_each": n_vae}


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def upstream_state():
    """Commit from .git without a git binary (compute nodes have none) + integrity of tracked files
    against the sha256 list recorded on the login node right after cloning."""
    head = (REPO / ".git" / "HEAD").read_text().strip()
    commit = head
    if head.startswith("ref: "):
        ref = head[5:]
        f = REPO / ".git" / ref
        if f.exists():
            commit = f.read_text().strip()
        else:
            for line in (REPO / ".git" / "packed-refs").read_text().splitlines():
                if line.endswith(" " + ref):
                    commit = line.split()[0]
    manifest = REPO.parent.parent / "provenance" / "upstream_tracked_sha256.txt"
    changed = []
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest, rel = line.split(maxsplit=1)
            if sha256(REPO / rel) != digest:
                changed.append(rel)
    return {"upstream_commit": commit, "upstream_changed_files": changed if manifest.exists() else "no manifest"}


def gpu_peak_gib():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def fake_sampler(lq_reader, model, temporal_sf=4, **_):
    """Linear blend between genuine frames, yielded in 20-frame chunks like the real sampler (CPU test only)."""
    n = len(lq_reader)
    total = n * temporal_sf
    frames = []
    for k in range(total):
        j, f = divmod(k, temporal_sf)
        a = lq_reader[min(j, n - 1)]
        b = lq_reader[min(j + 1, n - 1)]
        frames.append(a * (1 - f / temporal_sf) + b * (f / temporal_sf))
        if len(frames) == 20:
            yield None, torch.stack(frames)
            frames = []
    if frames:
        yield None, torch.stack(frames)


def psnr16(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(65535.0 ** 2 / mse))


def as_uint16(a):
    if a.dtype == np.uint16:
        return a
    if a.dtype == np.uint8:
        return a.astype(np.uint16) * 257
    raise TypeError(f"unsupported source dtype {a.dtype}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of TIFF/PNG frames, or a video file")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", required=True, help="folder that contains transformer/ (from_pretrained subfolder)")
    ap.add_argument("--vae_path", required=True)
    ap.add_argument("--temporal_sf", type=int, default=10)
    ap.add_argument("--fps", type=float, default=25.0, help="playback rate of the OUTPUT (metadata / timing only)")
    ap.add_argument("--input_stride", type=int, default=1,
                    help="hold-out test: feed only every k-th input frame (use temporal_sf=k) and score the dropped ones")
    ap.add_argument("--max_input_frames", type=int, default=None)
    ap.add_argument("--vae_dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--no_paste_originals", action="store_true",
                    help="top-level sequence = pure model output (default pastes original pixels at genuine positions)")
    # upstream quick_start/generate.sh defaults
    ap.add_argument("--attention_type", default="slide_chunk_all_block_2x1x1")
    ap.add_argument("--temporal_upsample", default="nearest")
    ap.add_argument("--num_frames", type=int, default=60)
    ap.add_argument("--vae_batch_size", type=int, default=16)
    ap.add_argument("--tile_min_h", type=int, default=256)
    ap.add_argument("--tile_min_w", type=int, default=256)
    ap.add_argument("--tile_min_t", type=int, default=20)
    ap.add_argument("--tile_stride_h", type=int, default=192)
    ap.add_argument("--tile_stride_w", type=int, default=192)
    ap.add_argument("--sampling_steps", type=int, default=16)
    ap.add_argument("--t_shift", type=float, default=8.0)
    ap.add_argument("--t_cond", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write_workers", type=int, default=8)
    ap.add_argument("--fake_model", action="store_true",
                    help="CPU pipeline test: replace the model by linear blending (checks IO/indexing/reports only)")
    args = ap.parse_args()

    t_start = time.time()
    out = Path(args.out_dir)
    for sub in ("reconstructed", "original", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if any(out.glob("frame_*.tif")) or any((out / "reconstructed").glob("frame_*.tif")):
        sys.exit(f"ERROR: {out} already holds frames; use a new --out_dir (results are never overwritten)")
    log_dir = out / "logs"
    sf = args.temporal_sf
    if not 2 <= sf <= 16:
        sys.exit("temporal_sf must be in 2..16 (upstream README)")

    # ---- input
    src = Path(args.input)
    if src.is_file():
        files = video_to_frames(src, out / "decoded_source")
    else:
        files = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: natural_key(p.name))
    if args.max_input_frames:
        files = files[: args.max_input_frames]
    full = FloatFrameReader(files)
    reader = full if args.input_stride == 1 else SubsampledReader(full, args.input_stride)
    if args.input_stride > 1 and args.input_stride != sf:
        sys.exit("hold-out mode needs temporal_sf == input_stride")
    N = len(reader)
    n_out = (N - 1) * sf + 1
    print(f"input: {len(files)} files, {full.W}x{full.H}, {full.src_dtype}; model input N={N}; "
          f"sf={sf}; output frames={n_out}", flush=True)

    # ---- model
    device = torch.device("cpu" if args.fake_model else "cuda")
    G.set_seed(args.seed)
    vae_dtype = torch.float32 if args.vae_dtype == "fp32" else torch.bfloat16
    if vae_dtype == torch.float32:
        # torch enables TF32 for cuDNN convs by default; for a genuine float32 VAE pass, turn it off.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    if args.fake_model:
        model, counts = None, {"fake_model": True}
        build_float_sampler(log_dir)  # still validates the patch against upstream source
        sampler = fake_sampler
    else:
        model, counts = build_model(args, device, vae_dtype)
        sampler = build_float_sampler(log_dir)
        torch.cuda.reset_peak_memory_stats()
    t_load = time.time() - t0

    gen = sampler(
        reader, model, train_num_frames=args.num_frames, tile_min_t=args.tile_min_t,
        temporal_upsample=args.temporal_upsample, spatial_sf=1, temporal_sf=sf, nT_cond=1,
        num_steps=args.sampling_steps, t_cond=args.t_cond, t_shift=args.t_shift,
        device=device, weight_dtype=torch.bfloat16, vae_dtype=vae_dtype, verbose=True,
    )

    # ---- streaming writer
    pool = ThreadPoolExecutor(max_workers=args.write_workers)
    pending, rows, anchors, holdout = [], [], [], []

    def write_tif(path, arr):
        tifffile.imwrite(path, arr, photometric="rgb", compression=None, metadata=None)

    def handle(k, rec16):
        rp = out / "reconstructed" / f"frame_{k:06d}.tif"
        fp = out / f"frame_{k:06d}.tif"
        write_tif(rp, rec16)
        genuine = (k % sf == 0)
        j, frac = k // sf, (k % sf) / sf
        row = {"out_index": k, "time_s": round(k / args.fps, 6), "kind": "genuine" if genuine else "generated",
               "t_frac": frac, "prev_src": files[j * args.input_stride].name,
               "next_src": files[min(j + 1, N - 1) * args.input_stride].name}
        if genuine:
            orig = as_uint16(reader.native(j))
            os.symlink(os.path.relpath(files[j * args.input_stride], out / "original"), out / "original" / fp.name)
            anchors.append({"out_index": k, "src": files[j * args.input_stride].name,
                            "psnr_vae_vs_orig_db": psnr16(rec16, orig),
                            "mean_abs_diff_16bit": float(np.abs(rec16.astype(np.int32) - orig).mean()),
                            "max_abs_diff_16bit": int(np.abs(rec16.astype(np.int32) - orig).max())})
            if args.no_paste_originals:
                os.link(rp, fp)
                row["final_pixels"] = "model (VAE-reconstructed genuine frame)"
            else:
                write_tif(fp, orig)
                back = tifffile.imread(fp)
                if not np.array_equal(back, orig):
                    raise RuntimeError(f"pasted original not bit-exact at {fp}")
                row["final_pixels"] = "original source (bit-exact)" if full.src_dtype == np.uint16 else "original source (8-bit x257)"
        else:
            os.link(rp, fp)
            row["final_pixels"] = "model"
            if args.input_stride > 1:  # dropped real frame exists at this position
                gt = as_uint16(full.native(k))
                holdout.append({"out_index": k, "gt_src": files[k].name, "psnr_db": psnr16(rec16, gt),
                                "mean_abs_diff_16bit": float(np.abs(rec16.astype(np.int32) - gt).mean())})
        rows.append(row)

    k = 0
    chunk_times = []
    t_gen = time.time()
    t_prev = t_gen
    value_range = [float("inf"), float("-inf")]
    for _lq, pred in gen:
        pred = pred.float()
        value_range = [min(value_range[0], pred.min().item()), max(value_range[1], pred.max().item())]
        rec = (pred / 255.0).clamp_(0, 1).mul_(65535.0).round_().to(torch.int32)
        rec = rec.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint16)
        now = time.time()
        chunk_times.append({"frames": int(rec.shape[0]), "seconds": now - t_prev})
        t_prev = now
        for f in rec:
            if k >= n_out:
                break
            pending.append(pool.submit(handle, k, np.ascontiguousarray(f)))
            k += 1
        if len(pending) > 4 * args.write_workers:
            for p in pending:
                p.result()
            pending = []
        print(f"  wrote up to frame {k}/{n_out}  gpu peak {gpu_peak_gib():.1f} GiB", flush=True)
    for p in pending:
        p.result()
    pool.shutdown()
    t_gen = time.time() - t_gen
    if k != n_out:
        raise RuntimeError(f"sampler produced {k} frames, expected {n_out}")

    rows.sort(key=lambda r: r["out_index"])
    with open(out / "frame_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    anchors.sort(key=lambda r: r["out_index"])
    with open(log_dir / "anchor_fidelity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(anchors[0].keys()))
        w.writeheader()
        w.writerows(anchors)
    if holdout:
        holdout.sort(key=lambda r: r["out_index"])
        with open(log_dir / "holdout_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(holdout[0].keys()))
            w.writeheader()
            w.writerows(holdout)

    finite = [a["psnr_vae_vs_orig_db"] for a in anchors if np.isfinite(a["psnr_vae_vs_orig_db"])]
    report = {
        "input": str(src.resolve()), "input_files_used": len(files), "input_stride": args.input_stride,
        "source_dtype": str(full.src_dtype), "width": full.W, "height": full.H,
        "model_input_frames": N, "temporal_sf": sf, "output_frames": n_out,
        "input_duration_s_at_fps": N / args.fps, "output_duration_s_at_fps": n_out / args.fps,
        "expansion_factor": n_out / N, "fps": args.fps,
        "vae_dtype": args.vae_dtype, "dit_dtype": "bf16",
        "pred_value_range_0_255": value_range,
        "paste_originals": not args.no_paste_originals,
        "anchor_psnr_db_mean": float(np.mean(finite)) if finite else None,
        "anchor_psnr_db_min": float(np.min(finite)) if finite else None,
        "holdout_psnr_db_mean": float(np.mean([h["psnr_db"] for h in holdout])) if holdout else None,
        "timing_s": {"model_load": t_load, "generation_and_write": t_gen, "total": time.time() - t_start,
                     "sec_per_output_frame": t_gen / n_out, "chunks": chunk_times},
        "gpu": ({"name": torch.cuda.get_device_name(0),
                 "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
                if torch.cuda.is_available() and not args.fake_model else None),
        "software": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(), **upstream_state(),
                     "generate_py_sha256": sha256(REPO / "generate.py")},
        "model": {"model_path": args.model_path, "vae_path": args.vae_path, **counts},
        "settings": vars(args),
        "slurm": {k2: os.environ.get(k2) for k2 in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
                                                     "CUDA_VISIBLE_DEVICES")},
        "output_dir": str(out.resolve()),
    }
    (log_dir / "run_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k2: report[k2] for k2 in ("output_frames", "expansion_factor", "anchor_psnr_db_mean",
                                                "holdout_psnr_db_mean", "timing_s", "gpu")}, indent=1, default=str))


if __name__ == "__main__":
    main()
