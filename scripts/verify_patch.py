#!/usr/bin/env python
"""Prove the float-output sampler is numerically the upstream sampler minus the uint8 step.

Runs, on the same 8-bit frames and the same seed, (a) upstream `sample_skip_concat` fed by the
upstream PNG reader and (b) `sample_skip_concat_float` (bf16 VAE) fed by our float reader, for the
first M chunks, and compares upstream uint8 with floor(clip(ours, 0, 255)) — `.byte()` truncates.

Usage: python verify_patch.py --frames DIR_OF_PNG --model_path ... --vae_path ... [--chunks 3]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ldf_tiff_infer as W  # noqa: E402

G = W.G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vae_path", required=True)
    ap.add_argument("--temporal_sf", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--out_json", required=True)
    a = ap.parse_args()
    ns = argparse.Namespace(
        model_path=a.model_path, vae_path=a.vae_path, attention_type="slide_chunk_all_block_2x1x1",
        vae_batch_size=16, tile_min_h=256, tile_min_w=256, tile_min_t=20, tile_stride_h=192, tile_stride_w=192)
    device = torch.device("cuda")
    model, _ = W.build_model(ns, device, torch.bfloat16)
    float_sampler = W.build_float_sampler(Path(a.out_json).parent)
    common = dict(train_num_frames=60, tile_min_t=20, temporal_upsample="nearest", spatial_sf=1,
                  temporal_sf=a.temporal_sf, nT_cond=1, num_steps=16, t_cond=0.1, t_shift=8.0,
                  device=device, weight_dtype=torch.bfloat16, verbose=False)

    up_reader = G.VideoFrameLazyReader(a.frames, dimension_order="NCHW")
    files = sorted(Path(a.frames).glob("*.png"), key=lambda p: W.natural_key(p.name))
    fl_reader = W.FloatFrameReader(files)

    def run(sampler, reader, **extra):
        G.set_seed(42)
        outs = []
        for i, (_lq, pred) in enumerate(sampler(reader, model, **common, **extra)):
            outs.append(pred.cpu())
            if i + 1 >= a.chunks:
                break
        return torch.cat(outs)

    u8check_sampler = W.build_float_sampler(Path(a.out_json).parent, keep_uint8_output=True)

    def cmp(x, y):
        d = (x.to(torch.int16) - y.to(torch.int16)).abs()
        return {"identical_fraction": float((d == 0).float().mean()), "max_abs_diff_codes": int(d.max()),
                "mean_abs_diff_codes": float(d.float().mean())}

    with torch.no_grad():
        up = run(G.sample_skip_concat, up_reader)
        up_u8check = run(u8check_sampler, fl_reader, vae_dtype=torch.bfloat16)
        ours = run(float_sampler, fl_reader, vae_dtype=torch.bfloat16)
        up_again = run(G.sample_skip_concat, up_reader)
    ours_u8 = ours.clamp(0, 255).floor().to(torch.uint8)
    plumbing = cmp(up, up_u8check)
    res = {
        "frames_compared": int(up.shape[0]),
        "upstream_vs_upstream_rerun (GPU determinism)": cmp(up, up_again),
        "upstream_vs_all_patches_except_output + float reader (must be identical)": plumbing,
        "upstream_vs_floor(float output) (differs only by upstream's bf16 add/mul before .byte())": cmp(up, ours_u8),
        "float_output_fractional_part_mean": float((ours - ours.floor()).mean()),
        "verdict": "PLUMBING BIT-EXACT" if plumbing["max_abs_diff_codes"] == 0 else "PLUMBING DIFFERS - investigate",
    }
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
