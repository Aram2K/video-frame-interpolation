#!/bin/bash
# EMA-VFI (ours_t, arbitrary t) on one real shot through the shared 16-bit harness (A100 only).
#   SHOT=test01 VARIANT=ds025 sbatch scripts/envs/run_emavfi.sbatch
#   SHOT=test01 VARIANT=ds025_rec709 sbatch ...
# Input : input/$SHOT, or input/${SHOT}_rec709 when VARIANT ends in _rec709
# Output: output/$SHOT/ema-vfi/$VARIANT   (skipped if logs/run_report.json exists; an incomplete dir is removed)
# VARIANT = underscore-separated tokens, any order ("rec709" only as the last token):
#   ds025 | ds05 | ds1   flow down_scale 0.25 (default; official 4K setting) | 0.5 (official 2K) | 1.0 (not
#                        timed at steady state: pass a longer --time on the sbatch command line for ds1)
#   notta                test-time flip augmentation off (default on, as upstream for ours_t)
#   fasttta              batched TTA (more memory)
#   small                ours_small_t checkpoint (default ours_t)
#   norefine             output the mask-weighted warped blend, bypassing the refinement U-Net residual
#   rgb                  feed RGB instead of upstream's BGR order (lossless either way)
#   tf32                 allow TF32 kernels (faster; default is strict float32; differs by mean ~6, max ~400 codes)
#   bench                cuDNN autotuning (+~140 s warm-up, no real gain at 4K fp32)
#   ho2                  hold-out: every 2nd input frame, x2, scores the dropped real frames (SF ignored)
#   log                  label only (no effect)
#   rec709               read input/${SHOT}_rec709
# Optional env: SF (default 10), FPS (default 25), ADAPTER_ARGS="k=v k=v" (appended, override tokens).
# Time (smoke job 319369, A100 80GB, 3840x2160 16-bit): default config 11.2 s/pair of model time,
#   ~12.2 s/pair with TIFF IO, model load ~3 s -> 41 frames (40 pairs) ~8.5 min; ds05 11.7 s/pair.
#   GPU peak 21 GiB (ds05 at 4 frames: 34.6 GiB, job 319357). --time 00:30:00 = ~3.5x margin.
#   Hold-out (ho2) is much shorter. EXCEPTIONS needing a longer --time on the command line:
#   ds1 (down_scale=1.0) measured 154 s/pair -> 40 pairs ~1 h 43 min, use --time 04:00:00;
#   bench (cudnn_benchmark) adds a ~140 s one-off warm-up on the first pair.
#SBATCH --job-name=vfi_run_emavfi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/emavfi
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
mvfi_require_gpu
[ "$(command -v python)" = "$MVFI_ENV/bin/python" ] || { echo "ERROR: env $MVFI_ENV not active (python=$(command -v python))" >&2; exit 1; }

SHOT=${SHOT:?set SHOT}
VARIANT=${VARIANT:-ds025}
SF=${SF:-10}
FPS=${FPS:-25}
ADAPTER=$MVFI_ROOT/scripts/adapters/emavfi.py
HARNESS=$MVFI_ROOT/scripts/vfi_harness.py

IN=$MVFI_ROOT/input/$SHOT
toks=${VARIANT}
if [[ $VARIANT == *_rec709 || $VARIANT == rec709 ]]; then
  IN=$MVFI_ROOT/input/${SHOT}_rec709
  toks=${VARIANT%rec709}
fi
args=() hargs=(--temporal_sf "$SF")
for t in ${toks//_/ }; do
  case $t in
    ds025)    args+=(down_scale=0.25) ;;
    ds05)     args+=(down_scale=0.5) ;;
    ds1)      args+=(down_scale=1.0) ;;
    notta)    args+=(tta=false) ;;
    fasttta)  args+=(fast_tta=true) ;;
    small)    args+=(model=ours_small_t) ;;
    norefine) args+=(refine=false) ;;
    rgb)      args+=(channel_order=rgb) ;;
    tf32)     args+=(tf32=true) ;;
    bench)    args+=(cudnn_benchmark=true) ;;
    ho2)      hargs=(--temporal_sf 2 --input_stride 2) ;;
    log)      ;;
    *) echo "ERROR: unknown VARIANT token '$t' in '$VARIANT'" >&2; exit 1 ;;
  esac
done
for kv in ${ADAPTER_ARGS:-}; do args+=("$kv"); done
aargs=()
for kv in "${args[@]}"; do aargs+=(--adapter_arg "$kv"); done

if [ -z "$(find "$IN" -maxdepth 1 \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' \) -print -quit 2>/dev/null)" ]; then
  echo "ERROR: no input frames in $IN" >&2; exit 1
fi
OUT=$MVFI_ROOT/output/$SHOT/ema-vfi/$VARIANT
# run_report.json is written last by the harness: its presence marks a complete run.
if [ -e "$OUT/logs/run_report.json" ]; then echo "-- $OUT already complete, skipping"; exit 0; fi
if [ -e "$OUT" ]; then echo "-- removing incomplete $OUT"; rm -rf "${OUT:?}"; fi
mkdir -p "$OUT/logs"
mvfi_start_gpu_monitor "$OUT/logs/gpu_usage_${SLURM_JOB_ID}.csv"

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID shot=$SHOT variant=$VARIANT"
echo "   input=$IN ($(find "$IN" -maxdepth 1 \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' \) | wc -l) frames)"
echo "   harness args: ${hargs[*]} --fps $FPS | adapter args: ${args[*]:-(adapter defaults: ours_t ds0.25 tta fp32)}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,driver_version --format=csv,noheader
t0=$(date +%s)
python "$HARNESS" --adapter "$ADAPTER" --input "$IN" --out_dir "$OUT" "${hargs[@]}" --fps "$FPS" "${aargs[@]}"
echo "== harness finished in $(( $(date +%s) - t0 )) s"
python - "$OUT/logs/run_report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
st = r["model"]["output_stats"]
print("output_frames", r["output_frames"], "| sec/pair", round(r["timing_s"]["sec_per_pair_mean"], 2),
      "| gpu peak GiB", round(r["gpu"]["max_memory_allocated_gib"], 2), "| float stats", st,
      "| holdout PSNR", r["holdout_psnr_db_mean"])
if st["nonfinite_values"] or st["raw_min"] < -1e-5 or st["raw_max"] > 1 + 1e-5:
    sys.exit("WARNING: non-finite or out-of-range model output (see output_stats); outputs kept for inspection")
EOF
echo "peak nvidia-smi memory.used: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{print m" MiB"}' "$OUT/logs/gpu_usage_${SLURM_JOB_ID}.csv")"
echo "== done $(date -Is)"
