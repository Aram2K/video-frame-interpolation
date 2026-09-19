#!/bin/bash
# VTinker (fp32) on one real shot through the shared 16-bit harness (A100 only).
#   SHOT=test01 VARIANT=A_log sbatch scripts/envs/run_vtinker.sbatch       # input/test01          -> output/test01/vtinker/A_log
#   SHOT=test01 VARIANT=B_rec709 sbatch scripts/envs/run_vtinker.sbatch    # input/test01_rec709   -> output/test01/vtinker/B_rec709
# VARIANT ending in _rec709 selects input/${SHOT}_rec709, anything else input/$SHOT.
# FIRST, on any new shot, measure quality at t != 0.5 against real frames (the model was trained at t=0.5 only):
#   SHOT=test01 VARIANT=holdout SF=10 STRIDE=10 sbatch scripts/envs/run_vtinker.sbatch
#   -> output/test01/vtinker/holdout/logs/holdout_metrics.csv gives PSNR per output index, i.e. per t.
#   A U-shape with its minimum at t=0.5 is fine; a collapse at t=0.1/0.9 means restrict VTinker to SF=2 or drop it.
# Optional env: SF (default 10), STRIDE (default 1; hold-out mode needs SF == STRIDE, e.g. SF=2 STRIDE=2),
#   FPS (default 25), MAXF (limit input frames), ADAPTER_ARGS="k=v k=v" (see scripts/adapters/vtinker.py).
# A finished run is never redone (logs/run_report.json is written last); an incomplete output dir is removed.
# --time: passing smoke job 319365 (A100, 3840x2160, fp32, TF32 off) took 22.6 s per pair for the 9 frames
#   of x10 (23.6 s per pair incl. TIFF writes, peak 25.9 GiB allocated / 32 GiB nvidia-smi), model load 1 s.
#   41 input frames = 40 pairs -> ~16 min; x3 margin = ~48 min -> 01:00:00 (also covers 4096-wide DCI 4K).
#SBATCH --job-name=vfi_run_vtinker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/vtinker
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
mvfi_require_gpu
export PYTHONDONTWRITEBYTECODE=1   # keep src/VTinker free of __pycache__

SHOT=${SHOT:?set SHOT}
VARIANT=${VARIANT:?set VARIANT}
SF=${SF:-10}
STRIDE=${STRIDE:-1}
FPS=${FPS:-25}
case "$VARIANT" in
  *_rec709) IN=$MVFI_ROOT/input/${SHOT}_rec709 ;;
  *)        IN=$MVFI_ROOT/input/$SHOT ;;
esac
OUT=$MVFI_ROOT/output/$SHOT/vtinker/$VARIANT
ADAPTER=$MVFI_ROOT/scripts/adapters/vtinker.py

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID shot=$SHOT variant=$VARIANT sf=$SF stride=$STRIDE"
echo "   input=$IN"
echo "   output=$OUT"
if [ ! -d "$IN" ] || [ -z "$(ls "$IN" | grep -iE '\.(tif|tiff|png)$' | head -1)" ]; then
  echo "ERROR: no TIFF/PNG frames in $IN" >&2; exit 1
fi
if [ -e "$OUT/logs/run_report.json" ]; then
  echo "-- $OUT already complete (logs/run_report.json exists); skipping"; exit 0
fi
if [ -e "$OUT" ]; then
  echo "-- removing incomplete $OUT"; rm -rf "${OUT:?}"
fi
mkdir -p "$(dirname "$OUT")"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_CSV=$MVFI_ROOT/logs/gpu_usage_vtinker_${SHOT}_${VARIANT}_${SLURM_JOB_ID}.csv
mvfi_start_gpu_monitor "$GPU_CSV"

args=(--adapter "$ADAPTER" --input "$IN" --out_dir "$OUT" --temporal_sf "$SF" --fps "$FPS" --input_stride "$STRIDE")
[ -n "${MAXF:-}" ] && args+=(--max_input_frames "$MAXF")
for kv in ${ADAPTER_ARGS:-}; do args+=(--adapter_arg "$kv"); done

t0=$(date +%s)
python "$MVFI_ROOT/scripts/vfi_harness.py" "${args[@]}"
echo "-- harness wall time $(( $(date +%s) - t0 )) s"
echo "-- peak nvidia-smi memory.used: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{print m+0}' "$GPU_CSV") MiB"
cp "$GPU_CSV" "$OUT/logs/" 2>/dev/null || true
echo "== done $(date -Is)"
