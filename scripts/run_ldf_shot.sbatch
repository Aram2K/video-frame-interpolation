#!/bin/bash
# LDF-VFI on one shot (single GPU). Variants run in the order given; results are never overwritten.
#   SHOT=test01 VARIANTS="A B HA HB EVAL A32" sbatch scripts/run_ldf_shot.sbatch
#   A    Gen5-log TIFFs, x10                       -> output/$SHOT/ldf-vfi/A_log_sf10
#   B    technical Rec.709 TIFFs, x10              -> output/$SHOT/ldf-vfi/B_rec709_sf10
#   HA   hold-out: every 2nd log frame, x2 (scores the dropped real frames)
#   HB   hold-out: same on Rec.709
#   EVAL score HA/HB (and HA32 if present) in one common display space
#   A32  as A with the VAE in float32 instead of bfloat16 (HA32: its hold-out)
#SBATCH --job-name=vfi_ldf_shot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
mvfi_require_gpu
# 2026-09-17 run 319426 died with CUDA OOM at frame 80/401 on the A100: 43 GiB allocated but 75.8 GiB held,
# i.e. allocator fragmentation, when the concat window asked for one 13.8 GiB block. Expandable segments
# let the caching allocator grow/reuse segments instead of fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SHOT=${SHOT:?set SHOT}
VARIANTS=${VARIANTS:-"A B HA HB EVAL"}
MODEL=$MVFI_MODELS/LDF-VFI/transformer
VAE=$MVFI_MODELS/LDF-VFI/Wan2.1_VAE_cond_v2.pth
IN_LOG=$MVFI_ROOT/input/$SHOT
IN_709=$MVFI_ROOT/input/${SHOT}_rec709
OUT=$MVFI_ROOT/output/$SHOT/ldf-vfi
mkdir -p "$OUT"
mvfi_start_gpu_monitor "$OUT/gpu_usage_${SLURM_JOB_ID}.csv"
echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID shot=$SHOT variants=[$VARIANTS]"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cd "$MVFI_ROOT/scripts"

run() {  # name input sf extra-args...
  local name=$1 input=$2 sf=$3; shift 3
  # run_report.json is written last: its presence marks a complete run.
  if [ -e "$OUT/$name/logs/run_report.json" ]; then echo "-- $name already done, skipping"; return; fi
  if [ -e "$OUT/$name" ]; then echo "-- removing incomplete $OUT/$name"; rm -rf "${OUT:?}/$name"; fi
  echo "-- $(date -Is) start $name"
  local t0=$(date +%s)
  python ldf_tiff_infer.py --input "$input" --out_dir "$OUT/$name" --model_path "$MODEL" --vae_path "$VAE" \
    --temporal_sf "$sf" --fps 25 "$@"
  echo "-- $(date -Is) end $name ($(( $(date +%s) - t0 )) s)"
}

for v in $VARIANTS; do
  case $v in
    A)    run A_log_sf10 "$IN_LOG" 10 ;;
    B)    run B_rec709_sf10 "$IN_709" 10 ;;
    A32)  run A_log_sf10_vae32 "$IN_LOG" 10 --vae_dtype fp32 ;;
    HA)   run holdout/A_log_s2 "$IN_LOG" 2 --input_stride 2 ;;
    HB)   run holdout/B_rec709_s2 "$IN_709" 2 --input_stride 2 ;;
    HA32) run holdout/A_log_s2_vae32 "$IN_LOG" 2 --input_stride 2 --vae_dtype fp32 ;;
    EVAL)
      args=(--run "A_log_bf16:$OUT/holdout/A_log_s2:gen5" --run "B_rec709_bf16:$OUT/holdout/B_rec709_s2:rec709")
      [ -e "$OUT/holdout/A_log_s2_vae32/logs/run_report.json" ] && args+=(--run "A_log_fp32vae:$OUT/holdout/A_log_s2_vae32:gen5")
      python eval_holdout.py "${args[@]}" --out "$OUT/holdout/holdout_scores.json" ;;
    *) echo "unknown variant $v" >&2; exit 1 ;;
  esac
done
echo "== done $(date -Is)"
