#!/bin/bash
# GIMM-VFI smoke test on the A100: harness on input/synthetic4k (4 x 3840x2160 16-bit TIFF, x10 -> 31 frames),
# then scripts/envs/check_gimmvfi_smoke.py (frame count, bit-exact genuine frames, no NaN, float range,
# t=0.1 nearer img0 / t=0.9 nearer img1, no 8-bit quantisation, NaN -> clear error).
# Then short 2-frame runs of the other variants / ds 0.5 (EXTRAS; failures there are reported, not fatal).
#   sbatch scripts/envs/smoke_gimmvfi.sbatch
#   GIMM_MODEL=R DS=0.25 EXTRAS="" sbatch scripts/envs/smoke_gimmvfi.sbatch
#SBATCH --job-name=vfi_smoke_gimmvfi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/gimmvfi
set +u; source "$MVFI_ROOT/scripts/env.sh"; set -u
trap mvfi_cleanup EXIT
mvfi_require_gpu
GIMM_MODEL=${GIMM_MODEL:-R-P}
DS=${DS:-0.25}
EXTRAS=${EXTRAS-"R:0.25 F-P:0.25 F:0.25 R-P:0.5"}
IN=$MVFI_ROOT/input/synthetic4k
OUT=$MVFI_ROOT/output/_checks/gimmvfi_smoke_$SLURM_JOB_ID
HARNESS=$MVFI_ROOT/scripts/vfi_harness.py
ADAPTER=$MVFI_ROOT/scripts/adapters/gimmvfi.py
CHECK=$MVFI_ROOT/scripts/envs/check_gimmvfi_smoke.py
SMI=$MVFI_ROOT/logs/gpu_usage_gimmvfi_smoke_${SLURM_JOB_ID}.csv
peak_smi() { awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{print m" MiB"}' "$SMI"; }

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID model=$GIMM_MODEL ds=$DS extras=[$EXTRAS]"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
mvfi_start_gpu_monitor "$SMI"
T0=$(date +%s)
python "$HARNESS" --adapter "$ADAPTER" --input "$IN" --out_dir "$OUT" --temporal_sf 10 --fps 25 \
  --adapter_arg model_variant="$GIMM_MODEL" --adapter_arg ds_factor="$DS"
echo "-- harness wall time $(( $(date +%s) - T0 )) s; nvidia-smi peak so far $(peak_smi)"
python "$CHECK" --out "$OUT" --input "$IN" --sf 10 --direct --model_variant "$GIMM_MODEL" --ds_factor "$DS"
echo "SMOKE MAIN PASS (GIMM-VFI-$GIMM_MODEL ds=$DS) -> $OUT; nvidia-smi peak $(peak_smi)"

for e in $EXTRAS; do
  if [ $(( $(date +%s) - T0 )) -gt 1200 ]; then echo "-- skipping remaining extras (time budget)"; break; fi
  v=${e%%:*}; d=${e##*:}; o=${OUT}_${v}_ds${d}
  echo "-- $(date -Is) extra: $v ds=$d (2 input frames)"
  if python "$HARNESS" --adapter "$ADAPTER" --input "$IN" --out_dir "$o" --temporal_sf 10 --fps 25 \
       --max_input_frames 2 --adapter_arg model_variant="$v" --adapter_arg ds_factor="$d" \
     && python "$CHECK" --out "$o" --input "$IN" --sf 10 --max_input_frames 2; then
    echo "EXTRA PASS $v ds=$d"
  else
    echo "EXTRA FAIL $v ds=$d"
  fi
done
echo "nvidia-smi peak over the job: $(peak_smi)"
echo "== done $(date -Is) ($(( $(date +%s) - T0 )) s)"
