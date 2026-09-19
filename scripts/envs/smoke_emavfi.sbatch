#!/bin/bash
# EMA-VFI 4K smoke test through the shared harness (A100 only).
#   sbatch scripts/envs/smoke_emavfi.sbatch
#   EXTRAS="" sbatch ...     main run only
# Main: adapter defaults (ours_t, down_scale 0.25, TTA, strict fp32, no cuDNN autotune) on input/synthetic4k (4 frames) x10 -> 31 frames in
#       output/_checks/emavfi_smoke_$SLURM_JOB_ID, then check_emavfi_output.py (count, NaN, range, t ordering,
#       bit-exact genuine frames).
# Extras (2 input frames -> 11 frames each, output/_checks/emavfi_smoke_${SLURM_JOB_ID}_<name>), same checks,
# compared with the main run: ds05 (down_scale 0.5), notta, small (ours_small_t), norefine, tf32.
# Timing extras (not default; 4 input frames so the steady state after cuDNN warm-up is visible):
#   EXTRAS="ds05x4 bench ds1" sbatch ...   down_scale 0.5 | cudnn_benchmark=true | down_scale 1.0 (2 frames)
#SBATCH --job-name=vfi_smoke_emavfi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/emavfi
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
mvfi_require_gpu
[ "$(command -v python)" = "$MVFI_ENV/bin/python" ] || { echo "ERROR: env $MVFI_ENV not active (python=$(command -v python))" >&2; exit 1; }

EXTRAS=${EXTRAS-"ds05 notta small norefine tf32"}
IN=$MVFI_ROOT/input/synthetic4k
ADAPTER=$MVFI_ROOT/scripts/adapters/emavfi.py
HARNESS=$MVFI_ROOT/scripts/vfi_harness.py
CHECK=$MVFI_ROOT/scripts/envs/check_emavfi_output.py
OUT=$MVFI_ROOT/output/_checks/emavfi_smoke_$SLURM_JOB_ID
T0=$(date +%s)

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), torch.cuda.get_device_name(0))"
mkdir -p "$OUT/logs"
mvfi_start_gpu_monitor "$OUT/logs/gpu_usage.csv"

echo "== main run: 4 frames x10"
python "$HARNESS" --adapter "$ADAPTER" --input "$IN" --out_dir "$OUT" --temporal_sf 10 --fps 25
echo "== main run finished after $(( $(date +%s) - T0 )) s"
python "$CHECK" "$OUT" "$IN" 10 4
rc=0

extra() {  # name n_input_frames adapter-args...
  local name=$1 n=$2; shift 2
  local o=${OUT}_$name args=()
  for kv in "$@"; do args+=(--adapter_arg "$kv"); done
  if [ $(( $(date +%s) - T0 )) -gt 1260 ]; then echo "-- skip $name (time budget)"; return 0; fi
  echo "== extra $name: $* ($n frames x10)"
  python "$HARNESS" --adapter "$ADAPTER" --input "$IN" --out_dir "$o" --temporal_sf 10 --fps 25 \
    --max_input_frames "$n" "${args[@]}" && python "$CHECK" "$o" "$IN" 10 "$n" --ref "$OUT"
}
for e in $EXTRAS; do
  case $e in
    ds05)     extra ds05 2 down_scale=0.5 || rc=1 ;;
    notta)    extra notta 2 tta=false || rc=1 ;;
    small)    extra small 2 model=ours_small_t || rc=1 ;;
    norefine) extra norefine 2 refine=false || rc=1 ;;
    tf32)     extra tf32 2 tf32=true || rc=1 ;;
    ds05x4)   extra ds05x4 4 down_scale=0.5 || rc=1 ;;
    bench)    extra bench 4 cudnn_benchmark=true || rc=1 ;;
    ds1)      extra ds1 2 down_scale=1.0 || rc=1 ;;
    *) echo "unknown extra $e" >&2; rc=1 ;;
  esac
done

echo "peak nvidia-smi memory.used: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{print m" MiB"}' "$OUT/logs/gpu_usage.csv")"
echo "== done $(date -Is) total $(( $(date +%s) - T0 )) s, extras rc=$rc"
exit $rc
