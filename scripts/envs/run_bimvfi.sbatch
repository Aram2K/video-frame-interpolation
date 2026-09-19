#!/bin/bash
# BiM-VFI on one real shot through the shared 16-bit harness (A100). Never overwrites a finished run.
#   SHOT=test01 VARIANT=pyr7_sf10 sbatch scripts/envs/run_bimvfi.sbatch
#   SHOT=test01 VARIANT=pyr7_sf10_rec709 sbatch scripts/envs/run_bimvfi.sbatch
# Input:  input/$SHOT, or input/${SHOT}_rec709 when VARIANT ends in _rec709.
# Output: output/$SHOT/bim-vfi/$VARIANT   (skipped if logs/run_report.json exists; an incomplete dir is removed first)
# VARIANT tokens (anywhere in the name):
#   pyr8      pyr_level=8 instead of 7 (larger motion search; ~1.3x cost at 4K)
#   holdout   hold-out mode: every 2nd input frame, x2 (scores the dropped real frames); default is x10
#   tf32      allow TF32 convolutions (faster, not bit-exact fp32)
#   reflectpad  OPT-IN: centred reflect padding with a 64 px margin instead of the upstream default.
#             The default is upstream zero padding (bottom/right, via InputPadder). Reflect shifts the
#             model's global mean/std normalisation and so changes the whole frame, not just the border
#             (interior delta 0.00978 vs pyr8-vs-pyr7 0.00891); it has NOT been shown to beat upstream.
#             On the first real shot run pyr7_sf10 AND pyr7_sf10_reflectpad and compare on that footage.
#   constpad  accepted for compatibility; it is now the default, so the token is a no-op.
# Extra adapter args: ADAPTER_ARGS="margin=128 ..." (appended last, so they win).
# DRY_RUN=1 bash scripts/envs/run_bimvfi.sbatch   prints input/output/arguments and exits (no GPU touched).
# --time: smoke job 319361 (A100, 3840x2160 16-bit, pyr7): 13.65 s/pair incl. async TIFF writes, 1.5 s model load,
#   no warm-up; pyr8 1.73 s/t (~16 s/pair); 4096-wide DCI ~+6% (pyr7) / ~+25% (pyr8). 41 frames = 40 pairs:
#   ~9-13 min generation + ~4 min post-checks (reads ~400 TIFFs) -> ~13-17 min; x3 margin -> 01:00:00.
#SBATCH --job-name=vfi_run_bimvfi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/bimvfi
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
[ "${DRY_RUN:-0}" = 1 ] || mvfi_require_gpu
[ "$(command -v python)" = "$MVFI_ENV/bin/python" ] || { echo "ERROR: env $MVFI_ENV not active ($(command -v python))" >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1  # keep src/BiM-VFI free of __pycache__
unset SLURM_PROCID  # upstream utils/experiment.py would treat the job as a distributed launch

SHOT=${SHOT:?set SHOT}
VARIANT=${VARIANT:?set VARIANT, e.g. pyr7_sf10 or pyr7_sf10_rec709}
case $VARIANT in /*|*..*) echo "bad VARIANT $VARIANT" >&2; exit 1 ;; esac
if [[ $VARIANT == *_rec709 ]]; then IN=$MVFI_ROOT/input/${SHOT}_rec709; else IN=$MVFI_ROOT/input/$SHOT; fi
BASE=$MVFI_ROOT/output/$SHOT/bim-vfi
OUT=$BASE/$VARIANT
ADAPTER=$MVFI_ROOT/scripts/adapters/bimvfi.py

aargs=(--adapter_arg pyr_level=$([[ $VARIANT == *pyr8* ]] && echo 8 || echo 7))
[[ $VARIANT == *tf32* ]] && aargs+=(--adapter_arg tf32=true)
if [[ $VARIANT == *reflectpad* ]]; then aargs+=(--adapter_arg pad_mode=reflect --adapter_arg margin=64)
else aargs+=(--adapter_arg pad_mode=constant); fi  # constant = upstream, the default; constpad is a no-op
for kv in ${ADAPTER_ARGS:-}; do aargs+=(--adapter_arg "$kv"); done
if [[ $VARIANT == *holdout* ]]; then SF=2; hargs=(--temporal_sf 2 --input_stride 2); else SF=10; hargs=(--temporal_sf 10); fi

echo "== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-manual} shot=$SHOT variant=$VARIANT"
echo "   input=$IN  output=$OUT  adapter: ${aargs[*]}  harness: ${hargs[*]}"
[ -d "$IN" ] || { echo "ERROR: input folder $IN does not exist" >&2; exit 1; }
N=$(find "$IN" -maxdepth 1 \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' \) | wc -l)
[ "$N" -ge 2 ] || { echo "ERROR: $IN has $N frames" >&2; exit 1; }
if [[ $VARIANT == *holdout* ]]; then NU=$(( (N + 1) / 2 )); else NU=$N; fi
echo "   $N input frames, $NU used -> $(( (NU - 1) * SF + 1 )) output frames (x$SF)"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY_RUN: would run vfi_harness.py; done=$([ -e "$OUT/logs/run_report.json" ] && echo yes || echo no)"; exit 0; fi
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader

# run_report.json is written last by the harness: its presence marks a complete run.
if [ -e "$OUT/logs/run_report.json" ]; then echo "-- $OUT already complete, skipping"; exit 0; fi
if [ -e "$OUT" ]; then echo "-- removing incomplete $OUT"; rm -rf "${OUT:?}"; fi
mkdir -p "$BASE"
mvfi_start_gpu_monitor "$BASE/gpu_usage_${VARIANT//\//_}_${SLURM_JOB_ID}.csv"

cd "$MVFI_ROOT/scripts"
t0=$(date +%s)
env -u SLURM_PROCID python vfi_harness.py --adapter "$ADAPTER" --input "$IN" --out_dir "$OUT" \
  --fps 25 "${hargs[@]}" "${aargs[@]}"
echo "-- harness wall time $(( $(date +%s) - t0 )) s"

# Sanity checks (NaN/range/genuine bit-exactness/temporal order); informative for real footage, never deletes output.
if [[ $VARIANT != *holdout* ]]; then
  EXPECT=$(( (N - 1) * SF + 1 ))
  python "$MVFI_ROOT/scripts/envs/check_bimvfi_smoke.py" "$OUT" "$IN" --sf $SF --expect $EXPECT \
    > "$OUT/logs/post_checks.txt" 2>&1 && echo "-- post-checks PASS" || echo "WARNING: post-checks flagged issues, see $OUT/logs/post_checks.txt"
  tail -3 "$OUT/logs/post_checks.txt"
fi
echo "== done $(date -Is)"
