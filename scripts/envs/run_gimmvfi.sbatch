#!/bin/bash
# GIMM-VFI on one real shot through the shared harness (A100). Never overwrites a finished run.
#   SHOT=test01 VARIANT=R-P_ds0.25 sbatch scripts/envs/run_gimmvfi.sbatch
#   SHOT=test01 VARIANT=R-P_ds0.25_rec709 sbatch scripts/envs/run_gimmvfi.sbatch
# VARIANT = <model>_ds<factor>[_s2][_rec709]  (also the output folder name)
#   model     R-P (default: RAFT, perceptual/LPIPS checkpoint) | R | F-P | F   (F = FlowFormer flow)
#   ds        flow-estimation downscale: 0.25 (published 4K protocol) | 0.5
#   _s2       hold-out: every 2nd input frame, x2 (harness scores the dropped real frames)
#   _rec709   read input/${SHOT}_rec709 instead of input/$SHOT (log)
# input  $MVFI_ROOT/input/$SHOT or $MVFI_ROOT/input/${SHOT}_rec709
# output $MVFI_ROOT/output/$SHOT/gimm-vfi/$VARIANT  (run_report.json present = done -> skipped;
#        a folder without it is an interrupted run and is deleted first)
# Time (smoke job 319351, A100, 3840x2160): R-P ds0.25 3.3 s/pair steady (~6 s/pair incl. 10 TIFF writes),
#   R/F/F-P ds0.25 4.4-5.0 s (first pair), R-P ds0.5 12.4 s/pair (32.6 GiB peak). 41 inputs = 40 pairs:
#   ds0.25 ~4.5 min, ds0.5 ~10 min -> 00:45:00 = >3x margin for the slowest allowed setting.
#   ds 1.0 is not allowed here (untested at 4K; RAFT all-pairs correlation is estimated to exceed 80 GB).
#SBATCH --job-name=vfi_gimmvfi_shot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/gimmvfi
set +u; source "$MVFI_ROOT/scripts/env.sh"; set -u
trap mvfi_cleanup EXIT
mvfi_require_gpu
SHOT=${SHOT:?set SHOT}
VARIANT=${VARIANT:-R-P_ds0.25}
SF=${TEMPORAL_SF:-10}

v=$VARIANT
IN=$MVFI_ROOT/input/$SHOT
if [[ $v == *_rec709 ]]; then IN=$MVFI_ROOT/input/${SHOT}_rec709; v=${v%_rec709}; fi
STRIDE=1
if [[ $v == *_s2 ]]; then STRIDE=2; SF=2; v=${v%_s2}; fi
if [[ $v =~ ^(R-P|R|F-P|F)_ds(0\.25|0\.5)$ ]]; then
  GIMM_MODEL=${BASH_REMATCH[1]}; DS=${BASH_REMATCH[2]}
else
  echo "ERROR: VARIANT '$VARIANT' does not match <R-P|R|F-P|F>_ds<0.25|0.5>[_s2][_rec709]" >&2; exit 1
fi
OUT=$MVFI_ROOT/output/$SHOT/gimm-vfi/$VARIANT
NIN=$(find "$IN" -maxdepth 1 \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' \) 2>/dev/null | wc -l)
if [ "$NIN" -lt 2 ]; then echo "ERROR: $IN has $NIN frames" >&2; exit 1; fi

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID shot=$SHOT variant=$VARIANT"
echo "   model=GIMM-VFI-$GIMM_MODEL ds=$DS temporal_sf=$SF input_stride=$STRIDE input=$IN ($NIN frames) -> $OUT"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
if [ -e "$OUT/logs/run_report.json" ]; then echo "-- $OUT already done (run_report.json), skipping"; exit 0; fi
if [ -e "$OUT" ]; then echo "-- removing incomplete $OUT"; rm -rf "${OUT:?}"; fi
mkdir -p "$(dirname "$OUT")"
mvfi_start_gpu_monitor "$MVFI_ROOT/logs/gpu_usage_gimmvfi_${SHOT}_${VARIANT}_${SLURM_JOB_ID}.csv"
T0=$(date +%s)
python "$MVFI_ROOT/scripts/vfi_harness.py" --adapter "$MVFI_ROOT/scripts/adapters/gimmvfi.py" \
  --input "$IN" --out_dir "$OUT" --temporal_sf "$SF" --input_stride "$STRIDE" --fps 25 \
  --adapter_arg model_variant="$GIMM_MODEL" --adapter_arg ds_factor="$DS"
echo "== done $(date -Is) ($(( $(date +%s) - T0 )) s) -> $OUT"
