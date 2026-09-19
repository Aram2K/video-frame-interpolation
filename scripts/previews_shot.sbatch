#!/bin/bash
# CPU: previews / comparison material for every finished LDF-VFI run of a shot.
#   SHOT=test01 sbatch scripts/previews_shot.sbatch
#SBATCH --job-name=vfi_previews
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
SHOT=${SHOT:?set SHOT}
MODEL_DIR=${MODEL_DIR:-ldf-vfi}
OUT=$MVFI_ROOT/output/$SHOT/$MODEL_DIR
cd "$MVFI_ROOT/scripts"
for run in "$OUT"/*/; do
  run=${run%/}
  [ -e "$run/logs/run_report.json" ] || continue
  [ -e "$run/preview.mp4" ] && continue
  src=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['input'])" "$run/logs/run_report.json")
  case "$src" in
    *_rec709) views="as_encoded" ;;          # already display-referred
    *)        views="as_encoded rec709" ;;   # Gen5 log: also a technical 709 viewing version
  esac
  for view in $views; do
    echo "-- $(date -Is) $run ($view)"
    python make_previews.py "$run" --source_dir "$src" --fps 25 --view "$view"
  done
done
echo "== done $(date -Is)"
