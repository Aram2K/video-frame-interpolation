#!/bin/bash
# CPU prep for one shot: validate the TIFF sequence, then build the Test B (technical Rec.709) copy.
#   SHOT=test01 sbatch scripts/prep_shot.sbatch
#SBATCH --job-name=vfi_prep
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
SHOT=${SHOT:?set SHOT}
IN=$MVFI_ROOT/input/$SHOT
META=$MVFI_ROOT/output/$SHOT/input_check
mkdir -p "$META"
echo "== $(date -Is) host=$(hostname) shot=$SHOT"
chmod -R a-w "$IN"   # the genuine frames are never modified
cd "$MVFI_ROOT/scripts"
set +e
python inspect_input.py "$IN" "$META/input_metadata.json" --fps 25 --sf 10
rc=$?
set -e
(cd "$IN" && sha256sum *.tif *.tiff 2>/dev/null) > "$META/source_sha256.txt" || true
if [ $rc -ne 0 ]; then echo "INPUT PROBLEMS FOUND (see $META/input_metadata.txt); stopping." >&2; exit $rc; fi
if [ ! -d "$MVFI_ROOT/input/${SHOT}_rec709" ]; then
  python make_rec709.py "$IN" "$MVFI_ROOT/input/${SHOT}_rec709" --workers 16
  chmod -R a-w "$MVFI_ROOT/input/${SHOT}_rec709"
fi
echo "== done $(date -Is)"
