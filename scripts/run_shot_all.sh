#!/bin/bash
# Submit the whole per-shot pipeline as dependent Slurm jobs. Nothing heavy runs here.
#   scripts/run_shot_all.sh test01 [fast|ldf|all]
# Jobs are submitted from the project root (their #SBATCH --output=logs/... paths are relative to it);
# the Slurm account comes from SBATCH_ACCOUNT in config.local.sh.
#     fast  = prep + the four pairwise models (log + Rec.709) + comparison   (~1 h GPU)
#     ldf   = prep + LDF-VFI x10 on the log frames (~9 h GPU) + previews
#     all   = fast, then ldf, then the A/B hold-out scoring + comparison     (default)
set -euo pipefail
R=${MVFI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
SHOT=${1:?usage: run_shot_all.sh SHOT [fast|ldf|all]}
MODE=${2:-all}
if [ -f "$R/config.local.sh" ]; then source "$R/config.local.sh"; fi
export MVFI_ROOT=$R
cd "$R"
mkdir -p logs

sub() { echo "  $(sbatch --parsable "$@")"; }

PREP=$(SHOT=$SHOT sbatch --parsable scripts/prep_shot.sbatch)
echo "prep            $PREP"
dep="--dependency=afterok:$PREP"
last=$PREP

if [ "$MODE" = fast ] || [ "$MODE" = all ]; then
  # Pairwise models: quick, so both colour variants each. They run one after another on the single A100.
  for spec in "scripts/envs/run_emavfi.sbatch:ds025" "scripts/envs/run_emavfi.sbatch:ds025_rec709" \
              "scripts/envs/run_gimmvfi.sbatch:R-P_ds0.25" "scripts/envs/run_gimmvfi.sbatch:R-P_ds0.25_rec709" \
              "scripts/envs/run_bimvfi.sbatch:pyr7_sf10" "scripts/envs/run_bimvfi.sbatch:pyr7_sf10_rec709" \
              "scripts/envs/run_vtinker.sbatch:A_log" "scripts/envs/run_vtinker.sbatch:B_rec709"; do
    s=${spec%%:*}; v=${spec#*:}
    id=$(SHOT=$SHOT VARIANT=$v sbatch --parsable --dependency=afterok:$last "$s")
    echo "run $(basename $s .sbatch) $v -> $id"
    last=$id
  done
fi

if [ "$MODE" = ldf ] || [ "$MODE" = all ]; then
  id=$(SHOT=$SHOT VARIANTS="A" sbatch --parsable --dependency=afterok:$last scripts/run_ldf_shot.sbatch)
  echo "ldf A (x10, log)          $id"
  last=$id
fi

if [ "$MODE" = all ]; then
  id=$(SHOT=$SHOT VARIANTS="HA HB EVAL" sbatch --parsable --dependency=afterok:$last scripts/run_ldf_shot.sbatch)
  echo "ldf hold-out A/B + score  $id"
  last=$id
fi

PREV=$(SHOT=$SHOT sbatch --parsable --dependency=afterany:$last scripts/previews_shot.sbatch)
echo "previews (ldf)            $PREV"
CMP=$(sbatch --parsable --dependency=afterany:$PREV --job-name=vfi_compare -p standard \
      -c 8 --mem=32G -t 00:40:00 -o "$R/logs/%x_%j.log" \
      --wrap "source $R/scripts/env.sh; trap mvfi_cleanup EXIT; cd $R/scripts; python compare_models.py $SHOT --view as_encoded; python compare_models.py $SHOT --view rec709")
echo "cross-model comparison    $CMP"
echo
squeue -u "$USER" -o "%.8i %.24j %.9T %.9M %.20R"
