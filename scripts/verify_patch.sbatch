#!/bin/bash
# GPU: upstream uint8 sampler vs our float sampler on the bundled demo (same frames, seed, 3 chunks).
#SBATCH --job-name=vfi_verify_patch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
mvfi_require_gpu
D=$MVFI_ROOT/output/demo/ldf-vfi
mkdir -p "$D/patch_check"
cd "$MVFI_ROOT/scripts"
python verify_patch.py --frames "$D/decoded_source" --model_path "$MVFI_MODELS/LDF-VFI/transformer" \
  --vae_path "$MVFI_MODELS/LDF-VFI/Wan2.1_VAE_cond_v2.pth" --temporal_sf 8 --chunks 3 \
  --out_json "$D/patch_check/patch_equivalence_v2.json"
