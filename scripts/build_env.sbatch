#!/bin/bash
# Build the isolated LDF-VFI inference environment (CPU node; no GPU needed).
# Reference env from upstream: Python 3.11 / torch 2.5.1+cu124 (see src/LDF-VFI/env.yml, requirements.txt).
# Only the packages imported at inference are installed, at the exact versions upstream pins.
#   sbatch scripts/build_env.sbatch            # default TORCH_CUDA=cu124 (upstream reference)
#   TORCH_CUDA=cu121 sbatch scripts/build_env.sbatch   # fallback if cu124 fails on driver 545 / CUDA 12.3
#SBATCH --job-name=vfi_build_env
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail

ROOT=$MVFI_ROOT
ENV=$ROOT/env/ldfvfi
TORCH_CUDA=${TORCH_CUDA:-cu124}

# Keep caches and temp files inside the project directory (home quotas and login-node /tmp are often small on HPC clusters).
export PIP_CACHE_DIR=$ROOT/cache/pip
export CONDA_PKGS_DIRS=$ROOT/cache/conda_pkgs
export TMPDIR=$ROOT/cache/tmp/build_$SLURM_JOB_ID
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID TORCH_CUDA=$TORCH_CUDA"
source ${MVFI_CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh

if [ -e "$ENV" ]; then
  echo "ERROR: $ENV already exists; remove it deliberately before rebuilding." >&2
  exit 1
fi

# conda-forge only: python + the FFmpeg shared libraries torchcodec 0.1.0 needs (upstream env.yml pins ffmpeg=7.1.1).
conda create -y -p "$ENV" --override-channels -c conda-forge \
  python=3.11.9 ffmpeg=7.1.1 pip
conda activate "$ENV"
python -V
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "libx264|prores" || true

# PyTorch 2.5.1 (upstream pin). cu124 = upstream; cu121 = fully inside driver 545's CUDA 12.3.
pip install --index-url https://download.pytorch.org/whl/$TORCH_CUDA \
  torch==2.5.1 torchvision==0.20.1

# Inference imports of generate.py and training/*, pinned as in upstream requirements.txt / env.yml.
pip install \
  torchcodec==0.1.0 accelerate==1.5.2 diffusers==0.33.1 transformers==4.37.2 tokenizers==0.15.2 \
  datasets==3.5.0 huggingface_hub==0.34.4 hf-xet==1.1.10 safetensors==0.5.3 \
  einops==0.8.1 numpy==2.2.5 pillow==11.1.0 av==12.0.0 tqdm==4.67.1 \
  imageio==2.37.0 scipy==1.15.2 omegaconf==2.3.0 psutil==7.0.0 \
  lpips==0.1.4 piq==0.8.0 opencv-python-headless==4.11.0.86

# Pipeline-only additions (not used by upstream): 16-bit TIFF IO, BMD Gen5 / Rec.709 math.
pip install "tifffile>=2024.8.30,<2026" colour-science==0.4.6

echo "== import smoke test (CPU)"
cd $ROOT/src/LDF-VFI
python - <<'EOF'
import sys, torch, torchvision, diffusers, accelerate, transformers, datasets, numpy, PIL, av, tifffile, colour
print("python", sys.version)
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
from torchcodec.decoders import VideoDecoder
import generate  # upstream module: checks the whole import graph
print("upstream generate.py imports OK")
EOF

pip freeze > $ROOT/provenance/pip_freeze_ldfvfi.txt
conda list -p "$ENV" --explicit > $ROOT/provenance/conda_explicit_ldfvfi.txt
du -sh "$ENV"

# Free the package caches (envs keep their own hardlinked copies).
rm -rf "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
echo "== done $(date -Is)"
