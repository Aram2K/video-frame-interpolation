#!/bin/bash
# Build the isolated EMA-VFI inference env + fetch the official arbitrary-timestep checkpoints (CPU node).
#   sbatch scripts/envs/build_emavfi.sbatch
# Upstream (MCG-NJU/EMA-VFI @75b6f6a) pins torch 1.8.0 / python 3.8 / timm 0.6.11. The network is pure PyTorch
# (no custom CUDA kernels), so it runs on a modern stack. Stack used here:
#   python 3.10.14 (conda-forge) | torch 2.5.1+cu121, torchvision 0.20.1+cu121 (inside driver 545 / CUDA 12.3)
#   timm 0.6.13 (same 0.6 series as upstream's 0.6.11; only DropPath/to_2tuple/trunc_normal_ are used)
#   numpy 1.26.4, pillow 11.1.0, tifffile 2024.8.30 (harness IO). No opencv: the adapter does not need it.
# Weights: official Google Drive files (README folder 16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o); sha256 + size verified.
# Fallback source (byte-identical, verified by the same sha256): huggingface.co/xmanifold/emavfi.
# Re-running is safe: an existing env is kept (set REBUILD=1 to delete and rebuild it), verified weights are kept.
#SBATCH --job-name=vfi_env_emavfi
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail

ROOT=$MVFI_ROOT
ENV=$ROOT/env/emavfi
REPO=$ROOT/src/EMA-VFI
WDIR=$ROOT/models/EMA-VFI
HERE=$ROOT/scripts/envs

# Same per-job temp dir as env.sh (mvfi_cleanup removes it). Package caches live inside it (never $HOME, never /tmp).
export TMPDIR=$ROOT/cache/tmp/job_${SLURM_JOB_ID}
export PIP_CACHE_DIR=$TMPDIR/pip
export CONDA_PKGS_DIRS=$TMPDIR/conda_pkgs
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$WDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
source ${MVFI_CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh

# ---------------------------------------------------------------- weights (independent of the env)
fetch() {  # name drive_id sha256 bytes
  local name=$1 id=$2 sum=$3 size=$4 dst=$WDIR/$1 part=$WDIR/$1.part src
  if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ] && echo "$sum  $dst" | sha256sum -c --status; then
    echo "-- $name present and verified"; return 0
  fi
  for src in "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t" \
             "https://huggingface.co/xmanifold/emavfi/resolve/main/${name}"; do
    echo "-- fetching $name from $src"
    rm -f "$part"
    if curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 -sS -o "$part" "$src" \
       && [ "$(stat -c %s "$part")" = "$size" ] && echo "$sum  $part" | sha256sum -c --status; then
      mv -f "$part" "$dst"
      echo "$src" > "$dst.source_url"
      echo "-- $name OK ($size bytes, sha256 $sum)"; return 0
    fi
    echo "   failed or checksum/size mismatch (got $(stat -c %s "$part" 2>/dev/null || echo 0) bytes)"
  done
  rm -f "$part"
  echo "ERROR: could not fetch a verified $name" >&2; return 1
}
fetch ours_t.pkl       1mL4ht5DYFMA1CYiGDFbej-ufh5QoqfwN c09e82d193e303d7b91f1e36bf535886b820cb56105574355211d60a6b557a99 263760209
fetch ours_small_t.pkl 1mVC3gZVHibbwvDsw2Z1yuBkMjmDCZyG6 ad845671f7250d408937718aca19eaf87aa1ca79215070f7641eea332b703233 58535773
( cd "$WDIR" && sha256sum ours_t.pkl ours_small_t.pkl > SHA256SUMS && ls -l ours_t.pkl ours_small_t.pkl && cat SHA256SUMS )

# ---------------------------------------------------------------- env
if [ -e "$ENV" ] && [ "${REBUILD:-0}" = 1 ]; then echo "-- REBUILD=1: removing $ENV"; rm -rf "$ENV"; fi
if [ ! -e "$ENV/bin/python" ]; then
  [ -e "$ENV" ] && { echo "-- removing incomplete $ENV"; rm -rf "$ENV"; }
  conda create -y -p "$ENV" --override-channels -c conda-forge python=3.10.14 pip=24.2
fi
conda activate "$ENV"
python -V

# PyTorch from the cu121 index first, then everything else from PyPI with torch/torchvision held fixed.
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121
printf 'torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\n' > "$TMPDIR/constraints.txt"
pip install -c "$TMPDIR/constraints.txt" \
  numpy==1.26.4 pillow==11.1.0 tifffile==2024.8.30 \
  timm==0.6.13 huggingface_hub==0.25.2 pyyaml==6.0.2
pip check

echo "== import smoke test (CPU node: builds the net on CPU, strict weight load, tiny backbone forward)"
cd "$REPO"
python - <<'EOF'
import os
import sys, hashlib, torch, torchvision, timm, numpy, PIL, tifffile
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda build", torch.version.cuda,
      "| torchvision", torchvision.__version__, "| timm", timm.__version__, "| numpy", numpy.__version__,
      "| pillow", PIL.__version__, "| tifffile", tifffile.__version__)
sys.path.insert(0, ".")
import config as cfg
import Trainer  # upstream import graph (model.*, loss, warplayer)
from benchmark.utils.padder import InputPadder
W = os.environ["MVFI_ROOT"] + "/models/EMA-VFI"
for name, F_, depth in (("ours_t", 32, [2, 2, 2, 4, 4]), ("ours_small_t", 16, [2, 2, 2, 2, 2])):
    bcfg, mcfg = cfg.init_model_config(F=F_, depth=depth)
    bt, mt = cfg.MODEL_CONFIG["MODEL_TYPE"]
    net = mt(bt(**bcfg), **mcfg).eval()
    sd = torch.load(f"{W}/{name}.pkl", map_location="cpu", weights_only=True)
    dtypes = {str(v.dtype) for v in sd.values()}
    conv = {k.replace("module.", ""): v for k, v in sd.items()
            if "module." in k and "attn_mask" not in k and "HW" not in k}
    net.load_state_dict(conv, strict=True)
    n = sum(p.numel() for p in net.parameters())
    with torch.no_grad():
        x = torch.rand(1, 3, 128, 128)
        af, mf = net.feature_bone(x, x.flip(3))
    print(f"{name}: {len(sd)} keys ({len(sd) - len(conv)} dropped), dtypes {dtypes}, {n/1e6:.2f}M params, "
          f"feature shapes {[tuple(a.shape) for a in af]}, finite={all(torch.isfinite(a).all() for a in af)}")
# the adapter module must import cleanly (no GPU needed at import time)
import importlib.util
spec = importlib.util.spec_from_file_location("a", os.environ["MVFI_ROOT"] + "/scripts/adapters/emavfi.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("adapter module imports OK:", m.Adapter.__name__)
EOF

pip freeze > "$ROOT/provenance/pip_freeze_emavfi.txt"
conda list -p "$ENV" --explicit > "$HERE/conda_explicit_emavfi.txt"
du -sh "$ENV"
echo "== done $(date -Is)"
