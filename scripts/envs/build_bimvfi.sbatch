#!/bin/bash
# Build the BiM-VFI inference env (CPU node), fetch + verify the official checkpoint, import smoke test, pip freeze.
#   sbatch scripts/envs/build_bimvfi.sbatch            # create env if missing, then weights + tests (idempotent)
#   FRESH=1 sbatch scripts/envs/build_bimvfi.sbatch    # delete env/bimvfi first and rebuild it
#   BIMVFI_ENV=$PWD/env/bimvfi_rebuildtest sbatch ...   # reproducibility test into another prefix (no provenance
#                                                         files written; the freeze is diffed against provenance/)
# Upstream README asks for python 3.11, torch 2.4.1 / torchvision 0.19.1 and conda-forge cupy, plus a long list of
# training/metric packages (wandb, pyiqa, lpips, ...) that inference does not need: the adapter imports only the
# network, so only torch, torchvision (utils/misc.py), packaging, cupy and numpy are installed, plus the harness IO.
# torch cu121 wheels: CUDA 12.1 <= driver 545 (CUDA 12.3). cupy-cuda12x uses the NVRTC 12.1 from the torch wheel
# (nvidia-cuda-nvrtc-cu12==12.1.105), which the adapter preloads; no CUDA toolkit is needed.
#SBATCH --job-name=vfi_env_bimvfi
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail

ROOT=$MVFI_ROOT
ENV=${BIMVFI_ENV:-$ROOT/env/bimvfi}
CONSTRAINTS=$ROOT/scripts/envs/constraints_bimvfi.txt   # every package pinned (pip freeze of the tested env)
REPO=$ROOT/src/BiM-VFI
MODELDIR=$ROOT/models/BiM-VFI
WEIGHTS=$MODELDIR/bim_vfi.pth
WEIGHTS_ID=18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC
WEIGHTS_URL="https://drive.usercontent.google.com/download?id=${WEIGHTS_ID}&export=download&confirm=t"
WEIGHTS_BYTES=82749822
# No upstream checksum exists: pinned from the first download (job 319151, 2026-09-17; size matched Drive metadata).
# WEIGHTS_SHA256= (empty) records whatever is downloaded instead of verifying it.
WEIGHTS_SHA256=${WEIGHTS_SHA256-510cbcb07a7f8fcb17b4da241bbee2a6e20a8eb9bb423cf4df4815b7cd6d1cc0}

export PIP_CACHE_DIR=$ROOT/cache/tmp/build_bimvfi_$SLURM_JOB_ID/pip
export CONDA_PKGS_DIRS=$ROOT/cache/tmp/build_bimvfi_$SLURM_JOB_ID/conda_pkgs
export TMPDIR=$ROOT/cache/tmp/build_bimvfi_$SLURM_JOB_ID/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TORCH_HOME=$ROOT/cache/torch XDG_CACHE_HOME=$ROOT/cache/xdg CUPY_CACHE_DIR=$ROOT/cache/cupy
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR" "$MODELDIR"
trap 'rm -rf "$ROOT/cache/tmp/build_bimvfi_$SLURM_JOB_ID"' EXIT

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
source ${MVFI_CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh

if [ "${FRESH:-0}" = 1 ] && [ -e "$ENV" ]; then
  echo "-- FRESH=1: removing $ENV"; rm -rf "$ENV"
fi
if [ ! -e "$ENV/.build_complete" ]; then
  [ -e "$ENV" ] && { echo "-- removing incomplete env $ENV"; rm -rf "$ENV"; }
  conda create -y -p "$ENV" --override-channels -c conda-forge python=3.11.9 pip=24.2
  conda activate "$ENV"
  pip install --index-url https://download.pytorch.org/whl/cu121 -c "$CONSTRAINTS" \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121
  pip install -c "$CONSTRAINTS" numpy==1.26.4 cupy-cuda12x==13.6.0 fastrlock==0.8.3 packaging==24.2 \
    tifffile==2024.8.30 pillow==11.1.0
  touch "$ENV/.build_complete"
else
  echo "-- env exists: $ENV"
  conda activate "$ENV"
fi
python -V

echo "== weights"
if [ ! -s "$WEIGHTS" ]; then
  curl -fL --retry 3 --max-time 900 -o "$WEIGHTS.part" "$WEIGHTS_URL"
  mv "$WEIGHTS.part" "$WEIGHTS"
fi
size=$(stat -c %s "$WEIGHTS")
sum=$(sha256sum "$WEIGHTS" | cut -d' ' -f1)
echo "bim_vfi.pth bytes=$size sha256=$sum"
if [ "$size" != "$WEIGHTS_BYTES" ] || head -c 64 "$WEIGHTS" | grep -qi "<html\|<!doctype"; then
  echo "ERROR: $WEIGHTS is not the 82,749,822-byte checkpoint (size $size); removing it" >&2
  rm -f "$WEIGHTS"; exit 1
fi
if [ -n "$WEIGHTS_SHA256" ] && [ "$sum" != "$WEIGHTS_SHA256" ]; then
  echo "ERROR: sha256 mismatch: got $sum, pinned $WEIGHTS_SHA256" >&2; exit 1
fi
cat > "$MODELDIR/SHA256SUMS" <<EOF
$sum  bim_vfi.pth
EOF
cat > "$MODELDIR/PROVENANCE.txt" <<EOF
BiM-VFI official checkpoint (README "Pretrained Model" link)
file:    bim_vfi.pth
bytes:   $size
sha256:  $sum
source:  https://drive.google.com/file/d/$WEIGHTS_ID/view?usp=sharing
fetched: $WEIGHTS_URL
code:    https://github.com/KAIST-VICLab/BiM-VFI @ $(cat "$REPO/.git/$(sed 's/^ref: //' "$REPO/.git/HEAD")" 2>/dev/null || cat "$REPO/.git/HEAD")
checked: $(date -Is) by Slurm job $SLURM_JOB_ID
licence: research/education only, see LICENSE_NOTICE.txt
EOF
{
  echo "BiM-VFI licence terms (verbatim from $REPO/README.md, section 'License'; the repo has no LICENSE file)."
  echo "They apply to the source code AND the checkpoint in this folder."
  echo
  awk '/^## License/{f=1;next} /^\[\/\/\]/{if(f)exit} f' "$REPO/README.md" | sed '/^$/d'
  echo
  echo "Upstream: https://github.com/KAIST-VICLab/BiM-VFI (README.md). Training data: Vimeo90K (its own research-use terms)."
} > "$MODELDIR/LICENSE_NOTICE.txt"
cat "$MODELDIR/LICENSE_NOTICE.txt"

echo "== import smoke test (CPU node, no GPU)"
cd "$TMPDIR"
echo "-- NVRTC visible to cupy without preload?"
python -c "import torch, cupy; from cupy.cuda import nvrtc; print('nvrtc', nvrtc.getVersion())" 2>&1 | tail -2 || true
python - <<'EOF'
import os
import sys, importlib.util
import torch, torchvision, numpy, tifffile, PIL, packaging
spec = importlib.util.spec_from_file_location(
    "bimvfi_adapter", os.environ["MVFI_ROOT"] + "/scripts/adapters/bimvfi.py")
ad = importlib.util.module_from_spec(spec); spec.loader.exec_module(ad)
print("preloaded:", ad.preload_nvrtc())
import cupy
from cupy.cuda import nvrtc
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version(), "| torchvision", torchvision.__version__,
      "| numpy", numpy.__version__, "| cupy", cupy.__version__, "nvrtc", nvrtc.getVersion(),
      "| tifffile", tifffile.__version__, "| pillow", PIL.__version__)
ckpt, mode = ad.load_checkpoint(ad.WEIGHTS)
print("checkpoint load mode:", mode)
print("checkpoint keys:", sorted(ckpt.keys()), "| epoch", ckpt.get("epoch"), "iteration", ckpt.get("iteration"),
      "| run_id", ckpt.get("run_id"), "| lr_scheduler.anneal_func", (ckpt.get("lr_scheduler") or {}).get("anneal_func"))
model, meta = ad.load_model(ad.WEIGHTS, pyr_level=7, device="cpu")
print("model loaded strict=True:", meta)
heavy = [m for m in ("wandb", "pyiqa", "lpips", "cv2", "moviepy", "tensorboardX") if m in sys.modules]
print("training-stack modules imported:", heavy or "none")
assert not heavy
for h, w in [(2160, 3840), (2160, 4096), (1080, 1920)]:
    for p in (7, 8):
        l, r, t, b = ad.pad_amounts(h, w, 2 ** (p + 1), 64)
        print(f"pad {w}x{h} pyr{p}: -> {w+l+r}x{h+t+b} (l{l} r{r} t{t} b{b})")
EOF

pip check || true
if [ "$ENV" = "$ROOT/env/bimvfi" ]; then
  pip freeze > "$ROOT/provenance/pip_freeze_bimvfi.txt"
  conda list -p "$ENV" --explicit > "$ROOT/scripts/envs/conda_explicit_bimvfi.txt"
else
  pip freeze > "$TMPDIR/freeze.txt"
  echo "-- freeze of $ENV vs provenance/pip_freeze_bimvfi.txt:"
  diff "$ROOT/provenance/pip_freeze_bimvfi.txt" "$TMPDIR/freeze.txt" && echo "   identical"
fi
du -sh "$ENV"
echo "== done $(date -Is)"
