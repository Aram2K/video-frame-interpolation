#!/bin/bash
# Build the isolated VTinker inference environment + fetch the official weights (CPU node, no GPU).
#   sbatch scripts/envs/build_vtinker.sbatch             # fails if env/vtinker exists
#   REBUILD=1 sbatch scripts/envs/build_vtinker.sbatch   # deletes env/vtinker first (weights are kept if valid)
#
# Upstream (src/VTinker @ 6688e06, README) asks for Python 3.8.5 / torch 1.13.0+cu117 / cupy_cuda11x and a
# requirements.txt whose pins cannot be installed together (numpy 1.26.0 needs py>=3.9, scipy 1.5.2 only
# py<=3.8). Inference only imports torch + cupy (core/models/VTinker.py, softsplat, correlation), so we
# install just that, at versions that work with driver 545 (CUDA <= 12.3):
#   python 3.10.14 (conda-forge), torch 2.4.1+cu121, torchvision 0.19.1+cu121,
#   cupy-cuda12x 12.3.0 (last series that still has cupy.cuda.compile_with_cache, used by the kernels),
#   numpy 1.26.4 (<2 for cupy 12.3), tifffile + pillow (harness).
# CuPy 12.3 needs CUDA 12 runtime libraries (cudart, NVRTC + builtins, cuBLAS, ...). The torch cu121 wheels
# already pull them in as nvidia-*-cu12 pip packages (NVRTC 12.1.105 <= 12.3); an activate.d hook puts
# them on LD_LIBRARY_PATH and sets CUDA_PATH, so `conda activate env/vtinker` (scripts/env.sh) is enough.
#SBATCH --job-name=vfi_env_vtinker
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail

ROOT=$MVFI_ROOT
ENV=$ROOT/env/vtinker
REPO=$ROOT/src/VTinker
MODELS=$ROOT/models/VTinker
HF_REPO=wcy1234567/VTinker
HF_REV=f9f0f47e6fb5d35e73f38547e6a45521e41f0777   # HF commit (x-repo-commit), 2025-12-05
declare -A SHA=(
  [VTinker.pkl]=8ed128023af03332e8c2915b9b735aaff5c70667c054e0237ed032c47c43aa17        # 242743383 B, inference
  [UPR_ReDesgin.pkl]=13493f6832439a9b3f53a0445bf5e5388d9071afdd07145797ecfd7a11bd5ef4   # 6640821 B, training only
)
declare -A SIZE=([VTinker.pkl]=242743383 [UPR_ReDesgin.pkl]=6640821)

# Every cache/temp file on NFS, private to this job (never $HOME, never login-node /tmp, never shared caches).
export TMPDIR=$ROOT/cache/tmp/job_${SLURM_JOB_ID:-manual}
export PIP_CACHE_DIR=$TMPDIR/pip
export CONDA_PKGS_DIRS=$TMPDIR/conda_pkgs
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=$TMPDIR/xdg
export CUPY_CACHE_DIR=$ROOT/cache/cupy
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$CUPY_CACHE_DIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} cpus=${SLURM_CPUS_PER_TASK:-}"
source ${MVFI_CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh

# ---------------------------------------------------------------- weights (sha256-verified)
mkdir -p "$MODELS"
for f in VTinker.pkl UPR_ReDesgin.pkl; do
  if [ -f "$MODELS/$f" ] && echo "${SHA[$f]}  $MODELS/$f" | sha256sum -c --status; then
    echo "-- $f present and verified"
    continue
  fi
  url="https://huggingface.co/$HF_REPO/resolve/$HF_REV/$f"
  echo "-- downloading $url"
  curl -fL --retry 5 --retry-delay 10 -sS -o "$MODELS/$f.part" "$url"
  got_size=$(stat -c %s "$MODELS/$f.part")
  [ "$got_size" = "${SIZE[$f]}" ] || { echo "ERROR: $f size $got_size != ${SIZE[$f]}" >&2; exit 1; }
  echo "${SHA[$f]}  $MODELS/$f.part" | sha256sum -c - || { echo "ERROR: sha256 mismatch for $f" >&2; exit 1; }
  mv "$MODELS/$f.part" "$MODELS/$f"
done
chmod a-w "$MODELS"/*.pkl
(cd "$MODELS" && sha256sum VTinker.pkl UPR_ReDesgin.pkl > SHA256SUMS && cat SHA256SUMS)
# Model card (licence front-matter says MIT; the GitHub code licence is Pi-Lab 1.0 non-commercial).
curl -fsSL --retry 3 -o "$MODELS/HF_README.md" "https://huggingface.co/$HF_REPO/resolve/$HF_REV/README.md" || echo "WARN: model card download failed"
curl -fsSL --retry 3 -o "$MODELS/hf_model_info.json" "https://huggingface.co/api/models/$HF_REPO/revision/$HF_REV" || echo "WARN: HF API info download failed"

# ---------------------------------------------------------------- conda env
# A completed build leaves $ENV/.vtinker_build_ok; then only the checks below are re-run (unless REBUILD=1).
if [ -e "$ENV" ] && { [ "${REBUILD:-0}" = 1 ] || [ ! -e "$ENV/.vtinker_build_ok" ]; }; then
  if [ "${REBUILD:-0}" = 1 ]; then
    echo "-- REBUILD=1: removing $ENV"; rm -rf "${ENV:?}"
  else
    echo "ERROR: $ENV exists but is incomplete (no .vtinker_build_ok); use REBUILD=1." >&2; exit 1
  fi
fi
if [ ! -e "$ENV/.vtinker_build_ok" ]; then
conda create -y -p "$ENV" --override-channels -c conda-forge python=3.10.14 pip
conda activate "$ENV"
python -V

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
# nvjitlink: torch 2.4.1 leaves it unpinned (pip picks 12.9); keep every CUDA user-mode lib at 12.1 (<= driver 12.3).
pip install cupy-cuda12x==12.3.0 numpy==1.26.4 "fastrlock==0.8.2" tifffile==2024.8.30 pillow==10.4.0 \
  nvidia-nvjitlink-cu12==12.1.105

# activate.d hook: CUDA 12.1 runtime libs from the nvidia-* pip packages for CuPy (NVRTC JIT).
SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
mkdir -p "$ENV/etc/conda/activate.d" "$ENV/etc/conda/deactivate.d"
cat > "$ENV/etc/conda/activate.d/vtinker_cuda_libs.sh" <<EOF
# Added by scripts/envs/build_vtinker.sbatch: CUDA 12.1 runtime libraries (torch's nvidia-*-cu12 wheels) for CuPy 12.3.
export _VTINKER_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export _VTINKER_OLD_CUDA_PATH="\${CUDA_PATH:-}"
_vt_sp="$SP/nvidia"
_vt_libs=""
for _d in "\$_vt_sp"/*/lib; do _vt_libs="\${_vt_libs:+\$_vt_libs:}\$_d"; done
export LD_LIBRARY_PATH="\$_vt_libs\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export CUDA_PATH="\$_vt_sp/cuda_runtime"
unset _vt_sp _vt_libs _d
EOF
cat > "$ENV/etc/conda/deactivate.d/vtinker_cuda_libs.sh" <<'EOF'
export LD_LIBRARY_PATH="${_VTINKER_OLD_LD_LIBRARY_PATH:-}"
[ -z "$LD_LIBRARY_PATH" ] && unset LD_LIBRARY_PATH
if [ -n "${_VTINKER_OLD_CUDA_PATH:-}" ]; then export CUDA_PATH="$_VTINKER_OLD_CUDA_PATH"; else unset CUDA_PATH; fi
unset _VTINKER_OLD_LD_LIBRARY_PATH _VTINKER_OLD_CUDA_PATH
EOF
date -Is > "$ENV/.vtinker_build_ok"
conda deactivate
fi   # end of env creation

conda activate "$ENV"
SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_PATH=$CUDA_PATH"
ls "$CUDA_PATH"

echo "== CuPy install check (CPU nodes have no NVIDIA driver: libcuda.so.1 is the only allowed missing lib)"
missing=$(for so in "$SP"/cupy/_core/core.cpython-*.so "$SP"/cupy_backends/cuda/libs/nvrtc.cpython-*.so \
                   "$SP"/cupy_backends/cuda/api/runtime.cpython-*.so "$SP"/cupy_backends/cuda/libs/cublas.cpython-*.so; do
            ldd "$so" | grep "not found" || true; done | sort -u || true)
echo "unresolved: ${missing:-none}"
if [ -n "$(echo "$missing" | grep -v '^\s*$' | grep -v 'libcuda.so.1')" ]; then
  echo "ERROR: CuPy has unresolved CUDA libraries besides the driver" >&2; exit 1
fi
grep -n "compile_with_cache" "$SP"/cupy/cuda/__init__.py "$SP"/cupy/cuda/compiler.py | head -5 || true
python - <<'EOF'
import ctypes, glob, os
ctypes.CDLL("libnvrtc.so.12")
b = [p for d in os.environ["LD_LIBRARY_PATH"].split(":") for p in glob.glob(d + "/libnvrtc-builtins.so*")]
ctypes.CDLL(b[0])
print("NVRTC + builtins loadable:", b)
EOF

echo "== import smoke test (CPU node: cupy stubbed if the driver is absent; kernel JIT is tested in smoke_vtinker.sbatch)"
cd "$ROOT"
python - <<'EOF'
import os
import sys, hashlib, importlib, types
from pathlib import Path
import numpy, tifffile, PIL, torch, torchvision
print("python", sys.version.split()[0], "numpy", numpy.__version__, "tifffile", tifffile.__version__, "pillow", PIL.__version__)
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "torchvision", torchvision.__version__)
try:
    import cupy
    from cupy.cuda import nvrtc
    print("cupy", cupy.__version__, "nvrtc", nvrtc.getVersion(), "compile_with_cache present:",
          hasattr(cupy.cuda, "compile_with_cache"))
except ImportError as e:
    if "libcuda.so.1" not in str(e):
        raise
    print("cupy needs libcuda.so.1 (NVIDIA driver), absent on this CPU node -> stubbing cupy for the model/weights check")
    stub = types.ModuleType("cupy")
    stub.memoize = lambda **kw: (lambda f: f)
    sys.modules["cupy"] = stub
repo = Path(os.environ["MVFI_ROOT"] + "/src/VTinker").resolve()
sys.path.insert(0, str(repo))
m = importlib.import_module("core.models.VTinker")
assert Path(m.__file__).resolve().is_relative_to(repo), m.__file__
assert m.is_half is False
print("imported", m.__file__, "is_half =", m.is_half)
net = m.modelVFI()
sd = torch.load(os.environ["MVFI_ROOT"] + "/models/VTinker/VTinker.pkl",
                map_location="cpu", weights_only=True)
print("checkpoint type", type(sd).__name__, "entries", len(sd))
pref = {k[:7] for k in sd}
print("key prefixes", pref)
sd2 = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
own = net.state_dict()
missing = sorted(set(own) - set(sd2)); unexpected = sorted(set(sd2) - set(own))
print("missing", len(missing), missing[:10]); print("unexpected", len(unexpected), unexpected[:10])
dts = {}
for v in sd2.values():
    dts[str(v.dtype)] = dts.get(str(v.dtype), 0) + v.numel()
print("checkpoint dtypes (numel):", dts)
bad = [k for k in own if k in sd2 and own[k].shape != sd2[k].shape]
print("shape mismatches", bad)
net.load_state_dict({k: sd2[k] for k in own}, strict=True)
print("params", sum(p.numel() for p in net.parameters()))
assert not missing and not bad
print("IMPORT SMOKE TEST OK")
EOF

pip check || true
pip freeze > "$ROOT/provenance/pip_freeze_vtinker.txt"
conda list -p "$ENV" --explicit > "$MODELS/conda_explicit_vtinker.txt"
du -sh "$ENV"
echo "== done $(date -Is)"
