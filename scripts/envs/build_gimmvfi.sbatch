#!/bin/bash
# Build the GIMM-VFI inference env + download the official weights (CPU node, no GPU).
#   sbatch scripts/envs/build_gimmvfi.sbatch              # create env if missing, (re)install pins, fetch weights, test
#   REBUILD=1 sbatch scripts/envs/build_gimmvfi.sbatch    # delete env/gimmvfi first
# Stack (upstream pins py3.7/torch1.13/cu116 are not installable/usable here):
#   Python 3.10, not 3.11+: upstream dataclasses use instance defaults (HypoNetConfig() etc.), which 3.11 rejects
#   as "mutable default"; 3.10 accepts them, so no upstream patch is needed.
#   conda-forge: python 3.10.14, numpy 1.26.4, cupy 12.3.0 (still has cupy.cuda.compile_with_cache used by the
#                upstream softsplat.py) built for CUDA 12 with cuda-version=12.1 -> NVRTC 12.1 (<= driver 545 / CUDA 12.3)
#   pip (download.pytorch.org cu121): torch 2.5.1, torchvision 0.20.1 (torchvision: timm 0.4.12 imports it)
#   pip (PyPI): runtime imports of src/models + utils/config.py; timm 0.4.12 (FlowFormer's twins.py needs
#                timm.models.helpers.overlay_external_default_cfg, removed in later timm)
#SBATCH --job-name=vfi_env_gimmvfi
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
ROOT=$MVFI_ROOT
ENV=$ROOT/env/gimmvfi
REPO=$ROOT/src/GIMM-VFI
WDIR=$ROOT/models/GIMM-VFI
HF_REV=ab7735cdcfbd2e03c1bf2819380a25e8a4f321d1
ADAPTER=$ROOT/scripts/adapters/gimmvfi.py

# env.sh activates $MVFI_ENV; the model env may not exist yet, so start from conda base.
export MVFI_ENV=${MVFI_CONDA_BASE:-$HOME/miniconda3}
set +u; source $ROOT/scripts/env.sh; set -u
trap mvfi_cleanup EXIT
# package caches inside the per-job TMPDIR (removed by mvfi_cleanup), never $HOME or /tmp
export PIP_CACHE_DIR=$TMPDIR/pip CONDA_PKGS_DIRS=$TMPDIR/conda_pkgs PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"

if [ "${REBUILD:-0}" = 1 ] && [ -e "$ENV" ]; then
  echo "-- REBUILD=1: removing $ENV"; rm -rf "${ENV:?}"
fi
if [ ! -e "$ENV/conda-meta" ]; then
  rm -rf "${ENV:?}"
  conda create -y -p "$ENV" --override-channels -c conda-forge \
    python=3.10.14 pip numpy=1.26.4 cupy=12.3.0 cuda-version=12.1
fi
set +u; conda activate "$ENV"; set -u
python -V

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
pip install numpy==1.26.4 pillow==11.1.0 tifffile==2024.8.30 \
  omegaconf==2.3.0 PyYAML==6.0.2 easydict==1.13 yacs==0.1.8 einops==0.8.1 loguru==0.7.3 \
  scipy==1.13.1 timm==0.4.12

echo "== weights -> $WDIR (HF GSean/GIMM-VFI @ $HF_REV)"
mkdir -p "$WDIR"
cat > "$WDIR/SHA256SUMS" <<'EOF'
fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1  raft-things.pth
a7ae0cf958312bea17a0b01fe95b34a4fe3ff0c937d0dee3afeec872bd506ac8  flowformer_sintel.pth
1dff0ac3ca91483950eff814818115ff90cc3d16070089cc8cc3090eb3ee76f2  gimmvfi_r_arb.pt
b33e41536850594c01551978c4b63b1acd291c60d22310fbb6a4e57ac375033f  gimmvfi_r_arb_lpips.pt
e0ad627ea021b4234cd8ce33341d27abd505310b7d380e4c6cfbc8909f993a2e  gimmvfi_f_arb.pt
f2d52a1424d857a921e56fb82d9312ec27d58bbcbd58a7c3534593b9e6157858  gimmvfi_f_arb_lpips.pt
EOF
while read -r sha f; do
  if [ -f "$WDIR/$f" ] && [ "$(sha256sum "$WDIR/$f" | cut -d' ' -f1)" = "$sha" ]; then
    echo "   ok (cached) $f"; continue
  fi
  curl -fL --retry 5 --retry-delay 5 -sS -o "$WDIR/$f.part" "https://huggingface.co/GSean/GIMM-VFI/resolve/$HF_REV/$f"
  got=$(sha256sum "$WDIR/$f.part" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then echo "ERROR: sha256 mismatch for $f: $got" >&2; rm -f "$WDIR/$f.part"; exit 1; fi
  mv "$WDIR/$f.part" "$WDIR/$f"; echo "   downloaded $f"
done < "$WDIR/SHA256SUMS"
(cd "$WDIR" && sha256sum -c SHA256SUMS && ls -l --time-style=+ *.pt *.pth)
cat > "$WDIR/SOURCE.txt" <<EOF
Official GIMM-VFI checkpoints, https://huggingface.co/GSean/GIMM-VFI at revision $HF_REV (downloaded $(date -I), job $SLURM_JOB_ID).
Code: https://github.com/GSeanCDAT/GIMM-VFI (src/GIMM-VFI). Licence: S-Lab License 1.0, non-commercial
(src/GIMM-VFI/LICENSE); the HF repo itself states no licence. raft-things.pth = RAFT (BSD-3) release;
flowformer_sintel.pth = FlowFormer (Apache-2.0) as re-hosted by the GIMM-VFI authors.
gimm.pt (motion-model pretraining only) is not needed for inference and was not downloaded.
EOF
(cd "$WDIR" && sha256sum *.pt *.pth && stat -c '%s %n' *.pt *.pth) > $MVFI_ROOT/provenance/model_sha256_gimmvfi.txt

echo "== CuPy/NVRTC linkage (CPU nodes have no libcuda.so.1: only that one may be missing here)"
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
ldd "$SP"/cupy/_core/core.*.so "$SP"/cupy_backends/cuda/libs/nvrtc.*.so | grep -E "not found|nvrtc|cudart" | sort -u || true
python - "$ENV/lib/libnvrtc.so.12" <<'PYEOF'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1]); a, b = ctypes.c_int(), ctypes.c_int()
lib.nvrtcVersion(ctypes.byref(a), ctypes.byref(b)); print("conda-forge NVRTC", sys.argv[1], f"{a.value}.{b.value}")
PYEOF

echo "== import smoke test (CPU node: builds every variant with strict weight loading; no kernel run)"
cd "$ROOT"   # not the repo: proves the adapter does not rely on cwd-relative checkpoint paths
python - "$ADAPTER" <<'EOF'
import importlib.util, sys, time, types
from importlib.metadata import version
# CPU node: the real cupy cannot load (no libcuda.so.1). Upstream softsplat only needs cupy.memoize at import
# time; the real CuPy import + NVRTC JIT are exercised on the GPU by smoke_gimmvfi.sbatch.
fake = types.ModuleType("cupy"); fake.memoize = lambda **kw: (lambda f: f); fake.__version__ = "stub"
sys.modules["cupy"] = fake
import torch, torchvision, numpy, tifffile, PIL, timm, yacs, omegaconf, einops, loguru, scipy, easydict
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version(), "| torchvision", torchvision.__version__)
print("numpy", numpy.__version__, "| cupy (installed)", version("cupy"), "| tifffile", tifffile.__version__,
      "| pillow", PIL.__version__, "| timm", timm.__version__, "| omegaconf", omegaconf.__version__, "| scipy", scipy.__version__)
print("timm twins models:", timm.list_models("twins*"))
spec = importlib.util.spec_from_file_location("gimmvfi_adapter", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for v in ["R-P", "R", "F-P", "F"]:
    t0 = time.time()
    model, up, meta = mod.build_model(v)
    n = sum(p.numel() for p in model.parameters())
    dtypes = sorted({str(p.dtype) for p in model.state_dict().values()})
    print(f"{v}: built+loaded in {time.time()-t0:.1f}s, {n/1e6:.2f} M params, dtypes {dtypes}, "
          f"weights_only={meta['checkpoint_weights_only_load']}, ckpt keys {meta['checkpoint_top_level_keys']}")
    # informational: does the GIMM-VFI checkpoint carry its own flow-estimator weights (overwriting the standalone file)?
    flow_file = meta["weights"][0]["file"]
    ref = torch.load(flow_file, map_location="cpu", weights_only=True)
    ref = {k.replace("module.", ""): x for k, x in ref.items() if "module" in k}
    fe = model.flow_estimator.state_dict()
    diffs = [float((fe[k].float() - ref[k].float()).abs().max()) for k in ref if k in fe and fe[k].is_floating_point()]
    print(f"   flow_estimator vs {flow_file.split('/')[-1]}: {len(diffs)} tensors compared, max|diff| {max(diffs):.3g}")
    del model
print("IMPORT SMOKE OK")
EOF

pip check || true
pip freeze > $MVFI_ROOT/provenance/pip_freeze_gimmvfi.txt
conda list -p "$ENV" --explicit > $MVFI_ROOT/provenance/conda_explicit_gimmvfi.txt
conda list -p "$ENV" | grep -Ei "cuda|nvrtc|cupy|numpy|python " > $MVFI_ROOT/provenance/conda_cuda_pkgs_gimmvfi.txt || true
du -sh "$ENV"
echo "== done $(date -Is)"
