# Common environment for every movie_vfi Slurm job. Source it; do not execute it.
# All caches/temp files live inside the project directory (shared storage), not in $HOME or /tmp:
# home quotas and login-node /tmp are often small on HPC clusters.
export MVFI_ROOT=${MVFI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
# Site settings (Slurm account, conda location, login host): copy config.example.sh to config.local.sh.
if [ -f "$MVFI_ROOT/config.local.sh" ]; then source "$MVFI_ROOT/config.local.sh"; fi
export MVFI_ENV=${MVFI_ENV:-$MVFI_ROOT/env/ldfvfi}   # override per model before sourcing
export CUPY_CACHE_DIR=$MVFI_ROOT/cache/cupy
export PIP_CACHE_DIR=$MVFI_ROOT/cache/pip
export CONDA_PKGS_DIRS=$MVFI_ROOT/cache/conda_pkgs
export MVFI_REPO=$MVFI_ROOT/src/LDF-VFI
export MVFI_MODELS=$MVFI_ROOT/models

export HF_HOME=$MVFI_ROOT/cache/hf
export TORCH_HOME=$MVFI_ROOT/cache/torch
export XDG_CACHE_HOME=$MVFI_ROOT/cache/xdg
export TRITON_CACHE_DIR=$MVFI_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$MVFI_ROOT/cache/inductor
export MPLCONFIGDIR=$MVFI_ROOT/cache/mpl
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Per-job temp dir on NFS, removed when the job exits (the caller installs the trap).
export TMPDIR=$MVFI_ROOT/cache/tmp/job_${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUPY_CACHE_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$MPLCONFIGDIR"

source "${MVFI_CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
# Activate only if the env exists (build scripts source this file before creating it).
if [ -d "$MVFI_ENV/conda-meta" ]; then conda activate "$MVFI_ENV"; fi

# Defensive: refuse to run a GPU step outside a Slurm GPU allocation (not every cluster enforces device isolation).
mvfi_require_gpu() {
  if [ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ] || [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: no Slurm GPU allocation (use --gres=gpu:...); refusing to touch an unallocated GPU." >&2
    exit 2
  fi
}

# Sample GPU memory/utilisation of *our* allocated GPU every 5 s into a CSV; stopped by mvfi_cleanup.
mvfi_start_gpu_monitor() {
  local out=$1
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv -l 5 > "$out" 2>&1 &
  MVFI_SMI_PID=$!
}

mvfi_cleanup() {
  # Must never fail: it runs from an EXIT trap under `set -e`, where a non-zero status
  # would turn a successful job into FAILED and break dependent jobs.
  if [ -n "${MVFI_SMI_PID:-}" ]; then kill "$MVFI_SMI_PID" 2>/dev/null || true; fi
  rm -rf "$TMPDIR" 2>/dev/null || true
  return 0
}
