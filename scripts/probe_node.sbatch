#!/bin/bash
# Read-only hardware/software probe of a GPU node. Takes ~10 s.
# Submitted from the project root, inside a GPU allocation:
#   sbatch -p gpu --gres=gpu:1 scripts/probe_node.sbatch
#SBATCH --job-name=vfi_probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --output=logs/%x_%j_%N.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
uname -r; cat /etc/redhat-release 2>/dev/null; ldd --version | head -1
echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap,mig.mode.current --format=csv 2>&1
echo "== CPU"; lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)"
echo "== RAM"; free -g | head -2
echo "== disks"; df -h / /tmp /dev/shm "$MVFI_ROOT" 2>&1
echo "== internet from compute node"
for u in https://pypi.org/simple/ https://download.pytorch.org/whl/cu121/torch/ https://huggingface.co/api/models/onecat-ai/LDF-VFI https://github.com; do
  printf "%s -> " "$u"; timeout 10 curl -sS -o /dev/null -w "%{http_code}\n" "$u" 2>&1
done
echo "== ffmpeg: $(command -v ffmpeg || echo missing)"
echo "== done $(date -Is)"
