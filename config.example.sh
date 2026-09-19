# Site settings. Copy to config.local.sh (git-ignored) and fill in; scripts/env.sh and
# scripts/run_shot_all.sh source it. For manual `sbatch`, run `source config.local.sh` first
# and submit from the project root, because job logs go to logs/ relative to it.

export SBATCH_ACCOUNT=your_slurm_account          # read by sbatch for every job
export SLURM_ACCOUNT=$SBATCH_ACCOUNT              # read by srun
export MVFI_CONDA_BASE=$HOME/miniconda3           # conda installation used to create/activate envs
export MVFI_SSH_TARGET=user@cluster-login-host    # only used to print download commands
