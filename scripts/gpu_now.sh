#!/bin/bash
# Live GPU snapshot of my running job, taken inside its own allocation (safe: no extra GPU use).
USER=${USER:-$(id -un)}
J=$(squeue -u "$USER" -h -t RUNNING -o "%i %b" | awk '$2 ~ /gpu/ {print $1; exit}')
[ -z "$J" ] && { echo "no GPU job of mine is running right now"; exit 0; }
echo "job $J ($(squeue -h -j "$J" -o '%j') on $(squeue -h -j "$J" -o '%N'))"
srun --jobid="$J" --overlap nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv 2>/dev/null
