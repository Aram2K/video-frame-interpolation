# Reference hardware / software

| Item | Value |
|---|---|
| GPUs | 1 × NVIDIA H100 NVL 94 GB (compute capability 9.0), 1 × NVIDIA A100 80 GB PCIe (8.0), one per node |
| NVIDIA driver | 545.23.06 (CUDA ≤ 12.3); torch cu121/cu124 wheels work via CUDA minor-version compatibility |
| OS | EL8 (glibc 2.28) |
| Scheduler | Slurm; partitions `gpu` (GPU nodes) and `standard` (CPU nodes); GPUs requested with `--gres=gpu:A100:1` / `--gres=gpu:H100:1` |
| Storage | caches and temp files kept inside the project directory |
| Other | no system ffmpeg (conda-forge ffmpeg 7.1.1 used); no git binary on compute nodes |
