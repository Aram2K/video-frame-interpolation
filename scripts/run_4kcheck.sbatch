#!/bin/bash
# One-window 4K check (4 input frames x10 -> 31 output frames): real GPU memory and time per window at 4K,
# before committing hours of GPU time. Uses the real shot if it has been uploaded, else a synthetic 4K
# 16-bit sequence made from the upstream demo (upscaled; for timing/memory only).
#   SHOT=test01 sbatch scripts/run_4kcheck.sbatch
#SBATCH --job-name=vfi_ldf_4kcheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
mvfi_require_gpu
SHOT=${SHOT:-test01}
IN=$MVFI_ROOT/input/$SHOT
if [ -z "$(ls "$IN"/*.tif 2>/dev/null | head -1)" ]; then
  IN=$MVFI_ROOT/input/synthetic4k
  if [ "$(ls "$IN"/*.tif 2>/dev/null | wc -l)" -ne 4 ]; then
    rm -rf "$IN"
    python - "$IN" <<'PY'
import os
import sys, cv2, numpy as np, tifffile
from pathlib import Path
from torchcodec.decoders import VideoDecoder
out = Path(sys.argv[1]); out.mkdir(parents=True)
dec = VideoDecoder(os.environ["MVFI_ROOT"] + "/src/LDF-VFI/assets/demo.mp4", dimension_order="NHWC")
for i in range(4):
    a = cv2.resize(dec[i * 2].numpy(), (3840, 2160), interpolation=cv2.INTER_CUBIC).astype(np.uint16) * 257
    tifffile.imwrite(out / f"synth_{i:04d}.tif", a, photometric="rgb", metadata=None)
PY
  fi
fi
OUT=$MVFI_ROOT/output/_checks/ldf-vfi_4k_$(basename "$IN")_$SLURM_JOB_ID
mvfi_start_gpu_monitor "$MVFI_ROOT/logs/gpu_usage_4kcheck_${SLURM_JOB_ID}.csv"
echo "== $(date -Is) host=$(hostname) input=$IN"
cd "$MVFI_ROOT/scripts"
python ldf_tiff_infer.py --input "$IN" --out_dir "$OUT" --max_input_frames 4 \
  --model_path "$MVFI_MODELS/LDF-VFI/transformer" --vae_path "$MVFI_MODELS/LDF-VFI/Wan2.1_VAE_cond_v2.pth" \
  --temporal_sf 10 --fps 25
echo "peak nvidia-smi memory.used: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{print m" MiB"}' "$MVFI_ROOT/logs/gpu_usage_4kcheck_${SLURM_JOB_ID}.csv")"
python make_previews.py "$OUT" --source_dir "$IN" --fps 25 --view as_encoded
[ "$(basename "$IN")" != synthetic4k ] && python make_previews.py "$OUT" --source_dir "$IN" --fps 25 --view rec709
echo "== done $(date -Is)"
