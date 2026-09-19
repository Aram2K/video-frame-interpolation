#!/bin/bash
# Milestone step 4: upstream bundled demo (8x on assets/demo.mp4), then patch-equivalence check,
# then the same demo through our 16-bit TIFF wrapper.
# GPU: one A100 80GB (any GPU with >= 80 GB works).
#SBATCH --job-name=vfi_ldf_demo
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

OUT=$MVFI_ROOT/output/demo/ldf-vfi
MODEL=$MVFI_MODELS/LDF-VFI/transformer
VAE=$MVFI_MODELS/LDF-VFI/Wan2.1_VAE_cond_v2.pth
mkdir -p "$OUT"
mvfi_start_gpu_monitor "$OUT/gpu_usage_${SLURM_JOB_ID}.csv"
echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16); print("matmul ok", float((x @ x).float().abs().mean()))
EOF
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,pix_fmt -of csv=p=0 "$MVFI_REPO/assets/demo.mp4"

# (1) Upstream quick_start, unmodified code; only paths/output_dir differ (quick_start/generate.sh args verbatim).
cd "$MVFI_REPO"
T0=$(date +%s)
accelerate launch --num_machines=1 --num_processes=1 --mixed_precision=no --dynamo_backend=no generate.py \
  --model_path="$MODEL" --attention_type=slide_chunk_all_block_2x1x1 --temporal_upsample=nearest --num_frames=60 \
  --vae_type=wan2_1_cond_v2 --vae_path="$VAE" --vae_batch_size=16 \
  --tile_min_h=256 --tile_min_w=256 --tile_min_t=20 --tile_stride_h=192 --tile_stride_w=192 \
  --spatial_compression_ratio=8 --temporal_compression_ratio=4 \
  --sampling_steps=16 --t_shift=8 \
  --data=assets/demo.mp4 --temporal_sf=8 --output_dir="$OUT/upstream" --fps=30 --save_lq \
  --sp_size=1 2>&1 | tee "$OUT/upstream_log.txt"
echo "UPSTREAM_DEMO_SECONDS=$(( $(date +%s) - T0 ))"
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,nb_read_frames,r_frame_rate -of csv=p=0 "$OUT/upstream/demo/pred.mp4"
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$OUT/upstream/demo/lq.mp4"

# (2) Our wrapper on the same demo (decodes the mp4 once to lossless PNG, float path, 16-bit TIFF out).
cd "$MVFI_ROOT/scripts"
T0=$(date +%s)
python ldf_tiff_infer.py --input "$MVFI_REPO/assets/demo.mp4" --out_dir "$OUT/wrapper_sf8" \
  --model_path "$MODEL" --vae_path "$VAE" --temporal_sf 8 --fps 30
echo "WRAPPER_DEMO_SECONDS=$(( $(date +%s) - T0 ))"

# (3) Equivalence: upstream uint8 sampler vs float sampler, same frames, same seed, first 3 chunks.
python verify_patch.py --frames "$OUT/wrapper_sf8/decoded_source" --model_path "$MODEL" --vae_path "$VAE" \
  --temporal_sf 8 --chunks 3 --out_json "$OUT/wrapper_sf8/logs/patch_equivalence.json"

# (4) Viewing aids for the wrapper run.
python make_previews.py "$OUT/wrapper_sf8" --source_dir "$OUT/wrapper_sf8/decoded_source" --fps 30 --width 1280
echo "== done $(date -Is)"
