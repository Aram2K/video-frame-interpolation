#!/bin/bash
# Download the official LDF-VFI weights, pinned to one Hugging Face revision, and verify sha256.
#SBATCH --job-name=vfi_download_models
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
REPO_ID=onecat-ai/LDF-VFI
REV=87816885edfe4dba20190aea399b40181239824f
DEST=$MVFI_ROOT/models/LDF-VFI
mkdir -p "$DEST/transformer"
echo "== $(date -Is) host=$(hostname) $REPO_ID@$REV"

declare -A SHA=(
  [transformer/config.json]=""
  [transformer/diffusion_pytorch_model.safetensors]=73b82c245d9985f5311b0bb4568cba63bca0304ec83a26f17e57a89604174d0f
  [Wan2.1_VAE_cond_v2.pth]=96fcd7f89998868a681b715d030eac99420335421785fccc581c2541303b9515
  [README.md]=""
)
for f in "${!SHA[@]}"; do
  url="https://huggingface.co/$REPO_ID/resolve/$REV/$f"
  echo "-> $f"
  curl -fL --retry 5 --retry-delay 10 -C - -sS -o "$DEST/$f" "$url"
  got=$(sha256sum "$DEST/$f" | cut -d' ' -f1)
  want=${SHA[$f]}
  if [ -n "$want" ] && [ "$got" != "$want" ]; then
    echo "SHA256 MISMATCH for $f: $got != $want" >&2; exit 1
  fi
  echo "   $(stat -c %s "$DEST/$f") bytes sha256=$got ${want:+(verified)}"
done
cat > "$DEST/REVISION.json" <<EOF
{"repo_id": "$REPO_ID", "revision": "$REV", "downloaded": "$(date -Is)", "license": "apache-2.0"}
EOF
(cd "$DEST" && sha256sum transformer/config.json transformer/diffusion_pytorch_model.safetensors Wan2.1_VAE_cond_v2.pth README.md) \
  > $MVFI_ROOT/provenance/model_sha256.txt
echo "== done $(date -Is)"
