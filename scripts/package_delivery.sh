#!/bin/bash
# Build a download tree with no duplicated frames, then print the exact scp command for the PC.
#   ./package_delivery.sh test01
# Inside a run, the final sequence and reconstructed/ are hard links to the same files, so a plain
# "scp -r" would copy every frame twice. Here the final sequence is linked once, and only the frames
# that genuinely differ (the model's own versions of the genuine frames) are added.
set -euo pipefail
R=${MVFI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
if [ -f "$R/config.local.sh" ]; then source "$R/config.local.sh"; fi
SHOT=${1:?usage: package_delivery.sh SHOT}
DEL=${DEL:-$R/delivery}
OUT=$R/output/$SHOT
[ -d "$OUT" ] || { echo "no results for shot $SHOT" >&2; exit 1; }
rm -rf "$DEL/$SHOT"
mkdir -p "$DEL/$SHOT"

copy_run() {           # <run dir> <destination>
  local run=$1 dst=$2 sf
  mkdir -p "$dst"
  if [ "${ONLY:-}" = previews ]; then      # viewing material only (MB, not GB)
    for extra in frame_map.csv preview.mp4 preview_rec709.mp4; do
      [ -e "$run/$extra" ] && ln -f "$run/$extra" "$dst/" || true
    done
    [ -d "$run/comparison" ] && cp -al "$run/comparison" "$dst/comparison" || true
    [ -d "$run/logs" ] && cp -aL "$run/logs" "$dst/logs" || true
    return
  fi
  mkdir -p "$dst/frames"
  find "$run" -maxdepth 1 -name 'frame_*.tif' -exec ln -f {} "$dst/frames/" \;
  sf=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['temporal_sf'])" "$run/logs/run_report.json")
  # the model's own version of the genuine frames (differs from the pasted originals only for LDF-VFI)
  if [ -d "$run/reconstructed" ]; then
    mkdir -p "$dst/model_version_of_genuine_frames"
    for f in "$run"/reconstructed/frame_*.tif; do
      n=$(basename "$f" .tif); n=${n#frame_}
      if [ $((10#$n % sf)) -eq 0 ] && ! cmp -s "$f" "$run/frame_${n}.tif"; then
        ln -f "$f" "$dst/model_version_of_genuine_frames/"
      fi
    done
    rmdir "$dst/model_version_of_genuine_frames" 2>/dev/null || true
  fi
  for extra in frame_map.csv preview.mp4 preview_rec709.mp4; do
    [ -e "$run/$extra" ] && ln -f "$run/$extra" "$dst/" || true
  done
  [ -d "$run/comparison" ] && cp -al "$run/comparison" "$dst/comparison" || true
  [ -d "$run/logs" ] && cp -aL "$run/logs" "$dst/logs" || true
}

for rep in "$OUT"/*/*/logs/run_report.json "$OUT"/*/*/*/logs/run_report.json; do
  [ -e "$rep" ] || continue
  run=$(dirname "$(dirname "$rep")")
  rel=${run#$OUT/}
  echo "packaging $rel"
  copy_run "$run" "$DEL/$SHOT/$rel"
done
[ -d "$OUT/_comparison" ] && cp -aL "$OUT/_comparison" "$DEL/$SHOT/_comparison" || true
[ -d "$OUT/input_check" ] && cp -aL "$OUT/input_check" "$DEL/$SHOT/input_check" || true
cp -aL "$R/provenance" "$DEL/provenance"
cp -L "$R/README.md" "$DEL/README_project.md"
mkdir -p "$DEL/scripts" && cp -L "$R"/scripts/*.py "$R"/scripts/*.sbatch "$R"/scripts/*.sh "$DEL/scripts/" 2>/dev/null || true
cp -aL "$R/scripts/adapters" "$DEL/scripts/adapters" 2>/dev/null || true

cat > "$DEL/README_DELIVERY.md" <<EOF
# movie_vfi delivery - shot $SHOT - $(date -Is)

One folder per model and variant: <model>/<variant>/
  frames/frame_XXXXXX.tif   the sequence to bring back into Resolve (16-bit TIFF, 25 fps).
                            Genuine photographed frames sit at index k*sf and are your ORIGINAL pixels,
                            bit-for-bit; every other frame is generated.
  model_version_of_genuine_frames/   only for models that regenerate the genuine frames (LDF-VFI):
                            the model's own version of those frames, for comparison. Not used in frames/.
  frame_map.csv             per output frame: genuine or generated, t, source file names.
  preview*.mp4              8-bit H.264, viewing only (preview_rec709 = technical Rec.709 view of log frames).
  comparison/               side-by-side, contact sheets, anchor error maps, temporal profile.
  logs/run_report.json      frame counts, timing, GPU memory, versions, settings, model provenance.
_comparison/                all models side by side + summary table.
input_check/                validation of the source frames (numbering, bit depth, duplicates, cuts).
provenance/                 cluster inspection, package versions, model checksums, upstream commits.
scripts/                    everything used to produce this, including the model adapters.

PRECISION: LDF-VFI computes in bfloat16 from 8-bit-scaled values; its TIFFs are 16-bit containers
holding roughly 9 bits of real precision. The pairwise flow models (EMA-VFI, BiM-VFI, GIMM-VFI,
VTinker) warp the real 16-bit pixels in float32, so they preserve more of the source precision.
No grade, sharpening, denoise, grain or stabilisation was applied anywhere.
EOF

echo
du -sh "$DEL/$SHOT" "$DEL" 2>/dev/null
echo "files: $(find "$DEL" -type f | wc -l)"
echo
echo "Download from your PC (PowerShell), into your user folder:"
echo "  scp -r ${MVFI_SSH_TARGET:-user@cluster-login-host}:$DEL \"C:\\path\\to\\movie_vfi_results\""
