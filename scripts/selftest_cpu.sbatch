#!/bin/bash
# CPU self-test of the whole non-model pipeline (reader, float sampler patch, writer, paste-back,
# frame map, reports, previews, hold-out scoring) with --fake_model. No GPU.
#SBATCH --job-name=vfi_selftest
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
T=$MVFI_ROOT/output/_checks/selftest_$SLURM_JOB_ID
mkdir -p "$T"
cd "$MVFI_ROOT/scripts"
# synthetic 16-bit Gen5-like sequence, 6 frames, 640x360, moving gradient
python - "$T/src16" <<'PY'
import sys, numpy as np, tifffile
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True)
yy, xx = np.mgrid[0:360, 0:640].astype(np.float32)
for i in range(6):
    r = 0.1 + 0.6 * ((xx + 40 * i) % 640) / 640
    g = 0.1 + 0.6 * yy / 360
    b = np.full_like(r, 0.4 + 0.03 * i)
    a = np.round(np.stack([r, g, b], -1) * 65535).astype(np.uint16)
    tifffile.imwrite(out / f"clip{i + 90045:08d}.tif", a, photometric="rgb", metadata=None)
PY
python inspect_input.py "$T/src16" "$T/meta/input_metadata.json" --fps 25 --sf 10 || echo "inspect rc=$?"
python make_rec709.py "$T/src16" "$T/src16_rec709" --workers 4
for spec in "A:$T/src16:10:1" "B:$T/src16_rec709:10:1" "HA:$T/src16:2:2" "HB:$T/src16_rec709:2:2"; do
  IFS=: read -r name inp sf stride <<< "$spec"
  python ldf_tiff_infer.py --fake_model --input "$inp" --out_dir "$T/$name" --temporal_sf "$sf" --input_stride "$stride" \
    --model_path none --vae_path none --fps 25
done
python ldf_tiff_infer.py --fake_model --input "$MVFI_REPO/assets/demo.mp4" --out_dir "$T/demo" --temporal_sf 8 \
  --model_path none --vae_path none --fps 30 --max_input_frames 6
python make_previews.py "$T/A" --source_dir "$T/src16" --view as_encoded --width 640
python make_previews.py "$T/A" --source_dir "$T/src16" --view rec709 --width 640
python make_previews.py "$T/demo" --source_dir "$T/demo/decoded_source" --fps 30 --width 640
python eval_holdout.py --run "A:$T/HA:gen5" --run "B:$T/HB:rec709" --out "$T/holdout_scores.json" --lpips_width 320
echo "== checks"
python - "$T" <<'PY'
import sys, json, csv
from pathlib import Path
import numpy as np, tifffile
T = Path(sys.argv[1])
rep = json.loads((T / "A/logs/run_report.json").read_text())
assert rep["output_frames"] == (6 - 1) * 10 + 1 == len(list((T / "A").glob("frame_*.tif"))), rep["output_frames"]
rows = list(csv.DictReader(open(T / "A/frame_map.csv")))
assert [r["kind"] for r in rows[:11]] == ["genuine"] + ["generated"] * 9 + ["genuine"]
for j, src in enumerate(sorted((T / "src16").glob("*.tif"))):
    assert np.array_equal(tifffile.imread(T / f"A/frame_{j*10:06d}.tif"), tifffile.imread(src)), src
mid = tifffile.imread(T / "A/frame_000005.tif").astype(float)
a, b = tifffile.imread(T / "A/frame_000000.tif").astype(float), tifffile.imread(T / "A/frame_000010.tif").astype(float)
assert np.abs(mid - (a + b) / 2).max() <= 1.0, "t=0.5 frame is not the midpoint -> indexing/scale bug"
assert tifffile.imread(T / "A/frame_000005.tif").dtype == np.uint16
print("upstream:", rep["software"]["upstream_commit"], rep["software"]["upstream_changed_files"])
print("A ok; holdout:", json.loads((T / "holdout_scores.json").read_text())["results"]["A"]["psnr_display_db"])
print("demo frames:", json.loads((T / "demo/logs/run_report.json").read_text())["output_frames"])
PY
ls "$T/A" "$T/A/comparison" "$T/demo"
echo "== selftest PASSED $(date -Is)"
