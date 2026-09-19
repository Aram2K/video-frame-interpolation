#!/bin/bash
# VTinker hold-out check: compare generated frames at t != 0.5 against REAL frames, not against each other.
# The released model was trained at t = 0.5 only, so the t = 0.1..0.9 sweep of the deliverable is an
# extrapolation; every t-related number in the smoke is a self-consistency check that a cross-dissolve would
# also pass. The harness's hold-out mode (temporal_sf == input_stride) scores each generated index against the
# real frame at that index (logs/holdout_metrics.csv).
#
# input/synthetic4k has only 4 frames, so the reachable ground-truth timesteps are:
#   SF=STRIDE=3, pair (0,3):  t = 1/3 and t = 2/3   <- off-centre t against real frames
#   SF=STRIDE=2, pair (0,2):  t = 1/2
# The same runs are repeated with the shared _linear_blend adapter as the "is it better than a dissolve?"
# baseline. NOTE: a larger stride also means larger motion, so the PSNR level is not comparable to a
# stride-1 x10 delivery run; the VTinker-minus-blend gain at each t is the meaningful quantity here.
# The full t = 0.1..0.9 PSNR curve still needs a real shot (>= 31 frames):
#   SHOT=<shot> VARIANT=holdout SF=10 STRIDE=10 sbatch scripts/envs/run_vtinker.sbatch
#SBATCH --job-name=vfi_holdout_vtinker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/vtinker
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
mvfi_require_gpu
export PYTHONDONTWRITEBYTECODE=1

IN=$MVFI_ROOT/input/synthetic4k
OUT=$MVFI_ROOT/output/_checks/vtinker_holdout_$SLURM_JOB_ID
mkdir -p "$OUT"
export IN OUT
echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,driver_version --format=csv,noheader

for sf in 3 2; do
  for model in vtinker _linear_blend; do
    echo "-- hold-out sf=$sf stride=$sf adapter=$model"
    python "$MVFI_ROOT/scripts/vfi_harness.py" \
      --adapter "$MVFI_ROOT/scripts/adapters/$model.py" \
      --input "$IN" --out_dir "$OUT/sf${sf}_${model}" \
      --temporal_sf "$sf" --input_stride "$sf" --fps 25
  done
done

echo "== summary"
python - <<'EOF'
import csv, json, os
from pathlib import Path
out = Path(os.environ["OUT"])
rows, table = [], {}
for d in sorted(out.iterdir()):
    mf = d / "logs" / "holdout_metrics.csv"
    if not mf.is_file():
        continue
    sf = int(d.name.split("_")[0][2:])
    model = d.name.split("_", 1)[1]
    for r in csv.DictReader(open(mf)):
        k = int(r["out_index"])
        rec = {"config": d.name, "sf": sf, "model": model, "out_index": k, "t": k / sf,
               "gt": r["gt_src"], "psnr_db": float(r["psnr_db"]),
               "mean_abs_diff_16bit": float(r["mean_abs_diff_16bit"])}
        rows.append(rec)
        table.setdefault((sf, round(k / sf, 4)), {})[model] = rec
print(f"{'t':>6} {'gap':>4} {'VTinker PSNR':>13} {'blend PSNR':>11} {'gain dB':>8} "
      f"{'VTinker MAD':>12} {'blend MAD':>10}")
gains = []
for (sf, t), m in sorted(table.items()):
    v, b = m.get("vtinker"), m.get("_linear_blend")
    if not (v and b):
        continue
    g = v["psnr_db"] - b["psnr_db"]
    gains.append(g)
    print(f"{t:>6.3f} {sf:>4} {v['psnr_db']:>13.2f} {b['psnr_db']:>11.2f} {g:>+8.2f} "
          f"{v['mean_abs_diff_16bit']:>12.1f} {b['mean_abs_diff_16bit']:>10.1f}")
(out / "holdout_summary.json").write_text(json.dumps(
    {"rows": rows, "vtinker_minus_blend_db": gains,
     "note": "synthetic4k has 4 frames; reachable ground-truth t are 1/3, 2/3 (gap 3) and 1/2 (gap 2). "
             "A larger stride means larger motion, so PSNR levels are not comparable across gaps; the "
             "VTinker-minus-blend gain at each t is the comparable quantity."}, indent=1))
ok = all(g > 0 for g in gains)
print("\nHOLD-OUT", "PASSED (VTinker beats the cross-dissolve at every measured t)" if ok
      else "FAILED (VTinker does not beat a cross-dissolve at some measured t)")
raise SystemExit(0 if ok else 1)
EOF
echo "== done $(date -Is)  output: $OUT"
