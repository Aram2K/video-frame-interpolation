#!/bin/bash
# Independent re-verification of the reported VTinker bottom-row padding defect, read-only, CPU.
# Measures the signed per-row mean deviation of each generated frame from the 0.5*(img0+img1) blend
# near the bottom edge (rows with pad influence) and, as a control, near the right edge (no padding
# there, since 3840 is already a multiple of 128). Reads an EXISTING smoke output directory.
#   SMOKE=output/_checks/vtinker_smoke_319365 \
#     sbatch scripts/envs/check_vtinker_padbleed.sbatch
#SBATCH --job-name=vfi_check_vtinker_padbleed
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/vtinker
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
export PYTHONDONTWRITEBYTECODE=1

SMOKE=${SMOKE:-$MVFI_ROOT/output/_checks/vtinker_smoke_319365}
IN=$MVFI_ROOT/input/synthetic4k
OUT=$MVFI_ROOT/output/_checks/vtinker_padcheck_$SLURM_JOB_ID
mkdir -p "$OUT"
export SMOKE IN OUT
echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID smoke=$SMOKE"

python - <<'EOF'
import json, os, sys
from pathlib import Path
import numpy as np
import tifffile
sys.path.insert(0, os.environ["MVFI_ROOT"] + "/scripts")
from frameio import IMG_EXTS, natural_key

smoke, src_dir, out = Path(os.environ["SMOKE"]), Path(os.environ["IN"]), Path(os.environ["OUT"])
sf, nrows, ncols = 10, 40, 40
src = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: natural_key(p.name))

def mm(p):
    return tifffile.memmap(str(p), mode="r")

s = [mm(p) for p in src]
H, W = s[0].shape[:2]
print(f"{len(src)} sources {W}x{H} {s[0].dtype}; smoke {smoke}")
frames = sorted(smoke.glob("frame_*.tif"))
npairs = (len(frames) - 1) // sf

bot_rows, right_cols, per_t = [], [], {}
for j in range(npairs):
    b0 = 0.5 * (s[j][-nrows:].astype(np.float64) + s[j + 1][-nrows:].astype(np.float64))
    r0 = 0.5 * (s[j][:, -ncols:].astype(np.float64) + s[j + 1][:, -ncols:].astype(np.float64))
    for i in range(1, sf):
        k = j * sf + i
        a = mm(smoke / f"frame_{k:06d}.tif")
        db = (a[-nrows:].astype(np.float64) - b0).mean(axis=(1, 2))   # per row, over cols+channels
        dr = (a[:, -ncols:].astype(np.float64) - r0).mean(axis=(0, 2))  # per column, over rows+channels
        bot_rows.append(db)
        right_cols.append(dr)
        per_t.setdefault(round(i / sf, 1), []).append((j, db))
        del a

bot = np.stack(bot_rows)          # (27, nrows)
right = np.stack(right_cols)      # (27, ncols)
mean_bot = bot.mean(0)
mean_right = right.mean(0)
rows = list(range(H - nrows, H))
cols = list(range(W - ncols, W))
print("\nsigned per-ROW mean deviation from blend (16-bit levels), averaged over all "
      f"{bot.shape[0]} generated frames, last 20 rows:")
for r, v in zip(rows[-20:], mean_bot[-20:]):
    print(f"   row {r}: {v:+9.1f}")
print(f"\nstep between last two rows: {mean_bot[-1] - mean_bot[-2]:+.1f} levels")
print(f"rows with positive mean among the last 20: "
      f"{[r for r, v in zip(rows[-20:], mean_bot[-20:]) if v > 0]}")
print("\nCONTROL, signed per-COLUMN mean deviation, last 20 columns (width needs no padding):")
for c, v in zip(cols[-20:], mean_right[-20:]):
    print(f"   col {c}: {v:+9.1f}")
print(f"step between last two columns: {mean_right[-1] - mean_right[-2]:+.1f} levels")

print("\nper-t, row 2159 and 2158 (pair, t): deviation")
pt = {}
for t in sorted(per_t):
    for j, db in per_t[t]:
        pt[f"t={t} pair{j}"] = {"row_last": float(db[-1]), "row_last_minus_1": float(db[-2])}
    print(f"   t={t}: row {H-1} " + ", ".join(f"pair{j}={db[-1]:+.1f}" for j, db in per_t[t])
          + f" | row {H-2} " + ", ".join(f"pair{j}={db[-2]:+.1f}" for j, db in per_t[t]))

res = {"smoke_dir": str(smoke), "generated_frames": int(bot.shape[0]), "height": H, "width": W,
       "row_mean_dev_16bit": {str(r): float(v) for r, v in zip(rows, mean_bot)},
       "col_mean_dev_16bit": {str(c): float(v) for c, v in zip(cols, mean_right)},
       "last_row_step": float(mean_bot[-1] - mean_bot[-2]),
       "last_col_step": float(mean_right[-1] - mean_right[-2]),
       "per_t_row_last": pt}
(out / "padbleed.json").write_text(json.dumps(res, indent=1))
print("\nwrote", out / "padbleed.json")
EOF
echo "== done $(date -Is)"
