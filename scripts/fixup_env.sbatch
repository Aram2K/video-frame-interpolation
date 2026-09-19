#!/bin/bash
# Align transitive packages with upstream requirements.txt pins and add matplotlib (upstream pin) for plots.
#SBATCH --job-name=vfi_fixup_env
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
source "$MVFI_ROOT/scripts/env.sh"
export PIP_CACHE_DIR=$TMPDIR/pip
trap mvfi_cleanup EXIT
pip install pandas==2.2.3 pyarrow==19.0.1 matplotlib==3.10.1
python - <<'EOF'
import os
import sys
sys.path.insert(0, os.environ["MVFI_ROOT"] + "/src/LDF-VFI")
sys.path.insert(0, os.environ["MVFI_ROOT"] + "/scripts")
import generate, pandas, pyarrow, matplotlib, colour, tifffile
import colorxf
print("pandas", pandas.__version__, "pyarrow", pyarrow.__version__, "matplotlib", matplotlib.__version__,
      "colour", colour.__version__, "tifffile", tifffile.__version__)
print("Gen5 18% grey ->", float(colour.models.oetf_BlackmagicFilmGeneration5(0.18)))
print(colorxf.describe())
import numpy as np
x = np.linspace(0, 1, 5, dtype=np.float32)[:, None].repeat(3, 1)
print("Gen5 code -> 709 display:", colorxf.gen5_to_rec709_display(x)[:, 0].round(4).tolist())
EOF
pip check || true
pip freeze > $MVFI_ROOT/provenance/pip_freeze_ldfvfi.txt
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E " libx264 " || echo "WARNING: no libx264 in conda ffmpeg"
echo "== done $(date -Is)"
