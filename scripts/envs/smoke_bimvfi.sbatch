#!/bin/bash
# BiM-VFI smoke test on the A100: 4 synthetic 4K 16-bit frames x10 -> 31 frames through the shared harness, then checks
# (31 frames, no NaN/Inf, output range, genuine frames bit-exact, t=0.1 closer to img0 and t=0.9 closer to img1).
# Before the harness run, a short pre-check: CuPy cost-volume kernel vs a pure-torch reference, one 4K forward pass
# (time, memory) with the adapter defaults, pyr_level 8 (cost, CPU antialias fallback, t order);
# PRECHECK=full adds fp32 vs TF32; reflect/replicate vs the upstream constant padding default (per-edge, interior
# delta, the normalisation-statistics mechanism) and a ground-truth hold-out that can actually rank padding modes.
# PRECHECK=0 skips the pre-check entirely.
#   sbatch scripts/envs/smoke_bimvfi.sbatch            [ADAPTER_ARGS="pyr_level=7 ..."] [PRECHECK=0|1|full]
#SBATCH --job-name=vfi_smoke_bimvfi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export MVFI_ROOT=${MVFI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # run/submit from the project root, or export MVFI_ROOT
set -euo pipefail
export MVFI_ENV=$MVFI_ROOT/env/bimvfi
source "$MVFI_ROOT/scripts/env.sh"
trap mvfi_cleanup EXIT
mvfi_require_gpu
[ "$(command -v python)" = "$MVFI_ENV/bin/python" ] || { echo "ERROR: env $MVFI_ENV not active ($(command -v python))" >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1  # keep src/BiM-VFI free of __pycache__
unset SLURM_PROCID  # upstream utils/experiment.py would treat the job as a distributed launch

IN=$MVFI_ROOT/input/synthetic4k
OUT=$MVFI_ROOT/output/_checks/bimvfi_smoke_$SLURM_JOB_ID
ADAPTER=$MVFI_ROOT/scripts/adapters/bimvfi.py
ADAPTER_ARGS=${ADAPTER_ARGS:-"pyr_level=7 cudnn_benchmark=false"}  # pad_mode left at the adapter default (upstream constant)
SMI=$MVFI_ROOT/logs/gpu_usage_bimvfi_smoke_${SLURM_JOB_ID}.csv
echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES env=$MVFI_ENV"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
mvfi_start_gpu_monitor "$SMI"
PRE_FAIL=0

if [ "${PRECHECK:-1}" != 0 ]; then
echo "== pre-check $(date -Is)"
python - "$ADAPTER" "$IN" "${PRECHECK:-1}" <<'EOF' || PRE_FAIL=1
import os
import importlib.util, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.environ["MVFI_ROOT"] + "/scripts")
from frameio import IMG_EXTS, natural_key, read_image
from pathlib import Path
spec = importlib.util.spec_from_file_location("bimvfi_adapter", sys.argv[1])
ad = importlib.util.module_from_spec(spec); spec.loader.exec_module(ad)
dev = torch.device("cuda")
FULL = sys.argv[3] == "full"
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print("preloaded", ad.preload_nvrtc())
ad.import_bimvfi()
import cupy
from cupy.cuda import nvrtc
print("cupy", cupy.__version__, "nvrtc", nvrtc.getVersion(), "cupy sees", cupy.cuda.runtime.getDeviceCount(), "GPU(s)")
from modules.components.bim_vfi.costvol import costvol_func
torch.backends.cudnn.allow_tf32 = False
for dt in (torch.float32,):
    a = torch.randn(2, 8, 23, 37, device=dev, dtype=dt); b = torch.randn(2, 8, 23, 37, device=dev, dtype=dt)
    out = costvol_func.apply(a, b, 9)
    pat = F.unfold(b, 9, padding=4).view(2, 8, 81, 23, 37)
    ref = (a.unsqueeze(2) * pat).sum(1)
    err = (out - ref).abs().max().item()
    print(f"costvol kernel vs torch reference ({dt}): max abs err {err:.3g}")
    assert err < 1e-4, err

src = sorted([p for p in Path(sys.argv[2]).iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: natural_key(p.name))
def load(p):
    a = read_image(p)
    return torch.from_numpy(a.astype("float32") / 65535.0).permute(2, 0, 1).contiguous().to(dev)
i0, i1 = load(src[0]), load(src[1])
print("input", tuple(i0.shape), i0.dtype, float(i0.min()), float(i0.max()))

def run(adp, label, t=0.5, reps=2):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        t0 = time.time(); o = adp.interpolate(i0, i1, [t])[0]; torch.cuda.synchronize(); times.append(time.time() - t0)
    print(f"{label}: times {[round(x, 2) for x in times]} s, peak alloc {torch.cuda.max_memory_allocated()/2**30:.1f} GiB,"
          f" reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB, padded {adp.padded}, out {tuple(o.shape)} {o.dtype}"
          f" range [{float(o.min()):.4f}, {float(o.max()):.4f}]")
    return o

def order(adp, label):
    o1, o9 = adp.interpolate(i0, i1, [0.1, 0.9])
    m = lambda x, y: float((x.clamp(0, 1) - y).abs().mean())
    ok = m(o1, i0) < m(o1, i1) and m(o9, i1) < m(o9, i0)
    print(f"{label}: t=0.1 d(img0)={m(o1, i0):.4f} d(img1)={m(o1, i1):.4f} | t=0.9 d(img0)={m(o9, i0):.4f} "
          f"d(img1)={m(o9, i1):.4f} -> order_ok={ok}")
    assert ok

A = ad.Adapter(dev)  # adapter defaults (cudnn_benchmark=false, fp32, pyr7, upstream constant padding)
print("weights", A.meta)
DEFAULT_MODE, DEFAULT_MARGIN = A.pad_mode, A.margin
def set_mode(adp, mode, margin=64):
    adp.pad_mode, adp.margin = mode, (0 if mode == "constant" else margin)
def tag(mode, margin):
    return f"{mode}{margin if mode != 'constant' else ''}"
print(f"adapter default pad_mode={DEFAULT_MODE}"
      + (" (= upstream InputPadder)" if DEFAULT_MODE == "constant" else " (DEVIATION from upstream)"))
DEF = f"pyr7 {tag(DEFAULT_MODE, DEFAULT_MARGIN)}"
o32 = run(A, f"{DEF} fp32 cudnn_benchmark={torch.backends.cudnn.benchmark}", reps=3)
if FULL:
    o32b = A.interpolate(i0, i1, [0.5])[0]
    print("repeatability fp32 max abs diff:", float((o32 - o32b).abs().max()))
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = True
    otf = run(A, f"{DEF} tf32")
    print(f"tf32 vs fp32: max abs {float((otf - o32).abs().max()):.3g}, mean abs {float((otf - o32).abs().mean()):.3g}")
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    del otf, o32b; torch.cuda.empty_cache()

    MODES = (("constant", 0), ("reflect", 64), ("replicate", 64))
    outs = {}
    for mode, margin in MODES:
        set_mode(A, mode, margin)
        outs[mode] = run(A, f"pyr7 {tag(mode, margin)}", reps=1)

    # (1) The metric the earlier revision leaned on. Kept, but labelled for what it is.
    blend = 0.5 * (i0 + i1)
    def edge_report(o, label, ref, name, e=16):
        d = (o.clamp(0, 1) - ref).abs().mean(0)
        print(f"  {label}: |out-{name}| top {float(d[:e].mean()):.4f} bottom {float(d[-e:].mean()):.4f} "
              f"left {float(d[:, :e].mean()):.4f} right {float(d[:, -e:].mean()):.4f} "
              f"interior {float(d[e:-e, e:-e].mean()):.4f} whole {float(d.mean()):.4f}")
    print("edge bands at t=0.5 vs a 50/50 blend. CAUTION: a blend is not a reference -- this mostly")
    print("measures how far the model departs from linear blending (i.e. motion), so it does NOT rank")
    print("padding modes. Recorded only for continuity with job 319322.")
    for mode, margin in MODES:
        edge_report(outs[mode], tag(mode, margin), blend, "blend")

    # (2) How much of the frame does the padding choice actually move?
    for mode, margin in MODES[1:]:
        d = (outs[mode] - outs["constant"]).abs()
        print(f"  {tag(mode, margin)} vs constant(upstream): mean abs INTERIOR "
              f"{float(d[:, 16:-16, 16:-16].mean()):.5f} whole-frame {float(d.mean()):.5f} max {float(d.max()):.4f}")

    # (3) Mechanism: the model's normalisation is computed over the tensor it is handed, so pre-padding
    #     changes the statistics applied to EVERY pixel (bim_vfi.py:147-158, before InputPadder at :162).
    def norm_stats(a, b):
        st = [a, b]
        m = sum(t.mean([1, 2, 3], True) for t in st) / len(st)
        s = (sum(t.std([1, 2, 3], False, True).square() + (m - t.mean([1, 2, 3], True)).square()
                 for t in st) / len(st)).sqrt()
        return float(m), float(s)
    a0, a1 = i0.unsqueeze(0), i1.unsqueeze(0)
    set_mode(A, "reflect", 64)
    p0, _ = A._pad(a0); p1, _ = A._pad(a1)
    m_u, s_u = norm_stats(a0, a1)
    m_r, s_r = norm_stats(p0, p1)
    frac = 1 - (a0.shape[-1] * a0.shape[-2]) / (p0.shape[-1] * p0.shape[-2])
    print("normalisation statistics the model applies (bim_vfi.py:147-158, computed BEFORE InputPadder):")
    print(f"  constant/upstream (unpadded {tuple(a0.shape[-2:])}): mean {m_u:.6f} std {s_u:.6f}")
    print(f"  reflect64         (padded  {tuple(p0.shape[-2:])}): mean {m_r:.6f} std {s_r:.6f}"
          f"  -> dmean {m_r - m_u:+.6f} dstd {s_r - s_u:+.6f}")
    print(f"  {100 * frac:.1f}% of the reflect-padded tensor is fabricated pixels, and they enter the global")
    print("  mean/std -> reflect rescales the whole frame's input, it is not a border-only change.")
    del p0, p1; torch.cuda.empty_cache()

    # (4) The test that CAN rank padding modes: a ground-truth hold-out.
    #     Interpolate frames 0 and 2 at t=0.5 and score against the REAL frame 1.
    if len(src) >= 3:
        g0, g2, gt = load(src[0]), load(src[2]), load(src[1])
        print("HOLD-OUT vs ground truth (frames 0,2 -> t=0.5, scored against the real frame 1).")
        print("This has a true reference, so it CAN rank padding modes (lower = better):")
        for mode, margin in MODES:
            set_mode(A, mode, margin)
            edge_report(A.interpolate(g0, g2, [0.5])[0], tag(mode, margin), gt, "GT")
        del g0, g2, gt
        torch.cuda.empty_cache()
    else:
        print("HOLD-OUT skipped: need >= 3 input frames")
    set_mode(A, DEFAULT_MODE, DEFAULT_MARGIN)
    del outs, blend
    torch.cuda.empty_cache()
del A
torch.cuda.empty_cache()
B = ad.Adapter(dev, pyr_level=8)
o8 = run(B, f"pyr8 {tag(B.pad_mode, B.margin)} fp32", reps=2)
print("antialias CPU fallback calls so far:", ad.AA_CPU_FALLBACKS["calls"], "(expected 2 per pyr8 forward)")
print(f"pyr8 vs pyr7 at t=0.5: mean abs diff {float((o8 - o32).abs().mean()):.5f}")
order(B, "pyr8")
EOF
echo "== pre-check exit: $PRE_FAIL $(date -Is)"
fi

echo "== harness $(date -Is): $IN -> $OUT  adapter_args=[$ADAPTER_ARGS]"
aargs=(); for kv in $ADAPTER_ARGS; do aargs+=(--adapter_arg "$kv"); done
cd "$MVFI_ROOT/scripts"
t0=$(date +%s)
env -u SLURM_PROCID python vfi_harness.py --adapter "$ADAPTER" --input "$IN" --out_dir "$OUT" \
  --temporal_sf 10 --fps 25 "${aargs[@]}"
echo "== harness wall time $(( $(date +%s) - t0 )) s"
echo "peak nvidia-smi memory.used: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{print m" MiB"}' "$SMI")"
python -c "import json,sys; r=json.load(open(sys.argv[1])); m=r['model']; print(json.dumps({k: m[k] for k in ('repo_commit','weights','settings','raw_output_stats','runtime')}, indent=1))" "$OUT/logs/run_report.json"

echo "== checks $(date -Is)"
python "$MVFI_ROOT/scripts/envs/check_bimvfi_smoke.py" "$OUT" "$IN" --sf 10 --expect 31
echo "== done $(date -Is) pre-check fail=$PRE_FAIL"
exit $PRE_FAIL
