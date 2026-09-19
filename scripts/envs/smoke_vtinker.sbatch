#!/bin/bash
# VTinker smoke test on the A100: CuPy kernel JIT check, harness run on input/synthetic4k (x10), then checks:
# frame count, dtype/shape, genuine frames bit-exact, no non-finite network output, output range,
# t=0.1 closer to img0 / t=0.9 closer to img1 (mean abs diff), flow-cache == upstream per-t path,
# bottom-edge padding bleed (step 5: pad_mode=replicate vs the upstream constant 0.5),
# plus informational fp32 vs upstream-fp16 and TF32 comparisons. Exit code != 0 if a required check fails.
#   sbatch scripts/envs/smoke_vtinker.sbatch            # 4 frames -> 31
#   MAXF=2 sbatch scripts/envs/smoke_vtinker.sbatch     # quick: 2 frames -> 11
#SBATCH --job-name=vfi_smoke_vtinker
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
export PYTHONDONTWRITEBYTECODE=1   # keep src/VTinker free of __pycache__

MAXF=${MAXF:-4}
IN=$MVFI_ROOT/input/synthetic4k
OUT=$MVFI_ROOT/output/_checks/vtinker_smoke_$SLURM_JOB_ID
ADAPTER=$MVFI_ROOT/scripts/adapters/vtinker.py
REPO=$MVFI_ROOT/src/VTinker
GPU_CSV=$MVFI_ROOT/logs/gpu_usage_vtinker_smoke_$SLURM_JOB_ID.csv
export IN OUT ADAPTER REPO MAXF

echo "== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MAXF=$MAXF"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
echo "python $(command -v python); CUDA_PATH=${CUDA_PATH:-unset}"
mvfi_start_gpu_monitor "$GPU_CSV"

echo "== 1. CuPy NVRTC JIT of the upstream kernels"
python - <<'EOF'
import os, sys, torch
sys.path.insert(0, os.environ["REPO"])
import cupy
from cupy.cuda import nvrtc
print("torch", torch.__version__, torch.version.cuda, "| cupy", cupy.__version__, "nvrtc", nvrtc.getVersion(),
      "| device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
from core.models.softsplat import softsplat
from core.utils import correlation
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.rand(1, 3, 64, 80, device="cuda", generator=g)
flow = torch.zeros(1, 2, 64, 80, device="cuda"); flow[:, 0] = 3.0; flow[:, 1] = -2.0
y = softsplat.FunctionSoftsplat(tenInput=x, tenFlow=flow.contiguous(), tenMetric=None, strType="average")
err = (y[..., :-2, 3:] - x[..., 2:, :-3]).abs().max().item()
print("softsplat integer-shift max err", err); assert err < 1e-6
f = torch.randn(1, 64, 30, 40, device="cuda", generator=g)
c = correlation.FunctionCorrelation(tenFirst=f, tenSecond=f)
err = (c[:, 40] - (f * f).mean(1)).abs().max().item()
print("correlation shape", tuple(c.shape), "centre-channel max err", err); assert c.shape[1] == 81 and err < 1e-4
f2 = torch.roll(f, shifts=(0, 2), dims=(2, 3))   # f2(x) = f(x-2): displacement +2 in x -> channel 4*9+(4+2)=42
c2 = correlation.FunctionCorrelation(tenFirst=f, tenSecond=f2)
best = int(c2[0, :, 10:-10, 10:-10].mean((1, 2)).argmax())
print("correlation peak channel for +2 px x-shift:", best); assert best == 42
print("KERNEL JIT OK")
EOF

echo "== 2. harness run ($MAXF frames, x10)"
t0=$(date +%s)
python "$MVFI_ROOT/scripts/vfi_harness.py" --adapter "$ADAPTER" --input "$IN" --out_dir "$OUT" \
  --temporal_sf 10 --fps 25 --max_input_frames "$MAXF"
echo "harness wall time: $(( $(date +%s) - t0 )) s"
PEAK_SMI=$(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{print m+0}' "$GPU_CSV")
echo "peak nvidia-smi memory.used: $PEAK_SMI MiB"
export PEAK_SMI

echo "== 3. output checks"
python - <<'EOF'
import csv, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ["MVFI_ROOT"] + "/scripts")
from frameio import IMG_EXTS, natural_key, read_image

out, src_dir, maxf = Path(os.environ["OUT"]), Path(os.environ["IN"]), int(os.environ["MAXF"])
sf = 10
src = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: natural_key(p.name))[:maxf]
results, failed = [], []

def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        failed.append(name)

expected = (len(src) - 1) * sf + 1
frames = sorted(out.glob("frame_*.tif"))
check("frame count", len(frames) == expected, f"{len(frames)} files, expected {expected}")
rep = json.loads((out / "logs" / "run_report.json").read_text())
check("run_report output_frames", rep["output_frames"] == expected, str(rep["output_frames"]))
st = rep["model"]["stats"]
check("no non-finite network output", st["nonfinite_values"] == 0, f"nonfinite_values={st['nonfinite_values']}")
check("raw (pre-clamp) output range sane", -0.25 < st["raw_min"] and st["raw_max"] < 1.25,
      f"raw_min={st['raw_min']:.5f} raw_max={st['raw_max']:.5f} max clipped fraction={st['clipped_fraction_max']:.2e}")
check("generated frame count in adapter", st["frames"] == (len(src) - 1) * (sf - 1), str(st["frames"]))

srcs = [read_image(p).astype(np.float64) for p in src]
H, W = srcs[0].shape[:2]
fm = {int(r["out_index"]): r for r in csv.DictReader(open(out / "frame_map.csv"))}
per_frame, all_ok_shape, levels = [], True, set()
cache = {}
def frame(k):
    if k not in cache:
        cache[k] = read_image(out / f"frame_{k:06d}.tif")
    return cache[k]
for k in range(expected):
    a = frame(k)
    ok_shape = a.dtype == np.uint16 and a.shape == (H, W, 3)
    all_ok_shape &= ok_shape
    af = a.astype(np.float64)
    d = {"k": k, "kind": fm[k]["kind"], "t": float(fm[k]["t_frac"]), "min": int(a.min()), "max": int(a.max()),
         "mean": float(af.mean()), "std": float(af.std()), "finite": bool(np.isfinite(af).all())}
    if k % sf == 0:
        d["bit_exact_vs_source"] = bool(np.array_equal(a, read_image(src[k // sf])))
    else:
        j = k // sf
        d["mad_to_img0"] = float(np.abs(af - srcs[j]).mean() / 65535)
        d["mad_to_img1"] = float(np.abs(af - srcs[j + 1]).mean() / 65535)
        if len(levels) < 70000:
            levels.update(np.unique(a[::7, ::7]).tolist())
    per_frame.append(d)
    if k % sf == 0 and k > 0:   # free memory of the finished pair
        for kk in [x for x in cache if x < k]:
            del cache[kk]
check("dtype uint16 and shape (H, W, 3) for all frames", all_ok_shape, f"{W}x{H}")
check("all frames finite", all(d["finite"] for d in per_frame))
gen = [d for d in per_frame if d["kind"] == "generated"]
gm = [d for d in per_frame if d["kind"] == "genuine"]
check("genuine frames bit-exact", all(d["bit_exact_vs_source"] for d in gm), f"{len(gm)} frames")
check("generated frames non-degenerate", all(d["std"] > 100 and 0 <= d["min"] < d["max"] <= 65535 for d in gen),
      f"min std {min(d['std'] for d in gen):.1f}; range [{min(d['min'] for d in gen)}, {max(d['max'] for d in gen)}]")
smean = np.mean([s.mean() for s in srcs])
check("generated mean close to source mean", all(abs(d["mean"] - smean) < 0.03 * 65535 for d in gen),
      f"source mean {smean:.0f}; generated means {min(d['mean'] for d in gen):.0f}..{max(d['mean'] for d in gen):.0f}")
check("output not 8-bit quantised (many distinct 16-bit levels, not multiples of 257)",
      len(levels) > 1000 and any(v % 257 for v in levels), f"{len(levels)} distinct levels in subsample")
for j in range(len(src) - 1):
    d1 = next(d for d in gen if d["k"] == j * sf + 1)
    d9 = next(d for d in gen if d["k"] == j * sf + 9)
    base = float(np.abs(srcs[j] - srcs[j + 1]).mean() / 65535)
    check(f"pair {j}: t=0.1 closer to img0", d1["mad_to_img0"] < d1["mad_to_img1"],
          f"MAD0={d1['mad_to_img0']:.5f} MAD1={d1['mad_to_img1']:.5f} (img0-img1 MAD {base:.5f})")
    check(f"pair {j}: t=0.9 closer to img1", d9["mad_to_img1"] < d9["mad_to_img0"],
          f"MAD0={d9['mad_to_img0']:.5f} MAD1={d9['mad_to_img1']:.5f}")
    prof = [(d["t"], round(d["mad_to_img0"], 5), round(d["mad_to_img1"], 5)) for d in gen if d["k"] // sf == j]
    print(f"   pair {j} MAD profile (t, to img0, to img1): {prof}")
    m0 = [p[1] for p in prof]; m1 = [p[2] for p in prof]
    mono = all(b >= a - 1e-4 for a, b in zip(m0, m0[1:])) and all(b <= a + 1e-4 for a, b in zip(m1, m1[1:]))
    print(f"   pair {j} MAD monotonic in t (informational): {mono}")

summary = {"failed": failed, "checks": results, "per_frame": per_frame,
           "timing_s": rep["timing_s"], "gpu": rep["gpu"], "peak_nvidia_smi_mib": int(os.environ.get("PEAK_SMI", 0)),
           "adapter_stats": st, "adapter_settings": rep["model"]["settings"]}
(out / "logs" / "smoke_checks.json").write_text(json.dumps(summary, indent=1))
print("timing:", json.dumps(rep["timing_s"]), "gpu:", json.dumps(rep["gpu"]))
print("SMOKE CHECKS", "FAILED: " + ", ".join(failed) if failed else "PASSED")
sys.exit(1 if failed else 0)
EOF

echo "== 4. determinism + flow-cache equivalence, TF32 and upstream-fp16 comparisons (pair 0)"
python - <<'EOF'
import os
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.environ["MVFI_ROOT"] + "/scripts")
from frameio import IMG_EXTS, natural_key, read_image
src_dir, out = Path(os.environ["IN"]), Path(os.environ["OUT"])
src = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: natural_key(p.name))
img = [torch.from_numpy(read_image(p).astype(np.float32) / 65535.0).permute(2, 0, 1).contiguous().cuda() for p in src[:2]]
spec = importlib.util.spec_from_file_location("vt", os.environ["ADAPTER"]); vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)
ad = vt.Adapter(torch.device("cuda"))
ts = [0.1, 0.5, 0.9]
L = 65535.0
res = {}

def tm(fn):
    torch.cuda.synchronize(); t0 = time.time(); r = fn(); torch.cuda.synchronize(); return r, time.time() - t0

def cmp(name, xs, ys):
    d = torch.stack([(x - y).abs() for x, y in zip(xs, ys)])
    r = {"max_abs_16bit": float(d.max()) * L, "mean_abs_16bit": float(d.mean()) * L,
         "frac_px_diff_gt_1_8bit_level": float((d.amax(1) > 257 / L).float().mean()),
         "frac_px_diff_gt_0.5_16bit": float((d.amax(1) > 0.5 / L).float().mean())}
    res[name] = r
    print(f"{name}: max {r['max_abs_16bit']:.1f}/65535, mean {r['mean_abs_16bit']:.3f}/65535, pixels > 1/255: "
          f"{100 * r['frac_px_diff_gt_1_8bit_level']:.4f} %, pixels > 0.5/65535: {100 * r['frac_px_diff_gt_0.5_16bit']:.3f} %")
    return r

# flows: cached (estimated once, scaled per t) vs upstream get_flow(t)
# (padded the same way the adapter does by default: replicate, see step 5)
x0 = torch.nn.functional.pad(img[0][None], (0, 0, 0, 16), mode="replicate")
x1 = torch.nn.functional.pad(img[1][None], (0, 0, 0, 16), mode="replicate")
m = ad.model
half = m.get_flow(x0, x1, 0.5, 5, 1)
fd = 0.0
for t in ts:
    cached = torch.cat([half[:, :2] * 2 * t, half[:, 2:4] * 2 * (1 - t)], 1)
    fd = max(fd, float((cached - m.get_flow(x0, x1, t, 5, 1)).abs().max()))
g1, g2 = m.get_flow(x0, x1, 0.3, 5, 1), m.get_flow(x0, x1, 0.3, 5, 1)
res["flow_cache_vs_upstream_max_abs_px"] = fd
res["flow_run_to_run_max_abs_px"] = float((g1 - g2).abs().max())
res["flow_abs_max_px"] = float(half.abs().max() * 2)
print(f"flow: cached vs upstream max abs {fd:.3e} px; run-to-run {res['flow_run_to_run_max_abs_px']:.3e} px; "
      f"|flow| max {res['flow_abs_max_px']:.1f} px")

# synthesis with the SAME flow tensor twice: run-to-run nondeterminism of TBVFI (softsplat atomicAdd order etc.)
flow = torch.cat([half[:, :2] * 2 * 0.5, half[:, 2:4] * 2 * 0.5], 1)
s1 = m.main_model.inference(x0, x1, flow).clamp(0, 1)
s2 = m.main_model.inference(x0, x1, flow).clamp(0, 1)
cmp("synthesis_same_flow_run_to_run_t05", [s1[0]], [s2[0]])

a, dt_cache = tm(lambda: ad.interpolate(img[0], img[1], ts))
a2, _ = tm(lambda: ad.interpolate(img[0], img[1], ts))
ad.cache_flow = False
b, dt_upstream = tm(lambda: ad.interpolate(img[0], img[1], ts))
ad.cache_flow = True
rr = cmp("adapter_run_to_run", a, a2)
cu = cmp("flow_cache_vs_upstream_per_t", a, b)
res["time_3t_cache_s"], res["time_3t_upstream_s"] = dt_cache, dt_upstream
print(f"3 t: cache {dt_cache:.2f} s, upstream per-t {dt_upstream:.2f} s")
# Equivalent = same flows (to float rounding) and output differences of the same size as run-to-run noise.
ok = fd < 1e-3 and cu["mean_abs_16bit"] <= max(3 * rr["mean_abs_16bit"], 0.5) \
     and cu["frac_px_diff_gt_1_8bit_level"] <= max(3 * rr["frac_px_diff_gt_1_8bit_level"], 1e-4)

torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = True
c, dt_tf32 = tm(lambda: ad.interpolate(img[0], img[1], ts))
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
cmp("tf32_vs_fp32", a, c)
res["time_3t_tf32_s"] = dt_tf32
print(f"3 t: TF32 {dt_tf32:.2f} s")
try:  # upstream default: VTinker_half.py + model.half(), t = 0.5
    import core.models.VTinker_half as vh
    net = vh.modelVFI()
    sd = torch.load(vt.WEIGHTS, map_location="cpu", weights_only=True)
    net.load_state_dict({k: sd["module." + k] for k in net.state_dict()}); del sd
    net = net.half().eval().cuda()
    h16, dt16 = tm(lambda: net.inference(x0.half(), x1.half(), 0.5, 5, 1))
    h16 = h16.float().clamp(0, 1)[0, :, :2160, :3840]
    cmp("upstream_fp16_vs_fp32_t05", [h16], [a[1]])
    res["time_1t_fp16_upstream_incl_flow_s"] = dt16
    print(f"fp16 1 t incl. flow: {dt16:.2f} s")
except Exception as e:  # informational only
    res["fp16_error"] = repr(e); print("fp16 comparison skipped:", repr(e))
res["equivalence_pass"] = bool(ok)
(out / "logs" / "smoke_equivalence.json").write_text(json.dumps(res, indent=1))
print("EQUIVALENCE", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF
echo "== 5. bottom-edge padding: delivered output (pad_mode=replicate) vs upstream constant 0.5"
# 2160 is not a multiple of the 128 pad divisor, so 16 rows are appended below the picture and the forward
# warp can drag them back inside the crop. Statistic: signed per-row mean deviation from the 0.5*(img0+img1)
# blend, in 16-bit levels. With the upstream constant 0.5 the last delivered row stepped +1338 levels above
# its neighbour (smoke 319365); replicate must not.
python - <<'EOF'
import os
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, tifffile, torch
sys.path.insert(0, os.environ["MVFI_ROOT"] + "/scripts")
from frameio import IMG_EXTS, natural_key, read_image

out, src_dir, maxf = Path(os.environ["OUT"]), Path(os.environ["IN"]), int(os.environ["MAXF"])
sf, NR, NC = 10, 40, 40
src = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: natural_key(p.name))[:maxf]
res, failed = {}, []

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        failed.append(name)

# --- A. the delivered frames (written by the harness with adapter defaults)
s = [tifffile.memmap(str(p), mode="r") for p in src]
H, W = s[0].shape[:2]
rows, cols = [], []
for j in range(len(src) - 1):
    b = 0.5 * (s[j][-NR:].astype(np.float64) + s[j + 1][-NR:].astype(np.float64))
    r = 0.5 * (s[j][:, -NC:].astype(np.float64) + s[j + 1][:, -NC:].astype(np.float64))
    for i in range(1, sf):
        a = tifffile.memmap(str(out / f"frame_{j * sf + i:06d}.tif"), mode="r")
        rows.append((a[-NR:].astype(np.float64) - b).mean(axis=(1, 2)))
        cols.append((a[:, -NC:].astype(np.float64) - r).mean(axis=(0, 2)))
        del a
row_dev, col_dev = np.stack(rows).mean(0), np.stack(cols).mean(0)
row_step, col_step = float(row_dev[-1] - row_dev[-2]), float(col_dev[-1] - col_dev[-2])
band = float(np.abs(row_dev[-11:-1]).max())
print(f"delivered, mean over {len(rows)} generated frames, last 10 rows (16-bit levels): "
      + ", ".join(f"{H - 10 + i}:{v:+.1f}" for i, v in enumerate(row_dev[-10:])))
print(f"delivered, last 10 columns (control, width needs no padding): "
      + ", ".join(f"{W - 10 + i}:{v:+.1f}" for i, v in enumerate(col_dev[-10:])))
res["delivered"] = {"row_dev_last20": {str(H - 20 + i): float(v) for i, v in enumerate(row_dev[-20:])},
                    "col_dev_last20": {str(W - 20 + i): float(v) for i, v in enumerate(col_dev[-20:])},
                    "last_row_step": row_step, "last_col_step": col_step,
                    "neighbour_band_abs_max": band, "frames": len(rows)}
check("last delivered row has no padding step", abs(row_step) < 400,
      f"row {H - 1} - row {H - 2} = {row_step:+.1f} levels (constant 0.5 gave +1337.9)")
check("last delivered row within the neighbouring-row band", abs(float(row_dev[-1])) < 1.5 * band + 50,
      f"row {H - 1} = {row_dev[-1]:+.1f}, |dev| max over rows {H - 11}..{H - 2} = {band:.1f}")

# --- B. same pair, both pad modes, on the GPU
spec = importlib.util.spec_from_file_location("vt", os.environ["ADAPTER"]); vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)
ad = vt.Adapter(torch.device("cuda"))
check("default pad_mode is replicate", ad.pad_mode == "replicate", ad.pad_mode)
img = [torch.from_numpy(read_image(p).astype(np.float32) / 65535.0).permute(2, 0, 1).contiguous().cuda()
       for p in src[:2]]
b16 = 0.5 * (read_image(src[0]).astype(np.float64) + read_image(src[1]).astype(np.float64))
ts = [0.1, 0.5, 0.9]

def run(mode, pad_value=0.5):
    ad.pad_mode, ad.pad_value = mode, pad_value
    o = ad.interpolate(img[0], img[1], ts)
    u = [(x.clamp(0, 1) * 65535.0).round().permute(1, 2, 0).cpu().numpy().astype(np.float64) for x in o]
    return u, [float((x[-NR:] - b16[-NR:]).mean(axis=(1, 2))[-1]) for x in u]

u_rep, dev_rep = run("replicate")
u_rep2, _ = run("replicate")        # run-to-run baseline: cuDNN kernel selection is not deterministic
u_con, dev_con = run("constant", 0.5)
u_con0, dev_con0 = run("constant", 0.0)
u_repB, _ = run("replicate", 0.0)   # pad_value must be inert in replicate mode
ad.pad_mode, ad.pad_value = "replicate", 0.5
for name, dv in (("replicate", dev_rep), ("constant 0.5", dev_con), ("constant 0.0", dev_con0)):
    print(f"pair 0, row {H - 1} deviation from blend, {name:<13}: "
          + ", ".join(f"t={t}: {v:+.1f}" for t, v in zip(ts, dv)))
res["pair0_last_row_dev"] = {"ts": ts, "replicate": dev_rep, "constant_0.5": dev_con, "constant_0.0": dev_con0}
check("replicate removes most of the t=0.9 bottom-row bias", abs(dev_rep[-1]) < 0.34 * abs(dev_con[-1]),
      f"replicate {dev_rep[-1]:+.1f} vs constant {dev_con[-1]:+.1f} levels")
# mechanism: with constant padding the pad VALUE lands in the delivered last row. It also perturbs the
# interior a little (16 pad rows are 1/4 of the coarsest 68-row pyramid level, so the flow field moves
# slightly everywhere), but the delivered last row takes two orders of magnitude more.
def rowstat(a, b, r):
    d = np.abs(a[r] - b[r])
    return float(d.mean()), float(d.max())

last_mean, last_max = rowstat(u_con[-1], u_con0[-1], -1)
mid_mean, mid_max = rowstat(u_con[-1], u_con0[-1], H // 2)
print(f"constant pad_value 0.5 vs 0.0 at t=0.9: row {H - 1} mean |diff| {last_mean:.1f} (max {last_max:.0f}), "
      f"row {H // 2} mean |diff| {mid_mean:.1f} (max {mid_max:.0f}) 16-bit levels")
res["pad_value_probe_t09"] = {"last_row_mean_abs_16bit": last_mean, "last_row_max_abs_16bit": last_max,
                              "middle_row_mean_abs_16bit": mid_mean, "middle_row_max_abs_16bit": mid_max}
check("the constant pad value lands in the last delivered row", last_mean > 200 and last_mean > 10 * mid_mean,
      f"row {H - 1} mean |diff| {last_mean:.1f} vs row {H // 2} {mid_mean:.1f} levels")

def dstat(xs, ys):
    d = np.stack([np.abs(x - y) for x, y in zip(xs, ys)])
    return float(d.mean()), float(d.max())

n_mean, n_max = dstat(u_rep, u_rep2)    # same settings twice
p_mean, p_max = dstat(u_rep, u_repB)    # pad_value 0.5 vs 0.0, both replicate
print(f"replicate, pad_value 0.5 vs 0.0: mean {p_mean:.4f}/65535 max {p_max:.0f}; "
      f"run-to-run noise at identical settings: mean {n_mean:.4f}/65535 max {n_max:.0f}")
res["pad_value_inert_replicate"] = {"pad_value_mean": p_mean, "pad_value_max": p_max,
                                    "run_to_run_mean": n_mean, "run_to_run_max": n_max}
check("pad_value is inert when pad_mode=replicate (differs no more than run-to-run noise)",
      p_mean <= max(3 * n_mean, 0.05) and p_max <= max(3 * n_max, 100),
      f"mean {p_mean:.4f} vs noise {n_mean:.4f}, max {p_max:.0f} vs noise {n_max:.0f}")
res["failed"] = failed
(out / "logs" / "smoke_padding.json").write_text(json.dumps(res, indent=1))
print("PADDING CHECKS", "FAILED: " + ", ".join(failed) if failed else "PASSED")
sys.exit(1 if failed else 0)
EOF
echo "== SMOKE PASSED $(date -Is)  output: $OUT"
