"""VTinker adapter for vfi_harness.py (Wu et al., AAAI 2026, arXiv 2511.16124).

Model: official fp32 network src/VTinker/core/models/VTinker.py (is_half = False; the upstream tools use the
byte-identical VTinker_half.py + model.half()). Weights: models/VTinker/VTinker.pkl (HF wcy1234567/VTinker).

Precision path (no quantisation anywhere):
  harness float32 RGB [0,1] -> pad to a multiple of 2**(flow_deep+2) (replicate by default; upstream uses
     constant 0.5, which softsplat drags back inside the crop as a bright bottom line -- see pad_mode)
  -> fp32 network (TF32 disabled by default; the CuPy softsplat/correlation kernels are float32)
  -> crop -> clamp to [0,1] (as upstream) -> float32 tensors back to the harness (which rounds once to uint16).
Channel order: RGB (as upstream tools/test_single_case.py; training used random channel reversal).

How t is used (upstream modelVFI.get_flow): bidirectional flows F01, F10 are estimated once per pair,
independent of t, then scaled linearly: flow_0t = F01 * t, flow_1t = F10 * (1 - t); both frames are
forward-warped (softsplat 'average') and the synthesis net (TBVFI, no t input) builds the frame.
Upstream recomputes the identical flows for every t; with cache_flow=true (default) this adapter computes
them once per pair and only re-runs TBVFI per t (same maths; differences are floating-point rounding only).
The released model was trained only at t = 0.5, so the t = 0.1..0.9 sweep is an extrapolation. Measured
against REAL frames (hold-out mode, job 319409 on input/synthetic4k, the only sequence available so far):
VTinker beats a cross-dissolve at every reachable ground-truth t, by +2.15 dB at t = 0.5 (input gap 2) but
only +0.80 / +0.83 dB at t = 1/3 and 2/3 (input gap 3, i.e. more motion as well as off-centre t). The full
t = 0.1..0.9 PSNR curve needs a real shot of >= 31 frames:
    SHOT=<shot> VARIANT=holdout SF=10 STRIDE=10 sbatch scripts/envs/run_vtinker.sbatch
    -> output/<shot>/vtinker/holdout/logs/holdout_metrics.csv (PSNR per output index = per t)
Run that before committing a shot: a U-shape with its minimum at t = 0.5 is fine, a collapse at t = 0.1/0.9
means the sweep is not usable and VTinker should be restricted to t = 0.5 (SF=2) or dropped.

adapter_args (all optional):
  flow_deep=int, skip_num=int   pyramid depth / skip level (default: upstream rule from max(H, W)/1024:
                                >=3 -> 5,1 ; >=1.5 -> 4,0 ; else 3,0)
  pad_mode=replicate            replicate | reflect | constant. 2160 is not a multiple of the 128 pad divisor,
                                so 16 rows are appended below the picture. With the upstream 'constant' 0.5 the
                                forward warp (softsplat) pulls that flat grey back inside the crop: measured on
                                smoke 319365, the last delivered row is biased by +1229/65535 on average and up
                                to +3059 at t=0.9, while the unpadded right edge shows no such step -- a 1 px
                                flickering line, since genuine frames are pass-throughs. 'replicate' splats a
                                copy of the last real row instead. Keep 'constant' only for upstream-fidelity
                                comparisons.
  pad_value=0.5                 constant used for padding when pad_mode=constant (upstream: 0.5); ignored otherwise
  cache_flow=true               estimate flows once per pair (false = call model.inference per t, as upstream)
  allow_tf32=false              allow TensorFloat-32 convolutions/matmul (faster, not true fp32)
  cudnn_benchmark=false
  clamp=true                    clamp output to [0, 1] as upstream does
  channel_order=rgb             rgb | bgr (upstream test_video.py feeds BGR)
  nonfinite=raise               raise | warn when the network returns NaN/Inf
"""
import ctypes
import glob
import hashlib
import importlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "VTinker"
WEIGHTS = ROOT / "models" / "VTinker" / "VTinker.pkl"
REPO_URL = "https://github.com/Wucy0519/VTinker"
REPO_COMMIT_EXPECTED = "6688e06338e39b43764ef90455d4430fbaa563b4"
HF_REVISION = "f9f0f47e6fb5d35e73f38547e6a45521e41f0777"
WEIGHTS_URL = f"https://huggingface.co/wcy1234567/VTinker/resolve/{HF_REVISION}/VTinker.pkl"
WEIGHTS_SHA256 = "8ed128023af03332e8c2915b9b735aaff5c70667c054e0237ed032c47c43aa17"
WEIGHTS_BYTES = 242743383


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _git_commit(repo):
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (repo / ".git" / ref).exists():
            return (repo / ".git" / ref).read_text().strip()
        for line in (repo / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def _preload_cuda_libs():
    """CuPy 12.3 needs CUDA 12 runtime libs. The env's activate.d hook puts torch's nvidia-*-cu12 wheel libs on
    LD_LIBRARY_PATH; if the env python is run without `conda activate`, preload them here instead."""
    if "nvidia" in os.environ.get("LD_LIBRARY_PATH", ""):
        return
    import site
    loaded = []
    for sp in site.getsitepackages():
        for pat in ("cuda_runtime/lib/libcudart.so.12", "cuda_nvrtc/lib/libnvrtc.so.12",
                    "cuda_nvrtc/lib/libnvrtc-builtins.so.*", "nvjitlink/lib/libnvJitLink.so.12",
                    "cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12",
                    "cusparse/lib/libcusparse.so.12", "curand/lib/libcurand.so.10",
                    "cufft/lib/libcufft.so.11", "cusolver/lib/libcusolver.so.11"):
            for p in sorted(glob.glob(os.path.join(sp, "nvidia", pat))):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(p)
                except OSError:
                    pass
    if loaded:
        print(f"[vtinker] preloaded {len(loaded)} CUDA libs from nvidia pip wheels", flush=True)


def _bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class Adapter:
    def __init__(self, device, flow_deep=None, skip_num=None, pad_mode="replicate", pad_value=0.5,
                 cache_flow=True, allow_tf32=False, cudnn_benchmark=False, clamp=True, channel_order="rgb",
                 nonfinite="raise", verify_weights=True):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("VTinker's softsplat/correlation kernels are CUDA-only (CuPy); no CPU path")
        self.flow_deep = None if flow_deep is None else int(flow_deep)
        self.skip_num = None if skip_num is None else int(skip_num)
        if (self.flow_deep is None) != (self.skip_num is None):
            raise ValueError("give both flow_deep and skip_num, or neither")
        self.pad_mode = str(pad_mode).lower()
        if self.pad_mode not in ("replicate", "reflect", "constant"):
            raise ValueError("pad_mode must be replicate, reflect or constant")
        self.pad_value = float(pad_value)
        self.cache_flow = _bool(cache_flow)
        self.allow_tf32 = _bool(allow_tf32)
        self.clamp = _bool(clamp)
        self.channel_order = str(channel_order).lower()
        if self.channel_order not in ("rgb", "bgr"):
            raise ValueError("channel_order must be rgb or bgr")
        self.nonfinite = str(nonfinite).lower()
        if self.nonfinite not in ("raise", "warn"):
            raise ValueError("nonfinite must be raise or warn")

        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.benchmark = _bool(cudnn_benchmark)
        torch.set_grad_enabled(False)

        # --- weights (verify before use)
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"{WEIGHTS} missing: run scripts/envs/build_vtinker.sbatch")
        self.weights_sha256 = _sha256(WEIGHTS) if _bool(verify_weights) else None
        if self.weights_sha256 is not None and self.weights_sha256 != WEIGHTS_SHA256:
            raise RuntimeError(f"{WEIGHTS} sha256 {self.weights_sha256} != expected {WEIGHTS_SHA256}")

        # --- model code (namespace packages core/, core/models/: import from the repo root)
        _preload_cuda_libs()
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        mod = importlib.import_module("core.models.VTinker")
        if not Path(mod.__file__).resolve().is_relative_to(REPO.resolve()):
            raise ImportError(f"core.models.VTinker resolved to {mod.__file__}, not inside {REPO}")
        if mod.is_half is not False:
            raise RuntimeError("expected the fp32 model file (is_half = False)")
        import cupy  # noqa: F401  (imported by the model already; recorded below)
        from cupy.cuda import nvrtc
        self._versions = {"cupy": cupy.__version__, "nvrtc": ".".join(map(str, nvrtc.getVersion())),
                          "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                          "cudnn": torch.backends.cudnn.version()}

        net = mod.modelVFI()
        sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        own = net.state_dict()
        missing = sorted(set(own) - set(sd))
        if missing:
            raise RuntimeError(f"checkpoint lacks {len(missing)} model tensors, e.g. {missing[:5]}")
        self.ckpt_unused = sorted(set(sd) - set(own))
        self.ckpt_dtypes = sorted({str(v.dtype) for v in sd.values()})
        # load_state_dict copies into the fp32 parameters (upcasts if the checkpoint were fp16)
        net.load_state_dict({k: sd[k] for k in own}, strict=True)
        del sd
        self.model = net.float().eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        assert all(p.dtype == torch.float32 for p in self.model.parameters())

        self.stats = {"pairs": 0, "frames": 0, "raw_min": float("inf"), "raw_max": float("-inf"),
                      "clipped_fraction_max": 0.0, "nonfinite_values": 0, "flow_abs_max_px": 0.0,
                      "flow_s": 0.0, "synth_s": 0.0}
        self._settings = None
        print(f"[vtinker] model loaded on {self.device}: fp32, tf32={self.allow_tf32}, "
              f"cache_flow={self.cache_flow}, versions={self._versions}", flush=True)

    # upstream tools/test_single_case.py:get_deep_skip (computed on the unpadded size)
    def _deep_skip(self, h, w):
        if self.flow_deep is not None:
            return self.flow_deep, self.skip_num
        lenss = max(h, w) / 1024
        if lenss >= 3:
            return 5, 1
        if lenss >= 1.5:
            return 4, 0
        return 3, 0

    def _check(self, out, h, w):
        n_bad = int((~torch.isfinite(out)).sum())
        if n_bad:
            self.stats["nonfinite_values"] += n_bad
            msg = f"[vtinker] network output has {n_bad} non-finite values"
            if self.nonfinite == "raise":
                raise FloatingPointError(msg)
            print("WARNING " + msg, flush=True)
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        mn, mx = float(out.min()), float(out.max())
        self.stats["raw_min"] = min(self.stats["raw_min"], mn)
        self.stats["raw_max"] = max(self.stats["raw_max"], mx)
        clipped = float(((out < 0) | (out > 1)).float().mean())
        self.stats["clipped_fraction_max"] = max(self.stats["clipped_fraction_max"], clipped)
        return out

    @torch.no_grad()
    def interpolate(self, img0, img1, ts):
        x0 = img0.to(self.device, torch.float32, non_blocking=True).unsqueeze(0)
        x1 = img1.to(self.device, torch.float32, non_blocking=True).unsqueeze(0)
        if x0.shape != x1.shape or x0.shape[1] != 3:
            raise ValueError(f"bad input shapes {tuple(x0.shape)} {tuple(x1.shape)}")
        _, _, h, w = x0.shape
        deep, skip = self._deep_skip(h, w)
        div = 2 ** (deep + 2)
        ph, pw = -(-h // div) * div, -(-w // div) * div
        if (ph, pw) != (h, w):
            pad = (0, pw - w, 0, ph - h)
            if self.pad_mode == "constant":
                x0 = F.pad(x0, pad, "constant", self.pad_value)
                x1 = F.pad(x1, pad, "constant", self.pad_value)
            else:  # replicate/reflect: no foreign value for softsplat to drag back into the crop
                x0 = F.pad(x0, pad, mode=self.pad_mode)
                x1 = F.pad(x1, pad, mode=self.pad_mode)
        if self.channel_order == "bgr":
            x0, x1 = x0.flip(1), x1.flip(1)
        x0, x1 = x0.contiguous(), x1.contiguous()
        self._settings = {"flow_deep": deep, "skip_num": skip, "pad_divisor": div,
                          "padded_hw": [ph, pw], "input_hw": [h, w],
                          "pad_rows_added": ph - h, "pad_cols_added": pw - w}

        outs = []
        if self.cache_flow:
            torch.cuda.synchronize(self.device)
            t0 = time.time()
            # get_flow returns cat([F01 * t, F10 * (1 - t)]) / 4; at t = 0.5 the x2 below is exact.
            half = self.model.get_flow(x0, x1, 0.5, deep, skip)
            f01, f10 = half[:, :2] * 2.0, half[:, 2:4] * 2.0
            self.stats["flow_abs_max_px"] = max(self.stats["flow_abs_max_px"],
                                                float(torch.cat([f01, f10], 1).abs().max()))
            torch.cuda.synchronize(self.device)
            self.stats["flow_s"] += time.time() - t0
        for t in ts:
            t = float(t)
            if not 0.0 < t < 1.0:
                raise ValueError(f"t={t} outside (0, 1)")
            t0 = time.time()
            if self.cache_flow:
                flow = torch.cat([f01 * t, f10 * (1.0 - t)], dim=1)
                out = self.model.main_model.inference(x0, x1, flow)
            else:
                out = self.model.inference(x0, x1, t, deep, skip)
            out = out[:, :, :h, :w]
            if self.channel_order == "bgr":
                out = out.flip(1)
            out = self._check(out.float(), h, w)
            if self.clamp:
                out = out.clamp(0.0, 1.0)
            outs.append(out[0].contiguous())
            torch.cuda.synchronize(self.device)
            self.stats["synth_s"] += time.time() - t0
            self.stats["frames"] += 1
        self.stats["pairs"] += 1
        return outs

    def info(self):
        commit = _git_commit(REPO)
        return {
            "name": "VTinker",
            "paper": "Wu, Fu, Guo, Han, Li. VTinker: Guided Flow Upsampling and Texture Mapping for "
                     "High-Resolution Video Frame Interpolation. AAAI 2026 (arXiv 2511.16124)",
            "repo_url": REPO_URL, "repo_path": str(REPO), "repo_commit": commit,
            "repo_commit_matches_pinned": commit == REPO_COMMIT_EXPECTED,
            "model_file": "core/models/VTinker.py (is_half=False, fp32)",
            "weights": [{"file": str(WEIGHTS), "url": WEIGHTS_URL, "sha256": self.weights_sha256 or WEIGHTS_SHA256,
                         "sha256_verified_at_load": self.weights_sha256 is not None, "bytes": WEIGHTS_BYTES,
                         "checkpoint_dtypes": self.ckpt_dtypes, "checkpoint_unused_keys": self.ckpt_unused}],
            "settings": {"precision": "float32", "allow_tf32": self.allow_tf32,
                         "cudnn_benchmark": torch.backends.cudnn.benchmark, "cache_flow": self.cache_flow,
                         "pad_mode": self.pad_mode,
                         "pad_value": self.pad_value if self.pad_mode == "constant" else None,
                         "pad_note": "upstream pads constant 0.5; replicate avoids the warped pad bleeding "
                                     "into the last delivered row(s)",
                         "clamp": self.clamp, "channel_order": self.channel_order,
                         "time_use": "flow_0t = F01*t, flow_1t = F10*(1-t) (linear); synthesis net has no t input; "
                                     "trained at t=0.5 only",
                         "time_holdout_evidence": "vs real frames on input/synthetic4k (job 319409): PSNR gain "
                                                  "over a cross-dissolve +2.15 dB at t=0.5 (gap 2), +0.80/+0.83 dB "
                                                  "at t=1/3, 2/3 (gap 3). The full t=0.1..0.9 curve needs a real "
                                                  ">=31-frame shot: SF=10 STRIDE=10 via run_vtinker.sbatch",
                         **(self._settings or {})},
            "versions": self._versions,
            "stats": self.stats,
            "license": "code: Pi-Lab License 1.0 (non-commercial; src/VTinker/LICENSE), borrows UPR-Net "
                       "(CC BY-NC-ND 4.0), PerVFI (Apache-2.0), softsplat (academic use only). Weights: HF model "
                       "card says MIT (models/VTinker/HF_README.md) but they derive from the non-commercial code "
                       "-> treat as non-commercial research use only.",
        }
