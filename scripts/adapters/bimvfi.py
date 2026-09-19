"""BiM-VFI (Seo, Oh, Kim, CVPR 2025) adapter for vfi_harness.py.

Upstream: https://github.com/KAIST-VICLab/BiM-VFI (src/BiM-VFI, unmodified). Weights: bim_vfi.pth
(Google Drive id 18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC) in models/BiM-VFI/.
LICENCE: research and education only; commercial use needs written permission from the BiM-VFI authors
(README "License" section, copied to models/BiM-VFI/LICENSE_NOTICE.txt). No LICENSE file upstream.

What this adapter does (and why):
  * Imports only the network (modules.components.bim_vfi.BiMVFI). The package `modules/__init__.py`
    pulls in the whole training/metrics stack (wandb, pyiqa, lpips, ...), so a stub `modules` package
    is registered to skip it; nothing in the repo is patched. main.py / utils.experiment (whose
    SLURM_PROCID check starts distributed init) are never imported; SLURM_PROCID is removed anyway.
  * Calls the network once per t with time_step shaped (1,1,1), exactly like upstream
    inference_video.py (the demo's hard-coded 8x ratio is bypassed). Each t is an independent pass.
  * Padding: THE DEFAULT IS UPSTREAM (pad_mode=constant). The adapter hands the pair over unpadded
    and lets upstream's InputPadder do the work: pad bottom/right up to a multiple of
    2**(pyr_level+1) with zeros, inserted AFTER the model's own mean/std normalisation, so the added
    strip is the pair's mean colour rather than black; the model unpads its own output.
    pad_mode=reflect|replicate is an OPT-IN alternative in which the adapter pre-pads the pair
    centred to a multiple of the divisor with >= `margin` px per side (3840x2160 -> 4096x2304 at
    pyr7), which makes InputPadder a no-op, then crops back to H x W.
    Do NOT assume reflect suppresses the upstream edge-band artefact (issue #6). An earlier revision
    of this adapter defaulted to reflect and claimed exactly that; the measurements do not support
    the claim, and it has been withdrawn:
      - Job 319322 compared the modes at t=0.5 on the 4K smoke pair with a per-edge
        |out - 50/50 blend| metric. That metric cannot adjudicate padding: it mostly measures how
        far the model departs from a linear blend, i.e. motion, so it reads high wherever the
        content moves. Under reflect the top 16 px scored 0.0889 against an interior of 0.0636 --
        40% WORSE than the interior, the opposite of "no larger than the interior". Upstream
        constant scored top 0.0886 / interior 0.0638, i.e. indistinguishable from reflect there.
        Reflect was mildly better only at the two edges upstream actually pads (bottom 0.0315 vs
        0.0344, right 0.0259 vs 0.0268) and mildly worse at left (0.0730 vs 0.0691).
      - Pre-padding is NOT a border-only change. The model computes its normalisation mean and std
        over whatever tensor it is handed (bim_vfi.py:147-158, which runs BEFORE InputPadder at
        :162), so a centred reflect pad feeds 12.1% fabricated pixels into those global statistics
        and shifts the normalisation of EVERY pixel (measured on the smoke pair: mean 0.309944 ->
        0.311411, std 0.145377 -> 0.146000). Consequence: reflect vs constant differ by mean |delta|
        0.00978 in the INTERIOR (~640 16-bit codes) -- larger than the pyr8-vs-pyr7 difference
        (0.00817 under constant, 0.00891 under reflect) that this integration treats as a genuine
        modelling change.
      - Job 319406 then ran the test that CAN rank padding modes, because it has a true reference: a
        ground-truth hold-out (interpolate smoke frames 0 and 2 at t=0.5, score against the REAL
        frame 1). Mean |out - GT|, lower is better:
            constant (upstream): left 0.0912  top 0.0526  bottom 0.0410  right 0.0242
                                 interior 0.0590  whole 0.0588
            reflect64:           left 0.1038  top 0.0489  bottom 0.0412  right 0.0228
                                 interior 0.0592  whole 0.0590
            replicate64:         left 0.0880  top 0.0434  bottom 0.0399  right 0.0226
                                 interior 0.0586  whole 0.0584
        Reflect is WORSE than upstream whole-frame and in the interior, and 14% worse at the left
        edge -- the opposite of the withdrawn claim. Replicate is marginally best (whole 0.0584 vs
        0.0588, i.e. 0.7%), but on one pair of one synthetic clip that is not enough to justify
        deviating from upstream either, so it stays opt-in too.
    So the unearned deviation is gone: upstream behaviour is the default, reflect and replicate stay
    available for anyone who wants to test them on real footage. Numbers live in PADDING_EVIDENCE
    below and are copied into info()["settings"]["padding_evidence"], hence into run_report.json.
  * float32 end to end: no autocast, no fp16, TF32 disabled by default (tf32=true re-enables it for
    speed), no clamp/quantisation (the harness clamps and rounds once to uint16). The network's own
    mean/std normalisation is range-agnostic; 16-bit sources arrive as value/65535.
  * Raw (pre-clamp) output statistics and a non-finite check are kept and reported in info().
  * pyr_level=8: upstream downsamples the pair by 2**7 with F.interpolate(bilinear, antialias=True); torch
    2.4.1's CUDA kernel refuses that scale ("Too much shared memory required"). The adapter wraps `F` in the
    upstream bim_vfi module namespace so that exactly this call falls back to the same operator on CPU (float32)
    and the result is moved back; everything else stays on the GPU. Counted in info()["runtime"].

adapter args (--adapter_arg k=v):
  pyr_level=7      pyramid levels (7 for 4K as upstream; 8 allowed for very large motion; 3..8 accepted)
  pad_mode=constant  constant (DEFAULT = upstream: InputPadder zeros, bottom/right only, applied after
                   the model's normalisation) | reflect | replicate (both = adapter pre-pads centred;
                   opt-in, and they shift the model's global normalisation -- see above)
  margin=64        minimum padding on every side in px (ignored for pad_mode=constant)
  tf32=false       allow TF32 in cuDNN convolutions / matmul (faster on A100, not bit-exact fp32)
  cudnn_benchmark=false  true = cuDNN autotuning: ~2 min warm-up per process and ~74 GiB transient peak at 4K
  t_batch=1        timesteps per forward (batching t shares nothing, only raises memory)
  nonfinite=raise  raise | warn: what to do if the model outputs NaN/Inf
  weights=<path>   checkpoint path (default models/BiM-VFI/bim_vfi.pth, sha256-verified)
"""
import ctypes
import hashlib
import os
import pickle
import sys
import types
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("MVFI_ROOT", Path(__file__).resolve().parents[2])).resolve()
REPO = ROOT / "src" / "BiM-VFI"
MODEL_DIR = ROOT / "models" / "BiM-VFI"
WEIGHTS = MODEL_DIR / "bim_vfi.pth"
REPO_URL = "https://github.com/KAIST-VICLab/BiM-VFI"
PINNED_COMMIT = "df79cbcad4214d48911040def459174d0a4c9791"
WEIGHTS_URL = "https://drive.google.com/file/d/18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC/view"
WEIGHTS_BYTES = 82749822
WEIGHTS_SHA256 = "510cbcb07a7f8fcb17b4da241bbee2a6e20a8eb9bb423cf4df4815b7cd6d1cc0"  # first download, job 319151
LICENCE = ("Research and education only (README 'License' section, which covers the source code and the "
           "checkpoint); any commercial use needs formal permission from the BiM-VFI authors "
           "(see https://github.com/KAIST-VICLab/BiM-VFI). No LICENSE file upstream. Weights trained on Vimeo90K (research-use terms).")

# Why pad_mode defaults to upstream "constant". Measured on the 4K smoke pair, t=0.5, pyr7, fp32.
# Carried into info()["settings"]["padding_evidence"] so every run_report.json states it.
PADDING_EVIDENCE = {
    "default": "constant (upstream InputPadder: zeros bottom/right, applied after the model's own "
               "mean/std normalisation). reflect/replicate are opt-in.",
    "reflect_vs_constant_mean_abs_interior": 0.00978,
    "replicate_vs_constant_mean_abs_interior": 0.01045,
    "pyr8_vs_pyr7_mean_abs_reference": {"constant": 0.00817, "reflect": 0.00891},
    "normalisation_shift_from_prepadding": {"fabricated_fraction_of_tensor": 0.121,
                                            "mean": [0.309944, 0.311411], "std": [0.145377, 0.146000]},
    "holdout_vs_ground_truth_mean_abs": {
        "_what": "smoke frames 0,2 -> t=0.5 scored against the real frame 1; lower is better; job 319406",
        "constant": {"whole": 0.0588, "interior": 0.0590, "left": 0.0912, "top": 0.0526,
                     "bottom": 0.0410, "right": 0.0242},
        "reflect64": {"whole": 0.0590, "interior": 0.0592, "left": 0.1038, "top": 0.0489,
                      "bottom": 0.0412, "right": 0.0228},
        "replicate64": {"whole": 0.0584, "interior": 0.0586, "left": 0.0880, "top": 0.0434,
                        "bottom": 0.0399, "right": 0.0226}},
    "interpretation": "Pre-padding is not a border-only change: the model normalises by mean/std taken over the "
                      "tensor it is given (bim_vfi.py:147-158, before InputPadder at :162), so a centred reflect "
                      "pad puts 12.1% fabricated pixels into those global statistics and shifts every pixel. The "
                      "resulting interior delta (0.00978) exceeds the pyr8-vs-pyr7 delta (0.00817-0.00891) that is "
                      "treated as a real modelling change, so reflect is a whole-picture deviation, not a border "
                      "tweak. Against ground truth it is also slightly WORSE than upstream (whole 0.0590 vs 0.0588, "
                      "left edge 0.1038 vs 0.0912), so the deviation buys nothing and upstream is the default.",
    "edge_metric_verdict": "The |out - 50/50 blend| per-edge metric used in job 319322 does NOT discriminate "
                           "padding modes; it mostly tracks motion. Per-edge at t=0.5 -- reflect: top 0.0889 "
                           "bottom 0.0315 left 0.0730 right 0.0259 interior 0.0636; constant: top 0.0886 bottom "
                           "0.0344 left 0.0691 right 0.0268 interior 0.0638. Reflect's top strip is 40% ABOVE its "
                           "own interior, so the earlier claim that reflect suppresses upstream issue #6 is "
                           "WITHDRAWN as unsupported.",
    "evidence_jobs": ["319322 (first per-edge + reflect-vs-constant interior delta, reflect was still the default)",
                      "319406 (re-measured against this on-disk code, plus the normalisation-shift mechanism and "
                      "the ground-truth hold-out that actually ranks the modes)"],
    "open": "Decided on one synthetic 4K pair only. On the first real shot run VARIANT=pyr7_sf10 (upstream) and, "
            "if border quality is ever in question, VARIANT=pyr7_sf10_reflectpad, and decide from that footage. "
            "replicate64 was marginally best in the hold-out (whole 0.0584 vs 0.0588, 0.7%) and is worth re-testing "
            "on real footage, but one pair of one synthetic clip does not justify deviating from upstream.",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
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


def preload_nvrtc():
    """Load the NVRTC shipped with the torch cu12x wheel (nvidia-cuda-nvrtc-cu12) into the process, so
    CuPy's dlopen('libnvrtc.so.12') resolves to it without LD_LIBRARY_PATH or a system CUDA toolkit."""
    libdir = Path(torch.__file__).resolve().parent.parent / "nvidia" / "cuda_nvrtc" / "lib"
    loaded = []
    for pattern in ("libnvrtc-builtins.so*", "libnvrtc.so*"):
        for lib in sorted(libdir.glob(pattern)):
            if lib.name.endswith(".alt.so") or ".alt." in lib.name:
                continue
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            loaded.append(str(lib))
    return loaded


AA_CPU_FALLBACKS = {"calls": 0}


class _FunctionalWithAAFallback:
    """torch.nn.functional, except that an antialiased interpolate the CUDA kernel cannot handle runs on CPU."""

    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def interpolate(input, *args, **kwargs):  # noqa: A002
        try:
            return F.interpolate(input, *args, **kwargs)
        except RuntimeError as e:
            if not (input.is_cuda and kwargs.get("antialias") and "shared memory" in str(e)):
                raise
            AA_CPU_FALLBACKS["calls"] += 1
            return F.interpolate(input.float().cpu(), *args, **kwargs).to(input.device, input.dtype)


def import_bimvfi():
    """Return the upstream BiMVFI nn.Module class without importing the training stack."""
    os.environ.pop("SLURM_PROCID", None)  # upstream utils/experiment.py treats it as a distributed launch
    repo = str(REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    mod = sys.modules.get("modules")
    if mod is None:  # stub package: skips modules/__init__.py (training stack), keeps relative imports working
        mod = types.ModuleType("modules")
        mod.__path__ = [str(REPO / "modules")]
        sys.modules["modules"] = mod
    elif list(getattr(mod, "__path__", [])) != [str(REPO / "modules")]:
        raise ImportError(f"a different top-level 'modules' package is already imported: {mod}")
    from modules.components.bim_vfi import BiMVFI  # noqa: E402
    bim_mod = sys.modules["modules.components.bim_vfi.bim_vfi"]
    if not isinstance(bim_mod.F, _FunctionalWithAAFallback):
        bim_mod.F = _FunctionalWithAAFallback()
    import utils.padder  # noqa: E402  (upstream top-level 'utils' package; make sure it is the repo's)
    if not str(Path(utils.padder.__file__).resolve()).startswith(repo):
        raise ImportError(f"'utils' resolved outside the BiM-VFI repo: {utils.padder.__file__}")
    return BiMVFI


class _Inert:
    """Stand-in for training-only objects referenced by the checkpoint pickle (never called for real)."""

    def __init__(self, *args, **kwargs):
        self.args = tuple(map(repr, args))

    def __repr__(self):
        return f"<inert {self.args}>"


class _RestrictedUnpickler(pickle.Unpickler):
    """bim_vfi.pth holds, besides the weights, the OneCycleLR state, which pickles
    getattr(torch.optim.lr_scheduler.OneCycleLR, '_annealing_cos'); torch.load(weights_only=True) rejects
    builtins.getattr. Allow only what a state dict needs and replace those two globals by inert stubs, so no
    foreign code runs. (torch maps '*Storage' globals itself before calling this.)"""
    ALLOWED = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"),
               ("torch._utils", "_rebuild_parameter"), ("torch", "Size"), ("torch", "device")}
    INERT = {("__builtin__", "getattr"), ("builtins", "getattr"),
             ("torch.optim.lr_scheduler", "OneCycleLR")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        if (module, name) in self.INERT:
            return _Inert
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in checkpoint")


_restricted_pickle = types.SimpleNamespace(Unpickler=_RestrictedUnpickler, load=None, __name__="restricted_pickle")


def load_checkpoint(weights):
    try:
        return torch.load(weights, map_location="cpu", weights_only=True), "weights_only=True"
    except pickle.UnpicklingError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ckpt = torch.load(weights, map_location="cpu", weights_only=False, pickle_module=_restricted_pickle)
        return ckpt, "restricted unpickler (tensors + OrderedDict only; lr_scheduler getattr stubbed)"


def load_model(weights=WEIGHTS, pyr_level=7, device="cpu", verify_sha=True):
    weights = Path(weights)
    digest = _sha256(weights)
    if verify_sha and WEIGHTS_SHA256 != "PENDING" and digest != WEIGHTS_SHA256:
        raise RuntimeError(f"{weights}: sha256 {digest} != pinned {WEIGHTS_SHA256}")
    BiMVFI = import_bimvfi()
    model = BiMVFI(pyr_level=pyr_level, feat_channels=32)  # cfgs/bim_vfi*.yaml: feat_channels 32
    ckpt, load_mode = load_checkpoint(weights)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    ckpt_dtypes = sorted({str(v.dtype) for v in sd.values() if torch.is_tensor(v)})
    model.load_state_dict(sd, strict=True)
    model = model.float().eval().requires_grad_(False).to(device)
    meta = {"weights_file": str(weights), "weights_sha256": digest, "weights_bytes": weights.stat().st_size,
            "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "checkpoint_iteration": ckpt.get("iteration") if isinstance(ckpt, dict) else None,
            "checkpoint_tensor_dtypes": ckpt_dtypes, "checkpoint_load": load_mode,
            "n_params": sum(p.numel() for p in model.parameters())}
    return model, meta


def pad_amounts(h, w, divisor, margin):
    """(left, right, top, bottom): centred padding to a multiple of `divisor` with >= margin per side."""
    def one(n):
        tot = -(-(n + 2 * margin) // divisor) * divisor - n
        return tot // 2, tot - tot // 2
    left, right = one(w)
    top, bottom = one(h)
    return left, right, top, bottom


class Adapter:
    def __init__(self, device, pyr_level=7, pad_mode="constant", margin=64, tf32=False, cudnn_benchmark=False,
                 t_batch=1, nonfinite="raise", weights=str(WEIGHTS), **unknown):
        if unknown:
            raise TypeError(f"unknown adapter args: {sorted(unknown)}")
        if not 3 <= int(pyr_level) <= 8:
            raise ValueError("pyr_level must be in 3..8 (7 = upstream setting for 4K)")
        if pad_mode not in ("reflect", "replicate", "constant"):
            raise ValueError("pad_mode must be reflect, replicate or constant")
        if nonfinite not in ("raise", "warn"):
            raise ValueError("nonfinite must be raise or warn")
        self.device = torch.device(device)
        self.pyr_level = int(pyr_level)
        self.divisor = 2 ** (self.pyr_level + 1)
        self.pad_mode = pad_mode
        self.margin = int(margin) if pad_mode != "constant" else 0
        self.t_batch = max(1, int(t_batch))
        self.nonfinite = nonfinite
        self.tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = self.tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        self.nvrtc_preloaded = preload_nvrtc() if self.device.type == "cuda" else []
        self.model, self.meta = load_model(weights, self.pyr_level, self.device)
        self.commit = _git_commit(REPO)
        if self.commit != PINNED_COMMIT:
            warnings.warn(f"BiM-VFI repo at {self.commit}, adapter validated at {PINNED_COMMIT}")
        self.padded = None
        self.stats = {"frames": 0, "min": float("inf"), "max": float("-inf"), "n_below0": 0, "n_above1": 0,
                      "n_values": 0, "nonfinite": 0}

    def _pad(self, x):
        h, w = x.shape[-2:]
        if self.pad_mode == "constant":  # upstream InputPadder: zeros at bottom/right (applied after its
            return x, (0, 0, 0, 0)       # own normalisation, i.e. mean colour); leave it to the model
        l, r, t, b = pad_amounts(h, w, self.divisor, self.margin)
        mode = self.pad_mode
        if mode == "reflect" and (max(l, r) >= w or max(t, b) >= h):
            mode = "replicate"  # reflect needs pad < size (tiny test inputs only)
        return F.pad(x, (l, r, t, b), mode=mode), (l, r, t, b)

    def _track(self, pred):
        finite = torch.isfinite(pred)
        n_bad = int((~finite).sum())
        if n_bad:
            msg = f"BiM-VFI produced {n_bad} non-finite values"
            if self.nonfinite == "raise":
                raise FloatingPointError(msg)
            warnings.warn(msg)
            pred = torch.where(finite, pred, torch.zeros_like(pred))
        s = self.stats
        s["frames"] += pred.shape[0]
        s["nonfinite"] += n_bad
        s["min"] = min(s["min"], float(pred.min()))
        s["max"] = max(s["max"], float(pred.max()))
        s["n_below0"] += int((pred < 0).sum())
        s["n_above1"] += int((pred > 1).sum())
        s["n_values"] += pred.numel()
        return pred

    @torch.no_grad()
    def interpolate(self, img0, img1, ts):
        assert img0.shape == img1.shape and img0.dim() == 3 and img0.shape[0] == 3, img0.shape
        h, w = img0.shape[-2:]
        x0 = img0.to(self.device, torch.float32).unsqueeze(0)
        x1 = img1.to(self.device, torch.float32).unsqueeze(0)
        x0, (l, r, t, b) = self._pad(x0)
        x1, _ = self._pad(x1)
        self.padded = {"height": int(x0.shape[-2]), "width": int(x0.shape[-1]),
                       "left": l, "right": r, "top": t, "bottom": b}
        outs = []
        for i in range(0, len(ts), self.t_batch):
            chunk = [float(v) for v in ts[i:i + self.t_batch]]
            n = len(chunk)
            tt = torch.tensor(chunk, dtype=torch.float32, device=self.device)
            tt = tt.view(1, 1, 1) if n == 1 else tt.view(n, 1, 1, 1)  # upstream passes time_range[i]: (1,1,1)
            a, c = (x0, x1) if n == 1 else (x0.expand(n, -1, -1, -1), x1.expand(n, -1, -1, -1))
            res = self.model(img0=a, img1=c, time_step=tt, pyr_level=self.pyr_level)
            pred = res["imgt_pred"]
            del res
            if pred.dtype != torch.float32:
                raise TypeError(f"model returned {pred.dtype}, expected float32")
            pred = pred[..., t:t + h, l:l + w]
            pred = self._track(pred)
            outs.extend(pred[k].clone() for k in range(n))
        return outs

    def info(self):
        s = dict(self.stats)
        if s["n_values"]:
            s["frac_below0"] = s.pop("n_below0") / s["n_values"]
            s["frac_above1"] = s.pop("n_above1") / s["n_values"]
        try:
            import cupy
            from cupy.cuda import nvrtc
            cupy_info = {"cupy": cupy.__version__, "nvrtc": ".".join(map(str, nvrtc.getVersion()))}
        except Exception as e:  # noqa: BLE001
            cupy_info = {"cupy_error": repr(e)}
        return {
            "name": "BiM-VFI (CVPR 2025)", "repo_url": REPO_URL, "repo_path": str(REPO),
            "repo_commit": self.commit, "repo_patches": "none (network imported via a stub 'modules' package)",
            "weights": [{"file": self.meta["weights_file"], "url": WEIGHTS_URL, "sha256": self.meta["weights_sha256"],
                         "bytes": self.meta["weights_bytes"]}],
            "checkpoint": {k: self.meta[k] for k in ("checkpoint_epoch", "checkpoint_iteration",
                                                    "checkpoint_tensor_dtypes", "checkpoint_load", "n_params")},
            "license": LICENCE, "license_file": str(MODEL_DIR / "LICENSE_NOTICE.txt"),
            "settings": {"pyr_level": self.pyr_level, "divisor": self.divisor, "pad_mode": self.pad_mode,
                         "pad_is_upstream": self.pad_mode == "constant", "padding_evidence": PADDING_EVIDENCE,
                         "margin": self.margin, "padded": self.padded, "t_batch": self.t_batch,
                         "time_step_shape": "(1,1,1) per t" if self.t_batch == 1 else "(B,1,1,1)",
                         "dtype": "float32", "autocast": False, "tf32": self.tf32,
                         "cudnn_benchmark": torch.backends.cudnn.benchmark,
                         "output": "imgt_pred, cropped, unclamped (harness clamps to [0,1] and rounds to uint16)"},
            "raw_output_stats": s,
            "runtime": {**cupy_info, "nvrtc_preloaded": self.nvrtc_preloaded,
                        "antialias_downsample_cpu_fallback_calls": AA_CPU_FALLBACKS["calls"]},
        }
