"""EMA-VFI adapter for vfi_harness.py (Zhang et al., CVPR 2023; https://github.com/MCG-NJU/EMA-VFI).

Uses the ARBITRARY-TIMESTEP checkpoints (ours_t, default; ours_small_t optional). All requested t come from a
single upstream `Model.multi_inference(..., time_list=ts)` call per pair: the feature backbone runs once per pair
(per TTA pass) and only flow estimation + warp/refine run per t. Numerically this is the official 4K path
(benchmark/XTest_8X.py -> hr_inference(TTA=True, down_scale=0.25)) evaluated for several t at once.

Adapter args (--adapter_arg k=v):
  model=ours_t            | ours_small_t
  down_scale=0.25         flow is estimated on a copy downscaled by this factor (official: 0.25 for 4K,
                          0.5 for 2K); 1/down_scale must be an integer (1, 0.5, 0.25, 0.125)
  tta=true                flip test-time augmentation (upstream default for ours_t; false for ours_small_t)
  fast_tta=false          true: run the flipped pass batched with the normal one (more memory; implies tta)
  channel_order=bgr       order fed to the network. Upstream inference feeds cv2 BGR (training randomly swaps
                          the order, so rgb also works); the permutation is lossless and undone on output
  refine=true             false: return the mask-weighted warped blend, bypassing the refinement U-Net
                          residual (the temporal-stability variant reported at NTIRE 2026)
  tf32=false              true: allow TF32 in cuDNN/cuBLAS (faster on A100, fewer significant bits);
                          default keeps strict float32 arithmetic everywhere
  cudnn_benchmark=false   true = cuDNN autotuning (upstream benchmarks enable it). Measured at 3840x2160 fp32 on
                          the A100: autotuning costs ~140 s on the first pair and gains <0.5 s/pair, outputs
                          differ by <=2 16-bit codes, so it is off by default
  pad_divisor=0           0 = auto = 32/down_scale (flow net needs the DOWNSCALED frame divisible by 32)

Precision: input float32 [0,1] (16-bit source / 65535) -> float32 network -> float32 output in [0,1].
No uint8, no *255, no autocast/half. Upstream clamps its prediction to [0,1]; this adapter adds no clamp.
Padding: upstream InputPadder (replicate, centred) to a multiple of pad_divisor, cropped after inference.
"""
import os
import hashlib
import sys
from pathlib import Path

import torch

MVFI_ROOT = Path(os.environ.get("MVFI_ROOT", Path(__file__).resolve().parents[2])).resolve()
REPO = MVFI_ROOT / "src" / "EMA-VFI"
WEIGHTS_DIR = MVFI_ROOT / "models" / "EMA-VFI"
REPO_URL = "https://github.com/MCG-NJU/EMA-VFI"
EXPECTED_COMMIT = "75b6f6a889e695df875e103374040d47a4cfac7c"

CHECKPOINTS = {
    "ours_t": {
        "arch": {"F": 32, "depth": [2, 2, 2, 4, 4]},
        "sha256": "c09e82d193e303d7b91f1e36bf535886b820cb56105574355211d60a6b557a99",
        "bytes": 263760209,
        "url": "https://drive.google.com/uc?export=download&id=1mL4ht5DYFMA1CYiGDFbej-ufh5QoqfwN",
        "default_tta": True,
    },
    "ours_small_t": {
        "arch": {"F": 16, "depth": [2, 2, 2, 2, 2]},
        "sha256": "ad845671f7250d408937718aca19eaf87aa1ca79215070f7641eea332b703233",
        "bytes": 58535773,
        "url": "https://drive.google.com/uc?export=download&id=1mVC3gZVHibbwvDsw2Z1yuBkMjmDCZyG6",
        "default_tta": False,
    },
}


def _bool(v, name):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"emavfi: {name}={v!r} is not a boolean")


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


def _import_upstream():
    """Import the upstream modules by their (generic) top-level names, making sure they come from REPO."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    repo = str(REPO.resolve()) + "/"
    for mod in ("config", "Trainer", "model", "benchmark"):
        m = sys.modules.get(mod)
        if m is not None and not str(Path(getattr(m, "__file__", None) or "/").resolve()).startswith(repo):
            raise ImportError(f"emavfi: a foreign top-level module '{mod}' is already imported ({m})")
    import config as cfg
    return cfg


class Adapter:
    def __init__(self, device, model="ours_t", down_scale=0.25, tta=None, fast_tta=False, channel_order="bgr",
                 refine=True, tf32=False, cudnn_benchmark=False, pad_divisor=0, **unknown):
        if unknown:
            raise TypeError(f"emavfi: unknown adapter args {sorted(unknown)}")
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("emavfi: upstream code hard-codes .cuda(); a CUDA device is required")
        if model not in CHECKPOINTS:
            raise ValueError(f"emavfi: model must be one of {list(CHECKPOINTS)} (arbitrary-t checkpoints)")
        ck = CHECKPOINTS[model]
        self.device = device
        self.model_name = model
        self.down_scale = float(down_scale)
        inv = round(1.0 / self.down_scale)
        if not (0 < self.down_scale <= 1.0) or abs(inv * self.down_scale - 1.0) > 1e-9:
            raise ValueError("emavfi: down_scale must be 1/n (1, 0.5, 0.25, 0.125)")
        self.tta = ck["default_tta"] if tta is None else _bool(tta, "tta")
        self.fast_tta = _bool(fast_tta, "fast_tta") and self.tta  # upstream fast_TTA always applies TTA
        self.channel_order = str(channel_order).lower()
        if self.channel_order not in ("bgr", "rgb"):
            raise ValueError("emavfi: channel_order must be bgr or rgb")
        self.refine = _bool(refine, "refine")
        self.tf32 = _bool(tf32, "tf32")
        self.cudnn_benchmark = _bool(cudnn_benchmark, "cudnn_benchmark")
        self.pad_divisor = int(pad_divisor) or 32 * inv
        if self.pad_divisor % (32 * inv):
            raise ValueError(f"emavfi: pad_divisor must be a multiple of {32 * inv} for down_scale {down_scale}")

        # Strict float32 unless TF32 is explicitly allowed (process-wide switches; the harness runs one model).
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cudnn.allow_tf32 = self.tf32
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.cuda.set_device(device if device.index is not None else torch.cuda.current_device())

        cfg = _import_upstream()
        cfg.MODEL_CONFIG["LOGNAME"] = model
        cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(**ck["arch"])
        from Trainer import Model  # noqa: E402  (upstream; reads cfg.MODEL_CONFIG at construction)
        from benchmark.utils.padder import InputPadder  # noqa: E402
        from model.warplayer import warp  # noqa: E402
        self._InputPadder = InputPadder
        self._padders = {}

        wpath = WEIGHTS_DIR / f"{model}.pkl"
        size = wpath.stat().st_size
        digest = _sha256(wpath)
        if size != ck["bytes"] or digest != ck["sha256"]:
            raise RuntimeError(f"emavfi: {wpath} failed verification ({size} bytes, sha256 {digest})")

        self.model = Model(-1)  # builds the net and moves it to CUDA (upstream Model.device())
        sd = torch.load(wpath, map_location="cpu", weights_only=True)
        bad = {k: v.dtype for k, v in sd.items() if v.is_floating_point() and v.dtype != torch.float32}
        if bad:
            raise RuntimeError(f"emavfi: non-float32 tensors in checkpoint: {list(bad.items())[:5]}")
        # Same key conversion as upstream Model.load_model() (strip DDP prefix, drop cached window masks).
        sd = {k.replace("module.", ""): v for k, v in sd.items()
              if "module." in k and "attn_mask" not in k and "HW" not in k}
        self.model.net.load_state_dict(sd, strict=True)
        self.model.eval()
        self.model.net.float()
        self.n_params = sum(p.numel() for p in self.model.net.parameters())
        self.weights = {"file": str(wpath), "bytes": size, "sha256": digest, "url": ck["url"],
                        "source_url_used": (wpath.parent / (wpath.name + ".source_url")).read_text().strip()
                        if (wpath.parent / (wpath.name + ".source_url")).exists() else None}

        if not self.refine:
            def warp_blend_only(imgs, af, flow, mask):
                # upstream coraseWarp_and_Refine without the U-Net residual: the mask-weighted warped blend
                w0 = warp(imgs[:, :3], flow[:, :2])
                w1 = warp(imgs[:, 3:6], flow[:, 2:4])
                m = torch.sigmoid(mask)
                return w0 * m + w1 * (1 - m)
            self.model.net.coraseWarp_and_Refine = warp_blend_only  # instance attribute, upstream file untouched

        self.commit = _git_commit(REPO)
        self.stats = {"pairs": 0, "frames": 0, "nonfinite_values": 0,
                      "raw_min": float("inf"), "raw_max": float("-inf")}

    def _padder(self, shape):
        key = tuple(shape[-2:])
        if key not in self._padders:
            self._padders[key] = self._InputPadder(shape, divisor=self.pad_divisor)
        return self._padders[key]

    @torch.no_grad()
    def interpolate(self, img0, img1, ts):
        x0 = img0.to(self.device, torch.float32, non_blocking=True).unsqueeze(0)
        x1 = img1.to(self.device, torch.float32, non_blocking=True).unsqueeze(0)
        if self.channel_order == "bgr":
            x0, x1 = x0.flip(1), x1.flip(1)
        padder = self._padder(x0.shape)
        x0, x1 = padder.pad(x0, x1)
        preds = self.model.multi_inference(x0, x1, TTA=self.tta, down_scale=self.down_scale,
                                           time_list=[float(t) for t in ts], fast_TTA=self.fast_tta)
        outs = []
        for p in preds:  # (3, Hp, Wp) float32
            p = padder.unpad(p)
            if self.channel_order == "bgr":
                p = p.flip(0)
            if p.dtype != torch.float32:
                raise RuntimeError(f"emavfi: unexpected output dtype {p.dtype}")
            fin = torch.isfinite(p)
            lo, hi = torch.aminmax(torch.where(fin, p, p.new_tensor(0.5)))
            self.stats["nonfinite_values"] += int((~fin).sum())
            self.stats["raw_min"] = min(self.stats["raw_min"], float(lo))
            self.stats["raw_max"] = max(self.stats["raw_max"], float(hi))
            outs.append(p)
        self.stats["pairs"] += 1
        self.stats["frames"] += len(outs)
        return outs

    def info(self):
        import timm
        return {
            "name": "EMA-VFI",
            "paper": "Zhang et al., Extracting Motion and Appearance via Inter-Frame Attention for Efficient "
                     "Video Frame Interpolation, CVPR 2023 (arXiv 2303.00440)",
            "repo_url": REPO_URL,
            "repo_path": str(REPO),
            "repo_commit": self.commit,
            "repo_commit_expected": EXPECTED_COMMIT,
            "upstream_files_modified": False,
            "checkpoint": self.model_name,
            "weights": [self.weights],
            "n_params": self.n_params,
            "settings": {"model": self.model_name, "down_scale": self.down_scale, "tta": self.tta,
                         "fast_tta": self.fast_tta, "channel_order": self.channel_order, "refine": self.refine,
                         "tf32": self.tf32, "cudnn_benchmark": self.cudnn_benchmark,
                         "pad_divisor": self.pad_divisor, "pad_mode": "replicate (upstream InputPadder)",
                         "inference_call": "Trainer.Model.multi_inference (one call per pair, all t)"},
            "precision": "float32 end to end" + ("" if self.tf32 else " (TF32 disabled in cuDNN and cuBLAS)")
                         + "; no autocast/half; no uint8 or *255 round trip",
            "output_stats": dict(self.stats),
            "timm": timm.__version__,
            "license": "code: Apache-2.0 (src/EMA-VFI/LICENSE; README asks to also respect RIFE, PVT, IFRNet, "
                       "Swin, HRFormer licences). weights: no separate licence stated (project-level "
                       "Apache-2.0 presumed); trained on Vimeo90K (research terms).",
        }
