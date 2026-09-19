"""GIMM-VFI adapter for vfi_harness.py (Guo, Li, Loy, NeurIPS 2024; official code GSeanCDAT/GIMM-VFI).

    python vfi_harness.py --adapter adapters/gimmvfi.py ... \
        [--adapter_arg model_variant=R-P] [--adapter_arg ds_factor=0.25] [--adapter_arg allow_tf32=false]

adapter args
    model_variant  R-P (default: RAFT flow, LPIPS-finetuned "perceptual" checkpoint, upstream demo default)
                   R   (RAFT, checkpoint used for the paper's X4K numbers)
                   F-P / F (FlowFormer flow estimator; same two checkpoint flavours)
    ds_factor      0.25 (default; the published XTest-4K protocol). 0.5 and 1.0 are accepted. Only values
                   with an integer 1/ds are allowed: the upstream full-resolution upsampling uses
                   scale_factor = H/h and silently produces off-by-one sizes otherwise.
    allow_tf32     false (default): convolutions/matmuls in exact float32. true is faster on A100/H100
                   but TF32 lowers the internal precision of the flow/mask/residual networks.
    verify_sha256  true (default): check every weight file against the pinned Hugging Face LFS sha256.

How t is used: ONE model call per pair. Upstream GIMMVFI_*.forward computes the bidirectional flow
(RAFT or FlowFormer, both directions), the features and the correlation pyramid once, then, for each
requested t, splats the motion latents to t, queries the implicit motion field (SIREN hyponet) at
coordinate (t, x, y) and synthesises the frame. All len(ts) frames therefore share one motion
estimate; each t is synthesised independently (no recursion). t is passed exactly as given (float32),
both as the coordinate's time channel and as the per-frame timestep (upstream asserts they match).

Precision: inputs arrive as float32 RGB in [0, 1]; the adapter never quantises. The model is float32
end to end (RAFT mixed_precision=False, no autocast); the flow networks see 255*x as float, and the
final frame is a float32 bilinear warp of the FULL-RESOLUTION float inputs blended with an upsampled
float residual, clamped to [0, 1] by upstream. Padding: replicate to a multiple of 32 (as in upstream
video_Nx.py / X4K.py), removed from the outputs.

No files in the upstream repo are modified. Three things are adjusted at run time, in this process only:
  * checkpoint paths: upstream hard-codes relative paths 'pretrained_ckpt/raft-things.pth' and
    'pretrained_ckpt/flowformer_sintel.pth'; this adapter rebinds initialize_RAFT / get_cfg in the upstream
    module namespaces so they read the sha256-verified files in models/GIMM-VFI/.
  * FlowFormer (F variants only): the Twins encoders are built with pretrained=False instead of fetching
    ImageNet weights from the internet; they are overwritten anyway by flowformer_sintel.pth and then by
    the strict (all keys) load of the GIMM-VFI checkpoint, so the network is identical.
  * softsplat: upstream calls cupy.cuda.compile_with_cache (exists in the pinned CuPy 12.3; removed in
    CuPy 13). Only if it is missing, cuda_launch is replaced by an equivalent cupy.RawModule build.
    CUDA_HOME (used only for NVRTC -I flags) is set to the conda env if unset.
Upstream aborts with `assert False` when softsplat sees NaN; this adapter turns that (and any non-finite
input, flow or output) into a RuntimeError naming the problem instead of writing garbage.

Licence: S-Lab License 1.0, NON-COMMERCIAL use only (src/GIMM-VFI/LICENSE). Bundled RAFT is BSD-3,
FlowFormer Apache-2.0, softsplat "academic purposes only". The HF weights carry no licence statement.
"""
import functools
import hashlib
import importlib
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

MVFI_ROOT = Path(os.environ.get("MVFI_ROOT", Path(__file__).resolve().parents[2])).resolve()
REPO = MVFI_ROOT / "src" / "GIMM-VFI"
WEIGHTS = MVFI_ROOT / "models" / "GIMM-VFI"
REPO_URL = "https://github.com/GSeanCDAT/GIMM-VFI"
REPO_COMMIT_EXPECTED = "dbc56449994a3c2e045d46fd46bed570239913f8"
HF_REPO = "https://huggingface.co/GSean/GIMM-VFI"
HF_REVISION = "ab7735cdcfbd2e03c1bf2819380a25e8a4f321d1"

# file -> (bytes, sha256) from the Hugging Face LFS records of GSean/GIMM-VFI @ HF_REVISION
WEIGHT_FILES = {
    "raft-things.pth": (21108000, "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"),
    "flowformer_sintel.pth": (65066969, "a7ae0cf958312bea17a0b01fe95b34a4fe3ff0c937d0dee3afeec872bd506ac8"),
    "gimmvfi_r_arb.pt": (79305223, "1dff0ac3ca91483950eff814818115ff90cc3d16070089cc8cc3090eb3ee76f2"),
    "gimmvfi_r_arb_lpips.pt": (79308129, "b33e41536850594c01551978c4b63b1acd291c60d22310fbb6a4e57ac375033f"),
    "gimmvfi_f_arb.pt": (122818952, "e0ad627ea021b4234cd8ce33341d27abd505310b7d380e4c6cfbc8909f993a2e"),
    "gimmvfi_f_arb_lpips.pt": (122823503, "f2d52a1424d857a921e56fb82d9312ec27d58bbcbd58a7c3534593b9e6157858"),
}

# variant -> (upstream config, GIMM-VFI checkpoint, flow-estimator checkpoint)
VARIANTS = {
    "R-P": ("configs/gimmvfi/gimmvfi_r_arb.yaml", "gimmvfi_r_arb_lpips.pt", "raft-things.pth"),
    "R": ("configs/gimmvfi/gimmvfi_r_arb.yaml", "gimmvfi_r_arb.pt", "raft-things.pth"),
    "F-P": ("configs/gimmvfi/gimmvfi_f_arb.yaml", "gimmvfi_f_arb_lpips.pt", "flowformer_sintel.pth"),
    "F": ("configs/gimmvfi/gimmvfi_f_arb.yaml", "gimmvfi_f_arb.pt", "flowformer_sintel.pth"),
}

LICENSE_NOTE = ("S-Lab License 1.0 - non-commercial use only (text: src/GIMM-VFI/LICENSE). Bundled RAFT BSD-3, "
                "FlowFormer Apache-2.0, softsplat 'academic purposes only'. HF weights: no licence stated "
                "(assumed S-Lab 1.0).")


def normalise_variant(name):
    v = str(name).strip().upper().replace("_", "-")
    for prefix in ("GIMM-VFI-", "GIMMVFI-"):
        if v.startswith(prefix):
            v = v[len(prefix):]
    v = {"RP": "R-P", "FP": "F-P"}.get(v, v)
    if v not in VARIANTS:
        raise ValueError(f"unknown GIMM-VFI model_variant {name!r}; choose one of {sorted(VARIANTS)}")
    return v


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_weight(fname, verify=True):
    path = WEIGHTS / fname
    size, sha = WEIGHT_FILES[fname]
    if not path.is_file():
        raise FileNotFoundError(f"GIMM-VFI weight file missing: {path} (run scripts/envs/build_gimmvfi.sbatch)")
    if path.stat().st_size != size:
        raise RuntimeError(f"{path}: size {path.stat().st_size} != expected {size}")
    got = sha256_file(path) if verify else None
    if verify and got != sha:
        raise RuntimeError(f"{path}: sha256 {got} != expected {sha}")
    return {"file": str(path), "url": f"{HF_REPO}/resolve/{HF_REVISION}/{fname}", "bytes": size,
            "sha256": sha, "sha256_verified_at_load": bool(verify)}


def git_commit(repo):
    """Commit id without a git binary (compute nodes have none)."""
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
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ.setdefault("CUDA_HOME", sys.prefix)  # only used for NVRTC -I flags in softsplat.cuda_launch
    import cupy  # noqa: F401  (upstream softsplat needs it at import time)
    from models import create_model
    from utils.config import augment_defaults, load_config
    loaded = Path(sys.modules["models"].__file__).resolve()
    if REPO.resolve() not in loaded.parents:
        raise ImportError(f"'models' resolved to {loaded}, not the GIMM-VFI repo: sys.path clash")
    # import_module returns the real submodules: models.generalizable_INR defines *functions* named
    # gimmvfi_r / gimmvfi_f that shadow the submodule attributes, so `import a.b.gimmvfi_r as m` gives the function.
    mods = {k: importlib.import_module("models.generalizable_INR." + v) for k, v in {
        "ff_pkg": "flowformer", "gimmvfi_f": "gimmvfi_f", "gimmvfi_r": "gimmvfi_r",
        "softsplat": "modules.softsplat", "raft_pkg": "raft", "submission": "flowformer.configs.submission"}.items()}
    for k in ("gimmvfi_f", "gimmvfi_r"):
        assert isinstance(mods[k], types.ModuleType), k
    return dict(mods, create_model=create_model, augment_defaults=augment_defaults, load_config=load_config)


def _patch_softsplat_if_needed(softsplat_mod):
    """CuPy >= 13 removed compile_with_cache; rebuild cuda_launch with RawModule (kijai's fix). No-op on 12.x."""
    import cupy
    if hasattr(cupy.cuda, "compile_with_cache"):
        return "upstream cupy.cuda.compile_with_cache"

    @cupy.memoize(for_each_device=True)
    def cuda_launch(strKey):
        entry = softsplat_mod.objCudacache[strKey]
        inc = os.environ["CUDA_HOME"]
        mod = cupy.RawModule(code=entry["strKernel"], options=("-I " + inc, "-I " + inc + "/include"))
        return mod.get_function(entry["strFunction"])

    softsplat_mod.cuda_launch = cuda_launch
    return "cupy.RawModule (compile_with_cache missing in this CuPy)"


def _load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True), True
    except Exception:  # noqa: BLE001  (pickled non-tensor objects; file is sha256-verified above)
        return torch.load(path, map_location="cpu", weights_only=False), False


def build_model(variant="R-P", verify_sha256=True):
    """Construct the upstream model on CPU with the verified weights loaded (strict). Returns (model, meta)."""
    variant = normalise_variant(variant)
    cfg_rel, ckpt_name, flow_name = VARIANTS[variant]
    weights = [check_weight(flow_name, verify_sha256), check_weight(ckpt_name, verify_sha256)]
    up = _import_upstream()

    # Absolute checkpoint paths for the flow estimators (upstream uses cwd-relative 'pretrained_ckpt/...').
    up["gimmvfi_r"].initialize_RAFT = functools.partial(
        up["raft_pkg"].initialize_RAFT, model_path=str(WEIGHTS / "raft-things.pth"))

    def get_cfg():
        cfg = up["submission"].get_cfg()
        cfg.model = str(WEIGHTS / "flowformer_sintel.pth")
        cfg.latentcostformer.pretrain = False  # no ImageNet download; overwritten by the strict loads
        return cfg

    up["ff_pkg"].get_cfg = get_cfg

    config = up["augment_defaults"](up["load_config"](str(REPO / cfg_rel)))
    model, _ = up["create_model"](config.arch)
    sd, weights_only = _load_state_dict(WEIGHTS / ckpt_name)
    if not isinstance(sd, dict) or "state_dict" not in sd:
        raise RuntimeError(f"{ckpt_name}: unexpected checkpoint layout {type(sd)}")
    model.load_state_dict(sd["state_dict"], strict=True)
    model.eval().requires_grad_(False).float()
    meta = {"variant": variant, "config": str(REPO / cfg_rel), "arch_type": str(config.arch.type),
            "weights": weights, "checkpoint_weights_only_load": weights_only,
            "checkpoint_top_level_keys": sorted(sd.keys())}
    return model, up, meta


class _Padder:
    """Same geometry as upstream utils.utils.InputPadder (replicate, split evenly); upstream's module imports
    tensorboard, so it is re-implemented here."""

    def __init__(self, hw, divisor):
        self.ht, self.wd = hw
        ph = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pw = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self.pad_ = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]

    def pad(self, x):
        return F.pad(x, self.pad_, mode="replicate") if any(self.pad_) else x

    def unpad(self, x):
        h, w = x.shape[-2:]
        return x[..., self.pad_[2]:h - self.pad_[3], self.pad_[0]:w - self.pad_[1]]


class Adapter:
    def __init__(self, device, model_variant="R-P", ds_factor=0.25, allow_tf32=False, verify_sha256=True,
                 **unused):
        if unused:
            raise TypeError(f"unknown GIMM-VFI adapter args: {sorted(unused)}")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("GIMM-VFI needs CUDA (its softsplat kernel is CuPy/NVRTC only)")
        ds = float(ds_factor)
        inv = 1.0 / ds
        if not (0 < ds <= 1) or abs(inv - round(inv)) > 1e-9:
            raise ValueError(f"ds_factor={ds_factor}: use a value with integer 1/ds (0.25, 0.5, 1.0)")
        self.ds_factor = ds
        # pad so that the downscaled frame is an exact multiple of 8 (RAFT / FlowFormer 1/8 grid)
        self.pad_divisor = 32
        while (self.pad_divisor * ds) % 8 != 0:
            self.pad_divisor *= 2
        self.allow_tf32 = bool(allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.allow_tf32

        model, up, self.meta = build_model(model_variant, verify_sha256=bool(verify_sha256))
        self.softsplat_backend = _patch_softsplat_if_needed(up["softsplat"])
        self.model = model.to(self.device)
        self.variant = self.meta["variant"]

    def info(self):
        import cupy
        try:
            nvrtc = ".".join(map(str, cupy.cuda.nvrtc.getVersion()))
        except Exception as e:  # noqa: BLE001
            nvrtc = f"unknown ({e})"
        return {
            "name": f"GIMM-VFI-{self.variant}",
            "paper": "Guo, Li, Loy. Generalizable Implicit Motion Modeling for Video Frame Interpolation. "
                     "NeurIPS 2024 (arXiv 2407.08680)",
            "repo_url": REPO_URL, "repo_path": str(REPO), "repo_commit": git_commit(REPO),
            "repo_commit_expected": REPO_COMMIT_EXPECTED, "upstream_files_modified": False,
            "weights_repo": HF_REPO, "weights_revision": HF_REVISION, **self.meta,
            "license": LICENSE_NOTE, "license_file": str(REPO / "LICENSE"),
            "settings": {"model_variant": self.variant, "ds_factor": self.ds_factor,
                         "pad": f"replicate to multiple of {self.pad_divisor}",
                         "flow_iters": "RAFT 20 per direction (upstream fixed)" if self.variant.startswith("R")
                         else "FlowFormer upstream default",
                         "t_handling": "one forward per pair; all t share one bidirectional flow estimate",
                         "precision": "float32 end to end, no autocast",
                         "allow_tf32": self.allow_tf32,
                         "flowformer_twins_imagenet_init": False if self.variant.startswith("F") else None},
            "runtime": {"cupy": cupy.__version__, "nvrtc": nvrtc, "softsplat_backend": self.softsplat_backend,
                        "CUDA_HOME": os.environ.get("CUDA_HOME"),
                        "cudnn": torch.backends.cudnn.version()},
        }

    @torch.no_grad()
    def interpolate(self, img0, img1, ts):
        for name, im in (("img0", img0), ("img1", img1)):
            if im.ndim != 3 or im.shape[0] != 3:
                raise ValueError(f"{name}: expected (3, H, W), got {tuple(im.shape)}")
            if not torch.isfinite(im).all():
                raise ValueError(f"GIMM-VFI: {name} contains NaN/Inf")
        if img0.shape != img1.shape:
            raise ValueError(f"img0 {tuple(img0.shape)} != img1 {tuple(img1.shape)}")
        ts = [float(t) for t in ts]
        if not all(0.0 < t < 1.0 for t in ts):
            raise ValueError(f"ts must lie in (0, 1): {ts}")
        H, W = img0.shape[-2:]
        padder = _Padder((H, W), self.pad_divisor)
        x0 = padder.pad(img0.to(self.device, torch.float32).unsqueeze(0))
        x1 = padder.pad(img1.to(self.device, torch.float32).unsqueeze(0))
        xs = torch.cat((x0.unsqueeze(2), x1.unsqueeze(2)), dim=2)  # (1, 3, 2, Hp, Wp), as upstream
        s_shape = xs.shape[-2:]
        m = self.model
        coord_inputs = [(m.sample_coord_input(1, s_shape, [t], device=xs.device, upsample_ratio=self.ds_factor),
                         None) for t in ts]
        timesteps = [t * torch.ones(1, device=xs.device, dtype=torch.float32) for t in ts]
        try:
            out = m(xs, coord_inputs, t=timesteps, ds_factor=self.ds_factor)
        except AssertionError as e:
            raise RuntimeError(
                "GIMM-VFI aborted inside the model: an upstream assertion failed (softsplat uses `assert False` "
                "when NaN appears in the splatted latents/flow; see the 'NaN values detected' line above). "
                f"Pair shape {tuple(img0.shape)}, variant {self.variant}, ds_factor {self.ds_factor}. "
                "Refusing to write output for this pair.") from e
        flow = out["raft_flow"]
        if not torch.isfinite(flow).all():
            raise RuntimeError(f"GIMM-VFI: non-finite bidirectional flow ({(~torch.isfinite(flow)).sum().item()} "
                               "values); refusing to write output for this pair")
        frames = []
        for t, im in zip(ts, out["imgt_pred"]):
            im = padder.unpad(im)[0]
            if tuple(im.shape) != (3, H, W):
                raise RuntimeError(f"GIMM-VFI output {tuple(im.shape)} != {(3, H, W)} at t={t}")
            if im.dtype != torch.float32:
                raise RuntimeError(f"GIMM-VFI output dtype {im.dtype} (expected float32)")
            if not torch.isfinite(im).all():
                raise RuntimeError(f"GIMM-VFI: non-finite output frame at t={t}")
            frames.append(im)
        del out, xs, coord_inputs
        return frames
