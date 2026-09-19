"""Frame IO helpers shared by the movie_vfi scripts (no torch import)."""
import re
from pathlib import Path

import numpy as np
import tifffile

TIFF_EXTS = {".tif", ".tiff"}
IMG_EXTS = TIFF_EXTS | {".png"}


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]


def to_hwc3(a):
    """Return an (H, W, 3) view of a TIFF/PNG array, dropping alpha, handling planar layout."""
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)
    return a[..., :3]


def read_image(path):
    if Path(path).suffix.lower() in TIFF_EXTS:
        return to_hwc3(tifffile.imread(path, key=0))
    from PIL import Image
    with Image.open(path) as im:
        return to_hwc3(np.asarray(im))
