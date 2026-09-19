"""Technical (non-creative) colour transforms used by the Test B experiment and preview views.

Blackmagic Design Film Gen 5 log / Blackmagic Wide Gamut Gen 5  ->  Rec.709 primaries,
BT.1886 (pure 2.4 power, black = 0) display encoding.

Steps: Gen5 inverse OETF (scene linear) -> 3x3 gamut matrix (Bradford not needed: both D65)
-> hard clip to [0, 1] (no tone-mapping curve, no look) -> x ** (1/2.4).
Values above scene-linear 1.0 (about 2.5 stops over 18 % grey) clip: this transform is
deliberately simple and NOT invertible for highlights. Resolve's own CST (with DaVinci tone
mapping) is the authoritative alternative if a Resolve-rendered Rec.709 export is preferred.
"""
import numpy as np
import colour

_SRC = colour.models.RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT
_DST = colour.models.RGB_COLOURSPACE_BT709
M_BMDWG_TO_709 = colour.matrix_RGB_to_RGB(_SRC, _DST, chromatic_adaptation_transform=None).astype(np.float32)


def gen5_log_to_linear(x):
    return colour.models.oetf_inverse_BlackmagicFilmGeneration5(x).astype(np.float32)


def gen5_to_rec709_display(x, return_clip_fraction=False):
    """x: float array (..., 3) in [0, 1] (Gen5 log code values). Returns Rec.709 display values in [0, 1]."""
    lin = gen5_log_to_linear(np.asarray(x, dtype=np.float32))
    lin709 = lin @ M_BMDWG_TO_709.T
    clipped = np.clip(lin709, 0.0, 1.0)
    out = clipped ** (1.0 / 2.4)
    if return_clip_fraction:
        return out, float(np.mean(lin709 > 1.0)), float(np.mean(lin709 < 0.0))
    return out


def describe():
    return {
        "source": f"{_SRC.name} primaries/whitepoint, Blackmagic Film Generation 5 transfer",
        "target": "ITU-R BT.709 primaries, BT.1886 display encoding (gamma 2.4, Lw=1, Lb=0)",
        "matrix_bmdwg_to_709": M_BMDWG_TO_709.round(6).tolist(),
        "tone_mapping": "none (hard clip at scene-linear 1.0)",
        "colour_science_version": colour.__version__,
    }
