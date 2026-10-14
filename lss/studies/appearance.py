"""Surface colour under a light (design §7): tristimulus, CAM02-UCS and display sRGB.

A :class:`Light` is power per analysis-grid bin. A surface of reflectance R
(``lss.studies.samples``) under it has XYZ = Σ_bins power · R(centre) · cmf(centre),
scaled so the perfect white has Y = 100 under every light: the spectra are
relative, and the eye adapts to the light's own white.

Colour differences follow ANSI/IES TM-30-18 (CIE 224:2017): CIE 1964 10°
observer, CIECAM02 with L_A = 100 cd/m², Y_b = 20, average surround and the
illuminant discounted (full adaptation), then CAM02-UCS. The reference light at
the same CCT is TM-30's own, so a chart's colour shifts and the lamp sheet's Rf
measure the same thing. Display colours use the 2° observer (sRGB is defined on
it) and CAT02 adaptation from the light's white to D65.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from lss.spectral import colorimetry as cm
from lss.spectral.grids import analysis_grid

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import colour
    from colour.quality import cfi2017

L_A = 100.0  # adapting luminance, cd/m² (TM-30-18)
Y_B = 20.0  # relative background luminance (TM-30-18)
TM30_RANGE = (380.0, 780.0)  # nm; TM-30 assigns the reference CCT from this range


@dataclass(frozen=True)
class Light:
    name: str
    power: np.ndarray  # relative power per analysis-grid bin

    def __post_init__(self):
        p = np.asarray(self.power, dtype=float)
        if p.shape != (len(analysis_grid()),):
            raise ValueError("a light is power per analysis-grid bin")
        object.__setattr__(self, "power", p)


def lamp_light(lamp, level: int) -> Light:
    """A lamp at a model level, through its own envelope (the quick path)."""
    from lss.lamp import quickpath

    binned = quickpath.apply(lamp, lamp.generate(level)).spectrum.bin(analysis_grid())
    return Light(f"{lamp.id} L{level}", binned.values)


def reference_light(light: Light) -> tuple[Light, float]:
    """TM-30's reference illuminant for ``light`` (Planck, D series or their blend) and its CCT."""
    centres = analysis_grid().centres
    inside = (centres >= TM30_RANGE[0]) & (centres <= TM30_RANGE[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = colour.SpectralDistribution(light.power[inside], centres[inside])
        cct, _ = cfi2017.CCT_reference_illuminant(sd)
        shape = colour.SpectralShape(centres[0], centres[-1], 1.0)
        ref = cfi2017.sd_reference_illuminant(float(cct), shape)
    # A 1 nm density sampled at the bin centre, times the 1 nm bin width.
    return Light(f"reference {cct:.0f} K", np.asarray(ref.values, float) * analysis_grid().widths), float(cct)


def tristimulus(light: Light, reflectance: np.ndarray, observer: str = "10") -> tuple[np.ndarray, np.ndarray]:
    """XYZ of each sample (samples, 3) and of the perfect white, with the white's Y = 100."""
    weights = light.power[:, None] * cm.cmfs(observer)  # (bins, 3)
    white = weights.sum(axis=0)
    k = 100.0 / white[1]
    xyz = np.atleast_2d(np.asarray(reflectance, float).T) @ weights
    return k * xyz, k * white


def cam02ucs(light: Light, reflectance: np.ndarray) -> np.ndarray:
    """CAM02-UCS J′a′b′ of each sample under ``light`` in TM-30's viewing conditions."""
    xyz, white = tristimulus(light, reflectance, "10")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = colour.XYZ_to_CIECAM02(xyz, white, L_A, Y_B, colour.VIEWING_CONDITIONS_CIECAM02["Average"],
                                      discount_illuminant=True, compute_H=False)
        return colour.models.JMh_CIECAM02_to_CAM02UCS(np.stack([spec.J, spec.M, spec.h], axis=-1))


def srgb(light: Light, reflectance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Display colour of each sample adapted to D65: encoded sRGB in [0, 1] and an out-of-gamut flag."""
    xyz, white = tristimulus(light, reflectance, "2")
    white_xy = white[:2] / white.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb = colour.XYZ_to_sRGB(xyz / 100.0, illuminant=white_xy, chromatic_adaptation_transform="CAT02")
    rgb = np.atleast_2d(rgb)
    out = np.any((rgb < -1e-3) | (rgb > 1.0 + 1e-3), axis=1)
    return np.clip(rgb, 0.0, 1.0), out


def hex_colour(rgb: np.ndarray) -> str:
    r, g, b = (int(round(255 * c)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"
