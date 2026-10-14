"""Metameric pairs (design §7): two surfaces that match in daylight and split under a lamp.

The pair is CONSTRUCTED, not measured. The second sample is a measured base
reflectance plus a "metameric black": a reflectance change that leaves the
colour unchanged under D65. How far the pair splits under a lamp therefore
depends on the change chosen, not on any real material, and the result is a
demonstration of the effect, not a finding about real paints. A measured pair,
or the CIE's standard virtual pairs (ISO/CIE 23603, distributed only with the
standard), would make it a test.

The construction rule:

- *Match:* XYZ under D65 unchanged for both the 2° and the 10° observer
  (six linear conditions). They are imposed in the same arithmetic the studies
  evaluate with (``appearance.tristimulus``), so the match is exact there.
- *Smooth:* the change is a cosine series over 380–780 nm up to order 8
  (no feature narrower than a 50 nm half-period), held at its end values
  beyond, like the sample tables. This choice matters most, and it was made
  after a scan of orders 6–20 whose splits were seen: order 6 gives splits
  below 0.4, order 10 splits 1.2–1.5× those of order 8. Order 8 was picked by a
  test that does not involve the lamps: the partner is then about as steep as
  the steepest ColorChecker patch (0.024 against 0.0237 per nm).
- *Direction:* among such changes, the one with the largest XYZ change under
  CIE Illuminant A per unit RMS reflectance change. Daylight-to-tungsten is the
  CIE's standard change of illuminant for pairs matched in daylight; the
  direction is not tuned to any of the lamps the pair is then shown under. A
  still splits the pair least of the lamps shown: it is smooth like D65, so
  even this change barely separates under it, while spiky lamps pick out the
  change's wiggles.
- *Size:* the largest that keeps the second sample between 0.02 and 0.98 at
  every wavelength (a realisable reflectance).
"""

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import eigh, null_space

from lss.provenance import code_version
from lss.spectral import colorimetry as cm
from lss.spectral.continuum import TabulatedContinuum
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import Spectrum
from lss.studies import appearance as ap
from lss.studies import samples

BASIS_ORDER = 8  # highest cosine order: shortest half-period 400/8 = 50 nm
BASIS_RANGE_NM = (380.0, 780.0)
REFLECTANCE_LIMITS = (0.02, 0.98)
DEFAULT_BASE = "neutral 6.5 (.44 D)"


def d65() -> ap.Light:
    """CIE D65 from its tabulated SPD, binned exactly like a reference lamp."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour

    sd = colour.SDS_ILLUMINANTS["D65"]
    spectrum = Spectrum.of(TabulatedContinuum(np.asarray(sd.wavelengths, float), np.asarray(sd.values, float)))
    return ap.Light("CIE D65", spectrum.bin(analysis_grid()).values)


def illuminant_a() -> ap.Light:
    from lss.lamp import lamp as lamps

    return ap.lamp_light(lamps.load("ref:cie-a"), 0)


def _basis() -> np.ndarray:
    """Cosine series cos(kπx), k = 0 … BASIS_ORDER, x running 0 → 1 over the basis range, flat beyond."""
    lo, hi = BASIS_RANGE_NM
    x = (np.clip(analysis_grid().centres, lo, hi) - lo) / (hi - lo)
    return np.cos(np.pi * np.arange(BASIS_ORDER + 1)[None, :] * x[:, None])


def _weights(light: ap.Light, observer: str) -> np.ndarray:
    """(3, bins): XYZ of a reflectance under ``light`` (white Y = 100) is weights @ R."""
    w = light.power[:, None] * cm.cmfs(observer)
    return (100.0 / w[:, 1].sum()) * w.T


def metameric_black(match: ap.Light, split: ap.Light) -> np.ndarray:
    """Unit-RMS smooth reflectance change, invisible under ``match`` (2° and 10°), largest under ``split``."""
    phi = _basis()
    constraints = np.vstack([_weights(match, "2"), _weights(match, "10")]) @ phi
    constraints /= np.linalg.norm(constraints, axis=1, keepdims=True)
    free = phi @ null_space(constraints)  # smooth changes that keep the match
    response = _weights(split, "10") @ free
    g = response.T @ response
    h = free.T @ free / free.shape[0]  # mean square over the bins
    _, vectors = eigh(g, h)
    black = free @ vectors[:, -1]
    return black / np.sqrt(np.mean(black**2))


@dataclass(frozen=True)
class Pair:
    base: str
    first: np.ndarray  # the measured base reflectance on the analysis grid
    second: np.ndarray  # base + amplitude × black
    amplitude: float  # RMS reflectance change
    match: str
    design: str

    @property
    def reflectances(self) -> np.ndarray:
        return np.stack([self.first, self.second], axis=1)


def make(base: str = DEFAULT_BASE, match: ap.Light | None = None, design: ap.Light | None = None) -> Pair:
    """The constructed pair from a ColorChecker patch, by the module's fixed rule."""
    match = match or d65()
    design = design or illuminant_a()
    chart = samples.get("colorchecker")
    first = np.array(chart.values[:, chart.index(base)])
    black = metameric_black(match, design)
    lo, hi = REFLECTANCE_LIMITS
    best = 0.0
    for sign in (1.0, -1.0):  # the black's sign is arbitrary; take the one with more room
        b = sign * black
        with np.errstate(divide="ignore"):
            room = np.where(b > 0, (hi - first) / b, np.where(b < 0, (lo - first) / b, np.inf))
        a = float(room.min())
        if a > abs(best):
            best = sign * a
    if best == 0.0:
        raise ValueError(f"'{base}' lies outside {lo:g}-{hi:g} somewhere: no room for a partner")
    return Pair(base, first, first + best * black, abs(best), match.name, design.name)


@dataclass(frozen=True)
class Split:
    light: str
    delta_e: float  # CAM02-UCS ΔE′ between the two samples under this light
    srgb_first: str
    srgb_second: str


def split(pair: Pair, light: ap.Light) -> Split:
    jab = ap.cam02ucs(light, pair.reflectances)
    rgb, _ = ap.srgb(light, pair.reflectances)
    return Split(light.name, float(np.linalg.norm(jab[0] - jab[1])), ap.hex_colour(rgb[0]), ap.hex_colour(rgb[1]))


@dataclass(frozen=True)
class Study:
    base: str
    amplitude: float
    match: str
    design: str
    splits: list[Split]
    constructed: bool  # always True until a measured pair exists
    rule: str
    code_version: str

    def as_dict(self) -> dict:
        return asdict(self)


RULE = (f"base + metameric black: XYZ unchanged under D65 (2 and 10 deg); cosine series to order {BASIS_ORDER} "
        f"over {BASIS_RANGE_NM[0]:g}-{BASIS_RANGE_NM[1]:g} nm; direction of largest change under CIE A; "
        f"size the largest keeping {REFLECTANCE_LIMITS[0]:g}-{REFLECTANCE_LIMITS[1]:g}")


def study(lights: list[ap.Light], base: str = DEFAULT_BASE) -> tuple[Pair, Study]:
    pair = make(base)
    rows = [split(pair, d65())] + [split(pair, light) for light in lights]
    return pair, Study(base, pair.amplitude, pair.match, pair.design, rows, True, RULE, code_version())


def format_text(s: Study) -> str:
    lines = [
        f"Metameric pair: ColorChecker '{s.base}' and a CONSTRUCTED partner",
        "  matched under CIE D65; how far it splits depends on the chosen change (a demonstration,",
        "  not a measured pair). Rule: " + s.rule,
        f"  RMS reflectance change {s.amplitude:.3f}",
        "",
        f"  {'light':<28}{'dE':>7}   {'first':<8} {'second':<8}",
    ]
    for row in s.splits:
        lines.append(f"  {row.light:<28}{row.delta_e:7.2f}   {row.srgb_first} {row.srgb_second}")
    lines += [
        "",
        "  dE: CAM02-UCS difference between the two samples under each light (TM-30's conditions).",
        "  It is weak as a scale under a very orange light such as HPS: compare lamps, not exact values.",
        f"  code: {s.code_version}",
    ]
    return "\n".join(lines)
