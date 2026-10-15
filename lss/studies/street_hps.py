"""Street-lamp study: the rung-1 sodium arc against nine measured HPS lamps (design §4.4, §15).

The lamps are LICA-UCM's (:mod:`lss.accuracy.lica`): high-pressure sodium street
lamps measured in the street with a Jaz spectrograph, 1 nm steps, relative values, no
electrical data or arc-tube size. The model is the assumed arc of design §4.4 (nothing
fitted to these lamps), passed through the instrument's response before it is compared.

The response is fitted to LICA's mercury pencil lamp: three isolated lines, far
narrower than the instrument, at wavelengths measured by Sansonetti, Salit & Reader
(1996; ``lss/data/hg_lines.yaml``). It is checked on LICA's low-pressure sodium lamp,
whose lines are equally narrow. A second response, fitted to two argon lines of LICA's
thorium–argon lamp (another lamp, near 750 nm), shows how much each comparison depends
on the response; so does an extra offset of ±0.5 nm.

What can be compared (design §15): the HPS peaks, about 10 nm apart, are resolved; the
dip between them is not. So the study compares where the peaks are, how much power each
subordinate line carries against the D lines, and how bright the far wings are.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from lss.accuracy import lica
from lss.physics.hid import arc, sodium
from lss.physics.hid.composition import Fill
from lss.physics.thermal.tungsten import TungstenCoilContinuum
from lss.spectral.instrument import Kernel, KernelFit, fit_kernel, observe

#: Hg I lines of the pencil lamp, isolated at the instrument's resolution (air, nm;
#: Sansonetti, Salit & Reader, Appl. Opt. 35, 74 (1996)). 407.78 nm is 3.1 nm from
#: 404.66 and 6 % of it; 434.75 nm is 1 nm from 435.83 and 0.3 % of it.
HG_LINES_NM = (404.6565, 435.8335, 546.0750)
#: Ar I lines of the thorium–argon lamp with no Ar I or Ne I line above a fifth of their
#: NIST intensity within 3 nm (NIST ASD v5.12, observed air wavelengths, retrieved 2026-10-15).
AR_LINES_NM = (738.3980, 763.5106)
#: Na I doublets of the low-pressure sodium lamp used to check the response (NIST ASD; air, nm).
#: The 615/616 nm pair is left out: the lamp's neon starting gas has lines beside it.
LPS_DOUBLETS_NM = {"589 (D)": (588.9950, 589.5924), "568.8": (568.2633, 568.8205), "819": (818.3256, 819.4790)}

#: The assumed arc of design §4.4: not taken from a source or from these lamps.
ASSUMED_ARC = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=4000.0, wall_temperature_k=1500.0,
                            fill=Fill(sodium_pa=13e3, mercury_pa=70e3, xenon_pa=20e3), shells=60)
NODES_NM = np.arange(300.0, 901.0)

#: Subordinate lines: power above a straight baseline through 2 nm beside each edge.
BANDS_NM = {"498": (495.0, 501.0), "515": (512.0, 518.0), "568.8": (565.0, 572.0),
            "616": (612.0, 619.0), "819": (814.0, 825.0)}
#: The D lines' power, against which each band's is given.
D_BAND_NM = (575.0, 605.0)
#: Far wings, as their mean over the D band's highest sample. 525–540 nm is clear of
#: lines; nearer the D lines the measured lamps hold the Hg 546 nm line and a rise from
#: about 548 nm that the rung-1 model has no counterpart of.
RED_WING_NM = (640.0, 700.0)
BLUE_WING_NM = (525.0, 540.0)


@lru_cache(maxsize=None)
def response(lamp: str = "hg-pencil") -> KernelFit:
    """The instrument's response fitted to ``hg-pencil`` (default) or ``thorium-argon``."""
    lines = {"hg-pencil": HG_LINES_NM, "thorium-argon": AR_LINES_NM}[lamp]
    m = lica.load(lamp)
    return fit_kernel(m.wavelengths_nm, m.values, lines, drift=False)


def lps_check(kernel: Kernel | None = None) -> dict[str, float]:
    """Each LPS doublet fitted with the response held fixed (two powers and a straight
    background free): rms residual over the window's highest sample."""
    kernel = kernel or response().kernel
    m = lica.load("lps")
    out = {}
    for name, lines in LPS_DOUBLETS_NM.items():
        w = (m.wavelengths_nm >= np.mean(lines) - 5.0) & (m.wavelengths_nm <= np.mean(lines) + 5.0)
        x, y = m.wavelengths_nm[w], m.values[w]

        def residuals(p):
            return sum(kernel.line(x, c, a) for c, a in zip(lines, p[:2])) + p[2] + p[3] * (x - x.mean()) - y

        r = least_squares(residuals, [y.max(), y.max(), 0.0, 0.0], bounds=([0, 0, -np.inf, -np.inf], np.inf))
        out[name] = float(np.sqrt(np.mean(r.fun**2)) / y.max())
    return out


@dataclass(frozen=True)
class FluxCheck:
    temperature_k: float
    open_fraction: float
    bins_nm: tuple[tuple[float, float], ...]
    mean_relative_residual: tuple[float, ...]  # (measured − model)/measured, per bin


def flux_check(lo_nm: float = 450.0, hi_nm: float = 850.0, bin_nm: float = 50.0) -> FluxCheck:
    """LICA's halogen lamp against a coiled tungsten filament (design §4.1 L3), temperature,
    open fraction and scale free. A smooth tilt in LICA's response correction is absorbed
    by the temperature; anything less smooth would show in the residual."""
    m = lica.load("halogen")
    w = (m.wavelengths_nm >= lo_nm) & (m.wavelengths_nm <= hi_nm)
    x, y = m.wavelengths_nm[w], m.values[w]

    def residuals(p):
        c = TungstenCoilContinuum(p[0], open_fraction=p[1])
        return p[2] * c.density(x) / c.density(np.array([600.0]))[0] - y

    r = least_squares(residuals, [2800.0, 0.5, float(np.interp(600.0, x, y))],
                      bounds=([1700.0, 0.05, 0.0], [3150.0, 1.0, np.inf]), x_scale=[100.0, 0.1, 0.1])
    rel = -r.fun / y
    edges = np.arange(lo_nm, hi_nm, bin_nm)
    bins = tuple((float(a), float(a + bin_nm)) for a in edges)
    means = tuple(float(rel[(x >= a) & (x < b)].mean()) for a, b in bins)
    return FluxCheck(float(r.x[0]), float(r.x[1]), bins, means)


@lru_cache(maxsize=1)
def model_radiance(column: arc.ArcColumn = ASSUMED_ARC) -> tuple[np.ndarray, np.ndarray]:
    """(air wavelength nm, mean side-on radiance) of the arc on a grid resolving every line."""
    grid = arc.line_grid(sodium.data().lines, lo_nm=280.0, hi_nm=920.0, base_step_nm=0.05)
    return grid, arc.side_radiance(column, grid).mean_radiance


def _peak(x, y, lo, hi) -> float:
    """Vertex of the parabola through the highest sample in [lo, hi] and its neighbours."""
    idx = np.where((x >= lo) & (x <= hi))[0]
    i = idx[np.argmax(y[idx])]
    a, b, c = y[i - 1], y[i], y[i + 1]
    return float(x[i] + 0.5 * (a - c) / (a - 2.0 * b + c))


def _band(x, y, lo, hi, side=2.0) -> float:
    left = y[(x >= lo - side) & (x < lo)].mean()
    right = y[(x > hi) & (x <= hi + side)].mean()
    w = (x >= lo) & (x <= hi)
    return float(np.sum(y[w] - (left + (right - left) * (x[w] - lo) / (hi - lo))))


@dataclass(frozen=True)
class Features:
    blue_peak_nm: float
    red_peak_nm: float
    red_wing: float  # mean over RED_WING_NM, over the D band's highest sample
    blue_wing: float
    line_power: dict  # band name -> power over the D band's

    @property
    def separation_nm(self) -> float:
        return self.red_peak_nm - self.blue_peak_nm


def features(x, y) -> Features:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = (x >= D_BAND_NM[0]) & (x <= D_BAND_NM[1])
    top, d_power = y[d].max(), y[d].sum()

    def wing(lo, hi):
        return float(y[(x >= lo) & (x <= hi)].mean() / top)

    return Features(_peak(x, y, D_BAND_NM[0], 589.5), _peak(x, y, 589.5, D_BAND_NM[1]),
                    wing(*RED_WING_NM), wing(*BLUE_WING_NM),
                    {k: _band(x, y, lo, hi) / d_power for k, (lo, hi) in BANDS_NM.items()})


def model_features(kernel: Kernel, column: arc.ArcColumn = ASSUMED_ARC) -> Features:
    lam, radiance = model_radiance(column)
    return features(NODES_NM, observe(lam, radiance, kernel, NODES_NM))


def response_variants() -> dict[str, Kernel]:
    hg = response("hg-pencil").kernel
    return {"Hg pencil": hg, "Th-Ar": response("thorium-argon").kernel,
            "Hg pencil, +0.5 nm": replace(hg, offset_nm=hg.offset_nm + 0.5),
            "Hg pencil, -0.5 nm": replace(hg, offset_nm=hg.offset_nm - 0.5)}


def measured_features() -> dict[str, Features]:
    out = {}
    for key in lica.HPS:
        m = lica.load(key)
        out[key] = features(m.wavelengths_nm, m.values)
    return out


def _row(name, f: Features) -> str:
    return (f"{name:28s}{f.blue_peak_nm:8.1f}{f.red_peak_nm:8.1f}{f.separation_nm:7.1f}"
            f"{f.blue_wing:8.3f}{f.red_wing:8.3f}" + "".join(f"{f.line_power[k]:8.3f}" for k in BANDS_NM))


def format_text() -> str:
    hg, ar = response("hg-pencil"), response("thorium-argon")
    out = [f"Measured lamps: {lica.CITATION}", "",
           "Instrument response (Gaussian core convolved with a red exponential tail):"]
    for name, r in (("Hg pencil (404.7, 435.8, 546.1 nm)", hg), ("Th-Ar (738.4, 763.5 nm)", ar)):
        k = r.kernel
        out.append(f"  {name:36s} FWHM {k.fwhm_nm:.2f} nm, peak {k.mode_nm:+.2f} nm from the line, "
                   f"rms residual {r.rms_over_peak:.3f} of the peak")
    out.append("  checked on the low-pressure sodium lamp (Hg pencil response, rms residual / peak): "
               + ", ".join(f"{k} nm {v:.3f}" for k, v in lps_check().items()))
    fc = flux_check()
    out.append(f"LICA's response correction: halogen lamp against coiled tungsten at {fc.temperature_k:.0f} K "
               f"(open fraction {fc.open_fraction:.2f}): measured/model - 1 per 50 nm from 450 nm: "
               + " ".join(f"{v:+.3f}" for v in fc.mean_relative_residual))
    out += ["", "Assumed arc (design §4.4; nothing fitted): T 4000 K on the axis, 1500 K at a 3.8 mm wall; "
                "Na 13 kPa, Hg 70 kPa, Xe 20 kPa.",
            "Line powers are over the D lines' (575-605 nm); wings are mean over the D peak.", "",
            f"{'':28s}{'blue pk':>8s}{'red pk':>8s}{'sep':>7s}{'525-540':>8s}{'640-700':>8s}"
            + "".join(f"{k:>8s}" for k in BANDS_NM)]
    for name, k in response_variants().items():
        out.append(_row(f"model, {name}", model_features(k)))
    meas = measured_features()
    for key, f in meas.items():
        out.append(_row(key, f))
    return "\n".join(out)
