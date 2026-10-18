"""Flame study: soot's glow against LICA's lighter flame (design §4.5, first piece).

The flame is LICA-UCM's "Propane lighter" (:mod:`lss.accuracy.lica`): relative, 300–900 nm
at 1 nm, on the Jaz spectrograph of the street lamps. Each model has its temperature and
a scale fitted, by least squares on the relative residual (model/measured − 1) at the
window's nodes. The instrument's 1.6 nm response does nothing to a smooth continuum, so it
is not applied.

The rules were fixed before any fit (design §4.5): the window, the masks over band heads
and alkali lines, and both treatments of the flat floor the file sits on below 400 nm.
Soot's change of absorption with wavelength is never fitted, since it trades off with the
temperature; each set of optics in ``lss/data/soot_optics.yaml`` is run, and their spread
is the temperature's uncertainty.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from lss.accuracy import lica
from lss.physics.flame import soot
from lss.spectral.continuum import PlanckContinuum

WINDOW_NM = (450.0, 850.0)
#: C₂ Swan band heads (Δv = +1, 0, −1 at 473.7, 516.5, 563.6 nm), Na D and K, in case.
MASKS_NM = ((465.0, 476.0), (505.0, 519.0), (553.0, 566.0), (585.0, 593.0), (763.0, 773.0))
#: Where the file's flat floor is read: no soot at 2000 K reaches it here.
FLOOR_NM = (320.0, 380.0)
#: Optical depths at 550 nm run with Chang's optics (the thin limit is 1e-6).
DEPTHS = (0.1, 0.3, 1.0)
THIN = 1e-6
#: Propane in air, adiabatic, at 1 atm: a hard ceiling for soot's temperature (about 2270 K).
ADIABATIC_PROPANE_K = 2270.0
BAND_NM = 50.0


def models() -> dict:
    """name -> temperature -> continuum."""
    out = {"grey body (L1)": lambda t: PlanckContinuum(t)}
    for name in soot.sets():
        out[f"soot, thin, {name}"] = lambda t, n=name: soot.SootContinuum(t, THIN, n)
    for tau in DEPTHS:
        out[f"soot, depth {tau:g}, chang1990"] = lambda t, d=tau: soot.SootContinuum(t, d)
    return out


@dataclass(frozen=True)
class Measured:
    wavelengths_nm: np.ndarray  # the window's nodes, masks removed
    values: np.ndarray
    floor: float  # mean over FLOOR_NM, on the file's own scale
    floor_subtracted: bool


@lru_cache(maxsize=2)
def measured(floor_subtracted: bool) -> Measured:
    m = lica.load("lighter")
    x, y = m.wavelengths_nm, m.values
    floor = float(y[(x >= FLOOR_NM[0]) & (x <= FLOOR_NM[1])].mean())
    keep = (x >= WINDOW_NM[0]) & (x <= WINDOW_NM[1])
    for lo, hi in MASKS_NM:
        keep &= ~((x >= lo) & (x <= hi))
    y = y[keep] - (floor if floor_subtracted else 0.0)
    return Measured(x[keep], y, floor, floor_subtracted)


@dataclass(frozen=True)
class FlameFit:
    model: str
    floor_subtracted: bool
    temperature_k: float
    rms: float  # rms of model/measured − 1 over the window's nodes
    bins_nm: tuple[tuple[float, float], ...]
    mean_residual: tuple[float, ...]  # (measured − model)/measured, per bin


def fit(model: str, floor_subtracted: bool) -> FlameFit:
    make = models()[model]
    m = measured(floor_subtracted)
    x, y = m.wavelengths_nm, m.values
    ref = np.array([650.0])

    def shape(t):
        c = make(t)
        return c.density(x) / c.density(ref)[0]

    def residuals(p):
        return p[1] * shape(p[0]) / y - 1.0

    r = least_squares(residuals, [2000.0, float(np.interp(650.0, x, y))],
                      bounds=([1000.0, 0.0], [3500.0, np.inf]), x_scale=[100.0, 0.1])
    rel = -r.fun  # (measured − model)/measured
    edges = np.arange(WINDOW_NM[0], WINDOW_NM[1], BAND_NM)
    bins = tuple((float(a), float(a + BAND_NM)) for a in edges)
    means = tuple(float(rel[(x >= a) & (x < b)].mean()) for a, b in bins)
    return FlameFit(model, floor_subtracted, float(r.x[0]), float(np.sqrt(np.mean(r.fun**2))), bins, means)


def fits() -> list[FlameFit]:
    return [fit(name, sub) for sub in (False, True) for name in models()]


def format_text() -> str:
    m = lica.load("lighter")
    floor = measured(True).floor
    at = {w: float(np.interp(w, m.wavelengths_nm, m.values)) for w in (450.0, 500.0, 650.0)}
    out = [f"Measured flame: {lica.CITATION}",
           f"  file {lica.FILES['lighter'][0]}, header: \"{m.header.get('Other data', '')}\"", "",
           f"Window {WINDOW_NM[0]:g}-{WINDOW_NM[1]:g} nm; masked: "
           + ", ".join(f"{a:g}-{b:g}" for a, b in MASKS_NM) + " nm.",
           f"Floor (mean {FLOOR_NM[0]:g}-{FLOOR_NM[1]:g} nm): {floor:.4f}, "
           + ", ".join(f"{100 * floor / v:.0f} % of the signal at {w:g} nm" for w, v in at.items()) + ".",
           "Temperature and scale fitted; soot's change with wavelength fixed by its optics.",
           f"Propane-air adiabatic flame temperature, the ceiling: about {ADIABATIC_PROPANE_K:g} K.", ""]
    head = f"{'':34s}{'T (K)':>8s}{'rms':>7s}   (measured - model)/measured per {BAND_NM:g} nm from {WINDOW_NM[0]:g} nm"
    for sub in (False, True):
        out += ["Floor subtracted:" if sub else "Floor as measured:", head]
        for f in fits():
            if f.floor_subtracted != sub:
                continue
            flag = "  above the ceiling" if f.temperature_k > ADIABATIC_PROPANE_K else ""
            out.append(f"  {f.model:32s}{f.temperature_k:8.0f}{f.rms:7.3f}   "
                       + " ".join(f"{v:+.3f}" for v in f.mean_residual) + flag)
        out.append("")
    return "\n".join(out).rstrip()
