"""Each street lamp's arc temperature and sodium pressure, fitted (design §4.4, roadmap step 3).

The street-lamp study (:mod:`lss.studies.street_hps`) runs all nine lamps through one
assumed arc. Here each lamp gets its own axis temperature and sodium pressure; the
radius (3.8 mm), wall temperature (1500 K), parabolic profile, mercury (70 kPa) and
xenon (20 kPa) stay as assumed. With mercury held, more sodium also means fewer Na–Hg
pairs per Na–Na pair.

What is fitted, and what is held out, was set before any fit was run:

- *Fitted:* the blue D peak's wavelength, which in the model moves with sodium pressure
  (σ 0.3 nm), and the power of the 498, 515 and 568.8 nm lines over the D lines',
  which move with temperature (σ 10 % each). 819 nm is shown with and without: the
  instrument check fits it worst. 616 nm is left out (its baseline windows hold the
  pair absorption's peaks, design §4.4).
- *Not the peak spacing:* the red peak's place in the model depends on where the pair
  wings are spliced on (3, 6 or 12 nm from the lines move it by several nm); the blue
  peak's does not.
- *Held out:* the peak spacing, the red wing's level and slope, the 668–678 nm rise and
  the blue wing. The fit exists to ask whether the red wing's shortfall is the assumed
  arc's conditions; a fit that saw the red wing could not answer that.

Every physics variant left open in design §4.4 is a separate table: a central one
(sodium and mercury pairs, constant dipoles, Unsöld's ΔC₆, 6 nm splice) and one table
per change from it. A fitted value is reported as its range across them.

Measured values, for reading the fits (not used as bounds), from J. J. de Groot and
J. A. J. M. van Vliet, *The High-Pressure Sodium Lamp* (Philips Technical Library,
Macmillan 1986, doi:10.1007/978-1-349-09196-6):

- axis temperature: "between 4100 and 4300 K" in standard 400 W Na+Hg lamps, the
  consensus of six measurements (p. 133); about 4000 K without mercury (Table 3.5, p. 89);
- sodium pressure from the peak spacing Δλ: p_Na·√R/Δλ ≈ 2.7 ± 0.5 kPa nm⁻¹ mm^½ by
  the spectroscopic method (eq. 3.27, p. 85), 1.9 ± 0.5 from metal-bath experiments
  (eq. 3.28, p. 87); with R = 3.8 mm and the lamps' 7.8–19.2 nm, 11–27 kPa by the first;
- colour-improved lamps: Δλ about 25 and 45 nm (p. 301), wider than any of the nine.

The lamps' tube radius is not known. At a fixed spacing the rule scales p_Na as R^−½, so
a fitted sodium pressure is the one a 3.8 mm tube would need.

Each table is a grid over axis temperature and sodium pressure of the spectrum as the
instrument records it (1 nm nodes), cached on disk because it takes minutes. The fit
reads the features off it by cubic splines in (T, log p).
"""

import hashlib
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares

from lss import download
from lss.physics.hid import arc, mercury_pairs, pairs
from lss.spectral.instrument import Kernel, observe
from lss.studies import street_hps

#: Bump when the arc model changes, so cached tables are rebuilt.
TABLE_VERSION = 1
GRID_T_K = np.arange(3400.0, 4801.0, 200.0)
GRID_NA_PA = np.geomspace(5e3, 45e3, 23)

CENTRAL = replace(street_hps.ASSUMED_ARC, pairs=pairs.PairWings(splice_nm=6.0, dipole="constant"),
                  mercury_pairs=mercury_pairs.MercuryPairWings(splice_nm=6.0, dipole="constant", delta_c6_factor=1.0))


def _with_splice(column: arc.ArcColumn, splice_nm: float) -> arc.ArcColumn:
    return replace(column, pairs=replace(column.pairs, splice_nm=splice_nm),
                   mercury_pairs=replace(column.mercury_pairs, splice_nm=splice_nm))


#: The central arc, and one change from it per open choice of design §4.4.
VARIANTS = {
    "central": CENTRAL,
    "Hg pairs dC6 x2": replace(CENTRAL, mercury_pairs=replace(CENTRAL.mercury_pairs, delta_c6_factor=2.0)),
    "Hg pairs falling dipole": replace(CENTRAL, mercury_pairs=replace(CENTRAL.mercury_pairs, dipole="falling")),
    "Na pairs long-range dipole": replace(CENTRAL, pairs=replace(CENTRAL.pairs, dipole="long-range")),
    "splice 3 nm": _with_splice(CENTRAL, 3.0),
    "splice 12 nm": _with_splice(CENTRAL, 12.0),
}

#: Fitted targets: name -> σ. Line powers are compared as logarithms (σ is a fraction).
PEAK_SIGMA_NM = 0.3
LINE_SIGMA = 0.1
FIT_LINES = ("498", "515", "568.8")


def at(column: arc.ArcColumn, axis_temperature_k: float, sodium_pa: float) -> arc.ArcColumn:
    return replace(column, axis_temperature_k=float(axis_temperature_k),
                   fill=replace(column.fill, sodium_pa=float(sodium_pa)))


def observed(column: arc.ArcColumn, kernel: Kernel) -> np.ndarray:
    """The arc's mean side-on radiance as the instrument records it at the 1 nm nodes."""
    lam, radiance = street_hps.model_radiance(column)
    return observe(lam, radiance, kernel, street_hps.NODES_NM)


def _observed(job):
    return observed(*job)


def _key(column: arc.ArcColumn, kernel: Kernel) -> str:
    text = json.dumps([TABLE_VERSION, repr(column), repr(kernel), GRID_T_K.tolist(), GRID_NA_PA.tolist()])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def table_path(variant: str, kernel: Kernel | None = None) -> Path:
    kernel = kernel or street_hps.response().kernel
    return download.cache_dir() / "hid_tables" / f"{_key(VARIANTS[variant], kernel)}.npy"


def table(variant: str, kernel: Kernel | None = None, workers: int = 16) -> np.ndarray:
    """Recorded spectra, shape (temperature, sodium pressure, node), cached on disk."""
    kernel = kernel or street_hps.response().kernel
    column = VARIANTS[variant]
    path = table_path(variant, kernel)
    if path.exists():
        return np.load(path)
    jobs = [(at(column, t, p), kernel) for t in GRID_T_K for p in GRID_NA_PA]
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_observed, jobs)
    else:
        rows = [_observed(j) for j in jobs]
    out = np.array(rows).reshape(GRID_T_K.size, GRID_NA_PA.size, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.npy")
    np.save(tmp, out)
    tmp.replace(path)
    return out


def _vector(f: street_hps.Features) -> dict[str, float]:
    """Every feature, fitted or held out, as one flat dict (line powers as logarithms)."""
    out = {"blue peak": f.blue_peak_nm, "separation": f.separation_nm, "red wing": np.log(f.red_wing),
           "blue wing": np.log(f.blue_wing), "red slope": f.red_slope, "red bump": f.red_bump}
    for k, v in f.line_power.items():
        out[k] = np.log(v) if v > 0 else np.nan
    return out


@dataclass(frozen=True)
class Surface:
    """Each feature of one variant's table as a smooth function of (T, sodium pressure)."""

    variant: str
    splines: dict

    def __call__(self, temperature_k: float, sodium_pa: float) -> dict[str, float]:
        return {k: float(s(temperature_k, np.log(sodium_pa))[0, 0]) for k, s in self.splines.items()}


@lru_cache(maxsize=None)
def surface(variant: str) -> Surface:
    spectra = table(variant)
    feats = [[_vector(street_hps.features(street_hps.NODES_NM, spectra[i, j]))
              for j in range(GRID_NA_PA.size)] for i in range(GRID_T_K.size)]
    keys = feats[0][0].keys()
    splines = {}
    for k in keys:
        z = np.array([[row[j][k] for j in range(GRID_NA_PA.size)] for row in feats])
        if np.all(np.isfinite(z)):
            splines[k] = RectBivariateSpline(GRID_T_K, np.log(GRID_NA_PA), z)
    return Surface(variant, splines)


@dataclass(frozen=True)
class Fit:
    variant: str
    lamp: str
    with_819: bool
    temperature_k: float
    sodium_pa: float
    chi2_per_target: float
    at_bound: bool
    model: dict  # every feature at the fitted point
    measured: dict


def fit(surf: Surface, measured: street_hps.Features, lamp: str = "", with_819: bool = False,
        peak_offset_nm: float = 0.0) -> Fit:
    """Axis temperature and sodium pressure from the blue peak and the line powers.

    ``peak_offset_nm`` is added to the measured peak, to show how much the instrument's
    wavelength offset (±0.5 nm in the street-lamp study) moves the sodium pressure.
    """
    meas = _vector(measured)
    meas["blue peak"] += peak_offset_nm
    lines = FIT_LINES + (("819",) if with_819 else ())

    def residuals(p):
        m = surf(p[0], np.exp(p[1]))
        return np.array([(m["blue peak"] - meas["blue peak"]) / PEAK_SIGMA_NM]
                        + [(m[k] - meas[k]) / LINE_SIGMA for k in lines])

    lo = [GRID_T_K[0], np.log(GRID_NA_PA[0])]
    hi = [GRID_T_K[-1], np.log(GRID_NA_PA[-1])]
    best = None
    for t0 in (3800.0, 4200.0):  # two starts; keep the better
        for p0 in (10e3, 20e3):
            r = least_squares(residuals, [t0, np.log(p0)], bounds=(lo, hi), x_scale=[100.0, 0.2])
            if best is None or r.cost < best.cost:
                best = r
    t, lp = best.x
    edge = (min(t - lo[0], hi[0] - t) < 5.0) or (min(lp - lo[1], hi[1] - lp) < 0.01)
    return Fit(surf.variant, lamp, with_819, float(t), float(np.exp(lp)),
               float(2.0 * best.cost / best.fun.size), bool(edge), surf(t, np.exp(lp)), meas)


HELD_OUT = ("separation", "red wing", "red slope", "red bump", "blue wing")
#: de Groot & van Vliet (1986), eq. 3.27: p_Na·√R/Δλ, kPa nm⁻¹ mm^½, spectroscopic method.
BOOK_RULE, BOOK_RULE_ERR = 2.7, 0.5
#: The wavelength offset set before the fit, as in the street-lamp study.
PEAK_OFFSET_NM = 0.5


def book_rule_pa(separation_nm: float, radius_m: float = CENTRAL.radius_m) -> float:
    """The sodium pressure the book's rule of thumb gives for a peak spacing."""
    return BOOK_RULE * separation_nm / np.sqrt(radius_m * 1e3) * 1e3


def lamp_cct_k(key: str) -> float:
    """The correlated colour temperature LICA put in the lamp's name."""
    return float(key.split("-")[1].rstrip("k"))


@lru_cache(maxsize=None)
def lps_offset_nm() -> tuple[float, float]:
    """The instrument's wavelength offset near the D lines beyond the mercury response's:
    the LPS D and 568.8 nm doublets fitted with the response, an extra offset free
    (with each doublet's two powers and a straight background), weighted by their
    standard errors. Returns (offset, its standard error), nm; positive means the
    instrument reads lines further to the red than the mercury response says."""
    k0 = street_hps.response().kernel
    m = street_hps.lica.load("lps")
    values, weights = [], []
    for name in ("589 (D)", "568.8"):
        lines = street_hps.LPS_DOUBLETS_NM[name]
        w = (m.wavelengths_nm >= np.mean(lines) - 5.0) & (m.wavelengths_nm <= np.mean(lines) + 5.0)
        x, y = m.wavelengths_nm[w], m.values[w]

        def residuals(p):
            k = replace(k0, offset_nm=k0.offset_nm + p[4])
            return sum(k.line(x, c, a) for c, a in zip(lines, p[:2])) + p[2] + p[3] * (x - x.mean()) - y

        r = least_squares(residuals, [y.max(), y.max(), 0.0, 0.0, 0.0],
                          bounds=([0, 0, -np.inf, -np.inf, -2.0], [np.inf] * 4 + [2.0]))
        cov = np.linalg.inv(r.jac.T @ r.jac) * (2.0 * r.cost / (x.size - 5))
        values.append(r.x[4])
        weights.append(1.0 / cov[4, 4])
    values, weights = np.array(values), np.array(weights)
    return float(np.sum(values * weights) / weights.sum()), float(1.0 / np.sqrt(weights.sum()))


@lru_cache(maxsize=None)
def fits(with_819: bool = False, peak_offset_nm: float = 0.0) -> dict[str, dict[str, Fit]]:
    """variant -> lamp -> fit."""
    meas = street_hps.measured_features()
    return {v: {k: fit(surface(v), f, k, with_819, peak_offset_nm) for k, f in meas.items()} for v in VARIANTS}


def _direct(job):
    variant, t, p = job
    return _vector(street_hps.features(street_hps.NODES_NM, observed(at(VARIANTS[variant], t, p),
                                                                     street_hps.response().kernel)))


@lru_cache(maxsize=None)
def direct(workers: int = 16) -> dict[str, dict[str, dict]]:
    """variant -> lamp -> every feature from a direct run at the lamp's fitted point (no table)."""
    base = fits()
    jobs = [(v, f.temperature_k, f.sodium_pa) for v in VARIANTS for f in base[v].values()]
    with Pool(workers) as pool:
        rows = iter(pool.map(_direct, jobs))
    return {v: {lamp: next(rows) for lamp in base[v]} for v in VARIANTS}


def _held(m: dict, k: str) -> float:
    return float(np.exp(m[k])) if k in ("red wing", "blue wing") else m[k]


def format_text() -> str:
    base, run = fits(), direct()
    off, off_err = lps_offset_nm()
    out = ["Each lamp's axis temperature and sodium pressure, fitted to the blue D peak's wavelength (σ 0.3 nm)",
           "and the 498, 515, 568.8 nm line powers over the D lines' (σ 10 %). Radius, wall, profile, Hg and Xe",
           "as assumed. Tables: " + ", ".join(VARIANTS) + ".", ""]
    out.append(f"{'lamp':28s}{'T central':>10s}{'T range':>13s}{'Na kPa':>8s}{'range':>13s}{'chi2':>6s}"
               f"{'+819: T':>9s}{'Na':>6s}{'pk -/+0.5: Na':>15s}{'LPS off: Na':>12s}{'book rule':>11s}")
    plus = fits(with_819=True)
    lo_off, hi_off = fits(peak_offset_nm=-PEAK_OFFSET_NM), fits(peak_offset_nm=+PEAK_OFFSET_NM)
    lps = fits(peak_offset_nm=-off)  # the model read with the offset ≡ the lamp's peak moved by −offset
    for lamp, c in base["central"].items():
        ts = [base[v][lamp].temperature_k for v in VARIANTS]
        ps = [base[v][lamp].sodium_pa / 1e3 for v in VARIANTS]
        edge = "*" if any(base[v][lamp].at_bound for v in VARIANTS) else " "
        p8 = plus["central"][lamp]
        out.append(f"{lamp:28s}{c.temperature_k:10.0f}{min(ts):7.0f}-{max(ts):<5.0f}{c.sodium_pa / 1e3:8.1f}"
                   f"{min(ps):7.1f}-{max(ps):<5.1f}{c.chi2_per_target:6.1f}{edge}{p8.temperature_k:8.0f}{p8.sodium_pa / 1e3:6.1f}"
                   f"{lo_off['central'][lamp].sodium_pa / 1e3:9.1f}-{hi_off['central'][lamp].sodium_pa / 1e3:<5.1f}"
                   f"{lps['central'][lamp].sodium_pa / 1e3:12.1f}{book_rule_pa(c.measured['separation']) / 1e3:11.1f}")
    worst = max(abs(base[v][lamp].model["blue peak"] - run[v][lamp]["blue peak"]) for v in VARIANTS for lamp in base[v])
    out += ["(* a fit in some table sits on the grid's edge: T 3400-4800 K, Na 5-45 kPa)",
            f"(LPS off: the instrument's offset near the D lines from the LPS lamp, {off:+.2f} ± {off_err:.2f} nm "
            "beyond the mercury response's)",
            "(book rule: de Groot & van Vliet 1986, eq. 3.27, 2.7±0.5 kPa nm-1 mm^1/2, from the lamp's own "
            "peak spacing and R = 3.8 mm; measured axis temperature 4100-4300 K in standard 400 W lamps)",
            f"(table against a direct run at every fitted point: blue peak within {worst:.3f} nm)", "",
            "Held out, from a direct run at each lamp's fitted point (central table; range over the tables), "
            "against the lamp:"]
    out.append(f"{'lamp':28s}" + "".join(f"{k:>27s}" for k in HELD_OUT))
    for lamp, c in base["central"].items():
        cells = []
        for k in HELD_OUT:
            vals = [_held(run[v][lamp], k) for v in VARIANTS]
            fmt = "{:.3f}" if k in ("red wing", "blue wing") else "{:.2f}"
            cells.append(f"{fmt.format(_held(run['central'][lamp], k))} [{fmt.format(min(vals))},{fmt.format(max(vals))}] "
                         f"{fmt.format(_held(c.measured, k))}")
        out.append(f"{lamp:28s}" + "".join(f"{s:>27s}" for s in cells))
    out.append("(each cell: model [range over tables] lamp)")
    from scipy.stats import spearmanr
    cct = [lamp_cct_k(k) for k in base["central"]]
    rho, pval = spearmanr(cct, [f.sodium_pa for f in base["central"].values()])
    out += ["", f"Rank correlation of the fitted sodium pressure with the lamp's CCT (central): {rho:+.2f} "
                f"(p = {pval:.2f}, nine lamps)"]
    return "\n".join(out)
