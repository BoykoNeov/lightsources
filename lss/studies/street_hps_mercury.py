"""Each street lamp's mercury, read off its own peaks; the arc refitted with it (design §4.4).

The arc fit (:mod:`lss.studies.street_hps_fit`) held mercury at an assumed 70 kPa and left
seven lamps' red wings 1.2–5 times brighter than the model. Here each lamp's ratio of
mercury to sodium pressure is read off its own self-reversal peaks by de Groot & van
Vliet's practical formula (*The High-Pressure Sodium Lamp*, 1986, eq. 5.7, p. 159; from
350 W arc tubes of known amalgam, after de Groot & van Vliet 1979):

    p_Hg / p_Na = ((Δλ_R / Δλ_B)² − 1) / (0.057 · (Δλ_R^½ − 0.85)),

Δλ_B and Δλ_R the distances (nm) from the D lines' centre to the blue and red peaks. Mercury
widens the red side only, so it pushes the red peak out. The book gives the formula's
accuracy as 20–30 %, and calls it reliable only for p_Hg/p_Na > 10.

What is used, fitted and held out was set before any run with the lamps' own mercury:

- *Input:* the ratio, from the lamp's peaks. So the peak spacing is no longer held out.
- *Fitted:* axis temperature and sodium pressure, to the same targets as the arc fit (the
  blue peak, the 498, 515 and 568.8 nm line powers), with mercury tied to sodium by the ratio.
- *Held out:* the red wing's level and slope, the 668–678 nm rise and the blue wing.

The measured peaks are blurred and shifted by the instrument. That shift is estimated by
blurring the measured spectrum once more with the instrument's response, which moves the
peaks by about as much again; the model's own peaks cannot be used for it (their shape, on
the red side, is what is in question). The ratio's band covers: the D lines' centre as
their midpoint (589.29 nm) or weighted by strength (589.19 nm); the instrument's offset near
the D lines from the LPS lamp (±2 standard errors); half or all of the blur correction; and
the book's own 30 %, on top. The pre-set ±0.5 nm offset of the street-lamp study is shown
separately.

Only the central physics variant of the arc fit is tabulated here.

The lamps' order in red light is not a test (their wide spacing and bright red wing go
together, and eq. 5.7 reads the spacing); the red wing's level is. So the level (the
geometric mean of lamp/model over the nine) and the scatter about it are reported apart,
and a control fits every lamp at one shared ratio.

Found in the run, and checked after it (not set beforehand):

- At the lamps' mercury the model's red side has two maxima, either side of where the pair
  wings are spliced on. The outer one follows the splice (about 598 nm with 6 nm, 604 nm with
  12 nm; with 3 nm the two merge at 596 nm), so it is the join, which more mercury makes
  stronger. The lamps have one red peak. The model's peak spacing jumps between the two and
  is not reported.
- The extra light raises the D band's red half above the lamps' share (by up to 19 %), which
  lowers the line powers over the whole band and so raises the fitted temperature. As a check,
  the fit is also run with the line powers over the band's blue half (below 590 nm).
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from lss import download
from lss.accuracy import lica
from lss.physics.hid import arc
from lss.spectral.instrument import Kernel, observe
from lss.studies import street_hps, street_hps_fit as sf

#: p_Hg / p_Na, in steps of two, across the lamps' bands.
GRID_RATIO = np.geomspace(1.0, 128.0, 8)
#: The D lines' centre: their midpoint, and their centre weighted by strength (D2 twice D1).
D_CENTRES_NM = (589.29, 589.19)
#: The book's stated accuracy of eq. 5.7 (20–30 %), its upper end.
BOOK_ACCURACY = 0.3
#: Below this the book does not claim the formula.
BOOK_VALID_ABOVE = 10.0
#: The D band's blue half ends here (the check added after the run).
BLUE_HALF_NM = 590.0


def eq57(blue_nm: float, red_nm: float, centre_nm: float = D_CENTRES_NM[0]) -> float:
    """p_Hg/p_Na from the two self-reversal peaks (de Groot & van Vliet 1986, eq. 5.7)."""
    db, dr = centre_nm - blue_nm, red_nm - centre_nm
    return float(((dr / db) ** 2 - 1.0) / (0.057 * (np.sqrt(dr) - 0.85)))


def at(column: arc.ArcColumn, axis_temperature_k: float, sodium_pa: float, ratio: float) -> arc.ArcColumn:
    c = sf.at(column, axis_temperature_k, sodium_pa)
    return replace(c, fill=replace(c.fill, mercury_pa=float(ratio) * float(sodium_pa)))


def _observed(job):
    return sf.observed(*job)


def _key(column: arc.ArcColumn, kernel: Kernel) -> str:
    text = json.dumps([sf.TABLE_VERSION, "mercury ratio", repr(column), repr(kernel), sf.GRID_T_K.tolist(),
                       sf.GRID_NA_PA.tolist(), GRID_RATIO.tolist()])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def table_path(kernel: Kernel | None = None) -> Path:
    kernel = kernel or street_hps.response().kernel
    return download.cache_dir() / "hid_tables" / f"{_key(sf.CENTRAL, kernel)}.npy"


def table(kernel: Kernel | None = None, workers: int = 16) -> np.ndarray:
    """Recorded spectra, shape (temperature, sodium pressure, ratio, node), cached on disk."""
    kernel = kernel or street_hps.response().kernel
    path = table_path(kernel)
    if path.exists():
        return np.load(path)
    jobs = [(at(sf.CENTRAL, t, p, r), kernel) for t in sf.GRID_T_K for p in sf.GRID_NA_PA for r in GRID_RATIO]
    with Pool(workers) as pool:
        rows = pool.map(_observed, jobs)
    out = np.array(rows).reshape(sf.GRID_T_K.size, sf.GRID_NA_PA.size, GRID_RATIO.size, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.npy")
    np.save(tmp, out)
    tmp.replace(path)
    return out


def blur_shift_nm(x, y, kernel: Kernel | None = None) -> tuple[float, float]:
    """How far blurring the measured spectrum once more with the instrument's response moves
    its (blue, red) peaks: the estimate of how far the instrument moved them."""
    kernel = kernel or street_hps.response().kernel
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    f0 = street_hps.features(x, y)
    fine = np.arange(x[0], x[-1], 0.05)
    nodes = np.arange(np.ceil(x[0]) + 20.0, np.floor(x[-1]) - 20.0)
    f1 = street_hps.features(nodes, observe(fine, np.interp(fine, x, y), kernel, nodes))
    return f1.blue_peak_nm - f0.blue_peak_nm, f1.red_peak_nm - f0.red_peak_nm


@dataclass(frozen=True)
class Ratio:
    """A lamp's p_Hg/p_Na from eq. 5.7: central, its band, and the pre-set ±0.5 nm band."""

    central: float
    lo: float
    hi: float
    preset_lo: float
    preset_hi: float
    blur_blue_nm: float
    blur_red_nm: float

    @property
    def valid(self) -> bool:
        return self.central > BOOK_VALID_ABOVE


def ratio(lamp: str) -> Ratio:
    m = lica.load(lamp)
    f = street_hps.features(m.wavelengths_nm, m.values)
    sb, sr = blur_shift_nm(m.wavelengths_nm, m.values)
    off, err = sf.lps_offset_nm()

    def r(centre, offset, frac=1.0):
        # The instrument reads a line at true + offset (+ the blur): undo both.
        return eq57(f.blue_peak_nm - frac * sb - offset, f.red_peak_nm - frac * sr - offset, centre)

    central = r(D_CENTRES_NM[0], off)
    band = [r(c, o, frac) for c, o, frac in itertools.product(D_CENTRES_NM, (off - 2 * err, off, off + 2 * err), (0.5, 1.0))]
    preset = [r(c, o) for c in D_CENTRES_NM for o in (-sf.PEAK_OFFSET_NM, sf.PEAK_OFFSET_NM)]
    return Ratio(central, min(band) * (1.0 - BOOK_ACCURACY), max(band) * (1.0 + BOOK_ACCURACY),
                 min(preset), max(preset), sb, sr)


@lru_cache(maxsize=None)
def ratios() -> dict[str, Ratio]:
    return {lamp: ratio(lamp) for lamp in lica.HPS}


def red_share(x, y) -> float:
    """The D band's power above the blue half, over the whole band's."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = (x >= street_hps.D_BAND_NM[0]) & (x <= street_hps.D_BAND_NM[1])
    return float(y[d & (x > BLUE_HALF_NM)].sum() / y[d].sum())


def features(x, y, blue_half: bool = False) -> street_hps.Features:
    """The study's features; with ``blue_half``, the line powers over the D band's blue half."""
    f = street_hps.features(x, y)
    if not blue_half:
        return f
    share = 1.0 - red_share(x, y)
    return replace(f, line_power={k: v / share for k, v in f.line_power.items()})


def red_maxima_nm(x, y) -> list[float]:
    """The local maxima on the D lines' red side (589.5 nm to the band's end), above 5 % of its top."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    idx = np.where((x > 589.5) & (x < street_hps.D_BAND_NM[1]))[0]
    top = y[(x >= street_hps.D_BAND_NM[0]) & (x <= street_hps.D_BAND_NM[1])].max()
    return [float(x[i]) for i in idx if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > 0.05 * top]


def _vectors(spectra: np.ndarray, blue_half: bool) -> dict[str, np.ndarray]:
    shape = spectra.shape[:-1]
    feats = [sf._vector(features(street_hps.NODES_NM, spectra[idx], blue_half)) for idx in np.ndindex(shape)]
    return {k: np.array([f[k] for f in feats]).reshape(shape) for k in feats[0]
            if np.all(np.isfinite([f[k] for f in feats]))}


@lru_cache(maxsize=None)
def _node_features(blue_half: bool = False) -> dict[str, np.ndarray]:
    """Every feature at every node of the table, shape (temperature, sodium, ratio)."""
    return _vectors(table(), blue_half)


def _spline(z: np.ndarray) -> RectBivariateSpline:
    return RectBivariateSpline(sf.GRID_T_K, np.log(sf.GRID_NA_PA), z)


@lru_cache(maxsize=None)
def surface(hg_ratio: float, blue_half: bool = False) -> sf.Surface:
    """The table at one p_Hg/p_Na, as a smooth function of (T, sodium pressure)."""
    if not GRID_RATIO[0] <= hg_ratio <= GRID_RATIO[-1]:
        raise ValueError(f"ratio {hg_ratio:.2f} outside the table ({GRID_RATIO[0]:g}-{GRID_RATIO[-1]:g})")
    lr = np.log(GRID_RATIO)
    splines = {k: _spline(CubicSpline(lr, z, axis=2)(np.log(hg_ratio))) for k, z in _node_features(blue_half).items()}
    return sf.Surface(f"central, Hg/Na {hg_ratio:.3g}" + (", blue half" if blue_half else ""), splines)


@lru_cache(maxsize=None)
def surface_70_blue_half() -> sf.Surface:
    """The arc fit's central table (70 kPa), its line powers over the D band's blue half."""
    return sf.Surface("central, 70 kPa, blue half", {k: _spline(z) for k, z in _vectors(sf.table("central"), True).items()})


#: Each lamp is fitted at the low end of its band, its central ratio and the high end; and, as
#: a control, at one ratio shared by all nine (the median of the central ratios). The lamps'
#: order in red light is not a test (their wide spacing and bright red wing go together, and
#: eq. 5.7 reads the spacing); the red wing's level is. The control shows what the per-lamp
#: readout adds over a typical mercury.
ENDS = ("lo", "central", "hi")
ALL_ENDS = ENDS + ("common",)


def _clip(r: float) -> float:
    return float(np.clip(r, GRID_RATIO[0], GRID_RATIO[-1]))


def common_ratio() -> float:
    return float(np.median([r.central for r in ratios().values()]))


def ratio_at(lamp: str, end: str) -> float:
    return _clip(common_ratio() if end == "common" else getattr(ratios()[lamp], end))


@lru_cache(maxsize=None)
def fits() -> dict[str, dict[str, sf.Fit]]:
    """lamp -> end of the ratio band (or "common") -> fit (temperature and sodium; mercury tied to sodium)."""
    meas = street_hps.measured_features()
    return {lamp: {e: sf.fit(surface(ratio_at(lamp, e)), f, lamp) for e in ALL_ENDS} for lamp, f in meas.items()}


#: The check with the line powers over the blue half: each lamp's own ratio and the shared one.
BLUE_HALF_ENDS = ("central", "common")


@lru_cache(maxsize=None)
def blue_half_fits() -> dict[str, dict[str, sf.Fit]]:
    """lamp -> "70 kPa", "central" or "common" -> fit with the line powers over the D band's blue half."""
    out = {}
    for lamp in lica.HPS:
        m = lica.load(lamp)
        f = features(m.wavelengths_nm, m.values, blue_half=True)
        out[lamp] = {"70 kPa": sf.fit(surface_70_blue_half(), f, lamp)}
        out[lamp].update({e: sf.fit(surface(ratio_at(lamp, e), True), f, lamp) for e in BLUE_HALF_ENDS})
    return out


def _direct(job):
    t, p, r = job
    y = sf.observed(at(sf.CENTRAL, t, p, r), street_hps.response().kernel)
    out = sf._vector(street_hps.features(street_hps.NODES_NM, y))
    out["red maxima"] = red_maxima_nm(street_hps.NODES_NM, y)
    out["red share"] = red_share(street_hps.NODES_NM, y)
    return out


def _run(points: dict[tuple, tuple], workers: int) -> dict[str, dict[str, dict]]:
    with Pool(workers) as pool:
        rows = pool.map(_direct, list(points.values()))
    out: dict[str, dict[str, dict]] = {}
    for (lamp, e), row in zip(points, rows):
        out.setdefault(lamp, {})[e] = row
    return out


@lru_cache(maxsize=None)
def direct(workers: int = 16) -> dict[str, dict[str, dict]]:
    """lamp -> end -> every feature from a direct run at the fitted point (no table)."""
    base = fits()
    return _run({(l, e): (base[l][e].temperature_k, base[l][e].sodium_pa, ratio_at(l, e))
                 for l in base for e in ALL_ENDS}, workers)


@lru_cache(maxsize=None)
def blue_half_direct(workers: int = 16) -> dict[str, dict[str, dict]]:
    """As :func:`direct`, at the blue-half fits' points."""
    base = blue_half_fits()
    return _run({(l, e): (base[l][e].temperature_k, base[l][e].sodium_pa, ratio_at(l, e))
                 for l in base for e in BLUE_HALF_ENDS}, workers)


def format_text() -> str:
    rs, base, run = ratios(), fits(), direct()
    out = ["Each lamp's p_Hg/p_Na from its own peaks (de Groot & van Vliet 1986, eq. 5.7; claimed for > 10),",
           "the peaks first corrected for the instrument (blur shift from blurring the lamp again; LPS offset).",
           "Then axis temperature and sodium refitted (central physics variant) with mercury = ratio x sodium.", "",
           f"{'lamp':28s}{'blur B/R nm':>13s}{'Hg/Na':>7s}{'band':>14s}{'+-0.5 nm':>14s}"
           f"{'T K':>7s}{'Na kPa':>8s}{'Hg kPa':>8s}{'Hg band':>12s}{'chi2':>6s}"]
    for lamp, r in rs.items():
        c = base[lamp]["central"]
        lo, hi = base[lamp]["lo"], base[lamp]["hi"]
        flag = " " if r.valid else "?"
        edge = "*" if any(f.at_bound for f in base[lamp].values()) else " "
        out.append(f"{lamp:28s}{r.blur_blue_nm:6.2f}/{r.blur_red_nm:<6.2f}{r.central:6.1f}{flag}{r.lo:6.1f}-{r.hi:<7.1f}"
                   f"{r.preset_lo:6.1f}-{r.preset_hi:<7.1f}{c.temperature_k:7.0f}{c.sodium_pa / 1e3:8.1f}"
                   f"{ratio_at(lamp, 'central') * c.sodium_pa / 1e3:8.0f}"
                   f"{ratio_at(lamp, 'lo') * lo.sodium_pa / 1e3:6.0f}-{ratio_at(lamp, 'hi') * hi.sodium_pa / 1e3:<5.0f}"
                   f"{c.chi2_per_target:6.1f}{edge}")
    out += ["(? central ratio below 10, where the book does not claim the formula; * a fit on the grid's edge)",
            f"(band: D centre {D_CENTRES_NM[0]} or {D_CENTRES_NM[1]} nm, LPS offset +-2 s.e., half or all of the blur "
            f"shift, then the book's {BOOK_ACCURACY:.0%}; ratios outside {GRID_RATIO[0]:g}-{GRID_RATIO[-1]:g} clipped)", "",
            "Held out, from a direct run at each fitted point: model at the central ratio [low end, high end] lamp"]
    keys = ("red wing", "red slope", "red bump", "blue wing")
    out.append(f"{'lamp':28s}" + "".join(f"{k:>27s}" for k in keys) + f"{'lamp/model red':>18s}{'common':>8s}")
    for lamp in rs:
        cells = []
        for k in keys:
            fmt = "{:.3f}" if k in ("red wing", "blue wing") else "{:.2f}"
            v = [sf._held(run[lamp][e], k) for e in ("central", "lo", "hi")]
            cells.append(f"{fmt.format(v[0])} [{fmt.format(v[1])},{fmt.format(v[2])}] "
                         f"{fmt.format(sf._held(base[lamp]['central'].measured, k))}")
        q = {e: red_ratio(lamp, e) for e in ALL_ENDS}
        out.append(f"{lamp:28s}" + "".join(f"{s:>27s}" for s in cells)
                   + f"{q['central']:7.2f} [{q['hi']:.2f},{q['lo']:.2f}]{q['common']:8.2f}")
    out += ["(lamp/model red: central [high-mercury end, low-mercury end]; common: every lamp at the median "
            f"ratio, {common_ratio():.1f})", "",
            "Lamp/model red wing over the nine: level (geometric mean) and scatter (sd of its logarithm).",
            "The level is the test; the scatter is not (eq. 5.7 reads mercury from the spacing, which goes with the red wing).",
            f"  arc fit at 70 kPa: {level_70():.2f} [{scatter_70():.2f}]   own mercury: {level('central'):.2f} "
            f"[{scatter('central'):.2f}]   shared ratio: {level('common'):.2f} [{scatter('common'):.2f}]", "",
            "The model's red side at the central ratio: its maxima (nm) and the D band's share above 590 nm, model / lamp."]
    for lamp in rs:
        m = lica.load(lamp)
        mx = ", ".join(f"{v:.0f}" for v in run[lamp]["central"]["red maxima"])
        lx = ", ".join(f"{v:.0f}" for v in red_maxima_nm(m.wavelengths_nm, m.values))
        out.append(f"{lamp:28s} model {mx:14s} lamp {lx:10s} share {run[lamp]['central']['red share']:.3f} / "
                   f"{red_share(m.wavelengths_nm, m.values):.3f}")
    bh, brun = blue_half_fits(), blue_half_direct()
    out += ["(the outer red maximum follows the 6 nm splice: it is the join; the model's spacing is therefore not shown)", "",
            "Check added after the run: line powers over the D band's blue half (below 590 nm), not the whole band.",
            f"{'lamp':28s}{'T: 70 kPa':>10s}{'own Hg':>8s}{'shared':>8s}{'(whole band)':>14s}{'Na kPa':>8s}"
            f"{'lamp/model red':>16s}{'shared':>8s}"]
    for lamp in rs:
        b = bh[lamp]
        out.append(f"{lamp:28s}{b['70 kPa'].temperature_k:10.0f}{b['central'].temperature_k:8.0f}"
                   f"{b['common'].temperature_k:8.0f}{base[lamp]['central'].temperature_k:14.0f}"
                   f"{b['central'].sodium_pa / 1e3:8.1f}{red_ratio(lamp, 'central', True):16.2f}"
                   f"{red_ratio(lamp, 'common', True):8.2f}" + ("*" if any(f.at_bound for f in b.values()) else ""))
    out.append(f"level [scatter]: own mercury {level('central', True):.2f} [{scatter('central', True):.2f}]   "
               f"shared ratio {level('common', True):.2f} [{scatter('common', True):.2f}]")
    return "\n".join(out)


def red_ratio(lamp: str, end: str, blue_half: bool = False) -> float:
    """The lamp's red wing over the model's at the fitted point (held out)."""
    base, run = (blue_half_fits(), blue_half_direct()) if blue_half else (fits(), direct())
    return float(np.exp(base[lamp][end].measured["red wing"] - run[lamp][end]["red wing"]))


def _logs(end: str, blue_half: bool) -> np.ndarray:
    return np.log([red_ratio(lamp, end, blue_half) for lamp in lica.HPS])


def level(end: str, blue_half: bool = False) -> float:
    """Geometric mean over the nine lamps of lamp/model red wing: 1 is the right level."""
    return float(np.exp(_logs(end, blue_half).mean()))


def scatter(end: str, blue_half: bool = False) -> float:
    """Standard deviation over the nine lamps of log(lamp/model red wing), about its mean."""
    return float(_logs(end, blue_half).std())


def _logs_70() -> np.ndarray:
    base, run = sf.fits()["central"], sf.direct()["central"]
    return np.array([base[l].measured["red wing"] - run[l]["red wing"] for l in base])


def level_70() -> float:
    return float(np.exp(_logs_70().mean()))


def scatter_70() -> float:
    return float(_logs_70().std())


if __name__ == "__main__":
    table()
