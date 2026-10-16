"""Sodium–mercury pairs: the mercury share of the D lines' far wings (design §4.4, rung 2,
second part).

An HPS arc holds several times more mercury than sodium, so far from the D lines a sodium
atom absorbs mostly while a mercury atom is close. As for sodium pairs
(:mod:`lss.physics.hid.pairs`), the absorption is the classical (quasi-static) limit of the
Franck–Condon sum with LTE pair densities:

    k(ν̃) = n_Na n_Hg Σ_bands ∫ 4πR² e^{−V_X(R)/kT} S(R) δ(ν̃ − ΔV(R)/hc) dR,

    S(R) = (8π³ν / 3hc) |D(R)|² / 4πε₀,

with n_Na the ground-state sodium density and n_Hg the mercury density. The atoms differ,
so no ½; Na(3s) + Hg(¹S₀) meet only on X ²Σ⁺, so no spin weights. Three bands, the
Hund's case (c) curves from Na(3p) + Hg: I ½ and I 3/2 (from A ²Π) and II ½ (from B ²Σ⁺ at
short range). At long range they share the atomic 3s–3p strength equally, each with
|D|² = d_z², 3 d_z² fixed by the D lines' A values.

Data (``lss/data/nahg_pairs.yaml``): the X curve from Thiel, Hotop & Meyer's CCSD(T)
calculation (open in Thiel's thesis); the difference potentials digitised from Windholz,
Zerza, Pichler & Hess's Fig. 8 (Z. Phys. D 18, 373 (1991)), shifted by one constant so
that at 9.95 a₀ I 3/2 sits at the 3p₃/₂ asymptote minus ΔC₆/R⁶ (Unsöld's ΔC₆ times
``delta_c6_factor``, 1 to 2); beyond 9.95 a₀ each curve joins its asymptote smoothly. II ½ − X ½ has a minimum near 4.75 a₀; there the classical
sum piles up into the red satellite, which a quantum calculation would round off and
fringe (the measured S1–S4 undulations at 640–669 nm are not reproduced by this limit).

The transition dipoles are not given by the source beyond the A ← X oscillator strength
falling by 10× from 9.45 to 3.78 a₀: ``constant`` keeps d_z² at every R; ``falling``
lowers the two I bands' oscillator strength log-linearly by that factor and keeps II ½.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

from lss.physics.hid import broadening, sodium
from lss.physics.hid.pairs import BOHR_M, ELEMENTARY_CHARGE_C, EPSILON_0, HARTREE_CM1
from lss.spectral.units import C, H, K_B

BANDS = ("II1/2", "I3/2", "I1/2")
DIPOLE_FORMS = ("constant", "falling")
#: The A ← X oscillator strength falls by 10× from 500 to 200 pm (Windholz et al., p. 375).
FALL_R_A0 = (200e-12 / BOHR_M, 500e-12 / BOHR_M)
DELTA_C6_FACTORS = (1.0, 2.0)


@dataclass(frozen=True)
class MercuryPairData:
    asymptote_cm1: dict  # band -> its Na(3p_J) asymptote, vacuum cm⁻¹
    grids_a0: dict  # band -> R grid of the digitised curve
    plotted_cm1: dict  # band -> the curve as Windholz et al. plot it (not shifted)
    circles_cm1: dict  # band -> the plotted (computed) points, (R, E)
    reference_band: str
    reference_r_a0: float
    x_r_a0: np.ndarray
    x_hartree: np.ndarray
    x_c6_au: float
    source: str


def _grid(comment_free_rows, lo: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([np.array(row, dtype=float) for row in comment_free_rows])
    return np.round(lo + 0.05 * np.arange(values.size), 2), values


@lru_cache(maxsize=1)
def data() -> MercuryPairData:
    doc = yaml.safe_load(resources.files("lss.data").joinpath("nahg_pairs.yaml").read_text(encoding="utf-8"))
    lows = {"II1/2": 4.00, "I3/2": 4.00, "I1/2": 4.10}  # first grid point of each digitised curve
    grids, plotted = {}, {}
    for band in BANDS:
        grids[band], plotted[band] = _grid(doc["difference_potentials_cm1"][band], lows[band])
    asym = {band: float(doc["asymptote_cm1"][level]) for band, level in doc["upper_asymptote"].items()}
    lx = doc["lower_X"]
    return MercuryPairData(asym, grids, plotted, {k: np.array(v, dtype=float) for k, v in doc["circles_cm1"].items()},
                           doc["shift"]["reference_curve"], float(doc["shift"]["reference_r_a0"]),
                           np.array(lx["r_a0"], dtype=float), np.array(lx["hartree"], dtype=float),
                           float(lx["c6_au"]), str(doc["source"]))


@lru_cache(maxsize=1)
def _x_spline() -> CubicSpline:
    d = data()
    return CubicSpline(d.x_r_a0, d.x_hartree)


def lower_potential_cm1(r_a0) -> np.ndarray:
    """X ²Σ⁺, cm⁻¹ from the Na(3s) + Hg(¹S₀) asymptote (Thiel, Hotop & Meyer's CCSD(T) curve):
    the table inside 30 a₀, −C₆/R⁶ beyond, an exponential through the first two points inside."""
    d = data()
    r = np.asarray(r_a0, dtype=float)
    (r0, r1), (v0, v1) = d.x_r_a0[:2], d.x_hartree[:2]
    decay = np.log(v0 / v1) / (r1 - r0)
    inner = v0 * np.exp(-decay * (r - r0))
    outer = -d.x_c6_au / r**6
    middle = _x_spline()(np.clip(r, r0, 30.0))
    return np.where(r < r0, inner, np.where(r > 30.0, outer, middle)) * HARTREE_CM1


def delta_c6_unsold_au() -> float:
    """Unsöld's ΔC₆ between Na(3p)–Hg and Na(3s)–Hg (the estimate the impact widths use)."""
    return broadening.unsold_c6(sodium.data().line("D2"), "mercury") / broadening.C6_ATOMIC_UNIT


def shift_cm1(delta_c6_factor: float = 1.0) -> float:
    """The one constant added to all three digitised curves: it puts I 3/2 at 9.95 a₀ on the
    3p₃/₂ asymptote minus ΔC₆/R⁶, ΔC₆ being Unsöld's times ``delta_c6_factor``."""
    d = data()
    band, r = d.reference_band, d.reference_r_a0
    plotted = float(np.interp(r, d.grids_a0[band], d.plotted_cm1[band]))
    tail = delta_c6_factor * delta_c6_unsold_au() / r**6 * HARTREE_CM1
    return d.asymptote_cm1[band] - tail - plotted


def difference_potential_cm1(band: str, r_a0, delta_c6_factor: float = 1.0) -> np.ndarray:
    """Transition energy at separation R, vacuum cm⁻¹: the shifted digitised curve on its grid;
    beyond it asymptote − A₆/R⁶ − A₁₂/R¹², with A₆ and A₁₂ set so that the value and the slope
    are continuous at the grid's end (a slope step would make a step in the classical
    spectrum); NaN inside the grid (no data there)."""
    d = data()
    r = np.asarray(r_a0, dtype=float)
    grid, values = d.grids_a0[band], d.plotted_cm1[band] + shift_cm1(delta_c6_factor)
    r0, gap = grid[-1], d.asymptote_cm1[band] - values[-1]
    slope = (values[-1] - values[-2]) / (grid[-1] - grid[-2])
    a6 = (12.0 * gap - slope * r0) * r0**6 / 6.0
    a12 = (slope * r0 - 6.0 * gap) * r0**12 / 6.0
    tail = d.asymptote_cm1[band] - a6 / r**6 - a12 / r**12
    inside = np.interp(r, grid, values)
    return np.where(r < grid[0], np.nan, np.where(r > r0, tail, inside))


@lru_cache(maxsize=1)
def dipole_squared_au() -> float:
    """d_z²: each band's |D|² at long range, so that the three bands together carry the D
    lines' strength. From σ_line = (λ²/8π) A (g_k/g_i) = (8π³ν/3hc) · d²/4πε₀ summed."""
    total = 0.0
    for name in ("D1", "D2"):
        ln = sodium.data().line(name)
        lam = ln.vacuum_wavelength_nm * 1e-9
        sigma = lam**2 / (8.0 * np.pi) * ln.a_ki * ln.upper_g / ln.lower_g  # m²·Hz
        total += sigma * 4.0 * np.pi * EPSILON_0 * 3.0 * H * C / (8.0 * np.pi**3 * ln.frequency_hz)  # C²m²
    return total / 3.0 / (ELEMENTARY_CHARGE_C * BOHR_M) ** 2


def dipole_factor(band: str, r_a0, form: str = "constant", delta_c6_factor: float = 1.0) -> np.ndarray:
    """|D(R)|² / d_z². ``falling``: the I bands' oscillator strength (∝ ΔV·|D|²) falls 10×
    log-linearly from 9.45 to 3.78 a₀ and is held below; II ½ stays constant."""
    if form not in DIPOLE_FORMS:
        raise ValueError(f"dipole form must be one of {DIPOLE_FORMS}")
    r = np.asarray(r_a0, dtype=float)
    if form == "constant" or band == "II1/2":
        return np.ones_like(r)
    lo, hi = FALL_R_A0
    strength = 10.0 ** ((np.clip(r, lo, hi) - hi) / (hi - lo))
    dv = difference_potential_cm1(band, r, delta_c6_factor)
    dv_hi = difference_potential_cm1(band, hi, delta_c6_factor)
    return np.where(r >= hi, 1.0, strength * dv_hi / dv)


def quasistatic(temperature_k: float, wavenumber_edges_cm1, r_a0, transition_cm1, lower_cm1,
                dipole_squared_au) -> np.ndarray:
    """k / (n_a n_b) for one band of two unlike atoms, m⁵, averaged over each wavenumber bin:
    the R integral of 4πR² e^{−V/kT} S(R), binned at the transition energy (so an extremum
    of it gives a finite, integrable peak)."""
    r = np.asarray(r_a0, dtype=float)
    dv = np.asarray(transition_cm1, dtype=float)
    edges = np.asarray(wavenumber_edges_cm1, dtype=float)
    boltzmann = np.exp(-np.asarray(lower_cm1, dtype=float) * 100.0 * H * C / (K_B * temperature_k))
    dip2 = np.asarray(dipole_squared_au, dtype=float) * (ELEMENTARY_CHARGE_C * BOHR_M) ** 2
    strength = 8.0 * np.pi**3 * (dv * 100.0 * C) / (3.0 * H * C) * dip2 / (4.0 * np.pi * EPSILON_0)  # m²·Hz
    weight = 4.0 * np.pi * (r * BOHR_M) ** 2 * np.gradient(r) * BOHR_M * boltzmann * strength
    hist, _ = np.histogram(dv, bins=edges, weights=weight)
    return hist / (np.diff(edges) * 100.0 * C)


def reduced_absorption(temperature_k: float, wavenumber_edges_cm1, dipole: str = "constant",
                       delta_c6_factor: float = 1.0, r_max_a0: float = 400.0,
                       points: int = 100_000) -> dict[str, np.ndarray]:
    """k / (n_Na n_Hg) per band, m⁵, averaged over each wavenumber bin (vacuum cm⁻¹ edges)."""
    d = data()
    out = {}
    for band in BANDS:
        r = np.geomspace(d.grids_a0[band][0], r_max_a0, points)
        out[band] = quasistatic(temperature_k, wavenumber_edges_cm1, r,
                                difference_potential_cm1(band, r, delta_c6_factor), lower_potential_cm1(r),
                                dipole_squared_au() * dipole_factor(band, r, dipole, delta_c6_factor))
    return out


#: Wavenumber bins of the table the arc interpolates, vacuum cm⁻¹ (333–1250 nm).
TABLE_EDGES_CM1 = np.arange(8000.0, 30000.0 + 1.0, 2.0)


@lru_cache(maxsize=4096)
def _table(temperature_k: float, dipole: str, delta_c6_factor: float) -> dict[str, np.ndarray]:
    return reduced_absorption(temperature_k, TABLE_EDGES_CM1, dipole, delta_c6_factor)


@dataclass(frozen=True)
class MercuryPairWings:
    """Hand the mercury share of the D lines' wings over to the Na–Hg pair absorption.

    Each D line's mercury van der Waals share goes over to the bands on its own upper level
    (D1: I ½; D2: I 3/2 and II ½; ⅓ and ⅔ of the strength, as the lines) with distance from
    the line, by :func:`lss.physics.hid.arc.join_weight`: none inside ``join_nm``/2, all
    beyond 2·``join_nm`` (the resonance, xenon and Stark widths stay unless sodium pairs
    take the resonance share). The impact wing is the wrong form beyond about the
    Weisskopf detuning (0.36–0.47 nm at 2500–4000 K with Unsöld's C₆), outside which the
    quasi-static wing exceeds it as √(Δω/ω_W); the default hand-over is centred at about
    twice that.
    """

    join_nm: float = 0.8
    dipole: str = "constant"
    delta_c6_factor: float = 1.0

    def __post_init__(self):
        if self.join_nm <= 0 or self.dipole not in DIPOLE_FORMS or self.delta_c6_factor <= 0:
            raise ValueError("join and ΔC₆ factor must be positive and the dipole form known")

    def kappa(self, temperature_k: float, ground_m3: float, mercury_m3: float, frequency_hz: np.ndarray,
              line=None) -> np.ndarray:
        """Pair absorption coefficient, m⁻¹, before the stimulated-emission factor: the bands
        on ``line``'s upper level (a resonance :class:`~lss.physics.hid.sodium.Transition`),
        or all three (None)."""
        d = data()
        tables = _table(float(temperature_k), self.dipole, float(self.delta_c6_factor))
        bands = [b for b in BANDS if line is None
                 or abs(d.asymptote_cm1[b] - (line.upper_energy_cm1 - line.lower_energy_cm1)) < 0.01]
        if not bands:
            raise ValueError(f"no Na–Hg band ends on {line.name}'s upper level")
        centres = 0.5 * (TABLE_EDGES_CM1[1:] + TABLE_EDGES_CM1[:-1])
        per_pair = np.interp(np.asarray(frequency_hz) / (100.0 * C), centres, sum(tables[b] for b in bands),
                             left=0.0, right=0.0)
        return per_pair * ground_m3 * mercury_m3
