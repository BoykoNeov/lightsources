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
``delta_c6_factor``, 1 to 2); beyond 9.95 a₀ each curve joins its asymptote smoothly
(``tail``: fitted, or for I ½ Unsöld's ΔC₆ with the excess fading, a bracket). II ½ − X ½ has a minimum near 4.75 a₀; there the classical
sum piles up into the red satellite, which a quantum calculation would round off and
fringe (the measured S1–S4 undulations at 640–669 nm are not reproduced by this limit).

The transition dipoles are not given by the source beyond the A ← X oscillator strength
falling by 10× from 9.45 to 3.78 a₀: ``constant`` keeps d_z² at every R; ``falling``
lowers the two I bands' oscillator strength log-linearly by that factor and keeps II ½.

A second set of curves (``curves="czuchaj"``, ``lss/data/nahg_czuchaj.yaml``): Czuchaj,
Rebentrost, Stoll & Preuss's tables (Chem. Phys. Lett. 178, 246 (1991)), X, A ²Π and B ²Σ
with the σ→σ and σ→π transition dipoles, without spin–orbit coupling. Each Λ difference
potential is taken from its own computed asymptote and put on the 3p centroid, interpolated
out to 10 a₀, and only then coupled (Cohen–Schneider, the atomic fine structure at every R),
so the A–B crossing at 8–9 a₀ is resolved on the fine grid. The dipoles are mixed as
intensities by the coupled states' Σ and Π parts (``table``) or held at d_z² (``constant``).
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml
from scipy.interpolate import CubicSpline, PchipInterpolator

from lss.physics.hid import broadening, sodium
from lss.physics.hid.pairs import BOHR_M, ELEMENTARY_CHARGE_C, EPSILON_0, HARTREE_CM1
from lss.spectral.units import C, H, K_B

BANDS = ("II1/2", "I3/2", "I1/2")
#: The sets of upper curves: Windholz et al.'s digitised figure, or Czuchaj et al.'s tables.
CURVES = ("windholz", "czuchaj")
#: Dipole forms for Windholz's curves; Czuchaj's take :data:`CZUCHAJ_DIPOLE_FORMS`.
DIPOLE_FORMS = ("constant", "falling")
CZUCHAJ_DIPOLE_FORMS = ("table", "constant")
#: Czuchaj's curves are used out to here; beyond, the same join to the asymptote as Windholz's.
CZUCHAJ_R_MAX_A0 = 10.0
#: The splines run through the table's points out to here, so that their value and slope at
#: 10 a₀ (which set the join) are the table's, not a not-a-knot extrapolation from inside: cut
#: at 10 a₀, the Π curve's end slope came out 255 cm⁻¹/a₀, where the 12 a₀ point gives about 90
#: on average, and the join's A₆ went negative (I3/2 rose above its asymptote near 16.6 a₀).
CZUCHAJ_SPLINE_R_MAX_A0 = 12.0
#: Beyond the last tabulated dipole (15 a₀) the PCHIP reaches the atomic value here and stays.
CZUCHAJ_DIPOLE_ATOMIC_A0 = 20.0
#: The A ← X oscillator strength falls by 10× from 500 to 200 pm (Windholz et al., p. 375).
FALL_R_A0 = (200e-12 / BOHR_M, 500e-12 / BOHR_M)
DELTA_C6_FACTORS = (1.0, 2.0)
#: How each curve joins its asymptote beyond the digitised grid (:func:`difference_potential_cm1`).
TAILS = ("fitted", "dispersion")


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


@dataclass(frozen=True)
class CzuchajData:
    r_a0: np.ndarray  # the table's R up to CZUCHAJ_SPLINE_R_MAX_A0 (values used up to CZUCHAJ_R_MAX_A0)
    sigma_cm1: np.ndarray  # B − X, from its own computed asymptote (0 far out)
    pi_cm1: np.ndarray  # A − X, likewise
    dipole_r_a0: np.ndarray
    sigma_sigma_au: np.ndarray  # X → B
    sigma_pi_au: np.ndarray  # X → A, one Π component
    atomic_au: float
    source: str


@lru_cache(maxsize=1)
def czuchaj_data() -> CzuchajData:
    doc = yaml.safe_load(resources.files("lss.data").joinpath("nahg_czuchaj.yaml").read_text(encoding="utf-8"))
    pot, asym, dip = doc["potentials_hartree"], doc["asymptote_hartree"], doc["dipoles_au"]
    r = np.array(pot["r_a0"], dtype=float)
    keep = r <= CZUCHAJ_SPLINE_R_MAX_A0
    x = np.array(pot["X"], dtype=float) - asym["X"]

    def diff(name):
        return ((np.array(pot[name], dtype=float) - asym[name] - x) * HARTREE_CM1)[keep]

    return CzuchajData(r[keep], diff("B"), diff("A"), np.array(dip["r_a0"], dtype=float),
                       np.array(dip["sigma_sigma"], dtype=float), np.array(dip["sigma_pi"], dtype=float),
                       float(dip["atomic"]), str(doc["source"]))


@lru_cache(maxsize=1)
def _czuchaj_splines():
    cz = czuchaj_data()
    r_dip = np.append(cz.dipole_r_a0, CZUCHAJ_DIPOLE_ATOMIC_A0)
    return (CubicSpline(cz.r_a0, cz.sigma_cm1), CubicSpline(cz.r_a0, cz.pi_cm1),
            PchipInterpolator(r_dip, np.append(cz.sigma_sigma_au, cz.atomic_au)),
            PchipInterpolator(r_dip, np.append(cz.sigma_pi_au, cz.atomic_au)))


def czuchaj_coupled(r_a0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """band -> (transition energy, vacuum cm⁻¹; its Σ fraction c_Σ²) from Czuchaj's Λ curves,
    interpolated, put on the 3p centroid, then coupled with the atomic fine structure
    (Cohen–Schneider: a = ⅔ ΔE_fs; Ω = 3/2 is Π + a/2, the two Ω = ½ the eigenvalues of
    [[Π − a/2, a/√2], [a/√2, Σ]]). Beyond 10 a₀ the Λ curves are carried on as R⁻⁶ from their
    10 a₀ values: used for the Σ fractions only (the energies there are the join's)."""
    asym = data().asymptote_cm1
    r = np.asarray(r_a0, dtype=float)
    sig, pi, _, _ = _czuchaj_splines()
    r0 = CZUCHAJ_R_MAX_A0
    far = (r0 / np.maximum(r, r0)) ** 6
    vs = np.where(r <= r0, sig(np.minimum(r, r0)), float(sig(r0)) * far)
    vp = np.where(r <= r0, pi(np.minimum(r, r0)), float(pi(r0)) * far)
    p32, p12 = asym["I3/2"], asym["I1/2"]
    a = 2.0 * (p32 - p12) / 3.0
    centroid = (p12 + 2.0 * p32) / 3.0
    p = vp - a / 2.0
    q2 = a * a / 2.0
    mean, half = 0.5 * (p + vs), np.sqrt(0.25 * (p - vs) ** 2 + q2)
    upper = mean + half
    sigma_upper = (upper - p) ** 2 / (q2 + (upper - p) ** 2)  # eigenvector (q, λ − p)
    return {"II1/2": (centroid + upper, sigma_upper), "I1/2": (centroid + mean - half, 1.0 - sigma_upper),
            "I3/2": (centroid + vp + a / 2.0, np.zeros_like(r))}


def czuchaj_dipole_factor(band: str, r_a0) -> np.ndarray:
    """|D(R)|² / d_z² from Czuchaj's dipoles, mixed as intensities by the Σ and Π parts:
    c_Σ² D_σσ² + c_Π² D_σπ², over the table's atomic value squared (1 far out)."""
    _, _, dss, dsp = _czuchaj_splines()
    r = np.asarray(r_a0, dtype=float)
    rr = np.minimum(r, CZUCHAJ_DIPOLE_ATOMIC_A0)
    c_sigma = czuchaj_coupled(r)[band][1]
    return (c_sigma * dss(rr) ** 2 + (1.0 - c_sigma) * dsp(rr) ** 2) / czuchaj_data().atomic_au ** 2


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


def grid_start_a0(band: str, curves: str = "windholz") -> float:
    """The innermost R with data for the band."""
    return float(czuchaj_data().r_a0[0]) if curves == "czuchaj" else float(data().grids_a0[band][0])


def difference_potential_cm1(band: str, r_a0, delta_c6_factor: float = 1.0, tail: str = "fitted",
                             curves: str = "windholz") -> np.ndarray:
    """Transition energy at separation R, vacuum cm⁻¹: the shifted digitised curve on its grid
    (``curves="czuchaj"``: Czuchaj's coupled curves out to 10 a₀, unshifted);
    beyond it a join to the asymptote with the value and the slope continuous at the grid's
    end (a slope step would make a step in the classical spectrum); NaN inside the grid (no
    data there).

    ``fitted``: asymptote − A₆/R⁶ − A₁₂/R¹², A₆ and A₁₂ set by the value and slope (for I ½
    that makes A₆ 2.5–3 times Unsöld's ΔC₆). ``dispersion``, I ½ only: asymptote −
    ΔC₆/R⁶ − E₀ e^{−β(R−R₀)}, ΔC₆ Unsöld's times ``delta_c6_factor`` (the isotropic value
    3p₁/₂ has once the fine structure recouples, beyond about 16 a₀), the calculation's
    binding beyond it fading exponentially, E₀ and β set by the value and slope. The other
    two bands' fitted A₆ is 0.72–1.36 of Unsöld's already, and they keep it."""
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}")
    if curves not in CURVES:
        raise ValueError(f"curves must be one of {CURVES}")
    d = data()
    r = np.asarray(r_a0, dtype=float)
    if curves == "czuchaj":
        r_in, r0, step = grid_start_a0(band, curves), CZUCHAJ_R_MAX_A0, 1e-6
        end, before = czuchaj_coupled(np.array([r0, r0 - step]))[band][0]
        gap, slope = d.asymptote_cm1[band] - end, (end - before) / step
        inside = czuchaj_coupled(np.clip(r, r_in, r0))[band][0]
    else:
        grid, values = d.grids_a0[band], d.plotted_cm1[band] + shift_cm1(delta_c6_factor)
        r_in, r0, gap = grid[0], grid[-1], d.asymptote_cm1[band] - values[-1]
        slope = (values[-1] - values[-2]) / (grid[-1] - grid[-2])
        inside = np.interp(r, grid, values)
    if tail == "dispersion" and band == "I1/2":
        c6 = delta_c6_factor * delta_c6_unsold_au() * HARTREE_CM1
        excess = gap - c6 / r0**6
        beta = (slope - 6.0 * c6 / r0**7) / excess
        if excess <= 0 or beta <= 0:
            raise ValueError("the dispersion tail needs the curve bound beyond ΔC₆/R⁶ and falling faster")
        outer = d.asymptote_cm1[band] - c6 / r**6 - excess * np.exp(-beta * (r - r0))
    else:
        a6 = (12.0 * gap - slope * r0) * r0**6 / 6.0
        a12 = (slope * r0 - 6.0 * gap) * r0**12 / 6.0
        outer = d.asymptote_cm1[band] - a6 / r**6 - a12 / r**12
    return np.where(r < r_in, np.nan, np.where(r > r0, outer, inside))


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


def dipole_forms(curves: str = "windholz") -> tuple[str, ...]:
    return CZUCHAJ_DIPOLE_FORMS if curves == "czuchaj" else DIPOLE_FORMS


def dipole_factor(band: str, r_a0, form: str = "constant", delta_c6_factor: float = 1.0,
                  tail: str = "fitted", curves: str = "windholz") -> np.ndarray:
    """|D(R)|² / d_z². ``falling``: the I bands' oscillator strength (∝ ΔV·|D|²) falls 10×
    log-linearly from 9.45 to 3.78 a₀ and is held below; II ½ stays constant. ``table``
    (Czuchaj's curves): :func:`czuchaj_dipole_factor`."""
    if form not in dipole_forms(curves):
        raise ValueError(f"dipole form must be one of {dipole_forms(curves)} for {curves}'s curves")
    r = np.asarray(r_a0, dtype=float)
    if form == "table":
        return czuchaj_dipole_factor(band, r)
    if form == "constant" or band == "II1/2":
        return np.ones_like(r)
    lo, hi = FALL_R_A0
    strength = 10.0 ** ((np.clip(r, lo, hi) - hi) / (hi - lo))
    dv = difference_potential_cm1(band, r, delta_c6_factor, tail)
    dv_hi = difference_potential_cm1(band, hi, delta_c6_factor, tail)
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
                       points: int = 100_000, tail: str = "fitted",
                       curves: str = "windholz") -> dict[str, np.ndarray]:
    """k / (n_Na n_Hg) per band, m⁵, averaged over each wavenumber bin (vacuum cm⁻¹ edges)."""
    out = {}
    for band in BANDS:
        r = np.geomspace(grid_start_a0(band, curves), r_max_a0, points)
        out[band] = quasistatic(temperature_k, wavenumber_edges_cm1, r,
                                difference_potential_cm1(band, r, delta_c6_factor, tail, curves),
                                lower_potential_cm1(r),
                                dipole_squared_au() * dipole_factor(band, r, dipole, delta_c6_factor, tail, curves))
    return out


#: Wavenumber bins of the table the arc interpolates, vacuum cm⁻¹ (333–1250 nm).
TABLE_EDGES_CM1 = np.arange(8000.0, 30000.0 + 1.0, 2.0)


@lru_cache(maxsize=4096)
def _table(temperature_k: float, dipole: str, delta_c6_factor: float, tail: str,
           curves: str = "windholz") -> dict[str, np.ndarray]:
    return reduced_absorption(temperature_k, TABLE_EDGES_CM1, dipole, delta_c6_factor, tail=tail, curves=curves)


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
    twice that. ``tail``: how the curves join their asymptotes beyond the digitised grid
    (:func:`difference_potential_cm1`); it sets I ½ out to 15–20 nm red of D1. ``curves``:
    Windholz et al.'s digitised figure or Czuchaj et al.'s tables (whose dipole forms are
    ``table`` and ``constant``).
    """

    join_nm: float = 0.8
    dipole: str = "constant"
    delta_c6_factor: float = 1.0
    tail: str = "fitted"
    curves: str = "windholz"

    def __post_init__(self):
        if (self.join_nm <= 0 or self.curves not in CURVES or self.dipole not in dipole_forms(self.curves)
                or self.delta_c6_factor <= 0 or self.tail not in TAILS):
            raise ValueError("join and ΔC₆ factor must be positive and the curves, dipole form and tail known")

    def kappa(self, temperature_k: float, ground_m3: float, mercury_m3: float, frequency_hz: np.ndarray,
              line=None) -> np.ndarray:
        """Pair absorption coefficient, m⁻¹, before the stimulated-emission factor: the bands
        on ``line``'s upper level (a resonance :class:`~lss.physics.hid.sodium.Transition`),
        or all three (None)."""
        d = data()
        tables = _table(float(temperature_k), self.dipole, float(self.delta_c6_factor), self.tail, self.curves)
        bands = [b for b in BANDS if line is None
                 or abs(d.asymptote_cm1[b] - (line.upper_energy_cm1 - line.lower_energy_cm1)) < 0.01]
        if not bands:
            raise ValueError(f"no Na–Hg band ends on {line.name}'s upper level")
        centres = 0.5 * (TABLE_EDGES_CM1[1:] + TABLE_EDGES_CM1[:-1])
        per_pair = np.interp(np.asarray(frequency_hz) / (100.0 * C), centres, sum(tables[b] for b in bands),
                             left=0.0, right=0.0)
        return per_pair * ground_m3 * mercury_m3
