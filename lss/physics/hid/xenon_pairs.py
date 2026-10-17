"""Sodium–xenon pairs: the xenon share of the D lines' far wings (design §4.4, rung 2,
third part).

The same classical (quasi-static) limit as for sodium–mercury pairs
(:mod:`lss.physics.hid.mercury_pairs`), with LTE pair densities:

    k(ν̃) = n_Na n_Xe Σ_bands ∫ 4πR² e^{−V_X(R)/kT} S(R) δ(ν̃ − ΔV(R)/hc) dR,

n_Na the ground-state sodium density, n_Xe the xenon density; unlike atoms, so no ½;
Na(3s) + Xe meet only on X ²Σ⁺, so no spin weights. Two bands, X → A ²Π (red of the
lines; |D|² = 2 d_z²) and X → B ²Σ⁺ (blue, with a satellite where B − X has its maximum;
|D|² = d_z²), d_z² as for NaHg, held at every R (the source gives no dipoles).

Data (``lss/data/naxe_pairs.yaml``): Galbis et al.'s (2013) computed points, read from
their figures; X inside 8 a₀, which the figure does not show, extrapolated by a
Tang–Toennies form fitted to the points (``wall`` variants bracket it). Spin-free curves:
one asymptote at the 3p centroid. The detunings that matter are hundreds to thousands of
cm⁻¹, far beyond the 17 cm⁻¹ fine structure, so (as for Na–Na) each D line takes its share
of the table moved onto its own upper level.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.special import gammaincc

from lss.physics.hid import sodium
from lss.physics.hid.mercury_pairs import dipole_squared_au, quasistatic
from lss.physics.hid.pairs import BOHR_M, ELEMENTARY_CHARGE_C, EPSILON_0, HARTREE_CM1
from lss.spectral.units import C, H, K_B

BANDS = ("A-X", "B-X")
WALLS = ("central", "soft", "hard")
LAMBDA_FACTOR = {"A-X": 2.0, "B-X": 1.0}  # Π takes ⅔ of the 3s–3p strength, Σ ⅓
X_INNER_A0, X_BLEND_A0, OUTER_A0 = 8.0, 9.0, 16.0


@dataclass(frozen=True)
class XenonPairData:
    points: dict  # curve -> (R a₀, E cm⁻¹) array, E from its own asymptote
    c6_au: float
    wall_c8_au: dict
    source: str


@lru_cache(maxsize=1)
def data() -> XenonPairData:
    doc = yaml.safe_load(resources.files("lss.data").joinpath("naxe_pairs.yaml").read_text(encoding="utf-8"))
    return XenonPairData({k: np.array(v, dtype=float) for k, v in doc["points_cm1"].items()},
                         float(doc["dispersion_X_au"]["c6"]), {k: float(v) for k, v in doc["wall_c8_au"].items()},
                         str(doc["source"]))


def asymptote_cm1() -> float:
    """The 3p centroid (vacuum cm⁻¹), the spin-free curves' upper asymptote."""
    lines = sodium.resonance_lines()
    return sum(ln.upper_g * (ln.upper_energy_cm1 - ln.lower_energy_cm1) for ln in lines) / sum(ln.upper_g for ln in lines)


def _damped(n: int, x):
    """Tang–Toennies damping f_n(x) = 1 − e^{−x} Σ_{k≤n} x^k/k!."""
    return 1.0 - gammaincc(n + 1, x)


def _tang_toennies_cm1(params, r, c6: float, c8: float) -> np.ndarray:
    a, b = params
    c10 = 49.0 / 40.0 * c8 * c8 / c6
    return (a * np.exp(-b * r) - _damped(6, b * r) * c6 / r**6 - _damped(8, b * r) * c8 / r**8
            - _damped(10, b * r) * c10 / r**10) * HARTREE_CM1


@lru_cache(maxsize=8)
def wall_fit(wall: str = "central") -> tuple[float, float, float]:
    """(A, b, rms cm⁻¹): the Tang–Toennies form fitted to all the X points, C8 per ``wall``."""
    if wall not in WALLS:
        raise ValueError(f"wall must be one of {WALLS}")
    d = data()
    r, e = d.points["X"].T
    c8 = d.wall_c8_au[wall]
    fit = least_squares(lambda p: _tang_toennies_cm1(p, r, d.c6_au, c8) - e, [10.0, 1.0])
    return float(fit.x[0]), float(fit.x[1]), float(np.sqrt(np.mean(fit.fun**2)))


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lower_potential_cm1(r_a0, wall: str = "central") -> np.ndarray:
    """X ²Σ⁺, cm⁻¹ from the Na(3s) + Xe asymptote: the Tang–Toennies wall inside 8 a₀, the
    points (PCHIP) beyond 9 a₀, a smooth hand-over between; the form's tail beyond 16 a₀."""
    d = data()
    r = np.asarray(r_a0, dtype=float)
    a, b, _ = wall_fit(wall)
    c8 = d.wall_c8_au[wall]
    tt = _tang_toennies_cm1((a, b), r, d.c6_au, c8)
    pts = d.points["X"]
    inside = PchipInterpolator(*pts.T)(np.clip(r, pts[0, 0], pts[-1, 0]))
    c10 = 49.0 / 40.0 * c8 * c8 / d.c6_au
    tail = -(d.c6_au / r**6 + c8 / r**8 + c10 / r**10) * HARTREE_CM1
    tail = tail + (pts[-1, 1] - (-(d.c6_au / OUTER_A0**6 + c8 / OUTER_A0**8 + c10 / OUTER_A0**10) * HARTREE_CM1))
    w = _smoothstep((r - X_INNER_A0) / (X_BLEND_A0 - X_INNER_A0))  # 0: wall, 1: points
    v = (1.0 - w) * tt + w * inside
    return np.where(r > OUTER_A0, tail, v)


def upper_potential_cm1(curve: str, r_a0) -> np.ndarray:
    """A ²Π or B ²Σ⁺, cm⁻¹ from the Na(3p) + Xe asymptote: PCHIP through the points; inside
    the first point an exponential continuing its value and slope; beyond 16 a₀
    −A6/R⁶ − A8/R⁸ joined in value and slope."""
    pts = data().points[curve]
    r = np.asarray(r_a0, dtype=float)
    spline = PchipInterpolator(*pts.T)
    (r0, v0), r1 = pts[0], pts[-1, 0]
    s0 = float(spline.derivative()(r0))
    inner = v0 * np.exp(s0 / v0 * (r - r0)) if v0 > 0 else v0 + s0 * (r - r0)
    v1, s1 = float(spline(r1)), float(spline.derivative()(r1))
    # V = −a6/R⁶ − a8/R⁸ with V(r1) = v1, V'(r1) = s1
    a8 = (s1 * r1 + 6.0 * v1) * r1**8 / 2.0
    a6 = -v1 * r1**6 - a8 / r1**2
    outer = -a6 / r**6 - a8 / r**8
    return np.where(r < r0, inner, np.where(r > r1, outer, spline(np.clip(r, r0, r1))))


def difference_potential_cm1(band: str, r_a0, wall: str = "central") -> np.ndarray:
    """Transition energy at separation R, vacuum cm⁻¹."""
    upper = {"A-X": "A", "B-X": "B"}[band]
    return asymptote_cm1() + upper_potential_cm1(upper, r_a0) - lower_potential_cm1(r_a0, wall)


def reduced_absorption(temperature_k: float, wavenumber_edges_cm1, wall: str = "central",
                       r_min_a0: float = 4.0, r_max_a0: float = 400.0, points: int = 1_000_000) -> dict[str, np.ndarray]:
    """k / (n_Na n_Xe) per band, m⁵, averaged over each wavenumber bin (vacuum cm⁻¹ edges)."""
    r = np.geomspace(r_min_a0, r_max_a0, points)
    lower = lower_potential_cm1(r, wall)
    return {band: quasistatic(temperature_k, wavenumber_edges_cm1, r, difference_potential_cm1(band, r, wall), lower,
                              LAMBDA_FACTOR[band] * dipole_squared_au() * np.ones_like(r))
            for band in BANDS}


#: Wavenumber bins of the table the arc interpolates, vacuum cm⁻¹ (333–1250 nm).
TABLE_EDGES_CM1 = np.arange(8000.0, 30000.0 + 1.0, 2.0)


@lru_cache(maxsize=8)
def _table_geometry(wall: str, r_min_a0: float = 4.0, r_max_a0: float = 400.0, points: int = 1_000_000):
    """What :func:`reduced_absorption` bins, without the Boltzmann factor (the only part that
    depends on temperature): each R step's table bin, its weight at T → ∞, and V_X."""
    r = np.geomspace(r_min_a0, r_max_a0, points)
    lower = lower_potential_cm1(r, wall)
    bins, weights = [], []
    for band in BANDS:
        dv = difference_potential_cm1(band, r, wall)
        weights.append(quasistatic_weight(r, dv, LAMBDA_FACTOR[band] * dipole_squared_au()))
        bins.append(np.searchsorted(TABLE_EDGES_CM1, dv, side="right") - 1)
    idx, w = np.concatenate(bins), np.concatenate(weights)
    keep = (idx >= 0) & (idx < TABLE_EDGES_CM1.size - 1)
    return idx[keep], w[keep], np.concatenate([lower, lower])[keep]


def quasistatic_weight(r_a0, transition_cm1, dipole_squared_au: float) -> np.ndarray:
    """Each R step's 4πR² dR S(R), m⁵·Hz: the weight :func:`quasistatic` bins, before the
    Boltzmann factor."""
    dip2 = dipole_squared_au * (ELEMENTARY_CHARGE_C * BOHR_M) ** 2
    strength = 8.0 * np.pi**3 * (transition_cm1 * 100.0 * C) / (3.0 * H * C) * dip2 / (4.0 * np.pi * EPSILON_0)
    return 4.0 * np.pi * (r_a0 * BOHR_M) ** 2 * np.gradient(r_a0) * BOHR_M * strength


@lru_cache(maxsize=4096)
def _table(temperature_k: float, wall: str) -> np.ndarray:
    """k / (n_Na n_Xe) summed over the bands on :data:`TABLE_EDGES_CM1`, m⁵: the same sum as
    :func:`reduced_absorption`, with the temperature-free part computed once."""
    idx, w, lower = _table_geometry(wall)
    boltzmann = np.exp(-lower * 100.0 * H * C / (K_B * temperature_k))
    hist = np.bincount(idx, weights=w * boltzmann, minlength=TABLE_EDGES_CM1.size - 1)
    return hist / (np.diff(TABLE_EDGES_CM1) * 100.0 * C)


@dataclass(frozen=True)
class XenonPairWings:
    """Hand the xenon share of the D lines' wings over to the Na–Xe pair absorption.

    Each D line's xenon van der Waals share goes over to its own part of the pair absorption
    (its share of the D lines' strength, D1 ⅓, D2 ⅔, with the table moved onto its own upper
    level) with distance from the line, by :func:`lss.physics.hid.arc.join_weight`.
    """

    join_nm: float = 0.8
    wall: str = "central"

    def __post_init__(self):
        if self.join_nm <= 0 or self.wall not in WALLS:
            raise ValueError("join must be positive and the wall variant known")

    def kappa(self, temperature_k: float, ground_m3: float, xenon_m3: float, frequency_hz: np.ndarray,
              line=None) -> np.ndarray:
        """Pair absorption coefficient, m⁻¹, before the stimulated-emission factor: ``line``'s
        part (a resonance :class:`~lss.physics.hid.sodium.Transition`), or both lines' (None)."""
        centres = 0.5 * (TABLE_EDGES_CM1[1:] + TABLE_EDGES_CM1[:-1])
        table = _table(float(temperature_k), self.wall)
        nu = np.asarray(frequency_hz) / (100.0 * C)
        out = np.zeros(nu.shape)
        for ln in [line] if line is not None else sodium.resonance_lines():
            shift = ln.upper_energy_cm1 - ln.lower_energy_cm1 - asymptote_cm1()
            out += sodium.resonance_shares()[ln.name] * np.interp(nu - shift, centres, table, left=0.0, right=0.0)
        return out * ground_m3 * xenon_m3
