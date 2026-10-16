"""Sodium atom pairs: the D lines' far wings from the Na₂ curves (design §4.4, rung 2, first part).

Far from the D lines, a sodium atom absorbs only while another ground-state sodium atom
is close: the pair is briefly a molecule, and it absorbs at the gap between its upper and
lower curve at that distance. With LTE pair densities, and the classical limit of the
Franck–Condon sum (the quasi-static wing, which also counts bound pairs), the absorption
coefficient per wavenumber is

    k(ν̃) = ½ n² Σ_bands w_s ∫ 4πR² e^{−V_low(R)/kT} S(R) δ(ν̃ − ΔV(R)/hc) dR,

    S(R) = (8π³ν / 3hc) g_Λ |D(R)|² / 4πε₀     (integrated cross-section of one transition),

with n the ground-state atom density, ½ for counting each pair of identical atoms once,
w_s the spin weight of the lower curve (X: ¼, a: ¾) and g_Λ = 2 for a Π upper state.
The R integral is binned in wavenumber rather than evaluated through 1/|dΔV/dR|, so the
satellites, where ΔV has an extremum, come out as finite, integrable peaks.

Four bands: X→A and a→b on the red side, X→B and a→c on the blue (data and sources in
``lss/data/na2_pairs.yaml``). At long range each upper curve is the resonance
interaction c₃/R³, so the bands together reproduce the resonance-broadening wing; nearer
in, the curves' shapes and the bound X-state pairs (a 6022 cm⁻¹ well) make the wings
asymmetric and give the satellites at 551.5 and 804 nm.

Classical limit: no rovibrational structure, which is finer than the 1 nm the measured
lamps resolve. Near the satellites, where ΔV has an extremum, the classical limit is least
accurate and its peaks come out sharper than a quantum calculation's.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from lss.physics.hid import sodium
from lss.spectral.units import C, H, K_B

BOHR_M = 5.29177210903e-11  # CODATA 2018
BOHR_ANGSTROM = BOHR_M * 1e10
HARTREE_CM1 = 219474.6313632  # CODATA 2018
ELEMENTARY_CHARGE_C = 1.602176634e-19  # exact
EPSILON_0 = 8.8541878128e-12  # CODATA 2018
#: Reduced mass of ²³Na₂ in electron masses (AME2020 22.98976928 u, mᵤ/mₑ 1822.888486).
REDUCED_MASS_ME = 0.5 * 22.98976928 * 1822.888486
HARTREE_K = 315775.02480407  # CODATA 2018

DIPOLE_FORMS = ("long-range", "constant")


@dataclass(frozen=True)
class Band:
    name: str
    lower: str
    g_lambda: int
    c3_au: float
    dipole_a: float
    dipole_b: float
    p6_au: float  # long-range terms fitted to the digitised curve (lss/data/na2_pairs.yaml)
    p8_au: float
    difference_cm1: np.ndarray  # on the shared R grid, a₀

    def long_range_cm1(self, asymptote_cm1: float, r_a0) -> np.ndarray:
        r = np.asarray(r_a0, dtype=float)
        return asymptote_cm1 + (self.c3_au / r**3 + self.p6_au / r**6 + self.p8_au / r**8) * HARTREE_CM1


@dataclass(frozen=True)
class PairData:
    asymptote_cm1: float
    lower: dict
    long_range: dict
    r_grid_a0: np.ndarray
    bands: tuple[Band, ...]
    a_inner_r_a0: np.ndarray
    a_inner_cm1: np.ndarray
    source: str


def _grid(spec) -> np.ndarray:
    return spec["start"] + spec["step"] * np.arange(spec["count"])


@lru_cache(maxsize=1)
def data() -> PairData:
    doc = yaml.safe_load(resources.files("lss.data").joinpath("na2_pairs.yaml").read_text(encoding="utf-8"))
    diff = doc["difference_potentials_cm1"]
    bands = tuple(
        Band(name, b["lower"], int(b["g_lambda"]), float(b["c3"]), float(b["dipole_a"]), float(b["dipole_b"]),
             float(b["p6"]), float(b["p8"]), np.array(diff[name], dtype=float))
        for name, b in doc["bands"].items()
    )
    inner = doc["a_state_inner_cm1"]

    def numbers(d):  # YAML 1.1 reads 0.3e5 (no exponent sign) as a string
        return {k: np.array(v, dtype=float) if isinstance(v, list) else float(v) for k, v in d.items()}

    lower = {k: numbers(v) for k, v in doc["lower"].items()}
    return PairData(float(doc["asymptote_cm1"]), lower, lower["long_range"], _grid(diff["r_a0"]),
                    bands, _grid(inner["r_a0"]), np.array(inner["values"], dtype=float), str(doc["source"]))


def lower_potential_cm1(state: str, r_a0) -> np.ndarray:
    """X or a curve, cm⁻¹ from the 3s + 3s asymptote (Knoop et al. 2011; the a state's
    inner part, inside its fitted range, from Chung et al.'s figure)."""
    d = data()
    s, lr = d.lower[state], d.long_range
    r = np.asarray(r_a0, dtype=float) * BOHR_ANGSTROM
    xi = (r - s["r_m"]) / (r + s["b"] * s["r_m"])
    middle = np.polynomial.polynomial.polyval(xi, s["a"])
    outer = (-lr["c6"] / r**6 - lr["c8"] / r**8 - lr["c10"] / r**10
             + s["exchange_sign"] * lr["a_ex"] * r ** lr["gamma"] * np.exp(-lr["beta"] * r))
    inner = s["u1"] + s["u2"] / r**6
    v = np.where(r < s["r_inn"], inner, np.where(r > s["r_out"], outer, middle))
    if state == "a":
        ra = np.asarray(r_a0, dtype=float)
        v = np.where(ra < s["r_inn"] / BOHR_ANGSTROM, np.interp(ra, d.a_inner_r_a0, d.a_inner_cm1), v)
    return v


def difference_potential_cm1(band: Band, r_a0) -> np.ndarray:
    """Transition energy at separation R: the smoothed digitised curve inside 24 a₀, its
    long-range form outside."""
    d = data()
    r = np.asarray(r_a0, dtype=float)
    return np.where(r > d.r_grid_a0[-1], band.long_range_cm1(d.asymptote_cm1, r),
                    np.interp(r, d.r_grid_a0, band.difference_cm1))


def dipole_au(band: Band, r_a0, form: str = "long-range") -> np.ndarray:
    if form not in DIPOLE_FORMS:
        raise ValueError(f"dipole form must be one of {DIPOLE_FORMS}")
    r = np.asarray(r_a0, dtype=float)
    return band.dipole_a + (band.dipole_b / r**3 if form == "long-range" else 0.0 * r)


def reduced_absorption(temperature_k: float, wavenumber_edges_cm1, dipole: str = "long-range",
                       jmax: float | None = None, r_min_a0: float = 4.2, r_max_a0: float = 400.0,
                       points: int = 100_000) -> dict[str, np.ndarray]:
    """k/n² per band, m⁵, averaged over each wavenumber bin (vacuum cm⁻¹ edges).

    ``jmax`` keeps only pairs with rotational quantum number up to it (classically, a
    share 1 − exp(−J²/2μkTR²) at R). It is for comparing with a quantum calculation
    that stopped there, not for the physics.
    """
    d = data()
    edges = np.asarray(wavenumber_edges_cm1, dtype=float)
    r = np.geomspace(r_min_a0, r_max_a0, points)
    dr = np.gradient(r)
    out = {}
    for band in d.bands:
        dv = difference_potential_cm1(band, r)
        boltzmann = np.exp(-lower_potential_cm1(band.lower, r) * 100.0 * H * C / (K_B * temperature_k))
        if jmax is not None:
            boltzmann = boltzmann * -np.expm1(-jmax**2 * HARTREE_K / (2.0 * REDUCED_MASS_ME * temperature_k * r**2))
        dip = dipole_au(band, r, dipole) * ELEMENTARY_CHARGE_C * BOHR_M
        strength = 8.0 * np.pi**3 * (dv * 100.0 * C) / (3.0 * H * C) * band.g_lambda * dip**2 / (4.0 * np.pi * EPSILON_0)
        weight = (0.5 * d.lower[band.lower]["spin_weight"] * 4.0 * np.pi * (r * BOHR_M) ** 2 * dr * BOHR_M
                  * boltzmann * strength)  # m⁵·Hz per R step
        hist, _ = np.histogram(dv, bins=edges, weights=weight)
        out[band.name] = hist / (np.diff(edges) * 100.0 * C)
    return out


#: Wavenumber bins of the table the arc interpolates, vacuum cm⁻¹ (333–1250 nm).
TABLE_EDGES_CM1 = np.arange(8000.0, 30000.0 + 1.0, 2.0)


@lru_cache(maxsize=4096)
def _table(temperature_k: float, dipole: str) -> np.ndarray:
    return sum(reduced_absorption(temperature_k, TABLE_EDGES_CM1, dipole).values())


@dataclass(frozen=True)
class PairWings:
    """Hand the resonance share of the D lines' wings over to the pair absorption.

    Each D line's resonance share goes over to its own part of the pair absorption with
    distance from the line, by :func:`lss.physics.hid.arc.join_weight`: none inside
    ``join_nm``/2, all beyond 2·``join_nm`` (the Hg, Xe and Stark widths stay).

    The curves have no fine structure (one 3p asymptote). Each line's part is its share of
    the D lines' strength (D1 ⅓, D2 ⅔) with the table moved onto its own upper level, so
    near the lines each wing sits on its own line, and far out the two parts together sit
    on the 3p centroid, 11.5 cm⁻¹ above 3p₁/₂.
    """

    join_nm: float = 0.8
    dipole: str = "long-range"

    def __post_init__(self):
        if self.join_nm <= 0 or self.dipole not in DIPOLE_FORMS:
            raise ValueError("join must be positive and the dipole form known")

    def kappa(self, temperature_k: float, ground_m3: float, frequency_hz: np.ndarray, line=None) -> np.ndarray:
        """Pair absorption coefficient, m⁻¹, before the stimulated-emission factor: ``line``'s
        part (a resonance :class:`~lss.physics.hid.sodium.Transition`), or both lines' (None)."""
        centres = 0.5 * (TABLE_EDGES_CM1[1:] + TABLE_EDGES_CM1[:-1])
        table = _table(float(temperature_k), self.dipole)
        nu = np.asarray(frequency_hz) / (100.0 * C)
        out = np.zeros(nu.shape)
        for ln in [line] if line is not None else sodium.resonance_lines():
            shift = ln.upper_energy_cm1 - ln.lower_energy_cm1 - data().asymptote_cm1
            out += sodium.resonance_shares()[ln.name] * np.interp(nu - shift, centres, table, left=0.0, right=0.0)
        return out * ground_m3**2
