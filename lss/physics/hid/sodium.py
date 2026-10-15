"""Neutral sodium: levels, partition function and resonance lines (design §4.4 Stage A).

Data from the NIST Atomic Spectra Database (``lss/data/na_atomic.yaml``). Energies are
kept in cm⁻¹ as NIST gives them; ``hc/k`` turns them into temperatures.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from lss.spectral.units import C, C2, air_to_vacuum

#: hc/k in cm·K: an energy in cm⁻¹ times this, over T, is E/kT.
HC_OVER_K_CM = C2 * 100.0

#: Mass of the ²³Na atom (sodium is monoisotopic), kg. AME2020 atomic mass 22.98976928 u.
MASS_KG = 22.98976928 * 1.66053906660e-27

#: Rydberg constant for sodium's reduced mass, cm⁻¹ (R∞ = 109737.31568 cm⁻¹, CODATA 2018,
#: times 1/(1 + mₑ/M)); sets the effective quantum numbers of the levels.
RYDBERG_NA_CM1 = 109737.31568 / (1.0 + 5.48579909e-4 / 22.98976928)


@dataclass(frozen=True)
class Transition:
    """One line, upper level k → lower level i. ``wavelength_nm`` is in standard air."""

    name: str
    wavelength_nm: float
    a_ki: float  # s⁻¹
    lower_energy_cm1: float
    lower_g: int
    upper_energy_cm1: float
    upper_g: int
    self_broadening_m3_s: float  # Lorentzian FWHM in rad/s per ground-state atom per m³
    lower_l: int = 0  # orbital angular momentum of the valence electron, lower level
    upper_l: int = 1

    @property
    def vacuum_wavelength_nm(self) -> float:
        return float(air_to_vacuum(self.wavelength_nm))

    @property
    def frequency_hz(self) -> float:
        return C / (self.vacuum_wavelength_nm * 1e-9)


@dataclass(frozen=True)
class SodiumData:
    level_energy_cm1: np.ndarray
    level_g: np.ndarray
    level_l: np.ndarray  # orbital angular momentum of the valence electron
    ionisation_limit_cm1: float
    ion_ground_g: int
    lines: tuple[Transition, ...]
    source: str

    def line(self, name: str) -> Transition:
        for t in self.lines:
            if t.name == name:
                return t
        raise KeyError(name)


_L_OF = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


@lru_cache(maxsize=1)
def data() -> SodiumData:
    doc = yaml.safe_load(resources.files("lss.data").joinpath("na_atomic.yaml").read_text(encoding="utf-8"))
    levels = doc["levels"]
    energy = np.array([lv[4] for lv in levels], dtype=float)
    orbital = np.array([_L_OF[lv[0][-1]] for lv in levels])

    def l_at(e):
        return int(orbital[np.argmin(np.abs(energy - float(e)))])

    lines = tuple(
        Transition(
            name=d["name"],
            wavelength_nm=float(d["wavelength_nm"]),
            a_ki=float(d["a_ki"]),
            lower_energy_cm1=float(d["lower"]["energy_cm1"]),
            lower_g=int(d["lower"]["g"]),
            upper_energy_cm1=float(d["upper"]["energy_cm1"]),
            upper_g=int(d["upper"]["g"]),
            self_broadening_m3_s=float(d["self_broadening_cm3_s"]) * 1e-6,
            lower_l=l_at(d["lower"]["energy_cm1"]),
            upper_l=l_at(d["upper"]["energy_cm1"]),
        )
        for d in doc["lines"]
    )
    return SodiumData(
        level_energy_cm1=energy,
        level_g=np.array([lv[3] for lv in levels], dtype=float),
        level_l=orbital,
        ionisation_limit_cm1=float(doc["ionisation_limit_cm1"]),
        ion_ground_g=int(doc["ion_ground_g"]),
        lines=lines,
        source=str(doc["source"]),
    )


def partition_function(temperature_k, lowering_cm1: float = 0.0):
    """U(T) = Σ gᵢ exp(−Eᵢ/kT) over the bound levels below the lowered ionisation limit.

    NIST lists the levels up to n ≈ 20; the ones it omits lie within ~300 cm⁻¹ of the
    limit, inside the plasma's lowering at lamp densities, so they are not bound there.
    """
    d = data()
    keep = d.level_energy_cm1 < d.ionisation_limit_cm1 - lowering_cm1
    e, g = d.level_energy_cm1[keep], d.level_g[keep]
    t = np.asarray(temperature_k, dtype=float)
    return np.sum(g * np.exp(-np.multiply.outer(HC_OVER_K_CM / t, e)), axis=-1)


def level_population_fraction(energy_cm1: float, g: int, temperature_k, lowering_cm1: float = 0.0):
    """Fraction of neutral sodium atoms in one level (Boltzmann)."""
    t = np.asarray(temperature_k, dtype=float)
    return g * np.exp(-HC_OVER_K_CM * energy_cm1 / t) / partition_function(t, lowering_cm1)


def effective_quantum_number(energy_cm1: float) -> float:
    """n* = sqrt(R / (E_limit − E)) of a level of energy E."""
    return float(np.sqrt(RYDBERG_NA_CM1 / (data().ionisation_limit_cm1 - energy_cm1)))


def mean_square_radius_a0(energy_cm1: float, l: int) -> float:
    """⟨r²⟩ of the valence electron in units of a₀², hydrogenic with n* (Unsöld's approximation):
    ⟨r²⟩ = n*²/2 · (5n*² + 1 − 3l(l+1))."""
    n = effective_quantum_number(energy_cm1)
    return 0.5 * n * n * (5.0 * n * n + 1.0 - 3.0 * l * (l + 1))
