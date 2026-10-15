"""Composition of a sodium–mercury–xenon arc under LTE (design §4.4 Stage A).

The total pressure is the same everywhere in the arc, and each element keeps the same
share of the heavy particles at every radius (no demixing). Only sodium ionises at
lamp temperatures (5.14 eV; mercury 10.4 eV, xenon 12.1 eV), so the electrons come from
sodium, and Saha's equation for Na ⇌ Na⁺ + e closes the composition at each temperature.

The ionisation energy and the partition function's last bound level are lowered by the
Debye–Hückel estimate e²/(4πε₀λ_D), with λ_D from the electrons and ions together
(H. R. Griem, *Plasma Spectroscopy*, McGraw-Hill 1964, §6.3). At an HPS axis (≈ 4000 K,
nₑ ≈ 10²² m⁻³) this is ≈ 0.03 eV and raises nₑ by about 5 %.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from lss.physics.hid import sodium
from lss.spectral.units import H, K_B

ELECTRON_MASS_KG = 9.1093837015e-31  # CODATA 2018
ELEMENTARY_CHARGE_C = 1.602176634e-19  # exact
EPSILON_0 = 8.8541878128e-12  # CODATA 2018, F/m
#: One cm⁻¹ in joules.
CM1_J = H * 299792458.0 * 100.0


@dataclass(frozen=True)
class Fill:
    """The arc's gas: partial pressure of each element in the hot arc, Pa.

    These are the pressures the elements would exert un-ionised; their sum is the
    arc's total pressure (ionisation adds electrons, a few per cent of sodium's share).
    """

    sodium_pa: float
    mercury_pa: float = 0.0
    xenon_pa: float = 0.0

    def __post_init__(self):
        if self.sodium_pa <= 0 or self.mercury_pa < 0 or self.xenon_pa < 0:
            raise ValueError("pressures must be non-negative, and sodium's positive")

    @property
    def total_pa(self) -> float:
        return self.sodium_pa + self.mercury_pa + self.xenon_pa


@dataclass(frozen=True)
class Composition:
    """Number densities in m⁻³ at one temperature."""

    temperature_k: float
    sodium: float  # neutral sodium, all levels
    sodium_ion: float
    electrons: float
    mercury: float
    xenon: float
    lowering_cm1: float  # Debye–Hückel lowering of the ionisation energy

    @property
    def ionised_fraction(self) -> float:
        return self.sodium_ion / (self.sodium + self.sodium_ion)


def debye_lowering_cm1(electrons_m3: float, temperature_k: float) -> float:
    """e²/(4πε₀λ_D) in cm⁻¹, λ_D from electrons and singly charged ions at density nₑ each."""
    if electrons_m3 <= 0:
        return 0.0
    lam_d = np.sqrt(EPSILON_0 * K_B * temperature_k / (2.0 * electrons_m3 * ELEMENTARY_CHARGE_C**2))
    return ELEMENTARY_CHARGE_C**2 / (4.0 * np.pi * EPSILON_0 * lam_d) / CM1_J


def saha_factor(temperature_k: float, lowering_cm1: float) -> float:
    """nₑ·n(Na⁺)/n(Na) in m⁻³: 2 U⁺/U · (2π mₑ kT/h²)^{3/2} · exp(−(χ − Δχ)/kT)."""
    d = sodium.data()
    thermal = (2.0 * np.pi * ELECTRON_MASS_KG * K_B * temperature_k / H**2) ** 1.5
    chi = (d.ionisation_limit_cm1 - lowering_cm1) * sodium.HC_OVER_K_CM / temperature_k
    u_ratio = d.ion_ground_g / float(sodium.partition_function(temperature_k, lowering_cm1))
    return 2.0 * u_ratio * thermal * np.exp(-chi)


def composition(temperature_k: float, fill: Fill) -> Composition:
    """Densities at one temperature: ideal gas at the fill's total pressure, sodium in Saha balance."""
    t = float(temperature_k)
    x_na = fill.sodium_pa / fill.total_pa
    n_total = fill.total_pa / (K_B * t)  # all free particles, electrons included

    def heavy_and_electrons(alpha):
        heavy = n_total / (1.0 + alpha * x_na)  # nuclei: N(1 + α x_Na) = p/kT
        return heavy, alpha * x_na * heavy

    def mismatch(alpha):
        heavy, ne = heavy_and_electrons(alpha)
        n_na0 = (1.0 - alpha) * x_na * heavy
        return ne * ne - saha_factor(t, debye_lowering_cm1(ne, t)) * n_na0

    alpha = brentq(mismatch, 0.0, 1.0, xtol=1e-15, rtol=1e-12)  # mismatch(0) ≤ 0 < mismatch(1)
    heavy, ne = heavy_and_electrons(alpha)
    return Composition(
        temperature_k=t,
        sodium=(1.0 - alpha) * x_na * heavy,
        sodium_ion=ne,
        electrons=ne,
        mercury=fill.mercury_pa / fill.total_pa * heavy,
        xenon=fill.xenon_pa / fill.total_pa * heavy,
        lowering_cm1=debye_lowering_cm1(ne, t),
    )
