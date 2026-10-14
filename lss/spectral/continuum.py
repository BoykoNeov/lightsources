"""Continuum spectra: smooth or tabulated spectral densities (per nm).

A continuum supplies its density, its support, and the quadrature knots it
needs inside any interval, so that exact bin integrals can be formed by
segment quadrature (``lss.spectral.spectrum``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from lss.physics.thermal import planck
from lss.spectral.units import C2


class Continuum(ABC):
    @property
    @abstractmethod
    def support(self) -> tuple[float, float]:
        """(lo, hi) in nm; the density is zero outside (or negligible, see subclass)."""

    @abstractmethod
    def density(self, wavelength_nm): ...

    @abstractmethod
    def knots(self, lo: float, hi: float) -> np.ndarray:
        """Breakpoints within [lo, hi] such that the density is smooth between them."""

    def analytic_total(self) -> float | None:
        """∫ density over all wavelengths if known in closed form, else None."""
        return None


@dataclass(frozen=True)
class TabulatedContinuum(Continuum):
    """Piecewise-linear density between knots, zero outside the table."""

    wavelengths: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        lam = np.asarray(self.wavelengths, dtype=float)
        val = np.asarray(self.values, dtype=float)
        if lam.shape != val.shape or lam.ndim != 1 or lam.size < 2:
            raise ValueError("need 1-D wavelength and value arrays of equal length ≥ 2")
        if np.any(np.diff(lam) <= 0):
            raise ValueError("wavelengths must be strictly increasing")
        object.__setattr__(self, "wavelengths", lam)
        object.__setattr__(self, "values", val)

    @property
    def support(self):
        return float(self.wavelengths[0]), float(self.wavelengths[-1])

    def density(self, wavelength_nm):
        return np.interp(wavelength_nm, self.wavelengths, self.values, left=0.0, right=0.0)

    def knots(self, lo, hi):
        k = self.wavelengths
        return k[(k >= lo) & (k <= hi)]

    def analytic_total(self):
        return float(np.trapezoid(self.values, self.wavelengths))


@dataclass(frozen=True)
class PlanckContinuum(Continuum):
    """``scale`` × B_λ(T), in W·m⁻²·sr⁻¹·nm⁻¹ when scale is an emissivity.

    Support is taken as 1 nm – 10 mm, which holds all but ~1e-11 of the power
    for T ≥ 1000 K. Quadrature knots are spaced 1 % apart in wavelength.
    """

    temperature_k: float
    scale: float = 1.0
    c2: float = C2

    @property
    def support(self):
        return 1.0, 1e7

    def density(self, wavelength_nm):
        return self.scale * planck.spectral_radiance(wavelength_nm, self.temperature_k, self.c2)

    def knots(self, lo, hi):
        lo, hi = max(lo, 1.0), min(hi, 1e7)
        if hi <= lo:
            return np.array([])
        n = max(int(np.ceil(np.log(hi / lo) / np.log(1.01))), 1)
        return np.geomspace(lo, hi, n + 1)

    def analytic_total(self):
        if self.c2 != C2:
            return None
        return float(self.scale * planck.total_radiance(self.temperature_k))
