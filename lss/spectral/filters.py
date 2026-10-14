"""Spectral transmission curves (filters, envelopes on the quick path).

A transmission is piecewise linear between knots and constant beyond the end
knots. A repeated knot wavelength makes a sharp edge (a jump). Filters act on
line lists exactly (design §3.1): T is integrated over each line's profile
with the filter's knots as quadrature breakpoints, never read off a grid.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transmission:
    wavelengths: np.ndarray  # nm (air), non-decreasing; a repeat marks a jump
    values: np.ndarray  # dimensionless, usually in [0, 1]
    name: str = ""

    def __post_init__(self):
        lam = np.asarray(self.wavelengths, dtype=float)
        val = np.asarray(self.values, dtype=float)
        if lam.shape != val.shape or lam.ndim != 1 or lam.size < 1:
            raise ValueError("wavelengths and values must be 1-D arrays of equal length")
        if np.any(np.diff(lam) < 0):
            raise ValueError("wavelengths must be non-decreasing")
        object.__setattr__(self, "wavelengths", lam)
        object.__setattr__(self, "values", val)

    @property
    def knots(self) -> np.ndarray:
        return np.unique(self.wavelengths)

    def __call__(self, wavelength_nm):
        """T(λ). At a jump the value is undefined (measure zero); quadrature never samples it."""
        lam = np.asarray(wavelength_nm, dtype=float)
        k = self.wavelengths
        # Segment i spans [k[i], k[i+1]]; side="right" puts a point at a jump in the upper segment.
        i = np.clip(np.searchsorted(k, lam, side="right") - 1, 0, max(k.size - 2, 0))
        if k.size == 1:
            return np.full_like(lam, self.values[0])
        k0, k1 = k[i], k[i + 1]
        v0, v1 = self.values[i], self.values[i + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(k1 > k0, (lam - k0) / (k1 - k0), 0.0)
        out = v0 + np.clip(t, 0.0, 1.0) * (v1 - v0)
        out = np.where(lam < k[0], self.values[0], out)
        return np.where(lam >= k[-1], self.values[-1], out)

    @property
    def left_value(self) -> float:
        return float(self.values[0])

    @property
    def right_value(self) -> float:
        return float(self.values[-1])


def step_filter(edge_nm: float, below: float = 0.0, above: float = 1.0, name: str = "") -> Transmission:
    """Ideal sharp-edged filter: ``below`` for λ < edge, ``above`` for λ > edge."""
    return Transmission(np.array([edge_nm, edge_nm]), np.array([below, above]), name or f"step {edge_nm:g} nm")
