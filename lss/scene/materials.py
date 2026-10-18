"""Diffuse materials as spectral reflectance (design §8.2, §8.4).

A material is a piecewise-linear reflectance given at explicit knots. Mitsuba's
``irregular`` spectrum is 0 outside its knots (§8.4), so a table is passed with knots
over the whole render range: the analysis grid's values (end values held, as
``lss.studies.samples`` reads them) at every bin centre plus the two outer edges.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from lss.spectral.grids import analysis_grid


@dataclass(frozen=True)
class Material:
    id: str
    wavelengths: np.ndarray  # nm, increasing
    reflectance: np.ndarray
    source: str  # citation, or "constructed"

    def __post_init__(self):
        w = np.asarray(self.wavelengths, float)
        r = np.asarray(self.reflectance, float)
        if w.shape != r.shape or w.ndim != 1 or np.any(np.diff(w) <= 0):
            raise ValueError(f"{self.id}: knots must be one increasing row, as many as the values")
        if np.any(r < 0) or np.any(r > 1):
            raise ValueError(f"{self.id}: reflectance outside 0-1")
        object.__setattr__(self, "wavelengths", w)
        object.__setattr__(self, "reflectance", r)

    def at(self, wavelength_nm) -> np.ndarray:
        """Piecewise-linear reading, 0 outside the knots (as Mitsuba reads it)."""
        return np.interp(wavelength_nm, self.wavelengths, self.reflectance, left=0.0, right=0.0)

    def digest(self) -> str:
        h = hashlib.sha256(self.wavelengths.tobytes() + self.reflectance.tobytes())
        return h.hexdigest()[:16]


def from_analysis_grid(id: str, values: np.ndarray, source: str) -> Material:
    """Values at the analysis-grid centres, held to the grid's outer edges."""
    g = analysis_grid()
    v = np.asarray(values, float)
    w = np.concatenate([[g.lo], g.centres, [g.hi]])
    return Material(id, w, np.concatenate([[v[0]], v, [v[-1]]]), source)


def flat(id: str, value: float, source: str = "constructed") -> Material:
    return from_analysis_grid(id, np.full(len(analysis_grid()), float(value)), source)


def stepped(id: str, edges: np.ndarray, values: np.ndarray, source: str = "constructed",
            ramp_nm: float = 1e-3) -> Material:
    """Constant ``values[j]`` on each [edges[j], edges[j+1]], with ramps of ``ramp_nm`` at the
    inner edges and held across the outer ones (for checks that need a reflectance constant
    within each render bin)."""
    e = np.asarray(edges, float)
    v = np.asarray(values, float)
    w, r = [e[0] - ramp_nm], [v[0]]
    for j in range(1, e.size - 1):
        w += [e[j] - ramp_nm / 2, e[j] + ramp_nm / 2]
        r += [v[j - 1], v[j]]
    w.append(e[-1] + ramp_nm)
    r.append(v[-1])
    return Material(id, np.array(w), np.array(r), source)
