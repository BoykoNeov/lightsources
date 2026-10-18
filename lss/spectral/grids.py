"""Wavelength bin grids (design §3.2, §3.3).

A grid is a set of contiguous box bins given by their edges. Spectra are put on
a grid by *exact bin integrals* (power per bin), never by point sampling, so a
narrow line contributes its full power to whichever bin holds it.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinGrid:
    """Contiguous box bins in wavelength (nm, air)."""

    edges: np.ndarray
    name: str = ""

    def __post_init__(self):
        edges = np.asarray(self.edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("a grid needs at least two edges")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("grid edges must be strictly increasing")
        object.__setattr__(self, "edges", edges)

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def lo(self) -> float:
        return float(self.edges[0])

    @property
    def hi(self) -> float:
        return float(self.edges[-1])

    def __len__(self) -> int:
        return self.edges.size - 1


def uniform_grid(first_centre: float, last_centre: float, width: float, name: str = "") -> BinGrid:
    """Bins of equal ``width`` whose centres run from ``first_centre`` to ``last_centre``."""
    n = int(round((last_centre - first_centre) / width)) + 1
    centres = first_centre + width * np.arange(n)
    if not np.isclose(centres[-1], last_centre):
        raise ValueError("last_centre is not reachable in whole steps of width")
    edges = np.concatenate([centres - 0.5 * width, [centres[-1] + 0.5 * width]])
    return BinGrid(edges, name)


def analysis_grid() -> BinGrid:
    """360–830 nm in 1 nm bins centred on whole nanometres, matching the CIE 1 nm tables."""
    return uniform_grid(360.0, 830.0, 1.0, "analysis")


def extended_grid() -> BinGrid:
    """200–3000 nm for UV and IR quantities: 1 nm up to 830.5 nm, then 10 nm bins.

    Used for UV-A/UV-B dose and IR load. Total power is never taken from this
    grid (design §3.2); generators supply it.
    """
    fine = np.arange(199.5, 830.5 + 1e-9, 1.0)
    coarse = np.arange(840.5, 3000.5 + 1e-9, 10.0)
    return BinGrid(np.concatenate([fine, coarse]), "extended")


def render_grid(width: float = 10.0, lo: float = 359.5, hi: float = 830.5) -> BinGrid:
    """Coarse render bins of ``width`` nm spanning [lo, hi] (design §3.3, §8.4).

    The defaults put every edge on an analysis-grid edge, so each render bin is a whole
    number of 1 nm analysis bins and re-colouring uses the analysis grid's own sums. A
    last bin narrower than half ``width`` is merged into the one before it.
    """
    edges = np.arange(lo, hi, width)
    if hi - edges[-1] < 0.5 * width and edges.size > 1:
        edges = edges[:-1]
    edges = np.append(edges, hi)
    return BinGrid(edges, f"render {width:g} nm")
