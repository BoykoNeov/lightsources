"""Exact spectra (design §3.1) and their bin integrals (§3.2, §3.3).

A :class:`Spectrum` is a sum of components, each a line or a continuum with a
scale factor and the transmissions applied to it. Nothing is sampled at a
point: :meth:`Spectrum.bin` returns exact power per bin, plus the power that
falls below and above the grid, so the total is always accounted for.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from lss.spectral.continuum import Continuum
from lss.spectral.filters import Transmission
from lss.spectral.grids import BinGrid
from lss.spectral.lines import Line
from lss.spectral.quadrature import segment_integrals


@dataclass(frozen=True)
class Component:
    source: Line | Continuum
    scale: float = 1.0
    transmissions: tuple[Transmission, ...] = ()


@dataclass(frozen=True)
class BinnedSpectrum:
    """Power per bin (spectrum units × nm), with the power outside the grid on each side."""

    grid: BinGrid
    values: np.ndarray
    below: float
    above: float
    quantity: str = ""

    @property
    def density(self) -> np.ndarray:
        """Mean spectral density in each bin (spectrum units)."""
        return self.values / self.grid.widths

    @property
    def in_grid(self) -> float:
        return float(self.values.sum())

    @property
    def total(self) -> float:
        return self.in_grid + self.below + self.above


@dataclass(frozen=True)
class Spectrum:
    components: tuple[Component, ...] = ()
    quantity: str = "relative"  # e.g. "W m-2 sr-1 nm-1", or "relative"
    name: str = field(default="", compare=False)

    @classmethod
    def of(cls, *sources: Line | Continuum, quantity: str = "relative", name: str = "") -> "Spectrum":
        return cls(tuple(Component(s) for s in sources), quantity, name)

    @property
    def lines(self) -> list[Line]:
        return [c.source for c in self.components if isinstance(c.source, Line)]

    def __add__(self, other: "Spectrum") -> "Spectrum":
        if self.quantity != other.quantity:
            raise ValueError(f"cannot add {self.quantity!r} to {other.quantity!r}")
        return Spectrum(self.components + other.components, self.quantity, self.name or other.name)

    def scaled(self, factor: float) -> "Spectrum":
        comps = tuple(replace(c, scale=c.scale * factor) for c in self.components)
        return replace(self, components=comps)

    def filtered(self, transmission: Transmission) -> "Spectrum":
        comps = tuple(replace(c, transmissions=c.transmissions + (transmission,)) for c in self.components)
        return replace(self, components=comps)

    def bin(self, grid: BinGrid) -> BinnedSpectrum:
        values = np.zeros(len(grid))
        below = above = 0.0
        for comp in self.components:
            v, b, a = _bin_component(comp, grid.edges)
            values += v
            below += b
            above += a
        return BinnedSpectrum(grid, values, below, above, self.quantity)

    def integral(self, lo: float, hi: float) -> float:
        """Exact ∫_lo^hi of the spectrum."""
        return float(self.bin(BinGrid(np.array([lo, hi]))).values[0])

    def total(self) -> float:
        """Integral over all wavelengths (analytic where the components allow it)."""
        total = 0.0
        for comp in self.components:
            src = comp.source
            if not comp.transmissions and isinstance(src, Line):
                total += comp.scale * src.power
            elif not comp.transmissions and src.analytic_total() is not None:
                total += comp.scale * src.analytic_total()
            else:
                lo, hi = _component_extent(comp)
                v, b, a = _bin_component(comp, np.array([lo, hi]))
                total += float(v[0] + b + a)
        return total


def _transmission_product(transmissions, x):
    out = np.ones_like(x)
    for t in transmissions:
        out = out * t(x)
    return out


def _component_extent(comp: Component) -> tuple[float, float]:
    src = comp.source
    if isinstance(src, Line):
        return src.air_centre - 1.0, src.air_centre + 1.0
    return src.support


def _cumulative(segment):
    return np.concatenate([[0.0], np.cumsum(segment)])


def _bin_component(comp: Component, edges: np.ndarray):
    """(bin powers, power below edges[0], power above edges[-1]) for one component."""
    src = comp.source
    if isinstance(src, Line):
        f_edges, f_total = _line_cumulative(src, comp.transmissions, edges)
    else:
        f_edges, f_total = _continuum_cumulative(src, comp.transmissions, edges)
    s = comp.scale
    return s * np.diff(f_edges), s * f_edges[0], s * (f_total - f_edges[-1])


def _line_cumulative(line: Line, transmissions, edges):
    """F(edge) = ∫_{-∞}^{edge} power·φ·ΠT, and the transmitted total."""
    c = line.air_centre
    prof = line.profile
    if not transmissions:
        return line.power * prof.cdf(edges - c), line.power

    t_knots = np.concatenate([t.knots for t in transmissions])
    span = np.max(np.abs(np.concatenate([edges, t_knots]) - c)) * 1.01 + prof.fwhm
    points = np.unique(np.concatenate([c + prof.knots(span), t_knots, edges]))

    def integrand(x):
        return prof.pdf(x - c) * _transmission_product(transmissions, x)

    seg = segment_integrals(integrand, points[:-1], points[1:])
    # Beyond the outermost points every transmission is at its end value.
    t_left = np.prod([t.left_value for t in transmissions])
    t_right = np.prod([t.right_value for t in transmissions])
    left_tail = prof.cdf(points[0] - c) * t_left
    right_tail = (1.0 - prof.cdf(points[-1] - c)) * t_right
    f_points = left_tail + _cumulative(seg)
    f_edges = f_points[np.searchsorted(points, edges)]
    return line.power * f_edges, line.power * (f_points[-1] + right_tail)


def _continuum_cumulative(cont: Continuum, transmissions, edges):
    """F(edge) = ∫_{lo}^{edge} density·ΠT over the continuum's support, and the total."""
    lo, hi = cont.support
    inside = edges[(edges > lo) & (edges < hi)]
    parts = [np.array([lo, hi]), inside, cont.knots(lo, hi)]
    for t in transmissions:
        k = t.knots
        parts.append(k[(k > lo) & (k < hi)])
    points = np.unique(np.concatenate(parts))

    def integrand(x):
        return cont.density(x) * _transmission_product(transmissions, x)

    seg = segment_integrals(integrand, points[:-1], points[1:])
    f_points = _cumulative(seg)
    clipped = np.clip(edges, lo, hi)
    f_edges = f_points[np.searchsorted(points, clipped)]
    return f_edges, f_points[-1]
