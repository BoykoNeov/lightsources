"""A spectrum as a scanning instrument records it: triangular bandpass sampling.

A spectroradiometer stepped every ``h`` nm with a triangular bandpass of FWHM
``h`` reports at each node λ_k the mean density under a triangle of base 2h
centred there. Tables published that way (the CIE 5 nm tables) spread a line
over its two neighbouring nodes in proportion to its distance from each, so a
line model can be compared with them only after the same sampling (design §11,
like with like).

With FWHM equal to the step the triangles sum to one everywhere between the
end nodes, so the samples keep every line's power: h · Σ s_k equals the
integral, wherever the line sits.
"""

import numpy as np

from lss.spectral.continuum import TabulatedContinuum
from lss.spectral.filters import Transmission
from lss.spectral.grids import BinGrid
from lss.spectral.spectrum import Spectrum


def triangle_samples(spectrum: Spectrum, nodes) -> np.ndarray:
    """Mean density under a triangle of FWHM equal to the node spacing, at each node.

    Exact for lines and continua alike: two binnings on the intervals between
    nodes, one plain and one weighted by a ramp that rises from 0 to 1 across
    each interval, split every interval's power between its two end nodes.
    """
    nodes = np.asarray(nodes, dtype=float)
    steps = np.diff(nodes)
    if nodes.ndim != 1 or nodes.size < 2 or not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
        raise ValueError("triangular bandpass sampling needs at least two equally spaced nodes")
    h = float(steps[0])
    if h <= 0:
        raise ValueError("nodes must be increasing")
    # The end triangles reach one step beyond the end nodes.
    edges = np.concatenate([[nodes[0] - h], nodes, [nodes[-1] + h]])
    plain = spectrum.bin(BinGrid(edges)).values
    # Ramp 0 → 1 across each interval, dropping back to 0 at every interior edge.
    inner = edges[1:-1]
    knots = np.concatenate([[edges[0]], np.repeat(inner, 2), [edges[-1]]])
    values = np.concatenate([[0.0], np.tile([1.0, 0.0], inner.size), [1.0]])
    rising = spectrum.filtered(Transmission(knots, values, "ramp")).bin(BinGrid(edges)).values
    # Node k takes the rising part of the interval ending at it and the falling part of the one starting there.
    return (rising[:-1] + (plain - rising)[1:]) / h


def as_recorded(spectrum: Spectrum, nodes) -> Spectrum:
    """The spectrum as a table of triangular-bandpass samples, linear between nodes."""
    nodes = np.asarray(nodes, dtype=float)
    cont = TabulatedContinuum(nodes, triangle_samples(spectrum, nodes))
    return Spectrum.of(cont, quantity=spectrum.quantity, name=f"{spectrum.name} (as recorded)".strip())
