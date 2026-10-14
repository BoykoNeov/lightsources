"""Fixed-order Gauss–Legendre quadrature over many segments at once.

Exact integrals in this package are built from segment integrals between
breakpoints (line-local knots, filter knots, bin edges). Each segment is
smooth by construction, so an 8-point rule on it is accurate to near double
precision once segments are no wider than the local feature scale.
"""

import numpy as np

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)


def segment_integrals(func, a, b):
    """∫_a^b func for each pair (a[i], b[i]); ``func`` must accept arrays of any shape."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    x = mid[..., None] + half[..., None] * _NODES
    return half * (func(x) @ _WEIGHTS)
