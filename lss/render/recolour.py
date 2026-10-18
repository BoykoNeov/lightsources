"""Re-colouring lighting maps (design §3.3, §9.1, §8.4). No Mitsuba needed.

A lighting map holds, per pixel and render bin j, the bin's mean pixel radiance per
unit zone radiance (dimensionless). Under zone spectra S (radiance per 1 nm analysis
bin, ``Spectrum.bin(analysis_grid())``) a pixel's tristimulus values are

    XYZ = Σ_j map_j · W_j,   W_j = Σ_{i in bin j} S_i · cmf_i,

the sums ``colorimetry.tristimulus`` takes, so a map constant at ρ re-colours to the
tristimulus values of ρ·S exactly. That needs every render edge on an analysis edge.
"""

import numpy as np

from lss.spectral import colorimetry
from lss.spectral.grids import BinGrid, analysis_grid
from lss.spectral.spectrum import BinnedSpectrum


def bin_index(render: BinGrid) -> np.ndarray:
    """Render bin of each analysis bin (−1 outside the render range)."""
    a = analysis_grid().edges
    pos = np.searchsorted(a, render.edges)
    if not np.allclose(a[np.clip(pos, 0, a.size - 1)], render.edges):
        raise ValueError(f"{render.name or 'render grid'}: every render edge must be an analysis-grid edge")
    out = np.full(a.size - 1, -1)
    for j, (lo, hi) in enumerate(zip(pos[:-1], pos[1:])):
        out[lo:hi] = j
    return out


def weights(binned: BinnedSpectrum, render: BinGrid, observer: str = "2") -> np.ndarray:
    """W (render bins, 3): each render bin's Σ S_i·cmf_i over its analysis bins."""
    if binned.grid.edges.shape != analysis_grid().edges.shape:
        raise ValueError("re-colouring needs a spectrum binned on analysis_grid()")
    per_bin = binned.values[:, None] * colorimetry.cmfs(observer)
    idx = bin_index(render)
    w = np.zeros((len(render), 3))
    np.add.at(w, idx[idx >= 0], per_bin[idx >= 0])
    return w


def luminance_weights(binned: BinnedSpectrum, render: BinGrid) -> np.ndarray:
    """Each render bin's Σ S_i·V_i (the Y column of ``weights`` for the 2° observer)."""
    return weights(binned, render, "2")[:, 1]


def recolour(maps, spectra, render: BinGrid, observer: str = "2") -> np.ndarray:
    """XYZ image (..., 3) of Σ_k map_k under zone spectrum S_k.

    ``maps`` is a sequence of arrays (..., render bins); ``spectra`` the matching
    ``BinnedSpectrum`` values on the analysis grid.
    """
    out = None
    for m, s in zip(maps, spectra, strict=True):
        xyz = np.asarray(m, dtype=float) @ weights(s, render, observer)
        out = xyz if out is None else out + xyz
    return out
