"""Colour quantities from exact bin integrals (design §3.2, §3.4).

Thin wrappers over ``colour-science``. Inputs are :class:`BinnedSpectrum` on
the 1 nm analysis grid, whose bin centres coincide with the CIE 1 nm tables;
each bin's exact power is weighted by the table value at its centre.
"""

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import BinnedSpectrum
from lss.spectral.units import K_M

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # matplotlib-absent notice at import
    import colour

OBSERVERS = {
    "2": "CIE 1931 2 Degree Standard Observer",
    "10": "CIE 1964 10 Degree Standard Observer",
}
PHOTOPIC = "CIE 1924 Photopic Standard Observer"


@lru_cache(maxsize=None)
def _table(kind: str, name: str) -> np.ndarray:
    grid = analysis_grid()
    shape = colour.SpectralShape(grid.centres[0], grid.centres[-1], 1.0)
    if kind == "cmfs":
        data = colour.MSDS_CMFS[name].copy().align(shape)
    else:
        data = colour.SDS_LEFS[name].copy().align(shape)
    values = np.asarray(data.values, dtype=float)
    values.setflags(write=False)
    return values


def _check_grid(binned: BinnedSpectrum) -> None:
    ref = analysis_grid().edges
    if binned.grid.edges.shape != ref.shape or not np.allclose(binned.grid.edges, ref):
        raise ValueError("colorimetry needs a spectrum binned on analysis_grid()")


def cmfs(observer: str = "2") -> np.ndarray:
    """Colour-matching functions at the analysis-grid bin centres, shape (bins, 3)."""
    return _table("cmfs", OBSERVERS[observer])


def tristimulus(binned: BinnedSpectrum, observer: str = "2") -> np.ndarray:
    """Unnormalised XYZ = Σ_bins cmf(centre) · power(bin), in spectrum units."""
    _check_grid(binned)
    return binned.values @ _table("cmfs", OBSERVERS[observer])


def luminous(binned: BinnedSpectrum) -> float:
    """K_m Σ V(centre) · power(bin): lumens per (spectrum unit) when the spectrum is radiant."""
    _check_grid(binned)
    return float(K_M * binned.values @ _table("lef", PHOTOPIC))


def xy(binned: BinnedSpectrum, observer: str = "2") -> np.ndarray:
    xyz = tristimulus(binned, observer)
    return xyz[:2] / xyz.sum()


def uv_prime(binned: BinnedSpectrum, observer: str = "2") -> np.ndarray:
    """CIE 1976 u′v′."""
    x, y, z = tristimulus(binned, observer)
    d = x + 15 * y + 3 * z
    return np.array([4 * x / d, 9 * y / d])


def uv_1960(binned: BinnedSpectrum) -> np.ndarray:
    """CIE 1960 uv (the space CCT and Duv are defined in)."""
    u, v = uv_prime(binned)
    return np.array([u, 2.0 * v / 3.0])


def cct_duv(binned: BinnedSpectrum, method: str = "Ohno 2013") -> tuple[float, float]:
    """Correlated colour temperature (K) and Duv, CIE 1931 2° observer."""
    cct, duv = colour.uv_to_CCT(uv_1960(binned), method=method)
    return float(cct), float(duv)


def to_sd(binned: BinnedSpectrum, name: str = "") -> "colour.SpectralDistribution":
    """Mean density per 1 nm bin as a colour-science distribution (360–830 nm)."""
    _check_grid(binned)
    return colour.SpectralDistribution(binned.density, binned.grid.centres, name=name or "lss")


@dataclass(frozen=True)
class ColourRendering:
    ra: float  # CIE 13.3 general index
    ri: dict[int, float]  # special indices R1 … R14
    rf: float  # ANSI/IES TM-30-18 fidelity
    rg: float  # ANSI/IES TM-30-18 gamut

    @property
    def r9(self) -> float:
        return self.ri[9]


def colour_rendering(binned: BinnedSpectrum) -> ColourRendering:
    sd = to_sd(binned)
    with warnings.catch_warnings():
        # colour-science announces each internal resampling of its tables.
        warnings.filterwarnings("ignore", message="Aligning .*")
        cri = colour.colour_rendering_index(sd, additional_data=True)
        tm30 = colour.colour_fidelity_index(sd, additional_data=True, method="ANSI/IES TM-30-18")
    ri = {int(i): float(spec.Q_a) for i, spec in cri.Q_as.items()}
    return ColourRendering(float(cri.Q_a), ri, float(tm30.R_f), float(tm30.R_g))
