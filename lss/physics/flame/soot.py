"""Soot as the light of a flame (design §4.5 L2).

A flame's soot particles are far smaller than the wavelength, so a layer of them absorbs
(and, by Kirchhoff, emits) in the Rayleigh limit:

    κ_λ = 6π·E(m_λ)·f_v / λ,   radiance = (1 − exp(−κ_λ·L))·B_λ(T)

with E(m) = −Im[(m² − 1)/(m² + 2)] for soot's complex index m = n − ik, f_v the soot
volume fraction and L the path through the flame. Scattering is left out. f_v and L
only enter as their product, so the layer is set by its optical depth at 550 nm.

The optics are tabulated in ``lss/data/soot_optics.yaml``: Chang & Charalampopoulos's
measured index (the generator's), and two bounds on its change with wavelength, used to
bracket the temperature (the exponent and the temperature trade off, so it is never fitted).
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from lss.physics.thermal import planck
from lss.spectral.continuum import Continuum

REFERENCE_NM = 550.0


@lru_cache(maxsize=1)
def _doc() -> dict:
    return yaml.safe_load(resources.files("lss.data").joinpath("soot_optics.yaml").read_text(encoding="utf-8"))


def data_version() -> str:
    return _doc()["data_version"]


def sets() -> tuple[str, ...]:
    return tuple(_doc()["sets"])


def entry(name: str) -> dict:
    try:
        return _doc()["sets"][name]
    except KeyError:
        raise KeyError(f"no soot optics {name!r}; known: {', '.join(sets())}") from None


def range_nm(name: str) -> tuple[float, float]:
    lo, hi = entry(name)["range_nm"]
    return float(lo), float(hi)


def _poly_ln(coeffs, wavelength_nm):
    x = np.log(np.asarray(wavelength_nm, dtype=float) / 1000.0)
    return sum(c * x**i for i, c in enumerate(coeffs))


def refractive_index(wavelength_nm, name: str = "chang1990"):
    """Complex index m = n − ik of a dispersion set; NaN outside its range."""
    e = entry(name)
    if e["kind"] != "dispersion":
        raise ValueError(f"soot optics {name!r} give no refractive index")
    lam = np.asarray(wavelength_nm, dtype=float)
    lo, hi = range_nm(name)
    m = _poly_ln(e["n"], lam) - 1j * _poly_ln(e["k"], lam)
    return np.where((lam >= lo) & (lam <= hi), m, np.nan)


def absorption_function(m):
    """E(m) = −Im[(m² − 1)/(m² + 2)] for m = n − ik (positive for an absorber)."""
    m = np.asarray(m, dtype=complex)
    with np.errstate(invalid="ignore"):  # NaN outside a set's range stays NaN
        return -((m * m - 1.0) / (m * m + 2.0)).imag


def relative_absorption(wavelength_nm, name: str = "chang1990"):
    """κ_λ / κ_550: the absorption coefficient's shape, 1 at 550 nm; NaN outside the range."""
    e = entry(name)
    lam = np.asarray(wavelength_nm, dtype=float)
    lo, hi = range_nm(name)
    if e["kind"] == "dispersion":
        e_ratio = absorption_function(refractive_index(lam, name)) / absorption_function(
            refractive_index(REFERENCE_NM, name))
        shape = e_ratio * REFERENCE_NM / lam
    elif e["kind"] == "constant_e":
        shape = REFERENCE_NM / lam
    elif e["kind"] == "power_law":
        shape = (REFERENCE_NM / lam) ** float(e["alpha"])
    else:
        raise ValueError(f"soot optics {name!r}: unknown kind {e['kind']!r}")
    return np.where((lam >= lo) & (lam <= hi), shape, np.nan)


@dataclass(frozen=True)
class SootContinuum(Continuum):
    """(1 − exp(−τ_λ))·B_λ(T) in W·m⁻²·sr⁻¹·nm⁻¹: a uniform soot layer seen through.

    ``optical_depth`` is τ at 550 nm, κ_550·L. Small values give the optically thin
    shape, τ_λ·B_λ(T), scaled by τ. Support is the optics' range only: outside it soot's
    index is not given, so there is no total over all wavelengths.
    """

    temperature_k: float
    optical_depth: float
    optics: str = "chang1990"

    def __post_init__(self):
        if not self.temperature_k > 0:
            raise ValueError(f"soot temperature must be positive, got {self.temperature_k:g} K")
        if not self.optical_depth > 0:
            raise ValueError(f"optical depth must be positive, got {self.optical_depth:g}")
        entry(self.optics)

    @property
    def support(self):
        return range_nm(self.optics)

    def emissivity(self, wavelength_nm):
        """1 − exp(−τ_λ); zero outside the optics' range."""
        lam = np.asarray(wavelength_nm, dtype=float)
        lo, hi = self.support
        inside = (lam >= lo) & (lam <= hi)
        tau = self.optical_depth * relative_absorption(np.where(inside, lam, REFERENCE_NM), self.optics)
        return np.where(inside, -np.expm1(-tau), 0.0)

    def density(self, wavelength_nm):
        lam = np.asarray(wavelength_nm, dtype=float)
        lo, _ = self.support
        return self.emissivity(lam) * planck.spectral_radiance(np.maximum(lam, lo), self.temperature_k)

    def knots(self, lo, hi):
        lo, hi = max(lo, self.support[0]), min(hi, self.support[1])
        if hi <= lo:
            return np.array([])
        n = max(int(np.ceil(np.log(hi / lo) / np.log(1.01))), 1)  # Planck needs ~1 % steps
        return np.geomspace(lo, hi, n + 1)
