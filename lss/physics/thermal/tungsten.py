"""Tungsten as a selective emitter (design §4.1 L2).

ε(λ, T) is De Vos's measured normal spectral emissivity
(``lss/data/tungsten_emissivity.yaml``, which holds the provenance and limits).
Between tabulated points it is linear in wavelength and in temperature. Outside
the measured 1600–2800 K it is extrapolated linearly in T up to 3200 K, and the
caller is told so; outside the measured wavelengths it is not defined at all.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from lss.physics.thermal import planck
from lss.spectral.continuum import Continuum

#: Linear extrapolation in T is accepted this far: Kravchenko & Mamuta (2023) found
#: ε nearly linear in T and extrapolated De Vos to 3200 K; the tests check ours
#: against their 3000 K column.
T_MIN_K, T_MAX_K = 1600.0, 3200.0


@dataclass(frozen=True)
class EmissivityTable:
    wavelengths_nm: np.ndarray  # (n,)
    temperatures_k: np.ndarray  # (m,)
    values: np.ndarray  # (n, m)
    citation: str
    uncertainty_abs: float
    data_version: str

    @property
    def measured_range_k(self) -> tuple[float, float]:
        return float(self.temperatures_k[0]), float(self.temperatures_k[-1])

    @property
    def wavelength_range_nm(self) -> tuple[float, float]:
        return float(self.wavelengths_nm[0]), float(self.wavelengths_nm[-1])

    def extrapolated(self, temperature_k: float) -> bool:
        lo, hi = self.measured_range_k
        return not (lo <= temperature_k <= hi)

    def at_temperature(self, temperature_k: float) -> np.ndarray:
        """ε at every tabulated wavelength, linear in T (extrapolating from the end pair)."""
        if not (T_MIN_K <= temperature_k <= T_MAX_K):
            raise ValueError(f"tungsten emissivity is defined for {T_MIN_K:g}-{T_MAX_K:g} K, got {temperature_k:g} K")
        t = self.temperatures_k
        i = int(np.clip(np.searchsorted(t, temperature_k) - 1, 0, t.size - 2))
        w = (temperature_k - t[i]) / (t[i + 1] - t[i])
        return (1.0 - w) * self.values[:, i] + w * self.values[:, i + 1]


@lru_cache(maxsize=1)
def table() -> EmissivityTable:
    text = resources.files("lss.data").joinpath("tungsten_emissivity.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    rows = sorted((float(k), v) for k, v in doc["table"].items())
    temps = np.asarray(doc["temperatures_k"], dtype=float)
    values = np.asarray([v for _, v in rows], dtype=float)
    if values.shape != (len(rows), temps.size):
        raise ValueError("every table row needs one value per temperature")
    return EmissivityTable(
        wavelengths_nm=np.asarray([k for k, _ in rows]) * 1000.0,
        temperatures_k=temps,
        values=values,
        citation=doc["citation"],
        uncertainty_abs=float(doc["uncertainty_abs"]),
        data_version=doc["data_version"],
    )


def emissivity(wavelength_nm, temperature_k: float):
    """Normal spectral emissivity ε(λ, T); NaN outside the measured wavelengths."""
    tab = table()
    lo, hi = tab.wavelength_range_nm
    lam = np.asarray(wavelength_nm, dtype=float)
    out = np.interp(lam, tab.wavelengths_nm, tab.at_temperature(temperature_k))
    return np.where((lam >= lo) & (lam <= hi), out, np.nan)


@dataclass(frozen=True)
class TungstenContinuum(Continuum):
    """ε(λ, T) · B_λ(T) in W·m⁻²·sr⁻¹·nm⁻¹: radiance of a tungsten surface, normal direction.

    Support is the measured wavelength range only (300–2500 nm). Outside it the
    emissivity is unknown, so there is no total over all wavelengths.
    """

    temperature_k: float

    def __post_init__(self):
        table().at_temperature(self.temperature_k)  # range check up front

    @property
    def support(self):
        return table().wavelength_range_nm

    def density(self, wavelength_nm):
        tab = table()
        lo, hi = tab.wavelength_range_nm
        lam = np.asarray(wavelength_nm, dtype=float)
        eps = np.interp(lam, tab.wavelengths_nm, tab.at_temperature(self.temperature_k), left=0.0, right=0.0)
        inside = (lam >= lo) & (lam <= hi)
        return np.where(inside, eps * planck.spectral_radiance(np.where(inside, lam, lo), self.temperature_k), 0.0)

    def knots(self, lo, hi):
        lo, hi = max(lo, self.support[0]), min(hi, self.support[1])
        if hi <= lo:
            return np.array([])
        # ε has kinks at the table wavelengths; Planck needs ~1 % steps.
        n = max(int(np.ceil(np.log(hi / lo) / np.log(1.01))), 1)
        lam = table().wavelengths_nm
        return np.unique(np.concatenate([np.geomspace(lo, hi, n + 1), lam[(lam > lo) & (lam < hi)]]))


def cavity_emissivity(emissivity, open_fraction: float):
    """Effective emissivity of a cavity with walls of ``emissivity`` (design §4.1 L3).

    One-zone estimate for an isothermal cavity with diffuse walls, irradiated
    uniformly by itself: ε_eff = ε / (ε + (1 − ε)·f), where f is the ratio of the
    opening's area to the radiating wall area. f = 1 is a flat surface (ε_eff = ε);
    f → 0 a closed cavity, a blackbody. Since the gain is largest where ε is smallest,
    a cavity is greyer than its walls: a coiled filament is redder than a flat ribbon
    at the same temperature.
    """
    if not (0.0 < open_fraction <= 1.0):
        raise ValueError(f"open fraction must be in (0, 1], got {open_fraction:g}")
    eps = np.asarray(emissivity, dtype=float)
    return eps / (eps + (1.0 - eps) * open_fraction)


@dataclass(frozen=True)
class TungstenCoilContinuum(TungstenContinuum):
    """ε_eff(λ, T) · B_λ(T): a coiled tungsten filament as a cavity (design §4.1 L3).

    ``open_fraction`` stands for the coil's geometry, which the renderer's ray-traced
    coil will replace (the accurate step of L3).
    """

    open_fraction: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        cavity_emissivity(0.5, self.open_fraction)  # range check up front

    def density(self, wavelength_nm):
        tab = table()
        lo, hi = tab.wavelength_range_nm
        lam = np.asarray(wavelength_nm, dtype=float)
        inside = (lam >= lo) & (lam <= hi)
        eps = np.interp(np.where(inside, lam, lo), tab.wavelengths_nm, tab.at_temperature(self.temperature_k))
        eff = cavity_emissivity(eps, self.open_fraction)
        return np.where(inside, eff * planck.spectral_radiance(np.where(inside, lam, lo), self.temperature_k), 0.0)
