"""Emission bands shaped in photon energy (design §4.2 L1).

An LED pump and a phosphor emit bands whose natural shape is set in photon
energy, not wavelength: a Gaussian in energy is lopsided in wavelength, with a
longer red tail. Energy is carried here as wavenumber ν̃ = 10⁷/λ (cm⁻¹ for λ in
nm), which is proportional to it, so no physical constant enters.

Width convention
----------------
Widths are given in nanometres, measured on the *energy* shape: a half width
is the wavelength distance from the peak to the point where the per-energy
density has fallen to half. Peaks are likewise the per-energy maximum, quoted
as a wavelength. The per-nm density of the same band peaks slightly to the blue
of it, because the λ⁻² Jacobian falls with wavelength (as a blackbody's per-nm
peak lies blue of its per-frequency one), while its long tail is on the red side.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from lss.spectral.continuum import Continuum

#: λ (nm) × ν̃ (cm⁻¹).
_NM_CM = 1e7
_HALF_MAX_Z = np.sqrt(2.0 * np.log(2.0))  # half width at half maximum / σ
#: The band is cut off this many σ from its peak; the mass beyond is < 1e-15.
_CUTOFF_SIGMA = 8.0
#: A wide band is also cut off at 10 µm (1000 cm⁻¹), well short of zero energy;
#: the mass lost there is accounted for in :meth:`EnergyBand.analytic_total`.
_MIN_WAVENUMBER = 1e3
#: Quadrature knots per σ; the count is fixed, so knots move smoothly with the
#: parameters and a fit's finite-difference Jacobian is not disturbed.
_KNOTS_PER_SIGMA = 4


@dataclass(frozen=True)
class EnergyBand(Continuum):
    """``power`` × a split Gaussian in wavenumber: σ_short above the peak
    wavenumber (the short-wavelength side), σ_long below it. Density per nm."""

    peak_wavenumber: float  # cm⁻¹
    sigma_short: float  # cm⁻¹
    sigma_long: float  # cm⁻¹
    power: float = 1.0  # ∫ over all wavelengths, before the cut-off

    def __post_init__(self):
        if self.sigma_short <= 0 or self.sigma_long <= 0:
            raise ValueError("band widths must be positive")
        if self.peak_wavenumber <= _MIN_WAVENUMBER:
            raise ValueError(f"band peak beyond {_NM_CM / _MIN_WAVENUMBER:g} nm")

    @classmethod
    def split(cls, peak_nm: float, half_width_short_nm: float, half_width_long_nm: float,
              power: float = 1.0) -> "EnergyBand":
        """Peak and the two half widths at half maximum, in nm (see the module's width convention)."""
        if not (0 < half_width_short_nm < peak_nm) or half_width_long_nm <= 0:
            raise ValueError("need 0 < short half width < peak, and a positive long half width")
        nu = _NM_CM / peak_nm
        hw_short = _NM_CM / (peak_nm - half_width_short_nm) - nu
        hw_long = nu - _NM_CM / (peak_nm + half_width_long_nm)
        return cls(nu, hw_short / _HALF_MAX_Z, hw_long / _HALF_MAX_Z, power)

    @classmethod
    def symmetric(cls, peak_nm: float, fwhm_nm: float, power: float = 1.0) -> "EnergyBand":
        """Gaussian in energy whose two half-maximum points are ``fwhm_nm`` apart in wavelength."""
        if not (0 < fwhm_nm):
            raise ValueError("fwhm must be positive")
        nu = _NM_CM / peak_nm
        # 10⁷/(ν−h) − 10⁷/(ν+h) = W  ⇒  W h² + 2·10⁷ h − W ν² = 0.
        hw = (np.sqrt(_NM_CM**2 + (fwhm_nm * nu) ** 2) - _NM_CM) / fwhm_nm
        sigma = hw / _HALF_MAX_Z
        return cls(nu, sigma, sigma, power)

    @property
    def peak_nm(self) -> float:
        return _NM_CM / self.peak_wavenumber

    @property
    def _amplitude(self) -> float:
        """Peak density per cm⁻¹, so the untruncated band integrates to ``power``."""
        return self.power / (np.sqrt(np.pi / 2.0) * (self.sigma_short + self.sigma_long))

    @property
    def _wavenumber_range(self) -> tuple[float, float]:
        nu = self.peak_wavenumber
        return max(nu - _CUTOFF_SIGMA * self.sigma_long, _MIN_WAVENUMBER), nu + _CUTOFF_SIGMA * self.sigma_short

    @property
    def support(self):
        lo_nu, hi_nu = self._wavenumber_range
        return _NM_CM / hi_nu, _NM_CM / lo_nu

    def density(self, wavelength_nm):
        lam = np.asarray(wavelength_nm, dtype=float)
        lo, hi = self.support
        inside = (lam >= lo) & (lam <= hi)
        nu = _NM_CM / np.where(inside, lam, lo)
        d = nu - self.peak_wavenumber
        sigma = np.where(d > 0, self.sigma_short, self.sigma_long)
        per_wavenumber = self._amplitude * np.exp(-0.5 * (d / sigma) ** 2)
        return np.where(inside, per_wavenumber * _NM_CM / lam**2, 0.0)

    def knots(self, lo, hi):
        nu = self.peak_wavenumber
        steps = np.arange(1, int(_CUTOFF_SIGMA * _KNOTS_PER_SIGMA) + 1) / _KNOTS_PER_SIGMA
        offsets = np.concatenate([-self.sigma_long * steps[::-1], [0.0], self.sigma_short * steps])
        wavenumbers = nu + offsets
        lam = _NM_CM / wavenumbers[wavenumbers > self._wavenumber_range[0]][::-1]
        return lam[(lam >= lo) & (lam <= hi)]

    def analytic_total(self):
        """The integral over the support: ``power`` less the mass beyond the cut-offs."""
        lo_nu, _ = self._wavenumber_range
        z_long = (self.peak_wavenumber - lo_nu) / (self.sigma_long * np.sqrt(2.0))
        z_short = _CUTOFF_SIGMA / np.sqrt(2.0)
        side = self._amplitude * np.sqrt(np.pi / 2.0)  # each side's untruncated mass per unit σ
        return float(side * (self.sigma_short * erf(z_short) + self.sigma_long * erf(z_long)))
