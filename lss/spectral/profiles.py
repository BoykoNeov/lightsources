"""Generic normalised line profiles φ(Δλ) per nm (design §3.1, §4.4).

Every profile is normalised, ∫φ dΔλ = 1, and offers an exact (or near-machine
precision) cumulative distribution, so binning a line telescopes: bin powers
always sum to the line power, whatever the bin width.

Pressure-broadening physics (resonance, van der Waals, Stark widths, and the
quasi-static wings of HPS) lives in ``lss.physics.hid``; it produces the
widths these generic shapes take.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import erf, voigt_profile

from lss.spectral.quadrature import segment_integrals

_SIGMA_PER_FWHM = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


class Profile(ABC):
    """A symmetric, normalised line shape as a function of offset Δλ from line centre (nm)."""

    @property
    @abstractmethod
    def fwhm(self) -> float: ...

    @abstractmethod
    def pdf(self, dx): ...

    @abstractmethod
    def cdf(self, dx): ...

    def knots(self, span: float) -> np.ndarray:
        """Offsets that resolve the profile out to ±span, for line-local quadrature.

        Dense in the core (fwhm/40), growing geometrically by 25 % per step, so
        each segment is smooth on its own scale and an 8-point rule is accurate.
        """
        h0 = self.fwhm / 40.0
        span = max(span, 4.0 * self.fwhm)
        n = int(np.ceil(np.log1p(span / h0 * 0.25) / np.log(1.25))) + 1
        pos = h0 * (1.25 ** np.arange(n) - 1.0) / 0.25
        pos = pos[pos < span]
        pos = np.append(pos, span)
        return np.concatenate([-pos[:0:-1], pos])


class Gaussian(Profile):
    """Doppler-like profile with full width at half maximum ``fwhm`` (nm)."""

    def __init__(self, fwhm: float):
        if fwhm <= 0:
            raise ValueError("fwhm must be positive")
        self._fwhm = float(fwhm)
        self.sigma = self._fwhm * _SIGMA_PER_FWHM

    @property
    def fwhm(self) -> float:
        return self._fwhm

    def pdf(self, dx):
        z = np.asarray(dx, dtype=float) / self.sigma
        return np.exp(-0.5 * z * z) / (self.sigma * np.sqrt(2.0 * np.pi))

    def cdf(self, dx):
        return 0.5 * (1.0 + erf(np.asarray(dx, dtype=float) / (self.sigma * np.sqrt(2.0))))

    def knots(self, span: float) -> np.ndarray:
        # Beyond 40 σ the Gaussian is below 1e-300; no point resolving it further.
        return super().knots(min(span, 40.0 * self.sigma))

    def __repr__(self):
        return f"Gaussian(fwhm={self._fwhm:g})"


class Lorentzian(Profile):
    """Impact-broadened (dispersion) profile with full width at half maximum ``fwhm`` (nm)."""

    def __init__(self, fwhm: float):
        if fwhm <= 0:
            raise ValueError("fwhm must be positive")
        self._fwhm = float(fwhm)
        self.gamma = 0.5 * self._fwhm  # half width at half maximum

    @property
    def fwhm(self) -> float:
        return self._fwhm

    def pdf(self, dx):
        dx = np.asarray(dx, dtype=float)
        return self.gamma / (np.pi * (dx * dx + self.gamma * self.gamma))

    def cdf(self, dx):
        return 0.5 + np.arctan(np.asarray(dx, dtype=float) / self.gamma) / np.pi

    def __repr__(self):
        return f"Lorentzian(fwhm={self._fwhm:g})"


class Voigt(Profile):
    """Gaussian ⊗ Lorentzian, given by the two component FWHMs (nm).

    The CDF is built once by 8-point Gauss–Legendre quadrature on geometric
    line-local knots out to 10⁴ Voigt widths; beyond that the profile equals
    its Lorentzian far wing and the tail mass is analytic.
    """

    TAIL_START = 1e4  # in units of the Voigt FWHM

    def __init__(self, fwhm_gaussian: float, fwhm_lorentzian: float):
        if fwhm_gaussian <= 0 or fwhm_lorentzian <= 0:
            raise ValueError("both widths must be positive; use Gaussian or Lorentzian for a limit")
        self.fwhm_gaussian = float(fwhm_gaussian)
        self.fwhm_lorentzian = float(fwhm_lorentzian)
        self.sigma = self.fwhm_gaussian * _SIGMA_PER_FWHM
        self.gamma = 0.5 * self.fwhm_lorentzian
        self._fwhm = self._find_fwhm()

        x_max = self.TAIL_START * self._fwhm
        k = self.knots(x_max)
        self._knots = k[k >= 0.0]
        seg = segment_integrals(self.pdf, self._knots[:-1], self._knots[1:])
        self._half_mass = np.concatenate([[0.0], np.cumsum(seg)])  # ∫₀^knot φ
        self._x_max = self._knots[-1]
        half_total = self._half_mass[-1] + self._tail(self._x_max)
        if abs(half_total - 0.5) > 1e-9:
            raise ArithmeticError(f"Voigt normalisation off by {half_total - 0.5:.3g}")

    def _find_fwhm(self) -> float:
        from scipy.optimize import brentq

        peak = self.pdf(0.0)
        hi = self.fwhm_gaussian + self.fwhm_lorentzian  # FWHM_V never exceeds f_G + f_L
        return 2.0 * brentq(lambda x: self.pdf(x) - 0.5 * peak, 0.0, hi, xtol=1e-15, rtol=1e-14)

    @property
    def fwhm(self) -> float:
        return self._fwhm

    @staticmethod
    def olivero_fwhm(fwhm_gaussian: float, fwhm_lorentzian: float) -> float:
        """Olivero & Longbothum (1977) approximation, accurate to about 0.02 %."""
        f_l = fwhm_lorentzian
        return 0.5346 * f_l + np.sqrt(0.2166 * f_l * f_l + fwhm_gaussian**2)

    def pdf(self, dx):
        return voigt_profile(np.asarray(dx, dtype=float), self.sigma, self.gamma)

    def _tail(self, x):
        """Mass beyond |Δλ| = x in one wing (Lorentzian asymptote)."""
        return 0.5 - np.arctan(np.asarray(x, dtype=float) / self.gamma) / np.pi

    def cdf(self, dx):
        dx = np.asarray(dx, dtype=float)
        ax = np.abs(dx)
        inner = ax < self._x_max
        half = np.empty_like(ax)
        half[~inner] = 0.5 - self._tail(ax[~inner])
        xi = ax[inner]
        i = np.clip(np.searchsorted(self._knots, xi, side="right") - 1, 0, self._knots.size - 2)
        left = self._knots[i]
        half[inner] = self._half_mass[i] + segment_integrals(self.pdf, left, xi)
        return 0.5 + np.sign(dx) * half

    def __repr__(self):
        return f"Voigt(fwhm_gaussian={self.fwhm_gaussian:g}, fwhm_lorentzian={self.fwhm_lorentzian:g})"
