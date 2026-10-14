"""Planck's law in the package's units (design §4.1 L1, §3.2).

Spectral radiance is per nm: W·m⁻²·sr⁻¹·nm⁻¹, wavelength in nm.
Total and band integrals are analytic (never a grid sum), because most of a
thermal source's power lies outside any visible grid.
"""

import numpy as np
from scipy.special import bernoulli, factorial

from lss.spectral.units import C1L, C2, NM, SIGMA_SB, WIEN_B

_BERNOULLI = bernoulli(30)  # B_0 … B_30, with B_1 = −1/2


def spectral_radiance(wavelength_nm, temperature_k, c2: float = C2):
    """Blackbody spectral radiance B_λ(T) in W·m⁻²·sr⁻¹·nm⁻¹.

    ``c2`` may be overridden only to reproduce historical definitions
    (e.g. CIE Illuminant A); physics always uses the CODATA value.
    """
    lam_m = np.asarray(wavelength_nm, dtype=float) * NM
    x = c2 / (lam_m * temperature_k)
    with np.errstate(over="ignore"):
        per_m = C1L / lam_m**5 / np.expm1(x)
    return per_m * NM


def total_radiance(temperature_k):
    """∫₀^∞ B_λ dλ = σT⁴/π, in W·m⁻²·sr⁻¹."""
    return SIGMA_SB * np.asarray(temperature_k, dtype=float) ** 4 / np.pi


def fraction_below(wavelength_nm, temperature_k):
    """Fraction of total blackbody radiance at wavelengths below ``wavelength_nm``.

    With x = c₂/(λT):

    * x ≥ 1: F = (15/π⁴) Σₙ e^{-nx} (x³/n + 3x²/n² + 6x/n³ + 6/n⁴), which converges
      like e^{-nx} (60 terms are far below double precision);
    * x < 1 (long wavelengths): F = 1 − (15/π⁴) ∫₀ˣ t³/(eᵗ−1) dt, with the integral
      from its Bernoulli series Σ_k B_k x^{k+3} / (k! (k+3)), convergent for x < 2π.
    """
    lam_m = np.asarray(wavelength_nm, dtype=float) * NM
    x = np.atleast_1d(C2 / (lam_m * temperature_k)).astype(float)
    out = np.empty_like(x)

    big = x >= 1.0
    xb = x[big]
    n = np.arange(1, 61)[:, None]
    series = np.exp(-n * xb) * (xb**3 / n + 3 * xb**2 / n**2 + 6 * xb / n**3 + 6 / n**4)
    out[big] = 15.0 / np.pi**4 * series.sum(axis=0)

    xs = x[~big]
    k = np.arange(0, 31)
    coeff = _BERNOULLI[k] / (factorial(k) * (k + 3))
    integral = (coeff[:, None] * xs ** (k[:, None] + 3)).sum(axis=0)
    out[~big] = 1.0 - 15.0 / np.pi**4 * integral

    return out.reshape(np.shape(lam_m)) if np.ndim(lam_m) else float(out[0])


def band_radiance(lo_nm, hi_nm, temperature_k):
    """∫_lo^hi B_λ dλ in W·m⁻²·sr⁻¹, analytic."""
    frac = fraction_below(hi_nm, temperature_k) - fraction_below(lo_nm, temperature_k)
    return frac * total_radiance(temperature_k)


def peak_wavelength(temperature_k):
    """Wien's displacement law: wavelength of maximum B_λ, in nm."""
    return WIEN_B / np.asarray(temperature_k, dtype=float) / NM
