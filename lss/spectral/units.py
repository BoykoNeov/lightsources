"""Physical constants and the package's unit conventions.

Conventions
-----------
* Wavelength: nanometres (nm). Wavelengths are in standard air unless an
  object carries ``medium="vacuum"``; colorimetry (CIE tables) assumes air.
* Spectral densities: per nm. A spectral radiance is W·m⁻²·sr⁻¹·nm⁻¹,
  a spectral flux is W·nm⁻¹.
* Temperatures: kelvin.

Constants are CODATA 2018 exact or recommended values (SI).
"""

import numpy as np

H = 6.62607015e-34  # Planck constant, J·s (exact)
C = 299792458.0  # speed of light in vacuum, m/s (exact)
K_B = 1.380649e-23  # Boltzmann constant, J/K (exact)
SIGMA_SB = 5.670374419e-8  # Stefan–Boltzmann constant, W·m⁻²·K⁻⁴

C1L = 2.0 * H * C**2  # first radiation constant for radiance, W·m²·sr⁻¹
C2 = H * C / K_B  # second radiation constant, m·K
WIEN_B = 2.897771955e-3  # Wien displacement constant, m·K

#: Legacy second radiation constant used by the definition of CIE Illuminant A
#: (CIE 15:2018, §3.1.1). Only for reproducing that definition, never for physics.
C2_CIE_ILLUMINANT_A = 1.435e-2

K_M = 683.0  # maximum luminous efficacy for photopic vision, lm/W (CIE)

NM = 1e-9  # metres per nanometre


def vacuum_to_air(wavelength_nm):
    """Convert vacuum wavelengths to standard air (IAU convention, Morton 2000).

    Valid above 200 nm. Uses the Ciddor-based refractive index of standard dry
    air at 15 °C, 101 325 Pa, 0.045 % CO2 in the form given by Morton (2000,
    ApJS 130, 403, eq. 8).
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    s2 = (1e3 / lam) ** 2  # (wavenumber in µm⁻¹)²
    n = 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)
    return lam / n


def air_to_vacuum(wavelength_nm):
    """Inverse of :func:`vacuum_to_air`, by fixed-point iteration (converges to <1e-12 nm)."""
    air = np.asarray(wavelength_nm, dtype=float)
    vac = air.copy()
    for _ in range(6):
        vac = vac + (air - vacuum_to_air(vac))
    return vac
