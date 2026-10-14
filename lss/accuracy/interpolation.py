"""Interpolating a sparse measured table of a thermal source (design §11).

A spectral-irradiance calibration lists a lamp at a few dozen uneven wavelengths
(the NIST FEL tables: nine between 380 and 780 nm). Spectral errors are scored at
those wavelengths only, but colour needs the whole visible, so the table must be
filled in. Interpolating the raw values is poor: a thermal spectrum is far from
straight over 50-100 nm. Dividing by a blackbody first leaves a ratio that changes
slowly, and that ratio is interpolated instead.

* ``pchip`` (the default): the ratio to a blackbody at :func:`ratio_temperature`,
  interpolated by a monotone cubic (no overshoot between nodes).
* ``linear``: the same ratio, linear between nodes.
* ``nist-poly5``: NIST's recommended form for its FEL tables, a blackbody times a
  fifth-order polynomial in wavelength, fitted by least squares (here with the
  blackbody's temperature held at :func:`ratio_temperature`; NIST fits it too, in
  Wien's form λ⁻⁵exp(a₀ + a₁/λ)). It does not pass through the nodes, and it
  overshoots between them by up to ~1 % in the UV and 0.5-1 % in the visible and
  near IR (L. K. Huang, R. P. Cebula, E. Hilsenrath, "New procedure for interpolating
  NIST FEL lamp irradiances", Metrologia 35, 381-386 (1998)).

The spread between methods is the floor under any colour score against such a table
(``tests/test_nist_fel.py`` measures it).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

from lss.physics.thermal import planck

METHODS = ("pchip", "linear", "nist-poly5")


def ratio_temperature(wavelength_nm, values) -> float:
    """Temperature of the blackbody whose shape is closest to the table, in log space
    (scale free): the one that leaves the flattest ratio."""
    lam, val = np.asarray(wavelength_nm, float), np.asarray(values, float)

    def spread(t):
        r = np.log(val / planck.spectral_radiance(lam, t))
        return float(np.var(r))

    return float(minimize_scalar(spread, bounds=(1000.0, 4000.0), method="bounded",
                                 options={"xatol": 1e-3}).x)


def interpolate(wavelength_nm, values, at_nm, method: str = "pchip") -> np.ndarray:
    """The table's density at ``at_nm`` (inside the table's range)."""
    lam, val = np.asarray(wavelength_nm, float), np.asarray(values, float)
    at = np.asarray(at_nm, float)
    if np.any(at < lam[0]) or np.any(at > lam[-1]):
        raise ValueError("interpolation is not extrapolation: stay inside the table")
    t = ratio_temperature(lam, val)
    b_nodes, b_at = planck.spectral_radiance(lam, t), planck.spectral_radiance(at, t)
    ratio = val / b_nodes
    if method == "pchip":
        return b_at * PchipInterpolator(lam, ratio)(at)
    if method == "linear":
        return b_at * np.interp(at, lam, ratio)
    if method == "nist-poly5":
        # Scaled wavelength keeps the normal equations well conditioned.
        x, x_at = (lam - lam.mean()) / lam.std(), (at - lam.mean()) / lam.std()
        coeff, *_ = np.linalg.lstsq(np.vander(x, 6) * b_nodes[:, None], val, rcond=None)
        return b_at * (np.vander(x_at, 6) @ coeff)
    raise ValueError(f"unknown interpolation {method!r}; known: {METHODS}")
