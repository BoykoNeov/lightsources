"""Planck checks (design §12, spectral and colour core)."""

import numpy as np
import pytest
from scipy.integrate import quad

from lss.physics.thermal import planck
from lss.spectral.units import SIGMA_SB


@pytest.mark.parametrize("temperature", [1000.0, 2856.0, 6500.0])
def test_total_matches_stefan_boltzmann_on_wide_log_grid(temperature):
    # Integrate on a log grid from 50 nm to 1 mm; the design asks for 0.1 %.
    lam = np.geomspace(50.0, 1e6, 200_001)
    total = np.trapezoid(planck.spectral_radiance(lam, temperature), lam)
    assert total == pytest.approx(SIGMA_SB * temperature**4 / np.pi, rel=1e-3)


@pytest.mark.parametrize("temperature", [1500.0, 2856.0, 5000.0])
@pytest.mark.parametrize("band", [(360.0, 830.0), (200.0, 400.0), (780.0, 3000.0)])
def test_band_fraction_matches_numeric_integral(temperature, band):
    lo, hi = band
    analytic = planck.band_radiance(lo, hi, temperature)
    numeric, _ = quad(planck.spectral_radiance, lo, hi, args=(temperature,), epsabs=0, epsrel=1e-12)
    assert analytic == pytest.approx(numeric, rel=1e-9)


def test_fraction_is_continuous_across_series_switch():
    # x = c2/(λT) = 1 is where fraction_below switches series.
    temperature = 3000.0
    lam_switch = 1.4387768775e-2 / temperature / 1e-9
    below, above = planck.fraction_below(np.array([lam_switch * (1 - 1e-9), lam_switch * (1 + 1e-9)]), temperature)
    assert above - below == pytest.approx(0.0, abs=1e-9)


def test_fraction_limits():
    assert planck.fraction_below(1.0, 2856.0) == pytest.approx(0.0, abs=1e-15)
    assert planck.fraction_below(1e9, 2856.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("temperature", [2000.0, 2856.0, 6500.0])
def test_peak_at_wien_wavelength(temperature):
    peak = planck.peak_wavelength(temperature)
    lam = np.linspace(0.5 * peak, 1.5 * peak, 400_001)
    found = lam[np.argmax(planck.spectral_radiance(lam, temperature))]
    assert found == pytest.approx(peak, abs=0.01)


def test_about_three_quarters_of_2856k_power_is_beyond_1000nm():
    # The design's reason for never taking total power from a visible grid.
    assert 1.0 - planck.fraction_below(1000.0, 2856.0) == pytest.approx(0.76, abs=0.01)
