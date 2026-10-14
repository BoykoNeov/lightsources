"""Line profile checks (design §12: Voigt limits, Olivero–Longbothum FWHM, normalisation)."""

import numpy as np
import pytest
from scipy.integrate import quad

from lss.spectral.profiles import Gaussian, Lorentzian, Voigt

PROFILES = [Gaussian(0.4), Lorentzian(1.5), Voigt(0.3, 1.0), Voigt(1.0, 0.01), Voigt(0.002, 0.05)]


@pytest.mark.parametrize("profile", PROFILES, ids=repr)
def test_normalised(profile):
    assert profile.cdf(-1e12) == pytest.approx(0.0, abs=1e-12)
    assert profile.cdf(1e12) == pytest.approx(1.0, abs=1e-12)
    x = profile.knots(200 * profile.fwhm)
    numeric = sum(quad(profile.pdf, a, b, epsabs=0, epsrel=1e-12)[0] for a, b in zip(x[:-1], x[1:]))
    assert numeric == pytest.approx(profile.cdf(x[-1]) - profile.cdf(x[0]), abs=1e-11)


@pytest.mark.parametrize("profile", PROFILES, ids=repr)
@pytest.mark.parametrize("x", [0.05, 0.7, 3.0, 40.0])
def test_cdf_is_integral_of_pdf(profile, x):
    ref = 0.5 + quad(profile.pdf, 0.0, x, epsabs=0, epsrel=1e-12, limit=500)[0]
    assert profile.cdf(x) == pytest.approx(ref, abs=1e-12)
    assert profile.cdf(-x) == pytest.approx(1.0 - ref, abs=1e-12)


@pytest.mark.parametrize("profile", PROFILES, ids=repr)
def test_pdf_at_half_width_is_half_peak(profile):
    assert profile.pdf(0.5 * profile.fwhm) == pytest.approx(0.5 * profile.pdf(0.0), rel=1e-9)


def test_voigt_reduces_to_gaussian():
    v, g = Voigt(1.0, 1e-7), Gaussian(1.0)
    # The residual Lorentzian wing (~1e-9) outweighs the Gaussian beyond ~2 FWHM,
    # so compare the core relatively and the wings absolutely.
    core = np.linspace(-1.2, 1.2, 25)
    np.testing.assert_allclose(v.pdf(core), g.pdf(core), rtol=1e-5)
    x = np.linspace(-3, 3, 61)
    np.testing.assert_allclose(v.pdf(x), g.pdf(x), atol=1e-7)
    np.testing.assert_allclose(v.cdf(x), g.cdf(x), atol=1e-6)


def test_voigt_reduces_to_lorentzian():
    v, lor = Voigt(1e-7, 1.0), Lorentzian(1.0)
    x = np.linspace(-30, 30, 61)
    np.testing.assert_allclose(v.pdf(x), lor.pdf(x), rtol=1e-6)
    np.testing.assert_allclose(v.cdf(x), lor.cdf(x), atol=1e-9)


@pytest.mark.parametrize("ratio", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
def test_voigt_fwhm_matches_olivero_longbothum(ratio):
    # Olivero & Longbothum (1977) is itself accurate to about 0.02 %.
    f_g, f_l = 1.0, ratio
    v = Voigt(f_g, f_l)
    assert v.fwhm == pytest.approx(Voigt.olivero_fwhm(f_g, f_l), rel=3e-4)
