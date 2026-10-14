"""Line binning and filtering checks (design §12: power conservation, exact sharp filters)."""

import numpy as np
import pytest
from scipy.integrate import quad

from lss.spectral.continuum import PlanckContinuum, TabulatedContinuum
from lss.spectral.filters import Transmission, step_filter
from lss.spectral.grids import BinGrid, analysis_grid, extended_grid, render_grid
from lss.spectral.lines import Line
from lss.spectral.profiles import Gaussian, Lorentzian, Voigt
from lss.spectral.spectrum import Spectrum
from lss.spectral.units import air_to_vacuum, vacuum_to_air

PROFILES = [Gaussian(0.02), Gaussian(3.0), Lorentzian(0.5), Voigt(0.3, 1.0)]
GRIDS = [analysis_grid(), extended_grid(), render_grid(10.0), render_grid(47.0),
         BinGrid(np.arange(585.0, 595.0, 0.1))]


@pytest.mark.parametrize("profile", PROFILES, ids=repr)
@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: g.name or f"{len(g)} bins")
def test_binning_conserves_line_power(profile, grid):
    spec = Spectrum.of(Line(589.0, 2.5, profile), Line(436.3, 0.7, profile))
    binned = spec.bin(grid)
    assert binned.total == pytest.approx(3.2, rel=1e-13)
    assert np.all(binned.values >= -1e-15)


def test_narrow_line_lands_in_its_bin():
    binned = Spectrum.of(Line(546.07, 1.0, Gaussian(0.001))).bin(analysis_grid())
    assert binned.values[546 - 360] == pytest.approx(1.0, abs=1e-12)


def test_power_outside_grid_is_reported_not_dropped():
    binned = Spectrum.of(Line(254.0, 1.0, Gaussian(0.01)), Line(1014.0, 2.0, Gaussian(0.01))).bin(analysis_grid())
    assert binned.in_grid == pytest.approx(0.0, abs=1e-15)
    assert binned.below == pytest.approx(1.0)
    assert binned.above == pytest.approx(2.0)


@pytest.mark.parametrize("profile", [Gaussian(0.8), Lorentzian(0.8), Voigt(0.5, 0.5)], ids=repr)
@pytest.mark.parametrize("edge_offset", [-1.0, 0.0, 0.37, 5.0])
def test_sharp_filter_on_line_is_exact_and_grid_independent(profile, edge_offset):
    centre, power = 589.0, 2.0
    spec = Spectrum.of(Line(centre, power, profile)).filtered(step_filter(centre + edge_offset))
    expected = power * (1.0 - profile.cdf(edge_offset))
    totals = [spec.bin(g).total for g in GRIDS]
    for t in totals:
        assert t == pytest.approx(expected, rel=1e-12)
    # The edge falls inside a coarse bin; that bin still gets the exact split.
    coarse = spec.bin(BinGrid(np.array([500.0, 589.0 + edge_offset + 0.5, 700.0])))
    exact_first = power * (profile.cdf(edge_offset + 0.5) - profile.cdf(edge_offset))
    assert coarse.values[0] == pytest.approx(exact_first, rel=1e-12)


def test_sloped_filter_on_line_matches_quadrature():
    profile = Voigt(0.4, 0.6)
    t = Transmission(np.array([580.0, 590.0, 600.0]), np.array([0.2, 0.9, 0.5]))
    spec = Spectrum.of(Line(589.5, 1.0, profile)).filtered(t)
    ref = sum(
        quad(lambda x: profile.pdf(x - 589.5) * t(x), a, b, epsabs=0, epsrel=1e-13, limit=500)[0]
        for a, b in [(-np.inf, 580.0), (580.0, 589.5), (589.5, 590.0), (590.0, 600.0), (600.0, np.inf)]
    )
    assert spec.total() == pytest.approx(ref, rel=1e-10)


def test_tabulated_continuum_bins_exactly():
    cont = TabulatedContinuum(np.array([400.0, 500.0, 700.0]), np.array([0.0, 2.0, 1.0]))
    spec = Spectrum.of(cont)
    assert spec.total() == pytest.approx(100.0 + 300.0)
    assert spec.integral(450.0, 600.0) == pytest.approx(0.5 * (1.0 + 2.0) * 50 + 0.5 * (2.0 + 1.5) * 100)
    assert spec.bin(analysis_grid()).total == pytest.approx(400.0, rel=1e-13)


def test_filter_on_continuum_is_exact():
    cont = TabulatedContinuum(np.array([400.0, 700.0]), np.array([1.0, 1.0]))
    spec = Spectrum.of(cont).filtered(step_filter(512.3, below=0.25, above=1.0))
    assert spec.total() == pytest.approx(0.25 * 112.3 + 187.7, rel=1e-13)


def test_planck_continuum_bins_match_analytic_band():
    from lss.physics.thermal import planck

    binned = Spectrum.of(PlanckContinuum(2856.0)).bin(analysis_grid())
    assert binned.in_grid == pytest.approx(planck.band_radiance(359.5, 830.5, 2856.0), rel=1e-9)
    assert binned.total == pytest.approx(planck.total_radiance(2856.0), rel=1e-9)


def test_vacuum_line_is_binned_at_its_air_wavelength():
    # Na D2: 589.1583 nm in vacuum, 588.9950 nm in air (NIST ASD).
    assert vacuum_to_air(589.1583) == pytest.approx(588.9950, abs=2e-4)
    assert air_to_vacuum(588.9950) == pytest.approx(589.1583, abs=2e-4)
    line = Line(589.1583, 1.0, Gaussian(1e-4), medium="vacuum")
    assert line.air_centre == pytest.approx(588.9950, abs=2e-4)
    edges = np.array([588.9, 589.08, 589.2])
    binned = Spectrum.of(line).bin(BinGrid(edges))
    assert binned.values[0] == pytest.approx(1.0)
