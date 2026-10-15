"""The spectrograph's response (lss.spectral.instrument): blur, offset and 1 nm sampling."""

import numpy as np
import pytest
from scipy.stats import norm

from lss.spectral.instrument import Kernel, fit_kernel, observe

NODES = np.arange(400.0, 701.0)
KERNEL = Kernel(0.5, 0.8, -0.13)


def _narrow_line(centre, power=3.0, width=0.002):
    lam = np.sort(np.concatenate([np.linspace(380.0, 720.0, 20001), centre + np.linspace(-0.02, 0.02, 801)]))
    return lam, power * norm.pdf(lam, centre, width)


def test_kernel_has_unit_area_and_its_mean_is_offset_plus_tail():
    u = np.linspace(-10.0, 30.0, 400001)
    k = KERNEL.shape(u)
    assert np.trapezoid(k, u) == pytest.approx(1.0, rel=1e-9)
    assert np.trapezoid(u * k, u) + KERNEL.offset_nm == pytest.approx(KERNEL.offset_nm + KERNEL.tail_nm, rel=1e-9)


@pytest.mark.parametrize("centre", [500.0, 500.37, 550.5])
def test_observe_keeps_a_lines_power_and_places_it(centre):
    lam, f = _narrow_line(centre)
    s = observe(lam, f, KERNEL, NODES)
    # Sampling a 1.6 nm wide kernel every 1 nm keeps the power to a few tenths of a per cent,
    # depending on where the line falls between nodes.
    assert np.nansum(s) * 1.0 == pytest.approx(3.0, rel=5e-3)
    # The recorded centroid sits at the line plus the offset plus the tail.
    w = np.abs(NODES - centre) < 20
    assert np.sum(NODES[w] * s[w]) / np.sum(s[w]) == pytest.approx(centre + KERNEL.offset_nm + KERNEL.tail_nm, abs=0.01)
    # And it matches the kernel sampled at the nodes.
    assert s == pytest.approx(KERNEL.line(NODES, centre, 3.0), abs=3e-3 * s.max())


def test_observe_leaves_a_flat_density_flat_and_marks_the_edges():
    lam = np.linspace(380.0, 720.0, 3401)
    s = observe(lam, np.full(lam.size, 2.5), KERNEL, NODES)
    assert s[np.isfinite(s)] == pytest.approx(2.5, rel=1e-6)
    s = observe(np.linspace(450.0, 650.0, 2001), np.ones(2001), KERNEL, NODES)
    # A node gathers from 6σ to its red and 6σ + 15τ to its blue (after the offset).
    assert np.isnan(s[NODES < 450.0 + 15.0 - 0.13]).all() and np.isnan(s[NODES > 650.0 - 3.0 - 0.13]).all()
    assert np.isfinite(s[(NODES >= 466) & (NODES <= 646)]).all()


@pytest.mark.parametrize("drift", [False, True])
def test_fit_recovers_a_kernel_from_lines_sampled_every_nm(drift):
    true = Kernel(0.45, 0.7, -0.2, 0.08 if drift else 0.0)
    lines = (404.6565, 435.8335, 480.3, 546.075, 620.2)
    y = 0.3 + sum(true.line(NODES, c, p) for c, p in zip(lines, (4.0, 9.0, 2.0, 10.0, 5.0)))
    r = fit_kernel(NODES, y, lines, drift=drift)
    k = r.kernel
    assert (k.sigma_nm, k.tail_nm, k.offset_nm, k.offset_slope_per_100nm) == pytest.approx(
        (true.sigma_nm, true.tail_nm, true.offset_nm, true.offset_slope_per_100nm), abs=1e-4)
    assert r.powers == pytest.approx((4.0, 9.0, 2.0, 10.0, 5.0), rel=1e-4)
    assert r.rms_over_peak < 1e-6


def test_bad_kernels_and_crowded_lines_are_refused():
    with pytest.raises(ValueError):
        Kernel(0.0, 0.5)
    with pytest.raises(ValueError):
        fit_kernel(NODES, np.ones(NODES.size), (500.0, 505.0))
