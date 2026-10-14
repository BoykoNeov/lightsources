"""Measured halogen reference and tungsten Level 3 (design §4.1 L3, §11, §12)."""

import numpy as np
import pytest
from scipy.optimize import least_squares

from lss.accuracy import fit, interpolation, references, report
from lss.lamp import lamp as lamps
from lss.physics.thermal import tungsten
from lss.spectral import colorimetry as cm
from lss.spectral.continuum import TabulatedContinuum
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import Spectrum
from lss.spectral.units import C2

F455, F456 = "nist-fel-f455", "nist-fel-f456"


def _uncertainty_k2(lam):
    u = references.data_file("nist_fel.yaml")["uncertainty_k2_percent"]
    return np.interp(lam, [float(k) for k in u], [float(v) for v in u.values()]) / 100.0


def test_loader_reads_the_tables_in_si_units():
    a = references.get(F455)
    assert a.scored_at_nodes and a.sampling_nm is None and a.bandpass_nm is None
    assert a.kind == "measurement" and a.includes == frozenset({"envelope"})
    assert a.valid_range == (250.0, 2400.0) and a.nodes.size == 35
    assert 654.6 in a.nodes
    # 112.7 W/cm^3 at 555 nm is 0.1127 W m^-2 nm^-1.
    assert a.node_values[a.nodes == 555.0][0] == pytest.approx(0.1127)


def test_spectrum_passes_through_the_nodes():
    a = references.get(F455)
    np.testing.assert_allclose(report.density_at(a.spectrum, a.nodes[1:-1], width_nm=1e-3),
                               a.node_values[1:-1], rtol=1e-4)


def test_transcription_second_lamp_differs_only_smoothly():
    # Two lamps of one design differ by a scale and a small temperature difference:
    # ln(F-456 / F-455) = a + b/λ. A misread digit would stand out from that by ~1 %;
    # every node stays within half the stated k = 2 uncertainty.
    a, b = references.get(F455), references.get(F456)
    lam = a.nodes
    r = np.log(b.node_values / a.node_values)
    design = np.c_[np.ones_like(lam), 1.0 / lam]
    coeff, *_ = np.linalg.lstsq(design, r, rcond=None)
    assert np.all(np.abs(r - design @ coeff) < 0.5 * _uncertainty_k2(lam))
    # The fitted b is a temperature difference: a few kelvin, F-456 the hotter.
    dt = coeff[1] / (C2 * 1e9) * 3075.0**2
    assert 0.5 < dt < 5.0


def test_interpolation_floor_under_colour_scores():
    # How the table is filled in between its nine visible nodes moves its colour by
    # < 2e-4 in u'v' and < 3 K in CCT: the floor under the report's colour numbers.
    a = references.get(F455)
    dense = np.union1d(np.arange(250.0, 2401.0), a.nodes)
    uv, cct = [], []
    for method in interpolation.METHODS:
        s = Spectrum.of(TabulatedContinuum(dense, interpolation.interpolate(a.nodes, a.node_values, dense, method)))
        binned = s.bin(analysis_grid())
        uv.append(cm.uv_prime(binned))
        cct.append(cm.cct_duv(binned)[0])
    uv = np.array(uv)
    assert np.max(np.hypot(*(uv - uv[0]).T)) < 2e-4
    assert np.ptp(cct) < 3.0
    with pytest.raises(ValueError):
        interpolation.interpolate(a.nodes, a.node_values, [240.0])


def test_cavity_emissivity_limits():
    eps = np.array([0.2, 0.45])
    np.testing.assert_allclose(tungsten.cavity_emissivity(eps, 1.0), eps)
    near_closed = tungsten.cavity_emissivity(eps, 1e-6)
    np.testing.assert_allclose(near_closed, 1.0, atol=1e-5)
    eff = tungsten.cavity_emissivity(eps, 0.35)
    assert np.all(eff > eps) and np.all(eff < 1.0)
    # Greyer: the gain is largest where the wall emissivity is smallest.
    assert eff[0] / eps[0] > eff[1] / eps[1]
    with pytest.raises(ValueError):
        tungsten.cavity_emissivity(eps, 0.0)


def test_open_coil_is_the_flat_ribbon_and_a_coil_is_redder():
    lam = np.array([400.0, 700.0, 1500.0])
    flat = tungsten.TungstenContinuum(3000.0)
    np.testing.assert_allclose(tungsten.TungstenCoilContinuum(3000.0, open_fraction=1.0).density(lam),
                               flat.density(lam), rtol=1e-12)
    cct_flat = cm.cct_duv(Spectrum.of(flat).bin(analysis_grid()))[0]
    cct_coil = cm.cct_duv(Spectrum.of(tungsten.TungstenCoilContinuum(3000.0, open_fraction=0.35))
                          .bin(analysis_grid()))[0]
    assert cct_coil < cct_flat - 20.0


@pytest.fixture(scope="module")
def scores():
    return {(lid, level): report.score(lamps.load(lid), level)
            for lid in ("halogen_nist_fel_f455", "halogen_nist_fel_f456") for level in (1, 2, 3)}


def test_better_tungsten_physics_scores_better_on_a_measured_lamp(scores):
    # Against real tungsten (unlike Illuminant A, a blackbody by definition) the ladder
    # climbs: Planck, flat tungsten, coiled tungsten. Only the coil is within the total
    # stated uncertainty point by point (chi2 per node about 1, nodes independent); as a
    # test of shape that is generous (see the treatment test below).
    rows = [scores["halogen_nist_fel_f455", level] for level in (1, 2, 3)]
    assert rows[0].spectral_nrms > rows[1].spectral_nrms > rows[2].spectral_nrms
    assert rows[0].chi2_per_node > rows[1].chi2_per_node > 10.0 > 1.5 > rows[2].chi2_per_node
    l3 = rows[2]
    assert l3.delta_uv_prime < 3e-4 and abs(l3.delta_cct_k) < 10.0
    assert l3.empirical_params == 2


def test_fits_carry_over_to_the_second_lamp(scores):
    # F-456 is scored with F-455's values, not refitted.
    l3 = scores["halogen_nist_fel_f456", 3]
    assert l3.chi2_per_node < 1.5 and l3.delta_uv_prime < 5e-4


def test_scored_at_nodes_within_the_models_own_range(scores):
    l1, l2 = scores["halogen_nist_fel_f455", 1], scores["halogen_nist_fel_f455", 2]
    assert l1.range_nm == (250.0, 2400.0) and l1.reference_nodes == 35  # Planck is defined everywhere
    assert l2.range_nm == (300.0, 2400.0) and l2.reference_nodes == 30  # De Vos starts at 300 nm
    assert l2.reference_sampling_nm is None
    assert set(l1.band_nrms) == {"250-380", "380-480", "480-580", "580-680", "680-780", "780-1100",
                                 "1100-1700", "1700-2500"}


def test_node_residuals_are_the_reported_nrms():
    lamp = lamps.load("halogen_nist_fel_f455")
    ref = references.get(F455)
    c = report.compare(report.model_profile(lamp, 3, ref), ref)
    assert c.residuals.size == 30
    assert np.sqrt(np.mean(c.residuals**2)) == pytest.approx(report.score(lamp, 3).spectral_nrms, rel=1e-12)


def test_node_on_the_models_range_edge_is_not_averaged_with_zero():
    # De Vos's data start at 300 nm, a table node: its window must stay inside.
    lamp = lamps.load("halogen_nist_fel_f455")
    ref = references.get(F455)
    c = report.compare(report.model_profile(lamp, 3, ref), ref)
    assert c.node_wavelengths[0] == 300.0
    rel = c.node_model / c.node_reference
    assert abs(rel[0] - rel[1]) < 0.02  # not half (the bug this guards against)


def test_uncertainty_objective_minimises_the_reported_chi2():
    lamp = lamps.load("halogen_nist_fel_f455")
    ref = references.get(F455)
    c = report.compare(report.model_profile(lamp, 3, ref), ref)
    assert np.mean(c.pulls**2) == pytest.approx(report.score(lamp, 3).chi2_per_node, rel=1e-12)
    # The preset holds the uncertainty fit: refitting does not move it.
    result = fit.fit(lamp, 3, objective="uncertainty")
    assert result.after.chi2_per_node == pytest.approx(result.before.chi2_per_node, rel=1e-3)
    with pytest.raises(ValueError, match="uncertainty"):
        fit.fit(lamps.load("incandescent_cie_a"), 2, objective="uncertainty")


def _covariance(lam, treatment):
    """Relative covariance (k = 1) of the F-455 table at ``lam`` from the report's
    uncertainty parts: "independent" (each node's total, uncorrelated), "random" (random
    parts only) or "correlated" (random parts plus each systematic part fully correlated
    across wavelength). The report states no correlations; these bracket them."""
    comp = references.data_file("nist_fel.yaml")["uncertainty_components_k2_percent"]
    parts = [(np.interp(lam, comp["wavelengths_nm"], p["values"]) / 200.0, p["kind"]) for p in comp["parts"]]
    rand = sum(u**2 for u, kind in parts if kind == "random")
    systematic = [u for u, kind in parts if kind == "systematic"]
    if treatment == "independent":
        return np.diag(rand + sum(u**2 for u in systematic))
    if treatment == "random":
        return np.diag(rand)
    return np.diag(rand) + sum(np.outer(u, u) for u in systematic)


def _window_fit(lo, hi, treatment="independent"):
    """Temperature and open fraction fitted to the nodes in [lo, hi] only, by generalised
    least squares with the treatment's covariance (scale free within the window); values,
    standard errors and chi2 per degree of freedom."""
    lamp = lamps.load("halogen_nist_fel_f455")
    ref = references.get(F455)
    lam = ref.nodes[(ref.nodes >= max(lo, 300.0)) & (ref.nodes <= hi)]
    cov = _covariance(lam, treatment)

    def whitened(x):
        trial = fit.with_values(lamp, 3, {"filament_temperature": x[0], "coil_open_fraction": x[1]})
        c = report.compare(report.model_profile(trial, 3, ref), ref)
        sel = np.isin(c.node_wavelengths, lam)
        m, r = c.node_model[sel], c.node_reference[sel]
        chol = np.linalg.cholesky(cov * np.outer(r, r))
        wm, wr = np.linalg.solve(chol, m), np.linalg.solve(chol, r)
        return (wm @ wr) / (wm @ wm) * wm - wr

    sol = least_squares(whitened, [3075.0, 0.35], bounds=([2600.0, 0.05], [3200.0, 1.0]), x_scale=[100.0, 0.1])
    sd = np.sqrt(np.diag(np.linalg.inv(sol.jac.T @ sol.jac)))  # whitened: covariance in absolute terms
    return sol.x, sd, np.sum(sol.fun**2) / (sol.fun.size - 3)


def test_uncertainty_parts_add_up_to_the_stated_total():
    lam = references.get(F455).nodes
    total = np.sqrt(np.diag(_covariance(lam, "independent")))
    np.testing.assert_allclose(total, references.get(F455).node_sigma, rtol=0.04)
    visible = (lam >= 380) & (lam <= 780)
    assert np.all(np.sqrt(np.diag(_covariance(lam, "random")))[visible] < 0.0012)  # 0.1 % or less, k = 1


def _split(treatment):
    """Gap between the open fractions fitted to 300-800 and 800-2400 nm, in standard
    errors (each window's scaled by sqrt(chi2/dof) where that exceeds 1)."""
    (_, f_s), (_, sd_s), chi_s = _window_fit(300.0, 800.0, treatment)
    (_, f_l), (_, sd_l), chi_l = _window_fit(800.0, 2400.0, treatment)
    sd_s, sd_l = sd_s * np.sqrt(max(chi_s, 1.0)), sd_l * np.sqrt(max(chi_l, 1.0))
    return (f_l - f_s) / np.hypot(sd_s, sd_l)


def test_coil_shape_verdict_depends_on_how_the_uncertainty_correlates():
    # Design §4.1 L3, the treatment table. With each node's total uncertainty treated as
    # independent, one (T, f_open) fits 300-2400 nm and the two halves agree (1.7 sigma).
    # NIST's random part is far smaller than its total: with the random parts only, or
    # with each systematic part fully correlated, the fit fails (chi2/dof ~4-5) and the
    # halves differ by ~2.5-2.8 sigma even after scaling by sqrt(chi2/dof). Likely a real
    # shape miss; the report does not state the correlations that would settle it.
    assert _window_fit(300.0, 2400.0, "independent")[2] < 1.2
    assert 0.0 < _split("independent") < 2.0
    for treatment in ("random", "correlated"):
        assert _window_fit(300.0, 2400.0, treatment)[2] > 2.5
        assert _split(treatment) > 2.3
