"""Measured halogen reference and tungsten Level 3 (design §4.1 L3, §11, §12)."""

import numpy as np
import pytest

from lss.accuracy import interpolation, references, report
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
    # climbs: Planck, flat tungsten, coiled tungsten.
    nrms = [scores["halogen_nist_fel_f455", level].spectral_nrms for level in (1, 2, 3)]
    assert nrms[0] > nrms[1] > nrms[2]
    l3 = scores["halogen_nist_fel_f455", 3]
    assert l3.spectral_nrms < 0.01 and l3.delta_uv_prime < 1e-3
    assert l3.empirical_params == 2


def test_fits_carry_over_to_the_second_lamp(scores):
    # F-456 is scored with F-455's values, not refitted.
    l3 = scores["halogen_nist_fel_f456", 3]
    assert l3.spectral_nrms < 0.01 and l3.delta_uv_prime < 1e-3


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
