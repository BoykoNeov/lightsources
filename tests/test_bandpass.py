"""Triangular-bandpass sampling: a model seen as a 5 nm table would record it (design §11)."""

import numpy as np
import pytest

from lss.accuracy import references, report
from lss.accuracy.references import Reference
from lss.lamp import lamp as lamps
from lss.lamp.profile import EmissionProfile, Zone
from lss.spectral.bandpass import as_recorded, triangle_samples
from lss.spectral.continuum import TabulatedContinuum
from lss.spectral.lines import Line
from lss.spectral.profiles import Gaussian
from lss.spectral.spectrum import Spectrum

NODES = np.arange(380.0, 780.0 + 2.5, 5.0)


def _line(centre, power=2.0):
    return Spectrum.of(Line(centre, power, Gaussian(0.002)))


@pytest.mark.parametrize("centre", [380.02, 404.6565, 435.8335, 437.5, 550.0, 777.1, 779.99])
def test_samples_keep_a_lines_power_wherever_it_sits(centre):
    # Between the end nodes the triangles sum to exactly one.
    s = triangle_samples(_line(centre), NODES)
    assert 5.0 * s.sum() == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("centre", [380.0, 780.0])
def test_a_line_on_an_end_node_loses_only_the_half_beyond_the_table(centre):
    # Half the line lies outside the table, where the end triangle has fallen by |δ|/5:
    # the loss is ½ · mean|δ| / 5 = ½ · σ √(2/π) / 5.
    sigma = 0.002 / (2 * np.sqrt(2 * np.log(2)))
    lost = 0.5 * sigma * np.sqrt(2 / np.pi) / 5.0
    s = triangle_samples(_line(centre), NODES)
    assert 5.0 * s.sum() == pytest.approx(2.0 * (1 - lost), rel=1e-9)


def test_a_line_splits_between_its_two_nodes_by_distance():
    s = triangle_samples(_line(435.8335), NODES)
    k = np.searchsorted(NODES, 435.0)
    assert s[k] == pytest.approx(2.0 / 5.0 * (1 - 0.8335 / 5), rel=1e-12)
    assert s[k + 1] == pytest.approx(2.0 / 5.0 * (0.8335 / 5), rel=1e-12)
    assert np.count_nonzero(s > 1e-12) == 2


def test_a_line_on_a_node_stays_on_it_but_for_its_own_width():
    s = triangle_samples(_line(550.0), NODES)
    k = np.searchsorted(NODES, 550.0)
    assert s[k] == pytest.approx(2.0 / 5.0, rel=2e-4)
    assert s[k - 1] == pytest.approx(s[k + 1], rel=1e-9)
    assert s[k - 1] < 1e-4 * s[k]
    assert np.count_nonzero(s > 1e-12) == 3


def test_a_straight_continuum_is_unchanged():
    # A triangle averages a straight line to its value at the centre.
    lam = np.array([300.0, 900.0])
    flat = Spectrum.of(TabulatedContinuum(lam, 1.0 + 0.01 * (lam - 300.0)))
    assert triangle_samples(flat, NODES) == pytest.approx(1.0 + 0.01 * (NODES - 300.0), rel=1e-13)


def test_needs_equal_steps():
    with pytest.raises(ValueError, match="equally spaced"):
        triangle_samples(_line(500.0), np.array([400.0, 405.0, 412.0]))


def test_a_line_model_scores_zero_against_its_own_recorded_table():
    lines = _line(435.8335, 3.0) + _line(546.075, 5.0) + _line(611.2, 4.0)
    table = as_recorded(lines, NODES)
    ref = Reference(id="synthetic", family="fluorescent", description="lines recorded at 5 nm", spectrum=table,
                    valid_range=(380.0, 780.0), sampling_nm=5.0, kind="definition", uncertainty="exact",
                    includes=frozenset(), citation="test", data_version="test", bandpass_nm=5.0)
    profile = EmissionProfile(zones=(Zone(lines),))
    c = report.compare(profile, ref)
    assert np.sqrt(np.mean(c.residuals**2)) == pytest.approx(0.0, abs=1e-12)
    # The same lines scored at the 1 nm grid without the bandpass would be far off.
    raw = report.compare(profile, Reference(**{**ref.__dict__, "bandpass_nm": None}))
    assert np.sqrt(np.mean(raw.residuals**2)) > 1.0


def test_measured_tables_declare_a_bandpass_and_definitions_do_not():
    assert references.get("cie-fl11").bandpass_nm == 5.0
    assert references.get("cie-led-b5").bandpass_nm == 5.0
    assert references.get("cie-a").bandpass_nm is None
    row = report.score(lamps.load("led_cie_b5"), 1)
    assert row.reference_bandpass_nm == 5.0
