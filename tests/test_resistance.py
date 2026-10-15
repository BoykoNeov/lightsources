"""Filament temperature from resistance, and tungsten Level 4 (design §4.1 L4, §12)."""

import numpy as np
import pytest

from lss.accuracy import fit, references, report
from lss.lamp import lamp as lamps
from lss.physics.thermal import resistance

F455, F456 = "halogen_nist_fel_f455", "halogen_nist_fel_f456"


def _desai_polynomials(t):
    # Desai et al. (1984), the fits printed beside Table 5, in 1e-8 ohm m.
    t = np.asarray(t, dtype=float)
    low = -1.06871 + 2.06884e-2 * t + 1.27971e-6 * t**2 + 8.53101e-9 * t**3 - 5.14195e-12 * t**4
    high = -1.72573 + 2.14350e-2 * t + 5.74811e-6 * t**2 - 1.13698e-9 * t**3 + 1.1167e-13 * t**4
    return np.where(t < 750.0, low, high)


def test_table_matches_desais_polynomials_and_rises_steadily():
    tab = resistance.table()
    assert tab.resistivity(293.0) == pytest.approx(5.28)
    t = np.arange(293.0, 3600.0, 1.0)
    rho = tab.resistivity(t)
    assert np.all(np.diff(rho) > 0)
    # The interpolated table and Desai's own polynomials agree within 0.1 %: this checks
    # both the transcription and the interpolation between rows.
    np.testing.assert_allclose(rho, _desai_polynomials(t), rtol=1e-3)
    # Near lamp temperatures resistance rises a little faster than temperature.
    assert 1.1 < tab.log_slope(3000.0) < 1.3


def test_temperature_round_trips_and_refuses_impossible_ratios():
    tab = resistance.table()
    for t in (500.0, 2000.0, 2856.0, 3100.0):
        assert resistance.temperature(float(tab.ratio(t)) * 0.8, 0.8) == pytest.approx(t, abs=1e-6)
    assert resistance.temperature(1.0, 1.0) == pytest.approx(resistance.T_COLD_K)
    with pytest.raises(ValueError):
        resistance.temperature(0.5, 1.0)  # colder than room temperature
    with pytest.raises(ValueError):
        resistance.temperature(30.0, 1.0)  # past the table's 3600 K


def test_level_4_on_the_fitted_lamp_is_level_3_by_construction():
    # The cold resistance is fitted to F-455, so its Level 4 only restates Level 3's
    # temperature: the same spectrum, the same scores. No test of physics here.
    lamp = lamps.load(F455)
    l4 = lamp.generate(4)
    t4 = l4.metadata["derived"]["filament_temperature"]["value"]
    assert t4 == pytest.approx(lamp.physical["filament_temperature"].value, abs=0.05)
    assert "filament_temperature" not in l4.metadata["params"]
    ref = references.get("nist-fel-f455")
    s3, s4 = report.score(lamp, 3, ref), report.score(lamp, 4, ref)
    assert s4.chi2_per_node == pytest.approx(s3.chi2_per_node, abs=0.01)
    assert s4.empirical_params == s3.empirical_params == 2  # cold resistance swaps for temperature


def test_second_lamp_from_its_own_voltage_runs_hotter_than_its_spectrum_says():
    # The finding (design §4.1 L4). F-456 draws the same current at 0.37 % more voltage.
    # With F-455's cold resistance, tungsten's resistivity turns that into ~9 K hotter;
    # its spectrum, fitted for temperature alone, says ~3 K, and the shape score gets
    # worse, not better. Either the filaments differ in cold resistance by a few tenths of
    # a per cent, or the two voltages were read differently; the data cannot tell which.
    a, b = lamps.load(F455), lamps.load(F456)
    t_a = a.generate(4).metadata["derived"]["filament_temperature"]["value"]
    t_b = b.generate(4).metadata["derived"]["filament_temperature"]["value"]
    assert 8.0 < t_b - t_a < 11.0
    spectral = fit.fit(b, 3, objective="uncertainty", names=["filament_temperature"])
    t_spec = spectral.parameters[0].value
    assert 1.0 < t_spec - t_a < 5.0
    assert t_b - t_spec > 4.0
    ref = references.get("nist-fel-f456")
    assert report.score(b, 4, ref).chi2_per_node > 1.5 * report.score(b, 3, ref).chi2_per_node
    # The cold resistance F-456 would need (or its voltage's excess): a few tenths of a per cent.
    r_hot = b.physical["lamp_voltage"].value / b.physical["lamp_current"].value
    needed = r_hot / float(resistance.table().ratio(t_spec))
    assert 0.1 < 100 * (needed / b.physical["cold_resistance"].value - 1) < 0.5
