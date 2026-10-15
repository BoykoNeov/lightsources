"""The power balance: filament temperature from the current alone (design §4.1 L4, §12)."""

from dataclasses import replace
from importlib import resources

import numpy as np
import pytest
import yaml

from lss.accuracy import references, report
from lss.lamp import lamp as lamps
from lss.physics.thermal import balance

F455, F456 = "halogen_nist_fel_f455", "halogen_nist_fel_f456"

#: NIST: "measurements made on our FEL-type standard spectral irradiance lamps operating
#: at 8.000 A ... give a value of about 0.06% per 1 mA in irradiance at 655 nm" (Walker &
#: Thompson, J. Res. NIST 99, 255 (1994)). One figure, so "about 0.06" is read as
#: 0.05-0.07; the band was set before the model was run.
NIST_BAND = (0.05, 0.07)


def _doc():
    text = resources.files("lss.data").joinpath("tungsten_total_emissivity.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _balance(lamp_id=F455, **changes):
    p = lamps.load(lamp_id).physical
    args = dict(cold_resistance_ohm=p["cold_resistance"].value, wire_area_m2=p["filament_wire_area"].value,
                open_fraction=p["coil_open_fraction"].value, gas_share=p["gas_loss_share"].value,
                gas_exponent=p["gas_loss_exponent"].value)
    return balance.FilamentBalance(**{**args, **changes})


def _calibrated(gas_share, open_fraction=None, emissivity=None):
    """F-455's filament with the surface re-solved for another gas share or shape."""
    p = lamps.load(F455).physical
    f = p["coil_open_fraction"].value if open_fraction is None else open_fraction
    area = balance.solve_area(p["lamp_current"].value, p["lamp_voltage"].value, p["cold_resistance"].value,
                              f, gas_share, p["gas_loss_exponent"].value, emissivity)
    return _balance(wire_area_m2=area, open_fraction=f, gas_share=gas_share, emissivity=emissivity)


def test_total_emissivity_transcription():
    doc = _doc()
    # Each row's radiated power over e T^4 is the same constant (the radiation constant
    # of 1934, ~5.71e-12 W cm-2 K-4): a digit misread in either column, beyond the last,
    # would stand out. 0.3 % is the three-figure emissivity's rounding at 1000 K.
    c = np.array([n / (doc["table"][t] * float(t) ** 4) for t, n in doc["radiated_w_cm2"].items()])
    assert np.all(np.abs(c / np.median(c) - 1.0) < 3e-3)
    assert np.median(c) == pytest.approx(5.71e-12, rel=2e-3)
    # The 1945 table reprints the 1934 values where they overlap.
    for t, e in doc["forsythe_adams_1945"].items():
        assert doc["table"][t] == e
    tab = balance.total_emissivity_table()
    assert np.all(np.diff(tab.values) > 0)
    assert tab(3091.2) == pytest.approx(0.337, abs=1e-3)


def test_emissivity_slope_is_poorly_fixed_near_3000_k():
    # The slope the lamp's response depends on: from row differences 0.58 below 3000 K and
    # 0.32 above, and 0.53 from the 1934 table's own slope column. The module's result is
    # therefore reported across that spread, not from one interpolated slope.
    doc = _doc()
    e = doc["table"]
    below = np.log(e[3000] / e[2800]) / np.log(3000 / 2800)
    above = np.log(e[3200] / e[3000]) / np.log(3200 / 3000)
    assert below == pytest.approx(0.58, abs=0.01)
    assert above == pytest.approx(0.32, abs=0.01)
    assert doc["log_slope_n_3000k"] - 4.0 == pytest.approx(0.53)


def test_balance_closes_and_follows_the_current():
    bal = _balance()
    op = bal.operating_point(8.2)
    assert op["power_w"] == pytest.approx(op["radiated_w"] + op["gas_loss_w"], rel=1e-9)
    # The surface was solved at NIST's operating point: Level 3's temperature, NIST's voltage.
    assert op["temperature_k"] == pytest.approx(3091.21, abs=0.05)
    assert op["voltage_v"] == pytest.approx(109.4, abs=0.01)
    assert op["gas_loss_w"] / op["power_w"] == pytest.approx(0.094, abs=0.002)
    currents = np.linspace(7.0, 8.6, 9)
    temps = [bal.temperature(i) for i in currents]
    volts = [bal.operating_point(i)["voltage_v"] for i in currents]
    assert np.all(np.diff(temps) > 0) and np.all(np.diff(volts) > 0)
    with pytest.raises(ValueError):
        bal.temperature(30.0)  # past the emissivity table


def test_lamp_at_8_000_a_matches_nists_measured_response():
    # A sanity check, not a verdict (design §4.1 L4): NIST's one-figure 0.06 %/mA holds over
    # the whole plausible range of the unknowns, so passing it shows sensible behaviour,
    # not that the physics is right. It does bound the gas loss loosely from above.
    for g in (0.0, 0.10, 0.15):
        s = balance.irradiance_sensitivity(_calibrated(g), 8.000, 655.0)
        assert NIST_BAND[0] < s < NIST_BAND[1], (g, s)
    assert balance.irradiance_sensitivity(_calibrated(0.20), 8.000, 655.0) > NIST_BAND[1]
    # A straight wire (no cavity), and either end of the emissivity slope's spread.
    assert NIST_BAND[0] < balance.irradiance_sensitivity(_calibrated(0.10, open_fraction=1.0), 8.000, 655.0) < NIST_BAND[1]
    for slope in (0.32, 0.53):
        em = balance.power_law_emissivity(3091.2, slope)
        s = balance.irradiance_sensitivity(_calibrated(0.10, emissivity=em), 8.000, 655.0)
        assert NIST_BAND[0] < s < NIST_BAND[1], (slope, s)
    # The coil's cavity flattens the emissivity's rise with temperature, so a coil follows
    # the current more steeply than a straight wire.
    assert (balance.irradiance_sensitivity(_calibrated(0.10), 8.000, 655.0)
            > balance.irradiance_sensitivity(_calibrated(0.10, open_fraction=1.0), 8.000, 655.0))


def test_response_does_not_depend_on_the_filaments_size():
    # Twice the surface at the same temperature needs sqrt(2) times the current; the
    # response per fractional change of current is the same. So the NIST check tests
    # how the losses and the resistance scale with temperature, not the fitted surface.
    small = _balance()
    big = replace(small, wire_area_m2=2.0 * small.wire_area_m2)
    i_small, i_big = 8.0, 8.0 * np.sqrt(2.0)
    assert big.temperature(i_big) == pytest.approx(small.temperature(i_small), abs=1e-6)
    s_small = balance.irradiance_sensitivity(small, i_small, 655.0) * i_small
    s_big = balance.irradiance_sensitivity(big, i_big, 655.0, step_a=np.sqrt(2.0) * 1e-3) * i_big
    assert s_big == pytest.approx(s_small, rel=1e-4)


def test_implied_wire_is_plausible():
    # The cold resistance and the solved surface together imply the wire: about 0.29 mm
    # thick and 0.92 m long, a plausible 1 kW, 110 V filament; no datasheet gives it.
    w = _balance().wire()
    assert 0.25e-3 < w["diameter_m"] < 0.33e-3
    assert 0.8 < w["length_m"] < 1.05


def test_level_4_on_the_presets():
    # F-455: the surface was solved at its own operating point, so Level 4 restates Level 3.
    lamp = lamps.load(F455)
    d = lamp.generate(4).metadata["derived"]
    assert d["filament_temperature"]["value"] == pytest.approx(lamp.physical["filament_temperature"].value, abs=0.05)
    assert d["lamp_voltage"]["value"] == pytest.approx(lamp.physical["lamp_voltage"].value, abs=0.01)
    ref = references.get("nist-fel-f455")
    s3, s4 = report.score(lamp, 3, ref), report.score(lamp, 4, ref)
    assert s4.chi2_per_node == pytest.approx(s3.chi2_per_node, abs=0.01)
    assert s4.empirical_params == 5  # cold resistance, surface, open fraction, gas share and exponent
    # F-456 shares F-455's constants and current, so it comes out at F-455's temperature and
    # voltage: 0.4 V below its own reading. Its voltage is not used, which is all that makes
    # its score better than the resistance step's.
    b = lamps.load(F456)
    db = b.generate(4).metadata["derived"]
    assert db["filament_temperature"]["value"] == pytest.approx(d["filament_temperature"]["value"], abs=1e-6)
    assert b.physical["lamp_voltage"].value - db["lamp_voltage"]["value"] == pytest.approx(0.4, abs=0.01)
