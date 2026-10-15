"""LED Level 3: the drive heats the die and the phosphors (design §4.2 L3)."""

from dataclasses import replace

import numpy as np
import pytest

from lss.lamp import lamp as lamps
from lss.lamp.lamp import ModelSpec
from lss.params import Parameter
from lss.physics.led import thermal
from lss.spectral.grids import analysis_grid
from lss.studies import led_drive

LEDS = ["led_cie_b1", "led_cie_b2", "led_cie_b3", "led_cie_b4", "led_cie_b5"]


def test_junction_temperature_is_the_board_plus_the_heat_through_the_resistance():
    # 0.35 A × 3.0 V = 1.05 W in, 0.35 W out as light: 0.70 W of heat through 20 K/W.
    assert thermal.junction_temperature(0.35, 3.0, 0.35, 300.0, 20.0) == pytest.approx(314.0)
    assert thermal.junction_temperature(0.35, 3.0, 0.35, 300.0, 0.0) == 300.0
    with pytest.raises(ValueError):
        thermal.junction_temperature(0.35, 3.0, 2.0, 300.0, 20.0)  # more light than input
    with pytest.raises(ValueError):
        thermal.junction_temperature(0.35, 3.0, 0.35, 300.0, -1.0)


@pytest.mark.parametrize("retained, barrier", [(0.903, 0.3), (0.94, 0.3), (0.9, 0.15), (0.9, 0.8)])
def test_quenching_keeps_the_cited_fraction_at_150_c(retained, barrier):
    eta = lambda t: thermal.quantum_yield(t, retained, barrier)  # noqa: E731
    assert eta(thermal.T_HOT_K) / eta(thermal.T_ROOM_K) == pytest.approx(retained, rel=1e-12)
    temps = np.linspace(250.0, 500.0, 11)
    assert np.all(np.diff([eta(t) for t in temps]) < 0)  # one barrier: falls steadily with T
    assert thermal.relative_yield(350.0, 350.0, retained, barrier) == pytest.approx(1.0)


def test_quenching_rejects_what_one_barrier_cannot_do():
    with pytest.raises(ValueError):
        thermal.quantum_yield(350.0, 0.1, 0.05)  # a low barrier cannot lose 90 % over 125 K
    with pytest.raises(ValueError):
        thermal.relative_yield(300.0, 350.0, 0.9, 0.3)  # cooler than the anchor: yield above 1


def heated(lamp, warming_k: float):
    """The lamp with a Level 3 whose junction runs ``warming_k`` above Level 2's anchor."""
    anchor = lamp.physical["phosphor_temperature"].value
    drive = {"drive_current": (0.35, "A"), "forward_voltage": (3.0, "V"), "radiant_power": (0.35, "W"),
             "board_temperature": (anchor, "K"), "thermal_resistance": (warming_k / 0.70, "K/W")}
    physical = dict(lamp.physical)
    for name, (value, unit) in drive.items():
        physical[name] = Parameter(value, unit, empirical=True, reason="test operating point")
    return replace(lamp, physical=physical, models={**lamp.models, 3: ModelSpec(3, "led_thermal")})


def _binned(lamp, level):
    return lamp.generate(level).spectrum.bin(analysis_grid())


@pytest.mark.parametrize("preset", LEDS)
def test_level_3_is_level_2_at_its_anchor(preset):
    # Level 2 was fitted at its phosphor temperature; Level 3 shifts from there, so with the
    # junction at that temperature (and no other drive data, as for the CIE tables) it is
    # Level 2 exactly.
    lamp = heated(lamps.load(preset), 0.0)
    profile = lamp.generate(3)
    assert profile.metadata["junction_temperature_k"] == pytest.approx(lamp.physical["phosphor_temperature"].value)
    np.testing.assert_allclose(_binned(lamp, 3).values, _binned(lamp, 2).values, rtol=1e-12, atol=0)
    assert {"die.peak_shift", "garnet.quench_retained_423k"} <= set(profile.metadata["shared"])


def test_heating_moves_the_die_and_quenches_the_phosphors():
    lamp = heated(lamps.load("led_cie_b1"), 10.0)
    meta = lamp.generate(3).metadata
    assert meta["junction_temperature_k"] == pytest.approx(lamp.physical["phosphor_temperature"].value + 10.0)
    g, n = meta["relative_yields"]["garnet"], meta["relative_yields"]["nitride"]
    assert 0.98 < g < n < 1.0  # the garnet (90 % kept at 150 °C) quenches faster than the nitride (94 %)


def test_window_colour_ignores_light_outside_the_window():
    lamp = lamps.load("led_cie_b3")
    b = _binned(lamp, 2)
    outside = replace(b, values=np.where((b.grid.centres > 430) & (b.grid.centres < 715), b.values, 5 * b.values))
    np.testing.assert_allclose(led_drive.window_uv(outside), led_drive.window_uv(b), rtol=1e-12)


needs_data = pytest.mark.skipif(not led_drive.available(),
                                reason="measured LED data not downloaded (run `lss leddrive` once)")


@needs_data
def test_supplement_holds_two_leds_at_eight_currents():
    pytest.importorskip("openpyxl")
    for sample in led_drive.SAMPLES:
        trips = led_drive.readings(sample)
        assert [t[0].current_a for t in trips] == pytest.approx([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        for trip in trips:
            assert len({r.current_a for r in trip}) == 1
            assert all(r.wavelengths_nm[0] == 380 and r.wavelengths_nm[-1] == 780 for r in trip)
            assert all(r.power.max() == pytest.approx(100.0) for r in trip)
            # The instrument zeroes weak signal: nothing below 410 nm at any current.
            assert all(np.all(r.power[r.wavelengths_nm < 410] == 0) for r in trip)


@needs_data
def test_calibrated_findings_of_the_measured_led():
    # Recorded in the design doc (§4.2 L3); these hold in every colour window tried
    # (430-715, 445-690, 455-670 and 470-650 nm).
    pytest.importorskip("openpyxl")
    for sample in led_drive.SAMPLES:
        rows = led_drive.measured(sample)
        # Every triplicate warms (the voltage falls) and turns bluer (v' falls).
        assert all(r.drift_voltage_v < 0 and r.drift_uv[1] < 0 for r in rows)
        # From 150 to 500 mA v' falls too (by 1.3e-3 to 4e-3, depending on the window).
        assert rows[-1].uv[1] - rows[0].uv[1] < -1e-3
