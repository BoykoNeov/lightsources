"""LICA's lighter flame against a grey body and soot (design §4.5, first piece). Skipped
until the file is downloaded (`lss flame`)."""

import pytest

from lss.accuracy import lica
from lss.studies import flame

needs_data = pytest.mark.skipif(not lica.available(["lighter"]),
                                reason="LICA lighter flame not downloaded (run `lss flame` once)")


def _by(floor_subtracted):
    return {f.model: f for f in flame.fits() if f.floor_subtracted == floor_subtracted}


@needs_data
def test_the_file_is_the_stated_propane_flame_on_the_same_instrument():
    m = lica.load("lighter")
    assert m.header["Other data"] == "Propane lighter"
    assert "JAZA2310" in m.header["Correct for Electrical Dark"]
    assert flame.measured(True).floor == pytest.approx(0.0047, abs=0.0002)


@needs_data
@pytest.mark.parametrize("floor_subtracted", [False, True])
def test_temperatures_order_and_stay_under_the_flame_ceiling(floor_subtracted):
    f = _by(floor_subtracted)
    grey, flat, chang, hb = (f[k].temperature_k for k in (
        "grey body (L1)", "soot, thin, flat", "soot, thin, chang1990", "soot, thin, hottel_broughton"))
    assert grey > flat > chang > hb  # α = 0 (grey), 1, 1.28, 1.39
    assert 50.0 < flat - hb < 90.0  # the optics' spread, the temperature's uncertainty
    assert 1800.0 < chang < 2000.0 and 2000.0 < grey < 2200.0
    assert max(x.temperature_k for x in f.values()) < flame.ADIABATIC_PROPANE_K
    # Optical depth moves soot towards the grey body, but slowly: a third of the way at τ = 1.
    gap = grey - chang
    steps = [f[f"soot, depth {t:g}, chang1990"].temperature_k - chang for t in flame.DEPTHS]
    assert 0 < steps[0] < steps[1] < steps[2] < 0.4 * gap


@needs_data
def test_subtracting_the_floor_lowers_every_temperature():
    raw, sub = _by(False), _by(True)
    for name in raw:
        assert 10.0 < raw[name].temperature_k - sub[name].temperature_k < 60.0


@needs_data
def test_the_shape_does_not_pick_a_model_and_misses_by_a_few_per_cent():
    # Floor subtracted, every fit is within 2.5 % rms and within 1.3x of the others; as
    # measured the floor decides (the grey body best, Hottel-Broughton twice its rms).
    sub = [f.rms for f in _by(True).values()]
    assert max(sub) < 0.025 and max(sub) / min(sub) < 1.3
    raw = _by(False)
    assert min(raw.values(), key=lambda f: f.rms).model == "grey body (L1)"
    # Per 50 nm the residual reaches 2-5 %, beyond the 0.6 % LICA's correction showed on the
    # halogen lamp (design §15): one temperature does not fit the flame's shape.
    worst = max(abs(v) for f in raw.values() for v in f.mean_residual)
    assert worst > 0.02
