"""No-render studies (design §7, §12): reflectance samples and the colour chart."""

import json
import warnings

import numpy as np
import pytest

from lss import cli, lampsheet
from lss.lamp import lamp as lamps
from lss.spectral.grids import analysis_grid
from lss.studies import appearance as ap
from lss.studies import chart, samples

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import colour
    from colour.quality.cfi2017 import load_TCS_CIE2017


def light(name, level):
    return ap.lamp_light(lamps.load(name), level)


def test_sample_sets_load_with_their_ranges():
    lib = samples.library()
    assert len(lib["colorchecker"]) == 24 and lib["colorchecker"].measured_range == (380.0, 730.0)
    assert len(lib["cri"]) == 14 and lib["cri"].measured_range == (360.0, 830.0)
    assert len(lib["ces"]) == 99 and lib["ces"].measured_range == (380.0, 780.0)
    for s in lib.values():
        assert s.values.shape == (len(analysis_grid()), len(s))
        assert np.all((s.values >= 0.0) & (s.values <= 1.0))


def test_samples_sit_on_the_table_nodes_and_are_held_beyond():
    ces = samples.get("ces")
    table = load_TCS_CIE2017(colour.SpectralShape(380.0, 780.0, 1.0)).values
    centres = analysis_grid().centres
    inside = (centres >= 380.0) & (centres <= 780.0)
    np.testing.assert_array_equal(ces.values[inside], table)
    np.testing.assert_array_equal(ces.values[centres < 380.0], np.broadcast_to(table[0], (20, 99)))
    np.testing.assert_array_equal(ces.values[centres > 780.0], np.broadcast_to(table[-1], (50, 99)))


def test_planck_lamp_shifts_nothing():
    # Below 4500 K TM-30's reference is Planck at the CCT: a Planck lamp is its own reference.
    c = chart.make(light("incandescent_cie_a", 1), samples.get("colorchecker"))
    assert c.reference_cct_k == pytest.approx(2856.0, abs=2.0)
    assert max(p.delta_e for p in c.patches) < 0.02
    assert all(p.srgb_lamp == p.srgb_reference for p in c.patches)


@pytest.mark.parametrize("name, level", [("ref:cie-fl2", 0), ("ref:cie-hp1", 0), ("led_cie_b3", 2)])
def test_ces_chart_is_the_lamp_sheets_rf(name, level):
    # The chart and TM-30 share the reference and the colour space; with TM-30's own samples
    # they must give its Rf. The small gap is colour-science resampling the 5 nm tables.
    c = chart.make(light(name, level), samples.get("ces"))
    rf = lampsheet.make(lamps.load(name), level).rf
    assert c.fidelity == pytest.approx(rf, abs=0.1)


def test_sodium_desaturates_reds_and_shifts_most():
    hps = chart.make(light("ref:cie-hp1", 0), samples.get("colorchecker"))
    led = chart.make(light("led_cie_b3", 2), samples.get("colorchecker"))
    red = hps.patches[samples.get("colorchecker").index("red")]
    assert red.delta_c < -10.0  # HPS has almost no deep red to show a red surface with
    assert hps.mean_delta_e > 3.0 * led.mean_delta_e
    greys = [p for p in hps.patches if p.label.startswith(("neutral", "white", "black"))]
    assert all(p.delta_h_deg is None for p in greys)


def test_a_perfect_white_displays_as_white_under_any_light():
    flat = np.ones((len(analysis_grid()), 1))
    for name, level in [("ref:cie-hp1", 0), ("led_cie_b3", 2), ("incandescent_cie_a", 1)]:
        rgb, out = ap.srgb(light(name, level), flat)
        np.testing.assert_allclose(rgb[0], 1.0, atol=1e-3)  # D65 xy rounding
        assert not out[0]


def test_cli_chart(capsys, tmp_path):
    svg = tmp_path / "chart.svg"
    assert cli.main(["chart", "ref:cie-hp1", "--json", "--svg", str(svg)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["samples"] == "colorchecker" and len(doc["patches"]) == 24
    assert svg.read_text(encoding="utf-8").count("<rect") == 1 + 2 * 24
    assert cli.main(["chart", "led_cie_b3", "--samples", "cri"]) == 0
    assert "TCS14" in capsys.readouterr().out
