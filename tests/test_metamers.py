"""Constructed metameric pairs (design §7, §12)."""

import json

import numpy as np
import pytest

from lss import cli
from lss.lamp import lamp as lamps
from lss.spectral.grids import analysis_grid
from lss.studies import appearance as ap
from lss.studies import metamers, samples


@pytest.fixture(scope="module")
def pair():
    return metamers.make()


def test_pair_matches_under_d65_through_the_evaluation_path(pair):
    d65 = metamers.d65()
    for observer in ("2", "10"):
        xyz, _ = ap.tristimulus(d65, pair.reflectances, observer)
        np.testing.assert_allclose(xyz[0], xyz[1], rtol=0, atol=1e-9)
    jab = ap.cam02ucs(d65, pair.reflectances)
    assert np.linalg.norm(jab[0] - jab[1]) < 1e-6


def test_partner_is_a_realisable_smooth_reflectance(pair):
    lo, hi = metamers.REFLECTANCE_LIMITS
    assert pair.second.min() >= lo - 1e-12 and pair.second.max() <= hi + 1e-12
    # The size rule takes all the room: one limit is reached.
    assert min(pair.second.min() - lo, hi - pair.second.max()) < 1e-9
    # Smooth: about as steep as the steepest ColorChecker patch, no steeper.
    steepest_patch = np.max(np.abs(np.diff(samples.get("colorchecker").values, axis=0)))
    assert np.max(np.abs(np.diff(pair.second))) < 1.1 * steepest_patch
    assert pair.amplitude > 0.05


@pytest.mark.parametrize("name, level", [("ref:cie-a", 0), ("ref:cie-hp1", 0), ("ref:cie-fl11", 0),
                                         ("led_cie_b3", 2)])
def test_pair_splits_away_from_daylight(pair, name, level):
    s = metamers.split(pair, ap.lamp_light(lamps.load(name), level))
    assert s.delta_e > 1.0
    assert s.srgb_first != s.srgb_second


def test_brighter_daylight_keeps_the_match(pair):
    # A light with D65's own spectrum, merely scaled, cannot split the pair.
    d65 = metamers.d65()
    assert metamers.split(pair, ap.Light("2 x D65", 2.0 * d65.power)).delta_e < 1e-6


def test_black_is_invisible_under_the_match_light_only():
    black = metamers.metameric_black(metamers.d65(), metamers.illuminant_a())
    assert np.sqrt(np.mean(black**2)) == pytest.approx(1.0)
    w65 = metamers._weights(metamers.d65(), "10")
    wa = metamers._weights(metamers.illuminant_a(), "10")
    assert np.max(np.abs(w65 @ black)) < 1e-8 * np.max(np.abs(wa @ black))


def test_base_without_room_is_refused(monkeypatch):
    # The black patch reflects about 0.03: with a floor of 0.05 it has no room to move down or up.
    monkeypatch.setattr(metamers, "REFLECTANCE_LIMITS", (0.05, 0.98))
    with pytest.raises(ValueError):
        metamers.make("black 2 (1.5 D)")


def test_cli_metamers(capsys, tmp_path):
    csv = tmp_path / "pair.csv"
    assert cli.main(["metamers", "ref:cie-hp1", "--json", "--csv", str(csv)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["constructed"] is True
    assert [s["light"] for s in doc["splits"]] == ["CIE D65", "cie-hp1 L0"]
    assert doc["splits"][0]["delta_e"] < 1e-6 < doc["splits"][1]["delta_e"]
    rows = csv.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("# metameric pair, CONSTRUCTED") and len(rows) == 3 + len(analysis_grid())
    assert cli.main(["metamers"]) == 0
    assert "CONSTRUCTED" in capsys.readouterr().out
