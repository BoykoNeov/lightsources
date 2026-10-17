"""Sodium–xenon pairs in the nine street lamps (design §4.4, rung 2 third part): each lamp at
its fitted point with its own mercury, xenon at 10, 20 and 30 kPa with and without the pairs.
Skipped until the LICA data are downloaded; needs the mercury study's table (cached)."""

from dataclasses import replace

import numpy as np
import pytest

from lss.accuracy import lica
from lss.studies import street_hps, street_hps_fit as sf, street_hps_xenon as xe

needs_data = pytest.mark.skipif(not lica.available(),
                                reason="LICA lamp spectra not downloaded (run `lss streethps` once)")


@needs_data
def test_finding_the_double_count_is_bounded():
    """Set before running: eq. 5.7 read off the model's own spectrum, with xenon pairs over
    without, under 15 % for every lamp, or a targeted refit. At 10 and 20 kPa every lamp is
    under (largest 13 %, 2038 K, a lamp below eq. 5.7's range); at 30 kPa 2038 K moves 16 %."""
    for kpa in (10.0, 20.0):
        assert max(abs(xe.eq57_change(l, kpa)) for l in lica.HPS) < 0.15
    over = [l for l in lica.HPS if abs(xe.eq57_change(l, 30.0)) >= 0.15]
    assert over == ["hps-2038k-riba-roja-plaza"]


@needs_data
def test_the_targeted_refit_puts_the_mercury_back():
    """2038 K at 30 kPa with xenon pairs and its mercury lowered by the eq. 5.7 change: the
    model reads back the ratio it had without the pairs, and its red wing stays within 1 %
    of the fit's (20 kPa, no pairs): the xenon stands in for that much mercury, no more."""
    lamp = "hps-2038k-riba-roja-plaza"
    c = xe.column(lamp, 30.0, True)
    c = replace(c, fill=replace(c.fill, mercury_pa=c.fill.mercury_pa / (1.0 + xe.eq57_change(lamp, 30.0))))
    y = sf.observed(c, street_hps.response().kernel)
    x = street_hps.NODES_NM
    assert xe.eq57_of(x, y) == pytest.approx(xe.runs()[(lamp, 30.0, False)]["eq57"], rel=0.03)
    red = street_hps.features(x, y).red_wing
    assert red == pytest.approx(np.exp(xe.runs()[(lamp, 20.0, False)]["red wing"]), rel=0.01)


@needs_data
def test_finding_xenon_pairs_light_the_green_but_leave_the_red_alone():
    """At matched fill, 20 kPa: the red wing 1–5 % brighter, its slope steeper by at most 0.02
    (the lamps are 0.15–0.58 steeper), no 668–678 nm rise, 525–540 nm 1–5 % fainter, and the
    554–566 nm window 10–35 % brighter, which leaves every lamp still 1.4–3.3 times brighter
    there than the model."""
    r, m = xe.runs(), xe.measured()
    for lamp in lica.HPS:
        a, b = r[(lamp, 20.0, False)], r[(lamp, 20.0, True)]
        assert 0.005 < np.exp(b["red wing"] - a["red wing"]) - 1.0 < 0.06, lamp
        assert -0.025 < b["red slope"] - a["red slope"] <= 0.001, lamp
        assert abs(b["red bump"]) < 0.02, lamp
        assert -0.05 < np.exp(b["blue wing"] - a["blue wing"]) - 1.0 < 0.0, lamp
        assert 0.10 < b["green"] / a["green"] - 1.0 < 0.36, lamp
        assert 1.3 < m[lamp]["green"] / b["green"] < 3.4, lamp


@needs_data
def test_finding_high_xenon_would_close_the_green_and_take_a_share_of_the_mercury():
    """Added after the runs: at 160 kPa (a high-xenon lamp), temperature, sodium and mercury
    held, the 554-566 nm window comes to 0.6-1.25 of the lamps' (1.4-3.3 at 20 kPa); eq. 5.7 on
    the model moves by 14-39 %, so in such a lamp that much of eq. 5.7's mercury would be
    xenon; and still nothing for the red slope or the 668-678 nm rise."""
    hi, m = xe.runs(xe.HIGH_XENON_KPA), xe.measured()
    g = [m[l]["green"] / hi[(l, 160.0, True)]["green"] for l in lica.HPS]
    assert 0.6 < min(g) and max(g) < 1.3
    ch = [xe.eq57_change(l, 160.0) for l in lica.HPS]
    assert 0.13 < min(ch) and max(ch) < 0.40
    for l in ("hps-1732k-toledo", "hps-1819k-riba-roja-plaza", "hps-1869k-riba-roja",
              "hps-1895k-ager", "hps-1920k-alix", "hps-2005k-fisicas"):
        assert hi[(l, 160.0, True)]["red slope"] > m[l]["red slope"] + 0.1, l
        assert abs(hi[(l, 160.0, True)]["red bump"]) < 0.03, l
