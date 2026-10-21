"""Czuchaj et al.'s tabulated Na–Hg curves in the nine street lamps (design §4.4, "Czuchaj et
al.'s tabulated Na–Hg curves"): each lamp at its own-mercury fit, Windholz's curves against
Czuchaj's with the table's dipoles and constant ones. Skipped until the LICA data are
downloaded; needs the mercury study's table (cached)."""

from dataclasses import replace

import numpy as np
import pytest

from lss.accuracy import lica
from lss.studies import street_hps_czuchaj as sc, street_hps_fit as sf, street_hps_xenon as xe

needs_data = pytest.mark.skipif(not lica.available(),
                                reason="LICA lamp spectra not downloaded (run `lss streethps` once)")
WINDHOLZ, TABLE, CONSTANT = sc.SETS


def test_the_default_curves_keep_the_cached_tables():
    """The new field at its default is left out of the cache key, so no table is rebuilt."""
    assert "curves=" not in sf.column_key(sf.CENTRAL)
    other = replace(sf.CENTRAL, mercury_pairs=replace(sf.CENTRAL.mercury_pairs, curves="czuchaj"))
    assert "curves='czuchaj'" in sf.column_key(other)


@needs_data
def test_finding_czuchajs_curves_make_the_rise():
    """Predicted before the runs: +0.40 to +0.70 (table dipoles), +0.25 to +0.41 (constant).
    Found: +0.29 to +0.67 and +0.18 to +0.40 (2176 K under in both); Windholz's 0.00–0.01.
    Toledo, the strongest-rising lamp, within 0.01 with constant dipoles."""
    r = sc.runs()
    rise = {s: [r[(l, *s)]["red bump"] for l in lica.HPS] for s in sc.SETS}
    assert max(abs(v) for v in rise[WINDHOLZ]) < 0.01
    assert 0.28 < min(rise[TABLE]) and max(rise[TABLE]) < 0.70
    assert 0.17 < min(rise[CONSTANT]) and max(rise[CONSTANT]) < 0.41
    toledo = "hps-1732k-toledo"
    assert r[(toledo, *CONSTANT)]["red bump"] == pytest.approx(xe.measured()[toledo]["red bump"], abs=0.02)


@needs_data
def test_finding_the_red_wing_slope_and_616_line():
    """The red wing's level 1.12–1.31 times brighter (lamp/model 0.87–0.88); the slope steeper,
    0.22–0.48 with constant dipoles, 0.63–1.86 with the table's; the blue peak unmoved; the
    616 nm line usable again (lamp/model 0.93–1.16), where Windholz's curves left it at or
    below zero for seven lamps. The red peak out by 0.18–1.45 nm (predicted in by 0–0.5: missed)."""
    r, m = sc.runs(), xe.measured()
    for s in (TABLE, CONSTANT):
        gain = [np.exp(r[(l, *s)]["red wing"] - r[(l, *WINDHOLZ)]["red wing"]) for l in lica.HPS]
        assert 1.10 < min(gain) and max(gain) < 1.33
        assert sc.level_ratio(s)[0] == pytest.approx(0.875, abs=0.02)
        assert max(abs(r[(l, *s)]["blue peak"] - r[(l, *WINDHOLZ)]["blue peak"]) for l in lica.HPS) < 0.0005
        ratio = [np.exp(m[l]["616"] - r[(l, *s)]["616"]) for l in lica.HPS]
        assert 0.91 < min(ratio) and max(ratio) < 1.18
        out = [r[(l, *s)]["separation"] - r[(l, *WINDHOLZ)]["separation"] for l in lica.HPS]
        assert 0.15 < min(out) and max(out) < 1.5
    steeper = [r[(l, *WINDHOLZ)]["red slope"] - r[(l, *CONSTANT)]["red slope"] for l in lica.HPS]
    assert 0.20 < min(steeper) and max(steeper) < 0.50
    steeper = [r[(l, *WINDHOLZ)]["red slope"] - r[(l, *TABLE)]["red slope"] for l in lica.HPS]
    assert 0.60 < min(steeper) and max(steeper) < 1.90
    assert sum(np.isnan(r[(l, *WINDHOLZ)]["616"]) for l in lica.HPS) == 7
