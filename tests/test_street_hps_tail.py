"""The D1 band's long-range Na–Hg tail in the nine street lamps (design §4.4, "The I½ tail,
bracketed"): each lamp at its own-mercury fit, ΔC6 × 1 and × 2, the fitted tail against the
dispersion tail. Skipped until the LICA data are downloaded; needs the mercury study's table
(cached)."""

from dataclasses import replace

import numpy as np
import pytest

from lss.accuracy import lica
from lss.studies import street_hps_fit as sf, street_hps_tail as st

needs_data = pytest.mark.skipif(not lica.available(),
                                reason="LICA lamp spectra not downloaded (run `lss streethps` once)")


def test_the_default_tail_keeps_the_cached_tables():
    """The new field at its default is left out of the cache key, so no table is rebuilt."""
    assert "tail=" not in sf.column_key(sf.CENTRAL)
    other = replace(sf.CENTRAL, mercury_pairs=replace(sf.CENTRAL.mercury_pairs, tail="dispersion"))
    assert "tail='dispersion'" in sf.column_key(other)


@needs_data
def test_finding_the_tail_is_a_bounded_uncertainty_on_the_mercury():
    """Set before running: eq. 5.7 on the model, dispersion over fitted tail, under 15 % for
    every lamp. With Unsöld's ΔC6 it reads 2–8 % less mercury (predicted 5–15 %: smaller than
    predicted), with twice it 6–15 % more (predicted 5–20 %); the largest, 1920 K, 14.9 %."""
    ch1 = [st.eq57_change(l, 1.0) for l in lica.HPS]
    ch2 = [st.eq57_change(l, 2.0) for l in lica.HPS]
    assert -0.09 < min(ch1) and max(ch1) < -0.015
    assert 0.05 < min(ch2) and max(ch2) < st.BOUND


@needs_data
def test_finding_the_tail_moves_the_red_peak_only():
    """The red peak moves in by 0.03–0.27 nm (ΔC6 × 1) or out by 0.21–0.60 nm (× 2); the blue
    peak, the red wing's slope and the 668–678 nm rise do not move, the red wing by under 1 %."""
    r = st.runs()
    for f, lo, hi in ((1.0, -0.30, -0.02), (2.0, 0.19, 0.62)):
        for lamp in lica.HPS:
            a, b = r[(lamp, f, "fitted")], r[(lamp, f, "dispersion")]
            assert lo < b["separation"] - a["separation"] < hi, (lamp, f)
            assert abs(b["blue peak"] - a["blue peak"]) < 0.005, (lamp, f)
            assert abs(np.exp(b["red wing"] - a["red wing"]) - 1.0) < 0.01, (lamp, f)
            assert abs(b["red slope"] - a["red slope"]) < 0.001, (lamp, f)
            assert abs(b["red bump"] - a["red bump"]) < 0.001, (lamp, f)
