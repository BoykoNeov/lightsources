"""Each HPS street lamp's mercury from its own peaks, the arc refitted with it (design §4.4).
Skipped until the LICA data are downloaded; the first run builds the arc table with mercury
(about 40 minutes on 16 cores, cached)."""

import numpy as np
import pytest

from lss.accuracy import lica
from lss.studies import street_hps, street_hps_fit as sf, street_hps_mercury as hg

needs_data = pytest.mark.skipif(not lica.available(),
                                reason="LICA lamp spectra not downloaded (run `lss streethps` once)")


def test_eq57_is_zero_for_symmetric_peaks_and_matches_a_hand_value():
    c = hg.D_CENTRES_NM[0]
    assert hg.eq57(c - 5.0, c + 5.0) == pytest.approx(0.0, abs=1e-12)
    # Δλ_B 3 nm, Δλ_R 9 nm: (3² − 1) / (0.057 · (3 − 0.85)) = 65.28
    assert hg.eq57(c - 3.0, c + 9.0) == pytest.approx(8.0 / (0.057 * 2.15))


@needs_data
def test_the_instrument_moves_both_peaks_red_the_red_one_further():
    for lamp in lica.HPS:
        m = lica.load(lamp)
        b, r = hg.blur_shift_nm(m.wavelengths_nm, m.values)
        assert 0.2 < b < 0.5 and 0.5 < r < 1.1 and r > b, lamp


#: Between the table's nodes in every direction, across the range the lamps' fits fall in.
OFF_GRID = ((3900.0, 9, 2), (4300.0, 13, 4), (4100.0, 6, 5))


@needs_data
@pytest.mark.parametrize("t_k, i, j", OFF_GRID)
def test_the_table_reads_like_a_direct_run_and_a_fit_recovers_it(t_k, i, j):
    na = float(np.sqrt(sf.GRID_NA_PA[i] * sf.GRID_NA_PA[i + 1]))
    r = float(np.sqrt(hg.GRID_RATIO[j] * hg.GRID_RATIO[j + 1]))
    kernel = street_hps.response().kernel
    direct = street_hps.features(street_hps.NODES_NM, sf.observed(hg.at(sf.CENTRAL, t_k, na, r), kernel))
    surf = hg.surface(r)
    read, want = surf(t_k, na), sf._vector(direct)
    assert abs(read["blue peak"] - want["blue peak"]) < 0.15
    for k in sf.FIT_LINES:
        assert abs(read[k] - want[k]) < 0.02, k  # logarithms: 2 %
    f = sf.fit(surf, direct)
    assert abs(f.temperature_k - t_k) < 25.0
    assert abs(np.log(f.sodium_pa / na)) < 0.04


@needs_data
def test_every_fitted_point_reads_like_a_direct_run():
    fits, run = hg.fits(), hg.direct()
    for lamp in fits:
        for e in hg.ALL_ENDS:
            assert abs(fits[lamp][e].model["blue peak"] - run[lamp][e]["blue peak"]) < 0.15, (lamp, e)
            for k in sf.FIT_LINES:
                assert abs(fits[lamp][e].model[k] - run[lamp][e][k]) < 0.02, (lamp, e, k)


# What the run found (design §4.4). The red wing's level is the test; the lamps' order is not.

@needs_data
def test_with_the_lamps_own_mercury_the_red_wing_comes_to_the_lamps_level():
    assert hg.level_70() > 1.4  # the arc fit, mercury held at 70 kPa: the model too faint
    assert 0.85 < hg.level("central") < 1.15
    assert 0.85 < hg.level("central", blue_half=True) < 1.15
    # One shared ratio (the median) is nearly as close in level, much worse in scatter.
    assert 0.65 < hg.level("common") < 0.9
    assert hg.scatter("central") < 0.3 < 0.5 < hg.scatter("common")


@needs_data
def test_the_lamp_at_2038_k_overshoots_across_its_whole_band():
    lamp = "hps-2038k-riba-roja-plaza"
    assert not hg.ratios()[lamp].valid  # below 10: outside the book's claim
    assert max(hg.red_ratio(lamp, e) for e in hg.ENDS) < 1.0


@needs_data
def test_the_models_second_red_maximum_is_the_splice():
    # Toledo at its fitted point and own mercury: the outer red maximum follows the splice.
    lamp = "hps-1732k-toledo"
    f, r = hg.fits()[lamp]["central"], hg.ratio_at(lamp, "central")
    kernel = street_hps.response().kernel
    mx = {s: hg.red_maxima_nm(street_hps.NODES_NM, sf.observed(
        hg.at(sf._with_splice(sf.CENTRAL, s), f.temperature_k, f.sodium_pa, r), kernel)) for s in (3.0, 6.0, 12.0)}
    assert len(mx[3.0]) == 1 and 595 <= mx[3.0][0] <= 597
    assert len(mx[6.0]) == 2 and 592 <= mx[6.0][0] <= 594 and 597 <= mx[6.0][1] <= 599
    assert len(mx[12.0]) == 2 and mx[12.0][1] > 602
    m = lica.load(lamp)
    assert len(hg.red_maxima_nm(m.wavelengths_nm, m.values)) == 1


@needs_data
def test_most_of_the_temperature_rise_is_the_red_side_not_the_mercury():
    base, bh = hg.fits(), hg.blue_half_fits()
    whole = [base[l]["central"].temperature_k - sf.fits()["central"][l].temperature_k for l in base]
    blue = [bh[l]["central"].temperature_k - bh[l]["70 kPa"].temperature_k for l in bh]
    assert max(whole) > 250.0
    assert max(abs(d) for d in blue) < 100.0


@needs_data
def test_what_mercury_does_not_fix():
    run, base = hg.direct(), hg.fits()
    for lamp in base:
        assert abs(run[lamp]["central"]["red bump"]) < 0.02, lamp  # six lamps: +0.16 to +0.41
    for lamp in ("hps-1732k-toledo", "hps-1819k-riba-roja-plaza", "hps-1869k-riba-roja",
                 "hps-1895k-ager", "hps-1920k-alix", "hps-2005k-fisicas"):
        assert run[lamp]["central"]["red slope"] > base[lamp]["central"].measured["red slope"] + 0.1, lamp
