"""Each HPS street lamp's arc temperature and sodium pressure, fitted (design §4.4). Skipped
until the LICA data are downloaded; the first run builds the six arc tables (about half an hour
on 16 cores, cached)."""

import numpy as np
import pytest

from lss.accuracy import lica
from lss.studies import street_hps, street_hps_fit as sf

needs_data = pytest.mark.skipif(not lica.available(),
                                reason="LICA lamp spectra not downloaded (run `lss streethps` once)")

#: Halfway between the grid's nodes in both directions (geometric midpoints in pressure),
#: across the range the lamps' fits fall in.
OFF_GRID = tuple((t, float(np.sqrt(sf.GRID_NA_PA[i] * sf.GRID_NA_PA[i + 1])))
                 for t, i in ((3700.0, 5), (3900.0, 9), (4300.0, 13), (4500.0, 16)))


@needs_data
@pytest.mark.parametrize("t_k, na_pa", OFF_GRID)
def test_the_table_reads_like_a_direct_run_and_a_fit_recovers_it(t_k, na_pa):
    kernel = street_hps.response().kernel
    direct = street_hps.features(street_hps.NODES_NM, sf.observed(sf.at(sf.CENTRAL, t_k, na_pa), kernel))
    surf = sf.surface("central")
    read = surf(t_k, na_pa)
    want = sf._vector(direct)
    # Within 0.1 nm. (With the old cut, up to 0.3 nm at 21-24 kPa, where a hole in κ at the cut
    # made the blue peak jump; every fitted point is checked as well.)
    assert abs(read["blue peak"] - want["blue peak"]) < 0.1
    for k in sf.FIT_LINES:
        assert abs(read[k] - want[k]) < 0.02, k  # logarithms: 2 %
    f = sf.fit(surf, direct)
    assert abs(f.temperature_k - t_k) < 20.0
    assert abs(np.log(f.sodium_pa / na_pa)) < 0.03


@needs_data
def test_the_two_settings_move_different_targets():
    # Sodium moves the blue peak far more than temperature does; temperature moves the line powers.
    s = sf.surface("central")
    d_t = {k: s(4100.0, 15e3)[k] - s(3900.0, 15e3)[k] for k in ("blue peak", "568.8")}
    d_p = {k: s(4000.0, 16.5e3)[k] - s(4000.0, 13.6e3)[k] for k in ("blue peak", "568.8")}  # ±10 %
    assert abs(d_p["blue peak"]) > 3.0 * abs(d_t["blue peak"])
    assert abs(d_t["568.8"]) > 3.0 * abs(d_p["568.8"])


@needs_data
def test_every_fitted_point_reads_like_a_direct_run():
    fits, run = sf.fits(), sf.direct()
    for v in sf.VARIANTS:
        for lamp, f in fits[v].items():
            assert abs(f.model["blue peak"] - run[v][lamp]["blue peak"]) < 0.1, (v, lamp)
            for k in sf.FIT_LINES:
                assert abs(f.model[k] - run[v][lamp][k]) < 0.01, (v, lamp, k)


@needs_data
def test_the_d_band_red_half_does_not_bias_the_temperature_gauge():
    # The line powers are over the 575-605 nm power; the model's red peak sits elsewhere than
    # some lamps'. The share above 590 nm still agrees within 10 % at every fitted point.
    x = street_hps.NODES_NM
    d, r = (x >= 575) & (x <= 605), (x >= 575) & (x <= 605) & (x > 590)
    kernel = street_hps.response().kernel
    for lamp, f in sf.fits()["central"].items():
        y = sf.observed(sf.at(sf.CENTRAL, f.temperature_k, f.sodium_pa), kernel)
        m = lica.load(lamp).values
        assert abs((m[r].sum() / m[d].sum()) / (y[r].sum() / y[d].sum()) - 1.0) < 0.10, lamp


@needs_data
def test_the_lps_lamp_puts_the_offset_near_the_d_lines_slightly_blue():
    off, err = sf.lps_offset_nm()
    assert -0.3 < off < -0.05 and err < 0.08


@needs_data
def test_finding_temperatures_near_the_measured_range_pressures_inside_the_grid():
    """Recorded in the design doc (§4.4)."""
    fits = sf.fits()
    for v in sf.VARIANTS:
        for lamp, f in fits[v].items():
            assert not f.at_bound, (v, lamp)
            assert f.chi2_per_target < 1.0, (v, lamp)  # the three lines agree on one temperature
    central = fits["central"]
    t = [f.temperature_k for f in central.values()]
    assert 3900 < min(t) and max(t) < 4550  # the book: 4100-4300 K in standard 400 W lamps
    p = [f.sodium_pa for f in central.values()]
    assert 6.5e3 < min(p) and max(p) < 25e3


SHORT_SPACING = ("hps-1732k-toledo", "hps-1819k-riba-roja-plaza", "hps-1869k-riba-roja",
                 "hps-1895k-ager", "hps-1920k-alix")


@needs_data
def test_finding_the_model_keeps_the_books_rule_the_lamps_spread_wider():
    run, fits = sf.direct(), sf.fits()["central"]
    ratio = {lamp: f.sodium_pa / 1e3 * np.sqrt(sf.CENTRAL.radius_m * 1e3) / run["central"][lamp]["separation"]
             for lamp, f in fits.items()}
    # The book: 2.7 ± 0.5 spectroscopically, 1.9 ± 0.5 from metal-bath experiments. With the
    # smooth hand-over the model sits at the first's low end (with the 6 nm cut, 2.1-3.0).
    assert all(1.85 < q < 2.35 for q in ratio.values())
    # The book's rule from each lamp's own spacing gives more sodium than the fit: 1.5-1.9 times
    # for the five lamps whose peaks are further apart than the model's, 1.1-1.35 for the rest.
    q = {lamp: sf.book_rule_pa(f.measured["separation"]) / f.sodium_pa for lamp, f in fits.items()}
    assert sorted(lamp for lamp in q if q[lamp] > 1.45) == sorted(SHORT_SPACING)
    assert all(1.1 < v < 1.35 for lamp, v in q.items() if lamp not in SHORT_SPACING)


@needs_data
def test_finding_fitting_each_arc_does_not_close_the_red_wing():
    """Recorded in the design doc (§4.4). Held out: the fit never saw these."""
    run, meas = sf.direct(), street_hps.measured_features()
    red = {lamp: [meas[lamp].red_wing / np.exp(run[v][lamp]["red wing"]) for v in sf.VARIANTS] for lamp in meas}
    short = [lamp for lamp, q in red.items() if min(q) > 1.2]
    assert len(short) == 7  # brighter than the model in every table
    assert all(max(red[lamp]) < 1.15 for lamp in meas if lamp not in short)  # the other two: about as bright as the model
    # Five lamps' peaks are further apart than any table puts them at the fitted point; all five
    # are among the lamps whose red wing the model misses, by 1.7 times or more in every table.
    spacing = [lamp for lamp in meas
               if meas[lamp].separation_nm > max(run[v][lamp]["separation"] for v in sf.VARIANTS)]
    assert tuple(spacing) == SHORT_SPACING
    assert all(min(red[lamp]) > 1.6 for lamp in spacing)
    # The 668-678 nm rise is still missed everywhere.
    assert all(run[v][lamp]["red bump"] < 0.01 for v in sf.VARIANTS for lamp in meas)


@needs_data
def test_finding_four_lamps_fall_steeper_two_flatter_than_every_table():
    run, meas = sf.direct(), street_hps.measured_features()
    steeper = [l for l in meas if meas[l].red_slope < min(run[v][l]["red slope"] for v in sf.VARIANTS)]
    flatter = [l for l in meas if meas[l].red_slope > max(run[v][l]["red slope"] for v in sf.VARIANTS)]
    assert steeper == ["hps-1732k-toledo", "hps-1819k-riba-roja-plaza", "hps-1895k-ager", "hps-1920k-alix"]
    assert flatter == ["hps-2038k-riba-roja-plaza", "hps-2255k-paraninfo"]  # 2038 K by 0.01 only
