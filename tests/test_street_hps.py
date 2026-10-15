"""HPS street lamps (LICA-UCM) against the rung-1 sodium arc, through the instrument's
response (design §4.4, §15). Skipped until the data are downloaded (`lss streethps`)."""

import numpy as np
import pytest

from lss.accuracy import lica
from lss.studies import street_hps

needs_data = pytest.mark.skipif(not lica.available(),
                                reason="LICA lamp spectra not downloaded (run `lss streethps` once)")


@needs_data
def test_every_file_is_the_same_instrument_at_1_nm():
    for key in lica.FILES:
        m = lica.load(key)
        assert "JAZA2310" in m.header["Correct for Electrical Dark"]
        assert m.header["Correct for Flux"] == "Yes"
        assert m.wavelengths_nm[0] == 300 and m.wavelengths_nm[-1] == 900
        assert np.all(np.diff(m.wavelengths_nm) == 1.0)


@needs_data
def test_the_response_is_fitted_to_narrow_lines():
    hg = street_hps.response("hg-pencil")
    assert 1.55 < hg.kernel.fwhm_nm < 1.75 and hg.rms_over_peak < 0.025
    assert 0.25 < hg.kernel.mode_nm < 0.4  # lines read about 0.3 nm to the red near 400-550 nm
    ar = street_hps.response("thorium-argon")  # another lamp, near 750 nm
    assert 1.3 < ar.kernel.fwhm_nm < 1.5 and ar.rms_over_peak < 0.01
    assert abs(ar.kernel.mode_nm - hg.kernel.mode_nm) > 0.3  # the offset is not the same there


@needs_data
def test_the_mercury_response_fits_the_low_pressure_sodium_lines():
    # With the response held fixed, only each line's power and a straight background free.
    check = street_hps.lps_check()
    assert check["589 (D)"] < 0.03 and check["568.8"] < 0.02
    assert check["819"] < 0.09  # worse: the response near 800 nm differs from the mercury lines'


@needs_data
def test_lica_response_correction_leaves_a_halogen_lamp_smooth():
    # A smooth tilt would be absorbed by the temperature; anything else would show here.
    fc = street_hps.flux_check()
    assert max(abs(v) for v in fc.mean_relative_residual) < 0.01
    assert 2700 < fc.temperature_k < 3100


@needs_data
def test_finding_the_comparisons_do_not_depend_on_the_response():
    variants = [street_hps.model_features(k) for k in street_hps.response_variants().values()]
    sep = [f.separation_nm for f in variants]
    assert max(sep) - min(sep) < 0.2
    for band in street_hps.BANDS_NM:
        p = [f.line_power[band] for f in variants]
        assert max(p) / min(p) < 1.10
    red = [f.red_wing for f in variants]
    assert max(red) / min(red) < 1.1


@needs_data
def test_finding_lines_and_peaks_fall_among_the_lamps_but_the_red_wing_is_too_faint():
    """Recorded in the design doc (§4.4). The arc is the assumed one, nothing fitted."""
    model = street_hps.model_features(street_hps.response().kernel)
    lamps = list(street_hps.measured_features().values())
    # Every subordinate line's power over the D lines' lies within the nine lamps' range.
    for band in street_hps.BANDS_NM:
        measured = [f.line_power[band] for f in lamps]
        assert min(measured) <= model.line_power[band] <= max(measured), band
    sep = [f.separation_nm for f in lamps]
    assert min(sep) <= model.separation_nm <= max(sep)
    # The far blue wing agrees with most lamps ...
    blue = sorted(f.blue_wing for f in lamps)
    assert blue[1] <= model.blue_wing <= blue[-3]
    # ... but every lamp's red wing is at least three times the model's,
    assert min(f.red_wing for f in lamps) > 3.3 * model.red_wing
    # even the lamp whose peaks are closer than the model's (so not simply less sodium in the model).
    narrow = [f for f in lamps if f.separation_nm < model.separation_nm]
    assert narrow and all(f.red_wing > 3.0 * model.red_wing and f.blue_wing < model.blue_wing for f in narrow)


@needs_data
def test_finding_sodium_pairs_double_the_red_wing_but_do_not_close_it():
    """Recorded in the design doc (§4.4). The assumed arc with sodium-pair far wings, over
    both dipole forms and splices from 3 to 12 nm (the ends of each range are tested here)."""
    k = street_hps.response().kernel
    rung1 = street_hps.model_features(k)
    lamps = list(street_hps.measured_features().values())
    names = ("pairs, long-range dipole, splice 3 nm", "pairs, long-range dipole, splice 12 nm",
             "pairs, constant dipole, splice 6 nm")
    variants = [street_hps.model_features(k, street_hps.PAIR_VARIANTS[n]) for n in names]
    for f in variants:
        assert 1.9 < f.red_wing / rung1.red_wing < 2.5  # the red wing doubles ...
        assert min(m.red_wing for m in lamps) > 1.35 * f.red_wing  # ... and every lamp is still brighter
        assert 1.15 < f.blue_wing / rung1.blue_wing < 1.4
        blue = sorted(m.blue_wing for m in lamps)
        assert blue[1] <= f.blue_wing <= blue[-3]  # the blue wing stays among most lamps
        for band in street_hps.BANDS_NM:
            measured = [m.line_power[band] for m in lamps]
            assert min(measured) <= f.line_power[band] <= max(measured), band
    # The peak spacing depends on the splice: near the peaks it widens.
    assert variants[0].separation_nm > variants[1].separation_nm + 1.0
    sep = [m.separation_nm for m in lamps]
    assert all(min(sep) <= f.separation_nm <= max(sep) for f in variants)


@needs_data
def test_finding_mercury_pairs_bring_the_red_wing_among_the_lamps_but_not_the_bump():
    """Recorded in the design doc (§4.4). The assumed arc with sodium and sodium–mercury pair
    wings, over both dipole forms of each and both ends of the ΔC₆ range (the four corners for
    the mercury pairs with constant-dipole sodium pairs are tested here)."""
    k = street_hps.response().kernel
    lamps = list(street_hps.measured_features().values())
    names = [f"Hg pairs, {dip}, dC6 x{f}; Na pairs, constant" for dip in ("constant", "falling") for f in ("1", "2")]
    variants = [street_hps.model_features(k, street_hps.MERCURY_VARIANTS[n]) for n in names]
    red = sorted(m.red_wing for m in lamps)
    for f in variants:
        # The red wing is now among the lamps', at the faint end.
        assert red[0] <= f.red_wing <= red[2]
        # Its slope is the lamps' again (sodium pairs alone flattened it to about −1).
        assert -2.2 < f.red_slope < -1.4
        # The blue wing stays among most lamps.
        blue = sorted(m.blue_wing for m in lamps)
        assert blue[1] <= f.blue_wing <= blue[-3]
        # Held out: six lamps rise 16–41 % at 668–678 nm; the model does not rise.
        assert f.red_bump < 0.05
        for band in ("498", "515", "568.8", "819"):
            measured = [m.line_power[band] for m in lamps]
            assert min(measured) <= f.line_power[band] <= max(measured), band
    assert sum(m.red_bump > 0.15 for m in lamps) == 6
    # The 616 nm line's measure is not usable with these wings: the pair continuum's structure
    # from the avoided crossing falls on its baseline windows, and where depends on the shift.
    p616 = [f.line_power["616"] for f in variants]
    assert min(p616) < 0.01 and max(p616) > 0.035
