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
