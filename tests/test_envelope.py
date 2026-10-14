"""Glass envelopes on the quick path (design §5)."""

import numpy as np
import pytest

from lss import lampsheet
from lss.accuracy import fit, report
from lss.accuracy.references import Reference
from lss.lamp import lamp as lamps
from lss.lamp import quickpath
from lss.lamp.envelope import Envelope, glass
from lss.params import Parameter

GLASS = "soda-lime-rubin-clear"


def _envelope(mm=0.8):
    return Envelope(GLASS, Parameter(mm, "mm", source="test"))


def _planck(envelope=None, t=4000.0):
    doc = {"id": "t", "family": "fluorescent", "reference": None,
           "physical": {"filament_temperature": {"value": t, "unit": "K", "source": "test",
                                                 "bounds": [3000, 5000]}},
           "models": {1: {"generator": "planck"}}}
    if envelope is not None:
        doc["envelope"] = envelope.as_dict()
    return lamps.lamp_from_dict(doc)


def _reference(spectrum, includes=frozenset({"envelope"})):
    return Reference(id="synthetic", family="fluorescent", description="Planck through glass", spectrum=spectrum,
                     valid_range=(380.0, 780.0), sampling_nm=1.0, kind="definition", uncertainty="exact",
                     includes=includes, citation="test", data_version="test")


def test_glass_index_is_that_of_soda_lime():
    assert glass(GLASS).n(589.3) == pytest.approx(1.523, abs=0.002)


def test_wall_transmittance_follows_beer_lambert_from_k():
    t = _envelope(0.8).transmission()
    # k(600 nm) = 4.548e-7 in Rubin's table.
    assert t(600.0) == pytest.approx(np.exp(-4 * np.pi * 4.548e-7 * 0.8e6 / 600.0), rel=1e-12)
    assert 0.99 < t(600.0) < 0.995
    # Nearly clear across the visible, but iron absorption (Fe2+, centred near 1 µm)
    # takes ~3 % at the red end; cutting into the UV, opaque beyond the table.
    lam = np.arange(380.0, 781.0)
    assert 0.97 < t(lam).min() == pytest.approx(t(780.0))
    assert t(500.0) > 0.99
    assert t(310.0) < 0.25 and t(320.0) < 0.7 < 0.9 < t(340.0)
    assert t(300.0) == 0.0 and t(5000.0) == 0.0


def test_quick_path_applies_the_envelope_once():
    lamp = _planck(_envelope())
    bare = lamp.generate(1)
    once = quickpath.apply(lamp, bare)
    assert "envelope" not in bare.includes and "envelope" in once.includes
    assert quickpath.apply(lamp, once) is once
    assert once.metadata["quick_path"]["envelope"]["material"] == GLASS
    ratio = once.spectrum.integral(599.0, 601.0) / bare.spectrum.integral(599.0, 601.0)
    assert ratio == pytest.approx(lamp.envelope.transmission()(600.0), rel=1e-4)


def test_a_lamp_without_its_envelope_is_not_scored_and_with_it_matches_exactly():
    lamp = _planck(_envelope())
    ref = _reference(quickpath.apply(lamp, lamp.generate(1)).spectrum)
    row = report.score(lamp, 1, ref)
    assert row.comparable
    assert row.spectral_nrms == pytest.approx(0.0, abs=1e-12)

    bare = _planck()
    assert not report.score(bare, 1, ref).comparable
    with pytest.raises(ValueError, match="envelope"):
        fit.fit(bare, 1, ref)


def test_fit_goes_through_the_envelope_too():
    truth = _planck(_envelope(), t=4321.0)
    ref = _reference(quickpath.apply(truth, truth.generate(1)).spectrum)
    result = fit.fit(_planck(_envelope()), 1, ref)
    assert result.parameters[0].value == pytest.approx(4321.0, rel=1e-6)


def test_the_glass_shifts_colour_and_its_thickness_only_a_little():
    # Through 0.8 mm of glass vs none: Δu'v' ~4e-4 (the red-end loss), as large as a
    # good level-1 fit's error, so the envelope matters. The wall's 0.5-1.1 mm range
    # moves it only ~1.6e-4.
    lamp = _planck(_envelope(0.8))
    ref = _reference(quickpath.apply(lamp, lamp.generate(1)).spectrum)
    assert report.score(_planck(_envelope(1e-4)), 1, ref).delta_uv_prime > 3e-4
    for mm in (0.5, 1.1):
        row = report.score(_planck(_envelope(mm)), 1, ref)
        assert row.spectral_nrms < 5e-3
        assert row.delta_uv_prime < 2.5e-4


def test_reference_spectra_already_include_their_envelope():
    lamp = lamps.lamp_from_dict({"id": "t", "family": "fluorescent", "reference": "cie-fl11",
                                 "models": {0: {"generator": "reference"}}, "envelope": _envelope().as_dict()})
    profile = lamp.generate(0)
    assert "envelope" in profile.includes
    assert quickpath.apply(lamp, profile) is profile


def test_lamp_sheet_reports_the_envelope():
    sheet = lampsheet.make(_planck(_envelope()), 1)
    assert "envelope" in sheet.includes
    assert "soda-lime-rubin-clear, wall 0.8 mm" in lampsheet.format_text(sheet)
