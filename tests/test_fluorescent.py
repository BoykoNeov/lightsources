"""Fluorescent Level 1: Hg lines plus the shared tri-phosphor library (design §4.3)."""

from dataclasses import replace

import numpy as np
import pytest

from lss.accuracy import report
from lss.accuracy.references import Reference
from lss.lamp import lamp as lamps
from lss.lamp import phosphors, quickpath
from lss.lamp.profile import EmissionProfile, Zone
from lss.params import Parameter
from lss.physics.discharge import mercury
from lss.spectral.bandpass import as_recorded

FL = ["fluorescent_cie_fl10", "fluorescent_cie_fl11", "fluorescent_cie_fl12"]
NODES = np.arange(380.0, 780.0 + 2.5, 5.0)


def test_visible_hg_lines_keep_their_measured_ratios():
    lines = mercury.visible_lines(2.0, 320.0)
    assert [round(ln.centre_nm, 4) for ln in lines] == [404.6565, 407.7837, 434.7506, 435.8335, 546.075, 576.961,
                                                        579.067]
    assert sum(ln.power for ln in lines) == pytest.approx(2.0, rel=1e-12)
    by = {round(ln.centre_nm): ln.power for ln in lines}
    assert by[405] / by[436] == pytest.approx(0.44, rel=1e-12)
    assert by[546] / by[436] == pytest.approx(1.0, rel=1e-12)


def test_doppler_width_of_hg_at_room_temperature():
    # Δλ/λ = sqrt(8 kT ln2 / m c²) = 7.16e-7 sqrt(T / M[u]).
    assert mercury.doppler_fwhm(435.8335, 320.0) == pytest.approx(435.8335 * 7.16e-7 * np.sqrt(320 / 200.592),
                                                                  rel=1e-3)


def test_library_main_bands_have_no_share():
    lib = phosphors.library()
    assert set(lib) == {"BAM", "LAP", "YOX"}
    params = phosphors.parameters()
    for ph in lib.values():
        assert f"{ph.id}.{ph.main}.share" not in params
        for band in ph.bands:
            if band.name != ph.main:
                assert f"{ph.id}.{band.name}.share" in params


def test_phosphor_power_is_split_by_shares_and_kept():
    bands = phosphors.emission("YOX", lambda n: phosphors.parameters()[n], 3.0)
    assert sum(b.power for b in bands) == pytest.approx(3.0, rel=1e-12)
    main = next(i for i, b in enumerate(phosphors.library()["YOX"].bands) if b.name == "7F2-611")
    assert bands[main].power == max(b.power for b in bands)


@pytest.mark.parametrize("preset", FL)
def test_generator_powers_add_up(preset):
    lamp = lamps.load(preset)
    spec = lamp.models[1]
    profile = lamp.generate(1)
    assert profile.includes == {"phosphor"}
    expected = 1.0 + sum(lamp.parameter(n, spec).value for n in ("hg_to_green", "blue_to_green", "red_to_green"))
    assert profile.spectrum.total() == pytest.approx(expected, rel=1e-9)
    shared = profile.metadata["shared"]
    assert shared and all(phosphors.is_shared(n) for n in shared)
    assert "hg_to_green" not in shared


def test_a_lamps_own_value_overrides_the_library():
    lamp = lamps.load("fluorescent_cie_fl11")
    name = "LAP.7F5.peak"
    assert lamp.parameter(name) is phosphors.parameters()[name]
    own = Parameter(545.5, "nm", empirical=True, reason="test", bounds=(538, 550))
    lamp = replace(lamp, physical={**lamp.physical, name: own})
    assert lamp.parameter(name) is own
    assert lamp.generate(1).metadata["params"][name]["value"] == 545.5


@pytest.mark.parametrize("preset", FL)
def test_presets_are_scored_through_their_tube(preset):
    row = report.score(lamps.load(preset), 1)
    assert row.comparable and row.reference_bandpass_nm == 5.0
    assert quickpath.apply(lamps.load(preset), lamps.load(preset).generate(1)).includes == {"phosphor", "envelope"}


@pytest.mark.parametrize("band", ["YOX.7F3", "LAP.7F3", "LAP.7F2", "LAP.7F1", "LAP.7F0"])
def test_weak_band_widths_barely_matter_at_5_nm(band):
    # A narrow band's apparent width grows with its width squared, so a strong line
    # (YOX 611 nm) shows its width in a 5 nm table and is fitted; for these weak bands
    # the fixed 2 nm is a stated assumption that 1 or 3 nm would hardly change.
    lamp = lamps.load("fluorescent_cie_fl11")
    name = f"{band}.fwhm"
    assert phosphors.parameters()[name].bounds is None

    def model(fwhm):
        p = replace(phosphors.parameters()[name], value=fwhm)
        return replace(lamp, physical={**lamp.physical, name: p}).generate(1).spectrum

    ref = Reference(id="t", family="fluorescent", description="model at 2 nm, as a 5 nm table would record it",
                    spectrum=as_recorded(model(2.0), NODES), valid_range=(380.0, 780.0), sampling_nm=5.0,
                    kind="definition", uncertainty="exact", includes=frozenset({"phosphor"}), citation="test",
                    data_version="test", bandpass_nm=5.0)
    for fwhm in (1.0, 3.0):
        c = report.compare(EmissionProfile(zones=(Zone(model(fwhm)),)), ref)
        assert np.sqrt(np.mean(c.residuals**2)) < 5e-3


@pytest.mark.parametrize("preset", FL)
def test_calibrated_tubes_match_their_tables_in_sample(preset):
    # In-sample: the shared shapes were fitted to these three tables (leave-one-out
    # scores are in the design doc, §4.3).
    row = report.score(lamps.load(preset), 1)
    assert row.spectral_nrms < 0.08
    assert row.delta_uv_prime < 2e-3
    assert abs(row.delta_ra) < 1.0
    assert row.note.startswith("in-sample")
    assert row.empirical_shared > 30 and row.empirical_params == row.empirical_shared + 3


def test_shared_shape_findings_stay_at_their_bounds():
    # Values at a bound are findings, recorded in the library's notes: if a change frees
    # them, the notes and the design doc need revisiting.
    p = phosphors.parameters()
    assert p["YOX.7F1.fwhm"].value == p["YOX.7F1.fwhm"].bounds[1]
    assert p["YOX.7F2-611.fwhm"].value == pytest.approx(p["YOX.7F2-611.fwhm"].bounds[1], rel=0.01)
