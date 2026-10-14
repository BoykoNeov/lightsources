"""LED Level 1: energy-shaped bands and the two-band generator (design §4.2)."""

import numpy as np
import pytest
from scipy.optimize import brentq

from lss.accuracy import fit, report
from lss.accuracy.references import Reference
from lss.lamp import lamp as lamps
from lss.physics.led.bands import EnergyBand
from lss.spectral.grids import BinGrid
from lss.spectral.spectrum import Spectrum

PUMP = EnergyBand.symmetric(450.0, 20.0)
PHOSPHOR = EnergyBand.split(570.0, 50.0, 75.0, power=3.0)


@pytest.mark.parametrize("band", [PUMP, PHOSPHOR], ids=["pump", "phosphor"])
@pytest.mark.parametrize("width_nm", [0.1, 1.0, 5.0, 37.0])
def test_bins_account_for_the_whole_band(band, width_nm):
    s = Spectrum.of(band)
    edges = np.arange(400.0, 700.0 + width_nm / 2, width_nm)
    b = s.bin(BinGrid(edges))
    assert b.total == pytest.approx(s.total(), rel=1e-12)
    assert s.total() == pytest.approx(band.power, rel=1e-12)


def test_very_wide_band_is_cut_short_of_zero_energy_and_still_accounted_for():
    band = EnergyBand.split(650.0, 60.0, 400.0)
    assert band.support[1] == pytest.approx(1e4)
    s = Spectrum.of(band)
    assert s.bin(BinGrid(np.array([300.0, 800.0, 2000.0]))).total == pytest.approx(s.total(), rel=1e-10)
    assert s.total() < band.power


def _per_energy(band, lam):
    """Per-wavenumber density: the per-nm density undone by its λ⁻² Jacobian."""
    return band.density(lam) * lam**2 / 1e7


def test_width_convention_is_the_per_energy_half_maximum():
    # The two half-maximum points of the energy shape are 20 nm apart.
    peak = _per_energy(PUMP, PUMP.peak_nm)
    f = lambda lam: _per_energy(PUMP, lam) - peak / 2  # noqa: E731
    short, long_ = brentq(f, 400.0, 450.0), brentq(f, 450.0, 520.0)
    assert long_ - short == pytest.approx(20.0, abs=1e-9)
    # Symmetric in energy is lopsided in wavelength: the red half is wider.
    assert long_ - 450.0 > 450.0 - short

    peak = _per_energy(PHOSPHOR, 570.0)
    f = lambda lam: _per_energy(PHOSPHOR, lam) - peak / 2  # noqa: E731
    assert 570.0 - brentq(f, 450.0, 570.0) == pytest.approx(50.0, abs=1e-9)
    assert brentq(f, 570.0, 800.0) - 570.0 == pytest.approx(75.0, abs=1e-9)


def test_per_nm_peak_lies_blue_of_the_energy_peak():
    # The λ⁻² Jacobian falls with wavelength: d/dλ (S λ⁻²) = −2 S λ⁻³ at the energy peak.
    lam = np.linspace(440.0, 470.0, 30001)
    assert lam[np.argmax(PUMP.density(lam))] < 450.0


def _led(values=None):
    v = {"pump_peak": 450.0, "pump_fwhm": 22.0, "phosphor_peak": 565.0, "phosphor_half_width_short": 48.0,
         "phosphor_half_width_long": 72.0, "phosphor_to_pump": 3.5, **(values or {})}
    bounds = {"pump_peak": (435, 470), "pump_fwhm": (12, 35), "phosphor_peak": (520, 650),
              "phosphor_half_width_short": (20, 80), "phosphor_half_width_long": (30, 120),
              "phosphor_to_pump": (0.5, 30)}

    def p(name):
        return {"value": v[name], "unit": "1" if name == "phosphor_to_pump" else "nm", "empirical": True,
                "reason": "test", "bounds": list(bounds[name])}

    pump = ("pump_peak", "pump_fwhm")
    return lamps.lamp_from_dict({
        "id": "t", "family": "led", "reference": "cie-led-b5",
        "physical": {n: p(n) for n in pump},
        "models": {1: {"generator": "led_two_band", "params": {n: p(n) for n in v if n not in pump}}},
    })


def test_generator_declares_its_phosphor_and_counts_six_empirical_parameters():
    profile = _led().generate(1)
    assert profile.includes == {"phosphor"}
    assert profile.empirical_count == 6
    assert profile.spectrum.total() == pytest.approx(1.0 + 3.5, rel=1e-12)


def test_fit_recovers_every_parameter_of_a_synthetic_led():
    truth = {"pump_peak": 452.0, "pump_fwhm": 19.0, "phosphor_peak": 575.0, "phosphor_half_width_short": 55.0,
             "phosphor_half_width_long": 80.0, "phosphor_to_pump": 4.2}
    spectrum = _led(truth).generate(1).spectrum.scaled(2.3)
    ref = Reference(id="synthetic", family="led", description="two-band LED with known shape",
                    spectrum=spectrum, valid_range=(380.0, 780.0), sampling_nm=1.0, kind="definition",
                    uncertainty="exact", includes=frozenset({"phosphor"}), citation="test", data_version="test")
    result = fit.fit(_led(), 1, ref)
    assert result.success
    assert result.after.spectral_nrms < 1e-6
    for p in result.parameters:
        assert p.value == pytest.approx(truth[p.name], rel=1e-4), p.name
        assert not p.at_bound


@pytest.mark.parametrize("preset", ["led_cie_b3", "led_cie_b5"])
def test_led_presets_are_scored(preset):
    row = report.score(lamps.load(preset), 1)
    assert row.comparable and row.empirical_params == 6
    assert row.delta_uv_prime < 3e-3


def test_cool_white_fits_without_any_parameter_at_a_bound():
    # One phosphor band suits a cool-white LED; the neutral-white B3 needs a blend.
    assert fit.fit(lamps.load("led_cie_b5"), 1).at_bound == []
    assert fit.fit(lamps.load("led_cie_b3"), 1).at_bound == ["phosphor_half_width_short"]


def test_lamp_sheet_efficacy_covers_all_wavelengths():
    from lss import lampsheet

    sheet = lampsheet.make(lamps.load("led_cie_b5"), 1)
    assert sheet.ler_basis == "all wavelengths (analytic)"
    assert 250 < sheet.ler_lm_per_w < 350  # cool-white phosphor LEDs: about 280-330 lm/W
    assert sheet.visible_fraction > 0.99
