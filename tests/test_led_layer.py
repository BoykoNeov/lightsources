"""LED Level 2: the phosphor layer on a blue die (design §4.2 L2)."""

import numpy as np
import pytest
from scipy.special import expn

from lss.accuracy import report
from lss.lamp import lamp as lamps
from lss.lamp import phosphors
from lss.physics.led import layer

LEDS = ["led_cie_b1", "led_cie_b2", "led_cie_b3", "led_cie_b4", "led_cie_b5"]
D = layer._DNU
PUMP = layer.die_emission(450.0, 20.0, 350.0)
GARNET = layer.ce_garnet(20450.0, 6.0, 200.0, 350.0, 1500.0, 2000.0, 0.6)
NITRIDE = layer.eu_nitride(630.0, 90.0, 575.0, 1500.0)


def _two(tau_g=1.5, tau_n=0.4, eta=1.0):
    return [layer.Phosphor(*GARNET, tau_g, eta), layer.Phosphor(*NITRIDE, tau_n, eta)]


def _area(x):
    return float(np.sum(x * D))


@pytest.mark.parametrize("reflectance, eta", [(1.0, 1.0), (0.8, 1.0), (1.0, 0.85), (0.6, 0.7)])
def test_every_pump_photon_leaves_or_is_lost(reflectance, eta):
    out = layer.transfer(PUMP, _two(eta=eta), die_reflectance=reflectance)
    assert _area(out.photons()) + out.lost == pytest.approx(1.0, abs=1e-6)
    if reflectance == 1.0 and eta == 1.0:
        assert out.lost == 0.0


def test_an_empty_layer_passes_the_pump_unchanged():
    out = layer.transfer(PUMP, [layer.Phosphor(*GARNET, 0.0)])
    np.testing.assert_allclose(out.photons(), PUMP / _area(PUMP), atol=1e-15)


@pytest.mark.parametrize("tau", [0.05, 0.5, 2.0, 6.0])
def test_leaked_pump_is_a_lambertian_beer_lambert_transmission(tau):
    # A flat absorber of depth τ passes 2E₃(τ) of a Lambertian source: ∫ 2μ e^(−τ/μ) dμ.
    out = layer.transfer(PUMP, [layer.Phosphor(np.ones_like(PUMP), *GARNET[1:], tau)])
    assert _area(out.leaked.sum(axis=0)) == pytest.approx(2.0 * expn(3, tau), rel=1e-5)


def test_blue_leakage_falls_with_angle():
    # The path through the layer grows as 1/cos θ, so oblique light is yellower.
    out = layer.transfer(PUMP, _two())
    blue = [_area(out.leaked[i]) / _area(out.photons(i)) for i in range(len(out.zones_deg))]
    assert blue[0] > blue[1] > blue[2]
    assert sum(_area(out.photons(i)) for i in range(3)) == pytest.approx(_area(out.photons()), rel=1e-12)


def test_thicker_layer_leaks_less_blue():
    thin, thick = (layer.transfer(PUMP, _two(tau_g=t)) for t in (0.5, 2.0))
    assert _area(thick.leaked.sum(axis=0)) < _area(thin.leaked.sum(axis=0))


def test_nitride_takes_part_of_the_garnet_light():
    # Re-absorption: the nitride absorbs garnet photons as well as pump photons.
    alone = layer.transfer(PUMP, [layer.Phosphor(*GARNET, 1.5)])
    both = layer.transfer(PUMP, _two())
    garnet_out = lambda o: _area(o.converted[:, 0].sum(axis=0))  # noqa: E731
    assert garnet_out(both) < garnet_out(alone)
    pump_only = _two()[1]
    first = layer.transfer(PUMP, [layer.Phosphor(*GARNET, 0.0), pump_only])
    assert both.absorbed[1] > 0 and first.absorbed[1] > 0


def test_ce_bands_mirror_about_the_zero_phonon_line():
    zpl, s, hw = 20450.0, 6.0, 200.0
    a, e = layer.ce_garnet(zpl, s, hw, 300.0, 0.0, 2000.0, 0.0)
    nu = layer.NU
    assert nu[np.argmax(a)] == pytest.approx(zpl + s * hw, abs=20.0)
    assert nu[np.argmax(e)] == pytest.approx(zpl - s * hw, abs=20.0)
    assert _area(e) == pytest.approx(1.0, rel=1e-12)


def test_phonon_width_grows_with_temperature():
    # σ² = S (ħω)² coth(ħω / 2kT).
    def fwhm(t):
        a, _ = layer.ce_garnet(20450.0, 6.0, 200.0, t, 0.0, 2000.0, 0.0)
        above = layer.NU[a >= 0.5]
        return above.max() - above.min()

    kt = lambda t: t / layer.C2_CM_K  # noqa: E731
    ratio = np.sqrt(1.0 / np.tanh(200.0 / (2 * kt(450.0))) * np.tanh(200.0 / (2 * kt(300.0))))
    assert fwhm(450.0) / fwhm(300.0) == pytest.approx(ratio, rel=0.03)


def test_die_has_a_longer_short_wavelength_side():
    # The Boltzmann tail of hot carriers lies on the high-energy side.
    nu = layer.NU
    p = layer.die_emission(450.0, 20.0, 360.0)
    top = nu[np.argmax(p)]
    above = nu[p >= p.max() / 2]
    assert above.max() - top > top - above.min()


def test_library_holds_the_led_phosphor_constants():
    params = phosphors.parameters()
    assert {"garnet.huang_rhys", "garnet.phonon_energy", "nitride.emission_peak"} <= set(params)
    assert not params["garnet.phonon_energy"].empirical  # measured (Bachmann et al. 2009)
    assert set(phosphors.library()) == {"BAM", "LAP", "YOX"}  # band phosphors only


@pytest.mark.parametrize("preset", LEDS)
def test_generator_zones_make_up_the_lamp(preset):
    lamp = lamps.load(preset)
    profile = lamp.generate(2)
    assert profile.includes == {"phosphor"}
    assert [z.shape for z in profile.zones] == ["polar 0-30 deg", "polar 30-60 deg", "polar 60-90 deg"]
    total = sum(z.spectrum.total() for z in profile.zones)
    assert profile.spectrum.total() == pytest.approx(total, rel=1e-12)
    shared = set(profile.metadata["shared"])
    assert shared and all(phosphors.is_shared(n) for n in shared)
    assert "garnet_zero_phonon" not in shared  # each lamp's own garnet


def test_die_tail_and_phosphor_temperatures_are_separate_inputs():
    # Level 2 once used one temperature for both; the fits showed it is the die tail's shape
    # (design §4.2), so each now has its own input and each alone changes the spectrum.
    from lss.accuracy import fit
    from lss.spectral.grids import BinGrid

    lamp = lamps.load("led_cie_b4")
    assert {"die_tail_temperature", "phosphor_temperature"} <= set(lamp.generate(2).metadata["params"])
    grid = BinGrid(np.arange(380.0, 781.0, 5.0))

    def binned(values):
        return fit.with_values(lamp, 2, values).generate(2).spectrum.bin(grid).values

    cold = binned({})
    for name in ("die_tail_temperature", "phosphor_temperature"):
        hot = binned({name: lamp.physical[name].value + 40.0})
        assert np.max(np.abs(hot / hot.sum() - cold / cold.sum())) > 1e-4


@pytest.mark.parametrize("preset", LEDS)
def test_layer_beats_the_two_band_fit(preset):
    lamp = lamps.load(preset)
    l1, l2 = report.score(lamp, 1), report.score(lamp, 2)
    assert l2.spectral_nrms < l1.spectral_nrms


def test_calibrated_findings():
    # The layer explains L1's too-wide pump: the die widths fall inside InGaN's usual 18-25 nm.
    for preset in LEDS:
        assert 18.0 <= lamps.load(preset).physical["die_fwhm"].value <= 25.0
    # Findings the design doc records (§4.2); a refit that moves them should update it.
    warm = [lamps.load(p).physical["die_tail_temperature"] for p in LEDS[:3]]
    assert all(t.value == pytest.approx(t.bounds[0], abs=0.1) for t in warm)
    s = phosphors.parameters()["garnet.huang_rhys"].value
    assert s < 4.0  # below YAG:Ce's measured S ~ 6: half its Stokes shift
