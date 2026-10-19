"""LED Level 2: the phosphor layer on a blue die (design §4.2 L2)."""

import numpy as np
import pytest
from scipy.special import expn

from lss.accuracy import report
from lss.lamp import lamp as lamps
from lss.lamp import generators, phosphors
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


@pytest.mark.parametrize("n", [1.0, 1.53])
@pytest.mark.parametrize("reflectance, eta", [(1.0, 1.0), (0.8, 1.0), (1.0, 0.85), (0.6, 0.7)])
def test_every_pump_photon_leaves_or_is_lost(reflectance, eta, n):
    out = layer.transfer(PUMP, _two(eta=eta), die_reflectance=reflectance, refractive_index=n)
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


def test_fresnel_transmittance():
    # Normal incidence: 1 − ((n − 1)/(n + 1))²; nothing past the critical angle; all of it at n = 1.
    n = 1.53
    assert layer.fresnel_transmittance(1.0, n) == pytest.approx(1.0 - ((n - 1) / (n + 1)) ** 2, rel=1e-12)
    mu_c = np.sqrt(1.0 - 1.0 / n**2)
    assert layer.fresnel_transmittance(mu_c * 0.999, n) == 0.0
    assert 0.0 < layer.fresnel_transmittance(mu_c * (1 + 1e-6), n) < 0.02  # rises as √(μ − μc)
    np.testing.assert_allclose(layer.fresnel_transmittance(np.linspace(0.01, 1, 9), 1.0), 1.0, rtol=1e-12)


def test_an_empty_refracting_layer_sends_out_all_light_by_fresnel_shares():
    # With nothing to absorb and a white die, every photon leaves in the end; each pass sends
    # ∫T d(sin²θ) over a band (θ in air) out through it, so the bands share the light in that ratio.
    from scipy.integrate import quad

    n = 1.53
    out = layer.transfer(PUMP, [layer.Phosphor(*GARNET, 0.0)], refractive_index=n)
    np.testing.assert_allclose(out.photons(), PUMP / _area(PUMP), rtol=1e-9, atol=1e-15)

    def t_out(theta):  # transmittance for light leaving at θ in air
        return float(layer.fresnel_transmittance(np.sqrt(1.0 - (np.sin(theta) / n) ** 2), n))

    def band(lo, hi):
        return quad(lambda t: t_out(t) * np.sin(2 * t), np.radians(lo), np.radians(hi))[0]

    whole = band(0.0, 90.0)
    for i, (lo, hi) in enumerate(out.zones_deg):
        assert _area(out.photons(i)) == pytest.approx(band(lo, hi) / whole, rel=1e-8)


def test_band_layout_leaves_the_lamp_alone():
    # The light the top reflects is integrated on its own directions, so asking for more bands only
    # splits the same light.
    totals = [layer.transfer(PUMP, _two(), refractive_index=1.53, zones_deg=generators.equal_bands(k)).photons()
              for k in (1, 3, 6, 12)]
    for t in totals[1:]:
        np.testing.assert_allclose(t, totals[0], rtol=1e-9, atol=1e-18)


def test_directions_converge(monkeypatch):
    # The transmittance falls to 0 at the critical angle; integrated over the angle in air it is smooth.
    base = layer.transfer(PUMP, _two(), refractive_index=1.53, zones_deg=generators.equal_bands(6))
    monkeypatch.setattr(layer, "_DIRECTIONS", 2 * layer._DIRECTIONS)
    monkeypatch.setattr(layer, "_CONE_DIRECTIONS", 2 * layer._CONE_DIRECTIONS)
    fine = layer.transfer(PUMP, _two(), refractive_index=1.53, zones_deg=generators.equal_bands(6))
    for i in range(6):
        assert _area(fine.photons(i)) == pytest.approx(_area(base.photons(i)), rel=1e-9)
        np.testing.assert_allclose(fine.photons(i), base.photons(i), rtol=1e-7, atol=1e-12 * base.photons(i).max())


def test_refraction_narrows_the_colour_over_angle():
    # Inside, the bands reach grazing; in air all of 0–90° comes from inside 0–41° (n = 1.53), so
    # the blue's share changes less from the first band to the last, and still falls with angle.
    def blue(n):
        out = layer.transfer(PUMP, _two(), refractive_index=n)
        return [_area(out.leaked[i]) / _area(out.photons(i)) for i in range(3)]

    flat, refracted = blue(1.0), blue(1.53)
    assert refracted[0] > refracted[1] > refracted[2]
    assert refracted[0] / refracted[2] < 0.5 * flat[0] / flat[2]


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
    assert [z.shape for z in profile.zones] == [f"polar {b} deg fresnel 1.53" for b in ("0-30", "30-60", "60-90")]
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
        p = lamp.physical[name]
        step = 40.0 if p.bounds is None or p.value + 40.0 <= p.bounds[1] else -40.0  # B4's tail is at its top
        moved = binned({name: p.value + step})
        assert np.max(np.abs(moved / moved.sum() - cold / cold.sum())) > 1e-4


@pytest.mark.parametrize("preset", LEDS)
def test_layer_beats_the_two_band_fit(preset):
    lamp = lamps.load(preset)
    l1, l2 = report.score(lamp, 1), report.score(lamp, 2)
    assert l2.spectral_nrms < l1.spectral_nrms


def test_calibrated_findings():
    # The layer explains L1's too-wide pump: the die widths fall inside InGaN's usual 18-25 nm.
    for preset in LEDS:
        assert 18.0 <= lamps.load(preset).physical["die_fwhm"].value <= 25.0
    # Findings the design doc records (§4.2); a refit that moves them should update it. Since
    # the refraction refit (§8.6) the two warmest LEDs' tails sit at or near the lower bound and
    # the two coolest at the upper one.
    tails = [lamps.load(p).physical["die_tail_temperature"] for p in LEDS]
    assert all(t.value < t.bounds[0] + 10.0 for t in tails[:2])
    assert all(t.value > t.bounds[1] - 2.0 for t in tails[3:])
    s = phosphors.parameters()["garnet.huang_rhys"].value
    assert s < 4.0  # below YAG:Ce's measured S ~ 6: about two thirds of its Stokes shift
