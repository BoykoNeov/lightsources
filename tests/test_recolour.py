"""Re-colouring arithmetic, channels, materials and lamp placement (design §8.4, §8.5): no Mitsuba."""

import math
from dataclasses import replace

import numpy as np
import pytest

from lss.lamp.profile import fresnel_index, polar_band
from lss.render import maps, recolour
from lss.render.image import project
from lss.render.lamps import band_share, flux_lm, place, place_profile
from lss.render.mitsuba_scene import band_cosines, channel_knots
from lss.scene import materials
from lss.scene import study as studies
from lss.scene.study import LampPlacement
from lss.spectral import colorimetry
from lss.spectral.grids import analysis_grid, render_grid
from lss.spectral.spectrum import BinnedSpectrum


def binned(lamp: str, level: int = 0) -> BinnedSpectrum:
    from lss.lamp import lamp as lamps

    return lamps.load(lamp).generate(level).spectrum.bin(analysis_grid())


def test_render_bins_sit_on_analysis_edges():
    g = render_grid()
    assert len(g) == 47
    assert g.lo == 359.5 and g.hi == 830.5
    assert g.widths[-1] == pytest.approx(11.0) and np.allclose(g.widths[:-1], 10.0)
    idx = recolour.bin_index(g)
    assert (idx >= 0).all() and np.all(np.diff(idx) >= 0)
    with pytest.raises(ValueError):
        recolour.bin_index(render_grid(10.0, 360.0, 830.0))


@pytest.mark.parametrize("lamp", ["incandescent_cie_a", "hps_cie_hp1", "led_cie_b3"])
def test_constant_map_recolours_to_tristimulus(lamp):
    s = binned(lamp)
    rho = 0.37
    xyz = recolour.recolour([np.full((2, 3, 47), rho)], [s], render_grid())
    direct = colorimetry.tristimulus(BinnedSpectrum(s.grid, rho * s.values, 0, 0))
    assert np.allclose(xyz, direct, rtol=1e-12)


def test_map_constant_per_bin_recolours_exactly():
    s = binned("hps_cie_hp1")
    g = render_grid()
    rng = np.random.default_rng(3)
    per_bin = rng.uniform(0.1, 0.9, len(g))
    xyz = recolour.recolour([per_bin], [s], g)
    rho = per_bin[recolour.bin_index(g)]
    assert np.allclose(xyz, colorimetry.tristimulus(BinnedSpectrum(s.grid, rho * s.values, 0, 0)), rtol=1e-12)


def test_channels_sum_to_one_inside_the_range():
    g = render_grid()
    lam = np.linspace(g.lo + 1e-3, g.hi - 1e-3, 200_001)
    total = sum(np.interp(lam, *channel_knots(g, j), left=0, right=0) for j in range(len(g)))
    assert np.allclose(total, 1.0, atol=1e-9)


def test_materials_cover_the_render_range():
    m = materials.flat("f", 0.4)
    assert m.at(render_grid().lo) == pytest.approx(0.4) and m.at(render_grid().hi) == pytest.approx(0.4)
    assert m.at(300.0) == 0.0  # as Mitsuba reads it beyond the knots
    g = render_grid()
    v = np.linspace(0.1, 0.9, len(g))
    st = materials.stepped("s", g.edges, v)
    inner = g.centres + 0.4 * g.widths * np.array([-1, 1])[:, None]
    assert np.allclose(st.at(inner), v)


def test_lamp_is_scaled_to_the_study_flux():
    p = LampPlacement("x", "incandescent_cie_a", 1, 1234.0, (0, 0, 0), 0.05)
    pl = place(p)
    lm = sum(colorimetry.luminous(z.radiance) for z in pl.zones) * math.pi * 4 * math.pi * 0.05**2
    assert lm == pytest.approx(1234.0, rel=1e-12)
    assert "not the lamp's own output" in pl.note


def test_polar_zones_are_refused_on_a_sphere():
    with pytest.raises(ValueError, match="only 'isotropic'"):
        place(LampPlacement("x", "led_cie_b3", 2, 100.0, (0, 0, 0), 0.01))


def disc(lamp="led_cie_b3", level=2, flux=1000.0, bands=None):
    return LampPlacement("d", lamp, level, flux, (0, 0, 0), 0.05, "disc", (0, 0, -1), bands)


def test_polar_band_shapes():
    assert polar_band("isotropic") == (0.0, 90.0)
    assert polar_band("polar 7.5-15 deg") == (7.5, 15.0)
    assert polar_band("b 0-0.5") is None
    with pytest.raises(ValueError):
        polar_band("polar 60-30 deg")
    assert band_share("polar 0-90 deg") == pytest.approx(1.0)
    assert band_cosines("polar 60-90 deg") == (0.0, pytest.approx(0.5))
    assert band_cosines("isotropic") == (0.0, 1.0)
    assert maps.map_id("lamp", "polar 7.5-15 deg") == "lamp@polar-7.5-15"
    assert maps.map_id("lamp", "isotropic") == "lamp@isotropic"
    assert polar_band("polar 60-90 deg fresnel 1.53") == (60.0, 90.0)
    assert fresnel_index("polar 60-90 deg fresnel 1.53") == 1.53
    assert fresnel_index("polar 60-90 deg") == fresnel_index("isotropic") == 1.0
    # A shaped band is another map: a flat band's map is never reused for it.
    assert maps.map_id("lamp", "polar 60-90 deg fresnel 1.53") == "lamp@polar-60-90-fresnel-1.53"


def test_fresnel_band_share():
    """§8.6: a band shaped by the transmittance T sends ∫_band T d(sin²θ); the shares add up
    over the bands, and an empty layer's band powers (the layer's own Fresnel shares) come
    out at one radiance."""
    from scipy.integrate import quad

    from lss.lamp.generators import equal_bands
    from lss.physics.led import layer

    n = 1.53

    def t_out(theta):
        return float(layer.fresnel_transmittance(math.sqrt(1.0 - (math.sin(theta) / n) ** 2), n))

    for lo, hi in ((0.0, 30.0), (60.0, 90.0), (75.0, 90.0)):
        exact = quad(lambda t: t_out(t) * math.sin(2 * t), math.radians(lo), math.radians(hi))[0]
        assert band_share(f"polar {lo:g}-{hi:g} deg fresnel 1.53") == pytest.approx(exact, rel=1e-10)
    whole = band_share("polar 0-90 deg fresnel 1.53")
    assert sum(band_share(f"polar {lo:g}-{hi:g} deg fresnel 1.53") for lo, hi in equal_bands(6)) == pytest.approx(
        whole, rel=1e-12)
    out = layer.transfer(layer.die_emission(450.0, 20.0, 350.0), [layer.Phosphor(np.ones(layer.GRID_NM.size),
                         np.ones(layer.GRID_NM.size), 0.0)], refractive_index=n, zones_deg=equal_bands(6))
    power = [float(np.sum(out.photons(i))) for i in range(6)]
    radiance = [p / band_share(f"polar {lo:g}-{hi:g} deg fresnel 1.53") for p, (lo, hi) in zip(power, equal_bands(6))]
    assert np.allclose(radiance, radiance[0], rtol=1e-8)


def test_disc_bands_are_scaled_to_the_study_flux():
    """§8.5, §8.6: Σ_k π·A·s_k·K_m∫L_k V = Φ_v, with A the disc's front and s_k the band's
    share, ∫_band T d(sin²θ) for LED Level 2's Fresnel-shaped bands."""
    for bands in (None, 6):
        pl = place(disc(bands=bands))
        assert len(pl.zones) == (3 if bands is None else bands)
        assert flux_lm(pl) == pytest.approx(1000.0, rel=1e-12)


def test_a_lambertian_profile_gives_equal_band_radiance():
    """Bands holding a Lambertian surface's shares of one spectrum (what the layer gives with
    no phosphor) come out at one radiance, the same as a single isotropic zone."""
    from lss.lamp.generators import equal_bands
    from lss.lamp.profile import EmissionProfile, Zone
    from lss.spectral.continuum import PlanckContinuum
    from lss.spectral.spectrum import Spectrum

    def share(shape):
        return Spectrum.of(PlanckContinuum(3000.0, band_share(shape)))

    zones = tuple(Zone(share(f"polar {lo:g}-{hi:g} deg"), f"polar {lo:g}-{hi:g} deg") for lo, hi in equal_bands(3))
    pl = place_profile(disc(), EmissionProfile(zones), 2, "test")
    one = place_profile(disc(), EmissionProfile((Zone(share("isotropic")),)), 2, "test").zones[0].radiance.values
    for z in pl.zones:
        assert np.allclose(z.radiance.values, one, rtol=1e-12)


def test_angle_bands_split_the_same_light():
    """More bands only split the layer's output: the summed spectrum hardly moves."""
    from lss.lamp import lamp as lamps

    led = lamps.load("led_cie_b3")
    three = led.generate(2).spectrum.bin(analysis_grid()).values
    six = led.generate(2, angle_bands=6)
    assert [z.shape for z in six.zones][:2] == ["polar 0-15 deg fresnel 1.53", "polar 15-30 deg fresnel 1.53"]
    # The directions are taken over the angle in air, where the transmittance is smooth, and the
    # light the top reflects on its own directions (§8.6), so the sum moves only by rounding.
    assert np.allclose(six.spectrum.bin(analysis_grid()).values, three, rtol=1e-9, atol=1e-15 * three.max())
    with pytest.raises(ValueError, match="no angle bands"):
        lamps.load("incandescent_cie_a").generate(1, angle_bands=6)


def test_led_lab_study():
    s = studies.load("lab_led")
    p = s.lamps[0]
    assert p.kind == "disc" and p.normal == (0.0, 0.0, -1.0) and p.area == pytest.approx(math.pi * 0.05**2)
    assert s.geometry_hash() != studies.load("lab").geometry_hash()
    more = studies.Study(s.id, s.description, s.scene, s.camera, (replace(p, angle_bands=6),),
                         s.bin_nm, s.max_depth, s.shapes, s.materials)
    assert more.geometry_hash() == s.geometry_hash()  # a band's map does not depend on the others
    tilted = studies.Study(s.id, s.description, s.scene, s.camera, (replace(p, normal=(0, 1, -1)),),
                           s.bin_nm, s.max_depth, s.shapes, s.materials)
    assert tilted.geometry_hash() != s.geometry_hash()
    jobs = maps.jobs([place(p)])
    assert [shape for _, shape, _ in jobs] == [f"polar {b} deg fresnel 1.53" for b in ("0-30", "30-60", "60-90")]


def test_study_hash_ignores_the_lamp_physics():
    a = studies.load("lab")
    doc_lamp = a.lamps[0]
    b = studies.Study(a.id, a.description, a.scene, a.camera,
                      (LampPlacement(doc_lamp.id, "hps_cie_hp1", 0, 50.0, doc_lamp.centre, doc_lamp.radius),),
                      a.bin_nm, a.max_depth, a.shapes, a.materials)
    assert a.geometry_hash() == b.geometry_hash()
    moved = studies.Study(a.id, a.description, a.scene, a.camera,
                          (LampPlacement(doc_lamp.id, doc_lamp.lamp, None, 1.0, (0, 0, 1), doc_lamp.radius),),
                          a.bin_nm, a.max_depth, a.shapes, a.materials)
    assert a.geometry_hash() != moved.geometry_hash()


def test_camera_projection():
    cam = studies.load("lab").camera
    x, y = project(cam, cam.target)
    assert (x, y) == pytest.approx((cam.width / 2, cam.height / 2))
    right = np.asarray(cam.target) + (1.0, 0.0, 0.0)
    assert project(cam, right)[0] > cam.width / 2
    above = np.asarray(cam.target) + (0.0, 0.0, 0.5)
    assert project(cam, above)[1] < cam.height / 2
