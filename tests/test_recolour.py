"""Re-colouring arithmetic, channels, materials and lamp placement (design §8.4): no Mitsuba."""

import math

import numpy as np
import pytest

from lss.render import recolour
from lss.render.image import project
from lss.render.lamps import place
from lss.render.mitsuba_scene import channel_knots
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


def test_polar_zones_are_refused():
    with pytest.raises(ValueError, match="isotropic"):
        place(LampPlacement("x", "led_cie_b3", 2, 100.0, (0, 0, 0), 0.01))


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
