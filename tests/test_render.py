"""Renderer tests (design §12, §8.4): furnace, inverse square, re-colouring against a direct
render, and the direct calculation. Slow and on demand: ``pytest -m render``.
"""

import math
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.render

mi = pytest.importorskip("mitsuba")

from lss.render import env, recolour  # noqa: E402
from lss.render import mitsuba_scene as ms  # noqa: E402
from lss.render.lamps import place  # noqa: E402
from lss.scene import materials  # noqa: E402
from lss.scene.study import LampPlacement  # noqa: E402
from lss.spectral import colorimetry  # noqa: E402
from lss.spectral.grids import analysis_grid, render_grid  # noqa: E402
from lss.spectral.spectrum import BinnedSpectrum  # noqa: E402
from lss.studies import samples  # noqa: E402

GRID = render_grid()
T = None


@pytest.fixture(scope="module", autouse=True)
def variant():
    global T
    v = env.setup()
    T = mi.ScalarTransform4f
    return v


def ortho(origin, target, up, half, res, film=None):
    """Mitsuba's orthographic camera spans ±1 across and ±1/aspect up before ``to_world``,
    so one scale serves both axes; ``half`` must have the image's aspect."""
    assert math.isclose(half[0] / half[1], res[0] / res[1])
    return {"type": "orthographic",
            "to_world": T().look_at(origin, target, up) @ T().scale([half[0], half[0], 1]),
            "film": film or ms.film(GRID, *res), "sampler": {"type": "independent"}}


def mean_and_se(img):
    """Per-channel mean over pixels and its standard error (pixels are independent)."""
    px = np.asarray(img, dtype=np.float64).reshape(-1, np.asarray(img).shape[-1])
    return px.mean(0), px.std(0, ddof=1) / math.sqrt(px.shape[0])


def render_passes(scene, spp, passes):
    return np.mean([np.asarray(mi.render(scene, spp=spp, seed=s), dtype=np.float64) for s in range(passes)], axis=0)


def stepped_lamp(binned: BinnedSpectrum) -> dict:
    """A Mitsuba spectrum constant on each 1 nm analysis bin at the binned density."""
    g = analysis_grid()
    m = materials.stepped("lamp", g.edges, binned.values / g.widths)
    return ms.spectrum_dict(m.wavelengths, m.reflectance)


def cmf_channels(observer="2"):
    """Three film channels: the CMFs, constant on each 1 nm analysis bin at the bin-centre
    values ``colorimetry.tristimulus`` uses."""
    g = analysis_grid()
    cmf = colorimetry.cmfs(observer)
    out = []
    for c in range(3):
        m = materials.stepped("cmf", g.edges, np.clip(cmf[:, c], 0, None) / max(cmf[:, c].max(), 1.0))
        out.append((m.wavelengths, m.reflectance * max(cmf[:, c].max(), 1.0)))
    return out


def test_furnace():
    """Inside a closed box of albedo ρ emitting 1 everywhere: radiance 1/(1 − ρ) in every bin.

    A cube, not a sphere: Mitsuba's sphere, emitting and lit from inside, makes the first
    bounce 4.5 % too bright (0.5226 for 0.5; a cube gives 0.49999), so its furnace reads
    2.037 for 2 (§8.4). Lamps are spheres seen from outside, which the inverse-square
    test checks."""
    rho = 0.5
    sc = mi.load_dict({
        "type": "scene", "integrator": {"type": "path", "max_depth": -1},
        "sensor": {"type": "perspective", "fov": 60, "to_world": T().look_at([0, 0, 0], [0.3, 0.2, 1], [0, 1, 0]),
                   "film": ms.film(GRID, 32, 32), "sampler": {"type": "independent"}},
        "box": {"type": "cube", "flip_normals": True,
                "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": rho}},
                "emitter": {"type": "area", "radiance": {"type": "uniform", "value": 1.0}}}})
    m = ms.read_film(render_passes(sc, 1024, 4), GRID)
    mean, se = mean_and_se(m)
    z = (mean - 1 / (1 - rho)) / se
    print(f"furnace: mean/expected {np.min(mean) / 2:.5f}..{np.max(mean) / 2:.5f}, max |z| {np.abs(z).max():.2f}")
    assert np.abs(z).max() < 4.0
    assert mean.mean() == pytest.approx(2.0, rel=1e-3)  # all bins together


def patch_under_sphere(d, r=0.01, rho=0.5):
    """A patch at distance d from a sphere lamp (radiance 1), seen obliquely, 1 cm across."""
    view = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
    centre = np.array([0.0, d, 0.0])
    return mi.load_dict({
        "type": "scene", "integrator": {"type": "path", "max_depth": 2},
        "sensor": ortho((centre + 0.3 * view).tolist(), centre.tolist(), [0, 0, 1], (0.005, 0.005), (32, 32)),
        "patch": {"type": "rectangle", "to_world": T().translate(centre.tolist()) @ T().rotate([1, 0, 0], 90) @ T().scale(0.05),
                  "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": rho}}},
        "lamp": {"type": "sphere", "radius": r,
                 "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.0}},
                 "emitter": {"type": "area", "radiance": {"type": "uniform", "value": 1.0}}}})


def test_inverse_square():
    """A uniform sphere (radius r, radiance L) gives a facing patch irradiance πL(r/d)²."""
    out = {}
    for d in (0.5, 1.0):
        m = ms.read_film(render_passes(patch_under_sphere(d), 1024, 4), GRID)
        mean, se = mean_and_se(m)
        expect = 0.5 * (0.01 / d) ** 2
        # The 1 cm view: across it the irradiance falls by at most 1.5 (x/d)^2 ~ 3e-4 of itself.
        z = (mean - expect) / se
        out[d] = mean.mean()
        print(f"d {d}: mean/expected {mean.mean() / expect:.5f}, max |z| {np.abs(z).max():.2f}")
        assert np.abs(z).max() < 4.0
        assert mean.mean() == pytest.approx(expect, rel=2e-3)
    assert out[0.5] / out[1.0] == pytest.approx(4.0, rel=3e-3)


def lit_patches(reflectances: list[materials.Material], lamp_radiance: dict | None, film: dict, res, spp, passes,
                distance=10.0, radius=0.5, tilt_deg=30.0):
    """Patches 10 cm across in a 6-wide grid, lit by a distant sphere at ``tilt_deg`` from
    their normal, seen head-on. Returns (image, patch pixel boxes, each patch's irradiance
    per unit lamp radiance: πr²·cos/D² from its centre, exact for a uniform sphere)."""
    cols = 6
    rows = math.ceil(len(reflectances) / cols)
    pitch = 0.12
    d = {"type": "scene", "integrator": {"type": "path", "max_depth": 2}}
    boxes = []
    for i, m in enumerate(reflectances):
        x = (i % cols - (cols - 1) / 2) * pitch
        y = ((rows - 1) / 2 - i // cols) * pitch
        d[f"p{i:02d}"] = {"type": "rectangle", "to_world": T().translate([x, y, 0]) @ T().scale(0.05),
                          "bsdf": ms.bsdf(m)}
        boxes.append((x, y))
    half = (cols * pitch / 2, rows * pitch / 2)
    d["sensor"] = ortho([0, 0, 1], [0, 0, 0], [0, 1, 0], half, res, film)
    t = math.radians(tilt_deg)
    centre = [distance * math.sin(t), 0.0, distance * math.cos(t)]
    d["lamp"] = {"type": "sphere", "center": centre, "radius": radius,
                 "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.0}},
                 "emitter": {"type": "area", "radiance": lamp_radiance or {"type": "uniform", "value": 1.0}}}
    img = render_passes(mi.load_dict(d), spp, passes)
    w, h = res
    px = []
    for x, y in boxes:  # central 6 cm of each patch
        cx, cy = (x / half[0] + 1) / 2 * w, (1 - y / half[1]) / 2 * h
        rx, ry = 0.03 / half[0] / 2 * w, 0.03 / half[1] / 2 * h
        px.append((slice(int(cy - ry), int(cy + ry)), slice(int(cx - rx), int(cx + rx))))
    factors = []
    for x, y in boxes:
        v = np.asarray(centre) - (x, y, 0.0)
        dist = float(np.linalg.norm(v))
        factors.append(math.pi * (radius / dist) ** 2 * v[2] / dist)
    return img, px, factors


def test_recolouring_equals_a_direct_render():
    """On reflectances constant within each render bin, maps × spectrum is exact: equal to a
    render with the lamp's own spectrum, seen through the CMFs, within noise."""
    rng = np.random.default_rng(7)
    mats = [materials.stepped(f"m{i}", GRID.edges, rng.uniform(0.05, 0.95, len(GRID))) for i in range(6)]
    pl = place(LampPlacement("hp1", "hps_cie_hp1", 0, 1000.0, (0, 0, 0), 0.5))
    s = pl.zones[0].radiance
    scale = 1.0 / (s.values.max())  # keep the numbers near 1
    s1 = BinnedSpectrum(s.grid, s.values * scale, 0, 0, s.quantity)
    res = (192, 32)
    img_m, boxes, _ = lit_patches(mats, None, None, res, 1024, 4)
    maps_ = ms.read_film(img_m, GRID)
    xyz_maps = recolour.recolour([maps_], [s1], GRID)
    film_xyz = ms.film(GRID, *res, channels=cmf_channels())
    img_d, _, _ = lit_patches(mats, stepped_lamp(s1), film_xyz, res, 1024, 4)
    for i, b in enumerate(boxes):
        a, sa = mean_and_se(xyz_maps[b])
        dd, sd = mean_and_se(img_d[b])
        z = (a - dd) / np.hypot(sa, sd)
        print(f"patch {i}: maps/direct {np.round(a / dd, 4)}, z {np.round(z, 2)}")
        assert np.abs(z).max() < 4.0
        assert np.allclose(a, dd, rtol=5e-3)


def lab(xyz, white):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour

        return colour.XYZ_to_Lab(xyz / white[1], colour.XYZ_to_xy(white))


def delta_e00(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour

        return colour.delta_E(a, b, method="CIE 2000")


def chart_delta_e(lamp: str, spp=1024, passes=8):
    chart = samples.get("colorchecker")
    mats = [materials.from_analysis_grid(lb, chart.values[:, i], chart.citation) for i, lb in enumerate(chart.labels)]
    pl = place(LampPlacement("l", lamp, 0, 1000.0, (0, 0, 0), 0.5))
    s = pl.zones[0].radiance
    img, boxes, factors = lit_patches(mats, None, None, (240, 160), spp, passes)
    xyz_img = recolour.recolour([ms.read_film(img, GRID)], [s], GRID)
    to_direct, to_bins = {}, {}
    for i, (b, f) in enumerate(zip(boxes, factors)):
        white = colorimetry.tristimulus(BinnedSpectrum(s.grid, s.values * f / math.pi, 0, 0))
        rendered = xyz_img[b].reshape(-1, 3).mean(0)
        direct = colorimetry.tristimulus(BinnedSpectrum(s.grid, s.values * chart.values[:, i] * f / math.pi, 0, 0))
        binned = recolour.recolour([bin_means(mats[i]) * f / math.pi], [s], GRID)
        to_direct[chart.labels[i]] = float(delta_e00(lab(rendered, white), lab(direct, white)))
        to_bins[chart.labels[i]] = float(delta_e00(lab(rendered, white), lab(binned, white)))
    return to_direct, to_bins


def bin_means(m: materials.Material) -> np.ndarray:
    """Each render bin's mean reflectance as Mitsuba reads it (linear between knots): the
    noise-free map of a patch under a flat unit spectrum, up to the irradiance."""
    out = []
    for lo, hi in zip(GRID.edges[:-1], GRID.edges[1:]):
        x = np.linspace(lo, hi, 4001)
        out.append(np.trapezoid(m.at(x), x) / (hi - lo))
    return np.array(out)


@pytest.mark.parametrize("lamp", ["incandescent_cie_a", "led_cie_b3"])
def test_direct_calculation(lamp):
    """§12: ColorChecker patches through the maps path against SPD × reflectance (ΔE00 < 0.5)."""
    de, noise = chart_delta_e(lamp)
    worst = max(de, key=de.get)
    print(f"{lamp}: max dE00 {de[worst]:.3f} ({worst}), mean {np.mean(list(de.values())):.3f}; "
          f"against the noise-free bin means max {max(noise.values()):.3f}")
    assert max(de.values()) < 0.5
    assert max(noise.values()) < 0.06  # what is left beyond the bins' own error is noise (0.044 seen)


def test_direct_calculation_hps_records_the_bin_width():
    """Under HP1 the 10 nm bins cost more; recorded, not passed or failed (§8.4)."""
    de, noise = chart_delta_e("hps_cie_hp1")
    ranked = sorted(de.items(), key=lambda kv: -kv[1])
    print("hps_cie_hp1: " + ", ".join(f"{k} {v:.3f}" for k, v in ranked[:5]) + f"; mean {np.mean(list(de.values())):.3f}"
          f"; against the noise-free bin means max {max(noise.values()):.3f}")
    assert max(de.values()) < 5.0
    assert max(noise.values()) < 0.06
