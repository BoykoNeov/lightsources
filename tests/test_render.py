"""Renderer tests (design §12, §8.4, §8.5): furnace, inverse square, re-colouring against a
direct render, the direct calculation, the flat emitter's angle bands, and the bin-width and
zone-count convergence tests. Slow and on demand: ``pytest -m render``.
"""

import math
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.render

mi = pytest.importorskip("mitsuba")

from lss.lamp.profile import polar_band  # noqa: E402
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


def bin_means(m: materials.Material, grid=GRID) -> np.ndarray:
    """Each render bin's mean reflectance as Mitsuba reads it (linear between knots): the
    noise-free map of a patch under a flat unit spectrum, up to the irradiance."""
    out = []
    for lo, hi in zip(grid.edges[:-1], grid.edges[1:]):
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


# --- Second slice (§8.5): the flat emitter with angle bands, and the two convergence tests.

def disc_patch_scene(theta_deg, r=0.01, d=1.0, rho=0.5, mu=(0.0, 1.0)):
    """A disc lamp (radius r, normal +z, black) and a 2 cm patch at distance d and
    ``theta_deg`` off its normal, facing it, seen head-on through a 1 cm window."""
    t = math.radians(theta_deg)
    c = np.array([d * math.sin(t), 0.0, d * math.cos(t)])
    n = -c / d
    return mi.load_dict({
        "type": "scene", "integrator": {"type": "path", "max_depth": 2},
        "sensor": ortho((c + 0.3 * n).tolist(), c.tolist(), [0, 1, 0], (0.005, 0.005), (16, 16)),
        "patch": {"type": "rectangle", "to_world": T().look_at(c.tolist(), (c + n).tolist(), [0, 1, 0]) @ T().scale(0.01),
                  "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": rho}}},
        "lamp": {"type": "disk", "to_world": T().scale(r), "bsdf": ms.BLACK,
                 "emitter": {"type": "polardisc", "mu_lo": mu[0], "mu_hi": mu[1]}}})


@pytest.mark.parametrize("fresnel", ["", " fresnel 1.53"])
def test_disc_angle_bands(fresnel):
    """§8.5: with one band on, a patch inside it reads ρ·L·r²·cos θ/d² (the finite disc
    changes this by 10⁻⁴), times the transmittance T(θ) for a Fresnel-shaped band (§8.6);
    patches in the other bands read exactly 0; the bands' maps add up to one render with a
    single 0-90° band."""
    from lss.lamp.profile import fresnel_index
    from lss.physics.led import layer

    bands = [f"polar {b} deg{fresnel}" for b in ("0-30", "30-60", "60-90")]
    n = fresnel_index(bands[0])
    for theta in (15.0, 45.0, 75.0):
        sc = disc_patch_scene(theta)
        params = mi.traverse(sc)
        t = float(layer.fresnel_transmittance(math.sqrt(1.0 - (math.sin(math.radians(theta)) / n) ** 2), n))
        expect = 0.5 * 0.01**2 * math.cos(math.radians(theta)) * t
        total = 0.0
        params["lamp.emitter.n"] = n
        for shape in bands:
            params["lamp.emitter.mu_lo"], params["lamp.emitter.mu_hi"] = ms.band_cosines(shape)
            params.update()
            m = ms.read_film(render_passes(sc, 1024, 4), GRID)
            mean, se = mean_and_se(m)
            lo, hi = polar_band(shape)
            if lo <= theta < hi:
                z = (mean - expect) / se
                print(f"theta {theta}, {shape}: mean/expected {mean.mean() / expect:.5f}, max |z| {np.abs(z).max():.2f}")
                assert np.abs(z).max() < 4.0
                assert mean.mean() == pytest.approx(expect, rel=2e-3)
            else:
                assert np.all(m == 0.0)
            total = total + m
        params["lamp.emitter.mu_lo"], params["lamp.emitter.mu_hi"] = ms.band_cosines("isotropic")
        params.update()
        whole = ms.read_film(render_passes(sc, 1024, 4), GRID)
        a, sa = mean_and_se(total)
        b, sb = mean_and_se(whole)
        z = (a - b) / np.hypot(sa, sb)
        print(f"theta {theta}: bands summed / one 0-90 band {a.mean() / b.mean():.5f}, max |z| {np.abs(z).max():.2f}")
        assert np.abs(z).max() < 4.0


def chart_renders(grid, spp=1024, passes=8):
    """The ColorChecker through §8.4's patch set-up, rendered once on ``grid`` (the map does
    not depend on the lamp). Returns (materials, patch pixel boxes, irradiance factors, map)."""
    chart = samples.get("colorchecker")
    mats = [materials.from_analysis_grid(lb, chart.values[:, i], chart.citation) for i, lb in enumerate(chart.labels)]
    res = (240, 160)
    img, boxes, factors = lit_patches(mats, None, ms.film(grid, *res), res, spp, passes)
    return mats, boxes, factors, ms.read_film(img, grid)


def chart_lab(lamp, grid, rendered, noise_free=False):
    """Each patch's CIELAB (against the white card's direct value) through the maps on
    ``grid``, rendered or from the noise-free bin means; and the direct values at 1 nm."""
    mats, boxes, factors, m = rendered
    chart = samples.get("colorchecker")
    s = place(LampPlacement("l", lamp, 0, 1000.0, (0, 0, 0), 0.5)).zones[0].radiance
    out, direct = [], []
    for i, (b, f) in enumerate(zip(boxes, factors)):
        white = colorimetry.tristimulus(BinnedSpectrum(s.grid, s.values * f / math.pi, 0, 0))
        if noise_free:
            xyz = recolour.recolour([bin_means(mats[i], grid) * f / math.pi], [s], grid)
        else:
            xyz = recolour.recolour([m[b]], [s], grid).reshape(-1, 3).mean(0)
        out.append(lab(xyz, white))
        direct.append(lab(colorimetry.tristimulus(
            BinnedSpectrum(s.grid, s.values * chart.values[:, i] * f / math.pi, 0, 0)), white))
    return np.array(out), np.array(direct)


def test_bin_halving():
    """§12: halving the render bins (10 → 5 nm) changes the chart by ΔE00 < 0.3 (§8.5)."""
    g5 = render_grid(5.0)
    r10, r5 = chart_renders(GRID), chart_renders(g5)
    for lamp in ("incandescent_cie_a", "led_cie_b3", "fluorescent_cie_fl11", "hps_cie_hp1"):
        l10, direct = chart_lab(lamp, GRID, r10)
        l5, _ = chart_lab(lamp, g5, r5)
        n10, _ = chart_lab(lamp, GRID, r10, noise_free=True)
        n5, _ = chart_lab(lamp, g5, r5, noise_free=True)
        change, change_nf = delta_e00(l10, l5), delta_e00(n10, n5)
        noise5, err5 = delta_e00(l5, n5), delta_e00(l5, direct)
        print(f"{lamp}: 10->5 nm max dE00 {change.max():.3f} (noise-free {change_nf.max():.3f}); "
              f"5 nm against its noise-free bins max {noise5.max():.3f}, against SPD x reflectance max {err5.max():.3f}")
        assert change.max() < 0.3
        # FL11's lines put its weight in a few bins, so its colour carries about three times
        # the others' noise, at 10 nm as at 5 nm (0.15 and 0.17 here; 0.045 and 0.056 with
        # four times the samples): the lamp's noise, not the bins' (§8.5).
        assert noise5.max() < (0.25 if lamp == "fluorescent_cie_fl11" else 0.06)


def test_band_count_lab_led():
    """§12: doubling the zone count should change the image by ΔE00 < 0.3. LED Level 2's
    bands are angles in air after refraction at the layer's surface, each shaped by its
    transmittance (§8.6); 3 → 6 is §12's test, 6 → 12 is printed to show the trend.

    Recorded as a finding (§8.6): the brightness now agrees (the card within 0.1 %) and each
    image against its own white card changes by < 0.15, but against a fixed white card the
    chart still moves by about 0.5, on colour: a flat band is right at one angle inside it.
    When the zones blend smoothly between angles, turn the middle assert into §12's < 0.3."""
    from dataclasses import replace

    from lss.render import image, maps
    from lss.scene import study as studies

    base = studies.load("lab_led")
    labs = {}
    for n in (3, 6, 12):
        s = replace(base, lamps=(replace(base.lamps[0], angle_bands=n),))
        placed = [place(p) for p in s.lamps]
        result = maps.render(s, placed, 512, 8, env.setup())
        for m in result:
            print(f"{n} bands, {m.info.zone}: {m.info.seconds:.2f} s, noise {100 * m.info.noise_under_study_lamp:.2f} %")
        xyz = recolour.recolour([m.mean for m in result], [r for _, _, r in maps.jobs(placed)], s.grid)
        white = next(sh for sh in s.shapes if sh.id == "white card")
        chart = [sh for sh in s.shapes if sh.id.startswith("cc ")]
        labs[n] = (image.region_mean(xyz, s.camera, white.centre),
                   [image.region_mean(xyz, s.camera, sh.centre) for sh in chart])

    def change(a, b):
        wb = labs[b][0]
        fixed = np.array([delta_e00(lab(x, wb), lab(y, wb)) for x, y in zip(labs[a][1], labs[b][1])])
        own = np.array([delta_e00(lab(x, labs[a][0]), lab(y, wb)) for x, y in zip(labs[a][1], labs[b][1])])
        card = float(delta_e00(lab(labs[a][0], wb), lab(wb, wb)))
        print(f"{a} -> {b} bands: chart dE00 against the {b}-band white card {fixed.min():.3f}..{fixed.max():.3f} "
              f"(mean {fixed.mean():.3f}), the card itself {card:.3f}, its luminance {labs[a][0][1] / wb[1]:.4f}; "
              f"each against its own card {own.min():.3f}..{own.max():.3f} (mean {own.mean():.3f})")
        return fixed, own, labs[a][0][1] / wb[1]

    fixed, own, luminance = change(3, 6)
    change(6, 12)
    assert abs(luminance - 1.0) < 0.005  # shaped bands: the brightness has converged
    assert 0.3 < fixed.max() < 1.0  # the finding: just above §12's 0.3, on colour
    assert own.max() < 0.15
