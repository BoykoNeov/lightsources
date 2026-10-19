"""The viewer's Qt-free core (design §9.7): re-colouring, the probe, the display modes,
missing maps and noise notes, on small made-up maps. No Mitsuba, no Qt."""

import numpy as np
import pytest
import yaml

from lss.render import image, maps, recolour
from lss.scene import study as studies
from lss.spectral.units import K_M
from lss.ui.session import Choice, MissingMaps, Session

W, H = 64, 48


def small(name: str):
    """A packaged study at 64 × 48 pixels (its maps are made up here)."""
    from importlib import resources

    doc = yaml.safe_load(resources.files("lss.data").joinpath("studies", f"{name}.yaml").read_text(encoding="utf-8"))
    doc["camera"].update(width=W, height=H)
    doc["id"] = f"{name}_small"
    return studies.study_from_dict(doc)


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_CACHE", str(tmp_path))


def store(study, shape="isotropic", mean=None, noise_lamp="incandescent_cie_a level 2", passes=8, seed=0):
    rng = np.random.default_rng(seed)
    n = len(study.grid)
    mean = rng.uniform(0.05, 0.5, (H, W, n)) if mean is None else mean
    m2 = rng.uniform(0.0, 1e-3, (H, W, n))
    info = maps.MapInfo("lamp", passes, 64, "test", 1.0, 0.02, noise_lamp, shape)
    maps.save(study, maps.LightingMap(info, mean.astype(np.float32), m2.astype(np.float32)))
    return maps.load(study, "lamp", shape, mmap=False)


def test_image_is_the_maps_recoloured():
    study = small("lab")
    m = store(study)
    s = Session(study)
    (_, _, radiance), = s.jobs()
    want = recolour.recolour([m.mean], [radiance], study.grid)
    assert np.allclose(s.xyz(), want, rtol=1e-12, atol=0)


def test_probe_gives_the_image_tristimulus_values():
    study = small("lab")
    store(study)
    s = Session(study)
    xyz = s.xyz()
    for x, y, half in ((30, 20, 2), (5, 7, 0), (0, 0, 2)):
        p = s.probe(x, y, half)
        square = xyz[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1].reshape(-1, 3).mean(axis=0)
        assert np.allclose(p.xyz, square, rtol=1e-9, atol=0)
        assert p.luminance_cd_m2 == pytest.approx(K_M * p.xyz[1], rel=1e-3)  # ȳ is V(λ)


def test_probe_standard_error_from_the_maps():
    study = small("lab")
    m = store(study)
    s = Session(study)
    p = s.probe(10, 10, 0)
    (_, _, radiance), = s.jobs()
    idx = recolour.bin_index(study.grid)
    se_bin = np.sqrt(np.asarray(m.m2[10, 10], dtype=float) / (8 * 7))
    assert np.allclose(p.se, radiance.values * se_bin[idx], rtol=1e-6)


def test_dimming_scales_the_image():
    study = small("lab")
    store(study)
    s = Session(study)
    full = s.xyz()
    s.choices["lamp"] = Choice("incandescent_cie_a", 2, dimming=0.25)
    assert np.allclose(s.xyz(), 0.25 * full, rtol=1e-12)
    assert "dimmed to 25 %" in s.normalisation() and "1000 lm" in s.normalisation()


def test_missing_maps_are_named():
    study = small("lab_led")
    for shape in ("polar 0-30 deg fresnel 1.53", "polar 30-60 deg fresnel 1.53", "polar 60-90 deg fresnel 1.53"):
        store(study, shape)
    s = Session(study)
    assert s.missing() == []
    s.choices["lamp"] = Choice("led_cie_b3", 1)
    assert [maps.map_id(pl.placement.id, shape) for pl, shape, _ in s.missing()] == ["lamp@isotropic"]
    with pytest.raises(MissingMaps) as e:
        s.xyz()
    assert e.value.missing == ["lamp@isotropic"]


def test_a_sphere_refuses_angle_bands():
    s = Session(small("lab"))
    s.choices["lamp"] = Choice("led_cie_b3", 2)
    with pytest.raises(ValueError, match="sphere takes only 'isotropic'"):
        s.placed("lamp")


def test_noise_is_not_carried_to_another_lamp():
    study = small("lab")
    store(study)
    s = Session(study)
    assert "noise 2.00 % under incandescent_cie_a level 2" in s.noise_notes()[0]
    s.choices["lamp"] = Choice("hps_cie_hp1", 0)
    assert "not measured under hps_cie_hp1 level 0" in s.noise_notes()[0]


def test_white_card_mode_is_lss_renders_picture():
    study = small("lab")
    store(study)
    s = Session(study)
    xyz = s.xyz()
    white = image.region_mean(xyz, study.camera, next(x for x in study.shapes if x.id == "white card").centre)
    assert np.array_equal(s.picture(xyz, "card"), image.to_srgb(xyz, white))
    assert np.allclose(s.picture(xyz, "card", 1.0), image.to_srgb(2 * xyz, white))


def test_daylight_display_shows_sodium_orange():
    """A grey world (the map flat in wavelength): the card shows the light's own colour."""
    study = small("lab")
    store(study, mean=np.full((H, W, len(study.grid)), 0.3))
    s = Session(study)
    s.choices["lamp"] = Choice("hps_cie_hp1", 0)
    xyz = s.xyz()
    card = next(x for x in study.shapes if x.id == "white card").centre
    px, py = (int(round(v)) for v in image.project(study.camera, card))
    r, g, b = s.picture(xyz, "daylight")[py, px]
    assert r > g > b and r - b >= 0.15
    assert np.ptp(s.picture(xyz, "card")[py, px]) <= 0.02
    with pytest.raises(ValueError, match="display mode"):
        s.picture(xyz, "film")
