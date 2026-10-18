"""Lighting maps (design §9.1, §9.3, §8.4, §8.5): one per lamp and zone shape, rendered
with that lamp alone emitting a flat unit spectrum in that zone.

A map is the running mean of independent passes, kept per pixel and render bin as the
mean radiance per unit zone radiance (per nm, ``mitsuba_scene.read_film``), with
Welford's M2 over the pass means for the noise. Maps are stored outside the repository,
in ``cache_dir()/maps/<study geometry hash>/``, as ``<map id>.mean.npy``, ``.m2.npy`` and
``.json``, the map id being ``<lamp>@<zone>`` (``lamp@isotropic``, ``lamp@polar-0-30``).
The hash leaves out the lamps' spectra and fluxes, so a physics edit re-colours the
stored maps without a render; it also leaves out a disc's band edges, since one band's
map does not depend on the others (the emitter is black): a band whose edges recur in
another layout reuses its map (equal splits into different counts share none, §8.5).

For each map the render also reports the luminance-weighted relative standard error of
the map re-coloured under its zone's own spectrum in the study (Σ se(Y) / Σ Y over the
image, from each pass's re-coloured luminance), since a flat-spectrum noise flatters a
line lamp (§8.3). That number belongs to the spectrum it was measured under
(``MapInfo.noise_lamp``); maps reused for another lamp do not carry its noise.
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lss.download import cache_dir
from lss.render import recolour
from lss.render.lamps import PlacedLamp, map_zones
from lss.scene.study import Study
from lss.spectral.spectrum import BinnedSpectrum


@dataclass(frozen=True)
class MapInfo:
    lamp_id: str
    passes: int
    spp_per_pass: int
    variant: str
    seconds: float
    noise_under_study_lamp: float  # luminance-weighted relative standard error
    #: The spectrum that noise was measured under ("<lamp> level <n>"). Maps are reused
    #: for any lamp; under another spectrum their noise is not known (§8.3).
    noise_lamp: str = ""
    zone: str = "isotropic"  # the zone shape the map was rendered for

    @property
    def map_id(self) -> str:
        return map_id(self.lamp_id, self.zone)


@dataclass
class LightingMap:
    info: MapInfo
    mean: np.ndarray  # (h, w, bins), per nm
    m2: np.ndarray  # Welford's M2 over pass means, same shape

    def variance_of_mean(self) -> np.ndarray:
        n = self.info.passes
        return self.m2 / (n * (n - 1)) if n > 1 else np.full_like(self.m2, np.inf)


class Welford:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, x: np.ndarray) -> None:
        self.n += 1
        if self.mean is None:
            self.mean, self.m2 = x.astype(np.float64).copy(), np.zeros_like(x, dtype=np.float64)
            return
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def standard_error(self) -> np.ndarray:
        return np.sqrt(self.m2 / (self.n * (self.n - 1))) if self.n > 1 else np.full_like(self.mean, np.inf)


def map_id(lamp_id: str, shape: str) -> str:
    """File-safe id of a lamp's map for one zone shape: ``lamp@polar-0-30``."""
    slug = re.sub(r"[^0-9A-Za-z.]+", "-", shape.replace(" deg", "")).strip("-")
    return f"{lamp_id}@{slug}"


def noise_label(placed: PlacedLamp, shape: str = "isotropic") -> str:
    base = f"{placed.placement.lamp} level {placed.level}"
    return base if shape == "isotropic" else f"{base}, {shape}"


def luminance_noise(se_y: np.ndarray, y: np.ndarray) -> float:
    total = float(np.sum(y))
    return float(np.sum(se_y)) / total if total > 0 else float("inf")


def folder(study: Study) -> Path:
    return cache_dir() / "maps" / study.geometry_hash()


def save(study: Study, m: LightingMap) -> Path:
    d = folder(study)
    d.mkdir(parents=True, exist_ok=True)
    mid = m.info.map_id
    np.save(d / f"{mid}.mean.npy", m.mean.astype(np.float32))
    np.save(d / f"{mid}.m2.npy", m.m2.astype(np.float32))
    (d / f"{mid}.json").write_text(json.dumps(asdict(m.info), indent=1), encoding="utf-8")
    return d


def load(study: Study, lamp_id: str, shape: str = "isotropic") -> LightingMap | None:
    d = folder(study)
    mid = map_id(lamp_id, shape)
    meta = d / f"{mid}.json"
    if not meta.exists() and shape == "isotropic":  # stored before maps were per zone
        mid = lamp_id
        meta = d / f"{mid}.json"
    if not meta.exists():
        return None
    info = MapInfo(**json.loads(meta.read_text(encoding="utf-8")))
    mean = np.load(d / f"{mid}.mean.npy", mmap_mode="r")
    m2 = np.load(d / f"{mid}.m2.npy", mmap_mode="r")
    return LightingMap(info, mean, m2)


def jobs(placed: list[PlacedLamp]) -> list[tuple[PlacedLamp, str, BinnedSpectrum]]:
    """Every map a study needs: (lamp, zone shape, the zone's radiance)."""
    return [(pl, shape, r) for pl in placed for shape, r in map_zones(pl)]


def render(study: Study, placed: list[PlacedLamp], spp_per_pass: int, passes: int, variant: str,
           scene=None, progress=None, only=None) -> list[LightingMap]:
    """Render every lamp's maps (one per zone shape) in one loaded scene, switching the
    emitting lamp and zone in place. ``only``: a subset of ``jobs(placed)`` to render."""
    import mitsuba as mi

    from lss.render import mitsuba_scene as ms

    if passes < 2:
        raise ValueError("at least two passes, for the noise")
    grid = study.grid
    sc = scene if scene is not None else mi.load_dict(ms.scene_dict(study))
    params = mi.traverse(sc)
    out = []
    for pl, shape, radiance in (jobs(placed) if only is None else only):
        lamp_id = pl.placement.id
        ms.set_emitting(params, study, lamp_id, shape)
        wy = recolour.luminance_weights(radiance, grid)
        acc, acc_y = Welford(), Welford()
        t0 = time.perf_counter()
        label = map_id(lamp_id, shape)
        for i in range(passes):
            m = ms.read_film(mi.render(sc, spp=spp_per_pass, seed=i), grid)
            acc.add(m)
            acc_y.add(m @ wy)
            if progress:
                progress(label, i + 1, passes)
        info = MapInfo(lamp_id, passes, spp_per_pass, variant, round(time.perf_counter() - t0, 3),
                       luminance_noise(acc_y.standard_error(), acc_y.mean), noise_label(pl, shape), shape)
        out.append(LightingMap(info, acc.mean, acc.m2))
    return out
