"""Lighting maps (design §9.1, §9.3, §8.4): one per lamp, rendered with that lamp alone
emitting a flat unit spectrum.

A map is the running mean of independent passes, kept per pixel and render bin as the
mean radiance per unit lamp radiance (per nm, ``mitsuba_scene.read_film``), with
Welford's M2 over the pass means for the noise. Maps are stored outside the repository,
in ``cache_dir()/maps/<study geometry hash>/``, as ``<lamp>.mean.npy``, ``<lamp>.m2.npy``
and ``<lamp>.json``; the hash leaves out the lamps' spectra and fluxes, so a physics
edit re-colours the stored maps without a render.

For each map the render also reports the luminance-weighted relative standard error of
the map re-coloured under the study's own spectrum (Σ se(Y) / Σ Y over the image, from
each pass's re-coloured luminance), since a flat-spectrum noise flatters a line lamp (§8.3).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lss.download import cache_dir
from lss.render import recolour
from lss.render.lamps import PlacedLamp
from lss.scene.study import Study
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import BinnedSpectrum


@dataclass(frozen=True)
class MapInfo:
    lamp_id: str
    passes: int
    spp_per_pass: int
    variant: str
    seconds: float
    noise_under_study_lamp: float  # luminance-weighted relative standard error


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


def summed(placed: PlacedLamp) -> BinnedSpectrum:
    """The lamp's zones summed (they share one map: one shape on one sphere)."""
    g = analysis_grid()
    vals = sum(z.radiance.values for z in placed.zones)
    return BinnedSpectrum(g, vals, 0.0, 0.0, placed.zones[0].radiance.quantity)


def luminance_noise(se_y: np.ndarray, y: np.ndarray) -> float:
    total = float(np.sum(y))
    return float(np.sum(se_y)) / total if total > 0 else float("inf")


def folder(study: Study) -> Path:
    return cache_dir() / "maps" / study.geometry_hash()


def save(study: Study, m: LightingMap) -> Path:
    d = folder(study)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{m.info.lamp_id}.mean.npy", m.mean.astype(np.float32))
    np.save(d / f"{m.info.lamp_id}.m2.npy", m.m2.astype(np.float32))
    (d / f"{m.info.lamp_id}.json").write_text(json.dumps(asdict(m.info), indent=1), encoding="utf-8")
    return d


def load(study: Study, lamp_id: str) -> LightingMap | None:
    d = folder(study)
    meta = d / f"{lamp_id}.json"
    if not meta.exists():
        return None
    info = MapInfo(**json.loads(meta.read_text(encoding="utf-8")))
    mean = np.load(d / f"{lamp_id}.mean.npy", mmap_mode="r")
    m2 = np.load(d / f"{lamp_id}.m2.npy", mmap_mode="r")
    return LightingMap(info, mean, m2)


def render(study: Study, placed: list[PlacedLamp], spp_per_pass: int, passes: int, variant: str,
           scene=None, progress=None) -> list[LightingMap]:
    """Render every lamp's map in one loaded scene, switching the emitting lamp in place."""
    import mitsuba as mi

    from lss.render import mitsuba_scene as ms

    if passes < 2:
        raise ValueError("at least two passes, for the noise")
    grid = study.grid
    sc = scene if scene is not None else mi.load_dict(ms.scene_dict(study))
    params = mi.traverse(sc)
    keys = [ms.emitter_key(p.id) for p in study.lamps]
    out = []
    for p, pl in zip(study.lamps, placed, strict=True):
        ms.set_emitting(params, keys, ms.emitter_key(p.id))
        wy = recolour.luminance_weights(summed(pl), grid)
        acc, acc_y = Welford(), Welford()
        t0 = time.perf_counter()
        for i in range(passes):
            m = ms.read_film(mi.render(sc, spp=spp_per_pass, seed=i), grid)
            acc.add(m)
            acc_y.add(m @ wy)
            if progress:
                progress(p.id, i + 1, passes)
        info = MapInfo(p.id, passes, spp_per_pass, variant, round(time.perf_counter() - t0, 3),
                       luminance_noise(acc_y.standard_error(), acc_y.mean))
        out.append(LightingMap(info, acc.mean, acc.m2))
    return out
