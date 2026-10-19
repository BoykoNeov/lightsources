"""The viewer's state and arithmetic, without Qt (design §9.7). No Mitsuba needed.

A session holds a study, its lighting maps in memory (copied out of the stored files,
which are then closed, so the render worker can replace them), and for each placed lamp a
choice of preset, level and dimming. From these it gives the XYZ image (§9.1's sum, with
``recolour``'s weights), the display picture, the probe, and the maps the choice lacks.
The window draws what it returns.

Every lamp is scaled to the study's luminous flux (``lss.render.lamps``) and then dimmed:
the brightness is the study's setting, not the lamp's own output, and the window says so.
"""

import threading
from dataclasses import dataclass, replace

import numpy as np

from lss.lamp import lamp as lamps
from lss.lamp import quickpath
from lss.lamp.profile import EmissionProfile
from lss.render import image, maps, recolour
from lss.render.lamps import PlacedLamp, map_zones, place_profile
from lss.scene.study import Study
from lss.spectral import colorimetry
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import BinnedSpectrum

WHITE_CARD = "white card"
DISPLAY_MODES = {"card": "white card: balanced and exposed on the card, as lss render",
                 "daylight": "daylight display: no white balance (sRGB's D65 white), exposed on the card"}


@dataclass(frozen=True)
class Choice:
    lamp: str  # preset id, YAML path or ref:<id>
    level: int
    dimming: float = 1.0  # fraction of the study's luminous flux


@dataclass(frozen=True)
class Probe:
    x: int
    y: int
    half_px: int
    wavelengths: np.ndarray  # nm, the analysis grid's centres
    radiance: np.ndarray  # W m-2 sr-1 nm-1, mean over the square
    se: np.ndarray  # its standard error, from the maps' pass-to-pass scatter
    xyz: np.ndarray  # Σ radiance · cmf over 1 nm bins: the image's units
    luminance_cd_m2: float
    xy: tuple[float, float]
    cct_k: float
    duv: float


class MissingMaps(Exception):
    def __init__(self, missing: list[str]):
        super().__init__("needs render: " + ", ".join(missing))
        self.missing = missing


def default_level(preset: str) -> int:
    return max(lamps.load(preset).models)


def levels(preset: str) -> list[int]:
    return sorted(lamps.load(preset).models)


class Session:
    def __init__(self, study: Study):
        self.study = study
        self.choices = {p.id: Choice(p.lamp, p.level if p.level is not None else default_level(p.lamp))
                        for p in study.lamps}
        self._profiles: dict[tuple, EmissionProfile] = {}
        self._placed: dict[tuple, PlacedLamp] = {}  # by (lamp id, preset, level): no dimming
        self._lock = threading.Lock()
        self._maps: dict[tuple[str, str], maps.LightingMap | None] = {}
        self._white = next(s for s in study.shapes if s.id == WHITE_CARD)

    # --- lamps -------------------------------------------------------------------------

    def placement(self, lamp_id: str):
        return next(p for p in self.study.lamps if p.id == lamp_id)

    def _key(self, lamp_id: str, choice: Choice) -> tuple:
        return choice.lamp, choice.level, self.placement(lamp_id).angle_bands

    def has_profile(self, lamp_id: str, choice: Choice | None = None) -> bool:
        with self._lock:
            return self._key(lamp_id, choice or self.choices[lamp_id]) in self._profiles

    def profile(self, lamp_id: str, choice: Choice | None = None) -> EmissionProfile:
        """The choice's quick-path profile; runs the generator the first time (slow for some
        levels: call it off the window's thread)."""
        choice = choice or self.choices[lamp_id]
        key = self._key(lamp_id, choice)
        with self._lock:
            if key in self._profiles:
                return self._profiles[key]
        lamp = lamps.load(choice.lamp)
        prof = quickpath.apply(lamp, lamp.generate(choice.level, key[2]))
        with self._lock:
            self._profiles[key] = prof
        return prof

    def placed(self, lamp_id: str) -> PlacedLamp:
        """The current choice placed at the study's flux (dimming is applied when re-colouring)."""
        c = self.choices[lamp_id]
        key = (lamp_id, c.lamp, c.level)
        if key not in self._placed:
            p = replace(self.placement(lamp_id), lamp=c.lamp, level=c.level)
            self._placed[key] = place_profile(p, self.profile(lamp_id), c.level, lamps.load(c.lamp).id)
        return self._placed[key]

    def normalisation(self) -> str:
        parts = [f"{p.id}: scaled to the study's {p.flux_lm:g} lm (equal luminous flux), "
                 f"dimmed to {100 * self.choices[p.id].dimming:g} %" for p in self.study.lamps]
        return "; ".join(parts) + ". Not the lamps' own output."

    # --- maps --------------------------------------------------------------------------

    def lighting_map(self, lamp_id: str, shape: str) -> maps.LightingMap | None:
        key = (lamp_id, shape)
        if key not in self._maps:
            self._maps[key] = maps.load(self.study, lamp_id, shape, mmap=False)
        return self._maps[key]

    def forget_map(self, lamp_id: str, shape: str) -> None:
        """Drop a map held in memory, so the next use reads the stored one again."""
        self._maps.pop((lamp_id, shape), None)

    def jobs(self) -> list[tuple[PlacedLamp, str, BinnedSpectrum]]:
        return maps.jobs([self.placed(p.id) for p in self.study.lamps])

    def missing(self) -> list[tuple[PlacedLamp, str, BinnedSpectrum]]:
        return [j for j in self.jobs() if self.lighting_map(j[0].placement.id, j[1]) is None]

    def noise_notes(self) -> list[str]:
        """Each map's noise and the lamp it was measured under; 'not measured' under any
        other spectrum (§8.3: a flat-spectrum noise flatters a line lamp)."""
        out = []
        for pl, shape, _ in self.jobs():
            m = self.lighting_map(pl.placement.id, shape)
            if m is None:
                out.append(f"{maps.map_id(pl.placement.id, shape)}: no stored map")
                continue
            i, label = m.info, maps.noise_label(pl, shape)
            measured = f"{100 * i.noise_under_study_lamp:.2f} % under {i.noise_lamp or 'an unrecorded lamp'}"
            note = measured if i.noise_lamp == label else f"not measured under {label} ({measured})"
            out.append(f"{i.map_id}: {i.passes} x {i.spp_per_pass} spp on {i.variant}; noise {note}")
        return out

    # --- the image ---------------------------------------------------------------------

    def _terms(self):
        """(map, weights) for every map of every lamp; the weights include the dimming."""
        terms, missing = [], []
        for pl, shape, radiance in self.jobs():
            lamp_id = pl.placement.id
            m = self.lighting_map(lamp_id, shape)
            if m is None:
                missing.append(maps.map_id(lamp_id, shape))
                continue
            w = recolour.weights(radiance, self.study.grid) * self.choices[lamp_id].dimming
            terms.append((m, w, radiance, self.choices[lamp_id].dimming))
        if missing:
            raise MissingMaps(missing)
        return terms

    def xyz(self) -> np.ndarray:
        """XYZ image (h, w, 3): Σ_k map_k · W_k, summed in float64 (16 ms a map at 512 × 384)."""
        out = None
        for m, w, _, _ in self._terms():
            x = m.mean @ w
            out = x if out is None else out + x
        return out

    def white_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return image.region_mean(xyz, self.study.camera, self._white.centre)

    def picture(self, xyz: np.ndarray, mode: str = "card", stops: float = 0.0) -> np.ndarray:
        """Display sRGB (h, w, 3), 0–1 encoded; ``mode`` one of ``DISPLAY_MODES``."""
        if mode not in DISPLAY_MODES:
            raise ValueError(f"display mode {mode!r}; one of {', '.join(DISPLAY_MODES)}")
        return image.to_srgb(xyz * 2.0**stops, self.white_xyz(xyz), balance=mode == "card")

    def probe(self, x: int, y: int, half_px: int = 2) -> Probe:
        """The spectrum arriving at pixel (x, y), averaged over a (2·half_px + 1)² square:
        L(λ_i) = Σ_k map_k(bin of λ_i) · S_k(λ_i). Its tristimulus values are the image's."""
        cam = self.study.camera
        if not (0 <= x < cam.width and 0 <= y < cam.height):
            raise ValueError(f"pixel ({x}, {y}) is outside the {cam.width} x {cam.height} image")
        y0, y1 = max(0, y - half_px), min(cam.height, y + half_px + 1)
        x0, x1 = max(0, x - half_px), min(cam.width, x + half_px + 1)
        n_px = (y1 - y0) * (x1 - x0)
        grid = analysis_grid()
        idx = recolour.bin_index(self.study.grid)
        inside = idx >= 0
        radiance, var = np.zeros(len(grid)), np.zeros(len(grid))
        for m, _, s, dim in self._terms():
            mean = np.asarray(m.mean[y0:y1, x0:x1], dtype=np.float64).reshape(-1, len(self.study.grid)).mean(axis=0)
            n = m.info.passes
            v_px = np.asarray(m.m2[y0:y1, x0:x1], dtype=np.float64) / (n * (n - 1)) if n > 1 else np.inf
            v = v_px.reshape(-1, len(self.study.grid)).sum(axis=0) / n_px**2  # pixels independent
            radiance[inside] += dim * mean[idx[inside]] * s.values[inside]
            var[inside] += (dim * s.values[inside]) ** 2 * v[idx[inside]]  # maps independent
        binned = BinnedSpectrum(grid, radiance, 0.0, 0.0, "W m-2 sr-1 nm-1")
        xyz = colorimetry.tristimulus(binned)
        if xyz.sum() > 0:
            cct, duv = colorimetry.cct_duv(binned)
            xy = (float(xyz[0] / xyz.sum()), float(xyz[1] / xyz.sum()))
        else:
            cct, duv, xy = float("nan"), float("nan"), (float("nan"), float("nan"))
        return Probe(x, y, half_px, grid.centres, radiance, np.sqrt(var), xyz,
                     colorimetry.luminous(binned), xy, cct, duv)
