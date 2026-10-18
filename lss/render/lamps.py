"""A placed lamp's zone radiances (design §2, §8.4, §8.5). No Mitsuba needed.

The lamp's profile passes through its quick path (the envelope as a spectral factor,
as on the lamp sheet: the envelope is not yet geometry). The study places the lamp as
a sphere or a disc, and each zone becomes uniform radiance on it:

- an ``isotropic`` zone, the same in every direction (a Lambertian surface);
- a ``polar <lo>-<hi> deg`` zone (LED Level 2, on a disc only), radiance in the directions
  whose angle to the disc's normal is inside the band, none outside it.

A band's spectrum is the power the generator sends into it, P_k; a Lambertian surface
sends a share sin²θ_hi − sin²θ_lo of its power into a band, so the band's radiance is
L_k = c·P_k / (sin²θ_hi − sin²θ_lo), and an emitter of area A emits π·A·c·Σ_k P_k in
all. The zones are scaled together so that the lamp emits the study's luminous flux Φ_v:

    c = Φ_v / (K_m ∫ Σ_k P_k V dλ · π · A)

(A = 4πr² for a sphere, πr² for a disc; an isotropic zone is one band, 0–90°). The
brightness is therefore the study's setting, not the lamp's own output, even where a
model has one.
"""

import math
from dataclasses import dataclass

from lss.lamp import lamp as lamps
from lss.lamp import quickpath
from lss.lamp.profile import EmissionProfile, polar_band
from lss.scene.study import LampPlacement
from lss.spectral import colorimetry
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import BinnedSpectrum


@dataclass(frozen=True)
class PlacedZone:
    lamp_id: str
    zone: str  # zone name, or its index
    shape: str  # isotropic | polar <lo>-<hi> deg
    radiance: BinnedSpectrum  # W m-2 sr-1 per 1 nm analysis bin


@dataclass(frozen=True)
class PlacedLamp:
    placement: LampPlacement
    level: int
    zones: tuple[PlacedZone, ...]
    scale: float  # factor c applied to the profile's own zone spectra
    profile_power: float | None  # the model's own radiant flux, unused (see module note)
    note: str


def band_share(shape: str) -> float:
    """Share of a Lambertian surface's power sent into the zone's band: sin²θ_hi − sin²θ_lo."""
    lo, hi = polar_band(shape)
    return math.sin(math.radians(hi)) ** 2 - math.sin(math.radians(lo)) ** 2


def place(p: LampPlacement) -> PlacedLamp:
    lamp = lamps.load(p.lamp)
    level = p.level if p.level is not None else max(lamp.models)
    profile = quickpath.apply(lamp, lamp.generate(level, p.angle_bands))
    return place_profile(p, profile, level, lamp.id)


def place_profile(p: LampPlacement, profile: EmissionProfile, level: int, lamp_id: str) -> PlacedLamp:
    for z in profile.zones:
        if polar_band(z.shape) is None:
            raise ValueError(f"{p.id}: zone shape {z.shape!r}; the renderer takes 'isotropic' and 'polar' bands so far")
        if p.kind == "sphere" and z.shape != "isotropic":
            raise ValueError(f"{p.id}: a sphere takes only 'isotropic' zones, not {z.shape!r}; place it as a disc")
    grid = analysis_grid()
    binned = [z.spectrum.bin(grid) for z in profile.zones]
    total = binned[0]
    for b in binned[1:]:
        total = BinnedSpectrum(grid, total.values + b.values, 0.0, 0.0, total.quantity)
    lm_per_unit = colorimetry.luminous(total)
    if lm_per_unit <= 0:
        raise ValueError(f"{p.id}: the lamp emits nothing visible")
    scale = p.flux_lm / (lm_per_unit * math.pi * p.area)
    zones = []
    for k, (z, b) in enumerate(zip(profile.zones, binned)):
        f = scale / band_share(z.shape)
        zones.append(PlacedZone(p.id, z.name or str(k), z.shape,
                                BinnedSpectrum(grid, b.values * f, b.below * f, b.above * f, "W m-2 sr-1 nm-1")))
    note = (f"{p.id}: {lamp_id} level {level}, scaled to {p.flux_lm:g} lm by the study, "
            f"not the lamp's own output")
    return PlacedLamp(p, level, tuple(zones), scale, profile.power, note)


def flux_lm(placed: PlacedLamp) -> float:
    """Luminous flux the placed zones emit: Σ_k π·A·share_k·K_m∫L_k V dλ (the study's Φ_v)."""
    a = placed.placement.area
    return sum(math.pi * a * band_share(z.shape) * colorimetry.luminous(z.radiance) for z in placed.zones)


def map_zones(placed: PlacedLamp) -> list[tuple[str, BinnedSpectrum]]:
    """One (shape, radiance) per lighting map: zones of the same shape share a map and are
    re-coloured with their summed radiance (design §8.5)."""
    out: dict[str, BinnedSpectrum] = {}
    for z in placed.zones:
        r = z.radiance
        if z.shape in out:
            s = out[z.shape]
            r = BinnedSpectrum(s.grid, s.values + r.values, s.below + r.below, s.above + r.above, s.quantity)
        out[z.shape] = r
    return list(out.items())
