"""A placed lamp's zone radiances (design §2, §8.4). No Mitsuba needed.

The lamp's profile passes through its quick path (the envelope as a spectral factor,
as on the lamp sheet: the envelope is not yet geometry). Each ``isotropic`` zone becomes
uniform radiance on the study's sphere, and the zones are scaled together so that the
lamp emits the study's luminous flux Φ_v:

    L_k(λ) = S_k(λ) · Φ_v / (K_m ∫ Σ_k S_k V dλ · π · 4πr²)

(a Lambertian sphere of radiance L emits π·L·4πr²). The brightness is therefore the
study's setting, not the lamp's own output, even where a model has one.
"""

import math
from dataclasses import dataclass

from lss.lamp import lamp as lamps
from lss.lamp import quickpath
from lss.scene.study import LampPlacement
from lss.spectral import colorimetry
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import BinnedSpectrum


@dataclass(frozen=True)
class PlacedZone:
    lamp_id: str
    zone: str  # zone name, or its index
    radiance: BinnedSpectrum  # W m-2 sr-1 per 1 nm analysis bin


@dataclass(frozen=True)
class PlacedLamp:
    placement: LampPlacement
    level: int
    zones: tuple[PlacedZone, ...]
    scale: float  # factor applied to the profile's own zone spectra
    profile_power: float | None  # the model's own radiant flux, unused (see module note)
    note: str


def place(p: LampPlacement) -> PlacedLamp:
    lamp = lamps.load(p.lamp)
    level = p.level if p.level is not None else max(lamp.models)
    profile = quickpath.apply(lamp, lamp.generate(level))
    shapes = {z.shape for z in profile.zones}
    if shapes != {"isotropic"}:
        raise ValueError(f"{p.id}: zone shapes {sorted(shapes)}; the renderer takes only 'isotropic' so far")
    grid = analysis_grid()
    binned = [z.spectrum.bin(grid) for z in profile.zones]
    total = binned[0]
    for b in binned[1:]:
        total = BinnedSpectrum(grid, total.values + b.values, 0.0, 0.0, total.quantity)
    lm_per_unit = colorimetry.luminous(total)
    if lm_per_unit <= 0:
        raise ValueError(f"{p.id}: the lamp emits nothing visible")
    area = 4 * math.pi * p.radius**2
    scale = p.flux_lm / (lm_per_unit * math.pi * area)
    zones = tuple(
        PlacedZone(p.id, z.name or str(k),
                   BinnedSpectrum(grid, b.values * scale, b.below * scale, b.above * scale, "W m-2 sr-1 nm-1"))
        for k, (z, b) in enumerate(zip(profile.zones, binned)))
    note = (f"{p.id}: {lamp.id} level {level}, scaled to {p.flux_lm:g} lm by the study, "
            f"not the lamp's own output")
    return PlacedLamp(p, level, zones, scale, profile.power, note)
