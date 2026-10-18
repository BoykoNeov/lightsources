"""EmissionProfile: what every lamp model produces (design §2).

The renderer never sees a lamp, only zones: fixed grey spatial/angular shapes,
each with its own spectrum. Most lamps have one shape, ``isotropic``. LED
Level 2 gives emission-angle bands, ``polar <lo>-<hi> deg`` (polar angle inside
its phosphor layer); the renderer gives these shapes their geometry, and adds
the rest (bands in impact parameter and angle, bands along a filament).
"""

import re
from dataclasses import dataclass, field

from lss.spectral.spectrum import Spectrum

_POLAR = re.compile(r"polar (\d+(?:\.\d+)?)-(\d+(?:\.\d+)?) deg")


def polar_band(shape: str) -> tuple[float, float] | None:
    """A shape's band of polar angle in degrees: ``isotropic`` is 0–90°, ``polar <lo>-<hi> deg``
    its own band; None for any other shape."""
    if shape == "isotropic":
        return (0.0, 90.0)
    m = _POLAR.fullmatch(shape)
    if m is None:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    if not 0.0 <= lo < hi <= 90.0:
        raise ValueError(f"angle band {shape!r} is not inside 0-90 deg")
    return lo, hi

#: Components a profile may already contain, so they are never applied twice.
COMPONENTS = frozenset({"phosphor", "envelope", "reflector", "luminaire"})


@dataclass(frozen=True)
class Zone:
    spectrum: Spectrum
    shape: str = "isotropic"
    name: str = ""


@dataclass(frozen=True)
class EmissionProfile:
    zones: tuple[Zone, ...]
    #: Total radiant flux over all wavelengths in W, from the generator's energy
    #: budget. None when the model has no absolute scale (relative tables,
    #: a Planck curve without filament geometry).
    power: float | None = None
    includes: frozenset[str] = frozenset()
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        unknown = set(self.includes) - COMPONENTS
        if unknown:
            raise ValueError(f"unknown components in includes: {sorted(unknown)}")
        if not self.zones:
            raise ValueError("an emission profile needs at least one zone")

    @property
    def empirical_count(self) -> int:
        """Empirical parameters this model actually used (from ``metadata["params"]``,
        which every generator fills with the parameters it read)."""
        return sum(bool(p.get("empirical")) for p in self.metadata.get("params", {}).values())

    @property
    def spectrum(self) -> Spectrum:
        """Zone-summed spectrum."""
        total = self.zones[0].spectrum
        for z in self.zones[1:]:
            total = total + z.spectrum
        return total
