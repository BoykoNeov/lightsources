"""EmissionProfile: what every lamp model produces (design §2).

The renderer never sees a lamp, only zones: fixed grey spatial/angular shapes,
each with its own spectrum. Most lamps have one shape, ``isotropic``. LED
Level 2 gives emission-angle bands, ``polar <lo>-<hi> deg`` (polar angle inside
its phosphor layer); the renderer gives these shapes their geometry, and adds
the rest (bands in impact parameter and angle, bands along a filament).
"""

from dataclasses import dataclass, field

from lss.spectral.spectrum import Spectrum

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
