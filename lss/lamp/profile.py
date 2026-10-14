"""EmissionProfile: what every lamp model produces (design §2).

The renderer never sees a lamp, only zones: fixed grey spatial/angular shapes,
each with its own spectrum. Step 1 has only one shape, ``isotropic``; real
zone shapes (bands in impact parameter and angle, emission-angle bands, bands
along a filament) arrive with the renderer.
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
