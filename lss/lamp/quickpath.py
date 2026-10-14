"""The quick path (design §5): a generator's bare emission passed through the lamp's
own components as spectral factors, without the renderer.

Only the envelope has a quick-path model so far. A component the profile already
includes (a reference spectrum measured through its envelope) is never applied
twice.
"""

from dataclasses import replace

from lss.lamp.lamp import Lamp
from lss.lamp.profile import COMPONENTS, EmissionProfile


def apply(lamp: Lamp, profile: EmissionProfile, components=COMPONENTS) -> EmissionProfile:
    """``profile`` with each of ``components`` that the lamp has, and the profile lacks, applied."""
    wanted = frozenset(components) - profile.includes
    applied = {}
    zones = profile.zones
    if "envelope" in wanted and lamp.envelope is not None:
        t = lamp.envelope.transmission()
        zones = tuple(replace(z, spectrum=z.spectrum.filtered(t)) for z in zones)
        applied["envelope"] = lamp.envelope.as_dict()
    if not applied:
        return profile
    metadata = {**profile.metadata, "quick_path": {**profile.metadata.get("quick_path", {}), **applied}}
    return replace(profile, zones=zones, includes=profile.includes | applied.keys(), metadata=metadata)

