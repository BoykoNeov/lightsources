"""Lamp presets: one physical description shared by every model level (design §4, §5).

A preset is a YAML file in ``lss/data/lamps/``. It names the lamp's family,
its physical description (parameters with units and sources), its envelope if
it has one, the reference it is scored against, and the model levels available
for it.
"""

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from lss.lamp.envelope import Envelope
from lss.lamp.profile import EmissionProfile
from lss.params import Parameter


@dataclass(frozen=True)
class ModelSpec:
    level: int
    generator: str
    params: dict[str, Parameter] = field(default_factory=dict)


@dataclass(frozen=True)
class Lamp:
    id: str
    name: str
    family: str
    physical: dict[str, Parameter]
    reference: str | None
    models: dict[int, ModelSpec]
    envelope: Envelope | None = None  # applied after the generator (design §5)

    def generate(self, level: int) -> EmissionProfile:
        from lss.lamp.generators import run

        if level not in self.models:
            raise KeyError(f"{self.id} has no level {level}; available: {sorted(self.models)}")
        return run(self, self.models[level])

    def parameter(self, name: str, spec: ModelSpec | None = None) -> Parameter:
        """A model's own parameter if it sets one, else the physical description's, else
        the shared phosphor library's (``lss.lamp.phosphors``)."""
        if spec is not None and name in spec.params:
            return spec.params[name]
        if name in self.physical:
            return self.physical[name]
        from lss.lamp import phosphors

        if phosphors.is_shared(name):
            return phosphors.parameters()[name]
        raise KeyError(f"{self.id}: parameter {name!r} is not in the physical description")


def _params(d: dict | None) -> dict[str, Parameter]:
    return {k: Parameter.from_dict(v) for k, v in (d or {}).items()}


def lamp_from_dict(doc: dict) -> Lamp:
    models = {}
    for level, m in (doc.get("models") or {}).items():
        level = int(level)
        models[level] = ModelSpec(level, m["generator"], _params(m.get("params")))
    return Lamp(
        id=doc["id"],
        name=doc.get("name", doc["id"]),
        family=doc["family"],
        physical=_params(doc.get("physical")),
        reference=doc.get("reference"),
        models=models,
        envelope=Envelope.from_dict(doc["envelope"]) if doc.get("envelope") else None,
    )


def preset_ids() -> list[str]:
    folder = resources.files("lss.data").joinpath("lamps")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".yaml"))


def load(name: str) -> Lamp:
    """Load a preset by id, a YAML path, or ``ref:<reference id>`` for a Level-0-only lamp."""
    if name.startswith("ref:"):
        from lss.accuracy import references

        ref = references.get(name[4:])
        return Lamp(ref.id, ref.description, ref.family, {}, ref.id, {0: ModelSpec(0, "reference")})
    path = Path(name)
    if path.suffix in (".yaml", ".yml") and path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        res = resources.files("lss.data").joinpath("lamps", f"{name}.yaml")
        if not res.is_file():
            raise KeyError(f"no lamp preset {name!r}; presets: {', '.join(preset_ids())}")
        text = res.read_text(encoding="utf-8")
    return lamp_from_dict(yaml.safe_load(text))
