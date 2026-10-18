"""Studies: a fixed camera, scene and lamp placements (design §9.2, §8.4).

A study is a YAML file in ``lss/data/studies/`` (or a path). It names the scene
(so far only ``reference-lab``), the camera, the render settings and the lamps.
No preset has an emitting surface yet, so the study places each lamp as a sphere
(``isotropic`` zones only) or as a disc emitting from its front (``isotropic`` and
``polar`` angle-band zones, §8.5), and gives its luminous flux: the renderer scales
every lamp to it, so a render's brightness is the study's setting, not the lamp's
own output.

Nothing here imports Mitsuba; ``lss.render`` turns a study into a scene.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from lss.scene import lab
from lss.scene.geometry import Shape
from lss.scene.materials import Material
from lss.spectral.grids import BinGrid, render_grid


@dataclass(frozen=True)
class Camera:
    origin: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_deg: float = 50.0  # horizontal
    width: int = 512
    height: int = 384


@dataclass(frozen=True)
class LampPlacement:
    id: str
    lamp: str  # preset id, YAML path or ref:<id>
    level: int | None  # None: the lamp's highest
    flux_lm: float  # the study's normalisation (§6: equal luminous flux)
    centre: tuple[float, float, float]
    radius: float  # m
    kind: str = "sphere"  # sphere | disc
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)  # disc: the side it emits from
    #: Ask the generator for this many equal angle bands (the zone-count knob, §8.5);
    #: None keeps its own.
    angle_bands: int | None = None

    def __post_init__(self):
        if self.kind not in ("sphere", "disc"):
            raise ValueError(f"{self.id}: lamp kind {self.kind!r}; 'sphere' or 'disc'")
        if self.radius <= 0:
            raise ValueError(f"{self.id}: radius must be positive")
        if self.kind == "disc" and sum(c * c for c in self.normal) == 0:
            raise ValueError(f"{self.id}: a disc needs a normal")

    @property
    def area(self) -> float:
        """Emitting area: a sphere's whole surface, a disc's front."""
        return (4.0 if self.kind == "sphere" else 1.0) * math.pi * self.radius**2


@dataclass(frozen=True)
class Study:
    id: str
    description: str
    scene: str
    camera: Camera
    lamps: tuple[LampPlacement, ...]
    bin_nm: float = 10.0
    max_depth: int = 8
    shapes: tuple[Shape, ...] = field(default=(), compare=False)
    materials: dict[str, Material] = field(default_factory=dict, compare=False)

    @property
    def grid(self) -> BinGrid:
        return render_grid(self.bin_nm)

    def geometry_hash(self) -> str:
        """Hash of everything the lighting maps depend on: not the lamps' spectra or fluxes."""
        doc = {
            "scene": self.scene, "camera": asdict(self.camera), "bin_nm": self.bin_nm,
            "max_depth": self.max_depth,
            # Not the angle bands: a band's map does not depend on the others (§8.5).
            # A sphere's entry is as it was before discs, so stored maps stay valid.
            "lamps": [(p.id, p.centre, p.radius) + (("disc", p.normal) if p.kind == "disc" else ())
                      for p in self.lamps],
            "shapes": [s.as_dict() for s in self.shapes],
            "materials": {k: m.digest() for k, m in sorted(self.materials.items())},
        }
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def _vec(v) -> tuple[float, float, float]:
    x = tuple(float(a) for a in v)
    if len(x) != 3:
        raise ValueError(f"expected three coordinates, got {v!r}")
    return x


def study_from_dict(doc: dict) -> Study:
    if doc.get("scene") != "reference-lab":
        raise ValueError(f"unknown scene {doc.get('scene')!r}; only 'reference-lab' so far")
    cam = doc["camera"]
    camera = Camera(_vec(cam["origin"]), _vec(cam["target"]), _vec(cam.get("up", (0, 0, 1))),
                    float(cam.get("fov_deg", 50.0)), int(cam.get("width", 512)), int(cam.get("height", 384)))
    lamps = []
    for d in doc["lamps"]:
        kinds = [k for k in ("sphere", "disc") if k in d]
        if len(kinds) != 1:
            raise ValueError(f"{d.get('id')}: give the lamp one of 'sphere' or 'disc'")
        g = d[kinds[0]]
        bands = d.get("angle_bands")
        lamps.append(LampPlacement(d["id"], d["lamp"], d.get("level"), float(d["flux_lm"]),
                                   _vec(g["centre"]), float(g["radius"]), kinds[0],
                                   _vec(g["normal"]) if kinds[0] == "disc" else (0.0, 0.0, 1.0),
                                   int(bands) if bands is not None else None))
    if len({p.id for p in lamps}) != len(lamps):
        raise ValueError("lamp ids in a study must be distinct")
    render = doc.get("render") or {}
    shapes, materials = lab.reference_lab()
    return Study(doc["id"], doc.get("description", ""), doc["scene"], camera, tuple(lamps),
                 float(render.get("bin_nm", 10.0)), int(render.get("max_depth", 8)), shapes, materials)


def load(name: str) -> Study:
    """A study by id (``lss/data/studies/<id>.yaml``) or by YAML path."""
    path = Path(name)
    if path.suffix in (".yaml", ".yml") and path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        res = resources.files("lss.data").joinpath("studies", f"{name}.yaml")
        if not res.is_file():
            raise KeyError(f"no study {name!r}")
        text = res.read_text(encoding="utf-8")
    return study_from_dict(yaml.safe_load(text))
