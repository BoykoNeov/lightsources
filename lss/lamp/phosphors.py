"""Shared phosphor library (``lss/data/phosphors.yaml``, design §4.3 L1).

A phosphor's emission shape belongs to the phosphor, not to a lamp: every lamp
using the rare-earth tri-phosphor blend shares the same bands and differs only
in how much of each phosphor it has. A lamp reads a library parameter by its
full name, ``<phosphor>.<band>.<field>`` (e.g. ``LAP.7F6.share``), through
:meth:`lss.lamp.lamp.Lamp.parameter`; a value the lamp sets itself overrides it,
which is how a fit tries new shapes without editing the library.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import yaml

from lss.params import Parameter

LIBRARY_PATH = "lss/data/phosphors.yaml"
SHAPES = {"symmetric": ("peak", "fwhm"), "split": ("peak", "half_width_short", "half_width_long")}


@dataclass(frozen=True)
class Band:
    name: str
    shape: str  # symmetric | split

    @property
    def fields(self) -> tuple[str, ...]:
        return SHAPES[self.shape]


@dataclass(frozen=True)
class Phosphor:
    id: str
    name: str
    main: str
    bands: tuple[Band, ...]


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, Phosphor], dict[str, Parameter]]:
    doc = yaml.safe_load(resources.files("lss.data").joinpath("phosphors.yaml").read_text(encoding="utf-8"))
    phosphors, params = {}, {}
    for pid, ph in doc["phosphors"].items():
        if "." in pid:
            raise ValueError(f"phosphor id {pid!r} may not contain '.'")
        bands = []
        for bname, b in ph["bands"].items():
            if b["shape"] not in SHAPES:
                raise ValueError(f"{pid}.{bname}: unknown shape {b['shape']!r}")
            band = Band(bname, b["shape"])
            fields = band.fields + (() if bname == ph["main"] else ("share",))
            for f in fields:
                params[f"{pid}.{bname}.{f}"] = Parameter.from_dict(b[f])
            if bname == ph["main"] and "share" in b:
                raise ValueError(f"{pid}.{bname} is the main band; its share is 1 by definition")
            bands.append(band)
        if ph["main"] not in ph["bands"]:
            raise ValueError(f"{pid}: main band {ph['main']!r} is not one of its bands")
        phosphors[pid] = Phosphor(pid, ph["name"], ph["main"], tuple(bands))
    return phosphors, params


def library() -> dict[str, Phosphor]:
    return _load()[0]


def parameters() -> dict[str, Parameter]:
    """Every library parameter by full name."""
    return _load()[1]


def is_shared(name: str) -> bool:
    """True for a parameter that belongs to the library (shared by lamps), not to one lamp."""
    return name in parameters()


def emission(phosphor_id: str, read: Callable[[str], Parameter], power: float) -> list:
    """The phosphor's bands carrying ``power`` in total, split by their shares.

    ``read`` resolves a full parameter name (normally ``lamp.parameter``), so a lamp's
    own value overrides the library's.
    """
    from lss.physics.led.bands import EnergyBand

    ph = library()[phosphor_id]
    shares, shapes = [], []
    for band in ph.bands:
        key = f"{ph.id}.{band.name}"
        values = []
        for f in band.fields:
            p = read(f"{key}.{f}")
            if p.unit != "nm":
                raise ValueError(f"{key}.{f} must be in nm")
            values.append(p.value)
        shapes.append((band.shape, values))
        if band.name == ph.main:
            shares.append(1.0)
        else:
            s = read(f"{key}.share")
            if s.unit != "1":
                raise ValueError(f"{key}.share is a power ratio, unit 1")
            shares.append(s.value)
    total = sum(shares)
    out = []
    for (shape, v), share in zip(shapes, shares):
        w = power * share / total
        out.append(EnergyBand.symmetric(*v, power=w) if shape == "symmetric" else EnergyBand.split(*v, power=w))
    return out
