"""Lamp envelopes on the quick path (design §5).

A lamp's glass envelope transmits, absorbs and reflects; it never shifts
wavelength, so the renderer traces it as geometry. The quick path (lamp sheets,
the accuracy report) instead applies it as one spectral transmission factor.

Quick-path approximations, each stated:

* **Normal incidence.** The path through the wall is its thickness. Light leaving a
  diffuse phosphor coat crosses at an angle and travels further; the instrument path
  (§9.5) measures that.
* **Absorption only.** Surface reflection (about 4 % at the outer glass–air face) is
  left out: it is flat across the visible to 0.2 %, and on a coated tube the reflected
  light goes back into the phosphor rather than being lost. Absolute flux, the only
  number it would shift, is not modelled yet. Absorption is not flat: iron in the
  glass takes a few per cent more at the red end than in the green.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from lss.params import Parameter
from lss.spectral.filters import Transmission

#: Glass data files in ``lss/data``, by material id.
MATERIALS = {"soda-lime-rubin-clear": "glass_soda_lime.yaml", "fused-silica-franta": "glass_fused_silica.yaml"}


@dataclass(frozen=True)
class Glass:
    id: str
    name: str
    k_wavelength_nm: np.ndarray
    k: np.ndarray
    n_formula: dict

    @property
    def range_nm(self) -> tuple[float, float]:
        return float(self.k_wavelength_nm[0]), float(self.k_wavelength_nm[-1])

    def n(self, wavelength_nm):
        um = np.asarray(wavelength_nm, dtype=float) / 1000.0
        f = self.n_formula
        return f["n0"] + f["a"] * um**2 + f["b"] / um**2

    def internal_transmittance(self, thickness_mm: float) -> np.ndarray:
        """exp(−4πk·d/λ) at the table's own wavelengths."""
        d_nm = thickness_mm * 1e6
        return np.exp(-4.0 * np.pi * self.k * d_nm / self.k_wavelength_nm)


@lru_cache(maxsize=None)
def glass(material: str) -> Glass:
    try:
        name = MATERIALS[material]
    except KeyError:
        raise KeyError(f"unknown envelope material {material!r}; known: {sorted(MATERIALS)}") from None
    doc = yaml.safe_load(resources.files("lss.data").joinpath(name).read_text(encoding="utf-8"))
    table = np.asarray(doc["k"], dtype=float)
    return Glass(doc["id"], doc["name"], table[:, 0] * 1000.0, table[:, 1], doc["n_formula"])


@dataclass(frozen=True)
class Envelope:
    material: str
    wall_thickness: Parameter  # mm

    def __post_init__(self):
        if self.wall_thickness.unit != "mm":
            raise ValueError("envelope wall_thickness must be in mm")
        glass(self.material)  # fail on an unknown material at load time

    @property
    def glass(self) -> Glass:
        return glass(self.material)

    def transmission(self) -> Transmission:
        """Internal transmittance of the wall, linear between the table's wavelengths.

        Outside the table the glass is taken as opaque. For soda-lime that is right
        (below 310 nm its absorption edge, k already growing ~4× per 10 nm; beyond
        4.6 µm the silicate bands); fused silica's table stops at 240 nm and 2.6 µm,
        so a model is not scored outside them.
        """
        g = self.glass
        lam = g.k_wavelength_nm
        t = g.internal_transmittance(self.wall_thickness.value)
        knots = np.concatenate([[lam[0]], lam, [lam[-1]]])
        values = np.concatenate([[0.0], t, [0.0]])
        return Transmission(knots, values, f"{g.name}, {self.wall_thickness.value:g} mm wall")

    def as_dict(self) -> dict:
        return {"material": self.material, "wall_thickness": self.wall_thickness.as_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Envelope":
        return cls(d["material"], Parameter.from_dict(d["wall_thickness"]))
