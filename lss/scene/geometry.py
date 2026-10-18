"""Scene shapes, independent of the renderer (design §8.2).

Three kinds, all that the first reference lab needs:
- ``rectangle``: centre, outward normal, an up direction in its plane, width × height;
- ``sphere``: centre and radius;
- ``room``: an axis-aligned box seen from inside (normals inward), by its two corners.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Shape:
    id: str
    kind: str  # rectangle | sphere | room
    material: str
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)  # rectangle
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)  # rectangle
    size: tuple[float, float] = (1.0, 1.0)  # rectangle: width, height
    radius: float = 0.0  # sphere
    lo: tuple[float, float, float] = (0.0, 0.0, 0.0)  # room
    hi: tuple[float, float, float] = (1.0, 1.0, 1.0)  # room

    def __post_init__(self):
        if self.kind not in ("rectangle", "sphere", "room"):
            raise ValueError(f"{self.id}: unknown shape kind {self.kind!r}")
        if self.kind == "rectangle" and abs(float(np.dot(self.normal, self.up))) > 1e-9:
            raise ValueError(f"{self.id}: a rectangle's up must lie in its plane")

    def as_dict(self) -> dict:
        d = {"id": self.id, "kind": self.kind, "material": self.material}
        if self.kind == "rectangle":
            d.update(centre=self.centre, normal=self.normal, up=self.up, size=self.size)
        elif self.kind == "sphere":
            d.update(centre=self.centre, radius=self.radius)
        else:
            d.update(lo=self.lo, hi=self.hi)
        return d

    def rectangle_frame(self) -> np.ndarray:
        """4×4 matrix taking Mitsuba's unit rectangle ([−1, 1]², normal +z) to this one."""
        n = np.asarray(self.normal, float)
        n = n / np.linalg.norm(n)
        v = np.asarray(self.up, float)
        v = v / np.linalg.norm(v)
        u = np.cross(v, n)  # right-handed: u × v = n
        m = np.eye(4)
        m[:3, 0] = u * self.size[0] / 2
        m[:3, 1] = v * self.size[1] / 2
        m[:3, 2] = n
        m[:3, 3] = self.centre
        return m
