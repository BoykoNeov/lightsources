"""The reference lighting lab, first version (design §8.2, §8.4).

A closed room 6 × 6 × 3 m (x across, y away from the camera, z up), flat 0.5. On the
back wall (y = 3 m): the ColorChecker (BabelColor average, 6 × 4 patches of 10 cm), a
white card (0.9) and a grey card (0.18) on its left, the constructed metameric pair
(§7) on its right. A matte grey sphere (0.5) stands on the floor. The flat materials are
constructed, not measured. Lamps are placed by the study.
"""

from lss.scene.geometry import Shape
from lss.scene.materials import Material, flat, from_analysis_grid
from lss.studies import metamers, samples

ROOM_LO = (-3.0, -3.0, 0.0)
ROOM_HI = (3.0, 3.0, 3.0)
WALL_Y = ROOM_HI[1] - 0.001  # patches 1 mm in front of the back wall
FACING = (0.0, -1.0, 0.0)  # towards the camera
UP = (0.0, 0.0, 1.0)
PATCH = 0.10  # m
GAP = 0.02
CHART_CENTRE = (0.0, 1.4)  # x, z
CARD = 0.25


def _patch_centre(row: int, col: int) -> tuple[float, float, float]:
    pitch = PATCH + GAP
    x = CHART_CENTRE[0] + (col - 2.5) * pitch
    z = CHART_CENTRE[1] + (1.5 - row) * pitch
    return (x, WALL_Y, z)


def chart_patch_id(label: str) -> str:
    return "cc " + label


def reference_lab() -> tuple[tuple[Shape, ...], dict[str, Material]]:
    chart = samples.get("colorchecker")
    materials = {"grey50": flat("grey50", 0.5), "white90": flat("white90", 0.9), "grey18": flat("grey18", 0.18)}
    shapes = [Shape("room", "room", "grey50", lo=ROOM_LO, hi=ROOM_HI)]
    for i, label in enumerate(chart.labels):
        mid = chart_patch_id(label)
        materials[mid] = from_analysis_grid(mid, chart.values[:, i], chart.citation)
        shapes.append(Shape(mid, "rectangle", mid, centre=_patch_centre(i // 6, i % 6), normal=FACING, up=UP,
                            size=(PATCH, PATCH)))
    x_left = CHART_CENTRE[0] - 3 * (PATCH + GAP) - 0.1 - CARD / 2
    shapes.append(Shape("white card", "rectangle", "white90", centre=(x_left, WALL_Y, CHART_CENTRE[1] + 0.14),
                        normal=FACING, up=UP, size=(CARD, CARD)))
    shapes.append(Shape("grey card", "rectangle", "grey18", centre=(x_left, WALL_Y, CHART_CENTRE[1] - 0.14),
                        normal=FACING, up=UP, size=(CARD, CARD)))
    pair = metamers.make()
    x_right = CHART_CENTRE[0] + 3 * (PATCH + GAP) + 0.1 + CARD / 2
    for k, (name, values) in enumerate((("pair first", pair.first), ("pair second", pair.second))):
        materials[name] = from_analysis_grid(name, values, f"constructed metameric pair from {pair.base} (§7)")
        shapes.append(Shape(name, "rectangle", name, centre=(x_right, WALL_Y, CHART_CENTRE[1] + 0.14 - 0.28 * k),
                            normal=FACING, up=UP, size=(CARD, CARD)))
    shapes.append(Shape("sphere", "sphere", "grey50", centre=(0.9, 2.2, 0.35), radius=0.35))
    return tuple(shapes), materials
