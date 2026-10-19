"""A study as a Mitsuba scene (design §8.1, §8.4). Needs Mitsuba with a variant selected
(``lss.render.env.setup``).

Every lamp is an emitter with a flat spectrum on its sphere or disc; which lamp, and
which of a disc's angle bands, emits is switched in place (``set_emitting``), so one
loaded scene serves every lighting map. Mitsuba merges equal textures, and a merged
radiance would then be missing from ``mi.traverse``, so sphere k starts at radiance
1 + 10⁻⁶·k. A lamp's sphere or disc is black: it absorbs returning light (one fixed
boundary, §2). A disc emits from its front only (``lss.render.emitters``).
"""

import math

import numpy as np

from lss.lamp.profile import fresnel_index, polar_band

from lss.scene.geometry import Shape
from lss.scene.materials import Material
from lss.scene.study import Study
from lss.spectral.grids import BinGrid

CHANNEL_RAMP_NM = 1e-3


def _fmt(a) -> str:
    return ", ".join(f"{float(x):.10g}" for x in a)


def channel_knots(grid: BinGrid, j: int, ramp: float = CHANNEL_RAMP_NM) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity of channel j: 1 inside the bin, ramps of ``ramp`` centred on its edges, so
    neighbouring channels are complementary and all of them sum to 1 inside the range."""
    lo, hi = grid.edges[j], grid.edges[j + 1]
    return (np.array([lo - ramp / 2, lo + ramp / 2, hi - ramp / 2, hi + ramp / 2]), np.array([0.0, 1.0, 1.0, 0.0]))


def spectrum_dict(wavelengths, values) -> dict:
    return {"type": "irregular", "wavelengths": _fmt(wavelengths), "values": _fmt(values)}


def film(grid: BinGrid, width: int, height: int, channels: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    """A ``specfilm``: by default one box channel per render bin."""
    if channels is None:
        channels = [channel_knots(grid, j) for j in range(len(grid))]
    f = {"type": "specfilm", "width": width, "height": height, "rfilter": {"type": "box"},
         "component_format": "float32"}
    for j, (w, v) in enumerate(channels):
        f[f"ch{j:03d}"] = spectrum_dict(w, v)
    return f


def read_film(image, grid: BinGrid) -> np.ndarray:
    """Per-nm map (h, w, bins) from a rendered ``specfilm`` image: each channel holds the
    integral over its bin, so it is divided by the bin width here and nowhere else."""
    a = np.asarray(image, dtype=np.float64)
    if a.shape[-1] != len(grid):
        raise ValueError(f"film has {a.shape[-1]} channels, the grid {len(grid)} bins")
    return a / grid.widths


def bsdf(material: Material) -> dict:
    return {"type": "diffuse", "reflectance": spectrum_dict(material.wavelengths, material.reflectance)}


def shape_dict(shape: Shape, materials: dict[str, Material]) -> dict:
    import mitsuba as mi

    b = bsdf(materials[shape.material])
    if shape.kind == "sphere":
        return {"type": "sphere", "center": list(shape.centre), "radius": shape.radius, "bsdf": b}
    if shape.kind == "rectangle":
        return {"type": "rectangle", "to_world": mi.ScalarTransform4f(shape.rectangle_frame().tolist()), "bsdf": b}
    lo, hi = np.asarray(shape.lo), np.asarray(shape.hi)
    T = mi.ScalarTransform4f
    return {"type": "cube", "flip_normals": True, "bsdf": b,
            "to_world": T().translate((0.5 * (lo + hi)).tolist()) @ T().scale((0.5 * (hi - lo)).tolist())}


def emitter_key(lamp_id: str) -> str:
    return f"lamp_{lamp_id}"


BLACK = {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.0}}


def disc_frame(centre, normal, radius: float) -> np.ndarray:
    """4×4 matrix taking Mitsuba's unit disk (radius 1, normal +z) to this disc."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)  # right-handed: u × v = n
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = u * radius, v * radius, n, centre
    return m


def lamp_shapes(study: Study) -> dict:
    """One black emitting sphere or disc per lamp. Zones of one shape share a lighting
    map, re-coloured with their summed spectrum; a disc's bands each have their own."""
    import mitsuba as mi

    out = {}
    for k, p in enumerate(study.lamps):
        if p.kind == "sphere":
            out[emitter_key(p.id)] = {
                "type": "sphere", "center": list(p.centre), "radius": p.radius, "bsdf": BLACK,
                "emitter": {"type": "area", "radiance": {"type": "uniform", "value": 1.0 + 1e-6 * k}}}
        else:
            out[emitter_key(p.id)] = {
                "type": "disk", "to_world": mi.ScalarTransform4f(disc_frame(p.centre, p.normal, p.radius).tolist()),
                "bsdf": BLACK, "emitter": {"type": "polardisc"}}
    return out


def scene_dict(study: Study) -> dict:
    import mitsuba as mi

    c = study.camera
    d = {"type": "scene",
         "integrator": {"type": "path", "max_depth": study.max_depth},
         "sensor": {"type": "perspective", "fov": c.fov_deg, "fov_axis": "x",
                    "to_world": mi.ScalarTransform4f().look_at(list(c.origin), list(c.target), list(c.up)),
                    "film": film(study.grid, c.width, c.height), "sampler": {"type": "independent"}}}
    for i, s in enumerate(study.shapes):
        d[f"s{i:03d}"] = shape_dict(s, study.materials)
    d.update(lamp_shapes(study))
    return d


def band_cosines(shape: str) -> tuple[float, float]:
    """(mu_lo, mu_hi) of a zone's band: directions with mu_lo < cos θ ≤ mu_hi."""
    lo, hi = polar_band(shape)
    return (0.0 if hi >= 90.0 else math.cos(math.radians(hi))), math.cos(math.radians(lo))


def set_emitting(params, study: Study, lamp_id: str, shape: str) -> None:
    """Lamp ``lamp_id`` emits radiance 1 in zone ``shape``; every other lamp is off
    (radiance and sampling weight 0)."""
    for p in study.lamps:
        k, on = emitter_key(p.id), p.id == lamp_id
        if p.kind == "sphere":
            if on and shape != "isotropic":
                raise ValueError(f"{p.id}: a sphere has no zone {shape!r}")
            params[f"{k}.emitter.radiance.value"] = 1.0 if on else 0.0
        else:
            params[f"{k}.emitter.value"] = 1.0 if on else 0.0
            if on:
                params[f"{k}.emitter.mu_lo"], params[f"{k}.emitter.mu_hi"] = band_cosines(shape)
                params[f"{k}.emitter.n"] = fresnel_index(shape)
        params[f"{k}.emitter.sampling_weight"] = 1.0 if on else 0.0
    params.update()
