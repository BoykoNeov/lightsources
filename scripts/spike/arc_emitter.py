"""Renderer spike, item 1 (design doc §8.3): a Python emitter with (b, θ) zone masks.

Attached to a cylinder (the arc's boundary), it gives each surface point p and outgoing
direction d the chord's impact parameter b (the distance between the line through p along
d and the arc's axis; 0 to the radius) and its angle θ to the axis, as `side_radiance` in
`lss/physics/hid/arc.py` defines them, and returns a flat unit spectrum times a grey mask:
1 inside the selected (b, θ) zone, 0 outside. The renderer never sees a lamp spectrum.

    python scripts/spike/arc_emitter.py cuda_ad_spectral [res spp]
"""
import json
import math
import sys
import time

import drjit as dr
import mitsuba as mi
import numpy as np

R, Z0, Z1 = 0.05, -0.5, 0.5  # arc radius and ends along z (m)


def register():
    class ArcZoneEmitter(mi.Emitter):
        """Zone (b_lo ≤ b/R < b_hi, th_lo ≤ θ < th_hi, θ in degrees) of an arc on the z axis.

        ``code`` instead returns 1 + (b band) + 10·(θ band) for 4 b bands × 2 θ bands, a
        test pattern for reading the (b, θ) the emitter computes back out of an image.
        """

        def __init__(self, props):
            super().__init__(props)
            self.b_lo = props.get("b_lo", 0.0)
            self.b_hi = props.get("b_hi", 1.0)
            self.th_lo = math.radians(props.get("th_lo", 0.0))
            self.th_hi = math.radians(props.get("th_hi", 90.0001))
            self.code = props.get("code", False)
            self.m_flags = mi.EmitterFlags.Surface | mi.EmitterFlags.SpatiallyVarying

        def mask(self, p, d):
            u = mi.Vector3f(0, 0, 1)
            n = dr.cross(u, d)  # |n| = sin θ
            s = dr.norm(n)
            b = dr.abs(dr.dot(mi.Vector3f(p.x, p.y, 0), n)) / dr.maximum(s, 1e-7) / R
            th = dr.acos(dr.clip(dr.abs(d.z), 0, 1))  # 0 along the axis, π/2 across it
            if self.code:
                bb = dr.minimum(mi.Float(mi.Int32(b * 4)), 3)
                return 1 + bb + 10 * dr.select(th < math.radians(60), 1.0, 0.0)
            return dr.select((b >= self.b_lo) & (b < self.b_hi) & (th >= self.th_lo) & (th < self.th_hi),
                             1.0, 0.0)

        def eval(self, si, active=True):
            front = mi.Frame3f.cos_theta(si.wi) > 0
            v = self.mask(si.p, si.to_world(si.wi))
            return mi.Spectrum(dr.select(active & front, v, 0.0))

        def sample_direction(self, it, sample, active=True):
            ds = self.get_shape().sample_direction(it, sample, active)
            ok = active & (dr.dot(ds.d, ds.n) < 0) & (ds.pdf != 0)
            v = self.mask(ds.p, -ds.d) / ds.pdf
            ds.emitter = mi.EmitterPtr(self)
            return ds, mi.Spectrum(dr.select(ok, v, 0.0))

        def pdf_direction(self, it, ds, active=True):
            ok = active & (dr.dot(ds.d, ds.n) < 0)
            return dr.select(ok, self.get_shape().pdf_direction(it, ds, ok), 0.0)

        def eval_direction(self, it, ds, active=True):
            ok = active & (dr.dot(ds.d, ds.n) < 0)
            return mi.Spectrum(dr.select(ok, self.mask(ds.p, -ds.d), 0.0))

        def sample_ray(self, time, sample1, sample2, sample3, active=True):
            raise NotImplementedError("not needed by the path tracer")

        def traverse(self, callback):
            pass

        def to_string(self):
            return "ArcZoneEmitter"

    mi.register_emitter("arczone", lambda props: ArcZoneEmitter(props))


def film(res_x, res_y, channels=4):
    f = {"type": "specfilm", "width": res_x, "height": res_y, "rfilter": {"type": "box"},
         "component_format": "float32"}
    for i in range(channels):  # a few bins are enough; the emission is flat
        lo = 400 + 100 * i
        f[f"ch{i}"] = {"type": "irregular", "wavelengths": f"{lo - 1e-3}, {lo}, {lo + 100}, {lo + 100 + 1e-3}",
                       "values": "0, 1, 1, 0"}
    return f


def arc(emitter):
    return {"type": "cylinder", "p0": [0, 0, Z0], "p1": [0, 0, Z1], "radius": R,
            "emitter": emitter}


def pattern_test(view_deg):
    """Orthographic view of the bare arc at view_deg to its axis; column x gives b = |x|/R."""
    T = mi.ScalarTransform4f
    t = math.radians(view_deg)
    d = [0.0, math.sin(t), math.cos(t)]  # camera looks along −d
    up = [0.0, -math.cos(t), math.sin(t)]
    nx, ny = 256, 64
    sc = mi.load_dict({"type": "scene", "integrator": {"type": "path", "max_depth": 1},
                       "sensor": {"type": "orthographic",
                                  "to_world": T().look_at([2 * c for c in d], [0, 0, 0], up)
                                  @ T().scale([R * 1.25, R * 1.25 * ny / nx, 1]),
                                  "film": film(nx, ny, 1), "sampler": {"type": "independent"}},
                       "arc": arc({"type": "arczone", "code": True})})
    img = np.array(mi.render(sc, spp=64))[..., 0]  # (y, x) — value × 100 nm bin width
    img = img / 100.0
    x = ((np.arange(nx) + 0.5) / nx * 2 - 1) * R * 1.25  # signed offset from the axis (m)
    b = np.abs(x) / R
    inside = b < 0.98
    edge_ok = np.abs((b * 4) - np.round(b * 4)) > 0.03  # skip pixels straddling a band edge
    expect = 1 + np.minimum(np.floor(b * 4), 3) + 10 * (view_deg < 60)
    row = img[ny // 2]
    sel = inside & edge_ok
    good = np.isclose(row[sel], expect[sel], atol=1e-3)
    outside = float(np.abs(row[~inside & (b > 1.02)]).max(initial=0.0))
    return {"view_deg": view_deg, "pixels_checked": int(sel.sum()), "fraction_matching": round(float(good.mean()), 4),
            "max_outside_arc": outside}


def flux_scene(emitter, spp_res=64):
    """A 1 m² patch 0.3 m from the arc's side, seen head-on; returns the scene."""
    T = mi.ScalarTransform4f
    return mi.load_dict({
        "type": "scene", "integrator": {"type": "path", "max_depth": 2},
        "sensor": {"type": "orthographic",
                   "to_world": T().look_at([0, -1, 0], [0, 0.3, 0], [0, 0, 1]) @ T().scale([0.5, 0.5, 1]),
                   "film": film(spp_res, spp_res), "sampler": {"type": "independent"}},
        "patch": {"type": "rectangle",
                  "to_world": T().translate([0, 0.3, 0]) @ T().rotate([1, 0, 0], 90) @ T().scale(0.5),
                  "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.5}}},
        "arc": arc(emitter)})


def stats(sc, spp, reps=5):
    imgs, times = [], []
    for s in range(reps + 1):
        t0 = time.perf_counter()
        a = np.array(mi.render(sc, spp=spp, seed=s))
        times.append(time.perf_counter() - t0)
        imgs.append(a)
    imgs = np.stack(imgs[1:])  # drop the compiling first render
    per_rep = imgs.mean(axis=(1, 2, 3))
    return float(per_rep.mean()), float(per_rep.std(ddof=1) / math.sqrt(reps)), times[0], min(times[1:])


def main():
    mi.set_variant(sys.argv[1])
    register()
    res, spp = (int(a) for a in sys.argv[2:4]) if len(sys.argv) > 3 else (64, 256)
    out = {"variant": sys.argv[1], "res": res, "spp": spp,
           "patterns": [pattern_test(90), pattern_test(45)]}
    builtin = flux_scene({"type": "area", "radiance": {"type": "uniform", "value": 1.0}}, res)
    ours = flux_scene({"type": "arczone"}, res)
    zoned = flux_scene({"type": "arczone", "b_lo": 0.25, "b_hi": 0.5, "th_lo": 60, "th_hi": 90.0001}, res)
    for name, sc in [("builtin_area", builtin), ("arczone_all", ours), ("arczone_zone", zoned)]:
        m, se, t_first, t_later = stats(sc, spp)
        out[name] = {"mean": m, "se": se, "first_s": round(t_first, 3), "later_s": round(t_later, 4)}
    a, b = out["builtin_area"], out["arczone_all"]
    out["all_vs_builtin_z"] = round((b["mean"] - a["mean"]) / math.hypot(a["se"], b["se"]), 2)
    out["cost_ratio"] = round(b["later_s"] / a["later_s"], 2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
