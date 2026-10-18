"""Renderer spike, items 4 and 5 (design doc §8.3): CPU against GPU, and recompilation.

A reference-lab-sized scene: a closed room, a 6×4 chart of cubes with distinct
reflectances, a matte and a rough-metal sphere, a tessellated sphere (to put a few
thousand triangles in the BVH) and a rectangular area light; path tracer, max depth 8,
specfilm with 47 box channels of 10 nm over 360–830 nm.

    python scripts/spike/lab_timing.py cuda_ad_spectral [--res 512x384 --spp 64]
    python scripts/spike/lab_timing.py llvm_ad_spectral --recompile
"""
import argparse
import json
import os
import tempfile
import time

import mitsuba as mi
import numpy as np

PLY = os.path.join(tempfile.gettempdir(), "lss_spike_tess_sphere.ply")


def box(lo, hi, eps=1e-3):
    return {"type": "irregular", "wavelengths": f"{lo - eps}, {lo}, {hi}, {hi + eps}",
            "values": "0, 1, 1, 0"}


def film(w, h):
    f = {"type": "specfilm", "width": w, "height": h, "rfilter": {"type": "box"},
         "component_format": "float32"}
    for i in range(47):
        f[f"ch{i:02d}"] = box(360 + 10 * i, 370 + 10 * i)
    return f


def uv_sphere(n_lat=40, n_lon=80, centre=(0.0, 0.0, 0.0), r=1.0):
    th = np.linspace(0, np.pi, n_lat + 1)
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    t, p = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1).reshape(-1, 3)
    v = v * r + np.asarray(centre)
    f = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            c, d = a + n_lon, b + n_lon
            f += [[a, c, b], [b, c, d]]
    m = mi.Mesh("tess", vertex_count=len(v), face_count=len(f), has_vertex_normals=False)
    params = mi.traverse(m)
    params["vertex_positions"] = mi.Float(v.astype(np.float32).ravel())
    params["faces"] = mi.UInt32(np.asarray(f, dtype=np.uint32).ravel())
    params.update()
    return m, len(f)


def lab(w, h, max_depth):
    T = mi.ScalarTransform4f
    rng = np.random.default_rng(1)
    d = {"type": "scene", "integrator": {"type": "path", "max_depth": max_depth},
         "sensor": {"type": "perspective", "fov": 60,
                    "to_world": T().look_at([0, -3.5, 1.2], [0, 0, 0.9], [0, 0, 1]),
                    "film": film(w, h), "sampler": {"type": "independent"}}}
    wall = {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.6}}
    for name, tf in {"floor": T().scale(2),
                     "ceil": T().translate([0, 0, 2.5]) @ T().rotate([1, 0, 0], 180) @ T().scale(2),
                     "back": T().translate([0, 2, 1.25]) @ T().rotate([1, 0, 0], 90) @ T().scale([2, 1.25, 1]),
                     "left": T().translate([-2, 0, 1.25]) @ T().rotate([0, 1, 0], 90) @ T().scale([1.25, 2, 1]),
                     "right": T().translate([2, 0, 1.25]) @ T().rotate([0, 1, 0], -90) @ T().scale([1.25, 2, 1]),
                     "front": T().translate([0, -4, 1.25]) @ T().rotate([1, 0, 0], -90) @ T().scale([2, 1.25, 1])}.items():
        d[name] = {"type": "rectangle", "to_world": tf, "bsdf": wall}
    wl = "360, 420, 480, 540, 600, 660, 720, 780, 830"
    for i in range(24):
        r, c = divmod(i, 6)
        vals = ", ".join(f"{v:.3f}" for v in rng.uniform(0.03, 0.9, 9))
        d[f"patch{i}"] = {"type": "cube",
                          "to_world": T().translate([-1.0 + 0.4 * c, 1.9, 1.6 - 0.4 * r]) @ T().scale([0.17, 0.05, 0.17]),
                          "bsdf": {"type": "diffuse", "reflectance": {"type": "irregular", "wavelengths": wl, "values": vals}}}
    d["matte"] = {"type": "sphere", "center": [-0.8, 0.5, 0.35], "radius": 0.35,
                  "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.8}}}
    d["chrome"] = {"type": "sphere", "center": [0.8, 0.5, 0.35], "radius": 0.35,
                   "bsdf": {"type": "roughconductor", "material": "Cr", "alpha": 0.05}}
    mesh, ntri = uv_sphere(centre=(0.0, -0.3, 0.3), r=0.3)
    mesh.write_ply(PLY)
    d["tess"] = {"type": "ply", "filename": PLY,
                 "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.4}}}
    d["light"] = {"type": "rectangle",
                  "to_world": T().translate([0, 0, 2.49]) @ T().rotate([1, 0, 0], 180) @ T().scale(0.4),
                  "emitter": {"type": "area", "radiance": {"type": "uniform", "value": 10.0}}}
    return d, ntri


def load(w, h, max_depth):
    d, ntri = lab(w, h, max_depth)
    return mi.load_dict(d), ntri


def timed(sc, spp, seed=0):
    t0 = time.perf_counter()
    img = mi.render(sc, spp=spp, seed=seed)
    arr = np.array(img)  # forces evaluation
    return time.perf_counter() - t0, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--res", default="512x384")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--recompile", action="store_true")
    ap.add_argument("--threads", type=int, help="Dr.Jit CPU worker threads (default: all)")
    args = ap.parse_args()
    mi.set_variant(args.variant)
    if args.threads:
        import drjit as dr
        dr.set_thread_count(args.threads)
    w, h = map(int, args.res.split("x"))
    out = {"variant": args.variant, "res": [w, h], "spp": args.spp}
    sc, ntri = load(w, h, 8)
    out["mesh_triangles"] = ntri
    first, img = timed(sc, args.spp)
    later = [timed(sc, args.spp, seed=s + 1)[0] for s in range(args.repeat)]
    out.update(first_s=round(first, 3), later_s=[round(t, 3) for t in later],
               ms_per_spp=round(1000 * min(later) / args.spp, 2),
               mean_px=float(img.mean()), finite=bool(np.isfinite(img).all()))
    if args.recompile:
        # An edit in place, on the loaded scene: a chart patch's reflectance and the light's radiance.
        params = mi.traverse(sc)
        params["patch0.bsdf.reflectance.values"] *= 0.9
        params["light.emitter.radiance.value"] *= 1.1
        params.update()
        a, _ = timed(sc, args.spp, seed=7)
        b, _ = timed(sc, args.spp, seed=8)
        rec = {"edit in place": {"first_s": round(a, 3), "second_s": round(b, 3)}}
        for label, (ww, hh, md) in {"res 640x480": (640, 480, 8), "depth 4": (w, h, 4),
                                    "back to start": (w, h, 8)}.items():
            t0 = time.perf_counter()
            s2, _ = load(ww, hh, md)
            t_load = time.perf_counter() - t0
            a, _ = timed(s2, args.spp)
            b, _ = timed(s2, args.spp, seed=9)
            rec[label] = {"load_s": round(t_load, 3), "first_s": round(a, 3), "second_s": round(b, 3)}
        out["recompile"] = rec
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
