"""Renderer spike, items 2 and 4 (design doc §8.3): spectral output and the GPU.

A flat Lambertian patch (reflectance RHO) under a directional emitter of irradiance E at
normal incidence, seen head-on by an orthographic camera through a `specfilm` with box
channels. Each channel integrates radiance times its sensitivity over
wavelength, so its expected value is RHO·E/π times the bin width. All pixels have the same expectation,
so the spread across pixels at a given spp is the per-bin noise.

    python scripts/spike/specfilm_noise.py llvm_ad_spectral [--lo 360 --hi 830 --width 10]
"""
import argparse
import json
import math
import time

import mitsuba as mi

RHO, E = 0.5, 1.0


def box(lo, hi, eps=1e-3):
    return {"type": "irregular", "wavelengths": f"{lo - eps}, {lo}, {hi}, {hi + eps}",
            "values": "0, 1, 1, 0"}


def scene(edges, res, emit=None):
    film = {"type": "specfilm", "width": res, "height": res, "rfilter": {"type": "box"},
            "component_format": "float32"}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        film[f"ch{i:02d}"] = box(lo, hi)
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {"type": "orthographic",
                   "to_world": mi.ScalarTransform4f().look_at([0, 0, 1], [0, 0, 0], [0, 1, 0])
                   @ mi.ScalarTransform4f().scale([0.5, 0.5, 1]),
                   "film": film, "sampler": {"type": "independent"}},
        "patch": {"type": "rectangle", "bsdf": {"type": "diffuse",
                                                "reflectance": {"type": "uniform", "value": RHO}}},
        "light": {"type": "directional", "direction": [0, 0, -1],
                  "irradiance": emit or {"type": "uniform", "value": E}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--lo", type=float, default=360)
    ap.add_argument("--hi", type=float, default=830)
    ap.add_argument("--width", type=float, default=10)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--spp", default="16,64,256,1024")
    ap.add_argument("--threads", type=int, help="Dr.Jit CPU worker threads (default: all)")
    args = ap.parse_args()
    mi.set_variant(args.variant)
    if args.threads:
        import drjit as dr
        dr.set_thread_count(args.threads)
    import numpy as np

    n = round((args.hi - args.lo) / args.width)
    edges = [args.lo + i * args.width for i in range(n + 1)]
    sc = mi.load_dict(scene(edges, args.res))
    expect = RHO * E / math.pi * args.width
    out = {"variant": args.variant, "channels": n, "expect": expect, "runs": []}
    for spp in map(int, args.spp.split(",")):
        t0 = time.perf_counter()
        img = np.array(mi.render(sc, spp=spp))
        dt = time.perf_counter() - t0
        px = img.reshape(-1, img.shape[-1])  # (pixel, channel)
        mean, sd = px.mean(0), px.std(0, ddof=1)
        rel_sd = sd / expect  # one pixel's relative noise
        z = (mean - expect) / (sd / math.sqrt(px.shape[0]))
        out["runs"].append({"spp": spp, "seconds": round(dt, 3), "shape": list(img.shape),
                            "mean_over_expect": [round(v, 4) for v in (mean / expect).tolist()],
                            "rel_sd_px": [round(v, 4) for v in rel_sd.tolist()],
                            "rel_sd_px_times_sqrt_spp_median": round(float(np.median(rel_sd)) * math.sqrt(spp), 3),
                            "max_abs_z": round(float(np.abs(z).max()), 2)})
        r = out["runs"][-1]
        print(f"spp {spp:5d}  {dt:7.3f} s  shape {img.shape}  mean/expect "
              f"{min(r['mean_over_expect']):.4f}..{max(r['mean_over_expect']):.4f}  "
              f"rel sd*sqrt(spp) median {r['rel_sd_px_times_sqrt_spp_median']}  max|z| {r['max_abs_z']}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
