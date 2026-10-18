"""Renderer spike, item 3 (design doc §8.3): can the spectral variants render outside 360–830 nm?

The patch scene of specfilm_noise.py, with the emitter's irradiance nonzero only in one
band, seen through 10 nm channels across 250–1000 nm. Light in a band that arrives in the
channels over that band means the band is renderable.

    python scripts/spike/wl_range.py cuda_ad_spectral
"""
import sys

import mitsuba as mi
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from specfilm_noise import RHO, box, scene  # noqa: E402


def main():
    mi.set_variant(sys.argv[1])
    edges = list(range(250, 1001, 10))
    for lo, hi in [(300, 350), (500, 550), (850, 900)]:
        try:
            sc = mi.load_dict(scene(edges, 16, emit=box(lo, hi)))
            img = np.array(mi.render(sc, spp=256)).reshape(-1, len(edges) - 1).mean(0)
        except Exception as e:  # the finding may be a refusal
            print(f"emit {lo}-{hi} nm: refused: {type(e).__name__}: {str(e)[:300]}")
            continue
        expect = RHO / np.pi * 10
        lit = [f"{edges[i]}-{edges[i + 1]}:{v / expect:.3f}" for i, v in enumerate(img) if abs(v) > 1e-6 * expect]
        print(f"emit {lo}-{hi} nm: channels with signal (value / expected for a fully lit bin): {lit or 'none'}")


if __name__ == "__main__":
    main()
