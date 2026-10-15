"""Reproduce the digitised Na₂ curves in ``lss/data/na2_pairs.yaml`` and the Fig. 6 window
means in ``tests/test_hid_pairs.py`` from Chung, Kirby & Babb, Phys. Rev. A 63, 032516
(2001), arXiv:physics/0008060 (design §4.4, rung 2).

The paper does not tabulate its curves. Its arXiv PDF draws them as vector paths, so the
plotted points are read directly from page 12 (Figs. 1 and 2: potentials and difference
potentials) and page 15 (Fig. 6: reduced absorption coefficients), calibrated on the
panels' frames.

Tool-only dependency, not needed by the package: PyMuPDF (``pip install pymupdf``).

    python scripts/digitise_na2_curves.py            # prints the YAML blocks and Fig. 6 means
    python scripts/digitise_na2_curves.py --check    # compares with the committed YAML
"""

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

ARXIV_PDF = "https://arxiv.org/pdf/physics/0008060"
HARTREE_CM1 = 219474.6313632
ASYMPTOTE_CM1 = 16956.172
X_DEPTH_CM1 = 6022.04  # Figs. 1(a), 2(a) put zero at the X minimum (Knoop et al. 2011: 6022.042)
C3 = {"A-X": -12.26, "B-X": 6.1486, "b-a": -12.26, "c-a": 6.1486}
UPPER = {"A-X": ("A", "X"), "B-X": ("B", "X"), "b-a": ("b", "a"), "c-a": ("c", "a")}
#: Where each curve hands over from the spline to its long-range form (blend over 2 a0).
JOIN_A0 = {"A-X": 16.0, "B-X": 11.0, "b-a": 11.0, "c-a": 11.0}
GRID_A0 = np.round(np.arange(4.20, 24.0001, 0.05), 2)
A_INNER_A0 = np.round(np.arange(4.20, 8.0801, 0.05), 2)


def fetch_pdf(cache: Path) -> Path:
    path = cache / "chung2001_physics0008060.pdf"
    if not path.exists():
        req = urllib.request.Request(ARXIV_PDF, headers={"User-Agent": "lss-digitiser"})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return path


def _segments(page):
    return [it for d in page.get_drawings() for it in d["items"] if it[0] == "l"], page.get_drawings()


def figure_curves(doc) -> dict:
    """Page 12: the ten curves of Figs. 1 and 2, as (R in a0, E in cm⁻¹)."""
    drawings = doc[11].get_drawings()
    frames = {"1a": (7, 0.0, 30e3), "1b": (14, 12e3, 20e3), "2a": (22, 0.0, 30e3), "2b": (29, 10e3, 20e3)}
    curves = {"X": (4, "1a"), "A": (5, "1a"), "B": (6, "1a"), "A-X": (12, "1b"), "B-X": (13, "1b"),
              "a": (19, "2a"), "b": (20, "2a"), "c": (21, "2a"), "b-a": (27, "2b"), "c-a": (28, "2b")}
    out = {}
    for name, (i, panel) in curves.items():
        frame, e0, e1 = frames[panel]
        r = drawings[frame]["rect"]
        pts = np.array([(p.x, p.y) for it in drawings[i]["items"] if it[0] == "l" for p in (it[1], it[2])])
        rr = 4.0 + (pts[:, 0] - r.x0) / (r.x1 - r.x0) * 20.0  # every panel spans 4-24 a0
        ee = e0 + (r.y1 - pts[:, 1]) / (r.y1 - r.y0) * (e1 - e0)
        ru, inv = np.unique(np.round(rr, 3), return_inverse=True)
        out[name] = (ru, np.bincount(inv, ee) / np.bincount(inv))
    return out


def raw_difference(cur, band, r):
    """Panel (b) where it is on scale, otherwise the difference of the panel (a) curves."""
    rb, eb = cur[band]
    up, lo = UPPER[band]
    da = np.interp(r, *cur[up], left=np.nan) - np.interp(r, *cur[lo], left=np.nan)
    on_scale = (r >= rb.min()) & (np.interp(r, rb, eb) < 19990) & (np.interp(r, rb, eb) > 10010)
    return np.where(on_scale, np.interp(r, rb, eb), da)


def smoothed_difference(cur, band):
    """Least-squares cubic spline joined to asymptote + c3/R³ + p6/R⁶ + p8/R⁸ (c3 fixed)."""
    j = JOIN_A0[band]
    rr = np.round(np.arange(4.20, 24.0001, 0.01), 2)
    raw = raw_difference(cur, band, rr)
    far = rr >= j + 1.0
    design = np.vstack([1 / rr[far] ** 6, 1 / rr[far] ** 8]).T
    p, *_ = np.linalg.lstsq(design, (raw[far] - ASYMPTOTE_CM1) / HARTREE_CM1 - C3[band] / rr[far] ** 3, rcond=None)

    def long_range(r):
        return ASYMPTOTE_CM1 + (C3[band] / r**3 + p[0] / r**6 + p[1] / r**8) * HARTREE_CM1

    near = rr <= j + 2.0
    knots = np.concatenate([np.arange(4.45, 12.0, 0.25), np.arange(12.0, j + 1.9, 0.5)])
    spline = LSQUnivariateSpline(rr[near], raw[near], knots, k=3)
    w = np.clip((GRID_A0 - j) / 2.0, 0.0, 1.0)
    curve = (1 - w) * spline(np.minimum(GRID_A0, j + 2.0)) + w * long_range(GRID_A0)
    resid = (np.sqrt(np.mean((raw[near] - spline(rr[near])) ** 2)), np.sqrt(np.mean((raw[far] - long_range(rr[far])) ** 2)))
    return curve, p, resid


def fig6_windows(doc) -> dict:
    """Page 15: mean k/n² (1e-37 cm⁵) over each window, panels (b) triplets and (c) all."""
    segs = np.array([(it[1].x, it[1].y, it[2].x, it[2].y) for d in doc[14].get_drawings()
                     for it in d["items"] if it[0] == "l"])
    rows = {3000: (95.6, 165.1, 10.0), 2000: (183.6, 253.0, 15.0), 1000: (271.5, 341.0, 800.0)}
    cols = {"b": (271.1, 383.0), "c": (422.0, 534.1)}
    out = {}
    for col, (x0, x1) in cols.items():
        for t, (y0, y1, top) in rows.items():
            inside = ((segs[:, 0] >= x0 - .5) & (segs[:, 2] <= x1 + .5)
                      & (np.minimum(segs[:, 1], segs[:, 3]) >= y0 - .5) & (np.maximum(segs[:, 1], segs[:, 3]) <= y1 + .5))
            s = segs[inside]
            axis = (np.abs(s[:, 0] - s[:, 2]) < 1e-3) | (np.abs(s[:, 1] - s[:, 3]) < 1e-3)
            edge = ((np.minimum(s[:, 0], s[:, 2]) < x0 + 3.5) | (np.maximum(s[:, 0], s[:, 2]) > x1 - 3.5)
                    | (np.maximum(s[:, 1], s[:, 3]) > y1 - 3.5) | (np.minimum(s[:, 1], s[:, 3]) < y0 + .5))
            c = s[~(axis & edge)]  # drop the tick marks
            x = np.concatenate([c[:, 0], c[:, 2]])
            k = (y1 - np.concatenate([c[:, 1], c[:, 3]])) / (y1 - y0) * top
            lam = 500.0 + (x - (x0 + 15.35)) / 0.254167  # ticks: 500 nm at 15.35 pt, 25.42 pt per 100 nm
            for lo, hi in ((610, 630), (640, 700), (525, 540)):
                m = (lam >= lo) & (lam <= hi)
                edges = np.arange(lo, hi + .001, .5)
                idx = np.digitize(lam[m], edges) - 1
                means = [k[m][idx == i].mean() for i in range(len(edges) - 1) if np.any(idx == i)]
                out[(col, t, lo, hi)] = float(np.mean(means)) if means else float("nan")
    return out


def main(argv=None) -> int:
    import pymupdf  # tool-only dependency

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with lss/data/na2_pairs.yaml")
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()))
    args = ap.parse_args(argv)
    doc = pymupdf.open(fetch_pdf(args.cache))
    cur = figure_curves(doc)

    curves, fits = {}, {}
    for band in C3:
        curve, p, resid = smoothed_difference(cur, band)
        curves[band], fits[band] = np.round(curve), p
        print(f"# {band}: p6 {p[0]:.5g}, p8 {p[1]:.5g} a.u.; spline rms {resid[0]:.1f}, long-range rms {resid[1]:.1f} cm-1")
    a_inner = np.round(np.interp(A_INNER_A0, *cur["a"]) - X_DEPTH_CM1)
    windows = fig6_windows(doc)
    for (col, t, lo, hi), v in sorted(windows.items()):
        if np.isfinite(v):
            print(f"# Fig. 6({col}) {t} K {lo}-{hi} nm: {v:.2f} e-37 cm5")

    if args.check:
        import yaml
        root = Path(__file__).resolve().parents[1]
        d = yaml.safe_load((root / "lss" / "data" / "na2_pairs.yaml").read_text(encoding="utf-8"))
        worst = max(float(np.abs(np.array(d["difference_potentials_cm1"][b], float) - curves[b]).max()) for b in C3)
        worst = max(worst, float(np.abs(np.array(d["a_state_inner_cm1"]["values"], float) - a_inner).max()))
        par = max(abs(float(d["bands"][b][k]) / fits[b][i] - 1) for b in C3 for i, k in enumerate(("p6", "p8")))
        print(f"largest difference from the committed curves: {worst:.0f} cm-1; p6/p8 relative: {par:.1e}")
        return 0 if worst <= 1.0 and par < 1e-3 else 1

    print("difference_potentials_cm1:")
    print(f"  r_a0: {{start: 4.20, step: 0.05, count: {GRID_A0.size}}}")
    for band, curve in curves.items():
        print(f"  {band}: [" + ", ".join(f"{v:.0f}" for v in curve) + "]")
    print("a_state_inner_cm1:")
    print(f"  r_a0: {{start: 4.20, step: 0.05, count: {A_INNER_A0.size}}}")
    print("  values: [" + ", ".join(f"{v:.0f}" for v in a_inner) + "]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
