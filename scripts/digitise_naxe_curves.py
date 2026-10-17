"""Reproduce the NaXe curves in ``lss/data/naxe_pairs.yaml`` from Galbis, Douady, Jacquet,
Giglio & Gervais, J. Chem. Phys. 138, 014314 (2013), Figs. 2 and 7 (design §4.4, rung 2).

The paper puts its computed values in a supplement; the article's PDF draws its figures as
vector paths, and each computed point as a small marker. So the markers are read directly:
Fig. 2 (page 3), X ²Σ⁺, red filled markers (Na, CCSD(T)); Fig. 7 (page 7), A ²Π, red
filled markers (CCSD(T)), and B ²Σ⁺, red open markers (CASPT2). The axes are calibrated on
the frames' tick marks: each tick label is matched to its nearest tick and a straight line
fitted through the ticks' positions. (The labels' own centres sit up to 0.6 pt off their
ticks, which read every A and B point 7.4 cm⁻¹ low.) Read so, X's well is −93.0 cm⁻¹ at
9.5 a₀ and A's −1094.3 cm⁻¹ at 6.1 a₀, the values the paper states. The markers sit
0.002–0.008 a₀ short of a 0.1 a₀ grid, so R is rounded to it.

The paper is not openly available, so the PDF is not downloaded here: pass the path to a
copy. Tool-only dependency, not needed by the package: PyMuPDF (``pip install pymupdf``).

    python scripts/digitise_naxe_curves.py galbis2013.pdf          # prints the YAML blocks
    python scripts/digitise_naxe_curves.py galbis2013.pdf --check  # compares with the committed YAML
"""

import argparse
import sys
from pathlib import Path

import numpy as np

RED = (0.93, 0.12, 0.14)  # Na in both figures (Li black, K blue)
#: (page index, y of the R-axis labels' centres, curve: marker drawing type)
FIGURES = {"X": (3, 366.6, "fs"), "A": (7, 365.7, "fs"), "B": (7, 365.7, "s")}


def _ticks(page) -> tuple[np.ndarray, np.ndarray]:
    """The frame's tick marks: short segments at its left edge (y) and bottom edge (x)."""
    xs, ys = set(), set()
    for d in page.get_drawings():
        r = d["rect"]
        if not (r.x1 < 300 and r.x0 < 90 and r.width > 150 and r.height > 250):
            continue
        for it in d["items"]:
            if it[0] != "l" or abs(it[1].x - it[2].x) + abs(it[1].y - it[2].y) > 10:
                continue
            a, b = it[1], it[2]
            if abs(a.y - b.y) < 0.01 and min(a.x, b.x) < r.x0 + 6:
                ys.add(round(a.y, 3))
            if abs(a.x - b.x) < 0.01 and max(a.y, b.y) > r.y1 - 6:
                xs.add(round(a.x, 3))
    return np.array(sorted(xs)), np.array(sorted(ys))


def _calibration(page, axis_y: float):
    """Pixel → value on each axis: each tick label is matched to its nearest tick mark, and a
    straight line is fitted through the ticks' positions (label centres sit up to 0.6 pt off
    their ticks, which is 9 cm⁻¹ in Fig. 7)."""
    tx, ty = _ticks(page)
    xs, ys = [], []
    for w in page.get_text("words"):
        if not (40 < w[1] < 375 and w[0] < 300):
            continue
        try:
            v = float(w[4].replace("−", "-"))
        except ValueError:
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if abs(cy - axis_y) < 4:
            xs.append((tx[np.argmin(np.abs(tx - cx))], v))
        else:
            ys.append((ty[np.argmin(np.abs(ty - cy))], v))
    return np.polyfit(*np.array(xs).T, 1), np.polyfit(*np.array(ys).T, 1)


def read(pdf: Path) -> dict[str, np.ndarray]:
    import pymupdf

    doc = pymupdf.open(pdf)
    out = {}
    for curve, (pno, axis_y, kind) in FIGURES.items():
        page = doc[pno]
        fx, fy = _calibration(page, axis_y)
        pts = []
        for d in page.get_drawings():
            col = tuple(round(c, 2) for c in (d.get("color") or (9, 9, 9)))
            r = d["rect"]
            if col == RED and d["type"] == kind and r.width < 3:
                pts.append((round(float(np.polyval(fx, (r.x0 + r.x1) / 2)), 1),
                            round(float(np.polyval(fy, (r.y0 + r.y1) / 2)), 1)))
        out[curve] = np.array(sorted(pts))
    return out


def yaml_blocks(curves: dict[str, np.ndarray]) -> str:
    lines = ["points_cm1:  # [R a0, E cm-1 from the Na(3s or 3p) + Xe asymptote]"]
    for curve, pts in curves.items():
        rows = ", ".join(f"[{r:.1f}, {e:.1f}]" for r, e in pts)
        lines.append(f"  {curve}: [{rows}]")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--check", action="store_true", help="compare with lss/data/naxe_pairs.yaml")
    args = ap.parse_args(argv)
    curves = read(args.pdf)
    if not args.check:
        print(yaml_blocks(curves))
        return 0
    import yaml

    committed = yaml.safe_load((Path(__file__).parents[1] / "lss/data/naxe_pairs.yaml").read_text(encoding="utf-8"))
    worst = max(float(np.abs(np.array(committed["points_cm1"][c], dtype=float) - curves[c]).max()) for c in curves)
    print(f"largest difference from the committed points: {worst:.2f}")
    return 0 if worst < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
