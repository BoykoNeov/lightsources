"""Reproduce the Fig. 5.9b window values in ``tests/test_hid_xenon_pairs.py`` (design §4.4,
rung 2, third part): de Groot & van Vliet, *The High-Pressure Sodium Lamp* (1986), Fig. 5.9b,
k(λ)/([Na][Xe]) in 1e-47 m⁵: Schlejen & Woerdman's calculation at 2000 and 4000 K, and Jongerius
& Ras's spectrum derived from Na–Hg–Xe discharges at 3000 K.

The book's PDF is a scan, so the figure is a raster image. The page (PDF page 142, book page
138) was rendered at 400 dpi with poppler's ``pdftoppm``; the axes are calibrated on the frame
(bottom 0 at y = 2421 px, top 3 at 1510.5 px) and the x ticks (400 nm at 600 px, 800 nm at 1977
px). Each wavelength column was then listed as its runs of dark pixels (``--runs``), and the
runs were assigned to the three curves by hand, using their order (on the red side: derived >
4000 K > 2000 K) and continuity, and skipping the axis label and legend: the tables below.
Reading error about ±0.02. Where a dashed curve has a gap the entry is None. The calculated
curves end (fall to 0) at about 678 nm (2000 K) and 688 nm (4000 K). On the blue side the
2000 K branch cannot be told from the 4000 K one beyond 562 nm, so it is not used.

The book is not openly available, so the PDF is not downloaded here: pass the path to a copy.

    python scripts/digitise_fig59b.py                    # prints the window values the tests use
    python scripts/digitise_fig59b.py --runs book.pdf    # lists the dark runs per 2 nm column
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: nm: (derived 3000 K, calculated 4000 K, calculated 2000 K); None where a dash has a gap.
RED = {
 610:(1.313,1.245,1.132),612:(1.176,1.124,0.997),614:(1.061,0.924,0.82),616:(0.967,0.843,0.755),618:(0.878,0.768,0.651),
 620:(0.817,0.707,0.586),622:(0.759,0.651,0.507),624:(0.712,0.601,0.476),626:(None,0.557,0.432),628:(0.649,0.521,None),
 630:(0.603,0.484,0.362),632:(0.557,0.453,0.331),634:(None,0.42,0.305),636:(0.502,0.389,0.282),638:(None,0.357,None),
 640:(0.451,0.334,0.229),642:(None,0.31,0.211),644:(0.405,0.29,0.196),646:(0.392,0.27,0.18),648:(None,0.249,None),
 650:(0.349,0.229,None),652:(0.338,0.213,0.132),654:(0.311,0.198,0.119),656:(0.30,0.183,0.105),658:(None,0.17,None),
 660:(0.264,0.158,None),662:(0.255,0.148,0.077),664:(None,0.138,0.066),666:(0.226,0.129,0.056),668:(0.221,0.12,0.048),
 670:(None,0.112,None),672:(0.198,0.105,None),674:(0.191,0.097,0.026),676:(None,0.10,0.021),678:(0.171,0.107,0.016),
 680:(None,0.102,0.0),682:(None,0.087,0.0),684:(0.15,0.071,0.0),686:(0.143,0.049,0.0),688:(None,0.026,0.0),690:(0.127,0.0,0.0),
 692:(0.124,0.0,0.0),698:(0.105,0.0,0.0),700:(0.104,0.0,0.0),706:(0.091,0,0),712:(0.077,0,0),720:(0.064,0,0),728:(0.054,0,0),
 740:(0.038,0,0),750:(0.028,0,0)}
#: Derived (3000 K), blue side.
BLUE_DERIVED = {549:0.02,550:0.036,551:0.063,552:0.12,553:0.16,554:0.27,555:0.39,556:0.75,557:1.35,558:1.7,559:1.93,560:1.97,560.5:2.02,
 561:1.85,562:1.6,563:1.45,564:1.35,565:1.318,566:1.269,567:1.269,568:1.27,569:1.318,570:1.35,571:1.41,572:1.475,573:1.547,
 574:1.62,575:1.74,576:1.81,577:1.92,578:2.03,579:2.18,580:2.33}
#: Calculated 4000 K, blue side.
BLUE_4000 = {536:0.0,537:0.021,538:0.035,539:0.058,540:0.084,541:0.127,542:0.17,543:0.277,544:0.387,545:0.644,546:1.3,547:1.532,548:1.9,548.3:2.0,
 549:1.6,550:1.40,551:1.30,552:1.196,553:1.166,554:1.15,555:1.145,556:1.147,557:1.155,558:1.165,559:1.178,560:1.196,561:1.232,
 562:1.264,563:1.305,564:1.353,565:1.425,566:1.484,567:1.554,568:1.629,569:1.699,570:1.792,571:1.855,572:1.952,573:2.035,574:2.112,
 575:2.219,576:2.298,577:2.353}

X0_PX, PX_PER_NM = 1291.0, (1977 - 600) / 400.0  # 600 nm, from the 400 and 800 nm ticks
Y0_PX, PX_PER_UNIT = 2421.0, (2421 - 1510.5) / 3.0


def _series(d, i=None):
    pts = sorted((k, v if i is None else v[i]) for k, v in d.items())
    return np.array([(k, v) for k, v in pts if v is not None]).T


def mean(d, lo, hi, i=None) -> float:
    x, y = _series(d, i)
    g = np.linspace(lo, hi, 2001)
    return float(np.trapezoid(np.interp(g, x, y), g) / (hi - lo))


def integral(d, lo, hi, i=None) -> float:
    x, y = _series(d, i)
    g = np.linspace(lo, hi, 2001)
    return float(np.trapezoid(np.interp(g, x, y, left=0.0), g))


def runs(pdf: Path, lo_nm: float = 530.0, hi_nm: float = 760.0, step_nm: float = 2.0) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-f", "142", "-l", "142", "-r", "400", "-gray", str(pdf), str(stem)], check=True)
        raw = next(Path(tmp).glob("p*.pgm")).read_bytes()
    parts = raw.split(b"\n", 3)
    w, h = map(int, parts[1].split())
    img = np.frombuffer(parts[3][: w * h], np.uint8).reshape(h, w)
    for lam in np.arange(lo_nm, hi_nm + 1e-9, step_nm):
        x = int(round(X0_PX + (lam - 600.0) * PX_PER_NM))
        idx = np.where(img[1515:2417, x - 1:x + 2].min(1) < 140)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if idx.size else []
        print(f"{lam:6.1f}", [round(float((Y0_PX - (1515 + g.mean())) / PX_PER_UNIT), 3) for g in groups])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, metavar="PDF", help="list the dark runs per column from a copy of the book")
    args = ap.parse_args(argv)
    if args.runs:
        runs(args.runs)
        return 0
    for name, i, t in (("calculated", 2, 2000), ("derived", 0, 3000), ("calculated", 1, 4000)):
        print(f"{t} K {name}: mean 640-700 nm {mean(RED, 640, 700, i):.3f}, 640-670 nm {mean(RED, 640, 670, i):.3f}")
    print(f"integral 540-577 nm: derived (3000 K) {integral(BLUE_DERIVED, 540, 577):.1f}, "
          f"calculated (4000 K) {integral(BLUE_4000, 540, 577):.1f}")
    x, y = _series(BLUE_DERIVED)
    band = x < 566.0  # beyond the satellite's dip the curve climbs to the lines
    print(f"derived satellite: {x[band][np.argmax(y[band])]:.1f} nm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
