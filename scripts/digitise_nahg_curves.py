"""Reproduce the digitised NaHg difference potentials in ``lss/data/nahg_pairs.yaml`` from
Windholz, Zerza, Pichler & Heß, Z. Phys. D 18, 373 (1991), Fig. 8 (design §4.4, rung 2).

The paper does not tabulate its curves, and its PDF is a scan: Fig. 8 is a raster image.
So the page is rendered at 300 dpi, the axes are calibrated on the frame's tick marks
(both sides, so the scan's slight tilt is taken out), each drawn line is traced column by
column, and the plotted circles (the computed points) are centred with a ring template.

The paper is not openly available, so the PDF is not downloaded here: pass the path to a
copy. Tool-only dependency, not needed by the package: poppler's ``pdftoppm`` on the PATH.

    python scripts/digitise_nahg_curves.py windholz1991.pdf          # prints the YAML blocks
    python scripts/digitise_nahg_curves.py windholz1991.pdf --check  # compares with the committed YAML
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicHermiteSpline, LSQUnivariateSpline, PchipInterpolator

#: Fig. 8 on page 4, at 300 dpi: the crop's corner and size in pixels.
CROP = dict(x=330, y=2000, W=720, H=1000)
R_TICKS = np.arange(3, 12)  # the R axis, 3 to 11 a0, a tick every a0
E_TICKS = (20000, 15000, 10000, 5000)  # labelled energy ticks, cm⁻¹
#: Where each line is picked up, just right of its first circle (x, y in crop pixels, initial
#: slope in px/px), and the R range kept. The three lines are traced in this order, each with the
#: earlier ones masked out. I½ is traced piecewise, restarting at each of its circles, because it
#: bends too fast near 5.5–6 a0 for one pass.
SEEDS = {"II1/2": (110, 336.0, 1.0), "I3/2": (110, 825.0, -3.0)}
I12_PIECES = [(110, 952.0, -3.0, 172), (190, 752.0, -3.0, 215), (233, 625.0, -2.4, 255),
              (276, 531.0, -1.75, 423), (441, 313.0, -0.69, 465), (483, 288.5, -0.17, 506),
              (524, 280.6, -0.27, 547), (565, 270.1, -0.21, 590)]
CIRCLE_R_A0 = (4.0, 5.0, 5.5, 6.0, 8.0, 8.5, 9.0, 9.5, 10.0)
CIRCLE_GUESS = {
    "II1/2": [(99, 325), (181, 354), (224, 334), (265, 315), (431, 283), (473, 272), (514, 263), (556, 250), (597, 244)],
    "I3/2": [(98, 844), (181, 678), (224, 568), (265, 484), (431, 296)],
    "I1/2": [(181, 779), (224, 645), (264, 553), (431, 319), (473, 290), (514, 283), (556, 271), (598, 263)],
}
#: Output grids, a0. I½ leaves the frame (below 4000 cm⁻¹) just inside 4.05 a0; the last
#: circles are at 9.96–9.98 a0, so the grids stop at 9.95.
GRID_A0 = {"II1/2": (4.00, 9.95), "I3/2": (4.00, 9.95), "I1/2": (4.10, 9.95)}
STEP_A0 = 0.05


def render(pdf: Path, out: Path) -> np.ndarray:
    stem = out / "fig8"
    subprocess.run(["pdftoppm", "-f", "4", "-l", "4", "-r", "300", "-gray", "-x", str(CROP["x"]), "-y", str(CROP["y"]),
                    "-W", str(CROP["W"]), "-H", str(CROP["H"]), str(pdf), str(stem)], check=True)
    raw = next(out.glob("fig8*.pgm")).read_bytes()
    fields, i = [], 0
    while len(fields) < 4:
        while raw[i:i + 1].isspace():
            i += 1
        j = i
        while not raw[j:j + 1].isspace():
            j += 1
        fields.append(raw[i:j])
        i = j
    w, h = int(fields[1]), int(fields[2])
    return np.frombuffer(raw[i + 1:i + 1 + w * h], dtype=np.uint8).reshape(h, w)


def _runs(idx) -> list[float]:
    idx = np.asarray(idx)
    return [float(r.mean()) for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)] if idx.size else []


class Calibration:
    """Pixel → (R, E) from the frame's ticks: R from the bottom ticks; E interpolated
    across the frame between the left and right ticks (the scan is tilted by about 0.1°)."""

    def __init__(self, dark: np.ndarray):
        h, w = dark.shape
        cols, rows = dark.sum(0), dark.sum(1)
        self.left, self.right = int(np.argmax(cols[: w // 2])), w // 2 + int(np.argmax(cols[w // 2:]))
        bottom = h // 2 + int(np.argmax(rows[h // 2:]))
        bt = _runs(np.where(dark[bottom - 16:bottom - 3, :].sum(0) >= 10)[0])
        lt = _runs(np.where(dark[:, self.left + 3:self.left + 15].sum(1) >= 10)[0])
        rt = _runs(np.where(dark[:, self.right - 12:self.right - 2].sum(1) >= 8)[0])
        if len(bt) != len(R_TICKS):
            raise RuntimeError(f"expected {len(R_TICKS)} R ticks, found {bt}")
        self.x_fit = np.polyfit(bt, R_TICKS, 1)
        # labelled ticks only: every 5000 cm⁻¹ on the left; the right side has one every 1000
        lt = [y for y in lt if 20 < y < bottom - 20]
        rt = [y for y in rt if 20 < y < bottom - 20]
        step = (lt[-1] - lt[0]) / (len(E_TICKS) - 1)
        rt = [min(rt, key=lambda y: abs(y - (lt[0] + k * step))) for k in range(len(E_TICKS))]
        self.e_left, self.e_right = np.polyfit(lt, E_TICKS, 1), np.polyfit(rt, E_TICKS, 1)

    def r(self, x):
        return np.polyval(self.x_fit, x)

    def x(self, r):
        return (np.asarray(r) - self.x_fit[1]) / self.x_fit[0]

    def e(self, x, y):
        f = (np.asarray(x) - self.left) / (self.right - self.left)
        return (1 - f) * np.polyval(self.e_left, y) + f * np.polyval(self.e_right, y)


def track(dark, circle_x, x0, y0, slope0, x_end, win=6, skip=9) -> np.ndarray:
    """Follow a line rightwards: in each column take the dark run nearest the prediction
    from the recent slope; bridge across circles and anything too thick to be the line."""
    pts, y, s, x = [(x0, y0)], y0, slope0, x0
    while x < x_end:
        x += 1
        yp = y + s
        if any(abs(x - cx) <= skip for cx in circle_x):
            y = yp
            continue
        lo, hi = int(yp - win - abs(s)), int(yp + win + abs(s)) + 1
        col = np.where(dark[lo:hi, x])[0] + lo
        if col.size == 0:
            y = yp
            continue
        runs = np.split(col, np.where(np.diff(col) > 1)[0] + 1)
        r = min(runs, key=lambda r: abs(r.mean() - yp))
        if r.size > 5 + 1.5 * abs(s) or abs(r.mean() - yp) > win:
            y = yp
            continue
        y = float(r.mean())
        pts.append((x, y))
        recent = np.array(pts[-10:])
        if len(recent) >= 4:
            s = np.polyfit(recent[:, 0], recent[:, 1], 1)[0]
    return np.array(pts)


def circle_centres(dark, guesses, w=7) -> np.ndarray:
    score = None
    for rad in (5.5, 6.0, 6.5):
        ring = np.zeros((19, 19))
        for t in np.linspace(0, 2 * np.pi, 64, endpoint=False):
            ring[int(round(9 + rad * np.sin(t))), int(round(9 + rad * np.cos(t)))] = 1
        sc = ndimage.correlate(dark.astype(float), ring / ring.sum())
        score = sc if score is None else np.maximum(score, sc)
    out = []
    for x, y in guesses:
        sub = score[y - w:y + w + 1, x - w:x + w + 1]
        j, i = np.unravel_index(np.argmax(sub), sub.shape)
        out.append((x - w + i, y - w + j))
    return np.array(out, dtype=float)


def digitise(pdf: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        grey = render(pdf, Path(tmp))
    dark = grey < 140
    cal = Calibration(dark)
    circle_x = [float(cal.x(r)) for r in CIRCLE_R_A0]

    def mask(img, pts, half=5):
        m = img.copy()
        for x, y in pts:
            m[max(0, int(y) - half):int(y) + half + 1, int(x)] = False
        return m

    traces = {"II1/2": track(dark, circle_x, *SEEDS["II1/2"], 600)}
    traces["I3/2"] = track(dark, circle_x, *SEEDS["I3/2"], 600, win=4)
    masked = mask(mask(dark, traces["II1/2"]), traces["I3/2"])
    pieces = [track(masked, circle_x, x0, y0, s0, x1, win=5 if x0 < 270 else 6) for x0, y0, s0, x1 in I12_PIECES]
    traces["I1/2"] = np.vstack(pieces)

    out, shared = {}, None
    for name, p in traces.items():
        c = circle_centres(dark, CIRCLE_GUESS[name])
        if name == "II1/2":
            shared = c[cal.r(c[:, 0]) > 8.4]  # beyond 8.5 a0 II½ and I3/2 are drawn as one line
        r_line, e_line = cal.r(p[:, 0]), cal.e(p[:, 0], p[:, 1])
        r_c, e_c = cal.r(c[:, 0]), cal.e(c[:, 0], c[:, 1])
        r_all = np.concatenate([r_line, r_c])
        e_all = np.concatenate([e_line, e_c])
        wt = np.concatenate([np.ones_like(r_line), np.full_like(r_c, 5.0)])  # the computed points count more
        o = np.argsort(r_all)
        r_all, e_all, wt = r_all[o], e_all[o], wt[o]
        keep = np.concatenate([[True], np.diff(r_all) > 1e-9])
        r_all, e_all, wt = r_all[keep], e_all[keep], wt[keep]
        lo, hi = GRID_A0[name]
        knots = np.arange(np.ceil((r_all[0] + 0.1) / 0.25) * 0.25, r_all[-1] - 0.1, 0.25)
        spline = LSQUnivariateSpline(r_all, e_all, knots, w=wt, k=3)
        grid = np.round(np.arange(lo, hi + 1e-9, STEP_A0), 2)
        values = spline(grid)
        # Beyond the computed point at 8 a0 the drawing has kinks at the avoided crossing (8.5 a0):
        # near-flat stretches, and on I½ a dip of a few cm⁻¹, which the classical sum would turn
        # into spurious satellites. The paper states that only II½ − X½ has an extremum, so there
        # the curve is a monotone cubic (PCHIP) through the computed points instead, starting from
        # the drawn line at the 8 a0 point.
        pts = np.array([c for c in zip(r_c, e_c) if c[0] > 7.9])
        if name == "I3/2":
            pts = np.vstack([pts, np.column_stack([cal.r(shared[:, 0]), cal.e(shared[:, 0], shared[:, 1])])])
        nodes_r = np.concatenate([[pts[0, 0]], pts[1:, 0]])
        nodes_e = np.concatenate([[spline(pts[0, 0])], pts[1:, 1]])
        beyond = grid >= nodes_r[0]
        # Monotone slopes at the nodes, except the first, which takes the drawn line's so that the
        # slope is continuous there (a slope step makes a step in the classical spectrum).
        slopes = PchipInterpolator(nodes_r, nodes_e).derivative()(nodes_r)
        slopes[0] = spline.derivative()(nodes_r[0])
        values[beyond] = CubicHermiteSpline(nodes_r, nodes_e, slopes, extrapolate=True)(grid[beyond])
        values = np.round(values)
        out[name] = {
            "grid": grid, "values": values, "rms": float(np.sqrt(np.mean((spline(r_line) - e_line) ** 2))),
            "circles": [(round(float(a), 3), round(float(b))) for a, b in zip(r_c, e_c)],
        }
    return out


def yaml_blocks(curves: dict) -> str:
    lines = []
    for name, d in curves.items():
        lo, hi = GRID_A0[name]
        lines.append(f"  {name}:  # R = {lo:.2f}, {lo + STEP_A0:.2f}, ... {hi:.2f} a0; spline rms {d['rms']:.0f} cm-1 against the trace")
        vals = [f"{v:.0f}" for v in d["values"]]
        for k in range(0, len(vals), 12):
            lines.append("    - [" + ", ".join(vals[k:k + 12]) + "]")
    lines.append("circles_cm1:  # the plotted (computed) points, [R a0, E cm-1] as read")
    for name, d in curves.items():
        lines.append(f"  {name}: " + "[" + ", ".join(f"[{a}, {b}]" for a, b in d["circles"]) + "]")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", type=Path, help="a copy of Windholz et al., Z. Phys. D 18, 373 (1991)")
    ap.add_argument("--check", action="store_true", help="compare with lss/data/nahg_pairs.yaml")
    args = ap.parse_args(argv)
    curves = digitise(args.pdf)
    if not args.check:
        print(yaml_blocks(curves))
        return 0
    import yaml
    root = Path(__file__).resolve().parents[1]
    doc = yaml.safe_load((root / "lss" / "data" / "nahg_pairs.yaml").read_text(encoding="utf-8"))
    worst = 0.0
    for name, d in curves.items():
        committed = np.concatenate([np.array(row, dtype=float) for row in doc["difference_potentials_cm1"][name]])
        diff = float(np.max(np.abs(committed - d["values"])))
        worst = max(worst, diff)
        print(f"{name}: max |committed - digitised| = {diff:.0f} cm-1")
    return 0 if worst <= 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
