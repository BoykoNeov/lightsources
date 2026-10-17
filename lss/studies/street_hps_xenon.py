"""Sodium–xenon pairs in the nine street lamps (design §4.4, rung 2, third part).

Each lamp is held at its fitted point with its own mercury (:mod:`lss.studies.street_hps_mercury`,
whole-band fit at the central ratio): axis temperature, sodium and mercury as fitted. The
lamps' xenon is not known (the book: 10–30 kPa in operation for xenon as a starting gas), so
each is run at 10, 20 and 30 kPa, with and without xenon pairs. The fill also sets xenon's
impact width, so the pairs are judged at matched fill; the fits used 20 kPa without them.

What was set before running:

- *The double count.* Each lamp's mercury comes from eq. 5.7 on its own measured peaks, which
  the model's xenon cannot change. What can double-count is the model's own red peak: eq. 5.7
  applied to the model's spectrum, with and without xenon pairs, measures how much mercury the
  xenon stands in for. Under 15 % for every lamp: bounded, nothing refitted.
- *The blue test:* the mean over 554–566 nm over the D band's highest sample, model against lamp
  (between the Na–Na satellite at 551.5 nm, the Hg 546 nm line and the Na 568 nm lines). The
  model's classical satellite is a narrow spike at 555 nm, 5 nm short of the measured 560 nm, so
  a pass or a miss here is partly its position.
- *Also reported:* the red wing (not scored), the red slope, the 668–678 nm rise, and the blue
  wing at 525–540 nm.
"""

from dataclasses import replace
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from lss.accuracy import lica
from lss.physics.hid import xenon_pairs
from lss.studies import street_hps, street_hps_fit as sf, street_hps_mercury as sm

XENON_KPA = (10.0, 20.0, 30.0)
GREEN_NM = (554.0, 566.0)


def green(x, y) -> float:
    """The mean over :data:`GREEN_NM` over the D band's highest sample."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    top = y[(x >= street_hps.D_BAND_NM[0]) & (x <= street_hps.D_BAND_NM[1])].max()
    return float(y[(x >= GREEN_NM[0]) & (x <= GREEN_NM[1])].mean() / top)


def eq57_of(x, y) -> float:
    """eq. 5.7 read off a spectrum the way each lamp's is (:func:`street_hps_mercury.ratio`,
    central: D midpoint, the LPS offset and the whole blur shift undone)."""
    f = street_hps.features(x, y)
    sb, sr = sm.blur_shift_nm(x, y)
    off, _ = sf.lps_offset_nm()
    return sm.eq57(f.blue_peak_nm - sb - off, f.red_peak_nm - sr - off, sm.D_CENTRES_NM[0])


def column(lamp: str, xenon_kpa: float, pairs: bool):
    fit = sm.fits()[lamp]["central"]
    c = sm.at(sf.CENTRAL, fit.temperature_k, fit.sodium_pa, sm.ratio_at(lamp, "central"))
    c = replace(c, fill=replace(c.fill, xenon_pa=float(xenon_kpa) * 1e3))
    if pairs:
        c = replace(c, xenon_pairs=xenon_pairs.XenonPairWings(join_nm=c.mercury_pairs.join_nm))
    return c


def _run(job):
    lamp, xe, pairs = job
    y = sf.observed(column(lamp, xe, pairs), street_hps.response().kernel)
    x = street_hps.NODES_NM
    out = sf._vector(street_hps.features(x, y))
    out["green"] = green(x, y)
    out["eq57"] = eq57_of(x, y)
    return out


@lru_cache(maxsize=None)
def runs(workers: int = 16) -> dict[tuple, dict]:
    """(lamp, xenon kPa, pairs) -> every feature of a direct run, plus the green window and
    eq. 5.7 read off the model's spectrum."""
    jobs = [(lamp, xe, p) for lamp in lica.HPS for xe in XENON_KPA for p in (False, True)]
    with Pool(workers) as pool:
        rows = pool.map(_run, jobs)
    return dict(zip(jobs, rows))


@lru_cache(maxsize=None)
def measured() -> dict[str, dict]:
    out = {}
    for lamp in lica.HPS:
        m = lica.load(lamp)
        out[lamp] = sf._vector(street_hps.features(m.wavelengths_nm, m.values))
        out[lamp]["green"] = green(m.wavelengths_nm, m.values)
    return out


def eq57_change(lamp: str, xenon_kpa: float) -> float:
    """eq. 5.7 on the model with xenon pairs over without, minus one, at matched fill."""
    r = runs()
    return r[(lamp, xenon_kpa, True)]["eq57"] / r[(lamp, xenon_kpa, False)]["eq57"] - 1.0


def format_text() -> str:
    r, m = runs(), measured()
    out = ["Sodium-xenon pairs at each lamp's fitted point (own mercury; temperature, sodium, mercury held).",
           "Xenon 10, 20, 30 kPa, without / with xenon pairs. Red wing and blue wings over the D band's top.", ""]
    head = f"{'lamp':28s}{'Xe':>4s}" + "".join(f"{k:>18s}" for k in ("red wing", "red slope", "668-678 rise",
                                                                    "525-540 nm", "554-566 nm", "eq.5.7 on model"))
    out.append(head)
    for lamp in lica.HPS:
        for xe in XENON_KPA:
            a, b = r[(lamp, xe, False)], r[(lamp, xe, True)]
            cells = [f"{np.exp(a['red wing']):.4f}/{np.exp(b['red wing']):.4f}",
                     f"{a['red slope']:.3f}/{b['red slope']:.3f}", f"{a['red bump']:.3f}/{b['red bump']:.3f}",
                     f"{np.exp(a['blue wing']):.4f}/{np.exp(b['blue wing']):.4f}",
                     f"{a['green']:.4f}/{b['green']:.4f}", f"{a['eq57']:.1f}/{b['eq57']:.1f}"]
            out.append(f"{lamp if xe == XENON_KPA[0] else '':28s}{xe:4.0f}" + "".join(f"{c:>18s}" for c in cells))
        mm = m[lamp]
        out.append(f"{'  lamp':28s}{'':4s}{np.exp(mm['red wing']):>18.4f}{mm['red slope']:>18.3f}{mm['red bump']:>18.3f}"
                   f"{np.exp(mm['blue wing']):>18.4f}{mm['green']:>18.4f}{sm.ratios()[lamp].central:>18.1f}")
    out += ["", "The double count: eq. 5.7 on the model, with xenon pairs over without, minus one (matched fill)."]
    for xe in XENON_KPA:
        ch = [eq57_change(lamp, xe) for lamp in lica.HPS]
        out.append(f"  {xe:4.0f} kPa: " + ", ".join(f"{c:+.1%}" for c in ch) + f"   largest {max(ch, key=abs):+.1%}")
    out += ["", "What the pairs change, with over without, at matched fill (range over the nine lamps):"]
    for xe in XENON_KPA:
        def rng(f):
            v = [f(r[(lamp, xe, True)], r[(lamp, xe, False)]) for lamp in lica.HPS]
            return f"{min(v):+.3f} to {max(v):+.3f}"
        out.append(f"  {xe:4.0f} kPa: red wing x(1{rng(lambda b, a: np.exp(b['red wing'] - a['red wing']) - 1)}), "
                   f"slope {rng(lambda b, a: b['red slope'] - a['red slope'])}, "
                   f"rise {rng(lambda b, a: b['red bump'] - a['red bump'])}, "
                   f"525-540 x(1{rng(lambda b, a: np.exp(b['blue wing'] - a['blue wing']) - 1)}), "
                   f"554-566 x(1{rng(lambda b, a: b['green'] / a['green'] - 1)})")
    return "\n".join(out)
