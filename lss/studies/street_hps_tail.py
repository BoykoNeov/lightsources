"""The D1 band's long-range tail in the nine street lamps (design §4.4, "The I½ tail, bracketed").

Beyond the digitised Na–Hg grid (9.95 a₀) the I ½ curve, which supplies D1's red side out to
15–20 nm, joins its asymptote by a fit whose A₆ is 2.5–3.1 times Unsöld's ΔC₆. The
alternative (``tail="dispersion"``) puts Unsöld's ΔC₆ there with the calculation's extra
binding fading exponentially. Each lamp is held at its own-mercury fit
(:mod:`lss.studies.street_hps_mercury`, central ratio: temperature, sodium and mercury) and
run with ΔC₆ × 1 (the fits' setting) and × 2, each with both tails.

What was set before running: eq. 5.7 applied to the model's own spectrum, dispersion tail over
fitted, measures how much of each lamp's mercury hangs on the tail. Under 15 % for every lamp:
bounded. Over 15 % for any: the tail is a leading uncertainty on the spacing and on the
mercury read from it. Also reported: the red peak, the spacing, the red wing, its slope and
the 668–678 nm rise.
"""

from dataclasses import replace
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from lss.accuracy import lica
from lss.studies import street_hps, street_hps_fit as sf, street_hps_mercury as sm, street_hps_xenon as xe

FACTORS = (1.0, 2.0)
TAILS = ("fitted", "dispersion")
#: The verdict's line, set before running: eq. 5.7 on the model moving by this much or more.
BOUND = 0.15


def column(lamp: str, delta_c6_factor: float, tail: str):
    fit = sm.fits()[lamp]["central"]
    c = sm.at(sf.CENTRAL, fit.temperature_k, fit.sodium_pa, sm.ratio_at(lamp, "central"))
    return replace(c, mercury_pairs=replace(c.mercury_pairs, delta_c6_factor=float(delta_c6_factor), tail=tail))


def _run(job):
    lamp, factor, tail = job
    y = sf.observed(column(lamp, factor, tail), street_hps.response().kernel)
    x = street_hps.NODES_NM
    out = sf._vector(street_hps.features(x, y))
    out["eq57"] = xe.eq57_of(x, y)
    out["red share"] = sm.red_share(x, y)
    return out


@lru_cache(maxsize=None)
def runs(workers: int = 16) -> dict[tuple, dict]:
    """(lamp, ΔC₆ factor, tail) -> every feature of a direct run, plus eq. 5.7 read off the
    model's spectrum and the D band's red share."""
    jobs = [(lamp, f, t) for lamp in lica.HPS for f in FACTORS for t in TAILS]
    with Pool(workers) as pool:
        rows = pool.map(_run, jobs)
    return dict(zip(jobs, rows))


def eq57_change(lamp: str, factor: float) -> float:
    """eq. 5.7 on the model with the dispersion tail over the fitted one, minus one."""
    r = runs()
    return r[(lamp, factor, "dispersion")]["eq57"] / r[(lamp, factor, "fitted")]["eq57"] - 1.0


def format_text() -> str:
    r, m = runs(), xe.measured()
    out = ["The D1 band's tail at each lamp's own-mercury fit (temperature, sodium, mercury held).",
           "fitted / dispersion tail; red wing over the D band's top.", ""]
    head = f"{'lamp':28s}{'dC6':>5s}" + "".join(f"{k:>16s}" for k in ("red peak nm", "spacing nm", "eq.5.7 on model",
                                                                     "red wing", "red slope", "668-678 rise"))
    out.append(head)
    for lamp in lica.HPS:
        for f in FACTORS:
            a, b = r[(lamp, f, "fitted")], r[(lamp, f, "dispersion")]
            cells = [f"{a['blue peak'] + a['separation']:.2f}/{b['blue peak'] + b['separation']:.2f}",
                     f"{a['separation']:.2f}/{b['separation']:.2f}", f"{a['eq57']:.1f}/{b['eq57']:.1f}",
                     f"{np.exp(a['red wing']):.4f}/{np.exp(b['red wing']):.4f}",
                     f"{a['red slope']:.3f}/{b['red slope']:.3f}", f"{a['red bump']:.3f}/{b['red bump']:.3f}"]
            out.append(f"{lamp if f == FACTORS[0] else '':28s}{f:5.0f}" + "".join(f"{c:>16s}" for c in cells))
        mm = m[lamp]
        out.append(f"{'  lamp':28s}{'':5s}{mm['blue peak'] + mm['separation']:>16.2f}{mm['separation']:>16.2f}"
                   f"{sm.ratios()[lamp].central:>16.1f}{np.exp(mm['red wing']):>16.4f}{mm['red slope']:>16.3f}"
                   f"{mm['red bump']:>16.3f}")
    out += ["", "Dispersion over fitted tail, range over the nine lamps:"]
    for f in FACTORS:
        def rng(g):
            v = [g(r[(l, f, "dispersion")], r[(l, f, "fitted")]) for l in lica.HPS]
            return f"{min(v):+.3f} to {max(v):+.3f}"
        ch = [eq57_change(l, f) for l in lica.HPS]
        out.append(f"  dC6 x{f:.0f}: eq. 5.7 on the model {min(ch):+.1%} to {max(ch):+.1%}; "
                   f"spacing {rng(lambda b, a: b['separation'] - a['separation'])} nm; "
                   f"blue peak {rng(lambda b, a: b['blue peak'] - a['blue peak'])} nm; "
                   f"red wing x(1{rng(lambda b, a: np.exp(b['red wing'] - a['red wing']) - 1)}); "
                   f"slope {rng(lambda b, a: b['red slope'] - a['red slope'])}; "
                   f"rise {rng(lambda b, a: b['red bump'] - a['red bump'])}")
        over = [l for l, c in zip(lica.HPS, ch) if abs(c) >= BOUND]
        out.append(f"           at or over {BOUND:.0%}: {', '.join(over) if over else 'none'}")
    return "\n".join(out)
