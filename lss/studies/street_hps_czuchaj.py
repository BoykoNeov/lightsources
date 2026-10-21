"""Czuchaj et al.'s tabulated Na–Hg curves in the nine street lamps (design §4.4, "Czuchaj et
al.'s tabulated Na–Hg curves").

Each lamp is held at its own-mercury fit (:mod:`lss.studies.street_hps_mercury`, central
ratio: temperature, sodium and mercury) and run with Windholz's digitised curves (the fits'
setting), then with Czuchaj's tables, with the table's dipoles and with constant ones. Nothing
is refitted.

Set before running (with κ computed, before any lamp): the 668–678 nm rise +0.40 to +0.70 with
the table's dipoles, +0.25 to +0.41 with constant ones (six lamps +0.16 to +0.41); the model's
red wing 1.2–1.35 times brighter; its slope steeper, by 0.5–1.2 and 0.15–0.5; the red peak
in by 0–0.5 nm, the blue peak unmoved.
"""

from dataclasses import replace
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from lss.accuracy import lica
from lss.studies import street_hps, street_hps_fit as sf, street_hps_mercury as sm, street_hps_xenon as xe

#: (curves, dipole form): Windholz's as fitted, then Czuchaj's with each dipole form.
SETS = (("windholz", "constant"), ("czuchaj", "table"), ("czuchaj", "constant"))
LABELS = {SETS[0]: "Windholz", SETS[1]: "Czuchaj table", SETS[2]: "Czuchaj const"}


def column(lamp: str, curves: str, dipole: str):
    fit = sm.fits()[lamp]["central"]
    c = sm.at(sf.CENTRAL, fit.temperature_k, fit.sodium_pa, sm.ratio_at(lamp, "central"))
    return replace(c, mercury_pairs=replace(c.mercury_pairs, curves=curves, dipole=dipole))


def _run(job):
    lamp, curves, dipole = job
    y = sf.observed(column(lamp, curves, dipole), street_hps.response().kernel)
    x = street_hps.NODES_NM
    out = sf._vector(street_hps.features(x, y))
    out["eq57"] = xe.eq57_of(x, y)
    return out


@lru_cache(maxsize=None)
def runs(workers: int = 16) -> dict[tuple, dict]:
    """(lamp, curves, dipole) -> every feature of a direct run, plus eq. 5.7 read off the model."""
    jobs = [(lamp, *s) for lamp in lica.HPS for s in SETS]
    with Pool(workers) as pool:
        rows = pool.map(_run, jobs)
    return dict(zip(jobs, rows))


def stand_in(temperature_k: float, curves: str, dipole: str) -> dict[str, float]:
    """The stand-in the lamp predictions rested on (design §4.4): the Na–Hg pair κ alone, times
    a Planck source, through the instrument's response at the 1 nm nodes, with the study's
    measures. Its rise with Windholz's curves (0.00–0.03 at 3600–4600 K) matched the lamp runs'."""
    from lss.physics.hid import mercury_pairs as mp
    from lss.spectral.instrument import observe
    from lss.spectral.units import C, H, K_B, vacuum_to_air

    edges = mp.TABLE_EDGES_CM1
    lam = vacuum_to_air(1e7 / (0.5 * (edges[1:] + edges[:-1])))[::-1]
    planck = 1.0 / (lam * 1e-9) ** 5 / np.expm1(H * C / (lam * 1e-9 * K_B * temperature_k))
    kappa = sum(mp.reduced_absorption(temperature_k, edges, dipole=dipole, curves=curves).values())[::-1]
    x = street_hps.NODES_NM
    f = street_hps.features(x, observe(lam, kappa * planck, street_hps.response().kernel, x))
    return {"rise": f.red_bump, "slope": f.red_slope}


def level_ratio(s: tuple) -> tuple[float, float]:
    """Lamp/model red wing: geometric mean over the nine and the standard deviation of its log."""
    r, m = runs(), xe.measured()
    logs = np.array([m[l]["red wing"] - r[(l, *s)]["red wing"] for l in lica.HPS])
    return float(np.exp(logs.mean())), float(logs.std())


def format_text() -> str:
    r, m = runs(), xe.measured()
    out = ["Czuchaj et al.'s Na-Hg curves at each lamp's own-mercury fit (temperature, sodium, mercury held).",
           "Windholz (as fitted) / Czuchaj, table dipoles / Czuchaj, constant dipoles; red wing over the D band's top.",
           ""]
    keys = ("red peak nm", "spacing nm", "eq.5.7 on model", "red wing", "red slope", "668-678 rise", "616 line")
    out.append(f"{'lamp':28s}{'curves':>15s}" + "".join(f"{k:>16s}" for k in keys))
    for lamp in lica.HPS:
        for s in SETS:
            a = r[(lamp, *s)]
            cells = [f"{a['blue peak'] + a['separation']:.2f}", f"{a['separation']:.2f}", f"{a['eq57']:.1f}",
                     f"{np.exp(a['red wing']):.4f}", f"{a['red slope']:.3f}", f"{a['red bump']:.3f}",
                     f"{np.exp(a['616']):.4f}"]
            out.append(f"{lamp if s == SETS[0] else '':28s}{LABELS[s]:>15s}" + "".join(f"{c:>16s}" for c in cells))
        mm = m[lamp]
        out.append(f"{'  lamp':28s}{'':15s}{mm['blue peak'] + mm['separation']:>16.2f}{mm['separation']:>16.2f}"
                   f"{sm.ratios()[lamp].central:>16.1f}{np.exp(mm['red wing']):>16.4f}{mm['red slope']:>16.3f}"
                   f"{mm['red bump']:>16.3f}{np.exp(mm['616']):>16.4f}")
    out += ["", "Against Windholz's curves, range over the nine lamps:"]
    for s in SETS[1:]:
        def rng(g):
            v = [g(r[(l, *s)], r[(l, *SETS[0])]) for l in lica.HPS]
            return f"{min(v):+.3f} to {max(v):+.3f}"
        out.append(f"  {LABELS[s]}: rise {rng(lambda b, a: b['red bump'] - a['red bump'])}; "
                   f"red wing x(1{rng(lambda b, a: np.exp(b['red wing'] - a['red wing']) - 1)}); "
                   f"slope {rng(lambda b, a: b['red slope'] - a['red slope'])}; "
                   f"red peak {rng(lambda b, a: b['blue peak'] + b['separation'] - a['blue peak'] - a['separation'])} nm; "
                   f"blue peak {rng(lambda b, a: b['blue peak'] - a['blue peak'])} nm")
    out += ["", "Lamp/model red wing, geometric mean over the nine (sd of log):"]
    for s in SETS:
        g, sd = level_ratio(s)
        out.append(f"  {LABELS[s]:15s} {g:.2f} ({sd:.2f})")
    return "\n".join(out)
