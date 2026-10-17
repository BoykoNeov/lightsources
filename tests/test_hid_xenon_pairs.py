"""Sodium–xenon pairs, the xenon share of the D lines' far wings (design §4.4, rung 2 third
part): the curves against their source, the extrapolated wall, the comparison with the
book's Fig. 5.9b against the mark set before it was run, and the swap into the arc."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import voigt_profile

from lss.physics.hid import arc, broadening, sodium, xenon_pairs as xp
from lss.physics.hid.composition import Fill, composition
from lss.spectral.units import C, H, K_B, air_to_vacuum, vacuum_to_air

HPS_FILL = Fill(sodium_pa=13e3, mercury_pa=200e3, xenon_pa=20e3)

#: de Groot & van Vliet, The High-Pressure Sodium Lamp (1986), Fig. 5.9b: k/([Na][Xe]),
#: 1e-47 m⁵, read column by column from a 400 dpi render of the page (runs assigned to the
#: three curves by hand, about ±0.02). Calculated (Schlejen & Woerdman 1985, from Laskowski
#: et al.'s 1981 curves) at 2000 and 4000 K; derived from Na–Hg–Xe discharges (Jongerius &
#: Ras 1985) at 3000 K. Mean over 640–700 and 640–670 nm (the calculated curves end at about
#: 678 nm, 2000 K, and 688 nm, 4000 K), and the integral over 540–577 nm (1e-47 m⁵·nm; the
#: 4000 K curve is drawn to 577 nm; the 2000 K blue branch cannot be told from it beyond
#: 562 nm, so it is not used).
FIG59B_RED_640_700 = {2000: 0.065, 3000: 0.231, 4000: 0.128}
FIG59B_RED_640_670 = {2000: 0.121, 3000: 0.314, 4000: 0.202}
FIG59B_BLUE_540_577 = {3000: 33.4, 4000: 49.7}
#: The derived (3000 K) spectrum's B–X satellite, nm (air); the book: "observed for NaXe
#: discharges at 560 nm".
FIG59B_SATELLITE_NM = 560.5


def model(temperature_k: float, wall: str = "central"):
    edges = np.arange(12000.0, 20000.0, 4.0)
    k = sum(xp.reduced_absorption(temperature_k, edges, wall).values()) / 1e-47
    lam = vacuum_to_air(1e7 / (0.5 * (edges[1:] + edges[:-1])))
    order = np.argsort(lam)
    return lam[order], k[order]


def window_mean(lam, k, lo, hi):
    g = np.linspace(lo, hi, 4001)
    return np.trapezoid(np.interp(g, lam, k), g) / (hi - lo)


def test_curves_match_the_values_the_paper_states():
    r = np.linspace(5.0, 16.0, 11001)
    x, a = xp.lower_potential_cm1(r), xp.upper_potential_cm1("A", r)
    assert r[np.argmin(x)] == pytest.approx(9.48, abs=0.05) and -x.min() == pytest.approx(93.0, abs=0.5)  # Table II
    assert r[np.argmin(a)] == pytest.approx(6.11, abs=0.05) and -a.min() == pytest.approx(1094.0, abs=2.0)  # Table V
    # The A state's long-range tail against the paper's Q(20) = R⁶V = 841 au (Table VI): the
    # points end at 16 a0, where Q is a little larger (C8, C10).
    q16 = -16.0**6 * float(xp.upper_potential_cm1("A", 16.0)) / xp.HARTREE_CM1
    assert 841.0 < q16 < 1000.0


def test_the_extrapolated_wall_fits_the_points_and_the_dispersion():
    """X inside 8 a0 is a Tang–Toennies form with C6, C8, C10 fixed from the paper's Table III
    (C8/R² + C10/R⁴ = 91 au at 20 a0) and two parameters fitted to the points."""
    d = xp.data()
    c8 = d.wall_c8_au["central"]
    assert c8 / 20.0**2 + 49.0 / 40.0 * c8**2 / d.c6_au / 20.0**4 == pytest.approx(91.0, abs=0.5)
    for wall in xp.WALLS:
        assert xp.wall_fit(wall)[2] < 4.0  # cm⁻¹ rms over the 28 points
    # The variants bracket the wall where the red wing is made (central about 2000 cm⁻¹ at 6 a0).
    at6 = {w: float(xp.lower_potential_cm1(6.0, w)) for w in xp.WALLS}
    assert at6["soft"] < at6["central"] < at6["hard"]
    assert 0.7 < at6["soft"] / at6["central"] and at6["hard"] / at6["central"] < 1.15
    # Smooth through the hand-overs (value and slope).
    for r0 in (xp.X_INNER_A0, xp.X_BLEND_A0, xp.OUTER_A0):
        v = xp.lower_potential_cm1(np.array([r0 - 1e-4, r0, r0 + 1e-4]))
        assert abs(v[2] - v[0]) < 0.05
    assert xp.lower_potential_cm1(5.0) > xp.lower_potential_cm1(6.0) > xp.lower_potential_cm1(7.0) > 0.0


def test_difference_potentials_have_only_the_extrema_the_curves_imply():
    """A − X rises steadily towards the line (no red satellite). B − X has one maximum, the
    blue satellite, and beyond it, where B's shallow long-range well (67 cm⁻¹ at 12.8 a0)
    lies below the line's level of X's, a minimum 41 cm⁻¹ (1.4 nm) red of the line, inside
    the hand-over. Near 15.1–15.4 a0, 4 cm⁻¹ from the line and inside the impact region,
    A − X is flat within the reading error, so that check stops at 14 a0."""
    r = np.linspace(5.0, 14.0, 9001)
    assert np.all(np.diff(xp.difference_potential_cm1("A-X", r)) > 0.0)
    r = np.linspace(5.0, 40.0, 35001)
    bx = xp.difference_potential_cm1("B-X", r)
    turns = r[1:-1][np.diff(np.sign(np.diff(bx))) != 0]
    assert turns.size == 2 and turns[0] < 8.0 < 12.0 < turns[1] < 15.0
    satellite = float(vacuum_to_air(1e7 / bx.max()))
    assert 552.0 < satellite < 557.0
    assert bx[r > 10].min() - xp.asymptote_cm1() == pytest.approx(-40.6, abs=1.0)


def test_the_arc_s_table_is_the_same_sum():
    """The table the arc reads computes the temperature-free part once; it is the same sum
    (np.histogram's shortcut for even bins puts some points that sit on an edge into the
    next bin, which matters at the satellite's spike, so the running sums are compared)."""
    for t in (2000.0, 4000.0):
        direct = sum(xp.reduced_absorption(t, xp.TABLE_EDGES_CM1).values())
        table = xp._table(t, "central")
        assert table.sum() == pytest.approx(direct.sum(), rel=1e-12)
        assert np.abs(np.cumsum(table) - np.cumsum(direct)).max() < 1e-4 * direct.sum()


def test_finding_the_mark_set_against_fig_5_9b_is_mostly_missed():
    """Set before the model was built (2026-10-17): within 30 % of Fig. 5.9b's mean over
    640–700 nm and integral over 540–580 nm, at 2000 and 4000 K (calculated) and 3000 K
    (derived), and the satellite within 5 nm of about 560 nm. The two references themselves
    differ by 1.8 times on the red side, so no model could meet all of it. The model: the
    blue integral matches the discharge-derived one (0.99) and is 0.67 of the calculated;
    the red wing is 0.65 of the derived (3000 K), 1.86 and 1.33 of the calculated (2000,
    4000 K); the satellite 5.4 nm short of the derived. Met: one of six intensity checks."""
    red = {t: window_mean(*model(float(t)), 640, 700) / FIG59B_RED_640_700[t] for t in (2000, 3000, 4000)}
    assert red[2000] == pytest.approx(1.86, abs=0.05)
    assert red[3000] == pytest.approx(0.65, abs=0.03)
    assert red[4000] == pytest.approx(1.33, abs=0.04)
    blue = {t: window_mean(*model(float(t)), 540, 577) * 37.0 / FIG59B_BLUE_540_577[t] for t in (3000, 4000)}
    assert blue[3000] == pytest.approx(0.99, abs=0.03)
    assert blue[4000] == pytest.approx(0.67, abs=0.03)
    lam, k = model(3000.0)
    band = (lam > 540) & (lam < 580)
    assert FIG59B_SATELLITE_NM - lam[band][np.argmax(k[band])] == pytest.approx(5.4, abs=0.3)
    # The wall's variants move the red wing by up to 14 % and the satellite by 2 nm: the miss is
    # not the extrapolation's.
    for wall in ("soft", "hard"):
        assert window_mean(*model(3000.0, wall), 640, 700) / FIG59B_RED_640_700[3000] < 0.75


def test_widths_keep_xenon_apart():
    comp = composition(3000.0, HPS_FILL)
    w = broadening.widths(sodium.data().line("D2"), comp)
    assert w.van_der_waals_xenon > 0.0 and w.van_der_waals_mercury > w.van_der_waals_xenon


def test_each_line_s_part_sits_on_its_own_upper_level():
    wings, t, nu = xp.XenonPairWings(), 3000.0, C / (np.array([560.0, 650.0]) * 1e-9)
    by_line = [wings.kappa(t, 1e23, 1e24, nu, ln) for ln in sodium.resonance_lines()]
    assert by_line[0] + by_line[1] == pytest.approx(wings.kappa(t, 1e23, 1e24, nu), rel=1e-12)
    shares = sodium.resonance_shares()
    names = [ln.name for ln in sodium.resonance_lines()]
    # Far out (hundreds of cm⁻¹) the two parts are nearly in the lines' ratio.
    assert by_line[0][1] / by_line[1][1] == pytest.approx(shares[names[0]] / shares[names[1]], rel=0.1)


def test_xenon_pairs_replace_only_the_xenon_share_of_the_far_wing():
    """Inside half the join the lines keep their impact profile; beyond twice it each
    resonance line has lost only its xenon width, and the pair absorption is added, on both
    sides (B − X to the blue, A − X to the red)."""
    t = 3000.0
    comp = composition(t, HPS_FILL)
    wings = xp.XenonPairWings(join_nm=6.0)
    nu_near = C / (air_to_vacuum(np.array([587.0, 591.6])) * 1e-9)
    assert arc.absorption_coefficient(comp, nu_near, xenon_pairs=wings) == pytest.approx(
        arc.absorption_coefficient(comp, nu_near), rel=1e-12)

    nu = C / (air_to_vacuum(np.array([560.0, 650.0])) * 1e-9)
    expected = arc.absorption_coefficient(comp, nu, [ln for ln in sodium.data().lines if ln.lower_energy_cm1 > 0])
    stimulated = -np.expm1(-H * nu / (K_B * t))
    ground = comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1)
    for ln in sodium.resonance_lines():
        sigma, gamma, shift = broadening.voigt_parameters(replace(broadening.widths(ln, comp), van_der_waals_xenon=0.0))
        phi = voigt_profile(nu - (ln.frequency_hz + shift), sigma, gamma)
        expected += C * C / (8 * np.pi * ln.frequency_hz**2) * ln.a_ki * (ln.upper_g / ln.lower_g) * ground * phi * stimulated
    expected += wings.kappa(t, ground, comp.xenon, nu) * stimulated
    got = arc.absorption_coefficient(comp, nu, xenon_pairs=wings)
    assert got == pytest.approx(expected, rel=1e-10)
    assert np.all(got > arc.absorption_coefficient(comp, nu))  # the pairs exceed the xenon impact wing both sides
