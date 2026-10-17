"""Sodium atom pairs, the D lines' far wings (design §4.4, rung 2 first part): the curves
against their sources, a check that needs no paper, and the comparison with Chung, Kirby
& Babb's quantum calculation, which missed the mark set before it was run."""

from dataclasses import replace

import numpy as np
import pytest

from scipy.special import voigt_profile

from lss.physics.hid import arc, broadening, mercury_pairs as mp, pairs, sodium, xenon_pairs as xp
from lss.physics.hid.composition import Fill, composition
from lss.spectral.units import C, H, K_B, air_to_vacuum

#: Chung, Kirby & Babb, PRA 63, 032516 (2001), Fig. 6, digitised from the arXiv PDF's
#: vector paths (2026-10-15): mean of k/n² over each window, 1e-37 cm⁵. Panel (c) is all
#: bands, panel (b) the triplets. The windows' "grass" (10th–90th percentile) spans
#: about ±20 % of these means.
CHUNG_FIG6 = {
    (2000, 640, 700): 3.21, (2000, 525, 540): 2.76, (2000, 610, 630): 6.10,
    (3000, 640, 700): 1.55, (3000, 525, 540): 1.01, (3000, 610, 630): 3.84,
    (1000, 640, 700): 120.7, (1000, 525, 540): 43.3,
}
CHUNG_FIG6_TRIPLET_610_630 = {2000: 3.53, 3000: 2.61}
#: Woerdman & de Groot, Chem. Phys. Lett. 80, 220 (1981), from an HPS discharge at 2000 K,
#: 5 cm⁻¹ resolution, as quoted by Chung et al.: 5(1)e-37 cm⁵ at 500 nm.
WOERDMAN_500NM = 5.0


def window(t, lo_nm, hi_nm, **kw) -> float:
    """Mean k/n² over a (vacuum) wavelength window, 1e-37 cm⁵."""
    edges = np.sort(1e7 / np.linspace(lo_nm, hi_nm, 61))
    return float(sum(pairs.reduced_absorption(t, edges, **kw).values()).mean() * 1e10 / 1e-37)


def test_lower_curves_reproduce_knoop_et_al_s_derived_constants():
    r = np.linspace(2.5, 20.0, 200_001) / pairs.BOHR_ANGSTROM
    x, a = pairs.lower_potential_cm1("X", r), pairs.lower_potential_cm1("a", r)
    assert x.min() == pytest.approx(-6022.0420, abs=0.01)  # T_e^X = -D_e^X (Table III)
    assert r[np.argmin(x)] * pairs.BOHR_ANGSTROM == pytest.approx(3.07895, abs=1e-3)
    assert a.min() == pytest.approx(-172.909, abs=0.05)  # Table IV
    assert r[np.argmin(a)] * pairs.BOHR_ANGSTROM == pytest.approx(5.1431, abs=2e-3)
    # The a state's inner part (from Chung et al.'s figure) joins Knoop's form within the digitising error.
    edge = 4.2780 / pairs.BOHR_ANGSTROM
    assert abs(pairs.lower_potential_cm1("a", edge - 1e-6) - pairs.lower_potential_cm1("a", edge + 1e-6)) < 150


def test_difference_potentials_hold_the_satellites():
    d = pairs.data()
    r = np.linspace(4.2, 24.0, 4000)
    band = {b.name: b for b in d.bands}
    ax = pairs.difference_potential_cm1(band["A-X"], r)
    ca = pairs.difference_potential_cm1(band["c-a"], r)
    ba = pairs.difference_potential_cm1(band["b-a"], r)
    # Extrema near the measured satellites (804, 551.5, 880 nm); Chung et al. state 809, 548, 913 nm
    # for their adopted curves.
    assert 1e7 / ax.min() == pytest.approx(809.0, abs=6.0)
    assert 1e7 / ca[r > 7].max() == pytest.approx(548.0, abs=4.0)
    assert 1e7 / ba.min() == pytest.approx(913.0, abs=8.0)
    # Every curve joins its long-range form at 24 a0.
    for b in d.bands:
        inside = pairs.difference_potential_cm1(b, 24.0 - 1e-9)
        assert abs(inside - b.long_range_cm1(d.asymptote_cm1, 24.0)) < 1.0, b.name
        # and the fitted long-range form is the resonance term there, within the plotting steps
        assert abs(b.long_range_cm1(d.asymptote_cm1, 24.0) - (d.asymptote_cm1 + b.c3_au / 24.0**3 * pairs.HARTREE_CM1)) < 10


def test_long_range_pairs_reproduce_the_measured_resonance_wing():
    """5 nm from the D lines the pairs are about 20 a0 apart, where every curve is nearly the
    resonance c3/R^3 term. There the pair absorption lands within a factor 2 of the impact
    wing of Huennekens & Gallagher's measured widths: the weights (spin, Λ, the ½ for
    identical atoms) are right to that level. Recorded: 1.37 (blue) and 1.99 (red), the red
    side at the edge of the mark."""
    t = 1500.0
    comp = composition(t, Fill(sodium_pa=13e3))
    d_lines = [ln for ln in sodium.data().lines if ln.lower_energy_cm1 == 0.0]
    ground = comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1)
    for air_nm in (584.0, 595.0):
        wn = 1e7 / air_to_vacuum(np.array([air_nm]))[0]
        k = sum(pairs.reduced_absorption(t, [wn - 5, wn + 5]).values())[0]
        lorentz = arc.absorption_coefficient(comp, np.array([wn * 100 * C]), d_lines)[0] / ground**2
        assert 1.2 < k / lorentz < 2.0, (air_nm, k / lorentz)


def test_finding_the_pairs_miss_the_mark_set_against_chung_et_al():
    """Set before running (2026-10-15): within 30 % of Chung et al.'s Fig. 6 in 640–700 and
    525–540 nm at 2000 and 3000 K. Three of the four miss: the semiclassical pairs run 29–51 %
    above. The 1981 discharge measurement at 500 nm is met."""
    ratios = {(t, lo): window(t, lo, hi) / CHUNG_FIG6[(t, lo, hi)] for t in (2000, 3000) for lo, hi in ((640, 700), (525, 540))}
    assert sum(r > 1.3 for r in ratios.values()) == 3
    assert all(1.25 < r < 1.55 for r in ratios.values())
    edges = np.sort(1e7 / np.array([500.0 - 0.0625, 500.0 + 0.0625]))  # 5 cm⁻¹ at 500 nm
    k500 = sum(pairs.reduced_absorption(2000.0, edges).values())[0] * 1e10 / 1e-37
    assert abs(k500 / WOERDMAN_500NM - 1) < 0.3


def test_finding_a_rotational_cut_explains_most_of_the_miss_near_the_lines():
    """Chung et al. include rotational levels up to J = 250. Applied to the classical pairs
    (comparison only), it brings the windows to 1.14–1.37 of theirs. Independently of any model
    of ours, their triplet wing near the lines falls from 2000 to 3000 K as the cut predicts,
    where an uncut free-pair wing would stay flat."""
    cut = [window(t, lo, hi, jmax=250) / CHUNG_FIG6[(t, lo, hi)]
           for t in (2000, 3000) for lo, hi in ((610, 630), (640, 700), (525, 540))]
    assert all(1.05 < r < 1.40 for r in cut)

    def triplet(t, **kw):
        edges = np.sort(1e7 / np.linspace(610, 630, 61))
        return pairs.reduced_absorption(t, edges, **kw)["b-a"].mean() * 1e10 / 1e-37

    assert triplet(3000) / triplet(2000) == pytest.approx(1.0, abs=0.03)
    predicted = triplet(3000, jmax=250) / triplet(2000, jmax=250)
    theirs = CHUNG_FIG6_TRIPLET_610_630[3000] / CHUNG_FIG6_TRIPLET_610_630[2000]
    assert predicted == pytest.approx(theirs, abs=0.05) and theirs < 0.8


def test_finding_the_two_dipole_forms_bracket_the_bound_pairs_wing():
    """At 1000 K, where bound X-state pairs dominate and the rotational cut does not matter,
    the long-range dipole form at every R overshoots Chung et al.'s red wing and the constant
    form undershoots it slightly."""
    ref = CHUNG_FIG6[(1000, 640, 700)]
    long_range, constant = window(1000, 640, 700), window(1000, 640, 700, dipole="constant")
    assert constant < ref < long_range
    assert long_range / ref > 1.5 and constant / ref > 0.85
    assert window(1000, 640, 700, jmax=250) == pytest.approx(long_range, rel=0.01)


def test_pairs_replace_only_the_resonance_share_of_the_far_wing():
    """Inside half the join the lines keep their impact profile; beyond twice it each
    resonance line has lost only its resonance width, and the pair absorption is added."""
    t = 3000.0
    comp = composition(t, Fill(sodium_pa=13e3, mercury_pa=70e3, xenon_pa=20e3))
    wings = pairs.PairWings(join_nm=6.0)
    nu_near = C / (air_to_vacuum(np.array([587.0, 591.6])) * 1e-9)
    assert arc.absorption_coefficient(comp, nu_near, pairs=wings) == pytest.approx(
        arc.absorption_coefficient(comp, nu_near), rel=1e-12)

    nu = C / (air_to_vacuum(np.array([560.0, 650.0])) * 1e-9)
    others = [ln for ln in sodium.data().lines if ln.lower_energy_cm1 > 0]
    expected = arc.absorption_coefficient(comp, nu, others)
    stimulated = -np.expm1(-H * nu / (K_B * t))
    for ln in (ln for ln in sodium.data().lines if ln.lower_energy_cm1 == 0.0):
        n_lower = comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1)
        sigma, gamma, shift = broadening.voigt_parameters(replace(broadening.widths(ln, comp), resonance=0.0))
        phi = voigt_profile(nu - (ln.frequency_hz + shift), sigma, gamma)
        expected += C * C / (8 * np.pi * ln.frequency_hz**2) * ln.a_ki * (ln.upper_g / ln.lower_g) * n_lower * phi * stimulated
    ground = comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1)
    expected += wings.kappa(t, ground, nu) * stimulated
    got = arc.absorption_coefficient(comp, nu, pairs=wings)
    assert got == pytest.approx(expected, rel=1e-10)
    assert np.all(got > 1.5 * arc.absorption_coefficient(comp, nu))  # the pairs exceed the impact wing there


def test_the_hand_over_weight():
    off = np.geomspace(0.01, 100.0, 20_001)
    w = arc.join_weight(off, 0.8)
    assert np.all(w[off <= 0.4] == 0.0) and np.all(w[off >= 1.6] == 1.0)
    assert np.all(np.diff(w) >= 0.0) and np.abs(np.diff(w)).max() < 1e-3  # smooth and rising
    assert float(arc.join_weight(-1.0, 0.8)) == float(arc.join_weight(1.0, 0.8))  # both sides alike
    assert float(arc.join_weight(0.8, 0.8)) == pytest.approx(0.5)  # half way at the centre


def test_each_line_hands_over_by_its_own_weight():
    """1.1 nm red of D1 (1.7 nm from D2), join 0.8 nm: D1 part way, D2 fully handed over; the
    profiles mix bilinearly and each line adds its own part of each kind of pair."""
    t = 3500.0
    comp = composition(t, Fill(sodium_pa=12e3, mercury_pa=240e3, xenon_pa=20e3))
    na, hg = pairs.PairWings(join_nm=0.8, dipole="constant"), mp.MercuryPairWings(join_nm=0.8)
    shares = sodium.resonance_shares()
    assert shares["D1"] == pytest.approx(1 / 3, rel=1e-3) and shares["D1"] + shares["D2"] == pytest.approx(1.0)
    d1 = sodium.data().line("D1")
    nu = np.array([C / ((d1.vacuum_wavelength_nm + 1.1) * 1e-9)])
    ground = comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1)
    stimulated = -np.expm1(-H * nu / (K_B * t))
    expected = arc.absorption_coefficient(comp, nu, [ln for ln in sodium.data().lines if ln.lower_energy_cm1 > 0])
    for ln in sodium.resonance_lines():
        w = float(arc.join_weight(C / nu[0] * 1e9 - ln.vacuum_wavelength_nm, 0.8))
        assert (0.0 < w < 1.0) if ln is d1 else (w == 1.0)
        full = broadening.widths(ln, comp)

        def phi(**drop):
            sigma, gamma, shift = broadening.voigt_parameters(replace(full, **drop))
            return voigt_profile(nu - (ln.frequency_hz + shift), sigma, gamma)

        mixed = ((1 - w) ** 2 * phi() + w * (1 - w) * (phi(resonance=0.0) + phi(van_der_waals_mercury=0.0))
                 + w * w * phi(resonance=0.0, van_der_waals_mercury=0.0))
        expected += (C * C / (8 * np.pi * ln.frequency_hz**2) * ln.a_ki * (ln.upper_g / ln.lower_g) * ground * mixed
                     + w * (na.kappa(t, ground, nu, ln) + hg.kappa(t, ground, comp.mercury, nu, ln))) * stimulated
    assert arc.absorption_coefficient(comp, nu, pairs=na, mercury_pairs=hg) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("t", [1600.0, 2500.0, 3500.0, 4500.0])
def test_the_wing_falls_steadily_through_the_hand_over(t):
    """κ falls monotonically from 0.3 to 2.5 nm on both sides of the D lines, so the
    self-reversed line can have no false maximum there. (A cut at 6 nm stepped κ up about
    3.3 times on the red side, which made a second red maximum.)"""
    comp = composition(t, Fill(sodium_pa=12e3, mercury_pa=240e3, xenon_pa=20e3))
    na, hg = pairs.PairWings(dipole="constant"), mp.MercuryPairWings()
    off = np.geomspace(0.3, 2.5, 2000)
    for base, sign in ((sodium.data().line("D1").vacuum_wavelength_nm, 1),
                       (sodium.data().line("D2").vacuum_wavelength_nm, -1)):
        k = arc.absorption_coefficient(comp, C / ((base + sign * off) * 1e-9), pairs=na, mercury_pairs=hg)
        assert np.all(np.diff(k) < 0.0)
        # With xenon pairs too (B − X has a shallow minimum 41 cm⁻¹ red of the line, inside the
        # hand-over, where the classical sum piles up).
        k = arc.absorption_coefficient(comp, C / ((base + sign * off) * 1e-9), pairs=na, mercury_pairs=hg,
                                       xenon_pairs=xp.XenonPairWings())
        assert np.all(np.diff(k) < 0.0)


def test_each_line_s_pairs_sit_on_its_own_upper_level():
    """The curves have one 3p asymptote; each line takes its share of the table at its own
    upper level. Near the lines the pairs then stay within 1.4-2.1 times the resonance
    impact wing on both sides (with the table on 3p₁/₂ alone: 6 times red of D1 and 0.14
    times blue of D2, at 0.2 nm)."""
    t = 3500.0
    comp = composition(t, Fill(sodium_pa=12e3, mercury_pa=240e3, xenon_pa=20e3))
    wings = pairs.PairWings(dipole="constant")
    ground = comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1)
    for base, sign in ((sodium.data().line("D1").vacuum_wavelength_nm, 1),
                       (sodium.data().line("D2").vacuum_wavelength_nm, -1)):
        for off in (0.4, 0.8, 1.6, 3.0):
            nu = np.array([C / ((base + sign * off) * 1e-9)])
            impact = 0.0
            for ln in sodium.resonance_lines():
                w = broadening.widths(ln, comp)
                pref = C * C / (8 * np.pi * ln.frequency_hz**2) * ln.a_ki * (ln.upper_g / ln.lower_g) * ground
                impact += pref * (arc._profile(nu, ln, w) - arc._profile(nu, ln, replace(w, resonance=0.0)))[0]
            assert 1.4 < wings.kappa(t, ground, nu)[0] / impact < 2.1, (sign, off)
