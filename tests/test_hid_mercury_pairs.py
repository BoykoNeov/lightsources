"""Sodium–mercury pairs, the mercury share of the D lines' far wings (design §4.4, rung 2
second part): the curves against their sources, the counting against an analytic wing,
the satellite position predicted before it was compared, and the swap into the arc."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import voigt_profile

from lss.physics.hid import arc, broadening, mercury_pairs as mp, sodium
from lss.physics.hid.composition import Fill, composition
from lss.spectral.units import C, H, K_B, air_to_vacuum, vacuum_to_air

HPS_FILL = Fill(sodium_pa=13e3, mercury_pa=70e3, xenon_pa=20e3)
#: Schlejen, Woerdman & Pichler, J. Mol. Spectrosc. 128, 1 (1988), as quoted by Windholz et
#: al. (1991): the NaHg red satellite S1 in absorption, nm.
S1_NM = 669.0


def test_ground_state_is_thiel_hotop_meyers_curve():
    r = np.linspace(4.5, 30.0, 25501)
    v = mp.lower_potential_cm1(r)
    assert r[np.argmin(v)] == pytest.approx(6.52, abs=0.05)  # thesis Table 3.13
    assert -v.min() / 8065.544 * 1000 == pytest.approx(69.0, abs=1.0)  # De, meV
    assert r[np.argmin(np.abs(v[r < 6.0]))] == pytest.approx(5.52, abs=0.02)  # thesis Table 3.11
    # The −C6/R^6 tail takes over at 30 a0 without a jump (table −7.2e-7 Eh there).
    assert mp.lower_potential_cm1(30.0 - 1e-6) == pytest.approx(mp.lower_potential_cm1(30.0 + 1e-6), abs=0.03)
    # Inside the table the wall keeps rising.
    assert mp.lower_potential_cm1(4.0) > mp.lower_potential_cm1(4.5) > 5000.0


def test_digitised_curves_keep_the_shape_the_paper_states():
    """Only II½ − X½ has an extremum, one minimum; the curves pass through the plotted
    (computed) points; each joins its asymptote smoothly beyond 9.95 a0."""
    d = mp.data()
    for band in mp.BANDS:
        r = np.linspace(d.grids_a0[band][0], 40.0, 40001)
        dv = mp.difference_potential_cm1(band, r)
        slope = np.diff(dv)
        turns = int(np.sum(np.diff(np.sign(slope[slope != 0])) != 0))
        assert turns == (1 if band == "II1/2" else 0), band
        assert abs(dv[-1] - d.asymptote_cm1[band]) < 1.0
        for rc, ec in d.circles_cm1[band]:
            if rc >= d.grids_a0[band][0]:
                plotted = np.interp(rc, d.grids_a0[band], d.plotted_cm1[band])
                assert abs(plotted - ec) < 30.0, (band, rc)
        g = np.gradient(dv, r)
        i = np.searchsorted(r, 9.95)
        assert g[i - 20] == pytest.approx(g[i + 20], rel=0.05)  # no slope step at the join
    r = np.linspace(4.0, 9.95, 5951)
    dv = mp.difference_potential_cm1("II1/2", r) - mp.shift_cm1()
    assert dv.min() == pytest.approx(14413.0, abs=2.0) and r[np.argmin(dv)] == pytest.approx(4.75, abs=0.05)


def test_the_bands_carry_the_d_lines_strength_at_long_range():
    """3 d_z² is fixed by the D lines' A values; d_z for Na 3s–3p is about 2.5 e·a0."""
    assert np.sqrt(mp.dipole_squared_au()) == pytest.approx(2.49, abs=0.01)
    assert mp.dipole_factor("I3/2", 12.0, "falling") == pytest.approx(1.0)
    assert mp.dipole_factor("II1/2", 4.5, "falling") == pytest.approx(1.0)
    # falling: oscillator strength ∝ ΔV·|D|² down 10× from 9.45 to 3.78 a0 (held below)
    lo, hi = mp.FALL_R_A0
    r = 5.0
    f = mp.dipole_factor("I1/2", r, "falling") * mp.difference_potential_cm1("I1/2", r) / mp.difference_potential_cm1("I1/2", hi)
    assert np.log10(f) == pytest.approx((r - hi) / (hi - lo), abs=1e-9)


def test_counting_matches_the_analytic_quasistatic_wing():
    """A check that needs no paper: for ΔV = A − C/R⁶, a flat lower curve and a constant
    dipole, the classical wing is k/(n_a n_b) = 4πR² |dR/dν| S(ν) with R = (C/δ)^{1/6}."""
    asym = 17000.0
    c6_cm1 = 700.0 * mp.HARTREE_CM1  # 700 au, in cm⁻¹·a0⁶
    r = np.geomspace(5.0, 400.0, 400_000)
    dv = asym - c6_cm1 / r**6
    for delta in (150.0, 400.0, 1000.0):
        edges = [asym - delta - 2.0, asym - delta + 2.0]
        k = mp.quasistatic(3000.0, edges, r, dv, np.zeros_like(r), 6.0)[0]
        rr = (c6_cm1 / delta) ** (1 / 6)
        drdnu = rr / (6.0 * delta) * mp.BOHR_M / (100.0 * C)  # m per Hz
        nu = (asym - delta) * 100.0 * C
        s = 8 * np.pi**3 * nu / (3 * H * C) * 6.0 * (mp.ELEMENTARY_CHARGE_C * mp.BOHR_M) ** 2 / (4 * np.pi * mp.EPSILON_0)
        expected = 4 * np.pi * (rr * mp.BOHR_M) ** 2 * drdnu * s
        assert k == pytest.approx(expected, rel=0.01), delta


def test_finding_the_satellite_lands_where_it_was_predicted():
    """Set before comparing (2026-10-15): one shift for the three curves, fixed at 9.95 a0 by
    Unsöld's ΔC₆ (1 to 2 times); the classical satellite at the II½ minimum (drawn 14 413 cm⁻¹,
    computed point 14 473 cm⁻¹) plus the shift. The window is 668.7–678.4 nm (667.8–679.4 nm with
    the digitising error); the measured S1 at 669 nm is inside it, near its blue edge. A coarse
    test: the window is 10 nm wide."""
    assert mp.shift_cm1(1.0) == pytest.approx(477.5, abs=1.0)
    assert mp.shift_cm1(2.0) == pytest.approx(322.6, abs=1.0)
    blue = float(vacuum_to_air(1e7 / (14473.0 + mp.shift_cm1(1.0))))
    r = np.linspace(4.0, 9.95, 5951)
    red = float(vacuum_to_air(1e7 / mp.difference_potential_cm1("II1/2", r, 2.0).min()))
    assert blue == pytest.approx(668.7, abs=0.1) and red == pytest.approx(678.4, abs=0.1)
    assert blue <= S1_NM <= red
    # with ±20 cm⁻¹ for digitising, as the mark included: 667.8–679.4 nm
    wide = [float(vacuum_to_air(1e7 / (e + mp.shift_cm1(k)))) for e, k in ((14473.0 + 20.0, 1.0), (14413.0 - 20.0, 2.0))]
    assert wide[0] == pytest.approx(667.8, abs=0.1) and wide[1] == pytest.approx(679.4, abs=0.1)


def test_finding_the_classical_satellite_is_weak():
    """The II½ minimum sits at 4.75 a0, about 3000 cm⁻¹ up the ground state's wall, so at arc
    temperatures it is a small peak and an edge, not a band: at 3000 K the II½ band is under
    a third of the pair absorption at 650–670 nm and vanishes beyond its minimum."""
    lam = np.arange(650.0, 690.0, 1.0)
    edges = np.sort(1e7 / air_to_vacuum(np.append(lam - 0.5, lam[-1] + 0.5)))
    ra = {b: v[::-1] for b, v in mp.reduced_absorption(3000.0, edges).items()}
    total = sum(ra.values())
    near = (lam >= 650) & (lam <= 670)
    assert ra["II1/2"][near].sum() / total[near].sum() < 0.33
    assert np.all(ra["II1/2"][lam >= 673] == 0.0)


def test_widths_keep_mercury_and_xenon_apart():
    comp = composition(3000.0, HPS_FILL)
    w = broadening.widths(sodium.data().line("D2"), comp)
    assert w.van_der_waals == w.van_der_waals_mercury + w.van_der_waals_xenon
    assert w.van_der_waals_mercury > 2.0 * w.van_der_waals_xenon  # 70 kPa Hg against 20 kPa Xe


def test_the_table_holds_the_analytic_wing_near_the_lines():
    """Where the impact wing hands over (from about 0.3 nm), the 2 cm⁻¹ table, read as the
    arc reads it, keeps the analytic C₆ wing within 2 % (0.5 % from 0.5 nm)."""
    asym, c6_cm1, t = 16973.36619, 685.0 * mp.HARTREE_CM1, 3500.0
    r = np.geomspace(4.0, 400.0, 100_000)
    table = mp.quasistatic(t, mp.TABLE_EDGES_CM1, r, asym - c6_cm1 / r**6, np.zeros_like(r), 6.0)
    centres = 0.5 * (mp.TABLE_EDGES_CM1[1:] + mp.TABLE_EDGES_CM1[:-1])
    for off_nm, tol in ((0.3, 0.02), (0.5, 0.005), (1.0, 0.005), (3.0, 0.005)):
        nu_cm1 = 1e7 / (1e7 / asym + off_nm)
        delta = asym - nu_cm1
        rr = (c6_cm1 / delta) ** (1 / 6)
        s = 8 * np.pi**3 * nu_cm1 * 100 * C / (3 * H * C) * 6.0 * (mp.ELEMENTARY_CHARGE_C * mp.BOHR_M) ** 2 / (4 * np.pi * mp.EPSILON_0)
        expected = 4 * np.pi * (rr * mp.BOHR_M) ** 2 * rr / (6.0 * delta) * mp.BOHR_M / (100.0 * C) * s
        assert np.interp(nu_cm1, centres, table) == pytest.approx(expected, rel=tol), off_nm


def test_each_line_s_bands_sit_on_its_own_upper_level():
    """D1 hands its wing to I ½ (3p₁/₂), D2 to I 3/2 and II ½ (3p₃/₂); together, all three."""
    wings, t, nu = mp.MercuryPairWings(), 3000.0, C / (np.array([600.0, 650.0]) * 1e-9)
    by_line = [wings.kappa(t, 1e23, 1e24, nu, ln) for ln in (sodium.data().line("D1"), sodium.data().line("D2"))]
    assert by_line[0] + by_line[1] == pytest.approx(wings.kappa(t, 1e23, 1e24, nu), rel=1e-12)
    assert np.all(by_line[0] > 0) and np.all(by_line[1] > 0)  # far out not in the lines' ratio: the curves differ


def test_mercury_pairs_replace_only_the_mercury_share_of_the_far_wing():
    """Inside half the join the lines keep their impact profile; beyond twice it each
    resonance line has lost only its mercury width, and the pair absorption is added (red
    side only)."""
    t = 3000.0
    comp = composition(t, HPS_FILL)
    wings = mp.MercuryPairWings(join_nm=6.0)
    nu_near = C / (air_to_vacuum(np.array([587.0, 591.6])) * 1e-9)
    assert arc.absorption_coefficient(comp, nu_near, mercury_pairs=wings) == pytest.approx(
        arc.absorption_coefficient(comp, nu_near), rel=1e-12)

    nu = C / (air_to_vacuum(np.array([560.0, 650.0])) * 1e-9)
    others = [ln for ln in sodium.data().lines if ln.lower_energy_cm1 > 0]
    expected = arc.absorption_coefficient(comp, nu, others)
    stimulated = -np.expm1(-H * nu / (K_B * t))
    ground = comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1)
    for ln in (ln for ln in sodium.data().lines if ln.lower_energy_cm1 == 0.0):
        sigma, gamma, shift = broadening.voigt_parameters(replace(broadening.widths(ln, comp), van_der_waals_mercury=0.0))
        phi = voigt_profile(nu - (ln.frequency_hz + shift), sigma, gamma)
        expected += C * C / (8 * np.pi * ln.frequency_hz**2) * ln.a_ki * (ln.upper_g / ln.lower_g) * ground * phi * stimulated
    expected += wings.kappa(t, ground, comp.mercury, nu) * stimulated
    got = arc.absorption_coefficient(comp, nu, mercury_pairs=wings)
    assert got == pytest.approx(expected, rel=1e-10)
    assert got[0] < arc.absorption_coefficient(comp, nu)[0]  # blue: the mercury width goes, nothing replaces it
    assert got[1] > 2.0 * arc.absorption_coefficient(comp, nu)[1]  # red: the pairs far exceed the impact wing
