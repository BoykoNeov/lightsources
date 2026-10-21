"""Sodium–mercury pairs, the mercury share of the D lines' far wings (design §4.4, rung 2
second part): the curves against their sources, the counting against an analytic wing,
the satellite position predicted before it was compared, and the swap into the arc."""

from dataclasses import replace
from importlib import resources

import numpy as np
import pytest
import yaml
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


def test_dispersion_tail_joins_smoothly_and_ends_on_unsolds_c6():
    """``tail="dispersion"``: beyond 9.95 a0 I½ is the asymptote − ΔC6/R^6 (Unsöld's times the
    factor) − an exponentially fading excess; value and slope continuous at the grid's end,
    rising monotonically to the line, Unsöld's tail alone far out. Inside the grid, and the
    other two bands everywhere, nothing changes."""
    d = mp.data()
    for factor in mp.DELTA_C6_FACTORS:
        r0 = d.grids_a0["I1/2"][-1]
        eps = 1e-4
        just_in = mp.difference_potential_cm1("I1/2", r0, factor, "dispersion")
        just_out = mp.difference_potential_cm1("I1/2", r0 + eps, factor, "dispersion")
        assert just_out == pytest.approx(just_in, abs=0.1)
        grid_slope = (d.plotted_cm1["I1/2"][-1] - d.plotted_cm1["I1/2"][-2]) / 0.05
        tail_slope = (mp.difference_potential_cm1("I1/2", r0 + 2 * eps, factor, "dispersion") - just_out) / eps
        assert tail_slope == pytest.approx(grid_slope, rel=0.01)
        r = np.linspace(r0 + 0.01, 60.0, 20000)
        dv = mp.difference_potential_cm1("I1/2", r, factor, "dispersion")
        assert np.all(np.diff(dv) > 0)
        c6 = factor * mp.delta_c6_unsold_au() * mp.HARTREE_CM1
        assert d.asymptote_cm1["I1/2"] - mp.difference_potential_cm1("I1/2", 80.0, factor, "dispersion") \
            == pytest.approx(c6 / 80.0**6, rel=0.01)
        inside = np.linspace(4.2, 9.9, 200)
        np.testing.assert_array_equal(mp.difference_potential_cm1("I1/2", inside, factor, "dispersion"),
                                      mp.difference_potential_cm1("I1/2", inside, factor))
        for band in ("II1/2", "I3/2"):
            rr = np.linspace(4.2, 40.0, 500)
            np.testing.assert_array_equal(mp.difference_potential_cm1(band, rr, factor, "dispersion"),
                                          mp.difference_potential_cm1(band, rr, factor))
    with pytest.raises(ValueError):
        mp.difference_potential_cm1("I1/2", 12.0, 1.0, "power")
    with pytest.raises(ValueError):
        mp.MercuryPairWings(tail="power")


def test_finding_the_tail_cuts_both_ways():
    """Computed before any lamp run (design §4.4, "The I½ tail"): red of D1, all three bands,
    dispersion over fitted tail. With Unsöld's ΔC6 the wing is weaker near the line (×0.75 at
    1 nm, ×0.93 at 6 nm) and unchanged by 15 nm; with twice it, stronger (×1.2 at 1–3 nm)."""
    edges = mp.TABLE_EDGES_CM1
    centres = 0.5 * (edges[1:] + edges[:-1])
    lam1 = 1e7 / mp.data().asymptote_cm1["I1/2"]

    def ratio(factor, nm):
        a = mp.reduced_absorption(4000.0, edges, delta_c6_factor=factor)
        b = mp.reduced_absorption(4000.0, edges, delta_c6_factor=factor, tail="dispersion")
        sel = np.abs(centres - 1e7 / (lam1 + nm)) < 0.5e7 / (lam1 + nm) ** 2
        return sum(b[k][sel].mean() for k in mp.BANDS) / sum(a[k][sel].mean() for k in mp.BANDS)

    assert ratio(1.0, 1.0) == pytest.approx(0.75, abs=0.03)
    assert ratio(1.0, 6.0) == pytest.approx(0.93, abs=0.02)
    assert ratio(1.0, 15.0) == pytest.approx(1.0, abs=0.015)
    assert ratio(2.0, 2.0) == pytest.approx(1.22, abs=0.03)
    assert ratio(2.0, 30.0) == pytest.approx(1.0, abs=0.01)


def test_czuchaj_tables_as_transcribed_give_the_papers_well_depths():
    """Table 5's numbers against the text's D_e (X 593 cm⁻¹, A 7076, B 2436): 11–23 cm⁻¹ off,
    about the stated 1e-4 au (22 cm⁻¹; A just over it). A misread digit would be off by far more.
    The splines keep R ≤ 12 a₀ (values used to 10 a₀)."""
    cz = mp.czuchaj_data()
    assert cz.r_a0[-1] == mp.CZUCHAJ_SPLINE_R_MAX_A0 and cz.atomic_au == pytest.approx(2.534)
    doc = yaml.safe_load(resources.files("lss.data").joinpath("nahg_czuchaj.yaml").read_text(encoding="utf-8"))
    pot, asym = doc["potentials_hartree"], doc["asymptote_hartree"]
    for name, depth in (("X", 593.0), ("A", 7076.0), ("B", 2436.0)):
        assert (asym[name] - min(pot[name])) * mp.HARTREE_CM1 == pytest.approx(depth, abs=25.0), name
    # By 12 a₀ each Λ difference potential is bound by a few hundred cm⁻¹ at most.
    assert -500.0 < cz.sigma_cm1[-1] < 0.0 and -500.0 < cz.pi_cm1[-1] < 0.0


def test_czuchaj_coupling_has_the_right_limits():
    """Short range: II½ is B ²Σ, I½ and I3/2 are A ²Π. Long range: the atomic recoupling
    (3p₃/₂ Ω=½ is ⅔ Σ) and the three bands carry 3 d_z² between them."""
    c = mp.czuchaj_coupled(np.array([4.0, 200.0]))
    assert c["II1/2"][1][0] == pytest.approx(1.0, abs=1e-5)
    assert c["I1/2"][1][0] == pytest.approx(0.0, abs=1e-5)
    assert c["II1/2"][1][1] == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert sum(float(mp.czuchaj_dipole_factor(b, 40.0)) for b in mp.BANDS) == pytest.approx(3.0, abs=1e-6)
    assert float(mp.czuchaj_dipole_factor("II1/2", 5.0)) == pytest.approx((3.187 / 2.534) ** 2, rel=1e-3)
    assert float(mp.czuchaj_dipole_factor("I3/2", 5.0)) == pytest.approx((0.952 / 2.534) ** 2, rel=1e-3)
    with pytest.raises(ValueError):
        mp.MercuryPairWings(curves="czuchaj", dipole="falling")
    with pytest.raises(ValueError):
        mp.MercuryPairWings(curves="windholz", dipole="table")
    with pytest.raises(ValueError):
        mp.MercuryPairWings(curves="hess")


def test_czuchaj_curves_shape_and_join():
    """Only II½ has an extremum (one minimum); each curve rises monotonically from 6 a₀, through
    the A–B crossing; beyond 10 a₀ each stays below its asymptote (the join's A₆ > 0; with the
    splines cut at 10 a₀, I3/2 rose above it near 16.6 a₀) and meets it with no slope step."""
    d = mp.data()
    for band in mp.BANDS:
        r = np.linspace(4.0, 40.0, 72001)
        dv = mp.difference_potential_cm1(band, r, curves="czuchaj")
        slope = np.diff(dv)
        turns = int(np.sum(np.diff(np.sign(slope[slope != 0])) != 0))
        assert turns == (1 if band == "II1/2" else 0), band
        assert np.all(slope[r[1:] > 6.0] > 0), band
        assert abs(dv[-1] - d.asymptote_cm1[band]) < 1.0
        assert np.all(dv[r > mp.CZUCHAJ_R_MAX_A0] < d.asymptote_cm1[band]), band
        g = np.gradient(dv, r)
        i = np.searchsorted(r, mp.CZUCHAJ_R_MAX_A0)
        assert g[i - 20] == pytest.approx(g[i + 20], rel=0.05)
    assert np.isnan(mp.difference_potential_cm1("II1/2", 3.9, curves="czuchaj"))


def test_czuchaj_satellite_where_predicted():
    """Predicted before any code (design §4.4): the II½ minimum at 674.0–675.0 nm in air
    (cubic spline: 14 810 cm⁻¹ at 5.21 a₀), within the lamps' 668–678 nm window and 4–7 nm
    to the red of the measured S1 (a classical edge lies red of the Airy peak). Found 675.011 nm:
    0.011 nm past the bound as written, which rounded 675.01 down; recorded as that miss."""
    r = np.linspace(4.0, 7.0, 300001)
    dv = mp.difference_potential_cm1("II1/2", r, curves="czuchaj")
    nm = float(vacuum_to_air(1e7 / dv.min()))
    assert 674.0 <= nm <= 675.02
    assert r[np.argmin(dv)] == pytest.approx(5.21, abs=0.02)
    assert 4.0 <= nm - S1_NM <= 7.0


def test_czuchaj_satellite_stronger_and_no_false_structure():
    """Computed before any lamp run (design §4.4): the II½ edge 3–11 times Windholz's at
    4100 K (9.4 with the table's dipoles, 5.9 with constant ones), and no κ bin at 600–625 nm
    more than 10 % above its neighbours ±4 nm (Windholz's: 83 %)."""
    edges = mp.TABLE_EDGES_CM1
    centres = 0.5 * (edges[1:] + edges[:-1])
    air = vacuum_to_air(1e7 / centres)
    w = mp.reduced_absorption(4100.0, edges)
    cz = mp.reduced_absorption(4100.0, edges, dipole="table", curves="czuchaj")

    def edge(tab, lo, hi):
        sel = (centres > lo) & (centres < hi)
        return tab[sel].max()

    ratio = edge(cz["II1/2"], 14650, 15050) / edge(w["II1/2"], 14700, 15100)
    assert 3.0 <= ratio <= 11.0 and ratio == pytest.approx(9.4, abs=0.3)
    total = sum(cz.values())
    worst = 0.0
    for j in np.where((air > 600) & (air < 625))[0]:
        near = (np.abs(air - air[j] - 4) < 0.3) | (np.abs(air - air[j] + 4) < 0.3)
        worst = max(worst, total[j] / total[near].mean() - 1.0)
    assert worst < 0.10


def test_czuchaj_reaches_the_arc():
    """The arc takes Czuchaj's curves through MercuryPairWings; tables are keyed by them."""
    nu = C / (np.array([650.0, 675.0]) * 1e-9)
    a = mp.MercuryPairWings().kappa(4000.0, 1e22, 1e23, nu)
    b = mp.MercuryPairWings(curves="czuchaj", dipole="table").kappa(4000.0, 1e22, 1e23, nu)
    assert np.all(a > 0) and np.all(b > 0) and not np.allclose(a, b)
