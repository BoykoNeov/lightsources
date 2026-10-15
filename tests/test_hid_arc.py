"""HID Stages A and B for sodium (design §4.4, §12): checks against physical limits and
the cited tables, independent of any measured lamp. The D-line tests pass the D lines
alone, so the lines ending on 3p do not change what they measure."""

import numpy as np
import pytest
from scipy.signal import argrelextrema

from lss.physics.hid import arc, broadening, sodium
from lss.physics.hid.composition import Fill, composition, saha_factor
from lss.physics.thermal import planck
from lss.spectral.continuum import TabulatedContinuum
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import Spectrum
from lss.spectral.units import C, H, K_B, air_to_vacuum

HPS_FILL = Fill(sodium_pa=13e3, mercury_pa=70e3, xenon_pa=20e3)
LINES = sodium.data().lines
D_LINES = (sodium.data().line("D2"), sodium.data().line("D1"))


def test_partition_function_is_the_ground_level_when_cold():
    assert sodium.partition_function(300.0) == pytest.approx(2.0, abs=1e-9)
    assert 2.0 < sodium.partition_function(4000.0) < 2.05  # 3p at 2.1 eV: ~0.7 % of atoms
    assert sodium.partition_function(6000.0) > sodium.partition_function(4000.0)


def test_composition_closes_pressure_and_saha():
    for t in (1500.0, 3000.0, 4000.0, 5000.0):
        c = composition(t, HPS_FILL)
        particles = c.sodium + c.sodium_ion + c.electrons + c.mercury + c.xenon
        assert particles * K_B * t == pytest.approx(HPS_FILL.total_pa, rel=1e-12)
        assert c.electrons**2 == pytest.approx(saha_factor(t, c.lowering_cm1) * c.sodium, rel=1e-8)
        # each element keeps its share of the heavy particles
        assert c.mercury / (c.sodium + c.sodium_ion) == pytest.approx(70.0 / 13.0, rel=1e-12)
    assert composition(1500.0, HPS_FILL).ionised_fraction < 1e-6
    hot = [composition(t, HPS_FILL).ionised_fraction for t in (3000.0, 4000.0, 5000.0)]
    assert hot == sorted(hot) and 0.01 < hot[1] < 0.1  # a few per cent at an HPS axis


def test_widths_follow_their_sources():
    c = composition(4000.0, HPS_FILL)
    d2 = sodium.data().line("D2")
    w = broadening.widths(d2, c)
    ground = c.sodium * sodium.level_population_fraction(0.0, 2, 4000.0, c.lowering_cm1)
    assert w.resonance == pytest.approx(4.67e-13 * ground, rel=1e-12)  # Huennekens & Gallagher
    assert w.shift_van_der_waals / w.van_der_waals == pytest.approx(-2.94 / 8.08, rel=1e-12)  # Lindholm–Foley
    # Sodium's own atoms dominate; the Unsöld estimate for Hg and Xe is a minor part.
    assert 0.05 < w.van_der_waals / w.resonance < 0.5
    # Doppler: σ_ν = ν₀ sqrt(kT/mc²)
    assert w.doppler_sigma_hz == pytest.approx(d2.frequency_hz * np.sqrt(K_B * 4000.0 / (sodium.MASS_KG * C**2)))


def test_absorption_integrates_to_the_line_strength():
    """∫κ_ν dν = c²/(8πν₀²) A (g_k/g_i) n_i (1 − e^{−hν₀/kT}) for a profile normalised in frequency."""
    c = composition(2000.0, Fill(sodium_pa=1e-3))
    d2 = sodium.data().line("D2")
    sigma, gamma, shift = broadening.voigt_parameters(broadening.widths(d2, c))
    nu = d2.frequency_hz + shift + np.linspace(-2e4, 2e4, 400_001) * sigma
    kappa = arc.absorption_coefficient(c, nu, lines=(d2,))
    n_lower = c.sodium * sodium.level_population_fraction(0.0, 2, 2000.0, c.lowering_cm1)
    expected = (C**2 / (8 * np.pi * d2.frequency_hz**2) * d2.a_ki * 2.0 * n_lower
                * -np.expm1(-H * d2.frequency_hz / (K_B * 2000.0)))
    lorentz_tails = 2.0 * gamma / (np.pi * 2e4 * sigma)  # mass outside the window
    assert np.trapezoid(kappa, nu) == pytest.approx(expected * (1.0 - lorentz_tails), rel=1e-4)


def test_chords_cross_the_whole_arc():
    edges = np.linspace(0.0, 3e-3, 31)
    b = np.array([0.0, 1e-3, 2.95e-3])
    ell = arc.chord_lengths(edges, b)
    assert 2.0 * ell.sum(axis=1) == pytest.approx(2.0 * np.sqrt(3e-3**2 - b**2), rel=1e-12)


def test_isothermal_thick_arc_is_a_blackbody_at_line_centre():
    col = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=3000.0, wall_temperature_k=3000.0, fill=HPS_FILL, shells=8)
    lam = np.array([588.995095, 589.592424])
    s = arc.side_radiance(col, lam)
    assert s.radiance == pytest.approx(np.broadcast_to(planck.spectral_radiance(air_to_vacuum(lam), 3000.0), s.radiance.shape), rel=1e-9)


def test_thin_arc_is_the_sum_of_line_emissions():
    """With τ ≪ 1 each chord gives ∫I dλ = Σ (hν₀/4π) A n_k · chord length."""
    t = 2000.0
    col = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=t, wall_temperature_k=t, fill=Fill(sodium_pa=1e-6), shells=4)
    lam = arc.line_grid(D_LINES, 588.5, 590.1, first_offset_nm=2e-5, ratio=1.01, base_step_nm=0.01)
    s = arc.side_radiance(col, lam, impact_parameters=5, lines=D_LINES)
    c = composition(t, col.fill)
    per_length = sum(H * ln.frequency_hz / (4 * np.pi) * ln.a_ki
                     * c.sodium * sodium.level_population_fraction(ln.upper_energy_cm1, ln.upper_g, t, c.lowering_cm1)
                     for ln in D_LINES)  # W m⁻³ sr⁻¹
    chord = 2.0 * np.sqrt(col.radius_m**2 - s.impact_parameter_m**2)
    assert s.radiance.max() < 1e-2 * planck.spectral_radiance(589.0, t)  # really thin
    assert np.trapezoid(s.radiance, lam, axis=1) == pytest.approx(per_length * chord, rel=2e-3)


def _peaks_around_d(col):
    lam = arc.line_grid(D_LINES, 560.0, 620.0)
    m = arc.side_radiance(col, lam, lines=D_LINES).mean_radiance
    peaks = argrelextrema(m, np.greater)[0]
    dips = argrelextrema(m, np.less)[0]
    return lam, m, lam[peaks], lam[dips]


def test_self_reversal_needs_a_cooler_mantle():
    hot_core = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=4000.0, wall_temperature_k=1500.0, fill=HPS_FILL)
    lam, m, peaks, dips = _peaks_around_d(hot_core)
    assert peaks[0] < 585.5 and peaks[-1] > 593.0  # the two emission peaks bracket both lines
    # A dark floor spans both lines; with enough shells a faint bump appears between them,
    # where the mantle absorbs least, but it stays far below the peaks.
    floor = (lam > 588.9) & (lam < 589.7)
    assert m[floor].max() < 0.01 * m.max() and len(dips) >= 1
    for centre in (588.995095, 589.592424):  # the mantle absorbs the line centres
        assert m[np.argmin(np.abs(lam - centre))] < 0.01 * m.max()

    uniform = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=3000.0, wall_temperature_k=3000.0, fill=HPS_FILL)
    lam, m, peaks, dips = _peaks_around_d(uniform)
    assert len(dips) == 0  # flat-topped at the blackbody, no dip
    ratio = m / planck.spectral_radiance(air_to_vacuum(lam), 3000.0)
    assert ratio.max() == pytest.approx(1.0, abs=1e-9) and ratio.max() <= 1.0 + 1e-12


def test_peaks_separate_as_sodium_pressure_rises():
    gaps = []
    for p_na in (6.5e3, 13e3, 26e3):
        fill = Fill(sodium_pa=p_na, mercury_pa=70e3, xenon_pa=20e3)
        col = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=4000.0, wall_temperature_k=1500.0, fill=fill)
        _, _, peaks, _ = _peaks_around_d(col)
        gaps.append(peaks[-1] - peaks[0])
    assert gaps == sorted(gaps) and gaps[-1] > 1.5 * gaps[0]


def test_thin_wings_brighten_at_grazing_angles():
    col = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=4000.0, wall_temperature_k=1500.0, fill=HPS_FILL, shells=20)
    lam = np.array([450.0])  # far wing of the D lines: optically thin
    side = arc.side_radiance(col, lam, lines=D_LINES).mean_radiance[0]
    grazing = arc.side_radiance(col, lam, theta_rad=np.pi / 6, lines=D_LINES).mean_radiance[0]
    assert grazing / side == pytest.approx(2.0, rel=1e-2)


def test_analysis_bins_converge_on_the_grids():
    """Refining the line-local grid or the shells 2× moves each 1 nm bin by < 0.1 % of
    itself outside the dark centre and by < 0.1 % of the peak bin everywhere. The dark
    centre's own level is set by the coolest shell and converges slowly (it is < 0.1 %
    of the peaks either way)."""
    grid = analysis_grid()
    outside = (grid.centres < 585.0) | (grid.centres > 594.0)

    def bins(shells=60, **kw):
        col = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=4000.0, wall_temperature_k=1500.0,
                            fill=HPS_FILL, shells=shells)
        lam = arc.line_grid(LINES, 359.5, 830.5, **kw)
        m = arc.side_radiance(col, lam, impact_parameters=120).mean_radiance
        return Spectrum.of(TabulatedContinuum(lam, m)).bin(grid).values

    base = bins()
    for refined in (bins(first_offset_nm=5e-5, ratio=np.sqrt(1.05), base_step_nm=0.5), bins(shells=120)):
        change = np.abs(refined - base)
        assert change.max() < 1e-3 * base.max()
        assert np.all(change[outside] < 1e-3 * base[outside])


def _multiplet(name):
    return tuple(ln for ln in LINES if ln.stark is not None and ln.stark.name == name)


def test_stark_widths_follow_stark_b_by_hand():
    """STARK-B gives the FWHM w and the shift d (positive to the red) in Å at the multiplet's
    mean wavelength; in angular frequency they are 2πc·w/λ², the same for every component."""
    table = _multiplet("3p-4d")[0].stark
    per_angstrom = 2 * np.pi * C * 1e-10 / (5686.4e-10) ** 2
    w, d = table.width_and_shift_rad_s(1e22, 5000.0)  # 10¹⁶ cm⁻³, a tabulated point
    assert w == pytest.approx(2.22 * per_angstrom, rel=1e-12)
    assert d == pytest.approx(-0.772 * per_angstrom, rel=1e-12)  # to the red: negative
    # Between grid points: per electron, linear in log N and log T. At √(10¹⁵·10¹⁶) cm⁻³
    # and √(2500·5000) K, the mean of the four per-electron values.
    n = np.sqrt(1e15 * 1e16)
    per_e = np.mean([0.253 / 1e15, 0.242 / 1e15, 2.25 / 1e16, 2.22 / 1e16])
    w, _ = table.width_and_shift_rad_s(n * 1e6, np.sqrt(2500.0 * 5000.0))
    assert w == pytest.approx(per_e * n * per_angstrom, rel=1e-12)
    # Below the table, proportional to nₑ; above 10 000 K, held.
    w_low, _ = table.width_and_shift_rad_s(1e17, 5000.0)
    assert w_low == pytest.approx(0.0243e-3 * per_angstrom, rel=1e-12)
    assert table.width_and_shift_rad_s(1e22, 20000.0) == table.width_and_shift_rad_s(1e22, 10000.0)


def test_lines_ending_on_3p_take_their_level_s_resonance_width():
    c = composition(3000.0, HPS_FILL)
    d2, d1 = (broadening.widths(ln, c).resonance for ln in D_LINES)
    for ln in LINES:
        if ln.lower_energy_cm1 > 0:
            expected = d2 if ln.lower_g == 4 else d1
            assert broadening.widths(ln, c).resonance == pytest.approx(expected, rel=1e-12)


def test_every_line_is_isolated_at_an_hps_axis():
    """The line set: every 3p-lower multiplet STARK-B tabulates at 10¹⁶ cm⁻³. Each stays
    inside its isolated-line limit at the axis's nₑ (3p–5d only just: 1.07×10²² m⁻³)."""
    axis = composition(4000.0, HPS_FILL).electrons
    assert {ln.stark.name for ln in LINES} == {"3s-3p", "3p-3d", "3p-4d", "3p-5d", "3p-4s", "3p-5s", "3p-6s", "3p-7s"}
    for ln in LINES:
        assert ln.stark.isolated_line_limit_m3() > axis


def test_stark_shift_moves_a_line_to_the_red():
    """On an HPS axis 3p–5d is Stark-dominated: its absorption peaks to the red of the line,
    by the profile's shift (Stark plus van der Waals), and Stark is most of that shift."""
    c = composition(4000.0, HPS_FILL)
    ln = sodium.data().line("3p1/2-5d3/2")
    w = broadening.widths(ln, c)
    assert w.shift_stark < 0 and w.shift_stark / w.shift > 0.8
    lam = np.linspace(ln.wavelength_nm - 0.5, ln.wavelength_nm + 0.5, 20001)
    kappa = arc.absorption_coefficient(c, C / (air_to_vacuum(lam) * 1e-9), lines=(ln,))
    expected = ln.vacuum_wavelength_nm**2 * 1e-9 * (-w.shift) / (2 * np.pi * C)  # nm
    assert lam[np.argmax(kappa)] - ln.wavelength_nm == pytest.approx(expected, rel=2e-3, abs=1e-4)


def test_thin_arc_gives_the_819_nm_lines_emission():
    """The absolute check of the D lines, for the 3p–3d lines: with τ ≪ 1 each chord gives
    ∫I dλ = Σ (hν₀/4π) A n_k · chord length, n_k the 3d population."""
    t = 2500.0
    lines = _multiplet("3p-3d")
    col = arc.ArcColumn(radius_m=3.8e-3, axis_temperature_k=t, wall_temperature_k=t, fill=Fill(sodium_pa=1e-3), shells=4)
    lam = arc.line_grid(lines, 817.5, 820.3, first_offset_nm=2e-5, ratio=1.01, base_step_nm=0.01)
    s = arc.side_radiance(col, lam, impact_parameters=5, lines=lines)
    c = composition(t, col.fill)
    per_length = sum(H * ln.frequency_hz / (4 * np.pi) * ln.a_ki
                     * c.sodium * sodium.level_population_fraction(ln.upper_energy_cm1, ln.upper_g, t, c.lowering_cm1)
                     for ln in lines)
    chord = 2.0 * np.sqrt(col.radius_m**2 - s.impact_parameter_m**2)
    assert s.radiance.max() < 1e-2 * planck.spectral_radiance(819.0, t)
    assert np.trapezoid(s.radiance, lam, axis=1) == pytest.approx(per_length * chord, rel=2e-3)
