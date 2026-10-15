"""Line broadening in the arc, rung 1: impact theory (design §4.4, "Line profile").

Each line gets a Voigt profile in frequency: the Doppler Gaussian of the emitters'
thermal motion, convolved with one Lorentzian whose width is the sum of

* the natural width, taken as the line's own A_ki. The full natural width is the sum of
  both levels' total decay rates, so this undercounts (for a line ending on 3p by about
  6×10⁷ s⁻¹), against a resonance width of at least 6.9×10¹⁰ rad/s anywhere in an HPS
  arc (13 kPa sodium);
* resonance broadening of the 3p level by ground-state sodium, measured (Huennekens &
  Gallagher 1983, in ``lss/data/na_atomic.yaml``), for the D lines and for every line
  ending on 3p; impact theory makes it independent of temperature;
* van der Waals broadening by mercury and xenon, from Lindholm–Foley impact theory with
  Unsöld's C₆: an estimate, known to fall short of measured widths by a factor of about 2
  (it stands in until measured Na–Hg and Na–Xe constants are adopted, e.g. M. J. Jongerius,
  J. Phys. B 20, 3345 (1987) for xenon);
* Stark broadening and shift by electron impacts, from STARK-B's semiclassical tables
  (Dimitrijević & Sahal-Bréchot; :class:`lss.physics.hid.sodium.StarkTable`). Small for
  the D lines (about 7 % of their resonance width on an HPS axis), dominant for the lines
  from loosely bound upper levels (4d, 5d → 3p).

Left out: broadening by the ions' quasi-static fields (up to about 25 % of the width of
the nd → 3p lines in HPS-like arcs, Mioković et al. 2004, arXiv physics/0410146), and the
forbidden components (3p–nf, ng) that appear where a line stops being isolated. Rung 1
is symmetric about the shifted centre; the asymmetric quasi-static wings (red Na–Hg
wing, the 551 nm satellite) are rung 2.

Widths are full widths at half maximum in angular frequency (rad/s), as the sources give
them; :func:`voigt_parameters` turns them into the σ and γ of a profile in Hz.
"""

from dataclasses import dataclass

import numpy as np

from lss.physics.hid import sodium
from lss.physics.hid.composition import Composition
from lss.spectral.units import C, K_B

ATOMIC_MASS_KG = 1.66053906660e-27
HBAR = 1.054571817e-34
#: Atomic unit of C₆, E_h·a₀⁶ in J·m⁶ (CODATA 2018: E_h = 4.3597447222071e-18 J,
#: a₀ = 5.29177210903e-11 m).
C6_ATOMIC_UNIT = 4.3597447222071e-18 * 5.29177210903e-11**6

#: Static dipole polarizabilities in atomic units (a₀³), the recommended values of
#: P. Schwerdtfeger and J. K. Nagle, "2018 Table of static dipole polarizabilities of the
#: neutral elements in the periodic table", Mol. Phys. 117, 1200 (2019).
POLARIZABILITY_AU = {"mercury": 33.91, "xenon": 27.32}  # ±0.34, ±0.20
MASS_U = {"mercury": 200.592, "xenon": 131.293}  # IUPAC standard atomic weights

#: Lindholm–Foley impact theory for an R⁻⁶ interaction: the Lorentzian FWHM is
#: 8.08 (C₆/ħ)^{2/5} v̄^{3/5} N and the shift −2.94 (C₆/ħ)^{2/5} v̄^{3/5} N (towards the red),
#: in angular frequency; as tabulated by D. F. Gray, *The Observation and Analysis of
#: Stellar Photospheres*, 3rd ed. (Cambridge 2005), ch. 11, with C₆ in angular units.
LF_WIDTH, LF_SHIFT = 8.08, 2.94


@dataclass(frozen=True)
class Widths:
    """Broadening of one line at one point of the arc, rad/s (FWHM) and rad/s (shift)."""

    doppler_sigma_hz: float  # standard deviation of the Gaussian, Hz
    natural: float
    resonance: float
    van_der_waals_mercury: float  # kept apart so mercury-pair wings can replace it (rung 2)
    van_der_waals_xenon: float
    stark: float
    shift_van_der_waals: float  # centre shifts, rad/s; negative is to the red
    shift_stark: float

    @property
    def van_der_waals(self) -> float:
        return self.van_der_waals_mercury + self.van_der_waals_xenon

    @property
    def lorentzian(self) -> float:
        return self.natural + self.resonance + self.van_der_waals + self.stark

    @property
    def shift(self) -> float:
        return self.shift_van_der_waals + self.shift_stark


def unsold_c6(line: sodium.Transition, perturber: str) -> float:
    """Unsöld's C₆ in J·m⁶: α_p · (⟨r²⟩_upper − ⟨r²⟩_lower), atomic units."""
    dr2 = (sodium.mean_square_radius_a0(line.upper_energy_cm1, line.upper_l)
           - sodium.mean_square_radius_a0(line.lower_energy_cm1, line.lower_l))
    return POLARIZABILITY_AU[perturber] * dr2 * C6_ATOMIC_UNIT


def _mean_relative_speed(perturber: str, temperature_k: float) -> float:
    mu = sodium.MASS_KG * MASS_U[perturber] * ATOMIC_MASS_KG / (sodium.MASS_KG + MASS_U[perturber] * ATOMIC_MASS_KG)
    return np.sqrt(8.0 * K_B * temperature_k / (np.pi * mu))


def widths(line: sodium.Transition, comp: Composition) -> Widths:
    t = comp.temperature_k
    ground = comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1)
    stark, stark_shift = line.stark.width_and_shift_rad_s(comp.electrons, t) if line.stark else (0.0, 0.0)
    vdw_width, vdw_shift = {}, 0.0
    for perturber, n in (("mercury", comp.mercury), ("xenon", comp.xenon)):
        vdw_width[perturber] = 0.0
        if n <= 0:
            continue
        scale = (unsold_c6(line, perturber) / HBAR) ** 0.4 * _mean_relative_speed(perturber, t) ** 0.6 * n
        vdw_width[perturber] = LF_WIDTH * scale
        vdw_shift -= LF_SHIFT * scale
    return Widths(
        doppler_sigma_hz=line.frequency_hz * np.sqrt(K_B * t / (sodium.MASS_KG * C * C)),
        natural=line.a_ki,
        resonance=line.self_broadening_m3_s * float(ground),
        van_der_waals_mercury=vdw_width["mercury"],
        van_der_waals_xenon=vdw_width["xenon"],
        stark=stark,
        shift_van_der_waals=vdw_shift,
        shift_stark=stark_shift,
    )


def voigt_parameters(w: Widths) -> tuple[float, float, float]:
    """(σ, γ, shift) in Hz for ``scipy.special.voigt_profile``: γ is the Lorentzian half width."""
    return w.doppler_sigma_hz, w.lorentzian / (4.0 * np.pi), w.shift / (2.0 * np.pi)
