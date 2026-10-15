"""LED Level 2: a phosphor layer on a blue die (design §4.2 L2).

The die emits blue light into a layer that holds one or more phosphors. Each
phosphor absorbs part of it (Beer–Lambert), re-emits the absorbed photons in its
own band, and the layer absorbs part of that light again on its way out, which
feeds the phosphors that absorb there (a red nitride takes some of the garnet's
green). The blue that leaks through, and the colour, follow from how much of
each phosphor the light crosses, instead of being fitted as a power ratio.

Everything is counted in photons on a wavenumber grid (ν̃ = 10⁷/λ, cm⁻¹, for λ
in nm), because a phosphor converts photons one for one (times its quantum
yield) and its band shapes are set in photon energy. The Stokes loss is what is
left when a blue photon becomes a yellow one: power per nm is photons × ν̃ ×
the ν̃²/10⁷ Jacobian.

The layer (what this rung leaves out is listed in the design doc)
-----------------------------------------------------------------
* A flat slab with no scattering. Its optical depth at wavenumber ν̃ along the
  normal is τ(ν̃) = Σᵢ τᵢ aᵢ(ν̃): phosphor i's depth where its absorption shape aᵢ
  is 1. A path at polar angle θ inside it crosses τ/cos θ. Real layers scatter,
  which lengthens the paths, so a fitted τᵢ is an effective depth.
* The die emits into the layer from below as a Lambertian source: the flux
  fraction per unit μ = cos θ is 2μ, so the fraction reaching depth z (0 at the
  die, 1 at the top) is 2E₃(τz).
* Phosphor light is emitted isotropically. Light going down meets the die and is
  reflected with reflectance R; light reaching the top leaves. Refraction and
  total internal reflection at the top surface are left out, so a zone's angles
  are inside the layer.
* Light the phosphors absorb a second time is re-emitted from the depth profile
  of the absorbing phosphor's first absorption (a one-profile approximation of
  the depth redistribution).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc, erfcx, expn

#: Wavelength grid of the layer calculation (nm). The CIE tables span 380–780 nm;
#: the margin holds the die's blue tail and the phosphors' red tails.
GRID_NM = np.arange(350.0, 900.0 + 0.25, 0.5)
NU = 1e7 / GRID_NM  # cm⁻¹, falling along the grid
_DNU = np.abs(np.gradient(NU))  # cm⁻¹ per grid point, for sums over ν̃
#: Second radiation constant hc/k in cm·K: k·T in cm⁻¹ is T / C2.
C2_CM_K = 1.438776877
_HALF_MAX_Z = np.sqrt(2.0 * np.log(2.0))
#: Depth slabs and Gauss–Legendre directions per angle zone.
_DEPTHS = 24
_DIRECTIONS = 12
#: Default emission-angle zones (polar angle inside the layer, degrees).
ZONES_DEG = ((0.0, 30.0), (30.0, 60.0), (60.0, 90.0))


def _unit_area(shape: np.ndarray) -> np.ndarray:
    return shape / np.sum(shape * _DNU)


def _gauss(nu0: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((NU - nu0) / sigma) ** 2)


def _sigma_from_fwhm_nm(peak_nm: float, fwhm_nm: float) -> float:
    """σ (cm⁻¹) of a Gaussian in energy whose half-maximum points are ``fwhm_nm`` apart in
    wavelength (the convention of :meth:`lss.physics.led.bands.EnergyBand.symmetric`)."""
    nu = 1e7 / peak_nm
    hw = (np.sqrt(1e14 + (fwhm_nm * nu) ** 2) - 1e7) / fwhm_nm
    return hw / _HALF_MAX_Z


def die_emission(peak_nm: float, fwhm_nm: float, temperature_k: float) -> np.ndarray:
    """Photons per cm⁻¹ (unit area) from an InGaN die.

    A Gaussian in energy (alloy disorder: ``peak_nm`` and ``fwhm_nm`` are its own, in
    the energy-shape convention) convolved with the Boltzmann tail exp(−ΔE/kT) of
    carriers above the band edge, which gives the emission a longer short-wavelength
    side (E. F. Schubert, *Light-Emitting Diodes*, 2nd ed., §5.1: the high-energy
    slope of an LED's spectrum is set by kT). T is the tail's slope written as a
    temperature. In the CIE LEDs' fits it is a shape parameter rather than the junction
    temperature: the warm ones want it below room temperature (design §4.2).
    """
    nu0 = 1e7 / peak_nm
    sigma = _sigma_from_fwhm_nm(peak_nm, fwhm_nm)
    kt = temperature_k / C2_CM_K
    d = NU - nu0
    return _unit_area(np.exp(-0.5 * (d / sigma) ** 2) * erfcx((sigma / kt - d / sigma) / np.sqrt(2.0)))


def ce_garnet(zero_phonon_cm: float, huang_rhys: float, phonon_cm: float, temperature_k: float,
              inhomogeneous_fwhm_cm: float, ground_splitting_cm: float, ratio_2f7_2f5: float):
    """Absorption shape (1 at its peak) and emission (photons per cm⁻¹, unit area) of Ce³⁺ in a garnet.

    Configurational-coordinate model with one effective phonon (B. Henderson and
    G. F. Imbusch, *Optical Spectroscopy of Inorganic Solids*, 1989, ch. 5): the
    4f → 5d₁ absorption peaks S·ħω above the zero-phonon line and the emission
    S·ħω below it, both Gaussian with σ² = S(ħω)² coth(ħω/2kT), widened by an
    inhomogeneous spread of sites. The emission ends on the two spin-orbit levels
    of the 4f ground state, ²F₅/₂ and ²F₇/₂, ``ground_splitting_cm`` apart, so it
    is two such Gaussians; ``ratio_2f7_2f5`` is the photon ratio of the lower-energy
    one to the other.
    """
    kt = temperature_k / C2_CM_K
    shift = huang_rhys * phonon_cm
    sigma_inh = inhomogeneous_fwhm_cm / (2.0 * _HALF_MAX_Z)
    sigma = np.sqrt(huang_rhys * phonon_cm**2 / np.tanh(phonon_cm / (2.0 * kt)) + sigma_inh**2)
    absorption = _gauss(zero_phonon_cm + shift, sigma)
    upper = _gauss(zero_phonon_cm - shift, sigma)
    lower = _gauss(zero_phonon_cm - shift - ground_splitting_cm, sigma)
    return absorption, _unit_area(upper + ratio_2f7_2f5 * lower)


def eu_nitride(emission_peak_nm: float, emission_fwhm_nm: float, absorption_edge_nm: float,
               edge_width_cm: float):
    """Absorption shape (→ 1 on the blue side) and emission (photons per cm⁻¹, unit area) of a
    red Eu²⁺ nitride.

    Phenomenological: the emission is one Gaussian in energy; the absorption, the
    overlapping 4f → 5d bands of Eu²⁺, is a smooth edge, 1/2 at ``absorption_edge_nm``
    and falling to the red over ``edge_width_cm`` (the 1/e scale of the error function).
    """
    nu_edge = 1e7 / absorption_edge_nm
    absorption = 0.5 * erfc((nu_edge - NU) / edge_width_cm)
    emission = _gauss(1e7 / emission_peak_nm, _sigma_from_fwhm_nm(emission_peak_nm, emission_fwhm_nm))
    return absorption, _unit_area(emission)


@dataclass(frozen=True)
class Phosphor:
    """One phosphor in the layer, on :data:`GRID_NM`."""

    absorption: np.ndarray  # shape, 1 where ``optical_depth`` is quoted
    emission: np.ndarray  # photons per cm⁻¹, unit area
    optical_depth: float  # along the normal, where the absorption shape is 1
    quantum_yield: float = 1.0


@dataclass(frozen=True)
class LayerOutput:
    """Photons per cm⁻¹ leaving the top of the layer in each angle zone, on :data:`GRID_NM`,
    for one pump photon."""

    zones_deg: tuple[tuple[float, float], ...]
    leaked: np.ndarray  # (zone, grid): pump light that crossed the layer
    converted: np.ndarray  # (zone, phosphor, grid): phosphor light that left the layer
    absorbed: np.ndarray  # (phosphor,): photons each phosphor absorbed, re-absorption included
    lost: float  # photons absorbed by the die (R < 1) or by a quantum yield < 1

    def photons(self, zone: int | None = None) -> np.ndarray:
        """Photons per cm⁻¹ in one zone, or summed over the zones."""
        total = self.leaked + self.converted.sum(axis=1)
        return total.sum(axis=0) if zone is None else total[zone]

    def power_per_nm(self, zone: int | None = None) -> np.ndarray:
        """Relative radiant power per nm: photons × photon energy (∝ ν̃) × the ν̃²/10⁷ Jacobian."""
        return self.photons(zone) * NU**3 / 1e7


def _directions(zones_deg):
    """Gauss–Legendre nodes and weights in μ = cos θ for each zone."""
    x, w = np.polynomial.legendre.leggauss(_DIRECTIONS)
    out = []
    for lo, hi in zones_deg:
        if not 0.0 <= lo < hi <= 90.0:
            raise ValueError(f"angle zone {lo}-{hi} deg is not inside 0-90 deg")
        mu_lo, mu_hi = np.cos(np.radians(hi)), np.cos(np.radians(lo))
        half = (mu_hi - mu_lo) / 2.0
        out.append((mu_lo + half * (x + 1.0), half * w))
    return out


def transfer(pump: np.ndarray, phosphors: list[Phosphor], die_reflectance: float = 1.0,
             zones_deg=ZONES_DEG) -> LayerOutput:
    """Run the pump (photons per cm⁻¹ on :data:`GRID_NM`, unit area) through the layer."""
    if not phosphors:
        raise ValueError("a phosphor layer needs at least one phosphor")
    if not 0.0 <= die_reflectance <= 1.0:
        raise ValueError("die reflectance must be in [0, 1]")
    if any(p.optical_depth < 0 or not 0.0 <= p.quantum_yield <= 1.0 for p in phosphors):
        raise ValueError("optical depths must be ≥ 0 and quantum yields in [0, 1]")
    pump = _unit_area(pump)
    taus = np.array([p.optical_depth * p.absorption for p in phosphors])  # (phosphor, grid)
    tau = taus.sum(axis=0)
    share = np.divide(taus, tau, out=np.zeros_like(taus), where=tau > 0)  # who absorbs at ν̃

    edges = np.linspace(0.0, 1.0, _DEPTHS + 1)
    z = (edges[:-1] + edges[1:]) / 2.0
    dirs = _directions(zones_deg)

    # Pump: what reaches each depth is 2E₃(τz); a slab absorbs the difference.
    reach = 2.0 * expn(3, tau[None, :] * edges[:, None])  # (edge, grid)
    slab = reach[:-1] - reach[1:]  # (depth, grid)
    leaked = _leak(tau, dirs) * pump[None, :]  # (zone, grid)
    first = share[:, None, :] * slab[None, :, :] * pump[None, None, :]  # (phosphor, depth, grid)
    profile = np.einsum("pdg,g->pd", first, _DNU)  # where each phosphor absorbs the pump
    absorbed_first = profile.sum(axis=1)
    norm = absorbed_first[:, None]
    profile = np.divide(profile, norm, out=np.full_like(profile, 1.0 / _DEPTHS), where=norm > 0)

    # Isotropic emission at depth z: half goes up and leaves through τ(1−z)/μ; half goes down,
    # reaches the die through τz/μ, and a fraction R returns through τ(1+z)/μ.
    esc = np.array([_escape(tau, z, m, w, die_reflectance) for m, w in dirs])  # (zone, depth, grid)
    lost_die = (1.0 - die_reflectance) * 0.5 * expn(2, tau[None, :] * z[:, None])  # (depth, grid)
    esc_p = np.einsum("pd,zdg->pzg", profile, esc)  # (phosphor, zone, grid)
    esc_total = esc_p.sum(axis=1)  # (phosphor, grid)
    lost_p = profile @ lost_die  # (phosphor, grid)
    reabsorbed = 1.0 - esc_total - lost_p  # an emitted photon is absorbed again in the layer

    # Photon balance: N = A + M·(η N), where N are absorbed photons per phosphor, η the
    # quantum yields and M[i, j] the chance that a photon phosphor j emits is absorbed by i.
    eta = np.array([p.quantum_yield for p in phosphors])
    emission = np.array([p.emission for p in phosphors])  # (phosphor, grid)
    m = np.einsum("jg,jg,ig,g->ij", emission, reabsorbed, share, _DNU)
    n = np.linalg.solve(np.eye(len(phosphors)) - m * eta[None, :], absorbed_first)
    emitted = eta * n
    converted = np.einsum("j,jg,jzg->zjg", emitted, emission, esc_p)
    lost = float(np.sum((1.0 - eta) * n) + np.sum(emitted * np.einsum("jg,jg,g->j", emission, lost_p, _DNU)))
    return LayerOutput(tuple(tuple(zd) for zd in zones_deg), leaked, converted, n, lost)


def _leak(tau, dirs):
    """Fraction of a Lambertian pump crossing the whole layer, per zone: ∫ 2μ e^(−τ/μ) dμ."""
    return np.array([np.sum((w * 2.0 * m)[:, None] * np.exp(-tau[None, :] / m[:, None]), axis=0)
                     for m, w in dirs])


def _escape(tau, z, mu, w, reflectance):
    """Chance that a photon emitted isotropically at depth z leaves the top in this zone."""
    up = np.exp(-tau[None, None, :] * (1.0 - z)[None, :, None] / mu[:, None, None])
    down = np.exp(-tau[None, None, :] * (1.0 + z)[None, :, None] / mu[:, None, None])
    return 0.5 * np.einsum("m,mdg->dg", w, up + reflectance * down)
