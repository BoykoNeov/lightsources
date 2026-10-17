"""Radiative transfer through a cylindrical arc (design §4.4 Stages A and B).

Stage A: in each radial shell the gas is in LTE at the shell's temperature, so each line
absorbs with

    κ_ν = c²/(8πν₀²) · A_ki · (g_k/g_i) · n_i · φ_ν(ν) · (1 − exp(−hν/kT))      (m⁻¹)

(the frequency form of the design's κ_λ; φ_ν per Hz) and emits ε_ν = κ_ν B_ν(T)
(Kirchhoff), so emission and absorption are consistent by construction.

Stage B: along a chord at impact parameter b and angle θ to the tube's axis the arc is a
stack of shells of constant temperature, and each crossing is solved exactly,

    I ← I·e^{−Δτ} + B(T)·(1 − e^{−Δτ}),      Δτ = κ·ℓ/sin θ,

from the far wall to the near one. A hot core seen through a cooler mantle gives the
self-reversed line: broad emission from the core, its centre absorbed by the mantle.

The temperature falls from the axis to the wall as T(r) = T_w + (T_ax − T_w)(1 − (r/R)^p),
prescribed (parabolic for p = 2) until Stage C solves it.

With ``pairs`` set, the D lines' wings go over from the resonance share of their impact
profile to sodium atom pairs (:mod:`lss.physics.hid.pairs`) with distance from the line;
with ``mercury_pairs``, from the mercury van der Waals share to sodium–mercury pairs
(:mod:`lss.physics.hid.mercury_pairs`). The hand-over is gradual (:func:`join_weight`):
the impact wing and the quasi-static wing do not meet at any one distance, so a cut
leaves a step in κ, which the self-reversed line shows as a false maximum.
"""

from dataclasses import dataclass, replace
import itertools

import numpy as np
from scipy.special import voigt_profile

from lss.physics.hid import broadening, mercury_pairs as hg_pairs, pairs as na_pairs, sodium
from lss.physics.hid.composition import Composition, Fill, composition
from lss.physics.thermal import planck
from lss.spectral.units import C, H, K_B, air_to_vacuum


@dataclass(frozen=True)
class ArcColumn:
    radius_m: float
    axis_temperature_k: float
    wall_temperature_k: float
    fill: Fill
    profile_exponent: float = 2.0
    shells: int = 60
    pairs: na_pairs.PairWings | None = None
    mercury_pairs: hg_pairs.MercuryPairWings | None = None

    def __post_init__(self):
        if self.radius_m <= 0 or self.shells < 1 or self.profile_exponent <= 0:
            raise ValueError("radius, shell count and profile exponent must be positive")
        if self.wall_temperature_k <= 0 or self.axis_temperature_k <= 0:
            raise ValueError("temperatures must be positive")

    def temperature(self, r_m):
        x = np.asarray(r_m, dtype=float) / self.radius_m
        return self.wall_temperature_k + (self.axis_temperature_k - self.wall_temperature_k) * (1.0 - x**self.profile_exponent)

    @property
    def shell_edges_m(self) -> np.ndarray:
        return np.linspace(0.0, self.radius_m, self.shells + 1)

    def shell_compositions(self) -> list[Composition]:
        """Each shell at the temperature of its mid-radius."""
        e = self.shell_edges_m
        return [composition(float(t), self.fill) for t in self.temperature(0.5 * (e[:-1] + e[1:]))]


def line_grid(lines, lo_nm: float = 350.0, hi_nm: float = 1000.0, first_offset_nm: float = 1e-4,
              ratio: float = 1.05, base_step_nm: float = 1.0) -> np.ndarray:
    """Air wavelengths resolving every line: offsets from each centre growing geometrically
    from ``first_offset_nm`` by ``ratio``, on top of a uniform ``base_step_nm`` grid."""
    n = int(np.ceil(np.log((hi_nm - lo_nm) / first_offset_nm) / np.log(ratio))) + 1
    offsets = first_offset_nm * ratio ** np.arange(n)
    pts = [np.arange(lo_nm, hi_nm + 0.5 * base_step_nm, base_step_nm)]
    for ln in lines:
        pts += [ln.wavelength_nm + offsets, ln.wavelength_nm - offsets, [ln.wavelength_nm]]
    grid = np.unique(np.concatenate(pts))
    return grid[(grid >= lo_nm) & (grid <= hi_nm)]


def _profile(frequency_hz, ln, w: broadening.Widths) -> np.ndarray:
    sigma, gamma, shift = broadening.voigt_parameters(w)
    return voigt_profile(frequency_hz - (ln.frequency_hz + shift), sigma, gamma)


def join_weight(offset_nm, join_nm: float) -> np.ndarray:
    """How much of a line's wing is handed to the pair absorption at ``offset_nm`` from it:
    0 inside ``join_nm``/2, 1 beyond 2·``join_nm``, between them a smooth step (3x² − 2x³)
    in log(offset), so κ and its slope stay continuous."""
    x = np.log(np.maximum(np.abs(np.asarray(offset_nm, dtype=float)), 1e-300) / (0.5 * join_nm)) / np.log(4.0)
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def absorption_coefficient(comp: Composition, frequency_hz: np.ndarray, lines=None,
                           pairs: na_pairs.PairWings | None = None,
                           mercury_pairs: hg_pairs.MercuryPairWings | None = None) -> np.ndarray:
    """κ_ν in m⁻¹ at the given (vacuum) frequencies, summed over ``lines`` (default: all).

    With ``pairs``, each resonance line (lower level 3s) hands its resonance share over to
    its own part of the sodium-pair absorption by the weight w = :func:`join_weight` of its
    distance: (1 − w) of the impact profile with the resonance width, w of it without, and
    w of the line's pairs. ``mercury_pairs`` does the same with the mercury van der Waals
    width and the sodium–mercury pairs. With several, the profiles mix multilinearly: each
    combination of dropped widths is weighted by the product of w for those dropped and
    1 − w for those kept.
    """
    lines = sodium.data().lines if lines is None else lines
    t = comp.temperature_k
    stimulated = -np.expm1(-H * frequency_hz / (K_B * t))
    vacuum_nm = C / frequency_hz * 1e9
    kappa = np.zeros_like(frequency_hz)
    ground = float(comp.sodium * sodium.level_population_fraction(0.0, 2, t, comp.lowering_cm1))
    handovers = []  # (impact width handed over, its join, the pair κ of a resonance line)
    if pairs is not None:
        handovers.append(("resonance", pairs.join_nm, lambda ln: pairs.kappa(t, ground, frequency_hz, ln)))
    if mercury_pairs is not None:
        handovers.append(("van_der_waals_mercury", mercury_pairs.join_nm,
                          lambda ln: mercury_pairs.kappa(t, ground, float(comp.mercury), frequency_hz, ln)))
    for ln in lines:
        n_lower = comp.sodium * sodium.level_population_fraction(ln.lower_energy_cm1, ln.lower_g, t, comp.lowering_cm1)
        w = broadening.widths(ln, comp)
        phi = _profile(frequency_hz, ln, w)
        if ln.lower_energy_cm1 == 0.0 and handovers:
            offset = vacuum_nm - ln.vacuum_wavelength_nm
            weights = [join_weight(offset, join) for _, join, _ in handovers]
            mixed = np.zeros_like(phi)
            for drops in itertools.product((False, True), repeat=len(handovers)):
                weight = np.prod([wt if d else 1.0 - wt for wt, d in zip(weights, drops)], axis=0)
                if not any(drops):
                    mixed = mixed + weight * phi
                elif weight.any():
                    wf = replace(w, **{field: 0.0 for (field, _, _), d in zip(handovers, drops) if d})
                    mixed = mixed + weight * _profile(frequency_hz, ln, wf)
            phi = mixed
            for (_, _, pair_kappa), wt in zip(handovers, weights):
                if wt.any():
                    kappa += wt * pair_kappa(ln)
        kappa += (C * C / (8.0 * np.pi * ln.frequency_hz**2)) * ln.a_ki * (ln.upper_g / ln.lower_g) * n_lower * phi
    return kappa * stimulated


@dataclass(frozen=True)
class ArcSpectrum:
    wavelength_nm: np.ndarray  # air
    impact_parameter_m: np.ndarray
    radiance: np.ndarray  # (impact parameter, wavelength), W m⁻² sr⁻¹ nm⁻¹ at the arc boundary
    theta_rad: float

    @property
    def mean_radiance(self) -> np.ndarray:
        """Radiance averaged over the arc's projected width (uniform in b), W m⁻² sr⁻¹ nm⁻¹."""
        return self.radiance.mean(axis=0)


def chord_lengths(edges_m: np.ndarray, b_m: np.ndarray) -> np.ndarray:
    """ℓ[k, j]: length of the chord at impact parameter b_k inside shell j, on one side of
    its closest approach (the chord crosses each shell it reaches twice)."""
    b = b_m[:, None]
    outer = np.sqrt(np.clip(edges_m[None, 1:] ** 2 - b * b, 0.0, None))
    inner = np.sqrt(np.clip(np.maximum(edges_m[None, :-1], b) ** 2 - b * b, 0.0, None))
    return np.clip(outer - inner, 0.0, None)  # (b, shell)


def side_radiance(column: ArcColumn, wavelength_nm: np.ndarray, theta_rad: float = np.pi / 2,
                  impact_parameters: int | None = None, lines=None) -> ArcSpectrum:
    """Radiance leaving the arc along chords at angle ``theta_rad`` to its axis.

    Impact parameters are the midpoints of ``impact_parameters`` equal strips across the
    radius (default: two per shell); the arc is taken as infinitely long.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    nu = C / (air_to_vacuum(lam) * 1e-9)
    comps = column.shell_compositions()
    kappa = np.array([absorption_coefficient(c, nu, lines, column.pairs, column.mercury_pairs) for c in comps])  # (shell, λ)
    source = np.array([planck.spectral_radiance(air_to_vacuum(lam), c.temperature_k) for c in comps])

    nb = impact_parameters or 2 * column.shells
    b = (np.arange(nb) + 0.5) * column.radius_m / nb
    ell = chord_lengths(column.shell_edges_m, b) / np.sin(theta_rad)  # (b, shell)

    intensity = np.zeros((nb, lam.size))
    order = list(range(column.shells - 1, -1, -1)) + list(range(column.shells))  # far side in, then out
    for j in order:
        tau = ell[:, j, None] * kappa[None, j, :]
        att = np.exp(-tau)
        intensity = intensity * att + source[None, j, :] * (1.0 - att)
    return ArcSpectrum(lam, b, intensity, theta_rad)
