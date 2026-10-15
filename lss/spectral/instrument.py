"""A spectrum as a spectrograph records it: blur, wavelength offset and sampling (design §11).

A line at λ₀ is recorded at the nodes x_k with weight K(x_k − λ₀ − δ(λ₀)), where K is a
Gaussian of width σ convolved with a one-sided exponential tail of length τ to the red
(an exponentially modified Gaussian, unit area) and δ is the instrument's wavelength
offset, which may drift linearly with wavelength. A continuum is recorded as the same
kernel's convolution: each node is the kernel-weighted integral of the density. Sampled
every Δx, a line's samples sum to its power, Σ_k s_k · Δx ≈ P, to the extent the kernel
is wide against Δx (within 0.3 % for this instrument's 1.6 nm FWHM at 1 nm steps).

The kernel is fitted to lamps whose lines are far narrower than the instrument
(:func:`fit_kernel`): each line has a free power and a flat background in its own
window; σ, τ and the offset are shared.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import exponnorm

#: Wavelength at which a linear offset drift is anchored (nm).
OFFSET_ANCHOR_NM = 600.0


@dataclass(frozen=True)
class Kernel:
    """Instrument response: Gaussian core σ, red tail τ, offset δ(λ) = δ₀ + slope·(λ − 600 nm)/100 nm."""

    sigma_nm: float
    tail_nm: float
    offset_nm: float = 0.0
    offset_slope_per_100nm: float = 0.0

    def __post_init__(self):
        if self.sigma_nm <= 0 or self.tail_nm <= 0:
            raise ValueError("kernel width and tail must be positive")

    def offset(self, wavelength_nm):
        return self.offset_nm + self.offset_slope_per_100nm * (np.asarray(wavelength_nm, dtype=float) - OFFSET_ANCHOR_NM) / 100.0

    def shape(self, u_nm):
        """K(u) per nm, unit area; u is recorded minus true wavelength, before the offset."""
        return exponnorm.pdf(np.asarray(u_nm, dtype=float), self.tail_nm / self.sigma_nm, scale=self.sigma_nm)

    def line(self, nodes_nm, centre_nm: float, power: float = 1.0) -> np.ndarray:
        """Samples a line of ``power`` at ``centre_nm`` gives at the nodes."""
        return power * self.shape(np.asarray(nodes_nm, dtype=float) - centre_nm - self.offset(centre_nm))

    @property
    def fwhm_nm(self) -> float:
        u = np.linspace(-6.0 * self.sigma_nm, 6.0 * self.sigma_nm + 12.0 * self.tail_nm, 40001)
        k = self.shape(u)
        above = u[k >= 0.5 * k.max()]
        return float(above[-1] - above[0])

    @property
    def mode_nm(self) -> float:
        """Where a line at 600 nm peaks, relative to 600 nm (offset included)."""
        u = np.linspace(-6.0 * self.sigma_nm, 6.0 * self.sigma_nm + 12.0 * self.tail_nm, 40001)
        return float(u[np.argmax(self.shape(u))] + self.offset(OFFSET_ANCHOR_NM))


def observe(wavelength_nm, density, kernel: Kernel, nodes_nm, step_nm: float = 0.01) -> np.ndarray:
    """The density (any units per nm, on any increasing grid) as the instrument records it at the nodes.

    The density is first averaged into bins of ``step_nm`` (exactly, from its trapezoid
    integral, so power is kept however fine its own grid), then convolved with the
    kernel. Nodes whose kernel reaches past the density's grid are NaN.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    f = np.asarray(density, dtype=float)
    nodes = np.asarray(nodes_nm, dtype=float)
    reach_lo = 6.0 * kernel.sigma_nm
    reach_hi = 6.0 * kernel.sigma_nm + 15.0 * kernel.tail_nm
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lam))])
    edges = np.arange(lam[0], lam[-1], step_nm)
    binned = np.diff(np.interp(edges, lam, cum)) / step_nm
    centres = 0.5 * (edges[1:] + edges[:-1])
    u = np.arange(-reach_lo, reach_hi, step_nm)
    conv = np.convolve(binned, kernel.shape(u) * step_nm, mode="full")
    xs = centres[0] + u[0] + np.arange(conv.size) * step_nm
    shifted = nodes - kernel.offset(nodes)
    out = np.interp(shifted, xs, conv)
    inside = (shifted - reach_hi >= lam[0]) & (shifted + reach_lo <= lam[-1])
    return np.where(inside, out, np.nan)


@dataclass(frozen=True)
class KernelFit:
    kernel: Kernel
    lines_nm: tuple[float, ...]
    powers: tuple[float, ...]
    rms_over_peak: float  # residual rms over all windows, over the highest sample
    worst_over_peak: tuple[float, ...]  # each window's largest |residual| over its own peak


def fit_kernel(nodes_nm, values, lines_nm, half_window_nm: float = 4.0, drift: bool = True,
               start: Kernel = Kernel(0.45, 0.6, 0.0)) -> KernelFit:
    """Fit the kernel to isolated narrow lines of known wavelength (air, nm).

    Each line has its own power and flat background over ±``half_window_nm``; σ, τ and the
    offset (with a linear drift if ``drift``) are shared. Lines must not share a window.
    """
    x = np.asarray(nodes_nm, dtype=float)
    y = np.asarray(values, dtype=float)
    lines = tuple(sorted(float(w) for w in lines_nm))
    if any(b - a < 2 * half_window_nm for a, b in zip(lines, lines[1:])):
        raise ValueError("calibration lines must be at least two half-windows apart")
    masks = [np.abs(x - w) <= half_window_nm for w in lines]
    n = len(lines)
    peak = max(y[m].max() for m in masks)

    def unpack(p):
        k = Kernel(p[0], p[1], p[2], p[3] if drift else 0.0)
        return k, p[4:4 + n], p[4 + n:]

    def residuals(p):
        k, powers, bg = unpack(p)
        return np.concatenate([k.line(x[m], w, a) + b - y[m] for m, w, a, b in zip(masks, lines, powers, bg)])

    p0 = [start.sigma_nm, start.tail_nm, start.offset_nm, 0.0] + [y[m].max() * 2.0 for m in masks] + [0.0] * n
    lo = [0.05, 0.01, -3.0, -2.0] + [0.0] * n + [-np.inf] * n
    hi = [3.0, 5.0, 3.0, 2.0] + [np.inf] * n + [np.inf] * n
    if not drift:
        lo[3], hi[3] = -1e-12, 1e-12
    r = least_squares(residuals, p0, bounds=(lo, hi), x_scale="jac")
    k, powers, bg = unpack(r.x)
    worst, i = [], 0
    for m in masks:
        seg = r.fun[i:i + m.sum()]
        worst.append(float(np.abs(seg).max() / y[m].max()))
        i += m.sum()
    return KernelFit(k, lines, tuple(float(a) for a in powers), float(np.sqrt(np.mean(r.fun**2)) / peak), tuple(worst))
