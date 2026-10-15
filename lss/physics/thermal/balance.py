"""Filament temperature from the current alone: the power balance (design §4.1 L4).

A filament on a current supply settles where the electrical power it takes,
I²·R(T), equals the power it loses:

    I² · R_cold · ρ(T)/ρ(293 K)  =  P_rad(T) + P_gas(T)

* **Electrical input:** the resistance rises with temperature as tungsten's resistivity
  does (Desai et al. 1984, ``resistance.py``). The resistance of the leads and seals
  is in series with the filament and carries the same current, so it changes the
  voltage read across the lamp but not the filament's own power: it has no term here.
* **Radiation:** P_rad = A · f · ε_eff · σT⁴, where A is the wire's surface, f the
  coil's open fraction (Level 3) and ε_eff the one-zone cavity emissivity built from
  tungsten's *total* hemispherical emissivity (Forsythe & Watson 1934,
  ``tungsten_total_emissivity.yaml``). The power leaves through the coil's opening,
  f·A, at ε_eff; f = 1 is a straight wire. Applying the cavity formula to a total
  rather than wavelength by wavelength is a first step: the mean of ε_eff(ε(λ)) is not
  ε_eff of the mean ε. De Vos's spectral data cannot do it properly, because it stops
  at 2.5 µm, short of about 13 % of the power at 3100 K.
* **Gas loss:** heat carried off by the fill gas and conducted into the supports and
  leads, as one term, P_gas = P_gas(3000 K) · (T/3000 K)^m. Langmuir's sheath model
  would compute it, but the fill, its pressure and the coil's dimensions are unknown for
  these lamps. So the share at 3000 K and the exponent m are empirical. With heavy-gas
  conductivity rising as about T^0.75 and a wall near 900 K, the sheath model's slope
  at 3100 K is about 2.

The wire's surface A is not measured either. It is solved from one measured operating
point (``solve_area``), where the voltage gives the temperature through the resistance
(Level 4's first step). What the balance adds is the lamp at another current: how its
temperature, voltage and spectrum follow the current, which depends on how the losses
and the resistance scale with temperature, and not on the filament's size.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from importlib import resources
from typing import Callable

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from lss.physics.thermal import resistance, tungsten

#: Stefan–Boltzmann constant, CODATA 2018 (exact in SI), W m⁻² K⁻⁴.
SIGMA = 5.670374419e-8
#: The temperature at which the gas loss's share is stated.
T_GAS_REF_K = 3000.0


@dataclass(frozen=True)
class TotalEmissivityTable:
    temperatures_k: np.ndarray
    values: np.ndarray
    citation: str
    data_version: str

    @property
    def range_k(self) -> tuple[float, float]:
        return float(self.temperatures_k[0]), float(self.temperatures_k[-1])

    @cached_property
    def _interp(self) -> PchipInterpolator:
        return PchipInterpolator(self.temperatures_k, self.values)

    def __call__(self, temperature_k):
        t = np.asarray(temperature_k, dtype=float)
        lo, hi = self.range_k
        if np.any((t < lo) | (t > hi)):
            raise ValueError(f"tungsten total emissivity is tabulated for {lo:g}-{hi:g} K")
        return self._interp(t)

    def log_slope(self, temperature_k: float) -> float:
        """d ln ε / d ln T."""
        return float(self._interp(temperature_k, 1) * temperature_k / self._interp(temperature_k))


@lru_cache(maxsize=1)
def total_emissivity_table() -> TotalEmissivityTable:
    text = resources.files("lss.data").joinpath("tungsten_total_emissivity.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    rows = sorted((float(k), float(v)) for k, v in doc["table"].items())
    return TotalEmissivityTable(
        temperatures_k=np.array([t for t, _ in rows]),
        values=np.array([v for _, v in rows]),
        citation=doc["citation"],
        data_version=doc["data_version"],
    )


def power_law_emissivity(temperature_k: float, log_slope: float) -> Callable:
    """Total emissivity through the table's value at ``temperature_k`` with a chosen
    slope d ln ε / d ln T: for showing how a result depends on that slope, which the
    table's three figures fix poorly near 3000 K."""
    e0 = float(total_emissivity_table()(temperature_k))
    return lambda t: e0 * (np.asarray(t, dtype=float) / temperature_k) ** log_slope


@dataclass(frozen=True)
class FilamentBalance:
    """A current-driven coiled tungsten filament (design §4.1 L4)."""

    cold_resistance_ohm: float
    wire_area_m2: float
    open_fraction: float = 1.0
    gas_share: float = 0.0  # of the input power, at 3000 K
    gas_exponent: float = 2.0
    emissivity: Callable | None = None  # total ε(T); the Forsythe & Watson table if None

    def __post_init__(self):
        if self.cold_resistance_ohm <= 0 or self.wire_area_m2 <= 0:
            raise ValueError("cold resistance and wire area must be positive")
        if not (0.0 <= self.gas_share < 1.0):
            raise ValueError(f"gas share must be in [0, 1), got {self.gas_share:g}")
        tungsten.cavity_emissivity(0.5, self.open_fraction)  # range check

    @property
    def range_k(self) -> tuple[float, float]:
        return total_emissivity_table().range_k

    def total_emissivity(self, temperature_k):
        return (self.emissivity or total_emissivity_table())(temperature_k)

    def radiated_power(self, temperature_k):
        """P_rad in W: the coil's opening (f·A) radiating at the cavity emissivity."""
        eps = self.total_emissivity(temperature_k)
        eff = tungsten.cavity_emissivity(eps, self.open_fraction)
        return self.wire_area_m2 * self.open_fraction * eff * SIGMA * np.asarray(temperature_k, dtype=float) ** 4

    def gas_loss(self, temperature_k):
        """P_gas in W; its share of the input power is ``gas_share`` at 3000 K."""
        at_ref = self.gas_share / (1.0 - self.gas_share) * self.radiated_power(T_GAS_REF_K)
        return at_ref * (np.asarray(temperature_k, dtype=float) / T_GAS_REF_K) ** self.gas_exponent

    def resistance(self, temperature_k):
        """The filament's own hot resistance, in ohm."""
        return self.cold_resistance_ohm * resistance.table().ratio(temperature_k)

    def temperature(self, current_a: float) -> float:
        """The temperature at which I²·R(T) equals the losses."""
        if current_a <= 0:
            raise ValueError("current must be positive")
        lo, hi = self.range_k

        def excess(t):
            return float(current_a**2 * self.resistance(t) - self.radiated_power(t) - self.gas_loss(t))

        if excess(hi) > 0:
            raise ValueError(f"{current_a:g} A would heat the filament past {hi:g} K")
        if excess(lo) < 0:
            raise ValueError(f"{current_a:g} A leaves the filament below {lo:g} K")
        return float(brentq(excess, lo, hi, xtol=1e-9))

    def operating_point(self, current_a: float) -> dict:
        t = self.temperature(current_a)
        r = float(self.resistance(t))
        p_rad, p_gas = float(self.radiated_power(t)), float(self.gas_loss(t))
        return {"temperature_k": t, "voltage_v": current_a * r, "power_w": current_a**2 * r,
                "radiated_w": p_rad, "gas_loss_w": p_gas}

    def wire(self) -> dict:
        """Wire diameter and length implied by the cold resistance and the surface:
        R_cold = 4ρL/(πd²) and A = πdL, with ρ at 293 K (Desai). For a plausibility check
        of the two fitted numbers, not an input."""
        rho = float(resistance.table().resistivity(resistance.T_COLD_K)) * 1e-8
        d = (4.0 * rho * self.wire_area_m2 / (np.pi**2 * self.cold_resistance_ohm)) ** (1.0 / 3.0)
        return {"diameter_m": d, "length_m": self.wire_area_m2 / (np.pi * d)}


def solve_area(current_a: float, voltage_v: float, cold_resistance_ohm: float, open_fraction: float = 1.0,
               gas_share: float = 0.0, gas_exponent: float = 2.0, emissivity: Callable | None = None) -> float:
    """The wire surface that balances one measured operating point.

    The voltage over the current gives the hot resistance and so the temperature
    (``resistance.temperature``); the losses at that temperature must then equal I·V.
    That assumes the voltage was taken across the filament alone; lead and contact
    resistance in the reading put the surface a little too high (0.1 % per 13 mΩ here).
    """
    t = resistance.temperature(voltage_v / current_a, cold_resistance_ohm)
    unit = FilamentBalance(cold_resistance_ohm, 1.0, open_fraction, gas_share, gas_exponent, emissivity)
    return current_a * voltage_v / float(unit.radiated_power(t) + unit.gas_loss(t))


def irradiance_sensitivity(balance: FilamentBalance, current_a: float, wavelength_nm: float,
                           step_a: float = 1e-3) -> float:
    """Per cent change of the spectral radiance at ``wavelength_nm`` per mA of current,
    by central difference. The spectrum is Level 3's coil (De Vos, cavity formula
    wavelength by wavelength); the envelope's transmission does not depend on the
    current, so the irradiance changes by the same fraction."""
    def radiance(i):
        t = balance.temperature(i)
        cont = tungsten.TungstenCoilContinuum(t, open_fraction=balance.open_fraction)
        return float(cont.density(np.array([wavelength_nm]))[0])

    up, down = radiance(current_a + step_a), radiance(current_a - step_a)
    return 100.0 * (up - down) / (0.5 * (up + down)) / (2.0 * step_a * 1e3)
