"""Filament temperature from its resistance (design §4.1 L4).

A tungsten wire's resistance rises with temperature as its resistivity does, so the
ratio of the hot resistance (voltage over current) to the cold one gives the
temperature: ρ(T)/ρ(293 K) = R_hot/R_cold. The resistivity is Desai et al.'s
recommended table (``lss/data/tungsten_resistivity.yaml``, which holds the
provenance and limits), uncorrected for thermal expansion because a wire's
resistance ratio is too. Between tabulated points it is interpolated monotonically
(PCHIP), so the temperature for a given ratio is unique and its slope smooth.

The temperature found this way is an average along the wire weighted by its
resistance, cooler supports and ends included. A spectrum is dominated by the
hottest parts, so the two need not agree on a filament that is not one temperature
(Level 5).
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from importlib import resources

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

#: The reference temperature of a cold resistance.
T_COLD_K = 293.0


@dataclass(frozen=True)
class ResistivityTable:
    temperatures_k: np.ndarray
    values: np.ndarray  # 1e-8 ohm m
    citation: str
    uncertainty_rel: float
    data_version: str

    @property
    def range_k(self) -> tuple[float, float]:
        return float(self.temperatures_k[0]), float(self.temperatures_k[-1])

    @cached_property
    def _interp(self) -> PchipInterpolator:
        return PchipInterpolator(self.temperatures_k, self.values)

    def resistivity(self, temperature_k):
        """ρ(T) in 1e-8 ohm m, inside the tabulated range only."""
        t = np.asarray(temperature_k, dtype=float)
        lo, hi = self.range_k
        if np.any((t < lo) | (t > hi)):
            raise ValueError(f"tungsten resistivity is tabulated for {lo:g}-{hi:g} K")
        return self._interp(t)

    def ratio(self, temperature_k):
        """R(T)/R(293 K) of a tungsten wire."""
        return self.resistivity(temperature_k) / self.resistivity(T_COLD_K)

    def log_slope(self, temperature_k: float) -> float:
        """d ln ρ / d ln T: how many per cent the resistance rises per per cent of temperature."""
        return float(self._interp(temperature_k, 1) * temperature_k / self._interp(temperature_k))


@lru_cache(maxsize=1)
def table() -> ResistivityTable:
    text = resources.files("lss.data").joinpath("tungsten_resistivity.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    rows = sorted((float(k), float(v)) for k, v in doc["table"].items())
    return ResistivityTable(
        temperatures_k=np.array([t for t, _ in rows]),
        values=np.array([v for _, v in rows]),
        citation=doc["citation"],
        uncertainty_rel=float(doc["uncertainty_rel"]),
        data_version=doc["data_version"],
    )


def temperature(hot_resistance_ohm: float, cold_resistance_ohm: float) -> float:
    """The filament temperature at which R(T)/R(293 K) equals ``hot / cold``."""
    if hot_resistance_ohm <= 0 or cold_resistance_ohm <= 0:
        raise ValueError("resistances must be positive")
    tab = table()
    q = hot_resistance_ohm / cold_resistance_ohm
    lo, hi = tab.range_k
    q_hi = float(tab.ratio(hi))
    if not (1.0 <= q <= q_hi):
        raise ValueError(f"resistance ratio {q:.4g} is outside tungsten's 1-{q_hi:.4g} ({T_COLD_K:g}-{hi:g} K)")
    return float(brentq(lambda t: float(tab.ratio(t)) - q, T_COLD_K, hi, xtol=1e-9))
