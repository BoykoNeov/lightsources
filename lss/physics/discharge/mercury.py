"""Mercury lines of a low-pressure discharge (design §4.3 L1).

Line positions and relative powers come from a measured Hg-Ar discharge
(``lss/data/hg_lines.yaml``). Each line gets its Doppler width at the gas
temperature; isotope and hyperfine structure (~0.02 nm) and the Doppler core
(~0.001 nm) are both far below any reference table's resolution.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from lss.spectral.lines import Line
from lss.spectral.profiles import Gaussian
from lss.spectral.units import C, K_B

#: Mean atomic mass of natural mercury (IUPAC 2021 standard atomic weight), kg.
MASS_KG = 200.592 * 1.66053906660e-27

#: Level 1 emits the visible lines only. The UV lines (253.7 nm resonance and the
#: 297-366 nm group) fall on the phosphor coat, which absorbs and converts most of
#: them: that is level 2's UV budget, not a fixed line ratio.
VISIBLE_FROM_NM = 400.0


@dataclass(frozen=True)
class LineTable:
    wavelengths_nm: np.ndarray
    relative_power: np.ndarray
    medium: str
    source: str


@lru_cache(maxsize=1)
def table() -> LineTable:
    doc = yaml.safe_load(resources.files("lss.data").joinpath("hg_lines.yaml").read_text(encoding="utf-8"))
    rows = np.asarray(doc["lines"], dtype=float)
    return LineTable(rows[:, 0], rows[:, 1], doc["medium"], doc["source"])


def doppler_fwhm(wavelength_nm: float, temperature_k: float) -> float:
    """Doppler full width at half maximum of a line, in nm."""
    return wavelength_nm * np.sqrt(8.0 * K_B * temperature_k * np.log(2.0) / (MASS_KG * C**2))


def visible_lines(total_power: float, gas_temperature_k: float) -> list[Line]:
    """The visible lines, sharing ``total_power`` in their measured ratios."""
    t = table()
    keep = t.wavelengths_nm >= VISIBLE_FROM_NM
    lam, rel = t.wavelengths_nm[keep], t.relative_power[keep]
    powers = total_power * rel / rel.sum()
    return [Line(float(w), float(p), Gaussian(doppler_fwhm(w, gas_temperature_k)), t.medium, f"Hg {w:.3f}")
            for w, p in zip(lam, powers)]
