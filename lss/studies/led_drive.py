"""Drive study: a measured warm-white LED at several currents, against LED Level 3 (design §4.2 L3).

The data are the supplement of L. Wu, Y. Li, H. Chen, J. Wang, X. Wang, Y. Xue, *Sci.
Rep.* 16, 7751 (2026) (doi 10.1038/s41598-026-39015-3): two phosphor-converted LEDs
the authors made (a blue die with the commercial phosphors "LCG525F2", green, and
"LCN650", red), each measured three times in quick succession at 150–500 mA in steps
of 50 mA, at 25 °C ambient, in an integrating sphere. Each record holds the spectrum
(1 nm, normalised to its peak) and the current, forward voltage and radiant power.

The supplement is licensed CC BY-NC-ND 4.0, so it is not part of this repository:
:func:`fetch` downloads it from the publisher into a local cache and checks its
SHA-256, and nothing derived from it is written back. Tests that need it are skipped
when it is absent.

What the data can test (design §4.2 L3): the instrument zeroes weak signal, and more
of it at low current (at 150 mA the spectra end near 427 and 730 nm, at 500 mA near
413 and 752 nm), so the edges, and any colour computed from them, depend on the
current and on the window; colour is computed over :data:`WINDOW_NM` only. The
phosphors could not be identified, and Level 2 fits this LED only with its shared
constants at their bounds, so the study compares *changes*, band by band
(:data:`BAND_EDGES_NM`, each band's share of :data:`NORM_NM`), within each triplicate:
the LED warming at a fixed current, where the edge effect is small.
"""

import io
import re
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from lss import download
from lss.spectral import colorimetry as cm
from lss.spectral.continuum import TabulatedContinuum
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import Spectrum

URL = ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-026-39015-3/MediaObjects/"
       "41598_2026_39015_MOESM1_ESM.zip")
SHA256 = "abf33d0390636aaa1017b97d36f9f462b8785f88dd491a12ba8db49cba4f2363"
CITATION = ("L. Wu, Y. Li, H. Chen, J. Wang, X. Wang, Y. Xue, Sci. Rep. 16, 7751 (2026), "
            "doi 10.1038/s41598-026-39015-3, Supplementary Information 1 (CC BY-NC-ND 4.0)")
#: Record numbers of each sample: 8 currents × 3 readings, in measurement order.
SAMPLES = {1: range(875, 899), 2: range(899, 923)}
#: The range the clipped spectra can be trusted over (nm).
WINDOW_NM = (430.0, 715.0)


def data_path() -> Path:
    return download.cache_dir() / "wu2026_sci_rep_supplement_1.zip"


def available() -> bool:
    return data_path().exists()


def fetch() -> Path:
    """The supplement in the cache, downloaded first if absent; its SHA-256 is checked either way."""
    return download.fetch(URL, data_path(), SHA256)


@dataclass(frozen=True)
class Reading:
    record: int
    current_a: float
    voltage_v: float
    radiant_power_w: float
    wavelengths_nm: np.ndarray
    power: np.ndarray  # relative, peak 100

    @property
    def spectrum(self) -> Spectrum:
        return Spectrum.of(TabulatedContinuum(self.wavelengths_nm, self.power), name=f"record {self.record}")


def _number(text) -> float:
    return float(re.search(r"[-\d.]+", str(text)).group())


def _parse(record: int, data: bytes) -> Reading:
    import openpyxl  # optional dependency: pip install "lss[data]"

    sheet = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True).worksheets[0]
    rows = [list(r) for r in sheet.iter_rows(values_only=True)]
    header = {}
    for r in rows[:19]:
        for i in range(len(r) - 1):
            if isinstance(r[i], str) and r[i].endswith(":"):
                header[r[i]] = r[i + 1]
    start = next(i for i, r in enumerate(rows) if r[0] == "波长") + 1  # "wavelength"
    table = np.array([(float(r[0]), float(r[1])) for r in rows[start:] if r[0] is not None])
    return Reading(record,
                   current_a=_number(header["正向电流:"]) / 1e3,  # forward current, mA
                   voltage_v=_number(header["正向压降:"]),  # forward voltage, V
                   radiant_power_w=_number(header["光功率:"]) / 1e3,  # radiant power, mW
                   wavelengths_nm=table[:, 0], power=table[:, 1])


@lru_cache(maxsize=1)
def _records() -> dict[int, Reading]:
    wanted = {n for numbers in SAMPLES.values() for n in numbers}
    out = {}
    with zipfile.ZipFile(fetch()) as z:
        for info in z.infolist():
            m = re.search(r"/(\d+)\.xlsx$", info.filename)
            if m and int(m.group(1)) in wanted:
                out[int(m.group(1))] = _parse(int(m.group(1)), z.read(info))
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"{data_path()}: records {sorted(missing)} not found")
    return out


def readings(sample: int) -> list[tuple[Reading, ...]]:
    """One triplicate per current, readings in the order they were taken."""
    recs = _records()
    numbers = list(SAMPLES[sample])
    return [tuple(recs[n] for n in numbers[i:i + 3]) for i in range(0, len(numbers), 3)]


def window_uv(binned) -> np.ndarray:
    """u′v′ of a spectrum binned on the analysis grid, counting only :data:`WINDOW_NM`."""
    lo, hi = WINDOW_NM
    c = binned.grid.centres
    return cm.uv_prime(replace(binned, values=np.where((c > lo) & (c < hi), binned.values, 0.0)))


def reading_uv(r: Reading) -> np.ndarray:
    return window_uv(r.spectrum.bin(analysis_grid()))


@dataclass(frozen=True)
class CurrentRow:
    """One current: the triplicate's mean, and its drift from the first reading to the third."""

    current_a: float
    voltage_v: float
    radiant_power_w: float
    uv: np.ndarray
    drift_uv: np.ndarray
    drift_power: float  # fractional change of the radiant power
    drift_voltage_v: float

    @property
    def power_per_amp(self) -> float:
        return self.radiant_power_w / self.current_a


def measured(sample: int) -> list[CurrentRow]:
    rows = []
    for trip in readings(sample):
        uvs = np.array([reading_uv(r) for r in trip])
        p = np.array([r.radiant_power_w for r in trip])
        v = np.array([r.voltage_v for r in trip])
        rows.append(CurrentRow(trip[0].current_a, float(v.mean()), float(p.mean()), uvs.mean(axis=0),
                               uvs[-1] - uvs[0], float((p[-1] - p[0]) / p.mean()), float(v[-1] - v[0])))
    return rows


#: 10 nm bands where every current's spectrum is well above the instrument's floor.
BAND_EDGES_NM = np.arange(450.0, 691.0, 10.0)
#: Each band is taken as its share of the light over this range.
NORM_NM = (450.0, 680.0)
#: Groups of bands the study reports.
GROUPS_NM = {"blue": (450.0, 490.0), "green": (500.0, 570.0), "red": (600.0, 680.0)}
#: The drift is reported per this voltage drop: about 4-9 K of junction warming for 5-6 dies in
#: series at 1.6-3.0 mV/K each (the range of most blue GaInN entries tabulated by Cengiz,
#: Azarifar, Arik, Micromachines 13, 1615 (2022)); the paper gives neither number.
DRIFT_VOLTAGE_V = 0.07
#: The steepest quenching the cited 150 °C points allow: near the smallest barrier for which
#: one-barrier curves through them exist (about 0.009 eV for the garnet), its slope at room
#: temperature is within 2 % of the chord between 25 and 150 °C.
STEEPEST_BARRIER_EV = 0.012


def band_shares(wavelengths_nm: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Each :data:`BAND_EDGES_NM` band's share of the light over :data:`NORM_NM`, from a 1 nm table."""
    w, p = wavelengths_nm, power
    total = p[(w >= NORM_NM[0]) & (w <= NORM_NM[1])].sum()
    return np.array([p[(w >= a) & (w < b)].sum() / total for a, b in zip(BAND_EDGES_NM[:-1], BAND_EDGES_NM[1:])])


def group_means(change: np.ndarray) -> dict[str, float]:
    lo = BAND_EDGES_NM[:-1]
    return {name: float(change[(lo >= a) & (lo < b)].mean()) for name, (a, b) in GROUPS_NM.items()}


def measured_drift(sample: int, first: int = 3) -> np.ndarray:
    """Fractional change of each band over a triplicate, per :data:`DRIFT_VOLTAGE_V` of voltage
    drop, averaged over the currents from index ``first`` (300-500 mA by default: the larger drifts)."""
    out = []
    for trip in readings(sample)[first:]:
        a, b = trip[0], trip[-1]
        change = band_shares(b.wavelengths_nm, b.power) / band_shares(a.wavelengths_nm, a.power) - 1.0
        out.append(change * DRIFT_VOLTAGE_V / (a.voltage_v - b.voltage_v))
    return np.mean(out, axis=0)


def model_change(preset: str, warming_k: float, overrides: dict[str, float] | None = None) -> np.ndarray:
    """Fractional change of each band when Level 3 runs the junction ``warming_k`` above Level 2's
    anchor; ``overrides`` set shared constants (e.g. a quenching barrier, or a die shift of 0)."""
    from lss.accuracy import fit
    from lss.lamp import lamp as lamps
    from lss.lamp.lamp import ModelSpec
    from lss.params import Parameter

    lamp = lamps.load(preset)
    anchor = lamp.physical["phosphor_temperature"].value
    physical = dict(lamp.physical)
    # Any operating point that puts the junction warming_k above the anchor: 1 W of heat.
    for name, value, unit in (("drive_current", 1.0, "A"), ("forward_voltage", 1.0, "V"),
                              ("radiant_power", 0.0, "W"), ("board_temperature", anchor, "K"),
                              ("thermal_resistance", warming_k, "K/W")):
        physical[name] = Parameter(value, unit, empirical=True, reason="study operating point")
    lamp = replace(lamp, physical=physical, models={**lamp.models, 3: ModelSpec(3, "led_thermal")})
    if overrides:
        lamp = fit.with_values(lamp, 3, overrides)
    grid = analysis_grid()
    hot, cold = (lamp.generate(level).spectrum.bin(grid).values for level in (3, 2))
    return band_shares(grid.centres, hot) / band_shares(grid.centres, cold) - 1.0


def die_warming(sample: int, preset: str) -> float:
    """The warming (K) the measured drift implies through the die's cited peak shift: the drift
    fitted by least squares as a sum of Level 3's die-only and phosphor-only changes per kelvin
    (the phosphors quenching as steeply as the cited points allow)."""
    steep = {"garnet.quench_activation_energy": STEEPEST_BARRIER_EV,
             "nitride.quench_activation_energy": STEEPEST_BARRIER_EV}
    die = model_change(preset, 1.0, {**steep, "garnet.quench_retained_423k": 1.0, "nitride.quench_retained_423k": 1.0})
    phosphors_ = model_change(preset, 1.0, {**steep, "die.peak_shift": 0.0, "die.fwhm_shift": 0.0})
    coef, *_ = np.linalg.lstsq(np.vstack([die, phosphors_]).T, measured_drift(sample), rcond=None)
    return float(coef[0])


def format_text(presets=("led_cie_b1", "led_cie_b2", "led_cie_b3")) -> str:
    lines = [f"Data: {CITATION}",
             f"colour over {WINDOW_NM[0]:g}-{WINDOW_NM[1]:g} nm only: the instrument zeroes weak signal, more of it "
             "at low current, so the edges, and the colour, depend on the window", ""]
    for sample in SAMPLES:
        rows = measured(sample)
        base = rows[0]
        lines.append(f"sample {sample}: current, voltage, radiant power per amp, u'v' change from "
                     f"{base.current_a * 1e3:.0f} mA (1e-3); drift over the triplicate: voltage, power, u'v' (1e-4)")
        for r in rows:
            d = (r.uv - base.uv) * 1e3
            lines.append(f"  {r.current_a * 1e3:4.0f} mA  {r.voltage_v:6.2f} V  {r.power_per_amp:5.3f} W/A  "
                         f"du' {d[0]:+5.2f} dv' {d[1]:+5.2f}  |  drift {r.drift_voltage_v:+.3f} V "
                         f"{r.drift_power * 100:+5.2f} %  du' {r.drift_uv[0] * 1e4:+5.2f} dv' {r.drift_uv[1] * 1e4:+5.2f}")
        lines.append("")

    def groups(change):
        return "  ".join(f"{k} {v * 100:+5.2f}" for k, v in group_means(change).items())

    lo, hi = NORM_NM
    lines.append(f"Warming at a fixed current (300-500 mA), each band group's change of its share of "
                 f"{lo:g}-{hi:g} nm (%), per {DRIFT_VOLTAGE_V:g} V of voltage drop (about 4-9 K):")
    warm = {}
    for sample in SAMPLES:
        lines.append(f"  measured, sample {sample}:  {groups(measured_drift(sample))}")
        warm[sample] = [die_warming(sample, p) for p in presets]
    lines.append("  the warming the blue's change implies through the die's cited peak shift (K), with the "
                 f"phosphors of {', '.join(presets)}:")
    for sample, ks in warm.items():
        lines.append(f"    sample {sample}: " + ", ".join(f"{k:.1f}" for k in ks))
    for sample in SAMPLES:
        k = float(np.mean(warm[sample]))
        for label, barrier in (("assumed barrier", None), ("steepest quenching the cited points allow",
                                                           STEEPEST_BARRIER_EV)):
            over = None if barrier is None else {"garnet.quench_activation_energy": barrier,
                                                 "nitride.quench_activation_energy": barrier}
            for p in presets:
                lines.append(f"  Level 3, {p}, {k:.1f} K ({label}):  {groups(model_change(p, k, over))}")
    return "\n".join(lines)
