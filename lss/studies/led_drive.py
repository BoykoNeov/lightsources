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

What the data can test (design §4.2 L3): the spectra are zero below 421 nm and fall
to zero within 35 nm above 700 nm, faster than any phosphor's tail, so the instrument
clipped them; colour is compared over :data:`WINDOW_NM` only, for the model too. The
phosphors could not be identified, and Level 2 fits this LED only with its shared
constants at their bounds, so the study compares *changes*: within each triplicate
(the LED warming at a fixed current) and across currents.
"""

import hashlib
import io
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

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


def cache_dir() -> Path:
    """``$LSS_CACHE``, or ``~/.cache/lss``."""
    return Path(os.environ.get("LSS_CACHE") or Path.home() / ".cache" / "lss")


def data_path() -> Path:
    return cache_dir() / "wu2026_sci_rep_supplement_1.zip"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available() -> bool:
    return data_path().exists()


def fetch() -> Path:
    """The supplement in the cache, downloaded first if absent; its SHA-256 is checked either way."""
    path = data_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "lss (light source simulator)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    digest = _sha256(path)
    if digest != SHA256:
        raise ValueError(f"{path}: SHA-256 {digest} is not the published supplement's ({SHA256}); delete it to refetch")
    return path


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


def format_text() -> str:
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
    return "\n".join(lines)
