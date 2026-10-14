"""Reference spectra library with provenance (design §11).

Entries live in ``lss/data/references.yaml``. Each loads into a
:class:`Reference`: a relative or absolute spectrum, the wavelength range
the data actually covers, its sampling, what it already includes, and where
it came from.
"""

import warnings
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from lss.lamp.profile import COMPONENTS
from lss.spectral.continuum import TabulatedContinuum
from lss.spectral.spectrum import Spectrum


@dataclass(frozen=True)
class Reference:
    id: str
    family: str
    description: str
    spectrum: Spectrum
    valid_range: tuple[float, float]  # nm; never score or extrapolate outside it
    sampling_nm: float
    kind: str  # representative-table | definition | measurement
    uncertainty: str
    includes: frozenset[str]
    citation: str
    data_version: str
    #: FWHM of the triangular bandpass the table was recorded with (nm), or None for a
    #: spectrum no instrument blurred (a definition). Models are compared with the
    #: table only after the same sampling (lss.spectral.bandpass).
    bandpass_nm: float | None = None

    @property
    def nodes(self) -> np.ndarray:
        """The table's wavelengths."""
        lo, hi = self.valid_range
        return lo + self.sampling_nm * np.arange(int(round((hi - lo) / self.sampling_nm)) + 1)


def _colour_dataset(key: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour

    sd = colour.SDS_ILLUMINANTS[key]
    return np.asarray(sd.wavelengths, float), np.asarray(sd.values, float), f"colour-science {colour.__version__}"


def _bandpass(e: dict, lam: np.ndarray) -> float | None:
    bp = e.get("bandpass_nm")
    if bp is None:
        return None
    bp = float(bp)
    steps = np.diff(lam)
    if not np.allclose(steps, e["sampling_nm"]) or bp != float(e["sampling_nm"]):
        raise ValueError(f"{e['id']}: a bandpass is supported only with FWHM equal to an even table step")
    return bp


def _load_entry(entry: dict, defaults: dict) -> Reference:
    e = {**defaults, **entry}
    src = e["source"]
    if src["type"] != "colour-science":
        raise ValueError(f"unknown reference source type {src['type']!r} in {e['id']}")
    lam, val, version = _colour_dataset(src["key"])
    includes = frozenset(e.get("includes", []))
    if includes - COMPONENTS:
        raise ValueError(f"{e['id']}: unknown includes {sorted(includes - COMPONENTS)}")
    spectrum = Spectrum.of(TabulatedContinuum(lam, val), quantity="relative", name=e["id"])
    return Reference(
        id=e["id"],
        family=e["family"],
        description=e["description"],
        spectrum=spectrum,
        valid_range=(float(lam[0]), float(lam[-1])),
        sampling_nm=float(e["sampling_nm"]),
        kind=e["kind"],
        uncertainty=str(e["uncertainty"]),
        includes=includes,
        citation=e["citation"],
        data_version=f"{version}, dataset {src['key']}",
        bandpass_nm=_bandpass(e, lam),
    )


@lru_cache(maxsize=1)
def library() -> dict[str, Reference]:
    text = resources.files("lss.data").joinpath("references.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    defaults = doc.get("defaults", {})
    refs = [_load_entry(e, defaults) for e in doc["references"]]
    ids = [r.id for r in refs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate reference ids")
    return {r.id: r for r in refs}


def get(ref_id: str) -> Reference:
    try:
        return library()[ref_id]
    except KeyError:
        raise KeyError(f"no reference {ref_id!r}; see `lss refs`") from None
