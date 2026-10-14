"""Reference spectra library with provenance (design §11).

Entries live in ``lss/data/references.yaml``. Each loads into a
:class:`Reference`: a relative or absolute spectrum, the wavelength range
the data actually covers, its sampling, what it already includes, and where
it came from.

Two kinds of source:

* ``colour-science``: a dataset shipped with that package (the CIE tables), evenly
  spaced; the spectrum is the table, linear between nodes.
* ``table``: a data file in ``lss/data`` with one column per lamp, at wavelengths
  that may be uneven (a calibration report). Spectral errors are then scored at the
  table's own wavelengths (``scored_at_nodes``), and the continuous spectrum, which
  colour needs, is the table interpolated through its ratio to a blackbody
  (``lss.accuracy.interpolation``).
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
    sampling_nm: float | None  # None: uneven, see ``nodes``
    kind: str  # representative-table | definition | measurement
    uncertainty: str
    includes: frozenset[str]
    citation: str
    data_version: str
    #: FWHM of the triangular bandpass the table was recorded with (nm), or None for a
    #: spectrum no instrument blurred (a definition). Models are compared with the
    #: table only after the same sampling (lss.spectral.bandpass).
    bandpass_nm: float | None = None
    #: The table as published, for a reference scored at its own wavelengths (an uneven
    #: calibration table); None for an even table, whose spectrum is the table itself.
    node_values: np.ndarray | None = None
    node_wavelengths: np.ndarray | None = None
    interpolation: str | None = None  # how the spectrum fills the gaps between nodes
    #: Relative standard uncertainty (k = 1) at each node, where the source states one.
    node_sigma: np.ndarray | None = None

    @property
    def scored_at_nodes(self) -> bool:
        return self.node_values is not None

    @property
    def nodes(self) -> np.ndarray:
        """The table's wavelengths."""
        if self.node_wavelengths is not None:
            return self.node_wavelengths
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


@lru_cache(maxsize=None)
def data_file(name: str) -> dict:
    return yaml.safe_load(resources.files("lss.data").joinpath(name).read_text(encoding="utf-8"))


def _table(src: dict):
    """(wavelengths, values in W m-2 nm-1, version, relative k = 1 uncertainty or None) of one
    lamp's column in a data file."""
    doc = data_file(src["file"])
    if doc["unit"] != "W cm-3":
        raise ValueError(f"{src['file']}: unsupported unit {doc['unit']!r}")
    col = doc["columns"].index(src["column"])
    rows = sorted((float(k), v[col]) for k, v in doc["irradiance"].items())
    lam = np.array([k for k, _ in rows])
    val = np.array([v for _, v in rows], dtype=float) * 1e-3  # W cm-2 per cm = 1e-3 W m-2 nm-1
    sigma = None
    if "uncertainty_k2_percent" in doc:  # stated at a few wavelengths: linear between them
        u = sorted((float(k), float(v)) for k, v in doc["uncertainty_k2_percent"].items())
        sigma = np.interp(lam, [k for k, _ in u], [v for _, v in u]) / 200.0
    return lam, val, f"{src['file']}, column {src['column']}", sigma


def _load_entry(entry: dict, defaults: dict) -> Reference:
    e = {**defaults, **entry}
    src = e["source"]
    includes = frozenset(e.get("includes", []))
    if includes - COMPONENTS:
        raise ValueError(f"{e['id']}: unknown includes {sorted(includes - COMPONENTS)}")
    if src["type"] == "table":
        return _load_table_entry(e, src, includes)
    if src["type"] != "colour-science":
        raise ValueError(f"unknown reference source type {src['type']!r} in {e['id']}")
    lam, val, version = _colour_dataset(src["key"])
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


def _load_table_entry(e: dict, src: dict, includes: frozenset) -> Reference:
    from lss.accuracy import interpolation

    lam, val, version, sigma = _table(src)
    if e.get("sampling_nm") is not None or e.get("bandpass_nm") is not None:
        raise ValueError(f"{e['id']}: an uneven table has no single sampling or bandpass")
    method = e.get("interpolation", "pchip")
    dense = np.union1d(np.arange(np.ceil(lam[0]), np.floor(lam[-1]) + 1.0), lam)
    cont = TabulatedContinuum(dense, interpolation.interpolate(lam, val, dense, method))
    return Reference(
        id=e["id"],
        family=e["family"],
        description=e["description"],
        spectrum=Spectrum.of(cont, quantity="W m-2 nm-1", name=e["id"]),
        valid_range=(float(lam[0]), float(lam[-1])),
        sampling_nm=None,
        kind=e["kind"],
        uncertainty=str(e["uncertainty"]),
        includes=includes,
        citation=e["citation"],
        data_version=f"{version}; between nodes: {method} interpolation of the ratio to a blackbody",
        bandpass_nm=None,
        node_values=val,
        node_wavelengths=lam,
        interpolation=method,
        node_sigma=sigma,
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
