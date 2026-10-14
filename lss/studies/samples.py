"""Reflectance sample sets on the analysis grid (design §7, no-render studies).

Entries live in ``lss/data/reflectances.yaml``. A table is read as piecewise
linear between its nodes and held at its end values beyond them, then taken at
the analysis-grid bin centres: the colour-matching functions are applied the
same way (``lss.spectral.colorimetry``), so a reflectance and a CMF meet on the
same 1 nm points.
"""

import warnings
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from lss.spectral.grids import analysis_grid


@dataclass(frozen=True)
class SampleSet:
    id: str
    description: str
    labels: tuple[str, ...]
    values: np.ndarray  # reflectance, (analysis-grid bins, samples)
    measured_range: tuple[float, float]  # nm; outside it the end values are held
    sampling_nm: float
    kind: str  # measurement | standard-set
    uncertainty: str
    citation: str
    data_version: str

    def __len__(self) -> int:
        return len(self.labels)

    def index(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(f"{self.id} has no sample {label!r}; samples: {', '.join(self.labels)}") from None


def _colour_dataset(dataset: str, key: str):
    """(wavelengths, values (nodes, samples), labels, version) from colour-science."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour
        from colour.quality.cfi2017 import load_TCS_CIE2017

    if dataset == "colourchecker":
        sds = colour.SDS_COLOURCHECKERS[key]
    elif dataset == "tcs-cie1995":
        sds = colour.quality.datasets.SDS_TCS[key]
    elif dataset == "tcs-cie2017":
        msds = load_TCS_CIE2017(colour.SpectralShape(380.0, 780.0, float(key)))
        lam = np.asarray(msds.wavelengths, float)
        labels = tuple(f"CES{i + 1:02d}" for i in range(msds.values.shape[1]))
        return lam, np.asarray(msds.values, float), labels, f"colour-science {colour.__version__}"
    else:
        raise ValueError(f"unknown colour-science dataset {dataset!r}")
    labels = tuple(sds)
    first = sds[labels[0]]
    lam = np.asarray(first.wavelengths, float)
    if any(not np.array_equal(sds[k].wavelengths, lam) for k in labels):
        raise ValueError(f"{dataset} {key}: samples on different wavelengths")
    values = np.stack([np.asarray(sds[k].values, float) for k in labels], axis=1)
    return lam, values, labels, f"colour-science {colour.__version__}"


def on_analysis_grid(lam: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Piecewise-linear reading at the bin centres, end values held beyond the table."""
    centres = analysis_grid().centres
    return np.stack([np.interp(centres, lam, col) for col in np.atleast_2d(values.T)], axis=1)


def _load_entry(e: dict) -> SampleSet:
    src = e["source"]
    if src["type"] != "colour-science":
        raise ValueError(f"unknown sample source type {src['type']!r} in {e['id']}")
    lam, values, labels, version = _colour_dataset(src["dataset"], src["key"])
    steps = np.diff(lam)
    if not np.allclose(steps, steps[0]):
        raise ValueError(f"{e['id']}: uneven table spacing")
    grid_values = on_analysis_grid(lam, values)
    grid_values.setflags(write=False)
    return SampleSet(
        id=e["id"],
        description=e["description"],
        labels=labels,
        values=grid_values,
        measured_range=(float(lam[0]), float(lam[-1])),
        sampling_nm=float(steps[0]),
        kind=e["kind"],
        uncertainty=str(e["uncertainty"]),
        citation=e["citation"],
        data_version=f"{version}, dataset {src['dataset']} {src['key']}",
    )


@lru_cache(maxsize=1)
def library() -> dict[str, SampleSet]:
    text = resources.files("lss.data").joinpath("reflectances.yaml").read_text(encoding="utf-8")
    sets = [_load_entry(e) for e in yaml.safe_load(text)["sets"]]
    return {s.id: s for s in sets}


def get(set_id: str) -> SampleSet:
    try:
        return library()[set_id]
    except KeyError:
        raise KeyError(f"no sample set {set_id!r}; sets: {', '.join(library())}") from None
