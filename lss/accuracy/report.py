"""Accuracy report (design §11): score every model level against its reference.

Rules
-----
* **Like with like.** A model is compared with a reference only after the
  lamp's own components that the reference includes (its envelope) are applied
  on the quick path, and only if it then contains every one of them. A
  component with no quick-path model leaves the pair listed as not
  comparable rather than scored wrongly.
* **Only where the reference has data.** Both spectra are cut to the
  reference's valid range by an exact window, never extrapolated.
* **At the reference's resolution.** A table recorded with a bandpass (the CIE
  5 nm tables) spread each lamp line over its neighbouring nodes; the model is
  sampled through the same triangular bandpass before scoring, so a sharp line
  is not scored against a smeared one. Lamp sheets never see this step.
* **Equal luminous flux** normalisation (recorded in every row).
* **Golden file.** Scores are stored in ``lss/data/accuracy_golden.json``. Any
  error that grows beyond tolerance, a lost row, or a new empirical parameter
  fails the check; improvements are reported.
"""

import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from lss.accuracy import references
from lss.lamp import lamp as lamps
from lss.lamp import quickpath
from lss.spectral import colorimetry as cm
from lss.spectral.bandpass import as_recorded
from lss.spectral.filters import Transmission
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import BinnedSpectrum

GOLDEN_PATH = Path(str(resources.files("lss.data").joinpath("accuracy_golden.json")))
BANDS_NM = [(380.0, 480.0), (480.0, 580.0), (580.0, 680.0), (680.0, 780.0)]
NORMALISATION = "equal luminous flux"

#: Allowed growth of each error before the check fails: (absolute, relative).
TOLERANCE = {
    "spectral_nrms": (1e-5, 0.01),
    "delta_uv_prime": (1e-6, 0.01),
    "delta_cct_k": (0.05, 0.01),
    "delta_ra": (0.01, 0.01),
    "delta_rf": (0.01, 0.01),
}


@dataclass(frozen=True)
class Row:
    lamp_id: str
    level: int
    generator: str
    reference: str
    comparable: bool
    note: str = ""
    path: str = ""  # quick or instrument (design §5)
    normalisation: str = NORMALISATION
    range_nm: tuple[float, float] | None = None
    reference_sampling_nm: float | None = None
    reference_bandpass_nm: float | None = None  # the model was sampled through it
    spectral_nrms: float | None = None
    band_nrms: dict[str, float] | None = None
    delta_uv_prime: float | None = None
    delta_cct_k: float | None = None
    delta_ra: float | None = None
    delta_rf: float | None = None
    delta_flux: float | None = None  # needs electrical/absolute data; none in step 1
    delta_efficacy: float | None = None
    empirical_params: int = 0

    @property
    def key(self) -> str:
        return f"{self.lamp_id}@L{self.level}"


def _window(lo: float, hi: float) -> Transmission:
    return Transmission(np.array([lo, lo, hi, hi]), np.array([0.0, 1.0, 1.0, 0.0]), f"window {lo:g}-{hi:g}")


def _nrms(model: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((model - ref) ** 2)) / np.mean(ref))


@dataclass(frozen=True)
class Comparison:
    """A model and its reference on the analysis grid, cut to the scored range and
    normalised to equal luminous flux. The report and the fitter both use this, so a
    fit minimises exactly the error the report shows."""

    model: BinnedSpectrum
    reference: BinnedSpectrum
    range_nm: tuple[float, float]
    inside: np.ndarray  # bins strictly inside the range

    @property
    def residuals(self) -> np.ndarray:
        """Per-bin density error relative to the mean reference density; its RMS is the NRMS."""
        rd = self.reference.density[self.inside]
        return (self.model.density[self.inside] - rd) / np.mean(rd)


def model_profile(lamp, level: int, ref):
    """The level's emission with the lamp's own components that ``ref`` includes applied
    on the quick path, so it can be compared with the reference like with like."""
    return quickpath.apply(lamp, lamp.generate(level), ref.includes)


def missing_components(profile, ref) -> frozenset[str]:
    """Components the reference includes that the model does not (like with like)."""
    return ref.includes - profile.includes


def compare(profile, ref) -> Comparison:
    grid = analysis_grid()
    # Score only where the reference has data and the colour tables exist.
    lo, hi = max(ref.valid_range[0], grid.lo), min(ref.valid_range[1], grid.hi)
    window = _window(lo, hi)
    model = profile.spectrum
    if ref.bandpass_nm is not None:
        model = as_recorded(model, ref.nodes)
    m = model.filtered(window).bin(grid)
    r = ref.spectrum.filtered(window).bin(grid)
    k = cm.luminous(r) / cm.luminous(m)
    m = type(m)(m.grid, m.values * k, m.below * k, m.above * k, r.quantity)
    return Comparison(m, r, (lo, hi), (grid.centres > lo) & (grid.centres < hi))


def score(lamp, level: int, ref=None) -> Row:
    """Score one model level; ``ref`` defaults to the lamp's own reference."""
    ref = ref if ref is not None else references.get(lamp.reference)
    profile = model_profile(lamp, level, ref)
    base = dict(lamp_id=lamp.id, level=level, generator=profile.metadata.get("generator", ""), reference=ref.id,
                empirical_params=profile.empirical_count)

    missing = missing_components(profile, ref)
    if missing:
        return Row(**base, comparable=False,
                   note=f"reference includes {', '.join(sorted(missing))}; no quick-path model for it yet")

    c = compare(profile, ref)
    m, r = c.model, c.reference
    lo, hi = c.range_nm
    md, rd = m.density[c.inside], r.density[c.inside]
    centres = m.grid.centres[c.inside]
    bands = {}
    for b_lo, b_hi in BANDS_NM:
        sel = (centres > max(b_lo, lo)) & (centres < min(b_hi, hi))
        if sel.any():
            bands[f"{b_lo:g}-{b_hi:g}"] = _nrms(md[sel], rd[sel])

    cct_m, _ = cm.cct_duv(m)
    cct_r, _ = cm.cct_duv(r)
    cr_m, cr_r = cm.colour_rendering(m), cm.colour_rendering(r)
    return Row(
        **base,
        comparable=True,
        path="quick",
        range_nm=(lo, hi),
        reference_sampling_nm=ref.sampling_nm,
        reference_bandpass_nm=ref.bandpass_nm,
        spectral_nrms=_nrms(md, rd),
        band_nrms=bands,
        delta_uv_prime=float(np.hypot(*(cm.uv_prime(m) - cm.uv_prime(r)))),
        delta_cct_k=cct_m - cct_r,
        delta_ra=cr_m.ra - cr_r.ra,
        delta_rf=cr_m.rf - cr_r.rf,
    )


def run() -> list[Row]:
    rows = []
    for preset in lamps.preset_ids():
        lamp = lamps.load(preset)
        if lamp.reference is None:
            continue
        for level, spec in sorted(lamp.models.items()):
            if spec.generator == "reference":
                continue  # Level 0 is the reference itself
            rows.append(score(lamp, level))
    return rows


def format_text(rows: list[Row]) -> str:
    duv = "du'v'"
    head = f"{'lamp @ level':<28}{'reference':<12}{'NRMS':>9}{duv:>10}{'dCCT K':>9}{'dRa':>7}{'dRf':>7}{'emp':>5}"
    out = [f"Accuracy report (normalisation: {NORMALISATION}; path: quick)", head, "-" * len(head)]
    for r in rows:
        if not r.comparable:
            out.append(f"{r.key:<28}{r.reference:<12}  not comparable: {r.note}")
            continue
        out.append(f"{r.key:<28}{r.reference:<12}{r.spectral_nrms:>9.2e}{r.delta_uv_prime:>10.1e}"
                   f"{r.delta_cct_k:>+9.2f}{r.delta_ra:>+7.2f}{r.delta_rf:>+7.2f}{r.empirical_params:>5}")
        out.append(f"{'':<28}range {r.range_nm[0]:g}-{r.range_nm[1]:g} nm, reference sampled every "
                   f"{r.reference_sampling_nm:g} nm"
                   + (f" (model seen through its {r.reference_bandpass_nm:g} nm bandpass)"
                      if r.reference_bandpass_nm else "")
                   + "; band NRMS "
                   + ", ".join(f"{b}: {v:.1e}" for b, v in r.band_nrms.items()))
    if not rows:
        out.append("(no model levels above Level 0 yet)")
    return "\n".join(out)


def _errors(row: dict) -> dict[str, float]:
    return {k: abs(row[k]) for k in TOLERANCE if row.get(k) is not None}


def write_golden(rows: list[Row], path: Path = GOLDEN_PATH) -> None:
    data = {r.key: asdict(r) for r in rows}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_golden(rows: list[Row], path: Path = GOLDEN_PATH) -> list[str]:
    """Regressions against the golden file; an empty list means the check passes."""
    golden = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current = {r.key: asdict(r) for r in rows}
    failures = []
    for key in sorted(set(golden) - set(current)):
        failures.append(f"{key}: row disappeared")
    for key, row in sorted(current.items()):
        if key not in golden:
            failures.append(f"{key}: new row, not in the golden file (review it, then --update-golden)")
            continue
        old = golden[key]
        if row["comparable"] != old["comparable"]:
            failures.append(f"{key}: comparable changed {old['comparable']} -> {row['comparable']}")
        if row["empirical_params"] > old["empirical_params"]:
            failures.append(f"{key}: empirical parameters {old['empirical_params']} -> {row['empirical_params']}")
        now, before = _errors(row), _errors(old)
        for name, (atol, rtol) in TOLERANCE.items():
            if name in now and name in before and now[name] > before[name] * (1 + rtol) + atol:
                failures.append(f"{key}: {name} grew {before[name]:.3g} -> {now[name]:.3g}")
    return failures
