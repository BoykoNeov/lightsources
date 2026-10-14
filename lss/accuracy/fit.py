"""Parameter fitting (design §11, ``lss fit``).

Fits a model level's free parameters to a reference with
``scipy.optimize.least_squares``, inside each parameter's physically plausible
bounds.

Rules
-----
* **Free parameters** are those the level actually reads (from the generator's
  ``metadata["params"]``) that carry ``bounds`` in the preset. A parameter
  without bounds is held fixed; bounds are part of the physical description,
  never invented here.
* **Same yardstick as the report.** The ``spectral`` objective's residuals are
  exactly the report's: per-bin density error after the equal-luminous-flux
  normalisation, over the reference's range, so the fit minimises the reported
  NRMS. The ``colour`` objective matches chromaticity (Δu′v′) instead.
* **Like with like.** A model that lacks a component the reference includes is
  not fitted at all.
* **A value at a bound is a finding.** It means the model only matches with an
  implausible parameter: missing physics, and a pointer to the next rung.
* The tool never edits preset files. It prints a YAML snippet with the fitted
  value tagged ``empirical`` (a calibration, not a mechanism) and its provenance.
"""

import textwrap
from dataclasses import asdict, dataclass, field, replace
from datetime import date

import numpy as np
from scipy.optimize import least_squares

from lss.accuracy import references
from lss.accuracy import report
from lss.lamp.lamp import Lamp
from lss.spectral import colorimetry as cm

OBJECTIVES = ("spectral", "colour")

#: A fitted value this close to a bound (as a fraction of the bound range) is flagged.
AT_BOUND = 0.01


@dataclass(frozen=True)
class FittedParameter:
    name: str
    unit: str
    start: float
    value: float
    bounds: tuple[float, float]
    read_from: str  # "model level N" or "physical description"

    @property
    def position(self) -> float:
        """Where the value sits in its bounds: 0 at the lower, 1 at the upper."""
        lo, hi = self.bounds
        return (self.value - lo) / (hi - lo)

    @property
    def at_bound(self) -> bool:
        return self.position < AT_BOUND or self.position > 1.0 - AT_BOUND


@dataclass(frozen=True)
class FitResult:
    lamp_id: str
    level: int
    reference: str
    objective: str
    parameters: tuple[FittedParameter, ...]
    before: report.Row
    after: report.Row
    success: bool
    message: str
    evaluations: int
    lamp: Lamp = field(compare=False, repr=False)  # the lamp with the fitted values

    @property
    def at_bound(self) -> list[str]:
        return [p.name for p in self.parameters if p.at_bound]

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("lamp", "parameters")}
        d["parameters"] = [{**asdict(p), "position": p.position, "at_bound": p.at_bound} for p in self.parameters]
        return d


def _read_from(lamp: Lamp, level: int, name: str) -> str:
    return f"model level {level}" if name in lamp.models[level].params else "physical description"


def free_parameters(lamp: Lamp, level: int) -> list[str]:
    """Parameters the level reads that have bounds, in the order the generator reports them."""
    spec = lamp.models[level]
    used = lamp.generate(level).metadata.get("params", {})
    return [name for name in used if lamp.parameter(name, spec).bounds is not None]


def with_values(lamp: Lamp, level: int, values: dict[str, float]) -> Lamp:
    """A copy of the lamp with each value written where the level reads it."""
    spec = lamp.models[level]
    spec_params, physical = dict(spec.params), dict(lamp.physical)
    for name, v in values.items():
        target = spec_params if name in spec_params else physical
        target[name] = replace(lamp.parameter(name, spec), value=float(v))
    models = {**lamp.models, level: replace(spec, params=spec_params)}
    return replace(lamp, physical=physical, models=models)


def _residuals(lamp: Lamp, level: int, ref, objective: str) -> np.ndarray:
    c = report.compare(report.model_profile(lamp, level, ref), ref)
    if objective == "spectral":
        return c.residuals
    return cm.uv_prime(c.model) - cm.uv_prime(c.reference)


def fit(lamp: Lamp, level: int, ref=None, objective: str = "spectral", names: list[str] | None = None) -> FitResult:
    """Fit ``names`` (default: every free parameter) of one model level to ``ref``
    (a :class:`~lss.accuracy.references.Reference`; default: the lamp's own)."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {', '.join(OBJECTIVES)}")
    if level not in lamp.models:
        raise KeyError(f"{lamp.id} has no level {level}; available: {sorted(lamp.models)}")
    if ref is None:
        if lamp.reference is None:
            raise ValueError(f"{lamp.id}: no reference to fit against")
        ref = references.get(lamp.reference)

    missing = report.missing_components(report.model_profile(lamp, level, ref), ref)
    if missing:
        raise ValueError(f"{lamp.id} level {level} cannot be fitted to {ref.id}: the reference includes "
                         f"{', '.join(sorted(missing))} and the model does not")

    free = free_parameters(lamp, level)
    if names is None:
        names = free
        if not names:
            raise ValueError(f"{lamp.id} level {level} has no free parameters: none of the parameters it reads "
                             "has bounds in the preset")
    else:
        for name in names:
            if name not in free:
                raise ValueError(f"{name!r} is not a free parameter of {lamp.id} level {level}; "
                                 f"free: {', '.join(free) or 'none'}")

    spec = lamp.models[level]
    params = [lamp.parameter(n, spec) for n in names]
    lo = np.array([p.bounds[0] for p in params])
    hi = np.array([p.bounds[1] for p in params])
    start = np.array([p.value for p in params])

    # Bounds must lie inside the model's validity, or a trial point fails mid-fit.
    for i, name in enumerate(names):
        for edge in (lo[i], hi[i]):
            try:
                with_values(lamp, level, {name: edge}).generate(level)
            except ValueError as e:
                raise ValueError(f"bounds of {name!r} reach {edge:g}, where level {level} is not defined: {e}") from None

    # Optimise in [0, 1] per parameter so every step size means the same thing.
    def lamp_at(x):
        return with_values(lamp, level, dict(zip(names, lo + x * (hi - lo))))

    sol = least_squares(lambda x: _residuals(lamp_at(x), level, ref, objective),
                        (start - lo) / (hi - lo), bounds=(0.0, 1.0), method="trf",
                        diff_step=1e-7, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    fitted = lamp_at(sol.x)
    values = lo + sol.x * (hi - lo)
    return FitResult(
        lamp_id=lamp.id,
        level=level,
        reference=ref.id,
        objective=objective,
        parameters=tuple(
            FittedParameter(n, p.unit, float(s), float(v), (float(a), float(b)), _read_from(lamp, level, n))
            for n, p, s, v, a, b in zip(names, params, start, values, lo, hi)
        ),
        before=report.score(lamp, level, ref),
        after=report.score(fitted, level, ref),
        success=bool(sol.success),
        message=str(sol.message),
        evaluations=int(sol.nfev),
        lamp=fitted,
    )


def _scores(row: report.Row) -> str:
    duv = "du'v'"
    return (f"NRMS {row.spectral_nrms:.3e}   {duv} {row.delta_uv_prime:.2e}   dCCT {row.delta_cct_k:+.2f} K   "
            f"dRa {row.delta_ra:+.2f}   dRf {row.delta_rf:+.2f}")


def format_text(result: FitResult) -> str:
    out = [f"Fit of {result.lamp_id} level {result.level} to {result.reference} "
           f"(objective: {result.objective}; normalisation: {report.NORMALISATION})", ""]
    for p in result.parameters:
        lo, hi = p.bounds
        flag = "   AT BOUND: the model only matches with an implausible value (missing physics?)" if p.at_bound else ""
        out.append(f"  {p.name}: {p.start:g} -> {p.value:.6g} {p.unit}   bounds {lo:g}-{hi:g} "
                   f"(at {100 * p.position:.1f} %), from the {p.read_from}{flag}")
    out += ["", f"  before: {_scores(result.before)}", f"  after:  {_scores(result.after)}", "",
            f"optimiser: {'converged' if result.success else 'DID NOT CONVERGE'} after {result.evaluations} "
            f"evaluations ({result.message})"]
    return "\n".join(out)


def yaml_snippet(result: FitResult, today: date | None = None) -> str:
    """The fitted parameters as preset YAML, tagged empirical, with where they go."""
    today = today or date.today()
    after = result.after
    reason = (f"fitted by `lss fit` ({result.objective} objective) to {result.reference} on {today.isoformat()}: "
              f"NRMS {after.spectral_nrms:.2e}, du'v' {after.delta_uv_prime:.1e}, dCCT {after.delta_cct_k:+.2f} K")
    where = sorted({"physical" if p.read_from == "physical description" else f"models: {result.level}: params"
                    for p in result.parameters})
    out = [f"# lss fit: replace under `{'` and `'.join(where)}` in {result.lamp_id}"]
    for p in result.parameters:
        lo, hi = p.bounds
        out += [f"{p.name}:", f"  value: {p.value:.6g}", f"  unit: {p.unit}", "  empirical: true", "  reason: >-",
                *textwrap.wrap(reason, 88, initial_indent="    ", subsequent_indent="    "),
                f"  bounds: [{lo:g}, {hi:g}]"]
    return "\n".join(out) + "\n"
