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
* **Shared shapes need several lamps.** Parameters read from the shared phosphor
  library are held fixed when one lamp is fitted (``shared=True`` frees them, at
  the risk of shaping the library to one lamp). :func:`fit_joint` fits them to
  several lamps at once, each lamp keeping its own mix, and :func:`hold_out`
  scores a lamp the shared shapes were not fitted to.
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
from lss.lamp import phosphors
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
    read_from: str  # "model level N", "physical description" or "phosphor library"
    owner: str = ""  # the lamp it belongs to; "" for a shared library parameter

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


LIBRARY = "phosphor library"


def _read_from(lamp: Lamp, level: int, name: str) -> str:
    spec = lamp.models[level]
    if name in spec.params:
        return f"model level {level}"
    return LIBRARY if lamp.reads_from_library(name, spec) else "physical description"


def free_parameters(lamp: Lamp, level: int, shared: bool = False) -> list[str]:
    """Parameters the level reads that have bounds, in the order the generator reports them;
    shared phosphor-library parameters only if ``shared``."""
    spec = lamp.models[level]
    used = lamp.generate(level).metadata.get("params", {})
    return [name for name in used if lamp.parameter(name, spec).bounds is not None
            and (shared or not lamp.reads_from_library(name, spec))]


def with_values(lamp: Lamp, level: int, values: dict[str, float]) -> Lamp:
    """A copy of the lamp with each value written where the level reads it (a library
    parameter as a trial override, so it stays a library value)."""
    spec = lamp.models[level]
    spec_params, physical, library = dict(spec.params), dict(lamp.physical), dict(lamp.library_overrides)
    for name, v in values.items():
        if name in spec_params:
            target = spec_params
        elif lamp.reads_from_library(name, spec):
            target = library
        else:
            target = physical
        target[name] = replace(lamp.parameter(name, spec), value=float(v))
    models = {**lamp.models, level: replace(spec, params=spec_params)}
    return replace(lamp, physical=physical, models=models, library_overrides=library)


def _residuals(lamp: Lamp, level: int, ref, objective: str) -> np.ndarray:
    c = report.compare(report.model_profile(lamp, level, ref), ref)
    if objective == "spectral":
        return c.residuals
    return cm.uv_prime(c.model) - cm.uv_prime(c.reference)


def fit(lamp: Lamp, level: int, ref=None, objective: str = "spectral", names: list[str] | None = None,
        shared: bool = False) -> FitResult:
    """Fit ``names`` (default: every free parameter, library ones only if ``shared``) of one
    model level to ``ref`` (a :class:`~lss.accuracy.references.Reference`; default: the lamp's own)."""
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

    free = free_parameters(lamp, level, shared=shared)
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
            FittedParameter(n, p.unit, float(s), float(v), (float(a), float(b)), _read_from(lamp, level, n),
                            "" if _read_from(lamp, level, n) == LIBRARY else lamp.id)
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


def _parameter_line(p: FittedParameter) -> str:
    lo, hi = p.bounds
    flag = "   AT BOUND: the model only matches with an implausible value (missing physics?)" if p.at_bound else ""
    return (f"  {p.name}: {p.start:g} -> {p.value:.6g} {p.unit}   bounds {lo:g}-{hi:g} "
            f"(at {100 * p.position:.1f} %), from the {p.read_from}{flag}")


def format_text(result: FitResult) -> str:
    out = [f"Fit of {result.lamp_id} level {result.level} to {result.reference} "
           f"(objective: {result.objective}; normalisation: {report.NORMALISATION})", ""]
    out += [_parameter_line(p) for p in result.parameters]
    out += ["", f"  before: {_scores(result.before)}", f"  after:  {_scores(result.after)}", "",
            f"optimiser: {'converged' if result.success else 'DID NOT CONVERGE'} after {result.evaluations} "
            f"evaluations ({result.message})"]
    return "\n".join(out)


def _entry(p: FittedParameter, reason: str) -> list[str]:
    lo, hi = p.bounds
    name = p.name.rsplit(".", 1)[-1] if p.read_from == LIBRARY else p.name
    return [f"{name}:", f"  value: {p.value:.6g}", f"  unit: {p.unit}", "  empirical: true", "  reason: >-",
            *textwrap.wrap(reason, 88, initial_indent="    ", subsequent_indent="    "),
            f"  bounds: [{lo:g}, {hi:g}]"]


def _library_block(params: list[FittedParameter], reason: str) -> list[str]:
    """Library parameters grouped as they sit in lss/data/phosphors.yaml."""
    out = [f"# lss fit: replace in {phosphors.LIBRARY_PATH}, under phosphors: <phosphor>: bands: <band>:"]
    groups: dict[str, list[FittedParameter]] = {}
    for p in params:
        groups.setdefault(p.name.rsplit(".", 1)[0], []).append(p)
    for key, ps in groups.items():
        ph, band = key.split(".", 1)
        out.append(f"# {ph} / {band}")
        for p in ps:
            out += _entry(p, reason)
    return out


def _lamp_block(lamp_id: str, level: int, params: list[FittedParameter], reason: str) -> list[str]:
    where = sorted({"physical" if p.read_from == "physical description" else f"models: {level}: params"
                    for p in params})
    out = [f"# lss fit: replace under `{'` and `'.join(where)}` in {lamp_id}"]
    for p in params:
        out += _entry(p, reason)
    return out


def _reason(objective: str, refs: str, rows, today: date) -> str:
    scores = "; ".join(f"NRMS {r.spectral_nrms:.2e}, du'v' {r.delta_uv_prime:.1e}, dCCT {r.delta_cct_k:+.2f} K"
                       + (f" ({r.reference})" if len(rows) > 1 else "") for r in rows)
    return f"fitted by `lss fit` ({objective} objective) to {refs} on {today.isoformat()}: {scores}"


def yaml_snippet(result: FitResult, today: date | None = None) -> str:
    """The fitted parameters as preset (and library) YAML, tagged empirical, with where they go."""
    today = today or date.today()
    reason = _reason(result.objective, result.reference, [result.after], today)
    own = [p for p in result.parameters if p.read_from != LIBRARY]
    lib = [p for p in result.parameters if p.read_from == LIBRARY]
    out = _lamp_block(result.lamp_id, result.level, own, reason) if own else []
    if lib:
        out += ["", "# WARNING: shared phosphor shapes fitted to one lamp only"] + _library_block(lib, reason)
    return "\n".join(out) + "\n"


# ---- Joint fit: shared phosphor shapes over several lamps -------------------------------


@dataclass(frozen=True)
class JointFitResult:
    objective: str
    items: tuple[tuple[str, int, str], ...]  # (lamp id, level, reference id)
    parameters: tuple[FittedParameter, ...]
    before: tuple[report.Row, ...]
    after: tuple[report.Row, ...]
    success: bool
    message: str
    evaluations: int
    lamps: tuple[Lamp, ...] = field(compare=False, repr=False)  # the lamps with the fitted values

    @property
    def shared(self) -> dict[str, float]:
        return {p.name: p.value for p in self.parameters if p.owner == ""}

    @property
    def at_bound(self) -> list[str]:
        return [f"{p.owner}: {p.name}" if p.owner else p.name for p in self.parameters if p.at_bound]

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("lamps", "parameters")}
        d["parameters"] = [{**asdict(p), "position": p.position, "at_bound": p.at_bound} for p in self.parameters]
        return d


def _items(items) -> list:
    out = []
    for lamp, level, ref in items:
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
        out.append((lamp, level, ref))
    return out


def fit_joint(items, objective: str = "spectral") -> JointFitResult:
    """Fit the shared library parameters that several lamps read, together with each lamp's
    own free parameters, to each lamp's reference at once.

    ``items`` is a sequence of ``(lamp, level, reference or None)``. The residuals of all
    lamps are minimised together, so a shared shape has to suit every lamp.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {', '.join(OBJECTIVES)}")
    items = _items(items)
    if len(items) < 2:
        raise ValueError("a joint fit needs at least two lamps")
    shared: list[str] = []
    own: list[tuple[int, str]] = []
    for i, (lamp, level, _) in enumerate(items):
        for n in free_parameters(lamp, level, shared=True):
            if _read_from(lamp, level, n) == LIBRARY:
                if n not in shared:
                    shared.append(n)
            else:
                own.append((i, n))
    if not shared:
        raise ValueError("none of these lamps reads a bounded phosphor-library parameter; fit them one by one")

    lib = phosphors.parameters()
    params = [lib[n] for n in shared] + [items[i][0].parameter(n, items[i][0].models[items[i][1]]) for i, n in own]
    lo = np.array([p.bounds[0] for p in params])
    hi = np.array([p.bounds[1] for p in params])
    start = np.array([p.value for p in params])
    ns = len(shared)
    reads = [set(lamp.generate(level).metadata.get("params", {})) for lamp, level, _ in items]

    def lamps_at(x):
        v = lo + x * (hi - lo)
        common = dict(zip(shared, v[:ns]))
        out = []
        for i, (lamp, level, _) in enumerate(items):
            mine = {n: v[ns + j] for j, (k, n) in enumerate(own) if k == i}
            out.append(with_values(lamp, level, {**{n: c for n, c in common.items() if n in reads[i]}, **mine}))
        return out

    def residuals(x):
        return np.concatenate([_residuals(fl, level, ref, objective)
                               for fl, (_, level, ref) in zip(lamps_at(x), items)])

    sol = least_squares(residuals, (start - lo) / (hi - lo), bounds=(0.0, 1.0), method="trf",
                        diff_step=1e-7, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    fitted = lamps_at(sol.x)
    values = lo + sol.x * (hi - lo)
    names = [("", n, LIBRARY) for n in shared] + [
        (items[i][0].id, n, _read_from(items[i][0], items[i][1], n)) for i, n in own]
    return JointFitResult(
        objective=objective,
        items=tuple((lamp.id, level, ref.id) for lamp, level, ref in items),
        parameters=tuple(
            FittedParameter(n, p.unit, float(s), float(v), (float(a), float(b)), where, owner)
            for (owner, n, where), p, s, v, a, b in zip(names, params, start, values, lo, hi)
        ),
        before=tuple(report.score(lamp, level, ref) for lamp, level, ref in items),
        after=tuple(report.score(fl, level, ref) for fl, (_, level, ref) in zip(fitted, items)),
        success=bool(sol.success),
        message=str(sol.message),
        evaluations=int(sol.nfev),
        lamps=tuple(fitted),
    )


def hold_out(items, held: int, objective: str = "spectral") -> tuple[JointFitResult, FitResult]:
    """Out-of-sample check: fit the shared shapes to every lamp but ``items[held]``, then fit
    only that lamp's own parameters with those shapes, and score it."""
    items = _items(items)
    joint = fit_joint([it for i, it in enumerate(items) if i != held], objective)
    lamp, level, ref = items[held]
    reads = lamp.generate(level).metadata.get("params", {})
    shaped = with_values(lamp, level, {n: v for n, v in joint.shared.items() if n in reads})
    return joint, fit(shaped, level, ref, objective)


def format_joint(result: JointFitResult) -> str:
    out = [f"Joint fit of {', '.join(f'{lid} L{lv}' for lid, lv, _ in result.items)} "
           f"(objective: {result.objective}; normalisation: {report.NORMALISATION})", "", "Shared (phosphor library)"]
    out += [_parameter_line(p) for p in result.parameters if p.owner == ""]
    for (lid, _, ref), before, after in zip(result.items, result.before, result.after):
        out += ["", f"{lid} (scored against {ref})"]
        out += [_parameter_line(p) for p in result.parameters if p.owner == lid]
        out += [f"  before: {_scores(before)}", f"  after:  {_scores(after)}"]
    out += ["", f"optimiser: {'converged' if result.success else 'DID NOT CONVERGE'} after {result.evaluations} "
                f"evaluations ({result.message})"]
    return "\n".join(out)


def joint_yaml_snippet(result: JointFitResult, today: date | None = None) -> str:
    """Library YAML for the shared values, then each lamp's own, all tagged empirical."""
    today = today or date.today()
    refs = ", ".join(ref for _, _, ref in result.items)
    out = _library_block([p for p in result.parameters if p.owner == ""],
                         _reason(result.objective, f"{refs} jointly", list(result.after), today))
    for (lid, level, _), after in zip(result.items, result.after):
        mine = [p for p in result.parameters if p.owner == lid]
        if mine:
            reason = _reason(result.objective, f"{after.reference}, jointly with {refs}", [after], today)
            out += [""] + _lamp_block(lid, level, mine, reason)
    return "\n".join(out) + "\n"
