"""Lamp sheet (design §7): the numbers that describe one lamp at one model level.

Quick path only (no renderer): the generator's emission passed through the
lamp's own envelope as a spectral factor (``lss.lamp.quickpath``). Relative spectra give chromaticity, colour
rendering and the luminous efficacy *of the radiation*; absolute flux and lamp
efficacy need a model with an energy budget and are reported as unavailable
until one exists.
"""

from dataclasses import asdict, dataclass

import numpy as np

from lss.lamp import quickpath
from lss.lamp.lamp import Lamp
from lss.provenance import code_version
from lss.spectral import colorimetry as cm
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import Spectrum


@dataclass(frozen=True)
class LampSheet:
    lamp_id: str
    lamp_name: str
    family: str
    level: int
    generator: str
    quantity: str
    includes: list[str]
    x: float
    y: float
    u_prime: float
    v_prime: float
    cct_k: float
    duv: float
    ra: float
    r9: float
    rf: float
    rg: float
    cri_ri: dict[int, float]  # CIE 13.3 special indices R1 … R14
    tm30_hue_bins: dict[str, list[float]]  # TM-30 per hue bin: rf, rcs (%), rhs (rad); 16 bins from red
    ler_lm_per_w: float  # luminous efficacy of the radiation
    ler_basis: str  # what the radiant watts in LER cover
    visible_fraction: float | None  # 380–780 nm share of all-wavelength radiant power
    radiant_flux_w: float | None
    luminous_flux_lm: float | None
    valid_range_nm: tuple[float, float] | None
    params: dict
    derived: dict  # quantities the model computed from its parameters (e.g. a temperature)
    quick_path: dict  # components applied after the generator, with their parameters
    data_versions: list[str]
    code_version: str

    def as_dict(self) -> dict:
        return asdict(self)


def _all_wavelength_total(spectrum: Spectrum) -> float | None:
    """Radiant total over all wavelengths if every component is unbounded physics, else None."""
    from lss.physics.led.bands import EnergyBand
    from lss.spectral.continuum import PlanckContinuum

    if all(isinstance(c.source, (PlanckContinuum, EnergyBand)) for c in spectrum.components):
        return spectrum.total()
    return None


def make(lamp: Lamp, level: int) -> LampSheet:
    profile = quickpath.apply(lamp, lamp.generate(level))
    spectrum = profile.spectrum
    binned = spectrum.bin(analysis_grid())
    meta = profile.metadata

    cct, duv = cm.cct_duv(binned)
    xy = cm.xy(binned)
    upvp = cm.uv_prime(binned)
    cr = cm.colour_rendering(binned)
    lum = cm.luminous(binned)

    valid = meta.get("valid_range_nm")
    total = _all_wavelength_total(spectrum)
    if total is not None:
        radiant, basis = total, "all wavelengths (analytic)"
        visible = spectrum.integral(380.0, 780.0) / total
    else:
        lo, hi = valid if valid else (analysis_grid().lo, analysis_grid().hi)
        radiant, basis = spectrum.integral(lo, hi), f"{lo:g}-{hi:g} nm only (table range)"
        visible = None

    flux_w = profile.power
    flux_lm = lum / radiant * flux_w if flux_w is not None else None

    return LampSheet(
        lamp_id=lamp.id,
        lamp_name=lamp.name,
        family=lamp.family,
        level=level,
        generator=meta.get("generator", ""),
        quantity=spectrum.quantity,
        includes=sorted(profile.includes),
        x=float(xy[0]),
        y=float(xy[1]),
        u_prime=float(upvp[0]),
        v_prime=float(upvp[1]),
        cct_k=cct,
        duv=duv,
        ra=cr.ra,
        r9=cr.r9,
        rf=cr.rf,
        rg=cr.rg,
        cri_ri=cr.ri,
        tm30_hue_bins={"rf": list(cr.rf_h), "rcs": list(cr.rcs_h), "rhs": list(cr.rhs_h)},
        ler_lm_per_w=float(lum / radiant),
        ler_basis=basis,
        visible_fraction=visible,
        radiant_flux_w=flux_w,
        luminous_flux_lm=flux_lm,
        valid_range_nm=tuple(valid) if valid else None,
        params=meta.get("params", {}),
        derived=meta.get("derived", {}),
        quick_path=meta.get("quick_path", {}),
        data_versions=list(meta.get("data_versions", [])),
        code_version=code_version(),
    )


def format_text(sheet: LampSheet) -> str:
    def row(label, value):
        return f"  {label:<28}{value}"

    lines = [
        f"{sheet.lamp_name}",
        f"  [{sheet.lamp_id}] family {sheet.family}, level {sheet.level} ({sheet.generator})",
        "",
        "Colour",
        row("chromaticity x, y", f"{sheet.x:.4f}, {sheet.y:.4f}"),
        row("u', v'", f"{sheet.u_prime:.4f}, {sheet.v_prime:.4f}"),
        row("CCT", f"{sheet.cct_k:.0f} K"),
        row("Duv", f"{sheet.duv:+.4f}"),
        row("CRI Ra / R9", f"{sheet.ra:.1f} / {sheet.r9:.1f}"),
        row("TM-30 Rf / Rg", f"{sheet.rf:.1f} / {sheet.rg:.1f}"),
        row("CRI R1-R14", " ".join(f"{sheet.cri_ri[i]:.0f}" for i in sorted(sheet.cri_ri))),
        "",
        "TM-30 by hue (16 bins of 22.5 deg, from red)",
        row("hue bin", "".join(f"{j:>6}" for j in range(1, 17))),
        row("local fidelity Rf,h", "".join(f"{v:6.0f}" for v in sheet.tm30_hue_bins["rf"])),
        row("chroma shift Rcs,h %", "".join(f"{v:+6.0f}" for v in sheet.tm30_hue_bins["rcs"])),
        row("hue shift Rhs,h rad", "".join(f"{v:+6.2f}" for v in sheet.tm30_hue_bins["rhs"])),
        "",
        "Power",
        row("efficacy of radiation", f"{sheet.ler_lm_per_w:.1f} lm/W  (radiant W: {sheet.ler_basis})"),
    ]
    if sheet.visible_fraction is None:
        lines.append(row("", "partial-range value: NOT comparable with all-wavelength efficacies"))
    if sheet.visible_fraction is not None:
        lines.append(row("radiant share 380-780 nm", f"{100 * sheet.visible_fraction:.1f} %"))
    if sheet.radiant_flux_w is None:
        lines.append(row("absolute flux", "not available (relative spectrum, no energy budget)"))
    else:
        lines.append(row("radiant flux", f"{sheet.radiant_flux_w:.3g} W"))
        lines.append(row("luminous flux", f"{sheet.luminous_flux_lm:.4g} lm"))
    lines += [
        "",
        "Provenance",
        row("spectrum units", sheet.quantity),
        row("already includes", ", ".join(sheet.includes) or "nothing (bare emitter)"),
    ]
    if sheet.valid_range_nm:
        lines.append(row("data range", f"{sheet.valid_range_nm[0]:g}-{sheet.valid_range_nm[1]:g} nm"))
    for name, p in sheet.params.items():
        src = p.get("source") or f"EMPIRICAL: {p.get('reason')}"
        lines.append(row(name, f"{p['value']:g} {p['unit']}  ({src})"))
    for name, d in sheet.derived.items():
        lines.append(row(name, f"{d['value']:.6g} {d['unit']}  (derived: {d['from']})"))
    env = sheet.quick_path.get("envelope")
    if env:
        w = env["wall_thickness"]
        src = w.get("source") or f"EMPIRICAL: {w.get('reason')}"
        lines.append(row("envelope (quick path)", f"{env['material']}, wall {w['value']:g} {w['unit']}  ({src})"))
    for v in sheet.data_versions:
        lines.append(row("data", v))
    lines.append(row("code", sheet.code_version))
    return "\n".join(lines)


def spd_rows(lamp: Lamp, level: int) -> tuple[np.ndarray, np.ndarray]:
    """1 nm analysis-grid SPD (mean density per bin) for export."""
    binned = quickpath.apply(lamp, lamp.generate(level)).spectrum.bin(analysis_grid())
    return binned.grid.centres, binned.density

