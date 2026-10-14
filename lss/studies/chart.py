"""Colour chart under a lamp (design §7, no-render study).

Each sample's colour under the lamp is compared with its colour under TM-30's
reference light at the same CCT, in CAM02-UCS (``lss.studies.appearance``).
The shift ΔE′ is split into lightness ΔJ′, chroma ΔC′ and hue Δh. With the 99
TM-30 samples (``--samples ces``) the chart's fidelity is the lamp's Rf.
"""

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from lss.provenance import code_version
from lss.studies import appearance as ap
from lss.studies.samples import SampleSet

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from colour.quality.cfi2017 import delta_E_to_R_f


#: Below this CAM02-UCS chroma (under both lights) a sample is near-neutral and its hue angle is noise.
NEUTRAL_CHROMA = 3.0


@dataclass(frozen=True)
class PatchShift:
    label: str
    delta_e: float  # CAM02-UCS ΔE′, lamp vs reference
    delta_j: float  # lightness J′, lamp minus reference
    delta_c: float  # chroma C′ = √(a′² + b′²), lamp minus reference
    delta_h_deg: float | None  # hue angle, lamp minus reference, in (−180, 180]; None if near-neutral
    srgb_lamp: str  # display colour under the lamp (adapted to D65)
    srgb_reference: str  # display colour under the reference
    out_of_gamut: bool  # either display colour was clipped to sRGB


@dataclass(frozen=True)
class Chart:
    light: str
    reference: str
    reference_cct_k: float
    samples: str
    patches: list[PatchShift]
    mean_delta_e: float
    fidelity: float  # TM-30 scaling of the mean ΔE′ (equals Rf for the CES set)
    measured_range_nm: tuple[float, float]  # sample data range; held flat beyond
    sample_citation: str
    data_versions: list[str]
    code_version: str

    def as_dict(self) -> dict:
        return asdict(self)


def _chroma_hue(jab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.hypot(jab[:, 1], jab[:, 2]), np.degrees(np.arctan2(jab[:, 2], jab[:, 1]))


def make(light: ap.Light, samples: SampleSet, data_versions=()) -> Chart:
    ref, cct = ap.reference_light(light)
    jab_t = ap.cam02ucs(light, samples.values)
    jab_r = ap.cam02ucs(ref, samples.values)
    de = np.linalg.norm(jab_t - jab_r, axis=1)
    c_t, h_t = _chroma_hue(jab_t)
    c_r, h_r = _chroma_hue(jab_r)
    dh = (h_t - h_r + 180.0) % 360.0 - 180.0
    rgb_t, out_t = ap.srgb(light, samples.values)
    rgb_r, out_r = ap.srgb(ref, samples.values)
    patches = [
        PatchShift(label, float(de[i]), float(jab_t[i, 0] - jab_r[i, 0]), float(c_t[i] - c_r[i]),
                   float(dh[i]) if max(c_t[i], c_r[i]) >= NEUTRAL_CHROMA else None,
                   ap.hex_colour(rgb_t[i]), ap.hex_colour(rgb_r[i]), bool(out_t[i] or out_r[i]))
        for i, label in enumerate(samples.labels)
    ]
    mean = float(de.mean())
    return Chart(
        light=light.name,
        reference=ref.name,
        reference_cct_k=cct,
        samples=samples.id,
        patches=patches,
        mean_delta_e=mean,
        fidelity=float(delta_E_to_R_f(mean)),
        measured_range_nm=samples.measured_range,
        sample_citation=samples.citation,
        data_versions=[*data_versions, samples.data_version],
        code_version=code_version(),
    )


def format_text(chart: Chart) -> str:
    lo, hi = chart.measured_range_nm
    lines = [
        f"{chart.light} vs {chart.reference} (TM-30 reference at the same CCT)",
        f"samples: {chart.samples}; colour shift in CAM02-UCS, 10 deg observer",
        "",
        f"  {'sample':<24}{'dE':>6}{'dJ':>7}{'dC':>7}{'dh deg':>8}   {'lamp':<8} {'ref':<8}",
    ]
    for p in chart.patches:
        flag = "  (clipped to sRGB)" if p.out_of_gamut else ""
        dh = f"{p.delta_h_deg:+8.1f}" if p.delta_h_deg is not None else f"{'-':>8}"
        lines.append(f"  {p.label:<24}{p.delta_e:6.2f}{p.delta_j:+7.2f}{p.delta_c:+7.2f}{dh}"
                     f"   {p.srgb_lamp} {p.srgb_reference}{flag}")
    worst = max(chart.patches, key=lambda p: p.delta_e)
    lines += [
        "",
        f"  mean dE {chart.mean_delta_e:.2f}; on TM-30's scale {chart.fidelity:.1f}"
        + ("  (= Rf: these are its samples)" if chart.samples == "ces" else ""),
        f"  largest shift: {worst.label} ({worst.delta_e:.2f})",
        "  dJ, dC: lightness and chroma under the lamp minus under the reference; dh: hue turn",
        f"  (blank for near-greys, chroma below {NEUTRAL_CHROMA:g}).",
        "  lamp / ref: display colours after adapting to each light's white (sRGB, D65).",
        "",
        f"  sample data {lo:g}-{hi:g} nm, end values held beyond (an extrapolation)",
        f"  {chart.sample_citation}",
    ]
    lines += [f"  data: {v}" for v in chart.data_versions]
    lines.append(f"  code: {chart.code_version}")
    return "\n".join(lines)


def svg(chart: Chart, columns: int | None = None) -> str:
    """Swatches: the upper half under the lamp, the lower half under the reference, ΔE′ beneath."""
    n = len(chart.patches)
    cols = columns or (6 if n <= 24 else 11)
    size, gap, label_h, top = 96, 12, 34, 44
    rows = -(-n // cols)
    width = gap + cols * (size + gap)
    height = top + rows * (size + label_h + gap)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#767676"/>',
        f'<text x="{gap}" y="20" font-size="14" fill="#fff">{_esc(chart.light)}: upper halves; '
        f'{_esc(chart.reference)}: lower halves</text>',
        f'<text x="{gap}" y="36" font-size="11" fill="#eee">colour shift dE (CAM02-UCS) under each patch; '
        f'mean {chart.mean_delta_e:.2f}</text>',
    ]
    half = size // 2
    for i, p in enumerate(chart.patches):
        x = gap + (i % cols) * (size + gap)
        y = top + (i // cols) * (size + label_h + gap)
        out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{half}" fill="{p.srgb_lamp}"/>')
        out.append(f'<rect x="{x}" y="{y + half}" width="{size}" height="{size - half}" fill="{p.srgb_reference}"/>')
        out.append(f'<text x="{x}" y="{y + size + 13}" font-size="10" fill="#fff">{_esc(p.label[:18])}</text>')
        mark = "*" if p.out_of_gamut else ""
        out.append(f'<text x="{x}" y="{y + size + 27}" font-size="11" fill="#fff">dE {p.delta_e:.1f}{mark}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
