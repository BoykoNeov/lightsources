"""Joint fit of shared phosphor shapes over several lamps (design §11, ``lss fit A B C``)."""

import numpy as np
import pytest

from lss.accuracy import fit, report
from lss.accuracy.references import Reference
from lss.lamp import lamp as lamps
from lss.lamp import phosphors, quickpath
from lss.spectral.bandpass import as_recorded

FL = ["fluorescent_cie_fl10", "fluorescent_cie_fl11", "fluorescent_cie_fl12"]
NODES = np.arange(380.0, 780.0 + 2.5, 5.0)

#: Mixes as alike as the real CIE lamps': red/green rises ~1.75x and blue/green falls
#: ~3.6x from FL10 to FL12, Hg/green barely changes. Distinct mixes would make
#: recovery trivially easy and prove nothing about these data.
MIXES = [{"hg_to_green": 0.30, "blue_to_green": 0.72, "red_to_green": 0.52},
         {"hg_to_green": 0.29, "blue_to_green": 0.45, "red_to_green": 0.66},
         {"hg_to_green": 0.27, "blue_to_green": 0.20, "red_to_green": 0.91}]


def _truth_shapes() -> dict[str, float]:
    """Library values moved inside their bounds: the shapes the fit must find."""
    out = {}
    bounded = [(n, p) for n, p in phosphors.parameters().items() if p.bounds is not None]
    for i, (name, p) in enumerate(bounded):
        lo, hi = p.bounds
        moved = p.value + (0.075 if i % 2 else -0.075) * (hi - lo)
        out[name] = min(max(moved, lo + 0.05 * (hi - lo)), hi - 0.05 * (hi - lo))
    return out


def _synthetic_items(shapes):
    items = []
    for preset, mix in zip(FL, MIXES):
        lamp = lamps.load(preset)
        truth = fit.with_values(lamp, 1, {**shapes, **mix})
        recorded = as_recorded(quickpath.apply(truth, truth.generate(1)).spectrum, NODES)
        ref = Reference(id=f"synthetic-{preset}", family="fluorescent", description="known shapes and mix",
                        spectrum=recorded, valid_range=(380.0, 780.0), sampling_nm=5.0, kind="definition",
                        uncertainty="exact", includes=frozenset({"phosphor", "envelope"}), citation="test",
                        data_version="test", bandpass_nm=5.0)
        items.append((lamp, 1, ref))
    return items


def test_single_lamp_fit_holds_the_shared_shapes():
    lamp = lamps.load("fluorescent_cie_fl11")
    assert fit.free_parameters(lamp, 1) == ["hg_to_green", "blue_to_green", "red_to_green"]
    shared = fit.free_parameters(lamp, 1, shared=True)
    assert len(shared) > 20 and all(phosphors.is_shared(n) for n in shared[3:])


def test_trial_values_stay_library_values():
    lamp = lamps.load("fluorescent_cie_fl11")
    tried = fit.with_values(lamp, 1, {"LAP.7F6.share": 0.3, "red_to_green": 0.7})
    assert tried.parameter("LAP.7F6.share").value == 0.3
    assert tried.reads_from_library("LAP.7F6.share")
    assert "LAP.7F6.share" not in tried.physical
    assert tried.parameter("red_to_green", tried.models[1]).value == 0.7


def test_joint_fit_recovers_shared_shapes_and_every_mix():
    shapes = _truth_shapes()
    result = fit.fit_joint(_synthetic_items(shapes))
    assert result.success
    for row in result.after:
        assert row.spectral_nrms < 1e-5
    for p in result.parameters:
        truth = shapes[p.name] if p.owner == "" else MIXES[FL.index(p.owner)][p.name]
        assert p.value == pytest.approx(truth, rel=1e-3, abs=1e-3), f"{p.owner or 'shared'}: {p.name}"


def test_joint_snippet_puts_shared_values_in_the_library():
    lamp = lamps.load("fluorescent_cie_fl11")
    params = (fit.FittedParameter("LAP.7F6.share", "1", 0.25, 0.31, (0, 1), fit.LIBRARY, ""),
              fit.FittedParameter("red_to_green", "1", 0.6, 0.7, (0.05, 5), "model level 1", lamp.id))
    row = report.score(lamp, 1)
    result = fit.JointFitResult("spectral", ((lamp.id, 1, "cie-fl11"),), params, (row,), (row,), True, "", 1, (lamp,))
    text = fit.joint_yaml_snippet(result)
    assert "lss/data/phosphors.yaml" in text and "# LAP / 7F6\nshare:\n  value: 0.31" in text
    assert f"`models: 1: params` in {lamp.id}\nred_to_green:\n  value: 0.7" in text


def test_joint_fit_needs_shared_parameters():
    a, b = lamps.load("led_cie_b3"), lamps.load("led_cie_b5")
    with pytest.raises(ValueError, match="phosphor-library"):
        fit.fit_joint([(a, 1, None), (b, 1, None)])
