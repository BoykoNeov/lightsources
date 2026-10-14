"""Parameter fitting (design §11, `lss fit`)."""

import json

import numpy as np
import pytest
import yaml

from lss import cli
from lss.accuracy import fit, references, report
from lss.accuracy.references import Reference
from lss.lamp import lamp as lamps
from lss.params import Parameter
from lss.physics.thermal.tungsten import TungstenContinuum
from lss.spectral.spectrum import Spectrum


def _lamp(t_k=2900.0, bounds=(2400, 3200), level_params=None):
    t = {"value": t_k, "unit": "K", "empirical": True, "reason": "test"}
    if bounds is not None:
        t["bounds"] = list(bounds)
    models = {1: {"generator": "planck"}, 2: {"generator": "tungsten"}}
    if level_params:
        models[1]["params"] = level_params
    return lamps.lamp_from_dict({"id": "t", "family": "incandescent", "reference": "cie-a",
                                 "physical": {"filament_temperature": t}, "models": models})


def _tungsten_reference(t_k: float) -> Reference:
    # Scale is irrelevant: both objectives are blind to absolute level.
    spectrum = Spectrum.of(TungstenContinuum(t_k), name="synthetic").scaled(3.7)
    return Reference(id="synthetic", family="incandescent", description="tungsten at a known T",
                     spectrum=spectrum, valid_range=(380.0, 780.0), sampling_nm=1.0, kind="definition",
                     uncertainty="exact", includes=frozenset(), citation="test", data_version="test")


@pytest.mark.parametrize("objective", fit.OBJECTIVES)
def test_recovers_a_known_temperature(objective):
    result = fit.fit(_lamp(2900), 2, _tungsten_reference(2650.0), objective)
    assert result.success
    (p,) = result.parameters
    assert p.start == 2900
    assert p.value == pytest.approx(2650.0, abs=0.05)
    assert not p.at_bound
    assert result.after.spectral_nrms < 1e-5 < result.before.spectral_nrms


def test_fit_minimises_exactly_what_the_report_scores():
    result = fit.fit(_lamp(), 2)
    rescored = report.score(result.lamp, 2)
    assert rescored.spectral_nrms == result.after.spectral_nrms
    c = report.compare(result.lamp.generate(2), references.get("cie-a"))
    assert np.sqrt(np.mean(c.residuals ** 2)) == pytest.approx(result.after.spectral_nrms, rel=1e-12)
    # A small step either way only makes it worse.
    t = result.parameters[0].value
    for dt in (-0.5, 0.5):
        moved = fit.with_values(result.lamp, 2, {"filament_temperature": t + dt})
        assert report.score(moved, 2).spectral_nrms > result.after.spectral_nrms


def test_planck_against_illuminant_a_lands_on_its_definition():
    # A is Planck at 2856 K with a legacy c2: about 2855.5 K with CODATA 2018 constants.
    result = fit.fit(_lamp(2600), 1)
    assert result.parameters[0].value == pytest.approx(2855.5, abs=1.0)
    assert result.after.spectral_nrms < 1e-3


def test_value_at_a_bound_is_flagged():
    result = fit.fit(_lamp(2500, bounds=(2400, 2600)), 1)
    (p,) = result.parameters
    assert p.value == pytest.approx(2600, abs=0.5)
    assert p.at_bound and result.at_bound == ["filament_temperature"]
    assert "AT BOUND" in fit.format_text(result)


def test_value_is_written_where_the_level_reads_it():
    own = {"filament_temperature": {"value": 2700, "unit": "K", "empirical": True, "reason": "t",
                                    "bounds": [2500, 3100]}}
    lamp = _lamp(2900, level_params=own)
    result = fit.fit(lamp, 1)
    assert result.parameters[0].read_from == "model level 1"
    assert result.parameters[0].bounds == (2500, 3100)
    assert result.lamp.physical["filament_temperature"].value == 2900  # untouched
    assert result.lamp.models[1].params["filament_temperature"].value == result.parameters[0].value
    assert "models: 1: params" in fit.yaml_snippet(result)


def test_unfittable_requests_fail_clearly():
    with pytest.raises(ValueError, match="no free parameters"):
        fit.fit(_lamp(bounds=None), 2)
    with pytest.raises(ValueError, match="includes envelope"):
        fit.fit(_lamp(), 2, references.get("cie-hp1"))
    with pytest.raises(ValueError, match="not a free parameter"):
        fit.fit(_lamp(), 2, names=["coil_factor"])
    with pytest.raises(ValueError, match="not defined"):
        fit.fit(_lamp(bounds=(1500, 3200)), 2)  # De Vos data start at 1600 K
    with pytest.raises(ValueError, match="unknown objective"):
        fit.fit(_lamp(), 2, objective="vibes")


def test_snippet_is_a_valid_empirical_parameter():
    result = fit.fit(_lamp(), 2)
    doc = yaml.safe_load(fit.yaml_snippet(result))
    p = Parameter.from_dict(doc["filament_temperature"])
    assert p.empirical and "lss fit" in p.reason and "cie-a" in p.reason
    assert p.bounds == (2400, 3200)
    assert p.value == pytest.approx(result.parameters[0].value, rel=1e-5)


def test_cli_fit(capsys):
    assert cli.main(["fit", "incandescent_cie_a", "--level", "2", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["reference"] == "cie-a" and out["objective"] == "spectral"
    assert out["parameters"][0]["name"] == "filament_temperature"
    assert out["after"]["spectral_nrms"] <= out["before"]["spectral_nrms"]
    assert cli.main(["fit", "incandescent_cie_a", "--level", "1"]) == 2
