"""Accuracy report (design §11): scoring rules and the golden-file check."""

import json
from dataclasses import replace

import pytest

from lss.accuracy import report
from lss.lamp import lamp as lamps


@pytest.fixture(scope="module")
def rows():
    return report.run()


def test_no_regression_against_golden_file(rows):
    assert report.check_golden(rows) == []


def test_reference_scored_against_itself_is_exact():
    row = report.score(lamps.load("incandescent_cie_a"), 0)
    assert row.comparable
    assert row.spectral_nrms == pytest.approx(0.0, abs=1e-15)
    assert row.delta_uv_prime == pytest.approx(0.0, abs=1e-15)
    assert row.delta_cct_k == pytest.approx(0.0, abs=1e-9)


def test_planck_level_1_matches_illuminant_a(rows):
    # Illuminant A is Planck with a legacy c2; the residual is that ~0.5 K
    # definitional offset plus reading the 5 nm table piecewise linearly.
    row = next(r for r in rows if r.key == "incandescent_cie_a@L1")
    assert row.comparable and row.path == "quick"
    assert row.normalisation == "equal luminous flux"
    assert row.spectral_nrms < 1e-3
    assert row.delta_uv_prime < 1e-4
    assert abs(row.delta_cct_k) < 2.0
    assert row.range_nm[1] == 780.0  # never scored beyond the table


def test_model_missing_a_component_is_not_scored():
    lamp = lamps.lamp_from_dict({
        "id": "planck_vs_hps", "family": "hid", "reference": "cie-hp1",
        "physical": {"filament_temperature": {"value": 2000, "unit": "K", "source": "test"}},
        "models": {1: {"generator": "planck"}},
    })
    row = report.score(lamp, 1)
    assert not row.comparable
    assert "envelope" in row.note
    assert row.spectral_nrms is None


def test_golden_check_catches_regressions(rows, tmp_path):
    path = tmp_path / "golden.json"
    report.write_golden(rows, path)
    assert report.check_golden(rows, path) == []

    data = json.loads(path.read_text())
    key = "incandescent_cie_a@L1"
    data[key]["spectral_nrms"] *= 0.5
    data[key]["empirical_params"] = -1
    data["vanished@L2"] = data[key]
    path.write_text(json.dumps(data))
    failures = report.check_golden(rows, path)
    assert any("spectral_nrms grew" in f for f in failures)
    assert any("empirical parameters" in f for f in failures)
    assert any("vanished@L2: row disappeared" in f for f in failures)


def test_new_row_requires_review(rows, tmp_path):
    path = tmp_path / "golden.json"
    report.write_golden(rows, path)
    extra = rows + [replace(rows[0], level=9)]
    assert any("new row" in f for f in report.check_golden(extra, path))
