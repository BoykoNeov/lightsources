"""Lamp sheet and CLI."""

import json

import numpy as np
import pytest

from lss import cli, lampsheet
from lss.lamp import lamp as lamps
from lss.physics.thermal import planck


def test_planck_sheet_matches_independent_numbers():
    sheet = lampsheet.make(lamps.load("incandescent_cie_a"), 1)
    t = 2856.0
    assert sheet.cct_k == pytest.approx(2856.0, abs=2.0)
    expected_visible = planck.fraction_below(780.0, t) - planck.fraction_below(380.0, t)
    assert sheet.visible_fraction == pytest.approx(expected_visible, rel=1e-9)
    # LER = K_m ∫V B dλ / (σT⁴/π); V from the CIE 1 nm table, B at the table points.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour
    v = colour.SDS_LEFS["CIE 1924 Photopic Standard Observer"]
    lam = np.asarray(v.wavelengths)
    ler = 683.0 * np.sum(np.asarray(v.values) * planck.spectral_radiance(lam, t)) / planck.total_radiance(t)
    assert sheet.ler_lm_per_w == pytest.approx(ler, rel=1e-4)
    assert sheet.ler_basis.startswith("all wavelengths")
    assert sheet.params["filament_temperature"]["value"] == 2856.0
    assert sheet.code_version.startswith("lss ")


def test_reference_sheet_states_its_limits():
    sheet = lampsheet.make(lamps.load("hps_cie_hp1"), 0)
    assert sheet.valid_range_nm == (380.0, 780.0)
    assert "table range" in sheet.ler_basis
    assert sheet.visible_fraction is None
    assert sheet.radiant_flux_w is None and sheet.luminous_flux_lm is None
    assert sheet.includes == ["envelope"]
    text = lampsheet.format_text(sheet)
    assert "not available" in text and "380-780 nm" in text
    assert "NOT comparable" in text


def test_tungsten_sheet_flags_its_partial_range_efficacy():
    # Level 2 knows ε only over 300-2500 nm, so its efficacy must not pass as an all-wavelength value.
    sheet = lampsheet.make(lamps.load("incandescent_cie_a"), 2)
    assert sheet.ler_basis.startswith("300-2500 nm only")
    assert sheet.visible_fraction is None
    assert "NOT comparable" in lampsheet.format_text(sheet)
    assert "NOT comparable" not in lampsheet.format_text(lampsheet.make(lamps.load("incandescent_cie_a"), 1))


def test_cli_lampsheet_json(capsys):
    assert cli.main(["lampsheet", "incandescent_cie_a", "--json"]) == 0
    sheets = json.loads(capsys.readouterr().out)
    assert [s["level"] for s in sheets] == [0, 1, 2]


def test_cli_spd_csv(tmp_path, capsys):
    out = tmp_path / "spd.csv"
    assert cli.main(["lampsheet", "ref:cie-fl11", "--spd-csv", str(out)]) == 0
    rows = out.read_text().splitlines()
    assert rows[1] == "wavelength_nm,density"
    assert len(rows) == 2 + 471


def test_cli_refs_and_errors(capsys):
    assert cli.main(["refs", "--family", "led"]) == 0
    assert "cie-led-b3" in capsys.readouterr().out
    assert cli.main(["lampsheet", "no_such_lamp"]) == 2


def test_sheet_carries_cri_and_tm30_hue_detail():
    sheet = lampsheet.make(lamps.load("hps_cie_hp1"), 0)
    assert sorted(sheet.cri_ri) == list(range(1, 15)) and sheet.cri_ri[9] == sheet.r9
    for key in ("rf", "rcs", "rhs"):
        assert len(sheet.tm30_hue_bins[key]) == 16
    # HPS: the reds and greens (bins 1 and 7) lose chroma, the oranges (bins 3-4) gain it.
    rcs = sheet.tm30_hue_bins["rcs"]
    assert rcs[0] < 0 and rcs[6] < 0 and max(rcs[2], rcs[3]) > 0
    text = lampsheet.format_text(sheet)
    assert "TM-30 by hue" in text and "CRI R1-R14" in text
