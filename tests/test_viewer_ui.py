"""The viewer window (design §9.7), offscreen: lamp switching, the probe, a missing map
rendered by the worker (numpy stand-in) and shown. ``pytest -m ui``."""

import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.ui

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from lss.render import maps  # noqa: E402
from lss.ui.session import Session  # noqa: E402
from lss.ui.viewer import Viewer  # noqa: E402
from test_render_worker import study_path  # noqa: E402, F401
from test_viewer_session import small, store  # noqa: E402

FAKE = {"sigma": 1.0, "s_per_spp": 2e-4}


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_CACHE", str(tmp_path / "cache"))


def wait(app, done, timeout=300.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if done():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def picture(w) -> np.ndarray:
    return np.array(w.img.image)


def test_lamp_switch_and_probe(app, tmp_path):
    study = small("lab")
    store(study)
    w = Viewer(Session(study), "lab_small", "fake", FAKE)
    try:
        assert "showing incandescent_cie_a level 2" in w.badge.text()
        before = picture(w)
        preset, level, dim = w.lamp_controls["lamp"]
        preset.setCurrentText("hps_cie_hp1")
        wait(app, lambda: "showing hps_cie_hp1 level 0" in w.badge.text())
        assert not np.array_equal(before, picture(w))
        w.mode_box.setCurrentIndex(w.mode_box.findData("daylight"))
        assert "daylight display" in w.footer.text()
        w.probe(30, 20)
        assert "cd/m²" in w.probe_text.text() and "hps_cie_hp1 level 0" in w.probe_text.text()
        dim.setValue(50)
        wait(app, lambda: "at 50 %" in w.badge.text())
        assert "dimmed to 50 %" in w.footer.text()
        wait(app, lambda: "HPS" in w.sheet_text.toPlainText() or "hps" in w.sheet_text.toPlainText())
        w.grab().save(str(tmp_path / "viewer.png"))
    finally:
        w.close()


def test_a_missing_map_is_rendered_and_shown(app, study_path):  # noqa: F811
    from lss.scene import study as studies

    study = studies.load(study_path)
    w = Viewer(Session(study), study_path, "fake", FAKE)
    try:
        assert "needs render: lamp@isotropic" in w.badge.text()
        w.render_missing()
        wait(app, lambda: "showing incandescent_cie_a level 2" in w.badge.text())
        assert maps.load(study, "lamp", "isotropic") is not None
        assert w.client.status["lamp@isotropic"].state == "converged"
        assert "Converged" in [w.table.item(r, 1).text() for r in range(w.table.rowCount())]
    finally:
        w.close()
