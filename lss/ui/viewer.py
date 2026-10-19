"""The viewer window (design §9.7): PySide6 and pyqtgraph, the ``ui`` extra.

It draws what ``lss.ui.session`` computes and runs the render worker through
``lss.render.client``. Generators and lamp sheets run on a thread of their own; the
picture keeps the previous lamp, under a badge, until the new one is ready or while its
maps are missing.
"""

import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from lss import lampsheet
from lss.lamp import lamp as lamps
from lss.render import maps
from lss.render.client import RenderClient
from lss.render.worker import JobSpec, MapJob
from lss.spectral.grids import analysis_grid
from lss.ui.session import DISPLAY_MODES, Choice, MissingMaps, Session, levels

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

BADGE = {"info": "background:#fff4ce;color:#5c4400", "error": "background:#fde7e9;color:#7a0010",
         "ok": "background:transparent;color:palette(text)"}


def _eta(s: float) -> str:
    if not math.isfinite(s):
        return "—"
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, session: Session, study_ref: str, backend: str = "mitsuba",
                 backend_options: dict | None = None):
        super().__init__()
        self.session, self.study_ref = session, study_ref
        self.client = RenderClient(backend, backend_options)
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="lss-generator")
        self.mode, self.stops = "card", 0.0
        self.xyz = None
        self.shown = ""  # what the picture shows ("<lamp> level <n>")
        self.pending = {}  # lamp id -> (choice, future): a generator running
        self.sheet_future = None
        self.probe_at: tuple[int, int] | None = None
        self.jobs_by_map: dict[str, tuple[str, str]] = {}
        self.setWindowTitle(f"lss view — {session.study.id}")
        self._build()
        self.timer = QtCore.QTimer(self, interval=100, timeout=self._tick)
        self.timer.start()
        self.refresh()
        self._update_sheet()

    # --- layout ------------------------------------------------------------------------

    def _build(self) -> None:
        split = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        self.badge = QtWidgets.QLabel(wordWrap=True)
        self.badge.setContentsMargins(8, 4, 8, 4)
        lv.addWidget(self.badge)
        self.view = pg.GraphicsLayoutWidget()
        self.vb = self.view.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.marker = pg.ScatterPlotItem(size=14, symbol="s", pen=pg.mkPen("c", width=2), brush=None)
        self.vb.addItem(self.marker)
        self.view.scene().sigMouseClicked.connect(self._clicked)
        lv.addWidget(self.view, 1)
        self.footer = QtWidgets.QLabel(wordWrap=True)
        lv.addWidget(self.footer)
        split.addWidget(left)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._lamp_tab(), "Lamp")
        self.tabs.addTab(self._probe_tab(), "Probe")
        self.tabs.addTab(self._sheet_tab(), "Lamp sheet")
        self.tabs.addTab(self._render_tab(), "Renders")
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 500])
        self.setCentralWidget(split)
        self.resize(1300, 760)

    def _lamp_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.lamp_controls = {}
        for p in self.session.study.lamps:
            box = QtWidgets.QGroupBox(f"Lamp '{p.id}' ({p.kind}, {p.flux_lm:g} lm in the study)")
            f = QtWidgets.QFormLayout(box)
            preset, level = QtWidgets.QComboBox(), QtWidgets.QComboBox()
            preset.addItems(lamps.preset_ids())
            c = self.session.choices[p.id]
            if preset.findText(c.lamp) < 0:
                preset.addItem(c.lamp)
            preset.setCurrentText(c.lamp)
            self._fill_levels(level, c.lamp, c.level)
            dim = QtWidgets.QSpinBox(minimum=1, maximum=100, value=100, suffix=" %")
            f.addRow("Preset", preset)
            f.addRow("Model level", level)
            f.addRow("Dimming", dim)
            preset.currentTextChanged.connect(lambda t, lid=p.id: self._preset_changed(lid, t))
            level.activated.connect(lambda _i, lid=p.id: self._choose(lid))
            dim.valueChanged.connect(lambda _v, lid=p.id: self._choose(lid))
            self.lamp_controls[p.id] = (preset, level, dim)
            v.addWidget(box)
        disp = QtWidgets.QGroupBox("Display")
        f = QtWidgets.QFormLayout(disp)
        self.mode_box = QtWidgets.QComboBox()
        for k, text in DISPLAY_MODES.items():
            self.mode_box.addItem(text.split(":")[0], k)
        self.mode_box.currentIndexChanged.connect(self._display_changed)
        self.stops_box = QtWidgets.QDoubleSpinBox(minimum=-6, maximum=6, singleStep=0.5, suffix=" stops")
        self.stops_box.valueChanged.connect(self._display_changed)
        f.addRow("White", self.mode_box)
        f.addRow("Exposure", self.stops_box)
        v.addWidget(disp)
        self.noise_label = QtWidgets.QLabel(wordWrap=True)
        self.noise_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(QtWidgets.QLabel("<b>Lighting maps</b>"))
        v.addWidget(self.noise_label)
        v.addStretch(1)
        return w

    def _probe_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel("Click the picture: the spectrum arriving there (5 × 5 pixels)."))
        self.probe_plot = pg.PlotWidget(labels={"bottom": "wavelength (nm)", "left": "radiance (W m⁻² sr⁻¹ nm⁻¹)"})
        self.probe_curve = self.probe_plot.plot(pen=pg.mkPen("#2a6fdb", width=1.5))
        self.probe_hi = self.probe_plot.plot(pen=None)
        self.probe_lo = self.probe_plot.plot(pen=None)
        self.probe_plot.addItem(pg.FillBetweenItem(self.probe_lo, self.probe_hi, brush=pg.mkBrush(42, 111, 219, 60)))
        self.probe_plot.setXRange(350, 840, padding=0)
        v.addWidget(self.probe_plot, 1)
        self.probe_text = QtWidgets.QLabel(wordWrap=True)
        self.probe_text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(self.probe_text)
        return w

    def _sheet_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.sheet_plot = pg.PlotWidget(labels={"bottom": "wavelength (nm)", "left": "relative power"})
        self.sheet_plot.addLegend(offset=(-10, 10))
        v.addWidget(self.sheet_plot, 1)
        self.sheet_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.sheet_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        v.addWidget(self.sheet_text, 1)
        return w

    def _render_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.worker_label = QtWidgets.QLabel("worker: not started")
        v.addWidget(self.worker_label)
        row = QtWidgets.QHBoxLayout()
        self.threshold_box = QtWidgets.QDoubleSpinBox(minimum=0.1, maximum=20, value=1.0, singleStep=0.1, suffix=" %")
        row.addWidget(QtWidgets.QLabel("Noise threshold"))
        row.addWidget(self.threshold_box)
        row.addStretch(1)
        v.addLayout(row)
        buttons = QtWidgets.QHBoxLayout()
        self.render_button = QtWidgets.QPushButton("Render missing maps", clicked=self.render_missing)
        buttons.addWidget(self.render_button)
        for text, slot in (("Pause", self._pause), ("Resume", self._resume), ("Cancel", self._cancel),
                           ("Restart worker", self._restart)):
            buttons.addWidget(QtWidgets.QPushButton(text, clicked=slot))
        v.addLayout(buttons)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["map", "state", "passes", "spp", "noise", "ETA", "progress"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        v.addWidget(self.table, 1)
        v.addWidget(QtWidgets.QLabel("ETA is for the map rendering now; maps still queued are counted, not timed."))
        return w

    # --- lamp choice -------------------------------------------------------------------

    @staticmethod
    def _fill_levels(box: QtWidgets.QComboBox, preset: str, current: int | None = None) -> None:
        box.blockSignals(True)
        box.clear()
        lv = levels(preset)
        for n in lv:
            box.addItem(f"Level {n}", n)
        box.setCurrentIndex(lv.index(current) if current in lv else len(lv) - 1)
        box.blockSignals(False)

    def _preset_changed(self, lamp_id: str, preset: str) -> None:
        self._fill_levels(self.lamp_controls[lamp_id][1], preset)
        self._choose(lamp_id)

    def _choice_from_controls(self, lamp_id: str) -> Choice:
        preset, level, dim = self.lamp_controls[lamp_id]
        return Choice(preset.currentText(), level.currentData(), dim.value() / 100.0)

    def _choose(self, lamp_id: str) -> None:
        choice = self._choice_from_controls(lamp_id)
        if self.session.has_profile(lamp_id, choice):
            self.pending.pop(lamp_id, None)
            self._apply(lamp_id, choice)
            return
        fut = self.pool.submit(self.session.profile, lamp_id, choice)
        self.pending[lamp_id] = (choice, fut)
        self._badge("info", f"computing {choice.lamp} level {choice.level}… (the picture shows {self.shown})")

    def _apply(self, lamp_id: str, choice: Choice) -> None:
        old = self.session.choices[lamp_id]
        self.session.choices[lamp_id] = choice
        try:
            self.session.jobs()
        except (ValueError, KeyError) as e:
            self.session.choices[lamp_id] = old
            self._badge("error", f"{choice.lamp} level {choice.level}: {e}")
            return
        self.refresh()
        if (old.lamp, old.level) != (choice.lamp, choice.level):
            self._update_sheet()

    def _label(self) -> str:
        return ", ".join(f"{c.lamp} level {c.level}" + (f" at {100 * c.dimming:g} %" if c.dimming != 1 else "")
                         for c in self.session.choices.values())

    # --- picture -----------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-colour for the current choices, or say which maps are missing."""
        try:
            self.xyz = self.session.xyz()
        except MissingMaps as e:
            shown = f" The picture shows {self.shown}." if self.shown else ""
            self._badge("info", f"needs render: {', '.join(e.missing)} (no stored map for "
                                f"{self._label()}).{shown} Renders tab → Render missing maps.")
            self._update_notes()
            return
        self.shown = self._label()
        self._badge("ok", f"showing {self.shown}")
        self._draw()
        self._update_notes()
        if self.probe_at is not None:
            self._update_probe()

    def _draw(self) -> None:
        if self.xyz is None:
            return
        rgb = self.session.picture(self.xyz, self.mode, self.stops)
        self.img.setImage(np.round(rgb * 255).astype(np.uint8), levels=(0, 255))
        self.footer.setText(f"<b>Normalisation:</b> {self.session.normalisation()}<br>"
                            f"<b>Display:</b> {DISPLAY_MODES[self.mode]}, exposure {self.stops:+g} stops")

    def _display_changed(self, *_):
        self.mode, self.stops = self.mode_box.currentData(), self.stops_box.value()
        self._draw()

    def _badge(self, kind: str, text: str) -> None:
        self.badge.setStyleSheet(BADGE[kind])
        self.badge.setText(text)

    def _update_notes(self) -> None:
        try:
            notes = self.session.noise_notes()
        except (ValueError, KeyError) as e:
            notes = [str(e)]
        self.noise_label.setText("<br>".join(notes))

    # --- probe -------------------------------------------------------------------------

    def _clicked(self, ev) -> None:
        p = self.vb.mapSceneToView(ev.scenePos())
        x, y = int(math.floor(p.x())), int(math.floor(p.y()))
        cam = self.session.study.camera
        if 0 <= x < cam.width and 0 <= y < cam.height and self.xyz is not None:
            self.probe_at = (x, y)
            self._update_probe()
            self.tabs.setCurrentIndex(1)

    def probe(self, x: int, y: int) -> None:
        """Probe pixel (x, y), as a click there does."""
        self.probe_at = (x, y)
        self._update_probe()

    def _update_probe(self) -> None:
        x, y = self.probe_at
        try:
            p = self.session.probe(x, y)
        except MissingMaps:
            return
        self.marker.setData([x + 0.5], [y + 0.5])
        self.probe_curve.setData(p.wavelengths, p.radiance)
        self.probe_hi.setData(p.wavelengths, p.radiance + p.se)
        self.probe_lo.setData(p.wavelengths, np.maximum(0.0, p.radiance - p.se))
        rel = float(np.sum(p.se * p.radiance) / np.sum(p.radiance**2)) if p.radiance.any() else float("nan")
        self.probe_text.setText(
            f"pixel ({x}, {y}), {self._label()}<br>"
            f"luminance <b>{p.luminance_cd_m2:.4g} cd/m²</b> (at the study's flux and dimming)<br>"
            f"x, y = {p.xy[0]:.4f}, {p.xy[1]:.4f}; CCT {p.cct_k:.0f} K, Duv {p.duv:+.4f}<br>"
            f"band: ±1 standard error from the maps' passes (about {100 * rel:.1f} % of the spectrum)")

    # --- lamp sheet --------------------------------------------------------------------

    def _update_sheet(self) -> None:
        lamp_id = self.session.study.lamps[0].id
        c = self.session.choices[lamp_id]
        self.sheet_text.setPlainText(f"computing the lamp sheet of {c.lamp} level {c.level}…")
        self.sheet_future = (lamp_id, c, self.pool.submit(self._sheet, lamp_id, c))

    def _sheet(self, lamp_id: str, c: Choice):
        sheet = lampsheet.make(lamps.load(c.lamp), c.level)
        prof = self.session.profile(lamp_id, c)
        g = analysis_grid()
        zones = [(z.name or z.shape, z.spectrum.bin(g).values) for z in prof.zones]
        return sheet, zones

    def _show_sheet(self, result) -> None:
        sheet, zones = result
        self.sheet_text.setPlainText(lampsheet.format_text(sheet))
        self.sheet_plot.clear()
        g = analysis_grid()
        total = np.sum([v for _, v in zones], axis=0)
        peak = total.max() or 1.0
        self.sheet_plot.plot(g.centres, total / peak, pen=pg.mkPen("#444", width=2), name="all zones")
        if len(zones) > 1:
            for k, (name, v) in enumerate(zones):
                self.sheet_plot.plot(g.centres, v / peak, pen=pg.mkPen(pg.intColor(k, len(zones)), width=1), name=name)

    # --- renders -----------------------------------------------------------------------

    def render_missing(self) -> None:
        try:
            missing = self.session.missing()
        except (ValueError, KeyError) as e:
            self._badge("error", str(e))
            return
        if not missing:
            self._badge("ok", f"showing {self.shown}; every map is stored")
            return
        if self.client.proc is None:
            self.client.start()
        mjs = []
        for pl, shape, radiance in missing:
            mid = maps.map_id(pl.placement.id, shape)
            self.jobs_by_map[mid] = (pl.placement.id, shape)
            mjs.append(MapJob(pl.placement.id, shape, radiance.values, maps.noise_label(pl, shape)))
        self.client.submit(JobSpec(self.study_ref, tuple(mjs), threshold=self.threshold_box.value() / 100))
        self.tabs.setCurrentIndex(3)

    def _pause(self):
        if self.client.proc is not None:
            self.client.pause()

    def _resume(self):
        if self.client.proc is not None:
            self.client.resume()

    def _cancel(self):
        if self.client.proc is not None:
            self.client.cancel()

    def _restart(self):
        if self.client.proc is not None:
            self.client.restart()

    def _update_table(self) -> None:
        rows = list(self.client.status.values())
        self.table.setRowCount(len(rows))
        for r, s in enumerate(rows):
            noise = f"{100 * s.noise:.2f} %" if math.isfinite(s.noise) else "—"
            cells = [s.map, s.label, str(s.pass_index), str(s.spp_done), noise,
                     _eta(s.eta_s) if s.state in ("rendering", "stalled") else "—"]
            for col, text in enumerate(cells):
                self.table.setItem(r, col, QtWidgets.QTableWidgetItem(text))
            bar = self.table.cellWidget(r, 6) or QtWidgets.QProgressBar(maximum=1000)
            bar.setValue(int(1000 * s.bar))
            self.table.setCellWidget(r, 6, bar)

    # --- the clock ---------------------------------------------------------------------

    def _tick(self) -> None:
        for lamp_id, (choice, fut) in list(self.pending.items()):
            if fut.done():
                del self.pending[lamp_id]
                if fut.exception() is not None:
                    self._badge("error", f"{choice.lamp} level {choice.level}: {fut.exception()}")
                elif choice == self._choice_from_controls(lamp_id):
                    self._apply(lamp_id, choice)
        if self.sheet_future is not None and self.sheet_future[2].done():
            _, c, fut = self.sheet_future
            self.sheet_future = None
            if fut.exception() is not None:
                self.sheet_text.setPlainText(f"{c.lamp} level {c.level}: {fut.exception()}")
            else:
                self._show_sheet(fut.result())
        if self.client.proc is None and not self.client.status:
            return
        for e in self.client.poll():
            if e["type"] == "saved" and e["map"] in self.jobs_by_map:
                self.session.forget_map(*self.jobs_by_map[e["map"]])
                self.refresh()
            elif e["type"] == "state" and e.get("state") == "error":
                self._badge("error", "render worker error: " + (e.get("error") or "").strip().splitlines()[-1])
        state = self.client.worker_state()
        active = self.client.active()
        prefix = f"map {active.map}: {active.label}; " if active else ""
        self.worker_label.setText(f"{prefix}worker: {state}" + (f", {self.client.variant}" if self.client.variant else ""))
        self._update_table()

    def closeEvent(self, ev) -> None:
        self.timer.stop()
        self.client.close()
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(ev)


def run(study_ref: str, backend: str = "mitsuba") -> int:
    from lss.scene import study as studies

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    w = Viewer(Session(studies.load(study_ref)), study_ref, backend)
    w.show()
    return app.exec()
