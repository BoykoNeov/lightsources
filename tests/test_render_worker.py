"""The render worker and its watchdog (design §9.4, §9.7) against the numpy stand-in
renderer: convergence, an honest ETA and bar, pause, cancel, a crashed or hung worker and
restart. Real processes, no Mitsuba (the Mitsuba backend: ``-m render``)."""

import math
import os
import signal
import time
from importlib import resources

import numpy as np
import pytest
import yaml

from lss.render import maps, worker
from lss.render.client import RenderClient
from lss.render.worker import JobSpec, MapJob
from lss.scene import study as studies
from lss.spectral.grids import analysis_grid

# σ = 1 needs about 17 000 spp for 0.2 %, about 17 passes of 0.2 s.
FAKE = {"sigma": 1.0, "s_per_spp": 2e-4}
THRESHOLD = 0.002


@pytest.fixture
def study_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_CACHE", str(tmp_path / "cache"))
    doc = yaml.safe_load(resources.files("lss.data").joinpath("studies", "lab.yaml").read_text(encoding="utf-8"))
    doc["camera"].update(width=64, height=48)
    doc["id"] = "lab_tiny"
    path = tmp_path / "lab_tiny.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(path)


def job(study_path, shapes=("isotropic",), **kw) -> JobSpec:
    flat = np.ones(len(analysis_grid()))
    return JobSpec(study_path, tuple(MapJob("lamp", s, flat, "test lamp") for s in shapes), **{"threshold": THRESHOLD, **kw})


def run_until(c: RenderClient, done, timeout=300.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        c.poll()
        if done(c):
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out; worker {c.worker_state()}, maps {[(s.map, s.state) for s in c.status.values()]}")


@pytest.fixture
def client():
    clients = []

    def make(**options):
        c = RenderClient("fake", {**FAKE, **options})
        c.start()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.close()


def progress(c):
    return [m for m in c.messages if m["type"] == "progress"]


def test_pass_spp_is_a_power_of_two_near_the_target():
    # a = 0.1 s, b = 1 ms per spp: 0.2 s at 100 spp, but rendering ≥ a needs 100 too → 128
    assert worker.pass_spp((16, 64), (0.116, 0.164), 0.2, 65536) == 128
    # no fixed cost, 0.5 ms per spp: 400 spp → 512 (log-nearest)
    assert worker.pass_spp((16, 64), (0.008, 0.032), 0.2, 65536) == 512
    # timing noise (b ≤ 0): the larger pass's rate
    assert worker.pass_spp((16, 64), (0.05, 0.04), 0.2, 65536) == 256
    assert worker.pass_spp((16, 64), (0.008, 0.032), 0.2, 64) == 64


def test_c4_and_eta():
    assert worker.c4(2) == pytest.approx(math.sqrt(2 / math.pi))
    assert worker.c4(4) == pytest.approx(0.9213, abs=1e-4)
    # noise twice the threshold: three times the spp done still to go
    assert worker.eta_s(0.02, 0.01, 1000, 1e-3, 10, 8, 0.1) == pytest.approx(3.0)
    # converged but passes owed
    assert worker.eta_s(0.005, 0.01, 1000, 1e-3, 5, 8, 0.1) == pytest.approx(0.3)
    assert worker.bar(0.02, 0.01) == pytest.approx(0.25)
    assert worker.bar(0.005, 0.01, 4, 8) == pytest.approx(0.5)


def test_converges_and_stores_the_map(study_path, client):
    c = client()
    c.submit(job(study_path))
    run_until(c, lambda c: not c.busy())
    s = c.status["lamp@isotropic"]
    assert s.state == "converged" and s.noise <= THRESHOLD
    m = maps.load(studies.load(study_path), "lamp", "isotropic", mmap=False)
    assert m.info.passes >= 8 and m.info.noise_under_study_lamp <= THRESHOLD and m.info.variant == "fake"
    assert m.info.noise_lamp == "test lamp"
    passes = progress(c)
    assert len({p["spp_per_pass"] for p in passes}) == 1
    assert all(0.1 <= p["pass_s"] <= 0.4 for p in passes[1:])


def test_eta_is_honest_and_the_bar_linear(study_path, client):
    c = client()
    c.submit(job(study_path))
    run_until(c, lambda c: not c.busy())
    passes = progress(c)
    saved = next(m for m in c.messages if m["type"] == "saved")
    assert len(passes) >= 12, "the noise, not the minimum passes, must decide"
    fourth = passes[3]
    actual = passes[-1]["t"] - fourth["t"]
    assert fourth["eta_s"] == pytest.approx(actual, rel=0.3)
    t = np.array([p["elapsed_s"] for p in passes[3:]])
    b = np.array([p["bar"] for p in passes[3:]])
    r = np.corrcoef(t, b)[0, 1]
    assert r * r >= 0.95
    assert saved["state"] == "converged"


def test_pause_resume_and_cancel(study_path, client):
    c = client()
    c.submit(job(study_path, shapes=("isotropic", "polar 0-30 deg")))
    run_until(c, lambda c: len(progress(c)) >= 3)
    c.pause()
    run_until(c, lambda c: c.status["lamp@isotropic"].state == "paused")
    n = len(progress(c))
    t = time.monotonic()
    while time.monotonic() - t < 1.0:
        c.poll()
        time.sleep(0.05)
    assert len(progress(c)) == n, "no passes while paused"
    assert c.responsive()
    c.resume()
    run_until(c, lambda c: len(progress(c)) > n)
    t_cancel = time.monotonic()
    c.cancel()
    run_until(c, lambda c: c.status["lamp@isotropic"].state == "cancelled")
    assert time.monotonic() - t_cancel <= 0.5
    run_until(c, lambda c: c.status["lamp@polar-0-30"].state == "cancelled")
    assert maps.load(studies.load(study_path), "lamp", "isotropic") is None


def test_a_crashed_worker_is_seen_and_restarted(study_path, client):
    c = client(crash_after=8)  # 6 calibration passes, then 2 main ones
    c.submit(job(study_path))
    run_until(c, lambda c: not c.responsive())
    assert c.worker_state().startswith("Worker unresponsive (exited")
    c.backend_options = dict(FAKE)
    c.restart()
    run_until(c, lambda c: not c.busy())
    assert c.status["lamp@isotropic"].state == "converged"


def test_a_killed_worker_is_seen_within_the_heartbeat_timeout(study_path, client):
    c = client()
    c.submit(job(study_path))
    run_until(c, lambda c: len(progress(c)) >= 2)
    t = time.monotonic()
    os.kill(c.pid, signal.SIGTERM)  # its own PID, as a hang that never recovers would end
    run_until(c, lambda c: not c.responsive(), timeout=10)
    assert time.monotonic() - t <= 3.5


def test_a_missing_heartbeat_reads_unresponsive(study_path, client):
    c = client()
    assert c.worker_state().startswith("starting") and c.responsive()
    assert c.worker_state(time.monotonic() + 121).startswith("Worker unresponsive (no heartbeat 12")
    run_until(c, lambda c: c.variant == "fake")
    assert c.worker_state().startswith("alive")
    assert c.worker_state(time.monotonic() + 3.1).startswith("Worker unresponsive (no heartbeat for")


def test_a_hung_render_reads_stalled(study_path, client):
    c = client(hang_after=8)  # 6 calibration passes, then 2 main ones
    c.submit(job(study_path))
    run_until(c, lambda c: c.status["lamp@isotropic"].state == "stalled")
    assert c.responsive(), "the heartbeat goes on: stalled, not unresponsive"
    s = c.status["lamp@isotropic"]
    assert s.pass_index == 2
