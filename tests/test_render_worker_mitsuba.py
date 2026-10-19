"""The render worker with Mitsuba (design §9.7's predictions): the lab under CIE A to 1 %
noise, pass times, the heartbeat through compiling, an honest ETA. ``pytest -m render``."""

import time

import numpy as np
import pytest

pytestmark = pytest.mark.render

pytest.importorskip("mitsuba")

from lss.render import maps  # noqa: E402
from lss.render.client import RenderClient  # noqa: E402
from lss.render.lamps import place  # noqa: E402
from lss.render.worker import JobSpec, MapJob  # noqa: E402
from lss.scene import study as studies  # noqa: E402


def test_lab_converges_to_one_percent(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_CACHE", str(tmp_path))
    study = studies.load("lab")
    (pl,) = [place(p) for p in study.lamps]
    (shape, radiance), = [(s, r) for _, s, r in maps.jobs([pl])]
    c = RenderClient("mitsuba")
    c.start()
    try:
        c.submit(JobSpec("lab", (MapJob(pl.placement.id, shape, radiance.values, maps.noise_label(pl, shape)),),
                         threshold=0.01))
        t0 = time.monotonic()
        while c.busy() and time.monotonic() - t0 < 600:
            c.poll()
            time.sleep(0.02)
        c.poll()
    finally:
        c.close()
    s = c.status["lamp@isotropic"]
    passes = [m for m in c.messages if m["type"] == "progress"]
    beats = np.diff([m["t"] for m in c.messages if m["type"] == "heartbeat"])
    spp = passes[0]["spp_per_pass"]
    pass_s = [p["pass_s"] for p in passes]
    fourth = passes[3]
    eta_ratio = fourth["eta_s"] / (passes[-1]["t"] - fourth["t"])
    t, b = np.array([p["elapsed_s"] for p in passes[3:]]), np.array([p["bar"] for p in passes[3:]])
    r2 = np.corrcoef(t, b)[0, 1] ** 2
    print(f"\n{s.spp_done} spp in {len(passes)} passes of {spp}, {passes[-1]['elapsed_s']:.2f} s rendering; "
          f"first pass {pass_s[0]:.3f} s, others {min(pass_s[1:]):.3f}-{max(pass_s[1:]):.3f} s "
          f"(median {np.median(pass_s[1:]):.3f}); heartbeat gap max {beats.max():.2f} s; "
          f"ETA at pass 4 / actual {eta_ratio:.2f}; bar R² {r2:.3f}")
    assert s.state == "converged" and s.noise <= 0.01
    assert spp & (spp - 1) == 0, "a power of two"
    assert 2800 <= s.spp_done <= 4700
    assert beats.max() <= 1.5
    # The GPU now and then takes several times as long over a pass (not the worker's own work):
    # an ETA cannot foresee that, so ±30 % is asserted only when no later pass stalled.
    later = pass_s[4:]
    if max(later) <= 3 * np.median(pass_s[1:]):
        assert 0.7 <= eta_ratio <= 1.3
    else:
        print(f"a GPU stall after pass 4 ({max(later):.2f} s): ETA ratio {eta_ratio:.2f} not asserted")
    assert r2 >= 0.95
