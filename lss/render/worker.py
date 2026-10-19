"""The render worker (design §9.4, §9.7): a separate process that renders lighting maps
until their noise falls below a threshold, reporting honest progress.

It owns one loaded scene per study and keeps it between jobs, so the kernel compiles
once a process. Mitsuba's kernel is specialised to the spp (a new spp compiles again,
1.5–3.6 s on the lab, found while building §9.7), so the spp per pass is a power of two:
few kernels, each compiled once and then found in Dr.Jit's disk cache. Per map: calibration
passes at 16 and 64 spp, each three times (``CALIBRATION_SPP``; the first of each may
compile), the faster of the other two timing a whole pass (render, read back, accumulate)
as a fixed cost a plus a cost b per spp (the faster, since a GPU pass now and then takes
several times as long, and one such pass would set the spp for the whole map); then the power of two nearest the spp whose pass takes
``target_pass_s``, or whose rendering takes at least as long as a, if that is more, so a
pass is never mostly fixed cost (reading back and accumulating 512 × 384 × 47 values is
not free).
The spp stays fixed so the passes are independent and identically distributed, and Welford
over the pass means stays valid.

Messages to the parent (``out`` queue, each a dict with a ``type``):

- ``heartbeat`` every ``HEARTBEAT_S`` from a thread: the process is alive;
- ``state`` (compiling, rendering, paused, cancelled, error) for a map;
- ``progress`` after every pass: ``{study, map, map_index, maps_total, pass_index,
  spp_done, spp_per_pass, pass_s, elapsed_s, noise, threshold, eta_s, bar}``;
- ``saved`` when a map is stored (``converged`` or ``capped``);
- ``job_done`` when a job's maps are all handled.

Commands (``cmd`` queue): ``("job", JobSpec)``, ``("pause",)``, ``("resume",)``,
``("cancel",)``, ``("quit",)``; a job's commands are read between passes.

The noise is the stored maps' measure (Σ se(Y) / Σ Y of the map re-coloured under its own
zone's spectrum, §8.4), with each pixel's standard error corrected by c4(n), the
standard deviation's bias over n passes (8 % low at 4 passes): the ETA leans on it. The
ETA's rate is the mean pass after the first, which may compile: the mean, not the median,
since the GPU now and then takes 2–10 times as long over a pass (0.25–1 s against 0.1 s at
256 spp on the lab, found while building §9.7; not the worker's own work), and those
passes are part of the time a map takes.

The renderer is a backend: Mitsuba, or a numpy stand-in (``FakeBackend``) for the tests.
"""

import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

HEARTBEAT_S = 0.5
CALIBRATION_SPP = (16, 64)  # each rendered three times: the first may compile


@dataclass(frozen=True)
class MapJob:
    lamp_id: str
    shape: str
    radiance: np.ndarray  # the zone's radiance on the analysis grid (power per 1 nm bin)
    noise_lamp: str  # what the noise is measured under ("<lamp> level <n>[, <shape>]")


@dataclass(frozen=True)
class JobSpec:
    study: str  # study id or YAML path
    maps: tuple[MapJob, ...]
    threshold: float = 0.01
    min_passes: int = 8
    cap_s: float = 300.0  # per map, rendering time
    target_pass_s: float = 0.2
    max_spp_per_pass: int = 65536


# --- backends ----------------------------------------------------------------------------


class MitsubaBackend:
    def __init__(self, variant: str | None = None):
        from lss.render import env

        self.variant = env.setup(variant)
        self._study_key, self._scene, self._params, self._study = None, None, None, None

    def prepare(self, study) -> None:
        import mitsuba as mi

        from lss.render import mitsuba_scene as ms

        key = study.geometry_hash()
        if key != self._study_key:
            self._scene = mi.load_dict(ms.scene_dict(study))
            self._params = mi.traverse(self._scene)
            self._study_key, self._study = key, study

    def select(self, lamp_id: str, shape: str) -> None:
        from lss.render import mitsuba_scene as ms

        ms.set_emitting(self._params, self._study, lamp_id, shape)

    def render(self, spp: int, seed: int) -> np.ndarray:
        import mitsuba as mi

        from lss.render import mitsuba_scene as ms

        return ms.read_film(mi.render(self._scene, spp=spp, seed=seed), self._study.grid)


class FakeBackend:
    """A known map plus Gaussian noise of ``sigma``/√spp (relative, per pixel and bin), at
    ``s_per_spp`` seconds per spp; the first render also waits ``compile_s``. ``hang_after``
    passes it sleeps for good (the heartbeat goes on); ``crash_after`` passes it exits."""

    variant = "fake"

    def __init__(self, sigma: float = 0.5, s_per_spp: float = 2e-4, compile_s: float = 0.0,
                 hang_after: int | None = None, crash_after: int | None = None):
        self.sigma, self.s_per_spp, self.compile_s = sigma, s_per_spp, compile_s
        self.hang_after, self.crash_after = hang_after, crash_after
        self._renders, self._truth = 0, None

    def prepare(self, study) -> None:
        c, n = study.camera, len(study.grid)
        yy, xx = np.mgrid[0:c.height, 0:c.width]
        base = 0.2 + 0.8 * np.exp(-((xx - c.width / 2) ** 2 + (yy - c.height / 2) ** 2) / (0.1 * c.width**2))
        self._truth = base[..., None] * np.linspace(0.5, 1.5, n)[None, None, :]

    def select(self, lamp_id: str, shape: str) -> None:
        pass

    def render(self, spp: int, seed: int) -> np.ndarray:
        self._renders += 1
        if self._renders == 1:
            time.sleep(self.compile_s)
        if self.crash_after is not None and self._renders > self.crash_after:
            import os

            os._exit(3)
        if self.hang_after is not None and self._renders > self.hang_after:
            while True:
                time.sleep(1.0)
        time.sleep(spp * self.s_per_spp)
        rng = np.random.default_rng(seed)
        return self._truth * (1.0 + self.sigma / math.sqrt(spp) * rng.standard_normal(self._truth.shape))


def make_backend(name: str, options: dict | None = None):
    options = options or {}
    if name == "mitsuba":
        return MitsubaBackend(**options)
    if name == "fake":
        return FakeBackend(**options)
    raise ValueError(f"unknown render backend {name!r}; 'mitsuba' or 'fake'")


# --- progress arithmetic -----------------------------------------------------------------


def c4(n: int) -> float:
    """E[s]/σ for n samples from a normal distribution."""
    return math.sqrt(2.0 / (n - 1)) * math.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2))


def eta_s(noise: float, threshold: float, spp_done: int, s_per_spp: float, passes: int, min_passes: int,
          pass_s: float) -> float:
    """Seconds still needed: spp_done · ((noise/threshold)² − 1) spp at the rate so far, and
    at least the passes still owed to ``min_passes``."""
    if not math.isfinite(noise):
        return float("inf")
    more = spp_done * max(0.0, (noise / threshold) ** 2 - 1.0) * s_per_spp
    return max(more, max(0, min_passes - passes) * pass_s)


def pass_spp(spps, times, target_s: float, most: int) -> int:
    """The power of two nearest (in log) the spp whose pass takes ``target_s``, or whose
    rendering b·spp at least equals the fixed cost a, from two passes timed as a + b·spp;
    with a non-positive b (timing noise), the larger pass's rate and no fixed cost."""
    (n1, n2), (t1, t2) = spps, times
    b = (t2 - t1) / (n2 - n1)
    a = max(0.0, t1 - b * n1) if b > 0 else 0.0
    if b <= 0:
        b = t2 / n2
    spp = max(1.0, (target_s - a) / b, a / b)
    return int(min(most, 2 ** round(math.log2(spp))))


def bar(noise: float, threshold: float, passes: int = 0, min_passes: int = 0) -> float:
    """(threshold/noise)², which grows linearly in time since noise falls as 1/√spp; while
    passes are still owed to ``min_passes``, no more than their share."""
    if not math.isfinite(noise) or noise <= 0:
        return 0.0
    owed = passes / min_passes if min_passes else 1.0
    return min(1.0, (threshold / noise) ** 2, owed)


# --- the process -------------------------------------------------------------------------


class _Cancelled(Exception):
    pass


@dataclass
class _Control:
    cmd: object
    out: object
    paused: bool = False
    pending: list = field(default_factory=list)  # jobs that arrived while one ran

    def check(self, map_id: str) -> None:
        """Handle every command waiting between passes; while paused, wait here for more."""
        while True:
            try:
                msg = self.cmd.get(timeout=0.1) if self.paused else self.cmd.get_nowait()
            except queue.Empty:
                if self.paused:
                    continue
                return
            kind = msg[0]
            if kind == "pause":
                self.paused = True
                self.out.put({"type": "state", "map": map_id, "state": "paused"})
            elif kind == "resume":
                self.paused = False
                self.out.put({"type": "state", "map": map_id, "state": "rendering"})
            elif kind == "cancel":
                self.paused = False
                raise _Cancelled
            elif kind == "quit":
                raise SystemExit(0)
            elif kind == "job":
                self.pending.append(msg[1])


def _heartbeat(out, stop: threading.Event) -> None:
    import os

    pid = os.getpid()
    while not stop.is_set():
        out.put({"type": "heartbeat", "t": time.time(), "pid": pid})
        stop.wait(HEARTBEAT_S)


def _accumulate(acc, acc_y, m: np.ndarray, wy: np.ndarray) -> None:
    acc.add(m)
    acc_y.add(m @ wy)


def render_map(backend, study, job: JobSpec, k: int, mj: MapJob, control: _Control, out) -> None:
    from lss.render import maps, recolour
    from lss.spectral.grids import analysis_grid
    from lss.spectral.spectrum import BinnedSpectrum

    mid = maps.map_id(mj.lamp_id, mj.shape)
    base = {"study": job.study, "map": mid, "map_index": k, "maps_total": len(job.maps)}
    out.put({**base, "type": "state", "state": "compiling"})
    backend.select(mj.lamp_id, mj.shape)
    wy = recolour.luminance_weights(BinnedSpectrum(analysis_grid(), mj.radiance, 0.0, 0.0), study.grid)
    seed, times = 0, []
    cal, cal_y = maps.Welford(), maps.Welford()  # thrown away; after its first, an add costs what a pass's does
    for spp in CALIBRATION_SPP:
        timed = []
        for _ in range(3):
            t = time.perf_counter()
            _accumulate(cal, cal_y, backend.render(spp, seed), wy)
            seed += 1
            timed.append(time.perf_counter() - t)
            control.check(mid)
        times.append(min(timed[1:]))
    spp = pass_spp(CALIBRATION_SPP, times, job.target_pass_s, job.max_spp_per_pass)
    out.put({**base, "type": "state", "state": "rendering", "spp_per_pass": spp})
    acc, acc_y = maps.Welford(), maps.Welford()
    rendering_s, pass_times = 0.0, []
    while True:
        t = time.perf_counter()
        _accumulate(acc, acc_y, backend.render(spp, seed), wy)
        seed += 1
        pass_s = time.perf_counter() - t
        rendering_s += pass_s
        pass_times.append(pass_s)
        n = acc.n
        noise = (maps.luminance_noise(acc_y.standard_error(), acc_y.mean) / c4(n)) if n > 1 else float("inf")
        spp_done = n * spp
        typical = float(np.mean(pass_times[1:] if n > 1 else pass_times))  # the first may compile
        eta = eta_s(noise, job.threshold, spp_done, typical / spp, n, job.min_passes, typical)
        out.put({**base, "type": "progress", "pass_index": n, "spp_done": spp_done, "spp_per_pass": spp,
                 "pass_s": pass_s, "elapsed_s": rendering_s, "noise": noise, "threshold": job.threshold,
                 "eta_s": eta, "bar": bar(noise, job.threshold, n, job.min_passes), "t": time.time()})
        converged = n >= job.min_passes and noise <= job.threshold
        if converged or rendering_s >= job.cap_s:
            break
        control.check(mid)
    info = maps.MapInfo(mj.lamp_id, n, spp, backend.variant, round(rendering_s, 3), float(noise), mj.noise_lamp,
                        mj.shape)
    maps.save(study, maps.LightingMap(info, acc.mean, acc.m2))
    out.put({**base, "type": "saved", "state": "converged" if converged else "capped", "noise": float(noise),
             "passes": n, "spp_done": spp_done, "elapsed_s": rendering_s})


def run_job(backend, job: JobSpec, control: _Control, out) -> None:
    from lss.render import maps
    from lss.scene import study as studies

    study = studies.load(job.study)
    backend.prepare(study)
    for k, mj in enumerate(job.maps):
        mid = maps.map_id(mj.lamp_id, mj.shape)
        try:
            render_map(backend, study, job, k, mj, control, out)
        except _Cancelled:
            out.put({"type": "state", "study": job.study, "map": mid, "state": "cancelled"})
            for later in job.maps[k + 1:]:
                out.put({"type": "state", "study": job.study, "map": maps.map_id(later.lamp_id, later.shape),
                         "state": "cancelled"})
            break
    out.put({"type": "job_done", "study": job.study})


def main(cmd, out, backend_name: str = "mitsuba", backend_options: dict | None = None) -> None:
    """The worker process's entry point."""
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(out, stop), daemon=True).start()
    control = _Control(cmd, out)
    try:
        backend = make_backend(backend_name, backend_options)
        out.put({"type": "ready", "variant": backend.variant})
        while True:
            if control.pending:
                job = control.pending.pop(0)
            else:
                msg = cmd.get()
                if msg[0] == "quit":
                    break
                if msg[0] != "job":
                    continue  # pause/resume/cancel with nothing running
                job = msg[1]
            control.paused = False
            try:
                run_job(backend, job, control, out)
            except SystemExit:
                raise
            except Exception:
                out.put({"type": "state", "study": job.study, "map": None, "state": "error",
                         "error": traceback.format_exc()})
    except SystemExit:
        pass
    except Exception:
        out.put({"type": "state", "map": None, "state": "error", "error": traceback.format_exc()})
    finally:
        stop.set()
