"""The render worker seen from the viewer (design §9.4, §9.7): start it, send it jobs,
read its messages, and watch it. No Qt and no Mitsuba in this process.

Each map has a state: Queued · Compiling (before its first pass returns) · Rendering ·
Converged · Stopped at cap · Paused · Cancelled · Stalled · Error. The worker has its own:
starting (until its first heartbeat, for at most ``start_s``: importing Mitsuba takes
seconds, longer on a loaded machine), alive, or "Worker unresponsive" when its heartbeat
has been missing for ``heartbeat_timeout`` seconds or its process has exited. A map is "Stalled" when the
heartbeat goes on but no pass has come back for ``stall_passes`` pass times (or
``first_pass_s`` before its first, which includes compiling); it goes back to Rendering
when the next pass arrives. Restart kills the worker by the PID it was started with,
starts another and sends it again every map not yet stored.
"""

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, replace

from lss.render import worker
from lss.render.worker import JobSpec

DONE = ("converged", "capped", "cancelled")
LABELS = {"queued": "Queued", "compiling": "Compiling", "rendering": "Rendering", "converged": "Converged",
          "capped": "Stopped at cap", "paused": "Paused", "cancelled": "Cancelled", "stalled": "Stalled",
          "error": "Error"}


@dataclass
class MapStatus:
    map: str
    state: str = "queued"
    pass_index: int = 0
    spp_done: int = 0
    spp_per_pass: int = 0
    pass_s: float = 0.0
    elapsed_s: float = 0.0
    noise: float = float("inf")
    threshold: float = 0.0
    eta_s: float = float("inf")
    bar: float = 0.0
    since: float = 0.0  # when the current state began (monotonic)
    last_pass: float | None = None  # monotonic time of the last pass

    @property
    def label(self) -> str:
        return LABELS[self.state]


class RenderClient:
    def __init__(self, backend: str = "mitsuba", backend_options: dict | None = None,
                 heartbeat_timeout: float = 3.0, stall_passes: float = 10.0, first_pass_s: float = 180.0,
                 start_s: float = 120.0):
        self.backend, self.backend_options = backend, backend_options or {}
        self.heartbeat_timeout, self.stall_passes, self.first_pass_s = heartbeat_timeout, stall_passes, first_pass_s
        self.start_s = start_s
        self.started: float | None = None
        self.status: dict[str, MapStatus] = {}
        self.jobs: list[JobSpec] = []
        self.proc = None
        self.pid: int | None = None
        self.variant: str | None = None
        self.error: str | None = None
        self.last_heartbeat: float | None = None
        self.messages: list[dict] = []  # every message read, in order (for tests and logs)

    # --- the process -------------------------------------------------------------------

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self.cmd, self.out = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=worker.main, args=(self.cmd, self.out, self.backend, self.backend_options),
                                daemon=True, name="lss-render-worker")
        self.proc.start()
        self.pid = self.proc.pid
        self.started, self.last_heartbeat = time.monotonic(), None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def worker_state(self, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        if self.proc is None:
            return "not started"
        if not self.proc.is_alive():
            return f"Worker unresponsive (exited, code {self.proc.exitcode})"
        if self.last_heartbeat is None:
            if now - self.started > self.start_s:
                return f"Worker unresponsive (no heartbeat {now - self.started:.0f} s after starting)"
            return f"starting (pid {self.pid})"
        if now - self.last_heartbeat > self.heartbeat_timeout:
            return f"Worker unresponsive (no heartbeat for {now - self.last_heartbeat:.1f} s)"
        return f"alive (pid {self.pid})"

    def responsive(self, now: float | None = None) -> bool:
        """Alive or still starting: not (yet) unresponsive."""
        return not self.worker_state(now).startswith("Worker unresponsive")

    def kill(self) -> None:
        """Kill the worker by the PID it was started with (its own process handle)."""
        if self.proc is not None and self.proc.is_alive():
            self.proc.kill()
            self.proc.join(5)

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.is_alive():
            self.cmd.put(("quit",))
            self.proc.join(3)
        self.kill()
        self.proc = None

    def restart(self) -> None:
        """A new worker, sent every map not yet stored or cancelled."""
        self.kill()
        self.start()
        jobs, self.jobs = self.jobs, []
        for job in jobs:
            left = tuple(m for m in job.maps if self.status[self._id(m)].state not in DONE)
            if left:
                self.submit(replace(job, maps=left))

    # --- jobs --------------------------------------------------------------------------

    @staticmethod
    def _id(mj) -> str:
        from lss.render.maps import map_id

        return map_id(mj.lamp_id, mj.shape)

    def submit(self, job: JobSpec) -> None:
        now = time.monotonic()
        for mj in job.maps:
            self.status[self._id(mj)] = MapStatus(self._id(mj), threshold=job.threshold, since=now)
        self.jobs.append(job)
        self.cmd.put(("job", job))

    def pause(self) -> None:
        self.cmd.put(("pause",))

    def resume(self) -> None:
        self.cmd.put(("resume",))

    def cancel(self) -> None:
        self.cmd.put(("cancel",))

    def active(self) -> MapStatus | None:
        return next((s for s in self.status.values() if s.state not in DONE + ("queued", "error")), None)

    def busy(self) -> bool:
        return any(s.state not in DONE + ("error",) for s in self.status.values())

    # --- reading -----------------------------------------------------------------------

    def poll(self, now: float | None = None) -> list[dict]:
        """Read every waiting message, update the states, and return the messages that
        are not heartbeats."""
        events = []
        while True:
            try:
                msg = self.out.get_nowait()
            except (queue.Empty, OSError, ValueError, EOFError):
                break
            self.messages.append(msg)
            t = time.monotonic()
            kind = msg["type"]
            if kind == "heartbeat":
                self.last_heartbeat = t
                continue
            events.append(msg)
            if kind == "ready":
                self.variant = msg["variant"]
            s = self.status.get(msg.get("map")) if msg.get("map") else None
            if kind == "state" and msg["state"] == "error":
                self.error = msg.get("error")
                for st in self.status.values():
                    if st.state not in DONE:
                        st.state, st.since = "error", t
            elif s is None:
                continue
            elif kind == "state":
                s.state, s.since = msg["state"], t
                if msg["state"] == "rendering":
                    s.spp_per_pass = msg.get("spp_per_pass", s.spp_per_pass)
                    s.last_pass = t  # the stall clock starts with the main passes
            elif kind == "progress":
                for f in ("pass_index", "spp_done", "spp_per_pass", "pass_s", "elapsed_s", "noise", "eta_s", "bar"):
                    setattr(s, f, msg[f])
                if s.state in ("stalled", "compiling"):
                    s.state, s.since = "rendering", t
                s.last_pass = t
            elif kind == "saved":
                s.state, s.since, s.noise = msg["state"], t, msg["noise"]
                s.eta_s, s.bar = 0.0, 1.0 if msg["state"] == "converged" else s.bar
        self._watch(time.monotonic() if now is None else now)
        return events

    def _watch(self, now: float) -> None:
        if not self.worker_state(now).startswith("alive"):
            return  # the worker's state says it; a map cannot stall without a running worker
        for s in self.status.values():
            if s.state == "compiling" and now - s.since > self.first_pass_s:
                s.state, s.since = "stalled", now
            elif s.state == "rendering" and s.last_pass is not None:
                limit = self.stall_passes * s.pass_s if s.pass_s > 0 else self.first_pass_s
                if now - s.last_pass > limit:
                    s.state, s.since = "stalled", now
