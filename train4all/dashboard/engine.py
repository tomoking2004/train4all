"""
Self-contained HTML training dashboard for train4all.

Architecture
────────────
  DashboardConfig  - immutable settings (poll interval, filenames, …)
  Dashboard        - writes a static HTML shell once on initialize(), then
                     writes a small JSON data file on every update(). Browser
                     JavaScript polls that JSON and patches the DOM in place —
                     no page reloads, continuous animation, step-level refresh.

Design
──────
A laboratory chart recorder. Training curves are ink traces on engineering
graph paper — a two-tier grid behind every plot — and the live step-loss is a
strip chart whose pen head sits fixed at the right edge while the paper
scrolls beneath it. Light theme is the drafting room by day; dark is the
instrument room at night. IBM Plex Mono carries the wordmark and every number
and axis, engraved-style letterspaced capitals label the readouts, and IBM
Plex Sans carries prose. Matte panels and precise hairlines — no glass. A
vivid display spectrum (gradient wordmark, gauge ring, runline, and three
quiet ambient washes) carries the colour the data inks deliberately restrain.

A large progress gauge anchors the page — concentric rings inside a fine
machined tick bezel (outer = overall run, sweeping the display spectrum and
crowned gold once the run completes; inner = the live phase's steps, gold in the
gaps between phases and blank once the run ends) with the overall percentage at
its centre, epoch divider ticks, and a gold ★ best-epoch marker on its rim. Run
progress is strictly monotonic: every phase of the epoch advances it in
proportion to its share of the epoch's steps, it never rewinds across a phase or
epoch boundary, and it holds full once the run completes.

The gauge is flanked by a uniform KPI grid (current metric, best monitored
value, throughput, ETA, learning rate, and a GPU-memory cell whose bar turns
red near capacity) and the live step-loss graph (an auto-scaled trace of the
active phase's recent per-step loss). Instantaneous readings (current metric,
throughput, ETA, step graph) blank between steps; standing values (best
monitored value, learning rate, GPU memory) persist.

Below, every metric gets its own dependency-free SVG chart in a uniform
two-column grid, the primary (loss) metric first. All charts share the same
epoch-level history — until the first epoch completes they hold an "awaiting"
placeholder rather than a one-off per-step view. Each has best-epoch markers,
lowercase axis titles (``epoch`` · the metric name), a gold hover readout,
a log-scale toggle, and vector export; they render at their container's exact
pixel width (a ResizeObserver re-renders on reflow) and gridlines snap to nice
values — powers of ten on the log scale.

The run declares its phases, and every phase owns a fixed ink on a
blue→violet→magenta spectrum — train blue, validation violet, test magenta,
any further phase the next free hue — assigned once from that declared order,
so a phase keeps its ink in every curve, legend, badge, and gauge ring on the
page. The inks are validated for colour-vision deficiency (Machado
protan/deutan ΔE ≥ 12 between adjacent inks, in both themes). Red means offline
(a plateau keeps the training blue — the gold ★ carries that signal); gold is
reserved for excellence (best epoch, completed run). No green. A fixed hairline
across the top of the viewport mirrors overall progress in the same spectrum,
and turns gold once the run completes.

Configuration, environment, and model tables close the page — nested-dict
config opens indented sub-groups; click any row to copy its value (a gold flash
confirms). Light and dark themes are both first-class: the dashboard follows
the system preference, and a header toggle (or the ``T`` key) persists it.

Layout
────────────────────────────────────────────────────────────────────
   ── overall-progress hairline (fixed, top of viewport) ──
   header   ·  gradient wordmark · started / elapsed · pill · theme
   hero    loss · best   ┃   ◎ overall gauge   ┃
           it/s · ETA    ┃   phase·epoch·step  ┃   ▁▂▃▅ step loss
           lr · gpu      ┃                     ┃   (active phase)
   metric SVG charts — uniform two-column grid, loss first
   configuration · environment · model   (click to copy)
────────────────────────────────────────────────────────────────────
"""

import contextlib
import functools
import http.server
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

from train4all._version import __version__
from train4all.utils import TIMESTAMP_FORMAT, MetricTable

__all__ = ["Dashboard", "DashboardConfig", "PhaseSpec"]

# Number of recent per-step loss samples retained for the live step-loss graph.
# Sampled at the dashboard write cadence, so this spans roughly the last minute
# of a phase — a "recent activity" window that complements the full epoch-level
# history shown in the charts below.
_STEP_HISTORY = 96


def _json_safe(value: Any) -> Any:
    """Replace every non-finite float with ``None``, recursively.

    ``json.dumps`` happily writes bare ``NaN`` / ``Infinity``, which are not JSON: the
    browser's ``JSON.parse`` rejects the *whole* document, so a single divergent metric
    would blank the entire dashboard rather than just its own readout. And a metric can
    go non-finite while the loss stays finite — a 0/0 rate, an empty-class F1 — so the
    trainer's loss guard is no protection here. ``None`` is what the front end already
    renders as an absent reading ("—").
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DashboardConfig:
    """
    Appearance and behaviour settings for the live training dashboard.

    All fields carry sensible defaults; specify only what you need to change.
    Whether there is a dashboard at all is not settled here — that is the
    trainer's ``use_dashboard``, the one switch.

    Attributes:
        filename: HTML shell filename written inside ``run_dir``.
        data_filename: JSON data file polled by the browser on every tick.
        poll_interval_ms: Browser polling interval in milliseconds.
        open_on_start: Open in the system browser when
            :meth:`Dashboard.initialize` is called.
        stale_after_ms: Declare training *Offline* after this many ms without a
            heartbeat — an absolute timeout independent of ``poll_interval_ms``.
            Size it above your slowest synchronous pause (large saves, heavy
            plotting).
        use_server: Start a local HTTP server so the browser can ``fetch()`` the
            JSON data file — required for Chrome and Edge, which block
            cross-origin ``fetch()`` on ``file://`` pages. The server runs in a
            daemon thread and exits with the process.
    """
    filename: str = "dashboard.html"
    data_filename: str = "dashboard_data.json"
    poll_interval_ms: int = 500
    open_on_start: bool = True
    stale_after_ms: int = 30000
    use_server: bool = True


# ── Phase Schedule ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """
    One phase of an epoch, as the dashboard needs to see it.

    The flat, serializable projection of a trainer phase: the dashboard renders
    a schedule, not a training loop, so it takes the shape (name, gradients,
    length, cadence) and never the DataLoader or the metric function behind it.
    The list of these, in the order the phases run, is the dashboard's model of
    an epoch — it lays out the progress gauge, assigns the phase inks, and
    labels the badges from it alone.

    Attributes:
        name: Phase name, shown on the badge and in every chart legend.
        training: Whether the phase performs gradient updates. Drives the state
            accent, the pill, and the throughput readout.
        steps: Steps in the phase, so overall progress can advance in proportion
            to it. ``0`` when unknown (a length-less ``IterableDataset``), in
            which case the gauge falls back to weighting every phase of the
            epoch equally.
        every: The phase runs on epochs divisible by this. Epochs that skip it
            redistribute its share of the gauge across the phases that do run.
    """

    name: str
    training: bool = False
    steps: int = 0
    every: int = 1


# ── Dashboard ─────────────────────────────────────────────────────────────────

class Dashboard:
    """
    Live training dashboard backed by a JSON data file.

    Write the HTML shell once with :meth:`initialize`, then call :meth:`update`
    on every step or epoch to overwrite the small JSON data file. Browser-side
    JavaScript polls that file at the configured interval and patches the DOM
    in place — no page reloads, flicker-free live updates.

    If the training process exits without calling :meth:`finalize`, the
    ``last_update_ms`` field in the JSON lets the browser detect staleness and
    switch to the *Offline* state automatically.

    Args:
        config: Appearance and behaviour settings.
        run_dir: Directory where the HTML shell and JSON data file are written.
    """

    def __init__(self, config: DashboardConfig, run_dir: Path) -> None:
        self._config = config
        self._html_path = (run_dir / config.filename).resolve()
        self._data_path = (run_dir / config.data_filename).resolve()
        self._started_at: datetime | None = None
        self._status: str = "idle"
        self._trainer_config: dict[str, Any] = {}
        self._env_summary: dict[str, Any] = {}
        self._model_summary: dict[str, Any] = {}
        self._phases: list[PhaseSpec] = []
        self._monitor: str = "loss"
        self._monitor_phase: str = "val"
        self._server_port: int | None = None
        self._last_max_step: int = 0
        self._data_lock = threading.Lock()
        self._keepalive_stop = threading.Event()
        # The most recent full JSON payload, kept so the heartbeat can refresh
        # its timestamp without re-reading and re-parsing the (growing) file.
        self._last_payload: dict[str, Any] | None = None
        # Live telemetry surfaced beside the gauge. The step-loss buffer is a
        # rolling window of recent training-step losses; learning rate and GPU
        # memory are the latest readings. All are held on the instance so the
        # final ``finalize`` snapshot carries them without re-plumbing.
        self._step_loss: deque[float] = deque(maxlen=_STEP_HISTORY)
        self._step_nums: deque[int] = deque(maxlen=_STEP_HISTORY)
        self._step_phase: str = ""
        self._learning_rate: float | list[float] | None = None
        self._gpu_mem: tuple[float, float] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Absolute path to the HTML shell file."""
        return self._html_path

    @property
    def url(self) -> str:
        """URL for opening the dashboard — HTTP when the server is active, ``file://`` otherwise."""
        if self._server_port is not None:
            return f"http://127.0.0.1:{self._server_port}/{self._html_path.name}"
        return self._html_path.as_uri()

    @property
    def active(self) -> bool:
        """``True`` between :meth:`initialize` and :meth:`finalize` — i.e. while updates should be written."""
        return self._status == "training"

    @property
    def elapsed(self) -> timedelta | None:
        """Wall-clock time since :meth:`mark_started`, or ``None`` if not yet started."""
        return datetime.now() - self._started_at if self._started_at else None

    @property
    def poll_s(self) -> float:
        """Configured poll interval converted to seconds."""
        return self._config.poll_interval_ms / 1000

    def initialize(
        self,
        trainer_config: dict[str, Any],
        env_summary: dict[str, Any] | None = None,
        model_summary: dict[str, Any] | None = None,
        phases: list[PhaseSpec] | None = None,
        monitor: str = "loss",
        monitor_phase: str = "val",
    ) -> None:
        """
        Write the HTML shell and the first JSON snapshot.

        Must be called exactly once before any :meth:`update` or
        :meth:`finalize` call. Optionally starts an HTTP server and opens the
        dashboard in the system browser, depending on the
        :class:`DashboardConfig` settings.

        Args:
            trainer_config: Trainer hyperparameters shown in the Configuration panel.
            env_summary: System and runtime details shown in the Environment panel.
            model_summary: Registered model names and parameter counts shown in
                the Model panel.
            phases: The run's phases, in the order they run within an epoch (see
                :class:`PhaseSpec`). This is the dashboard's whole model of an
                epoch: it drives the progress gauge, the phase inks, and the
                state accents. Left empty, the gauge falls back to tracking only
                the phase currently reporting steps.
            monitor: Name of the metric tracked for the best-value KPI, used to
                label it (e.g. ``"accuracy"`` → "Best Val Accuracy").
            monitor_phase: Name of the phase that metric is read from — the other
                half of that label.
        """
        self._started_at = datetime.now()
        self._status = "training"
        self._trainer_config = trainer_config
        self._env_summary = env_summary or {}
        self._model_summary = model_summary or {}
        self._phases = list(phases or [])
        self._monitor = monitor
        self._monitor_phase = monitor_phase

        self._html_path.parent.mkdir(parents=True, exist_ok=True)
        html_content = (
            _HTML_SHELL
            .replace("__T4A_CSS__", _CSS)
            .replace("__T4A_POLL_MS__", str(self._config.poll_interval_ms))
            .replace("__T4A_DATA_FILE__", self._config.data_filename)
            .replace("__T4A_STALE_MS__", str(self._config.stale_after_ms))
            .replace("__T4A_VERSION__", __version__)
        )
        self._atomic_write(self._html_path, html_content)
        self._write_data(
            epoch=0, max_epoch=0, step=0, max_step=0,
            epoch_metrics={}, step_metrics=None, phase_name="",
            best_metric=float("inf"), best_epoch=None,
        )

        if self._config.use_server:
            self._start_http_server()

        if self._config.open_on_start:
            self.open_browser()

        self._start_keepalive()

    def mark_started(self, dt: datetime | None = None) -> None:
        """
        Reset the elapsed-time origin used by :attr:`elapsed`.

        Args:
            dt: New start time. Defaults to the current wall-clock time.
        """
        self._started_at = dt or datetime.now()

    def update(
        self,
        epoch: int,
        max_epoch: int,
        epoch_metrics: MetricTable | None = None,
        best_metric: float = float("inf"),
        best_epoch: int | None = None,
        *,
        epochs_no_improve: int = 0,
        is_gradient_phase: bool = False,
        step: int = 0,
        max_step: int = 0,
        step_metrics: dict[str, float] | None = None,
        phase_name: str = "",
        learning_rate: float | list[float] | None = None,
        gpu_mem: tuple[float, float] | None = None,
    ) -> None:
        """
        Overwrite the JSON data file with the latest training state.

        Call after each step for step-level granularity, or after each epoch
        for epoch-level updates. A keepalive thread refreshes the timestamp
        independently so the browser can distinguish a live process from a
        crashed one.

        Args:
            epoch: Current epoch number (1-based).
            max_epoch: Total number of training epochs.
            epoch_metrics: Accumulated per-epoch metrics keyed by metric then
                phase name.
            best_metric: Best monitored value recorded so far.
            best_epoch: Epoch that achieved ``best_metric``.
            epochs_no_improve: Consecutive epochs without an improvement.
            is_gradient_phase: Whether the active phase performs gradient updates.
            step: Current step within the active phase (1-based).
            max_step: Total number of steps in the active phase.
            step_metrics: Per-metric scalar values for the most recent step.
            phase_name: Name of the active phase — one of the names given to
                :meth:`initialize` (e.g. ``"train"``).
            learning_rate: Current optimizer learning rate(s), shown live beside
                the gauge — a single value, or a list of per-group rates rendered
                as a range. ``None`` leaves the readout blank.
            gpu_mem: ``(used_gb, total_gb)`` GPU memory for the live footprint
                bar. ``None`` hides the readout.
        """
        if max_step > 0:
            self._last_max_step = max_step
        if learning_rate is not None:
            self._learning_rate = learning_rate
        if gpu_mem is not None:
            self._gpu_mem = gpu_mem
        # Roll the step-loss window for the live step graph. It tracks the active
        # phase's steps — training or validation — and resets when the phase
        # changes, so the graph always shows the current phase's recent loss in
        # that phase's colour. The true step number is kept alongside each loss
        # so the axis reports real steps, not the (throttled) sample count. The
        # framework always records a ``loss`` entry.
        if phase_name and step_metrics:
            loss = step_metrics.get("loss")
            if isinstance(loss, (int, float)) and math.isfinite(loss):
                if phase_name != self._step_phase:
                    self._step_loss.clear()
                    self._step_nums.clear()
                    self._step_phase = phase_name
                self._step_loss.append(float(loss))
                self._step_nums.append(int(step))
        self._write_data(
            epoch, max_epoch, step, max_step,
            epoch_metrics or {}, step_metrics, phase_name,
            best_metric, best_epoch, epochs_no_improve, is_gradient_phase,
        )

    def heartbeat(self) -> None:
        """Refresh the liveness timestamp without changing the displayed data.

        Cheap and idempotent — a no-op until the first :meth:`update` and after
        :meth:`finalize`. Call it around long synchronous work (saving
        checkpoints, plotting) that would otherwise starve the keepalive thread
        and let the browser flag a live run as *Offline*.
        """
        if self.active:
            self._heartbeat()

    def finalize(
        self,
        epoch: int,
        max_epoch: int,
        epoch_metrics: MetricTable | None = None,
        best_metric: float = float("inf"),
        best_epoch: int | None = None,
        *,
        epochs_no_improve: int = 0,
    ) -> None:
        """
        Write the final JSON snapshot and embed all data inline in the HTML.

        Stops the keepalive thread, sets the training status to
        ``"completed"``, writes one last JSON update, then inlines all data
        into the HTML shell so the dashboard remains fully self-contained and
        viewable offline after the process exits.

        Args:
            epoch: Final epoch number reached.
            max_epoch: Total number of training epochs.
            epoch_metrics: All accumulated epoch metrics.
            best_metric: Best monitored value achieved.
            best_epoch: Epoch that achieved ``best_metric``.
            epochs_no_improve: Consecutive epochs without an improvement.
        """
        self._keepalive_stop.set()
        self._status = "completed"
        ms = self._last_max_step
        self._write_data(
            epoch, max_epoch, ms, ms,
            epoch_metrics or {}, None, "",
            best_metric, best_epoch, epochs_no_improve,
        )
        self._embed_data_in_html()

    def open_browser(self) -> None:
        """Open the dashboard, preferring the machine you are working *from*.

        Honours ``$BROWSER`` first. Editors set it when you develop on a remote
        host — VS Code Remote-SSH / Dev Containers / WSL, JetBrains Gateway —
        pointing it at a helper that opens the URL on your **local** machine and
        forwards the port, so the dashboard appears where you are sitting rather
        than on the headless remote. Falls back to the platform default browser,
        then to a quiet no-op: over a plain SSH session there is no display, so
        forward the printed port and open the URL manually
        (``ssh -L 8080:127.0.0.1:<printed-port> user@host``).
        """
        url = self.url
        # A plain helper/command path (the editor case) — run it with the URL
        # directly, which is reliable across platforms; anything fancier
        # (inline args, ``%s``) is left to webbrowser below.
        entry = os.environ.get("BROWSER", "").split(os.pathsep)[0].strip()
        if entry and "%s" not in entry:
            try:
                import subprocess
                subprocess.Popen([entry, url])
                return
            except Exception:
                pass
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    # ── Internals ─────────────────────────────────────────────────────────────

    def _write_data(
        self,
        epoch: int,
        max_epoch: int,
        step: int,
        max_step: int,
        epoch_metrics: MetricTable,
        step_metrics: dict[str, float] | None,
        phase_name: str,
        best_metric: float,
        best_epoch: int | None,
        epochs_no_improve: int = 0,
        is_gradient_phase: bool = False,
    ) -> None:
        el = self.elapsed
        data: dict[str, Any] = {
            "status":             self._status,
            "current_epoch":      epoch,
            "max_epoch":          max_epoch,
            "current_step":       step,
            "max_step":           max_step,
            "epoch_metrics":      epoch_metrics,
            "last_step_metrics":  step_metrics,
            "last_phase":         phase_name,
            "phases":             [asdict(p) for p in self._phases],
            "is_gradient_phase":  is_gradient_phase,
            "monitor":            self._monitor,
            "monitor_phase":      self._monitor_phase,
            # ``best_epoch is None`` is the single source of truth for "no best
            # yet" — this avoids depending on the ±inf sentinel, which differs
            # between min- and max-mode monitoring.
            "best_metric":        None if best_epoch is None else best_metric,
            "best_epoch":         best_epoch,
            "epochs_no_improve":  epochs_no_improve,
            "step_loss":          list(self._step_loss),
            "step_loss_phase":    self._step_phase,
            "step_loss_first":    self._step_nums[0] if self._step_nums else None,
            "step_loss_last":     self._step_nums[-1] if self._step_nums else None,
            "step_loss_cap":      _STEP_HISTORY,
            "learning_rate":      self._learning_rate,
            "gpu_mem_used":       self._gpu_mem[0] if self._gpu_mem else None,
            "gpu_mem_total":      self._gpu_mem[1] if self._gpu_mem else None,
            "config":             self._trainer_config,
            "env_summary":        self._env_summary,
            "model_summary":      self._model_summary,
            "started_at":         self._started_at.strftime(TIMESTAMP_FORMAT) if self._started_at else None,
            "elapsed":            str(el).split(".")[0] if el else None,
            "updated_at":         datetime.now().strftime(TIMESTAMP_FORMAT),
            # The poll interval is not repeated here: the browser reads it from the
            # shell's ``poll-ms`` meta tag, which is the one place it is declared.
            "last_update_ms":     int(time.time() * 1000),
        }
        # Sanitized once, at the single point the payload is serialized — so the cached
        # copy the heartbeat re-publishes is safe too, and holds a snapshot of the metric
        # tables rather than aliasing the trainer's live ones.
        safe = _json_safe(data)
        with self._data_lock:
            self._last_payload = safe
            self._atomic_write(self._data_path, json.dumps(safe))

    def _heartbeat(self) -> None:
        """Rewrite the cached payload with a fresh ``last_update_ms``."""
        with self._data_lock:
            if self._last_payload is None:
                return
            self._last_payload["last_update_ms"] = int(time.time() * 1000)
            self._atomic_write(self._data_path, json.dumps(self._last_payload))

    def _embed_data_in_html(self) -> None:
        """Rewrite the HTML to embed the final data inline for offline viewing.

        Inlines the training data as ``window.__TRAIN4ALL_DATA__``. The
        dashboard has no external JavaScript dependencies — charts are
        hand-rolled SVG — so the resulting file is fully self-contained
        without a network connection.
        """
        if not self._html_path.exists() or not self._data_path.exists():
            return
        html = self._html_path.read_text(encoding="utf-8")
        data_text = self._data_path.read_text(encoding="utf-8")

        html = html.replace(
            "</head>",
            f"<script>window.__TRAIN4ALL_DATA__={data_text};</script>\n</head>",
            1,
        )
        self._atomic_write(self._html_path, html)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write *text* to *path* atomically via a temp file + ``Path.replace``.

        Readers (the browser over HTTP, or the keepalive thread) therefore only
        ever observe a complete file — never a half-written one. Retries briefly
        on Windows ``PermissionError`` from antivirus / indexer file locks.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        for _ in range(10):
            try:
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(path)   # atomic rename — readers never see a half-written file
                return
            except PermissionError:
                time.sleep(0.05)
            except OSError:
                break
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)

    def _start_http_server(self) -> None:
        """Start a daemon-thread HTTP server in run_dir so the browser can fetch() the JSON."""
        class _Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass  # suppress per-request logging

            def end_headers(self) -> None:
                self.send_header("Cache-Control", "no-store")
                super().end_headers()

        handler = functools.partial(_Handler, directory=str(self._html_path.parent))
        try:
            # Port 0 lets the OS assign a free port atomically — no probe/bind race.
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        except OSError:
            return  # could not bind — fall back to the file:// URL
        self._server_port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def _start_keepalive(self) -> None:
        """Refresh the heartbeat every poll interval so the browser sees *Offline*
        only when the process actually dies; the thread stops at :meth:`finalize`.

        Sharing the GIL, it can be delayed by a long synchronous call on the main
        thread, so ``stale_after_ms`` absorbs such pauses and the trainer also
        pulses :meth:`heartbeat` directly around them.
        """
        def _run() -> None:
            while not self._keepalive_stop.wait(self.poll_s):
                self._heartbeat()

        threading.Thread(target=_run, daemon=True).start()


# ── Static Assets ─────────────────────────────────────────────────────────────


def _asset(name: str) -> str:
    """Read one of the dashboard's web assets from the file it is authored in.

    The stylesheet and the shell are CSS and HTML, not Python, and live as such:
    an editor highlights and checks them, a diff to them reads as a diff to a web
    page, and this module stays the size of the logic it actually holds. They ship
    as package data, so an installed train4all finds them beside this file.
    """
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


_CSS = _asset("style.css")
_HTML_SHELL = _asset("shell.html")
