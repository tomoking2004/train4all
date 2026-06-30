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
A quiet, Apple-like instrument panel: one centred column, frosted-glass cards
over soft ambient colour, hairline rules, and a single state-driven accent.

A large progress gauge anchors the page — concentric rings (outer = overall
run, gold once the run completes; inner = the live phase's steps, gold in the
gaps between phases and blank once the run ends) with the overall percentage at
its centre, epoch divider ticks, and a gold ★ best-epoch marker on its rim. Run
progress is strictly monotonic: train and validation steps both advance it
proportionally, it never rewinds across a phase or epoch boundary, and it holds
full once the run completes.

The gauge is flanked by a uniform KPI grid (current metric, best monitored
value, throughput, ETA, learning rate, and a GPU-memory cell whose bar turns
red near capacity) and the live step-loss graph (an auto-scaled trace of the
active phase's recent per-step loss). Instantaneous readings (current metric,
throughput, ETA, step graph) blank between steps; standing values (best
monitored value, learning rate, GPU memory) persist.

Below, every metric gets its own zero-dependency SVG chart in a uniform
two-column grid, the primary (loss) metric first. All charts share the same
epoch-level history — until the first epoch completes they hold an "awaiting"
placeholder rather than a one-off per-step view. Each has best-epoch markers,
lowercase axis titles (``epoch`` · the metric name), a gold hover readout,
a log-scale toggle, and vector export; they render at their container's exact
pixel width (a ResizeObserver re-renders on reflow) and gridlines snap to nice
values — powers of ten on the log scale.

Every phase owns a fixed hue on a blue→violet→red spectrum — train blue,
validation purple, test pink — so curves, legends, the phase badge, the inner
gauge ring, and the state accents always agree. Red means offline (a plateau
keeps the training blue — the gold ★ carries that signal); gold is reserved for
excellence (best epoch, completed run). No green. A fixed hairline across the
top of the viewport mirrors overall progress in the same spectrum, and turns
gold once the run completes.

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

from __future__ import annotations

import functools
import http.server
import importlib.metadata
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    _VERSION = importlib.metadata.version("train4all")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "unknown"

from train4all.utils.dict_utils import MetricTable

__all__ = ["Dashboard", "DashboardConfig"]

# Number of recent per-step loss samples retained for the live step-loss graph.
# Sampled at the dashboard write cadence, so this spans roughly the last minute
# of a phase — a "recent activity" window that complements the full epoch-level
# history shown in the charts below.
_STEP_HISTORY = 96


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DashboardConfig:
    """Appearance and behaviour settings for the live training dashboard.

    All fields carry sensible defaults; specify only what you need to change.

    Attributes:
        enabled:          Master switch — ``False`` disables the dashboard
                          entirely regardless of all other settings.
        filename:         HTML shell filename written inside ``run_dir``.
        data_filename:    JSON data file polled by the browser on every tick.
        poll_interval_ms: Browser polling interval in milliseconds.
        open_on_start:    Open in the system browser when
                          :meth:`Dashboard.initialize` is called.
        stale_after_ms:   Declare training *Offline* after this many ms without
                          a heartbeat — an absolute timeout independent of
                          ``poll_interval_ms``. Size it above your slowest
                          synchronous pause (large saves, heavy plotting).
        use_server:       Start a local HTTP server so the browser can
                          ``fetch()`` the JSON data file — required for
                          Chrome and Edge, which block cross-origin
                          ``fetch()`` on ``file://`` pages. The server runs
                          in a daemon thread and exits with the process.
    """
    enabled: bool = True
    filename: str = "dashboard.html"
    data_filename: str = "dashboard_data.json"
    poll_interval_ms: int = 500
    open_on_start: bool = True
    stale_after_ms: int = 30000
    use_server: bool = True


# ── Dashboard ─────────────────────────────────────────────────────────────────

class Dashboard:
    """Live training dashboard backed by a JSON data file.

    Write the HTML shell once with :meth:`initialize`, then call :meth:`update`
    on every step or epoch to overwrite the small JSON data file. Browser-side
    JavaScript polls that file at the configured interval and patches the DOM
    in place — no page reloads, flicker-free live updates.

    If the training process exits without calling :meth:`finalize`, the
    ``last_update_ms`` field in the JSON lets the browser detect staleness and
    switch to the *Offline* state automatically.

    Args:
        config:  Appearance and behaviour settings.
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
        self._training_phases: list[str] = ["train"]
        self._monitor: str = "loss"
        self._train_steps: int = 0
        self._val_steps: int = 0
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
        self._learning_rate: float | None = None
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
        training_phases: list[str] | None = None,
        monitor: str = "loss",
        train_steps: int = 0,
        val_steps: int = 0,
    ) -> None:
        """Write the HTML shell and the first JSON snapshot.

        Must be called exactly once before any :meth:`update` or
        :meth:`finalize` call. Optionally starts an HTTP server and opens the
        dashboard in the system browser, depending on the
        :class:`DashboardConfig` settings.

        Args:
            trainer_config: Trainer hyperparameters shown in the Configuration panel.
            env_summary: System and runtime details shown in the Environment panel.
            model_summary: Registered model names and parameter counts shown in
                the Model panel.
            training_phases: Phase names that trigger gradient updates, used to
                drive the state-dependent accent, gauge, and captions correctly.
            monitor: Name of the validation metric tracked for the best-value KPI,
                used to label it (e.g. ``"accuracy"`` → "Best Val Accuracy").
            train_steps: Steps per training phase, used to make overall progress
                advance proportionally and monotonically. ``0`` when unknown.
            val_steps: Steps per validation phase. ``0`` when there is no
                validation pass or its length is unknown.
        """
        self._started_at = datetime.now()
        self._status = "training"
        self._trainer_config = trainer_config
        self._env_summary = env_summary or {}
        self._model_summary = model_summary or {}
        if training_phases is not None:
            self._training_phases = training_phases
        self._monitor = monitor
        self._train_steps = train_steps
        self._val_steps = val_steps

        self._html_path.parent.mkdir(parents=True, exist_ok=True)
        html_content = (
            _HTML_SHELL
            .replace("__T4A_CSS__", _CSS)
            .replace("__T4A_POLL_MS__", str(self._config.poll_interval_ms))
            .replace("__T4A_DATA_FILE__", self._config.data_filename)
            .replace("__T4A_STALE_MS__", str(self._config.stale_after_ms))
            .replace("__T4A_VERSION__", _VERSION)
        )
        self._atomic_write(self._html_path, html_content)
        self._write_data(0, 0, 0, 0, {}, None, "", float("inf"), None)

        if self._config.use_server:
            self._start_http_server()

        if self._config.open_on_start:
            self._open_browser()

        self._start_keepalive()

    def mark_started(self, dt: datetime | None = None) -> None:
        """Reset the elapsed-time origin used by :attr:`elapsed`.

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
        phase: str = "",
        learning_rate: float | list[float] | None = None,
        gpu_mem: tuple[float, float] | None = None,
    ) -> None:
        """Overwrite the JSON data file with the latest training state.

        Call after each step for step-level granularity, or after each epoch
        for epoch-level updates. A keepalive thread refreshes the timestamp
        independently so the browser can distinguish a live process from a
        crashed one.

        Args:
            epoch:             Current epoch number (1-based).
            max_epoch:         Total number of training epochs.
            epoch_metrics:     Accumulated per-epoch metrics keyed by metric then phase.
            best_metric:       Best monitored validation value recorded so far.
            best_epoch:        Epoch that achieved ``best_metric``.
            epochs_no_improve: Consecutive epochs without an improvement.
            is_gradient_phase: Whether the active phase performs gradient updates.
            step:              Current step within the epoch (1-based).
            max_step:          Total number of steps in the epoch.
            step_metrics:      Per-metric scalar values for the most recent step.
            phase:             Name of the active phase (e.g. ``"train"``).
            learning_rate:     Current optimizer learning rate(s), shown live
                               beside the gauge — a single value, or a list of
                               per-group rates rendered as a range. ``None``
                               leaves the readout blank.
            gpu_mem:           ``(used_gb, total_gb)`` GPU memory for the live
                               footprint bar. ``None`` hides the readout.
        """
        if not self._config.enabled:
            return
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
        if phase and step_metrics:
            loss = step_metrics.get("loss")
            if isinstance(loss, (int, float)) and loss == loss and abs(loss) != float("inf"):
                if phase != self._step_phase:
                    self._step_loss.clear()
                    self._step_nums.clear()
                    self._step_phase = phase
                self._step_loss.append(float(loss))
                self._step_nums.append(int(step))
        self._write_data(
            epoch, max_epoch, step, max_step,
            epoch_metrics or {}, step_metrics, phase,
            best_metric, best_epoch, epochs_no_improve, is_gradient_phase,
        )

    def finalize(
        self,
        epoch: int,
        max_epoch: int,
        epoch_metrics: MetricTable | None = None,
        best_metric: float = float("inf"),
        best_epoch: int | None = None,
        epochs_no_improve: int = 0,
    ) -> None:
        """Write the final JSON snapshot and embed all data inline in the HTML.

        Stops the keepalive thread, sets the training status to
        ``"completed"``, writes one last JSON update, then inlines all data
        into the HTML shell so the dashboard remains fully self-contained and
        viewable offline after the process exits.

        Args:
            epoch:             Final epoch number reached.
            max_epoch:         Total number of training epochs.
            epoch_metrics:     All accumulated epoch metrics.
            best_metric:       Best monitored validation value achieved.
            best_epoch:        Epoch that achieved ``best_metric``.
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
        """Open the HTML shell in the system's default browser."""
        self._open_browser()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _write_data(
        self,
        epoch: int,
        max_epoch: int,
        step: int,
        max_step: int,
        epoch_metrics: MetricTable,
        step_metrics: dict[str, float] | None,
        phase: str,
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
            "train_steps":        self._train_steps,
            "val_steps":          self._val_steps,
            "epoch_metrics":      epoch_metrics,
            "last_step_metrics":  step_metrics,
            "last_phase":         phase,
            "training_phases":    self._training_phases,
            "is_gradient_phase":  is_gradient_phase,
            "monitor":            self._monitor,
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
            "learning_rate":      self._learning_rate,
            "gpu_mem_used":       self._gpu_mem[0] if self._gpu_mem else None,
            "gpu_mem_total":      self._gpu_mem[1] if self._gpu_mem else None,
            "config":             self._trainer_config,
            "env_summary":        self._env_summary,
            "model_summary":      self._model_summary,
            "started_at":         self._started_at.strftime("%Y-%m-%d %H:%M:%S") if self._started_at else None,
            "elapsed":            str(el).split(".")[0] if el else None,
            "updated_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_update_ms":     int(time.time() * 1000),
            "poll_interval_ms":   self._config.poll_interval_ms,
        }
        with self._data_lock:
            self._last_payload = data
            self._atomic_write(self._data_path, json.dumps(data))

    def heartbeat(self) -> None:
        """Refresh the liveness timestamp without changing the displayed data.

        Cheap and idempotent — a no-op until the first :meth:`update` and after
        :meth:`finalize`. Call it around long synchronous work (saving
        checkpoints, plotting) that would otherwise starve the keepalive thread
        and let the browser flag a live run as *Offline*.
        """
        if self.active:
            self._heartbeat()

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
        """Write *text* to *path* atomically via a temp file + ``os.replace``.

        Readers (the browser over HTTP, or the keepalive thread) therefore only
        ever observe a complete file — never a half-written one. Retries briefly
        on Windows ``PermissionError`` from antivirus / indexer file locks.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        for _ in range(10):
            try:
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.05)
            except OSError:
                break
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    def _open_browser(self) -> None:
        """Open the dashboard, preferring the machine you are working *from*.

        Honors ``$BROWSER`` first. Editors set it when you develop on a remote
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
        interval = self._config.poll_interval_ms / 1000

        def _run() -> None:
            while not self._keepalive_stop.wait(interval):
                self._heartbeat()

        threading.Thread(target=_run, daemon=True).start()


# ── Static assets ─────────────────────────────────────────────────────────────

_CSS = r"""
/* ── Theme tokens ──────────────────────────────────────────── */
:root, [data-theme="dark"] {
  --bg:        #0a0b0d;
  --text:      #e9ebef;
  --dim:       #9aa1ac;
  --faint:     #5c6270;
  --line:      rgba(255, 255, 255, 0.07);
  --line-2:    rgba(255, 255, 255, 0.16);
  --hover:     rgba(255, 255, 255, 0.04);
  --tip-bg:    #15171c;
  --thumb:     rgba(255, 255, 255, 0.15);
  --shadow:    0 10px 32px rgba(0, 0, 0, 0.45);
  --gold:      #e3bd6a;
  --good:      #5e8bff;
  --bad:       #f25c6e;
  --st-training:   #5e8bff;
  --st-validating: #b262f4;
  --st-stagnant:   #5e8bff;  /* plateau keeps the training blue — gold ★ already tells the story */
  --st-completed:  #e3bd6a;
  --st-idle:       #8a90a0;
  --st-stopped:    #f25c6e;
  --glass:         rgba(17, 19, 25, 0.55);
  --glass-edge:    rgba(255, 255, 255, 0.06);
  --card-shadow:   0 10px 30px rgba(0, 0, 0, 0.28);
  --c1: #5e8bff; --c2: #7a7bff; --c3: #976dfd; --c4: #b262f4;
  --c5: #cc58e3; --c6: #e150c4; --c7: #ee559b; --c8: #f25c6e;
  --spectrum: linear-gradient(90deg, #5e8bff, #7a7bff, #976dfd, #b262f4, #cc58e3, #e150c4, #ee559b, #f25c6e);
  color-scheme: dark;
}
[data-theme="light"] {
  --bg:        #faf9f5;
  --text:      #1c1e24;
  --dim:       #5b6170;
  --faint:     #989eab;
  --line:      rgba(22, 24, 31, 0.09);
  --line-2:    rgba(22, 24, 31, 0.2);
  --hover:     rgba(22, 24, 31, 0.045);
  --tip-bg:    #ffffff;
  --thumb:     rgba(22, 24, 31, 0.2);
  --shadow:    0 10px 32px rgba(20, 22, 30, 0.12);
  --gold:      #9c7520;
  --good:      #3a63dd;
  --bad:       #cc3f56;
  --st-training:   #3a63dd;
  --st-validating: #863cc6;
  --st-stagnant:   #3a63dd;
  --st-completed:  #9c7520;
  --st-idle:       #6e7480;
  --st-stopped:    #cc3f56;
  --glass:         rgba(255, 255, 255, 0.62);
  --glass-edge:    rgba(255, 255, 255, 0.85);
  --card-shadow:   0 10px 30px rgba(22, 24, 31, 0.08);
  --c1: #3a63dd; --c2: #5256d8; --c3: #6c48d0; --c4: #863cc6;
  --c5: #a033b6; --c6: #b92e9c; --c7: #c63577; --c8: #cc3f56;
  --spectrum: linear-gradient(90deg, #3a63dd, #5256d8, #6c48d0, #863cc6, #a033b6, #b92e9c, #c63577, #cc3f56);
  color-scheme: light;
}
:root {
  --accent: var(--st-idle);
  --phasecol: var(--st-idle);
  --mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  --sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --ease: cubic-bezier(0.22, 0.61, 0.36, 1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
/* a touch larger than the browser default — the whole UI scales from here */
html { font-size: 17.5px; scrollbar-width: thin; scrollbar-color: var(--thumb) transparent; }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: var(--thumb); border-radius: 100px; }
::selection { background: color-mix(in srgb, var(--accent) 28%, transparent); }

body {
  font-family: var(--sans); color: var(--text); background: var(--bg);
  min-height: 100vh; line-height: 1.5; letter-spacing: -0.006em;
  font-feature-settings: 'cv05';
  -webkit-font-smoothing: antialiased;
  transition: background 0.3s var(--ease), color 0.3s var(--ease);
}
/* ambient washes — three quiet pools of spectrum colour that the
   frosted-glass panels pick up and blur */
body::before {
  content: ''; position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background:
    radial-gradient(44% 38% at 10% 6%,  color-mix(in srgb, var(--c1) 11%, transparent), transparent 70%),
    radial-gradient(38% 32% at 90% 10%, color-mix(in srgb, var(--c3) 9%, transparent), transparent 70%),
    radial-gradient(42% 36% at 78% 96%, color-mix(in srgb, var(--c8) 7%, transparent), transparent 72%);
}

/* overall run progress — a fixed hairline across the top of the viewport.
   The eight-colour spectrum is anchored to the viewport, so the line reveals
   more of the gradient as the run advances; it turns gold on completion to
   match the gauge's outer ring. */
.runline { position: fixed; top: 0; left: 0; height: 2px; width: 0%;
  background-image: var(--spectrum); background-size: 100vw 100%; background-repeat: no-repeat;
  z-index: 100; transition: width 0.6s var(--ease); }
.runline.done { background-image: none; background-color: var(--st-completed); }

/* fills the viewport on any display — only a hairline-thin breathing margin */
.app { max-width: none; margin: 0 auto; padding: 0 clamp(20px, 3vw, 56px) 64px;
  animation: rise 0.55s var(--ease) both; }
@keyframes rise { from { opacity: 0; transform: translateY(8px); } }

/* ── Header — the wordmark anchors the page ────────────────── */
.top { display: flex; align-items: center; gap: 18px; padding: 28px 0 24px; flex-wrap: wrap; }
.brand { display: flex; flex-direction: column; gap: 4px; }
.brand-link { display: inline-flex; align-items: center; gap: 9px; width: fit-content;
  color: inherit; text-decoration: none; }
/* gradient wordmark — blue → purple → pink, kept restrained */
.brand-name { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1;
  background-image: linear-gradient(100deg, var(--c1), var(--c4) 50%, var(--c7));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.gh { width: 19px; height: 19px; margin-top: 1px; }
.gh path { fill: var(--faint); transition: fill 0.2s var(--ease); }
.gs1 { stop-color: var(--c1); } .gs2 { stop-color: var(--c4); } .gs3 { stop-color: var(--c7); }
.brand-link:hover .gh path { fill: url(#ghGrad); }
.brand-cap { font-size: 0.7rem; font-weight: 550; letter-spacing: 0.01em; color: var(--faint); }
.top-meta { display: flex; gap: 22px; margin-left: auto; flex-wrap: wrap; }
.top-meta span { display: inline-flex; align-items: baseline; gap: 7px; }
.top-meta .k { font-size: 0.68rem; font-weight: 550; letter-spacing: 0.01em; color: var(--faint); }
.top-meta b { color: var(--dim); font-weight: 500; font-family: var(--mono); font-size: 0.78rem;
  font-variant-numeric: tabular-nums; }
.pill { display: inline-flex; align-items: center; gap: 7px; padding: 6px 13px; border-radius: 100px;
  white-space: nowrap; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.01em;
  color: var(--accent); background: color-mix(in srgb, var(--accent) 9%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 26%, transparent);
  transition: color 0.5s var(--ease), background 0.5s var(--ease), border-color 0.5s var(--ease); }
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pill.live .dot { animation: pulse 1.8s var(--ease) infinite; }
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 45%, transparent); }
  70%  { box-shadow: 0 0 0 5px transparent; }
  100% { box-shadow: 0 0 0 0 transparent; }
}
.tbtn { width: 32px; height: 32px; display: inline-flex; align-items: center; justify-content: center;
  border-radius: 9px; border: 1px solid transparent; background: none; color: var(--faint); cursor: pointer;
  transition: color 0.2s var(--ease), border-color 0.2s var(--ease); }
.tbtn:hover { color: var(--text); border-color: var(--line-2); }
.tbtn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tbtn svg { width: 16px; height: 16px; }
.tbtn .i-moon { display: none; }
[data-theme="light"] .tbtn .i-moon { display: block; }
[data-theme="light"] .tbtn .i-sun { display: none; }

/* ── Hero — instrument panel: gauge centred, signal-rich flanks ──── */
.hero { display: flex; justify-content: center;
  background: var(--glass); border: 1px solid var(--line); border-radius: 22px;
  box-shadow: inset 0 1px 0 var(--glass-edge), var(--card-shadow);
  padding: clamp(28px, 3.4vw, 46px) clamp(20px, 3.2vw, 54px);
  backdrop-filter: blur(20px) saturate(1.5); -webkit-backdrop-filter: blur(20px) saturate(1.5); }
/* KPI grid incl. learning rate + GPU (left) · overall gauge (centre) · live
   step-loss graph (right). The flanks stretch to the gauge's height, so the
   width either side of the gauge carries signal instead of empty margin. */
.hero-grid { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: stretch; gap: clamp(20px, 2.8vw, 56px); width: 100%; margin: 0 auto; }
.hero-center { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 18px;
  min-width: 0; padding: 2px clamp(16px, 2vw, 40px);
  border-left: 1px solid var(--line); border-right: 1px solid var(--line); }
/* flanks vertically centre their content against the gauge; the step graph
   (right) carries flex:1 so it fills the gauge's full height */
.hero-side { display: flex; flex-direction: column; justify-content: center;
  gap: clamp(18px, 2.2vw, 30px); min-width: 0; }
/* a uniform 2-column KPI grid — loss · best, throughput · ETA, learning rate ·
   GPU memory — six peer cells, no separate strip */
.kpi-grid { display: grid; grid-template-columns: 1fr 1fr;
  gap: clamp(18px, 2.4vw, 32px) clamp(20px, 2.8vw, 48px); min-width: 0; }

/* concentric gauge — outer ring = overall run (with per-epoch ticks + best ★),
   inner ring = live phase steps. It carries everything the bar used to. */
.gauge { position: relative; width: clamp(248px, 25vw, 304px); aspect-ratio: 1; }
.gauge svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.g-bg, .g-ring { fill: none; }
.g-bg { stroke: color-mix(in srgb, var(--text) 8%, transparent); }
.g-ring { stroke-linecap: round;
  transition: stroke-dashoffset 0.6s var(--ease), stroke 0.5s var(--ease), opacity 0.3s var(--ease); }
.g-ring.run  { stroke: url(#ringGrad); stroke-width: 13; }
.g-ring.run.done { stroke: var(--st-completed); }           /* the whole run is complete — crown the outer ring gold */
.g-ring.step { stroke: var(--phasecol); stroke-width: 7; }  /* live phase's steps; gold in the gap between phases, blank at run end */
/* epoch divider ticks ringing the gauge, and the gold best-epoch ★ */
.g-tick { stroke: var(--line-2); stroke-width: 1.5; stroke-linecap: round; }
.g-best { fill: var(--gold); font-size: 14px; font-weight: 700; text-anchor: middle;
  dominant-baseline: central; font-family: var(--sans);
  filter: drop-shadow(0 0 3px color-mix(in srgb, var(--gold) 55%, transparent)); }
.rs1 { stop-color: var(--c1); } .rs2 { stop-color: var(--c2); } .rs3 { stop-color: var(--c3); }
.rs4 { stop-color: var(--c4); } .rs5 { stop-color: var(--c5); } .rs6 { stop-color: var(--c6); }
.rs7 { stop-color: var(--c7); } .rs8 { stop-color: var(--c8); }
.gauge-center { position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; }
.g-pct { display: flex; align-items: baseline; gap: 3px; font-family: var(--mono);
  font-variant-numeric: tabular-nums; }
.g-pct b { font-size: 4rem; font-weight: 550; letter-spacing: -0.045em; line-height: 1; }
.g-pct span { font-size: 1.4rem; color: var(--dim); }
.gauge-center i { font-style: normal; font-size: 0.7rem; font-weight: 550;
  letter-spacing: 0.02em; color: var(--faint); margin-top: 9px; }

/* phase / epoch / step readout under the gauge */
.hero-meta { display: flex; align-items: center; justify-content: center; gap: 16px; flex-wrap: wrap;
  font-size: 0.86rem; color: var(--dim); font-variant-numeric: tabular-nums; }
.hero-meta b { color: var(--text); font-weight: 550; font-family: var(--mono); }
.hm-sep { color: var(--faint); }
.hm-phase { font-style: normal; padding: 2px 10px; border-radius: 100px; font-size: 0.68rem;
  font-weight: 600; letter-spacing: 0.01em; white-space: nowrap; }
.hm-phase.is-train { color: var(--st-training);
  background: color-mix(in srgb, var(--st-training) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--st-training) 26%, transparent); }
.hm-phase.is-eval { color: var(--st-validating);
  background: color-mix(in srgb, var(--st-validating) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--st-validating) 26%, transparent); }

/* ── KPI cells — a uniform grid; learning rate and GPU memory are peers of the
   loss / throughput / ETA cells, not a separate strip. ───────────── */
.kpi { min-width: 0; max-width: 100%; overflow: hidden; }   /* a long value clips, never overlaps a neighbour */
.k-label { display: block; font-size: 0.72rem; font-weight: 550; letter-spacing: 0.01em;
  color: var(--faint); margin-bottom: 8px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.k-val { display: flex; align-items: baseline; gap: 8px; font-family: var(--mono);
  font-size: 1.58rem; font-weight: 550; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.k-val .star { color: var(--gold); font-size: 0.92rem; }
.k-val .k-note { font-size: 0.82rem; font-weight: 500; color: var(--dim); letter-spacing: 0; }
/* a multi-group LR range is a longer string — step the value down so it never
   collides with the neighbouring cell */
.k-val.is-range { font-size: 1.12rem; }
.k-sub { margin-top: 5px; font-size: 0.74rem; color: var(--faint); font-family: var(--mono);
  font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
/* GPU-memory usage bar — sits where the sub line would, fills with the used /
   total ratio and turns red near capacity so memory pressure is obvious */
.kpi .gbar { margin-top: 10px; height: 5px; border-radius: 100px; overflow: hidden;
  background: color-mix(in srgb, var(--text) 9%, transparent); }
.kpi .gbar i { display: block; height: 100%; width: 0%; border-radius: inherit; background: var(--good);
  transition: width 0.5s var(--ease), background 0.4s var(--ease); }

/* ── Step graph — the live per-step loss for the active phase, filling the
   opposite flank on its own. A hand-rolled SVG (filled area + line + leading
   dot) that auto-scales to its rolling window and takes the phase's colour
   (train blue, validation purple). Shown only while a step is in progress, in
   step with the throughput / ETA readouts. ── */
.stepgraph { flex: 1; display: flex; flex-direction: column; gap: 9px; min-width: 0; }
.sg-head { display: flex; align-items: baseline; gap: 11px; }
.sg-title { font-size: 0.72rem; font-weight: 550; letter-spacing: 0.01em; color: var(--faint); }
.sg-head .hm-phase { padding: 1px 8px; }
.sg-body { position: relative; flex: 1; min-height: clamp(118px, 15vw, 188px); width: 100%; }
.sg-body svg { display: block; width: 100%; height: 100%; }
.spark-empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  font-size: 0.78rem; color: var(--faint); letter-spacing: 0.01em; }
.sg-foot { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 2px 14px;
  font-family: var(--mono); font-size: 0.68rem; color: var(--faint); font-variant-numeric: tabular-nums; }

/* ── Charts — frosted-glass cards over the ambient washes.
      The grid (and its top margin) only exist once a chart is mounted, so no
      empty placeholder reserves space before the first epoch. ── */
.charts { display: block; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; margin-top: 18px; }
.grid:empty { display: none; }
/* a lone trailing chart (odd count) is centred instead of leaving an empty cell */
.grid > .chart:last-child:nth-child(odd) { grid-column: 1 / -1; justify-self: center; width: calc(50% - 9px); }
.chart { position: relative; background: var(--glass); border: 1px solid var(--line); border-radius: 18px;
  box-shadow: inset 0 1px 0 var(--glass-edge), var(--card-shadow); padding: 16px 18px 12px;
  backdrop-filter: blur(20px) saturate(1.5); -webkit-backdrop-filter: blur(20px) saturate(1.5);
  transition: border-color 0.35s var(--ease), box-shadow 0.45s var(--ease);
  animation: rise 0.5s var(--ease) both; }
/* hover highlight — no motion. The card holds still while a soft golden halo
   blooms around it and a warm sheen rises from the top edge; the border warms
   to gold too. Always gold (never the run-state accent) — the same note of
   excellence as the best ★ and a completed run. */
.chart::after { content: ''; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
  opacity: 0; transition: opacity 0.45s var(--ease);
  background: radial-gradient(135% 95% at 50% -12%, color-mix(in srgb, var(--gold) 15%, transparent), transparent 60%); }
.chart:hover { border-color: color-mix(in srgb, var(--gold) 46%, var(--line-2));
  box-shadow: inset 0 1px 0 var(--glass-edge),
    0 0 0 1px color-mix(in srgb, var(--gold) 18%, transparent),
    0 16px 40px color-mix(in srgb, var(--gold) 18%, transparent), var(--card-shadow); }
.chart:hover::after { opacity: 1; }
.hovdot { transition: cx 0.09s linear, cy 0.09s linear; }
.chart-head { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.chart-name { font-size: 0.92rem; font-weight: 650; letter-spacing: -0.01em; }
.legend { display: inline-flex; gap: 11px; margin-left: 2px; font-size: 0.74rem; color: var(--dim); }
.legend span { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.legend i { width: 8px; height: 8px; border-radius: 50%; }
.chart-acts { margin-left: auto; display: inline-flex; gap: 6px; opacity: 0; transition: opacity 0.2s var(--ease); }
.chart:hover .chart-acts, .chart-acts:focus-within { opacity: 1; }
.cbtn { font-family: var(--mono); font-size: 0.66rem; font-weight: 600; letter-spacing: 0.04em;
  display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 6px; cursor: pointer;
  color: var(--dim); background: none; border: 1px solid var(--line);
  transition: color 0.2s var(--ease), border-color 0.2s var(--ease), background 0.2s var(--ease); }
.cbtn:hover { color: var(--text); border-color: var(--line-2); }
.cbtn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.cbtn.on { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 35%, transparent);
  background: color-mix(in srgb, var(--accent) 8%, transparent); }
.chart-body svg { display: block; width: 100%; height: auto; }

/* ── Run details — configuration · environment · model ────── */
.meta { display: grid; grid-template-columns: repeat(3, 1fr); column-gap: 0; row-gap: 26px;
  background: var(--glass); border: 1px solid var(--line); border-radius: 18px;
  box-shadow: inset 0 1px 0 var(--glass-edge), var(--card-shadow);
  backdrop-filter: blur(20px) saturate(1.5); -webkit-backdrop-filter: blur(20px) saturate(1.5);
  padding: 24px clamp(12px, 1.6vw, 22px) 20px; margin-top: 18px; }
/* vertical hairlines divide the three columns */
.meta > div { padding: 2px clamp(20px, 2.6vw, 40px); }
.meta > div + div { border-left: 1px solid var(--line); }
.meta h2 { font-size: 0.82rem; font-weight: 600; letter-spacing: 0.005em;
  color: var(--dim); margin-bottom: 11px; }
.kv { display: flex; flex-direction: column; }
.kv-row { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
  padding: 7px 8px; margin: 0 -8px; border-radius: 7px; font-size: 0.8rem; cursor: pointer; user-select: none;
  transition: background 0.15s var(--ease); }
.kv-row:hover { background: var(--hover); }
.kv-row:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.kv-row.copied .kv-v { color: var(--gold); }
.kv-k { color: var(--dim); flex-shrink: 0; }
.kv-v { font-family: var(--mono); font-size: 0.76rem; text-align: right; word-break: break-word;
  font-variant-numeric: tabular-nums; transition: color 0.2s var(--ease); }
.kv .none { color: var(--faint); font-size: 0.78rem; padding: 4px 0; }
/* nested configuration — a dict key opens an indented sub-group marked by a
   hairline guide; scalar leaves stay copyable rows, so a dict / nested-dict
   learning-rate or scheduler config reads cleanly instead of "[object Object]" */
.kv-grouphead { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.005em;
  color: var(--dim); padding: 10px 8px 3px; }
.kv-nest { margin: 1px 0 3px 7px; padding-left: 12px; border-left: 1px solid var(--line); }
.kv-nest .kv-grouphead:first-child { padding-top: 4px; }

/* ── Tooltip · toast · footer ──────────────────────────────── */
.tip { position: fixed; z-index: 90; pointer-events: none; min-width: 116px; padding: 9px 12px;
  border-radius: 10px; background: var(--tip-bg); border: 1px solid var(--line-2);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: var(--shadow); font-size: 0.74rem; opacity: 0; transition: opacity 0.12s var(--ease); }
.tip.show { opacity: 1; }
.tip-title { color: var(--faint); font-weight: 600; font-size: 0.72rem; letter-spacing: 0.005em;
  margin-bottom: 5px; }
.tip-row { display: flex; justify-content: space-between; gap: 14px; padding: 1px 0; }
.tip-row .tk { display: inline-flex; align-items: center; gap: 6px; color: var(--dim); }
.tip-row .tk i { width: 8px; height: 8px; border-radius: 50%; }
.tip-row .tv { font-family: var(--mono); color: var(--text); font-variant-numeric: tabular-nums; }
.toast { position: fixed; bottom: 26px; left: 50%; transform: translate(-50%, 10px); z-index: 95;
  padding: 8px 17px; border-radius: 100px; background: var(--tip-bg); border: 1px solid var(--line-2);
  color: var(--gold); font-size: 0.76rem; font-weight: 600; opacity: 0; pointer-events: none;
  box-shadow: var(--shadow); transition: opacity 0.2s var(--ease), transform 0.2s var(--ease); }
.toast.show { opacity: 1; transform: translate(-50%, 0); }
footer { padding: 36px 0 0; text-align: center; color: var(--faint);
  font-size: 0.7rem; font-family: var(--mono); letter-spacing: 0.04em; }

@media (max-width: 1080px) {
  /* stack: gauge on top, then the KPI grid, then the step graph */
  .hero-grid { display: flex; flex-direction: column; align-items: stretch; gap: 28px; max-width: 560px; }
  .hero-center { order: -1; border: 0; padding: 0; }
  .hero-side { gap: 20px; }
  .stepgraph { flex: none; }
}
@media (max-width: 1040px) {
  .meta { grid-template-columns: 1fr 1fr; }
  /* the third column wraps to a full-width row — divide it with a rule above */
  .meta > div:nth-child(3) { grid-column: 1 / -1; border-left: 0; border-top: 1px solid var(--line); padding-top: 24px; }
}
@media (max-width: 720px) {
  .grid { grid-template-columns: 1fr; }
  .meta { grid-template-columns: 1fr; }
  .meta > div { border-left: 0; }
  .meta > div + div { border-top: 1px solid var(--line); padding-top: 24px; }
  .meta > div:nth-child(3) { grid-column: auto; }
  .top { gap: 12px; }
  .brand-name { font-size: 1.4rem; }
  .top-meta { order: 4; width: 100%; margin-left: 0; }
  .pill { margin-left: auto; }
  .gauge { width: 214px; }
  .g-pct b { font-size: 3.1rem; }
  .g-pct span { font-size: 1.1rem; }
}
@media (max-width: 460px) {
  .k-val { font-size: 1.3rem; }
  .kpi-grid { gap: 16px 24px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition-duration: 0.01ms !important; }
}"""

_HTML_SHELL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="poll-ms"    content="__T4A_POLL_MS__">
<meta name="data-file"  content="__T4A_DATA_FILE__">
<meta name="stale-ms"   content="__T4A_STALE_MS__">
<meta name="version"    content="__T4A_VERSION__">
<title>train4all — Dashboard</title>
<script>(function () { try {
  var t = localStorage.getItem('t4a-theme')
       || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  document.documentElement.dataset.theme = t;
} catch (e) { document.documentElement.dataset.theme = 'dark'; } })();</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400..700&family=JetBrains+Mono:wght@400..600&display=swap" rel="stylesheet">
<style>
__T4A_CSS__
</style>
</head>
<body>
<div class="runline" id="run-line"></div>
<div class="app">

  <header class="top">
    <div class="brand">
      <a class="brand-link" href="https://github.com/tomoking2004/train4all" target="_blank" rel="noopener" aria-label="train4all on GitHub">
        <span class="brand-name">train4all</span>
        <svg class="gh" viewBox="0 0 24 24" aria-hidden="true">
          <defs>
            <linearGradient id="ghGrad" x1="0" y1="0" x2="1" y2="0.25">
              <stop class="gs1" offset="0"/><stop class="gs2" offset="0.5"/><stop class="gs3" offset="1"/>
            </linearGradient>
          </defs>
          <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
        </svg>
      </a>
      <span class="brand-cap">Live Training Dashboard</span>
    </div>
    <div class="top-meta">
      <span><span class="k">Started</span><b id="m-start">—</b></span>
      <span><span class="k">Elapsed</span><b id="m-elapsed">—</b></span>
    </div>
    <div class="pill" id="status" aria-live="polite"><span class="dot"></span><span id="status-text">Standby</span></div>
    <button class="tbtn" id="theme-btn" title="Toggle theme (T)" aria-label="Toggle light / dark theme">
      <svg class="i-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
        <circle cx="12" cy="12" r="4"/>
        <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>
      </svg>
      <svg class="i-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
      </svg>
    </button>
  </header>

  <!-- Hero: an instrument panel — KPI grid incl. learning rate + GPU (left) ·
       overall gauge (centre) · live step-loss graph (right). The flanks carry
       signal into the width either side of the gauge instead of empty margin. -->
  <section class="hero" aria-label="Run progress">
    <div class="hero-grid">
      <div class="hero-side hero-left">
        <div class="kpi-grid">
          <div class="kpi">
            <span class="k-label" id="k-primary-label">Metric</span>
            <div class="k-val"><span id="k-primary">—</span></div>
            <div class="k-sub" id="k-primary-sub">—</div>
          </div>
          <div class="kpi">
            <span class="k-label" id="k-best-label">Best Val</span>
            <div class="k-val"><span class="star" id="k-best-star" style="display:none">★</span><span id="k-best">—</span></div>
            <div class="k-sub" id="k-best-sub">—</div>
          </div>
          <div class="kpi">
            <span class="k-label">Throughput</span>
            <div class="k-val" id="k-speed">—</div>
            <div class="k-sub">it / s</div>
          </div>
          <div class="kpi">
            <span class="k-label">ETA</span>
            <div class="k-val" id="k-eta">—</div>
            <div class="k-sub" id="k-eta-sub">—</div>
          </div>
          <div class="kpi">
            <span class="k-label">Learning Rate</span>
            <div class="k-val" id="k-lr">—</div>
            <div class="k-sub" id="k-lr-sub"></div>
          </div>
          <div class="kpi" id="stat-gpu" style="display:none">
            <span class="k-label">GPU Memory</span>
            <div class="k-val"><span id="k-gpu-pct">—</span><span class="k-note" id="k-gpu"></span></div>
            <div class="gbar"><i id="gpu-bar"></i></div>
          </div>
        </div>
      </div>

      <div class="hero-center">
        <div class="gauge">
          <svg viewBox="0 0 300 300" aria-hidden="true">
            <defs>
              <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
                <stop class="rs1" offset="0"/><stop class="rs2" offset="0.18"/>
                <stop class="rs3" offset="0.36"/><stop class="rs4" offset="0.52"/>
                <stop class="rs5" offset="0.66"/><stop class="rs6" offset="0.8"/>
                <stop class="rs7" offset="0.9"/><stop class="rs8" offset="1"/>
              </linearGradient>
            </defs>
            <circle class="g-bg" cx="150" cy="150" r="124" stroke-width="13"/>
            <circle class="g-ring run" id="ring-run" cx="150" cy="150" r="124" stroke-dasharray="779.11" stroke-dashoffset="779.11"/>
            <circle class="g-bg" cx="150" cy="150" r="100" stroke-width="7"/>
            <circle class="g-ring step" id="ring-step" cx="150" cy="150" r="100" stroke-dasharray="628.32" stroke-dashoffset="628.32"/>
            <g id="g-marks"></g>
          </svg>
          <div class="gauge-center">
            <div class="g-pct"><b id="ov-pct">0</b><span>%</span></div>
            <i id="ov-label">Overall</i>
          </div>
        </div>
        <div class="hero-meta">
          <span class="hm-phase" id="hm-phase" style="display:none">—</span>
          <span id="hm-epoch">Epoch — / —</span>
          <span class="hm-sep">·</span>
          <span id="hm-step">Step — / —</span>
        </div>
      </div>

      <div class="hero-side hero-right">
        <div class="stepgraph" aria-label="Loss per step for the active phase">
          <div class="sg-head">
            <span class="sg-title">Step Loss</span>
            <span class="hm-phase" id="sg-phase" style="display:none">—</span>
          </div>
          <div class="sg-body" id="spark-body"><div class="spark-empty">awaiting steps…</div></div>
          <div class="sg-foot">
            <span id="sp-win">—</span>
            <span id="sp-range">—</span>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Metric charts — uniform two-column grid (loss first); the area
       materialises only once the first epoch metric exists -->
  <main class="charts">
    <div class="grid" id="charts-grid"></div>
  </main>

  <section class="meta">
    <div><h2>Configuration</h2><div class="kv" id="cfg-grid"><div class="none">Loading…</div></div></div>
    <div><h2>Environment</h2><div class="kv" id="env-grid"><div class="none">Loading…</div></div></div>
    <div><h2>Model</h2><div class="kv" id="model-grid"><div class="none">Loading…</div></div></div>
  </section>

  <footer id="footer">train4all</footer>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {
  'use strict';

  /* ── boot ──────────────────────────────────────────────── */
  const META      = (n) => document.querySelector('meta[name="' + n + '"]');
  const POLL_MS   = parseInt(META('poll-ms').content) || 1000;
  const DATA_FILE = META('data-file').content;
  const STALE_MS  = parseInt(META('stale-ms').content) || 30000;
  const VERSION   = META('version') ? META('version').content : '';
  const REDUCED   = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const el    = (id) => document.getElementById(id);
  const clamp = (x, a, b) => Math.min(Math.max(x, a), b);
  const lerp  = (a, b, k) => a + (b - a) * k;
  const titleCase = (s) => String(s).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  function setText(id, t) { const e = el(id); if (e && e.textContent !== t) e.textContent = t; }
  function fmt(v) { if (typeof v !== 'number' || !isFinite(v)) return String(v); const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2); if (a >= 100) return v.toFixed(2); return v.toFixed(4); }
  function fmtDur(s) { if (!isFinite(s) || s < 0) return '—'; s = Math.round(s);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
    if (h) return h + ':' + String(m).padStart(2, '0') + ':' + String(x).padStart(2, '0'); return m + ':' + String(x).padStart(2, '0'); }
  function fmtClock(d) { return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0'); }

  /* ── theme ─────────────────────────────────────────────── */
  let themeEpoch = 0;
  function themeColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    return { bg: v('--bg'), text: v('--text'), dim: v('--dim'), faint: v('--faint'),
      line: v('--line'), gold: v('--gold'),
      pal: [v('--c1'), v('--c2'), v('--c3'), v('--c4'), v('--c5'), v('--c6'), v('--c7'), v('--c8')] };
  }
  /* phase-fixed colours from the spectrum — train is always blue, val /
     eval purple, test pink; anything else takes the next free hue. Curves
     therefore always agree with the phase badge and the state accents. */
  function phasePal(C, names) {
    let next = 1;
    return names.map((nm) => {
      let i = /^train/i.test(nm) ? 0 : /^(val|dev|eval)/i.test(nm) ? 3 : /^test/i.test(nm) ? 6 : -1;
      if (i < 0) { i = next % C.pal.length; next += 2; }
      return C.pal[i];
    });
  }
  function accentColor() { return getComputedStyle(document.documentElement).getPropertyValue('--accent').trim(); }
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem('t4a-theme', t); } catch (e) {}
    themeEpoch++;
    setFavicon(accentColor());
    Object.keys(charts).forEach(renderChart);
    if (sparkData) renderSpark(sparkData);
  }
  el('theme-btn').addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
  });
  document.addEventListener('keydown', (e) => {
    if ((e.key === 't' || e.key === 'T') && !e.ctrlKey && !e.metaKey && !e.altKey
        && !/INPUT|TEXTAREA|SELECT/.test((e.target && e.target.tagName) || '')) el('theme-btn').click();
  });

  /* ── state accent · favicon · status pill ──────────────── */
  const favLink = document.createElement('link'); favLink.rel = 'icon'; document.head.appendChild(favLink);
  function setFavicon(color) { try {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const x = c.getContext('2d'); x.fillStyle = color; x.beginPath();
    if (x.roundRect) x.roundRect(13, 13, 38, 38, 12); else x.arc(32, 32, 19, 0, Math.PI * 2);
    x.fill();
    const g = x.createLinearGradient(0, 13, 0, 51);
    g.addColorStop(0, 'rgba(255,255,255,0.33)'); g.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = g; x.fill();
    favLink.href = c.toDataURL('image/png');
  } catch (e) {} }

  const MODE_LABEL = { training: 'Training', validating: 'Validating', stagnant: 'Plateau',
    completed: 'Completed', idle: 'Standby', stopped: 'Offline' };
  let curMode = '';
  function setMode(mode, ni) {
    const pill = el('status');
    pill.classList.toggle('live', mode === 'training' || mode === 'validating');
    pill.title = mode === 'stagnant' ? 'No validation improvement for ' + ni + ' epoch' + (ni === 1 ? '' : 's') : '';
    setText('status-text', MODE_LABEL[mode] || mode);
    if (mode === curMode) return;
    curMode = mode;
    document.documentElement.style.setProperty('--accent', 'var(--st-' + mode + ')');
    setFavicon(accentColor());
  }

  /* ── eased number tween ────────────────────────────────── */
  function tween(node, to, f) {
    if (!node) return;
    if (typeof to !== 'number' || !isFinite(to)) { node.textContent = (to === undefined || to === null) ? '—' : String(to); node._v = undefined; return; }
    const from = (typeof node._v === 'number') ? node._v : to; node._v = to;
    if (from === to || REDUCED) { node.textContent = f(to); return; }
    const t0 = performance.now(), d = 380;
    (function step(t) { const k = clamp((t - t0) / d, 0, 1), e = 1 - Math.pow(1 - k, 3);
      node.textContent = f(from + (to - from) * e); if (k < 1) requestAnimationFrame(step); })(performance.now());
  }

  /* ── click-to-copy rows ────────────────────────────────── */
  function showToast() { let t = el('toast'); if (!t) { t = document.createElement('div'); t.id = 'toast'; t.className = 'toast'; t.textContent = '✓ Copied'; document.body.appendChild(t); }
    t.classList.add('show'); clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove('show'), 1500); }
  function attachCopy(row) { if (row.dataset.copy) return; row.dataset.copy = '1';
    const go = () => { const v = row.getAttribute('data-v') || ''; if (!v || v === '—') return;
      (navigator.clipboard ? navigator.clipboard.writeText(v) : Promise.reject()).then(() => { row.classList.add('copied'); showToast(); setTimeout(() => row.classList.remove('copied'), 1000); }).catch(showToast); };
    row.addEventListener('click', go);
    row.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } }); }
  /* render a config / info object as copyable rows. A nested dict opens an
     indented sub-group (a hairline guide marks each level); an array of scalars
     collapses to an inline list, an array of dicts nests by index. So a dict or
     nested-dict learning-rate / scheduler config reads cleanly rather than
     rendering as "[object Object]". */
  function isPlainObj(v) { return v != null && typeof v === 'object' && !Array.isArray(v); }
  function fmtCfgNum(n) {
    if (!isFinite(n)) return String(n);
    if (Number.isInteger(n)) return String(n);
    const a = Math.abs(n);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return n.toExponential(2).replace(/\.?0+e/, 'e');
    return String(+n.toPrecision(6));
  }
  function fmtScalar(v) {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (typeof v === 'number') return fmtCfgNum(v);
    return String(v);
  }
  function fmtInline(v) {   /* a scalar, or an array / dict collapsed to one line */
    if (Array.isArray(v)) return '[' + v.map(fmtInline).join(', ') + ']';
    if (isPlainObj(v)) return '{' + Object.entries(v).map(([k, x]) => k + ': ' + fmtInline(x)).join(', ') + '}';
    return fmtScalar(v);
  }
  function kvRow(k, val) {
    return '<div class="kv-row" tabindex="0" role="button" data-v="' + esc(val) + '">'
      + '<span class="kv-k">' + esc(k) + '</span><span class="kv-v">' + esc(val) + '</span></div>';
  }
  function kvHTML(obj) {
    let h = '';
    for (const [k, v] of Object.entries(obj)) {
      if (isPlainObj(v)) {
        h += Object.keys(v).length
          ? '<div class="kv-grouphead">' + esc(k) + '</div><div class="kv-nest">' + kvHTML(v) + '</div>'
          : kvRow(k, '{ }');
      } else if (Array.isArray(v) && v.some(isPlainObj)) {
        const indexed = {}; v.forEach((x, i) => { indexed[i] = x; });
        h += '<div class="kv-grouphead">' + esc(k) + '</div><div class="kv-nest">' + kvHTML(indexed) + '</div>';
      } else {
        h += kvRow(k, Array.isArray(v) ? fmtInline(v) : fmtScalar(v));
      }
    }
    return h;
  }
  function renderKV(id, obj) {
    const c = el(id); if (!c) return;
    if (!obj || !Object.keys(obj).length) { c.innerHTML = '<div class="none">—</div>'; return; }
    c.innerHTML = kvHTML(obj);
    c.querySelectorAll('.kv-row').forEach(attachCopy);
  }

  /* ── throughput · ETA ──────────────────────────────────── */
  const spd = { ema: 0, lastStep: -1, lastT: 0 };
  function updateSpeed(d, grad) { if (d.status !== 'training' || !grad || !d.max_step) return;
    const g = (Math.max(d.current_epoch, 1) - 1) * d.max_step + (d.current_step || 0), now = performance.now();
    if (spd.lastStep >= 0 && g > spd.lastStep && spd.lastT) { const dt = (now - spd.lastT) / 1000; if (dt > 0.05) { const r = (g - spd.lastStep) / dt; spd.ema = spd.ema ? lerp(spd.ema, r, 0.3) : r; } }
    spd.lastStep = g; spd.lastT = now; }
  function etaSeconds(d) { if (!spd.ema || !d.max_step || !d.max_epoch) return NaN;
    const total = d.max_epoch * d.max_step, done = (Math.max(d.current_epoch, 1) - 1) * d.max_step + (d.current_step || 0); return Math.max(total - done, 0) / spd.ema; }

  /* ── metric helpers ────────────────────────────────────── */
  function primarySeries(phases) { if (!phases) return null; if (phases.train && phases.train.length) return { phase: 'train', vals: phases.train };
    for (const k of Object.keys(phases)) if (phases[k] && phases[k].length) return { phase: k, vals: phases[k] }; return null; }
  function metricNames(d) { return Array.from(new Set(Object.keys(d.epoch_metrics || {}).concat(Object.keys(d.last_step_metrics || {})))); }
  function pickPrimary(d) { const names = metricNames(d); return names.find((n) => /loss/i.test(n)) || names[0] || null; }
  function liveValue(d, m) { const sm = d.last_step_metrics || {}, ps = primarySeries((d.epoch_metrics || {})[m]); return (typeof sm[m] === 'number') ? sm[m] : (ps ? ps.vals[ps.vals.length - 1] : undefined); }

  /* ── progress model — strictly monotonic ───────────────────
     The overall run percentage and the progress bar advance through
     every training AND validation step, proportionally to each phase's
     step count, and never rewind. Updates that carry no active phase
     (the inter-phase metric flushes) return null so the caller holds the
     last value instead of dropping to zero. */
  function epochFraction(d) {
    const cs = d.current_step || 0, ms = d.max_step || 0, grad = !!d.is_gradient_phase;
    if (!d.last_phase || ms <= 0 || cs <= 0) return null;          /* idle / flush — hold previous */
    const TS = d.train_steps || 0, VS = d.val_steps || 0, total = TS + VS;
    if (TS > 0) {                                                  /* proportional global-step model */
      const done = grad ? cs : TS + cs;
      return clamp(done / total, 0, 1);
    }
    return grad ? clamp(cs / ms, 0, 1) : 1;                        /* sizes unknown — best effort */
  }
  const prog = { epFrac: 0, stepFrac: 0, phaseCol: 'var(--st-idle)', ran: false,
                 lastGrad: false, lastPhase: '', lastMax: 0 };

  /* ── gauge rings ───────────────────────────────────────── */
  const RING_RUN_C = 779.11, RING_STEP_C = 628.32;
  function setRing(id, circ, frac) {
    const e = el(id); if (!e) return;
    const f = clamp(frac, 0, 1);
    e.style.strokeDashoffset = (circ * (1 - f)).toFixed(1);
    e.style.opacity = f > 0.001 ? '1' : '0';   /* hide the round-cap dot at 0 % (e.g. standby) */
  }

  /* ── gauge detail — epoch divider ticks + the best-epoch ★, set around the
     outer ring. The svg is rotated -90° so its 0° sits at top, so ticks use
     plain angles; the ★ is counter-rotated +90° so the glyph stays upright. */
  let gaugeSig = '';
  function updateGauge(d) {
    const marks = el('g-marks'); if (!marks) return;
    const E = d.max_epoch || 0, best = d.best_epoch || 0, sig = E + ':' + best;
    if (sig === gaugeSig) return;
    gaugeSig = sig;
    let s = '';
    if (E >= 2 && E <= 48) {
      for (let k = 0; k < E; k++) {
        const a = (k / E) * 2 * Math.PI;
        s += '<line class="g-tick" x1="' + (150 + 135 * Math.cos(a)).toFixed(1) + '" y1="' + (150 + 135 * Math.sin(a)).toFixed(1)
          + '" x2="' + (150 + 142 * Math.cos(a)).toFixed(1) + '" y2="' + (150 + 142 * Math.sin(a)).toFixed(1) + '"/>';
      }
    }
    if (best >= 1 && E) {
      const a = ((best - 0.5) / E) * 2 * Math.PI;
      const x = (150 + 142 * Math.cos(a)).toFixed(1), y = (150 + 142 * Math.sin(a)).toFixed(1);
      s += '<text class="g-best" x="' + x + '" y="' + y + '" transform="rotate(90 ' + x + ' ' + y + ')"><title>Best epoch ' + best + '</title>★</text>';
    }
    marks.innerHTML = s;
  }

  /* ── KPI cells ─────────────────────────────────────────── */
  /* The live metric, throughput and ETA are instantaneous readings: they show
     only while a step is actually running and blank to "—" otherwise — nothing
     lingers once the run is idle or finished (the full history lives in the
     charts below). The best-metric KPI is a standing record, so it persists. */
  function updateKpis(d, primary, stepping) {
    if (primary && stepping) {
      setText('k-primary-label', titleCase(primary));
      tween(el('k-primary'), liveValue(d, primary), fmt);
      setText('k-primary-sub', titleCase(d.last_phase) + ' · step ' + (d.current_step || 0));
    } else {
      setText('k-primary-label', primary ? titleCase(primary) : 'Metric');
      tween(el('k-primary'), undefined); setText('k-primary-sub', '—');
    }
    setText('k-best-label', 'Best Val ' + titleCase(d.monitor || 'loss'));
    if (d.best_metric != null) {
      el('k-best-star').style.display = '';
      tween(el('k-best'), d.best_metric, fmt);
      const ni = d.epochs_no_improve || 0;
      setText('k-best-sub', 'epoch ' + d.best_epoch + (ni > 0 ? ' · ' + ni + ' since' : ''));
    } else { el('k-best-star').style.display = 'none'; tween(el('k-best'), undefined); setText('k-best-sub', '—'); }
  }

  /* ── Step graph + aux stats — live signal in the hero flanks.
     The step graph is an auto-scaled trace of the active phase's recent step
     loss (train blue / val purple), shown only while a step is in progress —
     in step with the throughput / ETA readouts. Aux stats carry learning rate
     and GPU memory. ── */
  let sparkData = null, sparkSig = '';
  function fmtLR(v) {
    if (Array.isArray(v)) {   /* per-group learning rates → single value, pair, or range */
      const u = Array.from(new Set(v.filter((x) => typeof x === 'number' && isFinite(x)))).sort((a, b) => a - b);
      if (!u.length) return '—';
      if (u.length === 1) return fmtLR(u[0]);
      if (u.length === 2) return fmtLR(u[0]) + ' / ' + fmtLR(u[1]);
      return fmtLR(u[0]) + ' – ' + fmtLR(u[u.length - 1]);
    }
    if (typeof v !== 'number' || !isFinite(v)) return '—';
    if (v === 0) return '0';
    if (v < 1e-3 || v >= 1e4) return v.toExponential(2).replace(/\.?0+e/, 'e');  /* compact: 1.00e-4 → 1e-4 */
    return String(+v.toPrecision(3));
  }
  function fmtGB(g) { return g >= 100 ? g.toFixed(0) : g.toFixed(g >= 10 ? 1 : 2); }
  function isStepping(d) { return !!(d.last_phase && d.max_step && d.current_step) && d.status === 'training'; }

  function sparkSVG(vals, W, H, phase) {
    const C = themeColors(), col = phasePal(C, [phase || 'train'])[0];   /* phase hue — echoes that phase's curve below */
    const PADX = 3, PADT = 10, PADB = 8, n = vals.length;
    let mn = Infinity, mx = -Infinity;
    for (const v of vals) { if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mn === mx) { mn -= 0.5; mx += 0.5; }
    const X = (i) => n < 2 ? W / 2 : PADX + (i / (n - 1)) * (W - 2 * PADX);
    const Y = (v) => PADT + (1 - (v - mn) / (mx - mn)) * (H - PADT - PADB);
    let dLine = '';
    for (let i = 0; i < n; i++) dLine += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(vals[i]).toFixed(1);
    const base = (H - PADB).toFixed(1);
    const dArea = dLine + 'L' + X(n - 1).toFixed(1) + ' ' + base + 'L' + X(0).toFixed(1) + ' ' + base + 'Z';
    const lx = X(n - 1).toFixed(1), ly = Y(vals[n - 1]).toFixed(1);
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg">'
      + '<defs><linearGradient id="spGrad" x1="0" y1="0" x2="0" y2="1">'
      + '<stop offset="0" stop-color="' + col + '" stop-opacity="0.22"/>'
      + '<stop offset="1" stop-color="' + col + '" stop-opacity="0"/></linearGradient></defs>'
      + '<path d="' + dArea + '" fill="url(#spGrad)"/>'
      + '<path d="' + dLine + '" fill="none" stroke="' + col + '" stroke-width="1.8" '
      + 'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
      + '<circle cx="' + lx + '" cy="' + ly + '" r="5.4" fill="' + col + '" opacity="0.18"/>'
      + '<circle cx="' + lx + '" cy="' + ly + '" r="2.7" fill="' + col + '" stroke="' + C.bg + '" stroke-width="1.4"/>'
      + '</svg>';
  }

  function renderSpark(d) {
    sparkData = d;
    const body = el('spark-body'); if (!body) return;
    /* shown only while a step is actually running — like throughput / ETA, and
       nothing lingers once the run is idle or finished */
    const stepping = isStepping(d), ph = d.step_loss_phase || '';
    const vals = (stepping && Array.isArray(d.step_loss)) ? d.step_loss.filter((v) => typeof v === 'number' && isFinite(v)) : [];
    if (vals.length) {
      let mn = Infinity, mx = -Infinity;
      for (const v of vals) { if (v < mn) mn = v; if (v > mx) mx = v; }
      const f = d.step_loss_first, l = d.step_loss_last;   /* real step numbers, not the sample count */
      setText('sp-win', (typeof f === 'number' && typeof l === 'number')
        ? (f === l ? 'step ' + l : 'steps ' + f + '–' + l) : '—');
      setText('sp-range', 'min ' + fmt(mn) + ' · max ' + fmt(mx));
    } else {
      setText('sp-win', '—'); setText('sp-range', '—');
    }
    const phEl = el('sg-phase');
    if (phEl) {
      if (stepping && ph) { phEl.style.display = ''; phEl.textContent = titleCase(ph);
        phEl.className = 'hm-phase ' + (/^train/i.test(ph) ? 'is-train' : 'is-eval'); }
      else phEl.style.display = 'none';
    }
    const W = Math.max(80, Math.round(body.clientWidth || 240));
    const sig = (stepping ? 1 : 0) + '|' + vals.length + '|' + (vals.length ? vals[vals.length - 1] : '')
      + '|' + ph + '|' + (d.step_loss_first || '') + '|' + themeEpoch + '|' + W;
    if (sig === sparkSig) return;
    sparkSig = sig;
    if (!vals.length) { body.innerHTML = '<div class="spark-empty">' + (stepping ? 'awaiting steps…' : '—') + '</div>'; return; }
    const H = Math.max(40, Math.round(body.clientHeight || 120));
    body.innerHTML = sparkSVG(vals, W, H, ph);
  }

  /* Learning rate and GPU memory are standing telemetry — like the best-metric
     KPI they persist rather than blanking between steps. */
  function updateTelemetry(d) {
    const lr = d.learning_rate;
    setText('k-lr', fmtLR(lr));
    const lrEl = el('k-lr');   /* multi-group LRs: hover to read every group's rate */
    const nGroups = Array.isArray(lr) ? new Set(lr.filter((x) => typeof x === 'number' && isFinite(x))).size : 0;
    if (lrEl) { lrEl.title = Array.isArray(lr) ? lr.map((x) => fmtLR(x)).join(', ') : '';
      lrEl.classList.toggle('is-range', nGroups > 1); }   /* a range is wider — shrink to fit the cell */
    setText('k-lr-sub', nGroups > 1 ? nGroups + ' groups' : '');

    const gu = d.gpu_mem_used, gt = d.gpu_mem_total, gpuEl = el('stat-gpu');
    if (!gpuEl) return;
    if (typeof gu === 'number' && typeof gt === 'number' && gt > 0) {
      gpuEl.style.display = '';
      const r = clamp(gu / gt, 0, 1);
      setText('k-gpu-pct', Math.round(r * 100) + '%');
      setText('k-gpu', fmtGB(gu) + ' / ' + fmtGB(gt) + ' GB');
      const bar = el('gpu-bar');
      if (bar) { bar.style.width = (r * 100).toFixed(1) + '%';
        bar.style.background = r > 0.9 ? 'var(--bad)' : 'var(--good)'; }
    } else gpuEl.style.display = 'none';
  }

  const sparkRO = new ResizeObserver(() => { if (sparkData) renderSpark(sparkData); });
  (function () { const sb = el('spark-body'); if (sb) sparkRO.observe(sb); })();

  /* ════════════════════════════════════════════════════════
     SVG CHART ENGINE — dependency-free metric history.
     Pure vector: hairline grid, per-phase curves, best-epoch
     marker, logarithmic toggle, hover readout, and native
     SVG export. Colours are read from the active theme.
     ════════════════════════════════════════════════════════ */
  const charts = {};
  let bestEpochGlobal = 0;
  /* plot padding — the left and bottom gutters carry the tick values *and* the
     axis titles (metric name on Y, "Epoch" on X), so they are deeper than the
     other two sides */
  const PL = 58, PR = 18, PT = 30, PB = 46;
  const cid = (m) => m.replace(/[^a-zA-Z0-9_-]/g, '_');

  function xTickStep(n, plotW) {
    const target = Math.max(4, Math.floor(plotW / 96));
    const raw = Math.max(n / target, 1), p = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p; return 10 * p;
  }

  /* nice axis ticks — linear ticks land on 1·2·2.5·5 steps; log ticks
     snap to powers of ten (with 2× / 5× mantissas when decades are few),
     falling back to nice raw values inside a single decade */
  function niceTicksLinear(mn, mx, target) {
    const span = mx - mn;
    if (!(span > 0) || !isFinite(span)) return [mn];
    const raw = span / Math.max(target, 1);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    let st = 10 * p;
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) { st = m * p; break; }
    const out = [];
    for (let v = Math.ceil(mn / st) * st; v <= mx + st * 1e-6; v += st) out.push(v);
    return out;
  }
  function logTicks(mnL, mxL, target) {
    const span = mxL - mnL;
    if (span >= 1) {
      const out = [];
      const step = Math.max(1, Math.round(span / Math.max(target, 1)));
      for (let d = Math.ceil(mnL); d <= Math.floor(mxL) + 1e-9; d += step) out.push(d);
      if (out.length < 3 && span < 3) {
        for (let d = Math.floor(mnL) - 1; d <= Math.ceil(mxL); d++)
          for (const m of [2, 5]) { const lv = d + Math.log10(m); if (lv > mnL && lv < mxL) out.push(lv); }
        out.sort((a, b) => a - b);
      }
      if (out.length >= 2) return out;
    }
    return niceTicksLinear(Math.pow(10, mnL), Math.pow(10, mxL), target)
      .filter((v) => v > 0).map((v) => Math.log10(v));
  }
  function fmtTick(v) {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a < 1e-3 || a >= 1e5) return v.toExponential(0).replace('+', '');
    return String(+v.toFixed(a < 1 ? 4 : a < 10 ? 3 : a < 100 ? 2 : 1));
  }

  function chartSVG(metric, phases, useLog, bestEpoch, W, H, ch) {
    const C = themeColors();
    const tf = useLog ? Math.log10 : (v) => v;
    const series = Object.entries(phases).filter(([, v]) => v && v.length);
    if (!series.length) return '';
    const maxLen = Math.max.apply(null, series.map(([, v]) => v.length));
    let mn = Infinity, mx = -Infinity;
    series.forEach(([, v]) => v.forEach((x) => { if (useLog && x <= 0) return; const y = tf(x); if (y < mn) mn = y; if (y > mx) mx = y; }));
    if (!isFinite(mn)) { mn = 0; mx = 1; }
    if (mn === mx) { mn -= 0.5; mx += 0.5; }
    const padY = (mx - mn) * 0.08; mn -= padY; mx += padY;
    const X = (i) => maxLen < 2 ? PL + (W - PL - PR) / 2 : PL + (i / (maxLen - 1)) * (W - PL - PR);
    const Y = (v) => PT + (1 - (tf(v) - mn) / (mx - mn)) * (H - PT - PB);
    const k = cid(metric), pal = phasePal(C, series.map(([p]) => p));
    if (ch) ch.geom = { X, Y, maxLen, series, pal };   /* shared with hover for the readout dots */
    let s = '<svg id="svg-' + k + '" viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, ui-monospace, monospace">';
    s += '<defs>';
    series.forEach((_, i) => { const c = pal[i];
      s += '<linearGradient id="g-' + k + '-' + i + '" x1="0" y1="0" x2="0" y2="1">'
        + '<stop offset="0" stop-color="' + c + '" stop-opacity="0.13"/>'
        + '<stop offset="0.8" stop-color="' + c + '" stop-opacity="0.01"/>'
        + '<stop offset="1" stop-color="' + c + '" stop-opacity="0"/></linearGradient>'; });
    s += '</defs>';
    /* y grid — nice ticks; on log scale they snap to powers of ten */
    const yTicks = useLog ? logTicks(mn, mx, Math.max(3, Math.round((H - PT - PB) / 52)))
                          : niceTicksLinear(mn, mx, Math.max(3, Math.round((H - PT - PB) / 52)));
    yTicks.forEach((tv) => {
      const gy = PT + (1 - (tv - mn) / (mx - mn)) * (H - PT - PB);
      if (gy < PT - 0.5 || gy > H - PB + 0.5) return;
      s += '<line x1="' + PL + '" y1="' + gy.toFixed(1) + '" x2="' + (W - PR) + '" y2="' + gy.toFixed(1) + '" stroke="' + C.line + '" stroke-width="1" vector-effect="non-scaling-stroke"/>';
      s += '<text x="' + (PL - 9) + '" y="' + (gy + 3.8).toFixed(1) + '" text-anchor="end" font-size="11" fill="' + C.faint + '">' + fmtTick(useLog ? Math.pow(10, tv) : tv) + '</text>';
    });
    /* baseline */
    s += '<line x1="' + PL + '" y1="' + (H - PB) + '" x2="' + (W - PR) + '" y2="' + (H - PB) + '" stroke="' + C.line + '" stroke-width="1" vector-effect="non-scaling-stroke"/>';
    /* x labels — epoch numbers */
    const st = xTickStep(maxLen, W - PL - PR);
    for (let e = st; e <= maxLen; e += st) {
      s += '<text x="' + X(e - 1).toFixed(1) + '" y="' + (H - PB + 19) + '" text-anchor="middle" font-size="11" fill="' + C.faint + '">' + e + '</text>';
    }
    /* axis titles — lowercase "epoch" beneath the x ticks, the metric name
       (lowercase) rotated up the y gutter; a quiet, understated caption */
    const axMidX = PL + (W - PL - PR) / 2, axMidY = PT + (H - PT - PB) / 2;
    const yTitle = String(metric).replace(/_/g, ' ').toLowerCase();
    s += '<text x="' + axMidX.toFixed(1) + '" y="' + (H - 7) + '" text-anchor="middle" font-size="11" letter-spacing="0.3" fill="' + C.dim + '">epoch</text>';
    s += '<text x="15" y="' + axMidY.toFixed(1) + '" text-anchor="middle" font-size="11" letter-spacing="0.3" fill="' + C.dim + '" transform="rotate(-90 15 ' + axMidY.toFixed(1) + ')">' + esc(yTitle) + '</text>';
    /* best-epoch marker */
    if (bestEpoch && bestEpoch >= 1 && bestEpoch <= maxLen) {
      const bx = X(bestEpoch - 1);
      s += '<line x1="' + bx.toFixed(1) + '" y1="' + PT + '" x2="' + bx.toFixed(1) + '" y2="' + (H - PB) + '" stroke="' + C.gold + '" stroke-opacity="0.55" stroke-width="1" stroke-dasharray="3 4" vector-effect="non-scaling-stroke"/>';
      s += '<text x="' + bx.toFixed(1) + '" y="' + (PT - 8) + '" text-anchor="middle" font-size="10.5" font-weight="600" fill="' + C.gold + '">★ best</text>';
    }
    /* series: area + line (stride-downsampled when very long) */
    const stride = maxLen > 1500 ? Math.ceil(maxLen / 1500) : 1;
    series.forEach(([, vals], i) => {
      const c = pal[i];
      let dLine = '', first = true, lastI = 0;
      for (let j = 0; j < vals.length; j += stride) {
        if (useLog && vals[j] <= 0) continue;
        dLine += (first ? 'M' : 'L') + X(j).toFixed(1) + ' ' + Y(vals[j]).toFixed(1); first = false; lastI = j;
      }
      const le = vals.length - 1;
      if (lastI !== le && !(useLog && vals[le] <= 0)) dLine += 'L' + X(le).toFixed(1) + ' ' + Y(vals[le]).toFixed(1);
      if (!dLine) return;
      const dArea = dLine + 'L' + X(le).toFixed(1) + ' ' + (H - PB) + 'L' + X(0).toFixed(1) + ' ' + (H - PB) + 'Z';
      s += '<path d="' + dArea + '" fill="url(#g-' + k + '-' + i + ')"/>';
      s += '<path d="' + dLine + '" fill="none" stroke="' + c + '" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>';
    });
    /* hover crosshair + one readout dot per series (positioned by attachHover) */
    s += '<line class="hovline" x1="0" y1="' + PT + '" x2="0" y2="' + (H - PB) + '" stroke="' + C.faint + '" stroke-opacity="0.6" stroke-width="1" vector-effect="non-scaling-stroke" visibility="hidden"/>';
    series.forEach((_, i) => { s += '<circle class="hovdot" r="4" fill="' + pal[i] + '" stroke="' + C.bg + '" stroke-width="1.6" visibility="hidden"/>'; });
    s += '</svg>';
    return s;
  }

  /* shared hover tooltip */
  const tip = document.createElement('div'); tip.className = 'tip'; document.body.appendChild(tip);
  function attachHover(card, metric) {
    const body = card.querySelector('.chart-body');
    body.addEventListener('pointermove', (e) => {
      const ch = charts[metric], svg = body.querySelector('svg');
      if (!ch || !svg || !ch.phases) return;
      const rect = svg.getBoundingClientRect();
      const series = Object.entries(ch.phases).filter(([, v]) => v && v.length);
      if (!series.length) return;
      const maxLen = Math.max.apply(null, series.map(([, v]) => v.length));
      const W = svg.viewBox.baseVal.width;
      const xv = (e.clientX - rect.left) / rect.width * W;
      const idx = clamp(Math.round((xv - PL) / Math.max(W - PL - PR, 1) * (maxLen - 1)), 0, maxLen - 1);
      const gx = maxLen < 2 ? PL + (W - PL - PR) / 2 : PL + (idx / (maxLen - 1)) * (W - PL - PR);
      const hov = svg.querySelector('.hovline');
      if (hov) { hov.setAttribute('x1', gx); hov.setAttribute('x2', gx); hov.setAttribute('visibility', 'visible'); }
      const dots = svg.querySelectorAll('.hovdot');
      series.forEach(([, vals], i) => { const dot = dots[i]; if (!dot) return;
        if (ch.geom && idx < vals.length && (!ch.log || vals[idx] > 0)) {
          dot.setAttribute('cx', ch.geom.X(idx).toFixed(1)); dot.setAttribute('cy', ch.geom.Y(vals[idx]).toFixed(1));
          dot.setAttribute('visibility', 'visible');
        } else dot.setAttribute('visibility', 'hidden'); });
      const C = themeColors(), pal = phasePal(C, series.map(([p]) => p));
      let html = '<div class="tip-title">Epoch ' + (idx + 1) + '</div>';
      series.forEach(([ph, vals], i) => { if (idx >= vals.length) return;
        html += '<div class="tip-row"><span class="tk"><i style="background:' + pal[i] + '"></i>' + esc(titleCase(ph)) + '</span><span class="tv">' + fmt(vals[idx]) + '</span></div>'; });
      tip.innerHTML = html; tip.classList.add('show');
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let px = e.clientX + 14, py = e.clientY - th - 10;
      if (px + tw > window.innerWidth - 8) px = e.clientX - tw - 14;
      if (py < 8) py = e.clientY + 16;
      tip.style.left = px + 'px'; tip.style.top = py + 'px';
    });
    body.addEventListener('pointerleave', () => { tip.classList.remove('show');
      const hov = body.querySelector('.hovline'); if (hov) hov.setAttribute('visibility', 'hidden');
      body.querySelectorAll('.hovdot').forEach((dt) => dt.setAttribute('visibility', 'hidden')); });
  }

  /* export the chart as a standalone SVG that mirrors the on-page card: a
     header band (title + per-phase legend) over the plot, on the theme bg */
  function saveSVG(metric) {
    const svg = document.getElementById('svg-' + cid(metric)); if (!svg) return;
    const C = themeColors(), NS = 'http://www.w3.org/2000/svg';
    const vb = svg.getAttribute('viewBox').split(' ');
    const W = +vb[2], H = +vb[3], HEAD = 42;
    const clone = svg.cloneNode(true); clone.removeAttribute('id');
    const hov = clone.querySelector('.hovline'); if (hov) hov.remove();
    clone.querySelectorAll('.hovdot').forEach((dt) => dt.remove());
    const inner = clone.innerHTML;
    const title = titleCase(metric);
    const SANS = 'Inter, system-ui, sans-serif';
    let head = '<text x="' + PL + '" y="26" font-family="' + SANS + '" font-size="15" font-weight="700" fill="' + C.text + '">' + esc(title) + '</text>';
    let lx = PL + title.length * 8.6 + 22;
    const ch = charts[metric];
    if (ch && ch.phases) {
      const ser = Object.entries(ch.phases).filter(([, v]) => v && v.length);
      const pal = phasePal(C, ser.map(([p]) => p));
      ser.forEach(([ph], i) => { const name = titleCase(ph);
        head += '<circle cx="' + lx + '" cy="21" r="3.4" fill="' + pal[i] + '"/>';
        head += '<text x="' + (lx + 9) + '" y="25" font-family="' + SANS + '" font-size="12" fill="' + C.dim + '">' + esc(name) + '</text>';
        lx += 9 + name.length * 7.1 + 18; });
    }
    const out = '<svg xmlns="' + NS + '" viewBox="0 0 ' + W + ' ' + (H + HEAD) + '" font-family="JetBrains Mono, ui-monospace, monospace">'
      + '<rect width="' + W + '" height="' + (H + HEAD) + '" fill="' + C.bg + '"/>'
      + head + '<g transform="translate(0 ' + HEAD + ')">' + inner + '</g></svg>';
    const blob = new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n' + out], { type: 'image/svg+xml' });
    const a = document.createElement('a'); a.download = metric.replace(/_/g, '-') + '.svg';
    a.href = URL.createObjectURL(blob); a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  /* charts render at the exact pixel width of their container, so the
     typography stays the same size on any screen; a ResizeObserver
     re-renders them when the layout reflows */
  const ro = new ResizeObserver((entries) => {
    for (const en of entries) {
      const m = en.target._metric, ch = charts[m];
      if (!ch) continue;
      const w = Math.round(en.contentRect.width);
      if (Math.abs(w - (ch.lastW || 0)) > 8) renderChart(m);
    }
  });
  function renderChart(metric) {
    const ch = charts[metric]; if (!ch) return;
    const W = Math.max(320, Math.round(ch.body.clientWidth || 560));
    /* uniform height across the grid — both columns share a width, so a
       width-derived height keeps every chart identical. A taller ratio keeps
       the curves readable rather than stretched flat. */
    const H = clamp(Math.round(W * 0.64), 300, 410);
    ch.lastW = W;
    ch.body.innerHTML = chartSVG(metric, ch.phases, ch.log, bestEpochGlobal, W, H, ch);
    const leg = ch.card.querySelector('.legend');
    if (leg) { const C = themeColors();
      const ser = Object.entries(ch.phases).filter(([, v]) => v && v.length);
      const pal = phasePal(C, ser.map(([p]) => p));
      leg.innerHTML = ser.map(([ph], i) => '<span><i style="background:' + pal[i] + '"></i>' + esc(titleCase(ph)) + '</span>').join(''); }
  }

  function makeCard(metric) {
    const card = document.createElement('div');
    card.className = 'chart';
    card.innerHTML = '<div class="chart-head"><span class="chart-name">' + esc(titleCase(metric)) + '</span>'
      + '<span class="legend" aria-hidden="true"></span>'
      + '<div class="chart-acts">'
      + '<button class="cbtn lg" title="Toggle logarithmic scale">log</button>'
      + '<button class="cbtn dl" title="Save as vector SVG">svg</button>'
      + '</div></div><div class="chart-body"></div>';
    return card;
  }

  function ensureChart(metric, mount) {
    if (charts[metric]) return charts[metric];
    const card = makeCard(metric);
    card.style.animationDelay = Math.min(Object.keys(charts).length, 8) * 0.05 + 's';  /* gentle stagger as charts appear */
    mount.appendChild(card);
    const ch = charts[metric] = { card, body: card.querySelector('.chart-body'), log: false, phases: {}, sig: '' };
    card.querySelector('.dl').addEventListener('click', () => saveSVG(metric));
    const lgBtn = card.querySelector('.lg');
    lgBtn.addEventListener('click', () => {
      const ok = Object.values(ch.phases).every((v) => v.every((x) => typeof x === 'number' && x > 0));
      if (!ch.log && !ok) return;
      ch.log = !ch.log; lgBtn.classList.toggle('on', ch.log); renderChart(metric);
    });
    attachHover(card, metric);
    ch.body._metric = metric;
    ro.observe(ch.body);
    return ch;
  }

  function updateCharts(metrics, primary) {
    const grid = el('charts-grid');
    const names = Object.keys(metrics || {}).filter((m) => Object.values(metrics[m] || {}).some((v) => v && v.length));
    names.sort((a, b) => (a === primary ? -1 : 1) - (b === primary ? -1 : 1) || a.localeCompare(b));
    /* every chart shows the same epoch-level history — the primary (loss) first.
       Until the first epoch completes there are no points yet, so the panel
       holds the "awaiting" placeholder rather than a one-off per-step view. */
    for (const metric of names) {
      const ch = ensureChart(metric, grid);
      ch.phases = metrics[metric];
      const sig = Object.entries(ch.phases).map(([p, v]) => p + ':' + (v ? v.length : 0)).join('|') + '|' + ch.log + '|' + bestEpochGlobal + '|' + themeEpoch;
      if (sig !== ch.sig) { ch.sig = sig; renderChart(metric); }
    }
  }

  /* ── main loop ─────────────────────────────────────────── */
  let lastData = null, staticDone = false;
  async function fetchData() { try { const r = await fetch(DATA_FILE + '?_=' + Date.now()); if (r.ok) { lastData = await r.json(); return lastData; } } catch (e) {} return lastData; }
  async function tick() {
    const d = window.__TRAIN4ALL_DATA__ || await fetchData();
    if (!d) return;
    if (d.status === 'training' && Date.now() - (d.last_update_ms || 0) > STALE_MS) d.status = 'stopped';

    const grad = !!d.is_gradient_phase, ni = d.epochs_no_improve || 0, done = d.status === 'completed';
    const active = !!(d.last_phase && d.max_step && d.current_step);
    const mode = done ? 'completed' : d.status === 'stopped' ? 'stopped'
      : !active ? 'idle' : grad ? (ni >= 1 ? 'stagnant' : 'training') : 'validating';
    setMode(mode, ni);
    setText('m-start', d.started_at || '—'); setText('m-elapsed', d.elapsed || '—');

    /* progress fractions advance while a phase is active and SNAP to the phase's
       true boundary when it finishes — the inter-phase flush update carries no step
       info, and polling rarely catches the exact final step, so without the snap
       both the inner ring and the % would stall just short of the boundary.
         · inner ring (current-phase steps): snaps to a full circle.
         · epoch fraction (overall % + bar): snaps to train_steps/total after the
           training phase (validation still pending) or to 1 after validation /
           when there is no validation — so it reliably reaches 100 % at the end. */
    const ef = epochFraction(d);
    const TS = d.train_steps || 0, VS = d.val_steps || 0, total = TS + VS;
    if (done) { prog.epFrac = 1; prog.stepFrac = 1; }
    else if (active) {
      prog.epFrac = ef;                                    // non-null whenever active
      prog.stepFrac = clamp(d.current_step / d.max_step, 0, 1);
      prog.lastGrad = grad; prog.lastPhase = d.last_phase; prog.lastMax = d.max_step; prog.ran = true;
    } else if (prog.ran) {                                 // a phase just finished
      prog.stepFrac = 1;
      prog.epFrac = (prog.lastGrad && VS > 0 && total > 0) ? clamp(TS / total, 0, 1) : 1;
    }
    const E = d.max_epoch || 0, completed = Math.max(d.current_epoch || 0, 1) - 1;
    const overall = done ? 1 : (E ? clamp((completed + prog.epFrac) / E, 0, 1) : 0);

    /* phase colour drives the inner step ring. While running it tracks the live
       phase (train blue / val purple); once a phase's steps finish it turns gold
       and holds full through the gap before the next phase — a small note of
       completion — and at standby (before any epoch) it is a neutral grey. On run
       completion the inner ring is blanked entirely; the gold crown moves to the
       outer run ring instead. */
    if (active) prog.phaseCol = grad ? 'var(--st-training)' : 'var(--st-validating)';
    else if (prog.ran) prog.phaseCol = 'var(--st-completed)';   /* steps done, between phases → gold */
    else prog.phaseCol = 'var(--st-idle)';
    document.documentElement.style.setProperty('--phasecol', prog.phaseCol);

    el('run-line').style.width = (overall * 100).toFixed(2) + '%';
    el('run-line').classList.toggle('done', done);   /* the overall-run hairline goes gold to match the gauge */
    el('ring-run').classList.toggle('done', done);   /* crown the overall-run ring gold at completion */
    setRing('ring-run', RING_RUN_C, overall);
    setRing('ring-step', RING_STEP_C, done ? 0 : prog.stepFrac);   /* inner ring is the live phase only — blank when done */
    tween(el('ov-pct'), Math.round(overall * 100), (v) => String(Math.round(v)));
    setText('ov-label', done ? 'Complete' : 'Overall');
    updateGauge(d);

    /* phase / epoch / step readout — the finished phase is held through the
       inter-phase gap (shown at max/max) so it never blanks while the rings
       already read complete */
    const showPhase = !done && (active || prog.ran);
    const phName = active ? d.last_phase : prog.lastPhase;
    const phb = el('hm-phase');
    if (showPhase && phName) {
      phb.style.display = ''; phb.textContent = titleCase(phName);
      phb.className = 'hm-phase ' + (prog.lastGrad ? 'is-train' : 'is-eval');
    } else phb.style.display = 'none';
    setText('hm-epoch', 'Epoch ' + (d.current_epoch || 0) + ' / ' + (E || '—'));
    const stepMax = active ? d.max_step : (prog.ran ? prog.lastMax : 0);
    const stepCur = active ? (d.current_step || 0) : stepMax;   // gap → completed (max / max)
    setText('hm-step', done ? 'Complete' : (stepMax ? 'Step ' + stepCur + ' / ' + stepMax : 'Step — / —'));

    updateSpeed(d, grad);
    const eta = etaSeconds(d), primary = pickPrimary(d);
    const stepping = active && d.status === 'training';   /* a step is actively in progress */
    updateKpis(d, primary, stepping);
    setText('k-speed', stepping && spd.ema ? (spd.ema < 10 ? spd.ema.toFixed(1) : String(Math.round(spd.ema))) : '—');
    setText('k-eta', stepping && spd.ema && isFinite(eta) ? fmtDur(eta) : '—');
    setText('k-eta-sub', done ? 'complete'
      : (stepping && spd.ema && isFinite(eta) ? 'ends ~' + fmtClock(new Date(Date.now() + eta * 1000)) : 'remaining'));

    updateTelemetry(d);
    renderSpark(d);

    bestEpochGlobal = d.best_epoch || 0;
    updateCharts(d.epoch_metrics || {}, primary);

    const liveV = primary !== null ? liveValue(d, primary) : undefined;
    document.title = Math.round(overall * 100) + '%' + (typeof liveV === 'number' ? ' · ' + titleCase(primary) + ' ' + fmt(liveV) : '') + ' — train4all';

    if (!staticDone && ((d.config && Object.keys(d.config).length) || (d.env_summary && Object.keys(d.env_summary).length))) {
      renderKV('cfg-grid', d.config); renderKV('env-grid', d.env_summary); renderKV('model-grid', d.model_summary);
      staticDone = true;
    }
    setText('footer', 'train4all' + (VERSION ? ' v' + VERSION : '') + ' · ' + (d.updated_at || ''));
  }

  setFavicon(accentColor());
  setInterval(tick, POLL_MS);
  tick();
});
</script>
</body>
</html>"""
