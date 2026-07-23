import abc
import inspect
import json
import math
import random
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from train4all.trainer.checkpoint import Checkpoint
from train4all.trainer.phase import Phase
from train4all.utils import (
    DEFAULT_KEY_WIDTH,
    Dashboard,
    DashboardConfig,
    GpuProbe,
    LogLevel,
    MetricTable,
    PhaseSpec,
    TrainerLogger,
    UnifiedLogger,
    copy_dir,
    cuda_index,
    empty_cuda_cache,
    env_summary,
    get_metric_plot_filename,
    get_metric_plot_title,
    gpu_temperature,
    print_dict_tree,
    remove_dir,
    replace_dict_keys,
    save_curves_plot,
    separator_rule,
)

__all__ = ["BaseTrainer"]

type ModuleSpec = str | nn.Module | list[str | nn.Module]
type _Scheduler = LRScheduler | ReduceLROnPlateau


def _require_setup[F: Callable[..., Any]](func: F) -> F:
    """Ensure ``setup()`` has been called before the decorated method runs."""
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        self.ensure_setup()
        return func(self, *args, **kwargs)
    return wrapper  # type: ignore[return-value]


class BaseTrainer(abc.ABC):
    """
    Generic, extensible training framework for PyTorch models.

    Subclass this and implement:
        - ``setup()``
        - ``compute_loss()``
        - ``compute_metrics()``

    Optionally override ``compute_test_metrics()`` to report heavier, test-only
    metrics during the final evaluation, or ``get_batch_weight()`` to change how
    per-step metrics are weighted when averaged over an epoch.

    An epoch is whatever sequence of :class:`~train4all.trainer.phase.Phase`
    objects you hand to :meth:`train` — the loop has no built-in notion of
    "train" or "val" beyond the names you give them::

        trainer.train(
            Phase("train", train_loader, training=True),
            Phase("val", val_loader),
        )

    Args:
        num_epochs: Total number of training epochs. Required by ``train()``;
            leave unset (``None``) to only evaluate (``test()``) or inspect checkpoints.
        batch_size: Batch size (informational; not used internally).
        learning_rate: Learning rate(s) forwarded to the optimizer in ``setup()``.
            ``None`` (default) sets no rate — leave it unset for learning-rate-free
            optimizers (e.g. Prodigy, D-Adaptation, Schedule-Free), which then
            keep ``learning_rate`` out of the saved config.
        max_grad_norm: Clip the global gradient norm to this value before each
            optimizer step. Disabled when ``None``. Gradients are unscaled
            first, so this is correct under fp16 AMP as well.
        accumulation_steps: Accumulate gradients over this many steps before
            each optimizer update, simulating a larger effective batch with no
            extra memory. Gradients are normalized as the weighted mean
            ``Σ wᵢ∇Lᵢ / Σ wᵢ`` with per-batch weights ``wᵢ`` from
            ``get_batch_weight`` — keep those equal to the loss's denominator
            (e.g. token count for per-token losses) so variable-length batches
            are not mis-weighted. Defaults to ``1`` (update every step; the
            weight is then unused).
        amp: Automatic mixed precision. ``None`` (default) auto-enables bf16 on
            CUDA and is a no-op elsewhere; an explicit ``True``/``"bf16"``/
            ``"fp16"`` is warned about when the device is not CUDA, and
            ``False`` forces the full-precision path used to reproduce a run.
        tf32: Allow TF32 for fp32 matmuls/convolutions on CUDA (Ampere+), and
            enable the cuDNN autotuner. ``None`` (default) auto-enables it only
            when ``seed`` is unset — speed when not reproducing, full precision
            when you are. ``True``/``False`` force it on/off. CUDA-only; a no-op
            on CPU/MPS. Independent of and complementary to ``amp``.
        patience: Early-stopping patience (epochs without improvement). Disabled if ``None``.
        monitor: Metric name that drives best-checkpoint selection and early
            stopping. Defaults to ``"loss"``. Read each epoch from the phase named
            by ``monitor_phase``.
        monitor_mode: ``"min"`` to treat lower ``monitor`` values as better
            (e.g. loss) or ``"max"`` for higher-is-better metrics (e.g. accuracy, F1).
        monitor_phase: Name of the phase the ``monitor`` metric is read from.
            Defaults to ``"val"`` — but it is just a name, so any phase in the
            schedule can drive selection and early stopping.
        device: Device string (e.g. ``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``).
            Auto-detected when ``None`` — prefers CUDA, then MPS, then CPU.
            On a multi-GPU machine, select a specific GPU with ``"cuda:<index>"``.
        seed: Random seed for reproducibility. Disabled if ``None``.
        run_dir: Output directory for checkpoints, metrics, logs, and plots.
        run_snapshot_dir: Directory for a lightweight snapshot copy of ``run_dir``.
            Snapshotting is disabled when ``None``.
        run_snapshot_exclude: Top-level entry names left out of every snapshot —
            e.g. ``["checkpoints"]`` to mirror only the metrics and plots.
            ``None`` (default) excludes nothing. What a mirror leaves behind is a
            standing property of the run, so it is configured here rather than
            only at a call site: the per-epoch mirror ``train()`` takes is
            unattended, and could otherwise never be anything but complete.
            :meth:`snapshot_run` overrides it for a single call.
        resume: Resume from the latest checkpoint at the start of training.
        save_interval: Save a periodic checkpoint every *N* epochs.
        record_step_metrics: Record per-step metrics. The master switch; each
            phase decides whether it takes part (see ``Phase.record_steps``,
            which defaults to the training phases).
        step_metric_names: Step metric names to record. ``None`` records all.
        pbar_metric_names: Metric names shown in the tqdm postfix. ``None`` hides all
            metrics; GPU memory is always shown on CUDA regardless.
        use_progress_bar: Show tqdm progress bars during epoch iteration.
        debug_mode: Enable debug-level logging (forwarded to the logger).
        logger: Any object satisfying the :class:`TrainerLogger` protocol.
            A default ``UnifiedLogger`` is created if ``None``.
        use_dashboard: Enable the live web dashboard.
        dashboard_config: Dashboard appearance and behaviour settings.
            A default :class:`DashboardConfig` is used when ``None``.
    """

    # Class-level constants — override in a subclass to customize.

    # ── Output layout: subdirectories of ``run_dir`` ──────────────────────────
    _CHECKPOINTS_DIRNAME: str = "checkpoints"
    _METRICS_DIRNAME: str     = "metrics"
    _PLOTS_DIRNAME: str       = "plots"

    # ── Output layout: file names ─────────────────────────────────────────────
    _LOG_FILENAME: str    = "log.txt"
    _CONFIG_FILENAME: str = "config.json"

    # ── Checkpoint file stems ─────────────────────────────────────────────────
    # The on-disk format itself lives in :class:`Checkpoint` (the schema and its
    # version are owned there); these are only the file names under the run.
    _CHECKPOINT_LATEST: str = "latest"
    _CHECKPOINT_BEST: str   = "best"

    # ── Metrics file stems ────────────────────────────────────────────────────
    _METRICS_EPOCH: str = "epoch_metrics"
    _METRICS_STEP: str  = "step_metrics"

    # ── Name of the phase ``test(loader)`` builds ─────────────────────────────
    # Only the default name of the shorthand's phase — the behaviour (no gradients,
    # ``compute_test_metrics``) travels in the :class:`Phase` that shorthand builds,
    # not in a lookup keyed off this string. Pass ``test()`` a Phase to name it
    # anything else.
    _TEST_PHASE: str = "test"

    # ── GPU probe tunables ────────────────────────────────────────────────────
    _GPU_TEMP_WARN_C: int = 85   # warn above this GPU temperature (°C)
    _GPU_MEM_TTL_S: float = 2.0  # cache nvidia-smi memory reads for this long

    # ── Console / display tunables ────────────────────────────────────────────
    # The width itself is owned by ``log_utils`` (see ``DEFAULT_KEY_WIDTH``), so a
    # trainer's tables and a standalone ``Checkpoint.print_summary()`` agree by
    # reference, not by coincidence.
    _KEY_WIDTH: int           = DEFAULT_KEY_WIDTH
    _KEEP_PROGRESS_BAR: bool  = False  # persist tqdm bars after an epoch completes
    _DASH_THROTTLE_S: float   = 0.5    # minimum seconds between dashboard step writes
    _DASH_EXTRA_WAIT_S: float = 0.5    # extra wait after dashboard finalize

    def __init__(
        self,
        num_epochs: int | None = None,
        *,
        batch_size: int | None = None,
        learning_rate: float | dict[str, float] | None = None,
        max_grad_norm: float | None = None,
        accumulation_steps: int = 1,
        amp: bool | str | None = None,
        tf32: bool | None = None,
        patience: int | None = None,
        monitor: str = "loss",
        monitor_mode: str = "min",
        monitor_phase: str = "val",
        device: str | None = None,
        seed: int | None = None,
        run_dir: Path | str = "run",
        run_snapshot_dir: Path | str | None = None,
        run_snapshot_exclude: list[str] | None = None,
        resume: bool = True,
        save_interval: int | None = None,
        record_step_metrics: bool = False,
        step_metric_names: list[str] | None = None,
        pbar_metric_names: list[str] | None = None,
        use_progress_bar: bool = True,
        debug_mode: bool = False,
        logger: TrainerLogger | None = None,
        use_dashboard: bool = False,
        dashboard_config: DashboardConfig | None = None,
    ) -> None:
        # ── Training / optimization ───────────────────────────────────────────
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = self._validate_accumulation_steps(accumulation_steps)
        self.amp = amp
        self.tf32 = tf32

        # ── Early stopping / monitoring ───────────────────────────────────────
        self.patience = patience
        self.monitor = monitor
        self.monitor_mode = self._validate_monitor_mode(monitor_mode)
        self.monitor_phase = monitor_phase

        # ── Device / reproducibility ──────────────────────────────────────────
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        ))
        # On a multi-GPU machine, make the explicitly chosen GPU the process
        # default too, so stray ``device="cuda"`` allocations land on it.
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)
        self.seed = seed

        # ── Persistence ───────────────────────────────────────────────────────
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir = self.run_dir / self._CHECKPOINTS_DIRNAME
        self._metrics_dir     = self.run_dir / self._METRICS_DIRNAME
        self._plots_dir       = self.run_dir / self._PLOTS_DIRNAME
        self.run_snapshot_dir = Path(run_snapshot_dir) if run_snapshot_dir else None
        self.run_snapshot_exclude = run_snapshot_exclude
        self.resume = resume
        self.save_interval = save_interval

        # ── Metric recording ──────────────────────────────────────────────────
        self.record_step_metrics = record_step_metrics
        self.step_metric_names = step_metric_names
        self.pbar_metric_names = pbar_metric_names

        # ── Display / logging ─────────────────────────────────────────────────
        self.use_progress_bar = use_progress_bar
        self.debug_mode = debug_mode
        self.logger = logger or self._create_default_logger()

        # ── Internal: models / optimization objects ───────────────────────────
        self._models: dict[str, nn.Module] = {}
        self._compiled_models: set[str] = set()
        self._optimizer: Optimizer | None = None
        self._scheduler: _Scheduler | None = None

        # ── Internal: training state ──────────────────────────────────────────
        self._current_epoch: int = 0
        self._best_metric: float = self._worst_metric()
        self._best_epoch: int | None = None
        self._epochs_no_improve: int = 0

        # ── Internal: gradient-accumulation weighting (see _optimizer_step) ───
        # Running weight sum of the in-progress accumulation cycle; unused when
        # accumulation_steps == 1.
        self._cycle_weight: float = 0.0

        # ── Internal: metrics ─────────────────────────────────────────────────
        self._epoch_metrics: MetricTable = {}
        self._step_metrics: MetricTable = {}

        # ── Internal: misc state ──────────────────────────────────────────────
        self._setup_done: bool = False
        self._cache: dict[str, Any] = {}
        self._ckpt_excludes: set[str] = set()
        self._ckpt_extras: dict[str, Any] = {}
        self._last_dash_write: float = 0.0

        # ── Internal: GPU-memory probe ────────────────────────────────────────
        # The probe owns its own NVML handle and smi cache, so the progress bar
        # never pays an init/shutdown per iteration (see ``utils.system``).
        self._gpu = GpuProbe(self._cuda_index, ttl_s=self._GPU_MEM_TTL_S)

        # ── Internal: AMP / TF32 (depends on the attributes set above) ────────
        # AMP resolves to state the step loop reads on every batch; TF32 resolves
        # to global torch flags, so it keeps nothing. _init_tf32 runs after
        # reset_seed, whose cuDNN determinism flags it may relax.
        self._amp_enabled, self._amp_dtype, self._scaler = self._init_amp()
        self.reset_seed()  # applies self.seed to the RNGs when set
        self._init_tf32()

        # Reproducibility config — only the arguments the caller actually
        # customized (anything left at its default is omitted), so the saved
        # config is minimal and round-trips via :meth:`from_config`.
        self._config: dict[str, Any] = self._customized_config({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "accumulation_steps": accumulation_steps,
            "amp": amp,
            "tf32": tf32,
            "patience": patience,
            "monitor": monitor,
            "monitor_mode": monitor_mode,
            "monitor_phase": monitor_phase,
            "seed": seed,
        })
        # Record the *resolved* device unconditionally (not via the filter
        # above): a raw ``device=None`` would be dropped and re-resolved
        # differently on another host, so pinning it keeps reproduction exact —
        # ``from_config`` then fails loudly on a host that lacks it (pass
        # ``device=`` to retarget).
        self._config["device"] = str(self.device)

        dashboard_config = dashboard_config or DashboardConfig()
        self._dashboard: Dashboard | None = (
            Dashboard(dashboard_config, self.run_dir) if use_dashboard else None
        )
        # Tracked even when the dashboard is disabled, so a fresh run can delete a
        # previous run's dashboard files (see ``clear_artifacts``).
        self._dashboard_files: tuple[Path, ...] = (
            self.run_dir / dashboard_config.filename,
            self.run_dir / dashboard_config.data_filename,
        )

    # ── Abstract Methods ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def setup(self) -> None:
        """
        Initialize and register models, optimizers, and schedulers.

        Called once before training or evaluation begins. Use
        ``set_models()``, ``set_optimizer()``, and ``set_scheduler()`` here.

        Example::

            def setup(self):
                self.backbone = Backbone()
                self.classifier = Classifier()
                self.set_models({"backbone": self.backbone, "classifier": self.classifier})
                self.freeze("backbone")
                optimizer = torch.optim.Adam(self.get_trainable_params(), lr=self.learning_rate)
                self.set_optimizer(optimizer)
        """

    @abc.abstractmethod
    def compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Compute the scalar loss for a batch.

        Intermediate tensors can be cached with ``set_cache()`` for reuse
        inside ``compute_metrics()``.

        Args:
            batch: A batch of input data.

        Returns:
            Scalar loss tensor.

        Example::

            def compute_loss(self, batch):
                x, y = batch["input"], batch["target"]
                logits = self.classifier(self.backbone(x))
                self.set_cache("logits", logits.detach())
                return F.cross_entropy(logits, y)
        """

    @abc.abstractmethod
    def compute_metrics(self, batch: Any) -> dict[str, float]:
        """
        Compute evaluation metrics for a batch.

        Args:
            batch: A batch of input data.

        Returns:
            Mapping of metric name to scalar value.

        Example::

            def compute_metrics(self, batch):
                preds = self.get_cache("logits").argmax(dim=1)
                acc = (preds == batch["target"]).float().mean().item()
                return {"accuracy": acc}
        """

    def compute_test_metrics(self, batch: Any) -> dict[str, float]:
        """
        Compute evaluation metrics for the final test phase.

        The metric function ``test(loader)`` gives the phase that shorthand builds.
        Final evaluation runs once, so it can afford heavier, report-only metrics
        (AUC, per-class F1, calibration, confusion matrices, …) that would be
        wasteful every epoch. The default delegates to ``compute_metrics``, so test
        mirrors validation until you override it.

        Nothing routes here by phase name, and no entry point injects it — this is
        simply the default the shorthand reaches for, and any phase can carry it
        (or any other metric function) explicitly::

            Phase("audit", audit_loader, metric_fn=self.compute_test_metrics)

        Args:
            batch: A batch of test data.

        Returns:
            Mapping of metric name to scalar value.

        Example::

            def compute_test_metrics(self, batch):
                metrics = self.compute_metrics(batch)  # reuse the shared metrics
                metrics["auc"] = roc_auc_score(...)    # plus report-only extras
                return metrics
        """
        return self.compute_metrics(batch)

    def get_batch_weight(self, batch: Any) -> int:
        """
        Return the weight of ``batch`` for metric averaging and gradient
        accumulation.

        Metrics are averaged as ``Σ(metric × weight) / Σweight`` across the
        steps in an epoch. When ``accumulation_steps > 1`` the same weight also
        normalizes the accumulated gradient; with ``accumulation_steps == 1``
        the weight affects only metric averaging, not the gradient.

        The weight must equal the denominator the per-batch loss was divided by
        — only then does the accumulated gradient reconstruct the true mean over
        the effective batch (and the reported loss become a true weighted mean);
        a mismatched weight biases both. The default is the sample count, which
        fits a per-sample mean loss.

        For a language/vision-language model whose loss is a mean over the
        supervised tokens, that denominator is the ``labels != -100`` count —
        *not* ``attention_mask.sum()``, since prompt/image tokens are masked out
        of the loss and so are absent from its denominator.

        Args:
            batch: The current batch.

        Returns:
            A positive integer weight.

        Example::

            def get_batch_weight(self, batch):
                # HF LM/VLM loss is a mean over labels != -100; weight must match.
                return int((batch["labels"] != -100).sum())
        """
        return self._batch_size(batch)

    # ── Lifecycle Hooks ───────────────────────────────────────────────────────

    def on_set_training_mode(self, training: bool) -> None:
        """
        Called whenever the training or evaluation mode is toggled.

        Args:
            training: ``True`` when switching to training mode, ``False`` for evaluation.
        """

    def on_after_backward(self) -> None:
        """
        Called immediately after ``loss.backward()``, before gradient unscaling
        and clipping.

        Under fp16 AMP the gradients are still scaled by the loss-scale factor
        at this point; use :meth:`on_before_optimizer_step` for the unscaled,
        post-clip view.
        """

    def on_before_optimizer_step(self) -> None:
        """
        Called after backward (and gradient unscaling/clipping) and immediately
        before the optimizer step, while gradients are populated.

        Useful for gradient inspection, logging gradient norms, or applying a
        custom clipping/regularization scheme beyond ``max_grad_norm``.
        """

    def on_training_start(self) -> None:
        """Called once immediately before the main training loop begins."""

    def on_training_end(self) -> None:
        """Called once immediately after the main training loop ends."""

    def on_exception(self, exc: BaseException) -> None:
        """
        Called when the training loop is aborted by an exception, including
        ``KeyboardInterrupt`` (Ctrl-C). The exception is re-raised afterwards.

        No checkpoint is saved automatically — a mid-epoch save would persist an
        incomplete state. Override to react to the failure (custom logging,
        alerting, or your own recovery logic).

        Args:
            exc: The exception that aborted training.
        """

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Called while a full checkpoint is being built, before it is written.

        Attach custom state to persist by indexing ``checkpoint`` like a dict
        (``checkpoint["ema"] = self.ema.state_dict()``) — the counterpart of
        :meth:`on_load_checkpoint`. Not called for weights-only saves.

        Args:
            checkpoint: The :class:`Checkpoint` about to be saved.
        """

    def on_load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Called after a full checkpoint has been restored, with the loaded
        :class:`Checkpoint`.

        Read back custom state written in :meth:`on_save_checkpoint` by indexing
        it (``checkpoint["ema"]``), or reach for its typed accessors. Extras from
        ``update_checkpoint_extras()`` restore automatically (read them via
        :meth:`get_checkpoint_extras`). Not called for weights-only loads.

        Args:
            checkpoint: The :class:`Checkpoint` that was just loaded.
        """

    def on_train_epoch_start(self, epoch: int) -> None:
        """
        Called at the start of each training-loop epoch.

        Args:
            epoch: Current epoch number (1-based).
        """

    def on_train_epoch_end(self, epoch: int) -> None:
        """
        Called at the end of each training-loop epoch, after artifacts are saved.

        Args:
            epoch: Current epoch number (1-based).
        """

    def on_phase_start(self, epoch: int | None, phase: Phase) -> None:
        """
        Called at the start of every phase, training or evaluation.

        The step cache has been cleared before this hook fires. The phase carries
        its own context — ``phase.name``, ``phase.loader``, ``phase.training`` —
        so a hook can branch on what the pass actually is rather than on its name.

        Args:
            epoch: Current epoch number, or ``None`` when called outside the training loop.
            phase: The phase about to run.
        """

    def on_phase_end(
        self, epoch: int | None, phase: Phase, metrics: dict[str, float],
    ) -> None:
        """
        Called at the end of every phase, training or evaluation.

        Epoch metrics for the completed phase are already recorded and accessible
        via :meth:`get_epoch_metrics` when this hook fires.

        Args:
            epoch: Current epoch number, or ``None`` when called outside the training loop.
            phase: The phase that just ran.
            metrics: Aggregated metrics computed during the phase.
        """

    def on_step_start(self, step: int | None, batch: Any, phase: Phase) -> None:
        """
        Called at the start of every step.

        Args:
            step: 1-based step index within the current phase, or ``None`` when called
                outside the standard epoch loop.
            batch: The batch about to be processed.
            phase: The phase this step belongs to.
        """

    def on_step_end(
        self, step: int | None, batch: Any, metrics: dict[str, float], phase: Phase,
    ) -> None:
        """
        Called at the end of every step.

        Args:
            step: 1-based step index within the current phase, or ``None`` when called
                outside the standard epoch loop.
            batch: The batch that was just processed.
            metrics: Step metrics from the phase's metric function, plus ``"loss"``.
            phase: The phase this step belongs to.
        """

    # ── Main Training Workflow ────────────────────────────────────────────────

    def train(self, *phases: Phase) -> None:
        """
        Train the model for the configured number of epochs.

        Each epoch runs *phases* in the order given. The loop knows nothing about
        "train" and "val" beyond the names you choose — the canonical run is::

            trainer.train(
                Phase("train", train_loader, training=True),
                Phase("val", val_loader),
            )

        and a schedule that measures expensive metrics on a slice of the training
        data, every fifth epoch, is the same expression with one more phase::

            trainer.train(
                Phase("train", train_loader, training=True, metric_fn=lambda _: {}),
                Phase("train_eval", train_subset_loader, every=5),
                Phase("val", val_loader),
            )

        Best-checkpoint selection and early stopping read ``monitor`` from the
        phase named by ``monitor_phase`` (``"val"`` by default), taking only the
        value produced *this* epoch — so a monitored phase that sits out an epoch
        (``every > 1``) yields no value rather than a stale one.

        Args:
            *phases: The phases of one epoch, in the order they run. Names must be
                unique.

        Raises:
            ValueError: If ``num_epochs`` was not set, if no phase was given, or
                if two phases share a name.
        """
        self._require_num_epochs()
        self._validate_phases(phases)
        self.prepare_training()

        if self.is_training_complete():
            self.print("\n⏹️  Training already completed.\n")
            return

        if self.should_stop_early():
            self.print("\n⏹️  Early stopping condition already met. No training will run.\n")
            return

        self.print_schedule_summary(*phases)
        self._dash_init(phases)

        start_time = datetime.now()
        self.print(f"\n🚀 Training started at {start_time:%Y-%m-%d %H:%M:%S}\n")
        if self._dashboard:
            self._dashboard.mark_started(start_time)
        self.on_training_start()

        # ``BaseException`` so a KeyboardInterrupt (Ctrl-C) or out-of-memory
        # error also reaches ``on_exception`` before propagating. No checkpoint
        # is written here: a mid-epoch save would persist an incomplete state,
        # so recovery is left to the hook (which is a no-op by default).
        try:
            for epoch, max_epoch in self.epoch_iterator():
                self.print(f"\n── Epoch {epoch} / {max_epoch}\n")
                self.on_train_epoch_start(epoch)

                # This epoch's results only. The monitor is read from here rather
                # than from the recorded history, so a phase skipped by ``every``
                # reports nothing instead of re-reporting its last value.
                results: dict[str, dict[str, float]] = {}
                for phase in phases:
                    if not phase.runs_at(epoch):
                        continue
                    results[phase.name] = self._execute_phase(phase, epoch=epoch)
                    self.print_metrics(results[phase.name], phase.name)

                monitor_value = results.get(self.monitor_phase, {}).get(self.monitor)
                self.finalize_train_epoch(monitor_value)
                self.save_artifacts()
                # Mirror the run once the epoch's artifacts are on disk, so the copy
                # is always a complete epoch. A no-op unless ``run_snapshot_dir`` is
                # set — but when it is, this is what makes the setting mean anything.
                # Bare on purpose: an unattended call can carry no policy of its own,
                # so the mirror's shape is the trainer's (``run_snapshot_exclude``).
                self.snapshot_run()
                self._dash_update()
                self.on_train_epoch_end(epoch)

                if self.should_stop_early():
                    self.print(f"⏹️  Early stopping triggered at epoch {epoch}.\n")
                    break

                self.print()
        except BaseException as exc:
            self.on_exception(exc)
            raise

        self.on_training_end()
        duration = datetime.now() - start_time
        self._dash_finalize()
        self.empty_cuda_cache()
        self.print(f"\n✅ Training completed. Duration: {str(duration).split('.')[0]}\n")

    @_require_setup
    def test(
        self,
        phase: Phase | DataLoader,
        use_best: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate the model once, outside the training loop.

        The final evaluation is one ordinary phase — nothing here is a special
        case — so it is spelled in the same vocabulary as :meth:`train`. Passing a
        DataLoader is shorthand for the canonical test phase::

            trainer.test(test_loader)
            trainer.test(Phase(self._TEST_PHASE, test_loader,
                               metric_fn=self.compute_test_metrics))  # the same

        Pass a :class:`~train4all.trainer.phase.Phase` to say anything the
        shorthand cannot — most of all a name, since the name is what the metrics,
        the plots, and the exports are filed under. Two test sets need two names,
        or their curves silently concatenate under one::

            trainer.test(Phase("test_id",  id_loader,  metric_fn=self.compute_test_metrics))
            trainer.test(Phase("test_ood", ood_loader, metric_fn=self.compute_test_metrics))

        A phase you pass means exactly what it means everywhere else — nothing is
        injected into it, so ``metric_fn=None`` is the trainer's ``compute_metrics``
        as always. ``compute_test_metrics`` is what the *shorthand* reaches for, not
        a rule this method applies.

        Args:
            phase: The phase to evaluate, or a DataLoader for the canonical test
                phase (named :attr:`_TEST_PHASE`, carrying ``compute_test_metrics``).
            use_best: Load the best **weights** before evaluating. Only the
                weights are loaded — evaluation never restores the training
                state or metric history, so it cannot rewind the epoch counter
                or truncate the recorded metrics to the best epoch (use
                :meth:`load_best_checkpoint` to deliberately rewind to best).

        Returns:
            Mapping of metric name to value.
        """
        if isinstance(phase, DataLoader):
            phase = Phase(self._TEST_PHASE, phase, metric_fn=self.compute_test_metrics)

        if use_best:
            self.print()
            self.load_best_weights()

        self.print(f"\n── {phase.name.capitalize()} Epoch\n")
        metrics = self._execute_phase(phase)
        self.print_metrics(metrics, phase.name)
        # Terminal operation, like the end of ``train()`` — release cached
        # blocks now that no further epochs depend on allocator reuse.
        self.empty_cuda_cache()
        return metrics

    @_require_setup
    def execute_phase(
        self,
        phase: Phase,
        epoch: int | None = None,
        print_metrics: bool = False,
    ) -> dict[str, float]:
        """
        Run one phase to completion, outside the training loop.

        The building block :meth:`train` is made of — reach for it to drive your
        own loop while keeping the trainer's metric aggregation, checkpointing,
        hooks, AMP, and gradient accumulation.

        Args:
            phase: The phase to run.
            epoch: Epoch number, used for hook callbacks.
            print_metrics: Print aggregated metrics after the phase.

        Returns:
            Aggregated metrics for the phase.
        """
        metrics = self._execute_phase(phase, epoch=epoch)
        if print_metrics:
            self.print_metrics(metrics, phase.name)
        return metrics

    @_require_setup
    def execute_step(
        self,
        batch: Any,
        phase: Phase,
        step: int | None = None,
        print_metrics: bool = False,
    ) -> dict[str, float]:
        """
        Run one step on a single batch.

        Args:
            batch: Batch of data.
            phase: The phase this step belongs to. Only its ``training`` flag and
                metric function are consulted; its loader is not iterated.
            step: 1-based step index for hook and dashboard bookkeeping. In a
                training phase it also drives gradient accumulation — the
                optimizer updates every ``accumulation_steps`` steps.
            print_metrics: Print computed metrics after the step.

        Returns:
            Metrics computed for the step.
        """
        metrics = self._execute_step(batch, phase, step=step)
        if print_metrics:
            self.print_metrics(metrics, phase.name)
        return metrics

    # ── Setup & State ─────────────────────────────────────────────────────────

    def prepare_training(self) -> None:
        """
        Prepare the trainer for a new run.

        When not resuming, first resets the in-memory state (see
        :meth:`reset_trainer`) and clears any previous run artifacts (see
        :meth:`clear_artifacts`), so a fresh run inherits neither stale state nor
        stale files. Then prints the environment summary, saves the config, calls
        ``ensure_setup()``, optionally resumes from the latest checkpoint, and
        prints model and optimization summaries.
        """
        if not self.resume:
            # A fresh run must be fresh in memory as well as on disk — otherwise a
            # reused trainer keeps its epoch counter, metrics, and already-built
            # models (skipping training as "already completed" or continuing the
            # old models). Reset state, then clear the directory, so the two agree.
            self.reset_trainer()
            self.clear_artifacts()
        self.print_env_summary()
        self.save_config()
        self.print_config()
        self.ensure_setup()

        if self.resume and self.has_latest_checkpoint():
            self.load_latest_checkpoint()

        self.print_model_summary()
        self.print_optimization_summary()
        self.print_status()

    def ensure_setup(self) -> None:
        """Call ``setup()`` exactly once; subsequent calls are no-ops."""
        if not self._setup_done:
            self.setup()
            self._setup_done = True

    def clear_setup(self) -> None:
        """
        Discard all resources created by ``setup()``.

        Clears models, optimizer, and scheduler, and marks setup as
        incomplete so the next call to ``ensure_setup()`` rebuilds them.
        """
        self.clear_models()
        self.clear_optimizer()
        self.clear_scheduler()
        self._setup_done = False

    def reset_trainer(self) -> None:
        """
        Reset the trainer to its freshly constructed state.

        Composes the individual resets — setup (:meth:`clear_setup`), training
        progress (:meth:`reset_training_state`), metrics (:meth:`clear_metrics`),
        and the step cache (:meth:`clear_cache`) — then rewinds the
        reproducibility sources that do not reset on their own, so a subsequent
        ``train()`` faithfully repeats the first. User-set checkpoint extras are
        configuration, not training state, and are kept.
        """
        self.clear_setup()
        self.reset_training_state()
        self.clear_metrics()
        self.clear_cache()
        # The RNGs and the scaler's fp16 loss scale would otherwise carry over;
        # the transient _cycle_weight / _last_dash_write counters reset on use.
        self.reset_seed()
        self.reset_scaler()

    def reset_seed(self) -> None:
        """
        Apply the configured ``seed`` to the Python, NumPy, and Torch RNGs.

        Called from ``__init__`` to seed the first run and from
        :meth:`reset_trainer` to rewind every RNG to that same state, so a
        subsequent run resamples identically; on CUDA it also re-pins the
        deterministic cuDNN flags. A no-op when no ``seed`` was set.
        """
        if self.seed is None:
            return
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif self.device.type == "mps":
            torch.mps.manual_seed(seed)

    def reset_scaler(self) -> None:
        """
        Re-initialize the AMP ``GradScaler``, preserving whether it is enabled.

        Its fp16 loss scale adapts during training; rebuilding discards that
        adaptation so a fresh run starts from the construction-time scale. (bf16
        and full precision keep a disabled, passthrough scaler.)
        """
        self._scaler = torch.amp.GradScaler(enabled=self._scaler.is_enabled())

    def clear_artifacts(self) -> None:
        """
        Delete this run's checkpoints, metrics, plots, and dashboard files from
        ``run_dir``.

        Removes only the trainer-owned artifacts — ``config.json``, the log, and
        any user files in ``run_dir`` are left untouched — for a clean slate. A
        no-op (and silent) when none of them exist. ``prepare_training`` calls
        this automatically when ``resume=False``.
        """
        dirs = (self._checkpoints_dir, self._metrics_dir, self._plots_dir)
        if not any(p.exists() for p in (*dirs, *self._dashboard_files)):
            return
        for directory in dirs:
            remove_dir(directory)
        for file in self._dashboard_files:
            file.unlink(missing_ok=True)
        self.print("🧹 Cleared previous run artifacts (checkpoints, metrics, plots, dashboard).")

    # ── Epoch Control ─────────────────────────────────────────────────────────

    def epoch_iterator(self) -> Iterator[tuple[int, int]]:
        """
        Yield ``(current_epoch, num_epochs)`` for each training epoch.

        Automatically increments the internal epoch counter.

        Raises:
            ValueError: If ``num_epochs`` was not set on the trainer.
        """
        num_epochs = self._require_num_epochs()
        while self._current_epoch < num_epochs:
            self._current_epoch += 1
            yield self._current_epoch, num_epochs

    def finalize_train_epoch(self, monitor_value: float | None = None) -> None:
        """
        Update early-stopping state and step the scheduler.

        Must be called *after* computing metrics and *before* saving
        checkpoints, as checkpoint decisions depend on the updated state.

        Args:
            monitor_value: The epoch's ``monitor`` metric from the validation
                phase, or ``None`` if validation was not performed.
        """
        self._update_early_stopping(monitor_value)
        self._step_scheduler(monitor_value)

    def reset_training_state(self) -> None:
        """Reset the epoch counter, best-metric tracking, and early-stopping counters."""
        self._current_epoch = 0
        self._best_metric = self._worst_metric()
        self._best_epoch = None
        self._epochs_no_improve = 0

    def is_training_complete(self) -> bool:
        """Return ``True`` if the epoch counter has reached ``num_epochs``.

        Always ``False`` when ``num_epochs`` is unset, since no training is configured.
        """
        return self.num_epochs is not None and self._current_epoch >= self.num_epochs

    def is_best_epoch(self) -> bool:
        """Return ``True`` if the current epoch achieved the best ``monitor`` value."""
        return self._current_epoch == self._best_epoch

    def should_stop_early(self) -> bool:
        """Return ``True`` if the early-stopping patience has been exhausted."""
        return self.patience is not None and self._epochs_no_improve >= self.patience

    # ── Model / Optimizer / Scheduler ─────────────────────────────────────────

    def set_models(
        self,
        models: dict[str, nn.Module],
        overwrite: bool = True,
        set_attr: bool = False,
        compile: bool = False,
    ) -> None:
        """
        Register multiple models, moving each to the training device.

        Args:
            models: Mapping of name to model instance.
            overwrite: Replace any existing entry with the same name.
            set_attr: Also assign each model as ``self.<name>``.
            compile: Compile each model in place with ``torch.compile()`` (PyTorch 2.0+).
        """
        for name, model in models.items():
            self.set_model(name, model, overwrite=overwrite, set_attr=set_attr, compile=compile)

    def set_model(
        self,
        name: str,
        model: nn.Module,
        overwrite: bool = True,
        set_attr: bool = False,
        compile: bool = False,
    ) -> None:
        """
        Register a single model, moving it to the training device.

        Args:
            name: Model name.
            model: Model instance.
            overwrite: Replace an existing entry with the same name.
            set_attr: Also assign the model as ``self.<name>``.
            compile: Compile the model in place with ``torch.compile()``
                (PyTorch 2.0+) for graph-level optimizations. The registered
                module itself is compiled, so any reference to it runs the
                optimized graph and checkpoints keep their original keys.
        """
        if not overwrite and name in self._models:
            return
        model = model.to(self.device)
        if compile:
            model.compile()  # in place: keeps the same object and state-dict keys
            self._compiled_models.add(name)
        else:
            self._compiled_models.discard(name)
        self._models[name] = model
        if set_attr:
            setattr(self, name, model)

    def clear_models(self) -> None:
        """Remove all registered models."""
        self._models.clear()
        self._compiled_models.clear()

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer."""
        self._optimizer = optimizer

    def clear_optimizer(self) -> None:
        """Remove the current optimizer."""
        self._optimizer = None

    def set_scheduler(self, scheduler: _Scheduler) -> None:
        """Set the learning-rate scheduler."""
        self._scheduler = scheduler

    def clear_scheduler(self) -> None:
        """Remove the current scheduler."""
        self._scheduler = None

    def get_trainable_params(
        self,
        targets: ModuleSpec | None = None,
        exclude_targets: ModuleSpec | None = None,
    ) -> list[nn.Parameter]:
        """
        Return deduplicated trainable parameters from the specified models.

        Args:
            targets: Models to include. ``None`` includes all registered models.
            exclude_targets: Models to exclude from the result.

        Returns:
            List of unique parameters with ``requires_grad=True``.
        """
        modules = self._resolve_modules(targets)
        if exclude_targets is not None:
            excluded = set(self._resolve_modules(exclude_targets))
            modules = [m for m in modules if m not in excluded]

        seen: set[int] = set()
        params: list[nn.Parameter] = []
        for m in modules:
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def freeze(self, targets: ModuleSpec) -> None:
        """Disable gradients for the specified model(s)."""
        self._set_requires_grad(targets, False)

    def unfreeze(self, targets: ModuleSpec) -> None:
        """Enable gradients for the specified model(s)."""
        self._set_requires_grad(targets, True)

    def reset_parameters(self, targets: ModuleSpec | None = None) -> None:
        """
        Reset parameters of the specified model(s).

        Calls ``reset_parameters()`` on every submodule that implements it;
        submodules without the method are silently skipped.

        Args:
            targets: Models to reset. ``None`` resets all registered models.
        """
        for module in self._resolve_modules(targets):
            module.apply(self._reset_module_parameters)

    # ── Checkpoints ───────────────────────────────────────────────────────────

    def save_artifacts(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> None:
        """
        Save checkpoints and export all metric artifacts for the current state.

        Args:
            metric_names: Metrics to include. ``None`` includes all.
            phase_names: Phase names to include. ``None`` includes all.
        """
        # These sub-steps are slow and GIL-holding (torch.save, matplotlib, JSON);
        # pulse the heartbeat between them so a slow one never trips *Offline*.
        self.save_checkpoints()
        self._dash_heartbeat()
        if self._epoch_metrics:
            self.save_epoch_metric_plots(metric_names=metric_names, phase_names=phase_names)
            self.export_epoch_metrics(metric_names=metric_names, phase_names=phase_names)
            self._dash_heartbeat()
        if self._step_metrics:
            self.save_step_metric_plots(metric_names=metric_names, phase_names=phase_names)
            self.export_step_metrics(metric_names=metric_names, phase_names=phase_names)
            self._dash_heartbeat()

    @_require_setup
    def save_checkpoints(self) -> None:
        """Save the latest, best, and periodic (if configured) checkpoints."""
        # No explicit mkdir: ``Checkpoint.save`` creates each destination's parent.
        checkpoint = self._build_checkpoint()

        latest_path = self.get_latest_checkpoint_path()
        self._write_checkpoint(latest_path, checkpoint, f"💾 Latest checkpoint saved: {latest_path.name}")

        if self.is_best_epoch():
            best_path = self.get_best_checkpoint_path()
            self._write_checkpoint(best_path, checkpoint, f"🏆 Best checkpoint saved: {best_path.name}")

        if self.save_interval and self._current_epoch % self.save_interval == 0:
            epoch_path = self.get_checkpoint_path(f"epoch_{self._current_epoch}")
            self._write_checkpoint(
                epoch_path, checkpoint,
                f"💾 Epoch {self._current_epoch} checkpoint saved: {epoch_path.name}",
            )

    @_require_setup
    def save_checkpoint(self, path: Path | str) -> None:
        """
        Save a full checkpoint to a specific path.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        self._write_checkpoint(path, self._build_checkpoint(), f"💾 Checkpoint saved: {path.name}")

    @_require_setup
    def save_weights(self, path: Path | str) -> None:
        """
        Save only model weights to a specific path.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        self._write_checkpoint(
            path, self._build_checkpoint(weights_only=True),
            f"💾 Model weights saved: {path.name}",
        )

    def backup_checkpoint(self, path: Path | str) -> None:
        """
        Copy a checkpoint file, appending ``.bak`` to the filename.

        Args:
            path: Path to the checkpoint to back up.
        """
        path = Path(path)
        if not path.exists():
            self.print(f"Checkpoint not found, backup skipped: {path.name}", level="warn")
            return
        backup_path = path.with_name(path.name + ".bak")
        shutil.copyfile(path, backup_path)
        self.print(f"📦 Backup created: {backup_path.name}")

    @_require_setup
    def load_checkpoint(
        self,
        path: Path | str,
        strict: bool = False,
        key_map: dict[str, str] | None = None,
    ) -> None:
        """
        Load a full checkpoint (models, optimizer, scheduler, and training state).

        Args:
            path: Path to the checkpoint file.
            strict: Enforce exact key matching when loading model state dicts.
            key_map: Optional mapping to rename state-dict keys before loading.
        """
        self._load_checkpoint(Path(path), "💾 Loading checkpoint", strict=strict, key_map=key_map)

    @_require_setup
    def load_weights(
        self,
        path: Path | str,
        strict: bool = False,
        key_map: dict[str, str] | None = None,
    ) -> None:
        """
        Load only model weights from a checkpoint.

        Args:
            path: Path to the checkpoint file.
            strict: Enforce exact key matching.
            key_map: Optional mapping to rename state-dict keys before loading.
        """
        self._load_checkpoint(
            Path(path), "💾 Loading model weights",
            strict=strict, key_map=key_map, weights_only=True,
        )

    @_require_setup
    def load_latest_checkpoint(self) -> None:
        """Load the most recently saved checkpoint."""
        self._load_checkpoint(self.get_latest_checkpoint_path(), "💾 Loading latest checkpoint")

    @_require_setup
    def load_best_checkpoint(self) -> None:
        """Load the full checkpoint from the best validation epoch.

        Restores everything — weights, optimizer, training state, and the metric
        history *as of the best epoch* — so the trainer rewinds to that epoch.
        To evaluate the best model without disturbing the current run, load only
        its weights with :meth:`load_best_weights` (what ``test(use_best=True)``
        does).
        """
        self._load_checkpoint(self.get_best_checkpoint_path(), "🏆 Loading best checkpoint")

    @_require_setup
    def load_best_weights(self) -> None:
        """Load only the model weights from the best checkpoint.

        Leaves the optimizer, training state, and recorded metrics untouched —
        the right tool for evaluating the best model mid-run without rewinding,
        unlike the full :meth:`load_best_checkpoint`.
        """
        self._load_checkpoint(
            self.get_best_checkpoint_path(), "🏆 Loading best weights", weights_only=True
        )

    def exclude_from_checkpoint(self, names: str | list[str]) -> None:
        """
        Exclude model(s) from future checkpoints.

        Args:
            names: Registered model name(s) to exclude.

        Raises:
            ValueError: If any name is not a registered model.
        """
        if isinstance(names, str):
            names = [names]
        invalid = [n for n in names if n not in self._models]
        if invalid:
            raise ValueError(f"Unregistered model(s) cannot be excluded: {invalid}")
        self._ckpt_excludes.update(names)

    def update_checkpoint_extras(self, extras: dict[str, Any]) -> None:
        """
        Add or overwrite entries in the checkpoint ``extras`` dict.

        Args:
            extras: Key-value pairs to merge into the extras dict.
        """
        self._ckpt_extras.update(extras)

    def get_checkpoint_extras(self) -> dict[str, Any]:
        """
        Return a copy of the checkpoint ``extras`` dict.

        Mirrors :meth:`update_checkpoint_extras`. After any checkpoint is loaded
        (full or weights-only), this reflects the restored extras, so it is the
        symmetric way to read back static metadata without overriding
        :meth:`on_load_checkpoint`.
        """
        return dict(self._ckpt_extras)

    def has_latest_checkpoint(self) -> bool:
        """Return ``True`` if ``latest.pth`` exists."""
        return self.get_latest_checkpoint_path().exists()

    def has_best_checkpoint(self) -> bool:
        """Return ``True`` if ``best.pth`` exists."""
        return self.get_best_checkpoint_path().exists()

    def get_latest_checkpoint_path(self) -> Path:
        """Return the path to the latest checkpoint."""
        return self.get_checkpoint_path(self._CHECKPOINT_LATEST)

    def get_best_checkpoint_path(self) -> Path:
        """Return the path to the best checkpoint."""
        return self.get_checkpoint_path(self._CHECKPOINT_BEST)

    def get_checkpoint_path(self, name: str) -> Path:
        """Return the path to ``{name}.pth`` in the checkpoints directory."""
        return self._checkpoints_dir / f"{name}.pth"

    # ── Config ────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: Path | str, **overrides: Any) -> Self:
        """
        Construct a trainer from a saved ``config.json`` — the inverse of
        :meth:`save_config`.

        The config records exactly the constructor arguments the original run
        customized (defaults omitted; see ``__init__``), so reconstruction is
        just ``cls(**config)``. Only ``BaseTrainer`` constructor arguments are
        consumed — filtered against the same signature that wrote them, so custom
        metadata added via :meth:`update_config` is ignored and a stale key from
        an older version is dropped rather than fatal. A subclass's own
        constructor arguments are not recorded in the base config and so must be
        supplied through *overrides*.

        Keyword *overrides* take precedence over the file: pass ``device`` to
        reload on a host without the original one, ``run_dir`` to write into a
        fresh directory, or any subclass argument the config omits.

        Args:
            path: The ``config.json`` file, or the run directory containing it.
            **overrides: Constructor arguments that replace the file's values.

        Returns:
            A new trainer instance.
        """
        path = Path(path)
        if path.is_dir():
            path = path / cls._CONFIG_FILENAME
        with path.open(encoding="utf-8") as f:
            config = json.load(f)
        params = cls._init_params()
        config = {key: value for key, value in config.items() if key in params}
        return cls(**{**config, **overrides})

    def update_config(self, entries: dict[str, Any]) -> None:
        """
        Add or overwrite entries in the trainer configuration.

        Args:
            entries: Key-value pairs to merge into the config.
        """
        self._config.update(entries)

    def save_config(self) -> None:
        """Serialize the trainer configuration to ``config.json`` in ``run_dir``."""
        self._write_json(self.get_config_path(), self._config, "config")

    def get_config_path(self) -> Path:
        """Return the path to the config JSON file in ``run_dir``."""
        return self.run_dir / self._CONFIG_FILENAME

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_epoch_metrics(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> MetricTable:
        """
        Return epoch-level metrics, optionally filtered.

        Args:
            metric_names: Metrics to include. ``None`` returns all.
            phase_names: Phase names to include. ``None`` returns all.

        Returns:
            Filtered metric table.
        """
        return self._filter_metrics(self._epoch_metrics, metric_names=metric_names, phase_names=phase_names)

    def get_step_metrics(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> MetricTable:
        """
        Return step-level metrics, optionally filtered.

        Args:
            metric_names: Metrics to include. ``None`` returns all.
            phase_names: Phase names to include. ``None`` returns all.

        Returns:
            Filtered metric table.
        """
        return self._filter_metrics(self._step_metrics, metric_names=metric_names, phase_names=phase_names)

    def clear_metrics(self) -> None:
        """Clear all recorded epoch and step metrics."""
        self._epoch_metrics.clear()
        self._step_metrics.clear()

    def save_epoch_metric_plots(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> None:
        """
        Save epoch-level metric curve plots.

        Args:
            metric_names: Metrics to plot. ``None`` plots all.
            phase_names: Phase names to include. ``None`` includes all.
        """
        metrics = self.get_epoch_metrics(metric_names=metric_names, phase_names=phase_names)
        self._save_metric_plots(metrics, xlabel="epoch", split_phases=False)
        self.print("📈 Epoch-level metric curves saved.")

    def save_step_metric_plots(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> None:
        """
        Save step-level metric curve plots.

        Args:
            metric_names: Metrics to plot. ``None`` plots all.
            phase_names: Phase names to include. ``None`` includes all.
        """
        metrics = self.get_step_metrics(metric_names=metric_names, phase_names=phase_names)
        self._save_metric_plots(
            metrics,
            xlabel="step",
            title_prefix="step-level",
            path_prefix="step",
            split_phases=True,
        )
        self.print("📈 Step-level metric curves saved.")

    def export_epoch_metrics(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> Path:
        """
        Export epoch-level metrics to a JSON file.

        Args:
            metric_names: Metrics to export. ``None`` exports all.
            phase_names: Phase names to include. ``None`` includes all.

        Returns:
            Path to the written JSON file.
        """
        metrics = self.get_epoch_metrics(metric_names=metric_names, phase_names=phase_names)
        path = self.get_epoch_metrics_path()
        self._export_metrics(metrics, path)
        self.print(f"📄 Epoch-level metrics exported: {path.name}")
        return path

    def export_step_metrics(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> Path:
        """
        Export step-level metrics to a JSON file.

        Args:
            metric_names: Metrics to export. ``None`` exports all.
            phase_names: Phase names to include. ``None`` includes all.

        Returns:
            Path to the written JSON file.
        """
        metrics = self.get_step_metrics(metric_names=metric_names, phase_names=phase_names)
        path = self.get_step_metrics_path()
        self._export_metrics(metrics, path)
        self.print(f"📄 Step-level metrics exported: {path.name}")
        return path

    def get_epoch_metrics_path(self) -> Path:
        """Return the path to the epoch metrics JSON file."""
        return self.get_metrics_path(self._METRICS_EPOCH)

    def get_step_metrics_path(self) -> Path:
        """Return the path to the step metrics JSON file."""
        return self.get_metrics_path(self._METRICS_STEP)

    def get_metrics_path(self, name: str) -> Path:
        """Return the path to ``{name}.json`` in the metrics directory."""
        return self._metrics_dir / f"{name}.json"

    def get_metric_plot_path(
        self,
        metric_name: str,
        phase_name: str | None = None,
        prefix: str | None = None,
    ) -> Path:
        """Return the output path for a metric curve plot PNG."""
        filename = get_metric_plot_filename(metric_name, phase_name=phase_name, prefix=prefix)
        return self._plots_dir / filename

    # ── Logging & Display ─────────────────────────────────────────────────────

    def get_env_summary(self) -> dict[str, Any]:
        """Return the system and runtime environment summary as a dict."""
        return env_summary(
            self.run_dir,
            gpu_index=self._cuda_index if torch.cuda.is_available() else None,
        )

    def print_env_summary(self) -> None:
        """Print a system and runtime environment summary for experiment reproducibility."""
        self.print_dict_tree(self.get_env_summary(), header="🖥️  Environment")

    def print_config(self) -> None:
        """Print the current trainer configuration."""
        self.print_dict_tree(self._config, header="⚙️  Configuration")

    def get_model_summary(self) -> dict[str, str]:
        """Return the name and parameter counts of all registered models as a dict."""
        result: dict[str, str] = {}
        for name, model in self._models.items():
            total = trainable = 0
            for p in model.parameters():
                n = p.numel()
                total += n
                if p.requires_grad:
                    trainable += n
            suffix = " [compiled]" if name in self._compiled_models else ""
            if trainable == total:
                result[name] = f"{total:,} params{suffix}"
            elif trainable:
                result[name] = f"{trainable:,} / {total:,} trainable{suffix}"
            else:
                result[name] = f"frozen{suffix}"
        return result

    def print_model_summary(self) -> None:
        """Print the name and parameter counts of all registered models."""
        self.print_dict_tree(self.get_model_summary(), header="🧠 Model")

    def print_optimization_summary(self) -> None:
        """Print the optimizer, scheduler, and gradient-accumulation settings."""
        tree: dict[str, str] = {
            "Optimizer": self._optimizer.__class__.__name__ if self._optimizer else "-",
            "Scheduler": self._scheduler.__class__.__name__ if self._scheduler else "-",
        }
        if self.accumulation_steps > 1:
            tree["Grad accumulation"] = f"{self.accumulation_steps} steps"
        self.print_dict_tree(tree, header="⚡ Optimization")

    @staticmethod
    def get_schedule_summary(*phases: Phase) -> dict[str, str]:
        """Return the shape of one epoch as a dict: each phase name mapped to how
        it runs.

        The schedule is an argument to ``train()``, not trainer state, so it is
        not part of ``config.json`` — that file holds constructor arguments and
        must unpack straight back through :meth:`from_config`. This is the shape's
        own summary, alongside the model's and the optimizer's.

        Args:
            *phases: The phases of one epoch, in the order they run.
        """
        def describe(phase: Phase) -> str:
            kind = "training" if phase.training else "eval"
            return kind if phase.every == 1 else f"{kind}, every {phase.every} epochs"

        return {p.name: describe(p) for p in phases}

    def print_schedule_summary(self, *phases: Phase) -> None:
        """Print the shape of one epoch — the phases, in the order they run."""
        self.print_dict_tree(self.get_schedule_summary(*phases), header="🗓️  Schedule")

    def print_status(self) -> None:
        """Print the current training state (epoch, best monitored value, and recent metrics)."""
        tree: dict[str, Any] = {
            "Completed epochs":   self._current_epoch,
            f"Best {self.monitor_phase} {self.monitor}": (
                f"{self._best_metric:.4f}  (epoch {self._best_epoch})"
                if self._best_epoch is not None else "-"
            ),
            "Stagnant epochs":    self._epochs_no_improve,
            "Last epoch metrics": self._format_epoch_metrics() or "-",
        }
        self.print_dict_tree(tree, header="📋 Status")

    def print_metrics(self, metrics: dict[str, float], phase_name: str) -> None:
        """
        Print a flat metrics table for a given phase.

        Args:
            metrics: Mapping of metric name to value.
            phase_name: Phase label shown in the header.
        """
        print_dict_tree(
            metrics,
            max_depth=0,
            header=f"📊 {phase_name.capitalize()}",
            key_width=self._KEY_WIDTH,
            float_fmt=4,
            trailing_newline=True,
            print_fn=self.print,
        )

    def print_dict_tree(
        self,
        tree: dict[str, Any],
        header: str | None = None,
        max_depth: int | None = None,
    ) -> None:
        """
        Pretty-print a nested dictionary in a tree format.

        Args:
            tree: Dictionary to display.
            header: Title shown above the tree.
            max_depth: Maximum nesting depth to expand. ``None`` is unlimited.
        """
        print_dict_tree(
            tree,
            max_depth=max_depth,
            header=header,
            key_width=self._KEY_WIDTH,
            trailing_newline=True,
            print_fn=self.print,
        )

    def print(self, msg: str | None = None, level: LogLevel = "info", *, indent: int = 0) -> None:
        """Forward a message to the logger."""
        self.logger.log(msg, level=level, indent=indent)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def set_cache(self, key: str, value: Any) -> None:
        """Store *value* under *key* for cross-method communication within a step."""
        self._cache[key] = value

    def get_cache(self, key: str, default: Any = None) -> Any:
        """Return the cached value for *key*, or *default* if absent."""
        return self._cache.get(key, default)

    def clear_cache(self) -> None:
        """Remove all entries from the step cache."""
        self._cache.clear()

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot_run(self, exclude: list[str] | None = None) -> None:
        """
        Mirror ``run_dir`` into ``run_snapshot_dir``.

        ``train()`` calls this after every epoch's artifacts are written, so a
        configured ``run_snapshot_dir`` keeps an up-to-date copy of the run without
        any further wiring — which is the point of a cloud-backed mirror on a host
        that may vanish. Call it yourself for a snapshot at any other moment.

        Repeating it is cheap and safe, which is what lets the loop do it unattended:
        :func:`~train4all.utils.copy_dir` copies only the files that changed, replaces
        each atomically, and prunes last — so an interrupted snapshot leaves the
        mirror whole rather than gone.

        What the mirror leaves behind belongs to the run, not to a call, so it is
        configured as ``run_snapshot_exclude`` — a bare call takes whatever the
        trainer is set to take, whether the caller is you or the epoch loop.
        Nothing is excluded by default: the checkpoints are exactly what a mirror
        exists to preserve.

        *exclude* overrides that policy for this call alone. ``None`` (the default)
        defers to it, and ``[]`` mirrors everything even when the policy excludes
        something.

        A no-op when ``run_snapshot_dir`` is ``None``.

        Args:
            exclude: Top-level entry names to omit from this snapshot. ``None``
                defers to ``run_snapshot_exclude``.
        """
        if self.run_snapshot_dir is None:
            return
        copy_dir(
            src=self.run_dir,
            dst=self.run_snapshot_dir,
            exclude=self.run_snapshot_exclude if exclude is None else exclude,
        )

    # ── GPU Utilities ─────────────────────────────────────────────────────────

    def print_gpu_temperature(self) -> None:
        """Print the current GPU temperature via ``nvidia-smi``; warn above
        :attr:`_GPU_TEMP_WARN_C`.

        The reading itself comes from :func:`~train4all.utils.system.gpu_temperature`;
        what is left here is the reporting, which is the trainer's job.
        """
        if not torch.cuda.is_available():
            self.print("CUDA not available. Skipping GPU temperature check.", level="warn")
            return

        try:
            temp = gpu_temperature(self._cuda_index)
        except FileNotFoundError:
            self.print("'nvidia-smi' not found. Skipping GPU temperature check.", level="warn")
            return
        except subprocess.CalledProcessError as e:
            self.print(f"'nvidia-smi' command failed: {e}", level="warn")
            return
        except Exception as e:
            self.print(f"Failed to get GPU temperature: {e}", level="warn")
            return

        if temp is None:
            self.print("GPU temperature info unavailable or invalid.", level="warn")
            return
        self.print(f"🌡️  GPU Temperature: {temp} °C")
        if temp > self._GPU_TEMP_WARN_C:
            self.print("GPU temperature is high! Consider cooling down.", level="warn")

    @staticmethod
    def empty_cuda_cache() -> None:
        """Free Python-held tensor references and clear the CUDA memory cache."""
        empty_cuda_cache()

    # ── Internal: Training Loop ───────────────────────────────────────────────

    def _execute_phase(self, phase: Phase, epoch: int | None = None) -> dict[str, float]:
        self.clear_cache()
        self._set_training_mode(phase.training)
        self.on_phase_start(epoch, phase)
        metrics = self._iterate_phase(phase)
        self._record_epoch_metrics(metrics, phase.name)
        self._dash_update()
        self.on_phase_end(epoch, phase, metrics)
        # NOTE: no per-phase ``empty_cache()`` here — releasing cached blocks
        # back to the driver every phase forces the allocator to re-acquire
        # them next phase, which slows training. A single cleanup runs at the
        # end of ``train()``; call ``empty_cuda_cache()`` manually if needed.
        return metrics

    def _iterate_phase(self, phase: Phase) -> dict[str, float]:
        # Start each training phase with a clean gradient state so any
        # unstepped accumulation from the previous one (possible for
        # IterableDataset loaders whose length is unknown) is discarded —
        # including the per-cycle weight that would normalize those gradients.
        if phase.training and self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)
            self._cycle_weight = 0.0
        pbar: tqdm | None = (
            tqdm(phase.loader, desc=f"{phase.name.capitalize()} Epoch", leave=self._KEEP_PROGRESS_BAR)
            if self.use_progress_bar else None
        )
        accumulated: dict[str, float] = {}
        total_weight = 0
        max_step = self._loader_len(phase.loader)  # 0 for length-less IterableDataset loaders
        for step, batch in enumerate(pbar or phase.loader, 1):
            weight = self.get_batch_weight(batch)
            metrics = self._execute_step(batch, phase, step=step, max_step=max_step, weight=weight)
            self._accumulate_metrics(accumulated, metrics, weight)
            total_weight += weight
            if pbar is not None:
                self._update_pbar(pbar, metrics)

        return self._average_metrics(accumulated, total_weight)

    def _execute_step(
        self,
        batch: Any,
        phase: Phase,
        step: int | None = None,
        max_step: int = 0,
        weight: float | None = None,
    ) -> dict[str, float]:
        batch = self._to_device(batch)
        self.on_step_start(step, batch, phase)
        # A training step fires the optimizer update only on the step that closes
        # an accumulation cycle (every step when accumulation_steps == 1).
        apply_update = phase.training and self._is_accumulation_boundary(step or 0, max_step)
        # The per-batch weight drives gradient-accumulation normalization. The
        # epoch loop already computes it (for metric averaging) and passes it in;
        # the standalone ``execute_step`` path computes it here on demand.
        if phase.training and weight is None:
            weight = self.get_batch_weight(batch)
        metrics = self._compute_step(batch, phase, weight=weight, apply_update=apply_update)
        # Throttle intermediate steps, but always write the final step of a
        # phase so the gauge's inner ring reaches 100% before the phase resets.
        self._dash_update(
            step=step or 0, max_step=max_step, step_metrics=metrics, phase=phase,
            throttle=max_step <= 0 or step != max_step,
        )
        if self.record_step_metrics and phase.records_steps:
            self._record_step_metrics(metrics, phase.name)
        self.on_step_end(step, batch, metrics, phase)
        return metrics

    def _compute_step(
        self,
        batch: Any,
        phase: Phase,
        *,
        weight: float | None,
        apply_update: bool = True,
    ) -> dict[str, float]:
        # ``weight`` has no default but may be ``None``: evaluation ignores it,
        # while training always supplies it (computed in ``_execute_step``).
        with torch.set_grad_enabled(phase.training):
            with self._autocast():
                loss = self.compute_loss(batch)
            # Validate *before* the optimizer touches the parameters. A non-finite
            # loss makes every gradient non-finite, and a single step on those
            # writes NaN into every weight — so the guard has to precede the step
            # or it reports the divergence over a model it has already destroyed.
            # Stopping here leaves the model intact and the run resumable from its
            # last checkpoint. (Reading the value syncs with the device; the step
            # pays that cost regardless, since ``loss`` is recorded as a metric.)
            loss_value = self._validated_loss(loss)
            # Backward and the optimizer update run outside autocast — as AMP
            # requires — while still under the grad-enabled context above.
            if phase.training:
                assert weight is not None  # guaranteed by _execute_step when training
                self._optimizer_step(loss, weight, apply_update=apply_update)
        # The phase brings its own metric function, falling back to the trainer's
        # shared one — so no phase is privileged by name.
        metric_fn = phase.metric_fn or self.compute_metrics
        # Metrics never need gradients. Computing them under no_grad avoids
        # building and immediately discarding a graph on every step, which
        # otherwise leaks both memory and time during training phases (where
        # grad would still be enabled by the context above).
        with torch.no_grad(), self._autocast():
            metrics = metric_fn(batch)
        metrics["loss"] = loss_value
        return metrics

    def _autocast(self) -> torch.autocast:
        """Autocast context for the configured AMP device/dtype.

        A single source of truth for the two call sites in :meth:`_compute_step`; a
        transparent no-op when AMP is disabled (``enabled=False``).
        """
        return torch.autocast(
            self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled,
        )

    # ── Internal: Optimizer / Scheduler ───────────────────────────────────────

    def _is_accumulation_boundary(self, step: int, max_step: int) -> bool:
        """Whether the 1-based ``step`` ends a gradient-accumulation cycle and
        should trigger an optimizer update.

        Every ``accumulation_steps``-th step is a boundary, as is the final step
        of a known-length epoch (``max_step``) — so a short tail cycle is flushed
        rather than dropped. Length-unknown loaders (``max_step == 0``) cannot
        detect their last step, so their tail is discarded by the next epoch's
        opening ``zero_grad``.
        """
        return step % self.accumulation_steps == 0 or (max_step > 0 and step == max_step)

    def _optimizer_step(
        self, loss: torch.Tensor, weight: float, *, apply_update: bool = True,
    ) -> None:
        if self._optimizer is None:
            raise RuntimeError("An optimizer is required for training.")

        # Fast path: with no accumulation, one backward is one full update and
        # the loss is already the per-batch mean, so no weighting is needed.
        # A disabled scaler (bf16 / no AMP) is a transparent passthrough.
        if self.accumulation_steps == 1:
            self._scaler.scale(loss).backward()
            self.on_after_backward()
            self._apply_optimizer_step(grad_scale=None)
            return

        # Gradient accumulation. Weight each micro-batch's loss by its sample/
        # token count so the accumulated gradient is the true weighted mean over
        # the whole effective batch — Σ wᵢ∇Lᵢ / Σ wᵢ. A plain loss/N is only
        # correct when every micro-batch carries the same number of items; with
        # variable-length sequences it over-weights the shorter batches.
        self._cycle_weight += weight
        self._scaler.scale(loss * weight).backward()
        # Fires on every backward, including mid-cycle accumulation steps. Under
        # fp16 AMP the gradients are still scaled here; see
        # ``on_before_optimizer_step`` for the unscaled, post-clip view.
        self.on_after_backward()
        # Normalize / clip / step only when the accumulation cycle is complete.
        if not apply_update:
            return
        # Divide the accumulated gradient by the cycle's total weight to recover
        # the weighted mean. A short tail cycle (final, partial accumulation
        # window) divides by its own weight, not a full N, so it is unbiased too.
        grad_scale = 1.0 / self._cycle_weight if self._cycle_weight > 0 else 1.0
        self._cycle_weight = 0.0
        self._apply_optimizer_step(grad_scale=grad_scale)

    def _apply_optimizer_step(self, *, grad_scale: float | None) -> None:
        """Renormalize (optionally), clip, and run one optimizer step.

        ``grad_scale`` multiplies every gradient before clipping/stepping — the
        ``1/Σw`` accumulation normalizer — or ``None`` to skip it. Multiplying by
        a constant commutes with the AMP loss-scale, so the normalization is
        applied to the still-scaled gradients and ``unscale_`` is only needed
        when clipping (which must see real-unit gradients).
        """
        assert self._optimizer is not None  # guarded by the caller
        if grad_scale is not None and grad_scale != 1.0:
            for group in self._optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.mul_(grad_scale)
        if self.max_grad_norm is not None:
            # Gradients must be unscaled into real units before clipping;
            # unscale_ is a no-op on a disabled scaler.
            self._scaler.unscale_(self._optimizer)
            # Clip exactly the parameters the optimizer owns — the same set
            # unscale_ just rescaled and step() will update.
            params = [p for group in self._optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.on_before_optimizer_step()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad(set_to_none=True)

    def _step_scheduler(self, monitor_value: float | None = None) -> None:
        if self._scheduler is None:
            return
        if isinstance(self._scheduler, ReduceLROnPlateau):
            if monitor_value is None:
                raise ValueError(
                    f"ReduceLROnPlateau requires the '{self.monitor}' metric from the "
                    f"'{self.monitor_phase}' phase, but it was None this epoch. Give the "
                    f"run a '{self.monitor_phase}' phase that reports '{self.monitor}' and "
                    "runs every epoch (every=1), or use a different scheduler."
                )
            self._scheduler.step(monitor_value)
        else:
            self._scheduler.step()

    # ── Internal: Early Stopping / Mode ───────────────────────────────────────

    def _require_num_epochs(self) -> int:
        """Return ``num_epochs``, guarding the training-only entry points against
        it being unset."""
        if self.num_epochs is None:
            raise ValueError(
                "train() requires num_epochs; pass it to the constructor "
                "(e.g. MyTrainer(num_epochs=10)). To only evaluate, use test() or "
                "execute_phase(); to inspect a saved file, use Checkpoint.load(...)."
            )
        return self.num_epochs

    @staticmethod
    def _validate_monitor_mode(monitor_mode: str) -> str:
        if monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max'; got {monitor_mode!r}")
        return monitor_mode

    @staticmethod
    def _validate_accumulation_steps(accumulation_steps: int) -> int:
        # Rejected rather than clamped to 1: silently rewriting the caller's value
        # would still record the *given* one in ``config.json``, so the saved run
        # would claim a setting it never trained with. ``Phase.every`` rejects the
        # same way.
        if accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be >= 1; got {accumulation_steps}"
            )
        return accumulation_steps

    def _worst_metric(self) -> float:
        """The sentinel best-so-far value: any real metric beats it."""
        return float("-inf") if self.monitor_mode == "max" else float("inf")

    def _is_improvement(self, value: float, best: float) -> bool:
        return value > best if self.monitor_mode == "max" else value < best

    def _update_early_stopping(self, monitor_value: float | None) -> None:
        if monitor_value is None:
            return
        if self._is_improvement(monitor_value, self._best_metric):
            self._best_metric = monitor_value
            self._best_epoch = self._current_epoch
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
            n = self._epochs_no_improve
            self.print(f"No improvement for {n} {'epoch' if n == 1 else 'epochs'}.", level="warn")

    def _set_training_mode(self, training: bool) -> None:
        for model in self._models.values():
            model.train(training)
        self.on_set_training_mode(training)

    # ── Internal: Phases ──────────────────────────────────────────────────────

    def _validate_phases(self, phases: tuple[Phase, ...]) -> None:
        """Reject a schedule that cannot mean what it says, and warn about the
        ones that can but almost certainly do not."""
        if not phases:
            raise ValueError(
                "train() requires at least one Phase, e.g. "
                "train(Phase('train', train_loader, training=True), Phase('val', val_loader))."
            )
        names = [p.name for p in phases]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                "Phase names must be unique — they key the metric tables, the plots, "
                f"and monitor_phase. Duplicated: {duplicates}"
            )
        if not any(p.training for p in phases):
            self.print(
                "No phase has training=True, so no gradient update will ever run. Pass "
                "training=True to the phase that should learn.",
                level="warn",
            )
        if self.patience is None:
            return
        monitored = next((p for p in phases if p.name == self.monitor_phase), None)
        if monitored is None:
            self.print(
                "Early stopping is enabled (patience set) but no phase is named "
                f"'{self.monitor_phase}' — it can never trigger without the "
                f"'{self.monitor}' metric from that phase. Rename the phase, or point "
                "monitor_phase at one in the schedule.",
                level="warn",
            )
        elif monitored.every > 1:
            self.print(
                f"The monitored phase '{self.monitor_phase}' runs only every "
                f"{monitored.every} epochs, so early stopping advances only on those "
                "epochs — patience counts them, not raw epochs.",
                level="warn",
            )

    def _phase_specs(self, phases: tuple[Phase, ...]) -> list[PhaseSpec]:
        """The schedule as the dashboard sees it — names, gradients, lengths, cadence."""
        return [
            PhaseSpec(
                name=p.name,
                training=p.training,
                steps=self._loader_len(p.loader),
                every=p.every,
            )
            for p in phases
        ]

    # ── Internal: Checkpoints (save) ──────────────────────────────────────────

    def _build_checkpoint(self, weights_only: bool = False) -> Checkpoint:
        # ``Checkpoint`` owns the schema; a weights-only checkpoint is just
        # models + extras, so the training components are never even assembled.
        models = {
            k: v.state_dict()
            for k, v in self._models.items()
            if k not in self._ckpt_excludes
        }
        extras = dict(self._ckpt_extras)
        if weights_only:
            return Checkpoint.build(models=models, extras=extras, weights_only=True)

        checkpoint = Checkpoint.build(
            models=models,
            extras=extras,
            optimizer=self._optimizer.state_dict() if self._optimizer else None,
            scheduler=self._scheduler.state_dict() if self._scheduler else None,
            scaler=self._scaler.state_dict(),
            training_state={
                "current_epoch":     self._current_epoch,
                "best_metric":       self._best_metric,
                "best_epoch":        self._best_epoch,
                "epochs_no_improve": self._epochs_no_improve,
            },
            metrics={
                "epoch_metrics": self._epoch_metrics,
                "step_metrics":  self._step_metrics,
            },
        )
        # Subclasses attach custom state to full checkpoints only;
        # the weights-only path already returned.
        self.on_save_checkpoint(checkpoint)
        return checkpoint

    def _write_checkpoint(self, path: Path, checkpoint: Checkpoint, success_msg: str) -> None:
        """Write *checkpoint* to *path*, logging *success_msg* or a warning on failure."""
        try:
            checkpoint.save(path)
            self.print(success_msg)
        except Exception as e:
            self.print(f"Failed to save {path.name}: {e}", level="warn")

    # ── Internal: Checkpoints (load) ──────────────────────────────────────────

    def _load_checkpoint(
        self,
        path: Path | str,
        label: str,
        strict: bool = False,
        key_map: dict[str, str] | None = None,
        weights_only: bool = False,
    ) -> None:
        self.print(f"{label} ...")
        self.print(separator_rule(self._KEY_WIDTH))
        ckpt = self._read_checkpoint(path)
        if not ckpt:
            return

        loaded: dict[str, str] = {}

        for name, state_dict in ckpt.models.items():
            status = self._load_model_state_dict(
                model=self._models.get(name),
                name=name,
                state_dict=state_dict,
                strict=strict,
                key_map=key_map,
            )
            if status is not None:
                loaded[name] = status

        # Extras ride along with both full and weights-only checkpoints (see
        # ``_build_checkpoint``), so they round-trip on either load.
        if ckpt.extras:
            self._ckpt_extras.update(ckpt.extras)
            loaded["extras"] = "restored"

        if not weights_only:
            # Optimizer, scheduler, and scaler share one load-and-record path.
            for name, obj, state in (
                ("optimizer", self._optimizer, ckpt.optimizer_state),
                ("scheduler", self._scheduler, ckpt.scheduler_state),
                ("scaler",    self._scaler,    ckpt.scaler_state),
            ):
                status = self._load_state_dict(obj, name, state)
                if status is not None:
                    loaded[name] = status

            # ``Checkpoint.training_state`` already normalizes legacy key names,
            # so older checkpoints restore through the canonical fields here.
            ts = ckpt.training_state
            self._current_epoch     = ts.get("current_epoch",     self._current_epoch)
            self._best_metric       = ts.get("best_metric",       self._best_metric)
            self._best_epoch        = ts.get("best_epoch",        self._best_epoch)
            self._epochs_no_improve = ts.get("epochs_no_improve", self._epochs_no_improve)
            loaded["training_state"] = "restored"

            self._epoch_metrics = ckpt.metrics.get("epoch_metrics", self._epoch_metrics)
            self._step_metrics  = ckpt.metrics.get("step_metrics",  self._step_metrics)
            loaded["metrics"] = "restored"

            # Let subclasses restore any custom state from the loaded
            # checkpoint (the counterpart of ``on_save_checkpoint``).
            self.on_load_checkpoint(ckpt)

        print_dict_tree(
            loaded,
            max_depth=0,
            key_width=self._KEY_WIDTH,
            print_fn=self.print,
        )

    def _read_checkpoint(self, path: Path | str) -> Checkpoint | None:
        """Read the checkpoint at *path*, or return ``None`` (logging a warning) on failure."""
        try:
            return Checkpoint.load(path, map_location=self.device)
        except FileNotFoundError:
            self.print(f"Checkpoint not found: {path}", level="warn")
        except Exception as e:
            self.print(f"Failed to load checkpoint '{path}': {e}", level="warn")
        self.print()
        return None

    def _load_model_state_dict(
        self,
        model: nn.Module | None,
        name: str,
        state_dict: dict[str, Any] | None,
        strict: bool = False,
        key_map: dict[str, str] | None = None,
    ) -> str | None:
        if state_dict is None:
            return None
        if model is None:
            self.print(f"{name}: not registered — skipped", level="warn", indent=2)
            return None
        if key_map:
            state_dict = replace_dict_keys(state_dict, key_map)
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            parts = ["weights loaded"]
            if not strict:
                if missing:
                    parts.append(f"missing: {missing}")
                if unexpected:
                    parts.append(f"unexpected: {unexpected}")
            return ", ".join(parts)
        except Exception as e:
            self.print(f"{name}: failed to load ({e})", level="warn", indent=2)
            return f"failed ({e})"

    def _load_state_dict(
        self,
        obj: Optimizer | _Scheduler | torch.amp.GradScaler | None,
        name: str,
        state_dict: dict[str, Any] | None,
    ) -> str | None:
        if obj is None or state_dict is None:
            return None
        try:
            obj.load_state_dict(state_dict)
            return "state loaded"
        except Exception as e:
            self.print(f"{name}: failed to load ({e})", level="warn", indent=2)
            return f"failed ({e})"

    # ── Internal: Metrics ─────────────────────────────────────────────────────

    def _record_epoch_metrics(self, metrics: dict[str, float], phase_name: str) -> None:
        self._record_metrics(self._epoch_metrics, metrics, phase_name)

    def _record_step_metrics(self, metrics: dict[str, float], phase_name: str) -> None:
        if self.step_metric_names is not None:
            metrics = {k: v for k, v in metrics.items() if k in self.step_metric_names}
        self._record_metrics(self._step_metrics, metrics, phase_name)

    @staticmethod
    def _record_metrics(target: MetricTable, metrics: dict[str, float], phase_name: str) -> None:
        for name, value in metrics.items():
            target.setdefault(name, {}).setdefault(phase_name, []).append(value)

    @staticmethod
    def _filter_metrics(
        metrics: MetricTable,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> MetricTable:
        result: MetricTable = {}
        for name, phase_dict in metrics.items():
            if metric_names is not None and name not in metric_names:
                continue
            filtered = {
                phase_name: values
                for phase_name, values in phase_dict.items()
                if (phase_names is None or phase_name in phase_names) and values
            }
            if filtered:
                result[name] = filtered
        return result

    @staticmethod
    def _accumulate_metrics(
        accumulated: dict[str, float],
        metrics: dict[str, float],
        weight: float,
    ) -> None:
        for name, value in metrics.items():
            accumulated[name] = accumulated.get(name, 0.0) + value * weight

    @staticmethod
    def _average_metrics(accumulated: dict[str, float], total_weight: float) -> dict[str, float]:
        if total_weight == 0:
            return {}
        return {k: v / total_weight for k, v in accumulated.items()}

    def _save_metric_plots(
        self,
        metrics: MetricTable,
        xlabel: str,
        title_prefix: str | None = None,
        path_prefix: str | None = None,
        split_phases: bool = False,
    ) -> None:
        for metric_name, phase_dict in metrics.items():
            if all(not v for v in phase_dict.values()):
                continue
            if split_phases:
                for phase_name, values in phase_dict.items():
                    if not values:
                        continue
                    save_curves_plot(
                        curves={phase_name: values},
                        path=self.get_metric_plot_path(
                            metric_name, phase_name=phase_name, prefix=path_prefix,
                        ),
                        title=get_metric_plot_title(
                            metric_name, phase_name=phase_name, prefix=title_prefix,
                        ),
                        xlabel=xlabel,
                        ylabel=metric_name,
                    )
            else:
                save_curves_plot(
                    curves={p: v for p, v in phase_dict.items() if v},
                    path=self.get_metric_plot_path(metric_name, prefix=path_prefix),
                    title=get_metric_plot_title(metric_name, prefix=title_prefix),
                    xlabel=xlabel,
                    ylabel=metric_name,
                )

    def _export_metrics(self, metrics: MetricTable, path: Path | str) -> None:
        self._write_json(Path(path), metrics, "metrics")

    def _write_json(self, path: Path, data: Any, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.print(f"Failed to write {label}: {e}\n", level="warn")

    def _format_epoch_metrics(self) -> dict[str, str]:
        return {
            metric_name: "  ".join(
                f"{phase_name}={values[-1]:.4f}" if values else f"{phase_name}=N/A"
                for phase_name, values in phase_dict.items()
            ) or "N/A"
            for metric_name, phase_dict in self._epoch_metrics.items()
        }

    # ── Internal: Display ─────────────────────────────────────────────────────

    def _update_pbar(self, pbar: tqdm, metrics: dict[str, float]) -> None:
        display = {
            k: f"{v:.4f}"
            for k, v in metrics.items()
            if self.pbar_metric_names and k in self.pbar_metric_names
        }
        if self.device.type == "cuda":
            used, total, _ = self._gpu.memory_mib()
            display["GPU"] = f"{used}/{total}"
        pbar.set_postfix(display)

    def _create_default_logger(self) -> UnifiedLogger:
        return UnifiedLogger(
            f"trainer_{id(self)}",
            log_path=self.run_dir / self._LOG_FILENAME,
            verbose=True,
            debug_mode=self.debug_mode,
            # Append across a resume so the log is continuous; start a fresh log
            # for a fresh run (resume=False), matching the cleared artifacts.
            file_mode="a" if self.resume else "w",
        )

    # ── Internal: Dashboard ───────────────────────────────────────────────────

    def _dash_init(self, phases: tuple[Phase, ...]) -> None:
        if self._dashboard is None:
            return
        self._dashboard.initialize(
            self._config,
            env_summary=self.get_env_summary(),
            model_summary=self.get_model_summary(),
            phases=self._phase_specs(phases),
            monitor=self.monitor,
            monitor_phase=self.monitor_phase,
        )
        self.print(f"🌐 Dashboard: {self._dashboard.url}\n")

    def _dash_heartbeat(self) -> None:
        """Refresh the dashboard liveness timestamp (no-op without a dashboard)."""
        if self._dashboard is not None:
            self._dashboard.heartbeat()

    def _dash_update(
        self,
        *,
        step: int = 0,
        max_step: int = 0,
        step_metrics: dict[str, float] | None = None,
        phase: Phase | None = None,
        throttle: bool = False,
    ) -> None:
        if self._dashboard is None or not self._dashboard.active:
            return
        now = time.time()
        if throttle and now - self._last_dash_write < self._DASH_THROTTLE_S:
            return
        self._last_dash_write = now
        # Collect the per-group learning rates so a dict / multi-group LR is
        # represented faithfully: a single value when they all agree, otherwise
        # the full list (the dashboard renders it as a range).
        lr: float | list[float] | None = None
        if self._optimizer is not None and self._optimizer.param_groups:
            lrs = [pg["lr"] for pg in self._optimizer.param_groups if pg.get("lr") is not None]
            if lrs:
                lr = lrs[0] if len(set(lrs)) == 1 else lrs
        self._dashboard.update(
            self._current_epoch,
            self.num_epochs,
            self._epoch_metrics,
            self._best_metric,
            self._best_epoch,
            epochs_no_improve=self._epochs_no_improve,
            is_gradient_phase=phase.training if phase else False,
            step=step,
            max_step=max_step,
            step_metrics=step_metrics,
            phase_name=phase.name if phase else "",
            learning_rate=lr,
            gpu_mem=self._dash_gpu_mem(),
        )

    def _dash_gpu_mem(self) -> tuple[float, float] | None:
        """``(used_gb, total_gb)`` for the dashboard's footprint bar, or ``None``
        when not on CUDA. The same probe the progress bar reads."""
        return self._gpu.memory_gb() if self.device.type == "cuda" else None

    def _dash_finalize(self) -> None:
        if self._dashboard is None:
            return
        self._dashboard.finalize(
            self._current_epoch,
            self.num_epochs,
            self._epoch_metrics,
            self._best_metric,
            self._best_epoch,
            epochs_no_improve=self._epochs_no_improve,
        )
        time.sleep(self._dashboard.poll_s + self._DASH_EXTRA_WAIT_S)

    # ── Internal: Module Utilities ────────────────────────────────────────────

    def _resolve_modules(self, targets: ModuleSpec | None) -> list[nn.Module]:
        if targets is None:
            return list(self._models.values())
        if not isinstance(targets, list):
            targets = [targets]
        return [self._resolve_module(t) for t in targets]

    def _resolve_module(self, target: str | nn.Module) -> nn.Module:
        if isinstance(target, str):
            if target not in self._models:
                raise ValueError(f"Model '{target}' is not registered.")
            return self._models[target]
        if isinstance(target, nn.Module):
            return target
        raise TypeError(f"Expected a model name or nn.Module, got {type(target)}")

    def _set_requires_grad(self, targets: ModuleSpec, flag: bool) -> None:
        for m in self._resolve_modules(targets):
            for p in m.parameters():
                p.requires_grad = flag

    @staticmethod
    def _reset_module_parameters(m: nn.Module) -> None:
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()

    # ── Internal: Data Utilities ──────────────────────────────────────────────

    @staticmethod
    def _loader_len(loader: DataLoader) -> int:
        """Return ``len(loader)`` for progress estimation, or ``0`` when unknown
        (e.g. an ``IterableDataset`` loader exposes no length)."""
        try:
            return len(loader)
        except TypeError:
            return 0

    def _to_device(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            moved = [self._to_device(v) for v in x]
            return type(x)(moved)
        return x

    @staticmethod
    def _batch_size(batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.size(0)
        if isinstance(batch, dict):
            return BaseTrainer._batch_size(next(iter(batch.values())))
        if isinstance(batch, (list, tuple)) and batch:
            return BaseTrainer._batch_size(batch[0])
        raise TypeError(f"Cannot infer batch size from {type(batch)}")

    @staticmethod
    def _validated_loss(loss: torch.Tensor) -> float:
        val = loss.item()
        if not math.isfinite(val):
            raise RuntimeError(f"Invalid loss value: {val}")
        return float(val)

    # ── Internal: Precision & Config ──────────────────────────────────────────

    def _init_amp(self) -> tuple[bool, torch.dtype, torch.amp.GradScaler]:
        """Resolve automatic mixed precision from the ``amp`` setting.

        Returns ``(enabled, dtype, scaler)`` — the resolved AMP flag, the
        autocast dtype, and a :class:`~torch.amp.GradScaler` kept live only for
        fp16, since bf16's fp32-range exponent cannot underflow gradients and so
        needs no loss scaling. A disabled scaler is a transparent passthrough,
        keeping the optimizer step uniform across precisions.
        """
        amp = self.amp
        # Autocast dtype: an explicit "bf16"/"fp16" selects it; anything else
        # (the ``None``/bool forms) defaults to bf16.
        if isinstance(amp, str):
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp.lower())
            if dtype is None:
                raise ValueError(f"amp must be a bool, 'bf16', 'fp16', or None; got {amp!r}")
        else:
            dtype = torch.bfloat16

        # Enabled unless explicitly disabled (``amp=False``), and only on CUDA;
        # an explicit request on any other device is warned about and ignored.
        on_cuda = self.device.type == "cuda"
        enabled = amp is not False and on_cuda
        if amp and not on_cuda:
            self.print(
                f"amp={amp!r} was requested but device is '{self.device.type}'; "
                "training in full precision (AMP only applies to CUDA).",
                level="warn",
            )

        # A GradScaler matters only for fp16; bf16 and full precision use a
        # disabled, passthrough scaler.
        scaler = torch.amp.GradScaler(enabled=enabled and dtype is torch.float16)
        return enabled, dtype, scaler

    def _init_tf32(self) -> None:
        """Configure TF32 and the cuDNN autotuner from the ``tf32`` setting.

        TF32 lives entirely in torch's global backend flags, so this resolves the
        setting into them and keeps nothing of its own. ``None`` auto-enables both
        only when no ``seed`` is set, trading exact reproducibility for speed. Must
        run *after* :meth:`reset_seed`, whose deterministic / ``benchmark=False``
        flags this may relax.
        """
        tf32 = self.tf32
        # TF32 only applies to CUDA; elsewhere it's a no-op, and an explicit
        # request that can't be honoured is warned about and ignored.
        if self.device.type != "cuda":
            if tf32:
                self.print(
                    f"tf32={tf32!r} was requested but device is '{self.device.type}'; "
                    "ignored (TF32 only applies to CUDA).",
                    level="warn",
                )
            return

        # ``None`` follows the seed (speed when not reproducing); a bool forces it.
        enabled = (self.seed is None) if tf32 is None else bool(tf32)
        if enabled and self.seed is not None:
            self.print(
                "tf32 is enabled alongside a fixed seed; runs are only approximately "
                "reproducible, since TF32 rounds fp32 matmul inputs to a 10-bit mantissa.",
                level="warn",
            )
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        torch.set_float32_matmul_precision("high" if enabled else "highest")
        # The cuDNN autotuner is nondeterministic and assumes fixed input sizes,
        # so enable it only when not seeding (it also conflicts with the
        # deterministic flags reset_seed applies for a fixed seed).
        if enabled and self.seed is None:
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def _init_params() -> MappingProxyType[str, inspect.Parameter]:
        """The constructor's parameters — the schema of the saved config.

        The one place both sides of ``config.json`` consult: ``_customized_config``
        filters what it writes against these defaults, and :meth:`from_config`
        filters what it reads against these names.
        """
        return inspect.signature(BaseTrainer.__init__).parameters

    def _customized_config(self, provided: dict[str, Any]) -> dict[str, Any]:
        """Return only the entries whose value differs from the constructor's
        default, so a saved config records exactly what the caller customized
        (e.g. ``num_epochs`` is recorded when set, omitted when left ``None``).
        """
        params = self._init_params()
        return {
            key: value
            for key, value in provided.items()
            if params[key].default is inspect.Parameter.empty
            or value != params[key].default
        }

    # ── Internal: GPU ─────────────────────────────────────────────────────────

    @property
    def _cuda_index(self) -> int:
        """Index of the CUDA device the trainer reports on and probes."""
        return cuda_index(self.device)
