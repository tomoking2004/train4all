import abc
import gc
import importlib.metadata
import inspect
import json
import math
import multiprocessing
import platform
import random
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from train4all.utils import (
    Dashboard,
    DashboardConfig,
    LogLevel,
    MetricTable,
    UnifiedLogger,
    copy_dir,
    get_metric_plot_filename,
    get_metric_plot_title,
    print_dict_tree,
    print_flat_dict_tree,
    replace_dict_keys,
    save_curves_plot,
)

__all__ = ["BaseTrainer"]

ModuleSpec: TypeAlias = str | nn.Module | list[str | nn.Module]
_Scheduler: TypeAlias = LRScheduler | ReduceLROnPlateau
_F = TypeVar("_F", bound=Callable[..., Any])


def _require_setup(func: _F) -> _F:
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
    metrics during the final evaluation.

    Args:
        num_epochs: Total number of training epochs.
        batch_size: Batch size (informational; not used internally).
        learning_rate: Learning rate(s) forwarded to the optimizer in ``setup()``.
            ``None`` (default) sets no rate — leave it unset for learning-rate-free
            optimizers (e.g. Prodigy, D-Adaptation, Schedule-Free), which then
            keep ``learning_rate`` out of the saved config.
        max_grad_norm: Clip the global gradient norm to this value before each
            optimizer step. Disabled when ``None``. Gradients are unscaled
            first, so this is correct under fp16 AMP as well.
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
        monitor: Validation metric name that drives best-checkpoint selection and
            early stopping. Defaults to ``"loss"``. The value is read from the
            validation phase metrics each epoch.
        monitor_mode: ``"min"`` to treat lower ``monitor`` values as better
            (e.g. loss) or ``"max"`` for higher-is-better metrics (e.g. accuracy, F1).
        training_phases: Phase names treated as training phases. Defaults to ``["train"]``.
        device: Device string (e.g. ``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``).
            Auto-detected when ``None`` — prefers CUDA, then MPS, then CPU.
            On a multi-GPU machine, select a specific GPU with ``"cuda:<index>"``.
        seed: Random seed for reproducibility. Disabled if ``None``.
        run_dir: Output directory for checkpoints, metrics, and logs.
        run_snapshot_dir: Directory for a lightweight snapshot copy of ``run_dir``.
            Snapshotting is disabled when ``None``.
        resume: Resume from the latest checkpoint at the start of training.
        save_interval: Save a periodic checkpoint every *N* epochs.
        record_step_metrics: Record per-step metrics during training phases.
        step_metric_names: Step metric names to record. ``None`` records all.
        pbar_metric_names: Metric names shown in the tqdm postfix. ``None`` hides all.
        use_progress_bar: Show tqdm progress bars during epoch iteration.
        keep_progress_bar: Persist progress bars after an epoch completes.
        key_width: Column width used when printing metric and summary tables.
        debug_mode: Enable debug-level logging (forwarded to the logger).
        logger: External logger instance. A default ``UnifiedLogger`` is created if ``None``.
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

    # ── Checkpoint file stems and format version ──────────────────────────────
    _CHECKPOINT_VERSION: str = "1.1"
    _CHECKPOINT_LATEST: str  = "latest"
    _CHECKPOINT_BEST: str    = "best"

    # ── Metrics file stems ────────────────────────────────────────────────────
    _METRICS_EPOCH: str = "epoch_metrics"
    _METRICS_STEP: str  = "step_metrics"

    # ── Phase that carries the final-evaluation responsibility ────────────────
    # The single phase whose per-step metrics come from ``compute_test_metrics``
    # instead of ``compute_metrics``. Final evaluation runs once, so it owns the
    # report and can compute heavier, report-only metrics.
    _TEST_PHASE: str = "test"

    # ── GPU probe tunables ────────────────────────────────────────────────────
    _GPU_TEMP_WARN_C: int = 85   # warn above this GPU temperature (°C)
    _GPU_MEM_TTL_S: float = 2.0  # cache nvidia-smi memory reads for this long

    # ── Console / dashboard tunables ──────────────────────────────────────────
    _SEPARATOR_PAD: int       = 48   # separator rule width = key_width + this pad
    _DASH_THROTTLE_S: float   = 0.5  # minimum seconds between dashboard step writes
    _DASH_EXTRA_WAIT_S: float = 0.5  # extra wait after dashboard finalize

    def __init__(
        self,
        num_epochs: int,
        *,
        batch_size: int | None = None,
        learning_rate: float | dict[str, float] | None = None,
        max_grad_norm: float | None = None,
        amp: bool | str | None = None,
        tf32: bool | None = None,
        patience: int | None = None,
        monitor: str = "loss",
        monitor_mode: str = "min",
        training_phases: list[str] | None = None,
        device: str | None = None,
        seed: int | None = None,
        run_dir: Path | str = "run",
        run_snapshot_dir: Path | str | None = None,
        resume: bool = True,
        save_interval: int | None = None,
        record_step_metrics: bool = False,
        step_metric_names: list[str] | None = None,
        pbar_metric_names: list[str] | None = None,
        use_progress_bar: bool = True,
        keep_progress_bar: bool = False,
        key_width: int = 32,
        debug_mode: bool = False,
        logger: UnifiedLogger | None = None,
        use_dashboard: bool = False,
        dashboard_config: DashboardConfig | None = None,
    ) -> None:
        # ── Training / optimization ───────────────────────────────────────────
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        # ``amp`` is wired below via ``_init_amp`` — it depends on ``device``.

        # ── Early stopping / phases ───────────────────────────────────────────
        self.patience = patience
        self.monitor = monitor
        self.monitor_mode = self._validate_mode(monitor_mode)
        self.training_phases = training_phases or ["train"]

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
        self.resume = resume
        self.save_interval = save_interval

        # ── Metric recording ──────────────────────────────────────────────────
        self.record_step_metrics = record_step_metrics
        self.step_metric_names = step_metric_names
        self.pbar_metric_names = pbar_metric_names

        # ── Display / logging ─────────────────────────────────────────────────
        self.use_progress_bar = use_progress_bar
        self.keep_progress_bar = keep_progress_bar
        self.key_width = key_width
        self.debug_mode = debug_mode
        self.logger = logger or self._create_default_logger()

        # ── Internal: models / optimization objects ───────────────────────────
        self._models: dict[str, nn.Module] = {}
        self._optimizer: Optimizer | None = None
        self._scheduler: _Scheduler | None = None

        # ── Internal: training state ──────────────────────────────────────────
        self._current_epoch: int = 0
        self._best_metric: float = self._worst_metric()
        self._best_epoch: int | None = None
        self._epochs_no_improve: int = 0

        # ── Internal: metrics ─────────────────────────────────────────────────
        self._epoch_metrics: MetricTable = {}
        self._step_metrics: MetricTable = {}

        # ── Internal: misc state ──────────────────────────────────────────────
        self._setup_done: bool = False
        self._cache: dict[str, Any] = {}
        self._ckpt_excludes: set[str] = set()
        self._ckpt_extras: dict[str, Any] = {}
        self._last_dash_write: float = 0.0

        # ── Internal: GPU-memory probe state ──────────────────────────────────
        # Initialized lazily and reused across steps so the progress bar never
        # pays an NVML init/shutdown per iteration.
        self._pynvml: Any = None
        self._nvml_handle: Any = None
        self._nvml_failed: bool = False
        self._gpu_mem_cache: tuple[int, int, int] = (0, 0, 0)
        self._gpu_mem_cache_t: float = 0.0

        # ── Derived initialization (depends on the attributes set above) ──────
        self._init_amp(amp)  # needs ``device`` and ``logger``

        # Reproducibility config — only the arguments the caller actually
        # customized (anything left at its default is omitted), so the saved
        # config is minimal and round-trips via ``MyTrainer(**config)``.
        self._config: dict[str, Any] = self._customized_config({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "amp": amp,
            "tf32": tf32,
            "patience": patience,
            "monitor": monitor,
            "monitor_mode": monitor_mode,
            "training_phases": training_phases,
            "seed": seed,
        })

        if self.seed is not None:
            self._set_seed(self.seed)  # needs ``device``
        self._init_tf32(tf32)  # needs ``device`` and ``seed``; runs after _set_seed

        cfg = dashboard_config or DashboardConfig()
        self._dashboard: Dashboard | None = (
            Dashboard(cfg, self.run_dir) if use_dashboard and cfg.enabled else None
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

        Train and validation share one cheap, per-epoch metric path
        (``compute_metrics``); the test phase runs once for final reporting, so
        it gets its own override here for heavier, report-only metrics (AUC,
        per-class F1, calibration, confusion matrices, …). The default simply
        delegates to ``compute_metrics``, so test mirrors validation until you
        override it.

        Only the ``"test"`` phase routes here (see ``_TEST_PHASE``); every other
        phase — train, val, and any custom phase — uses ``compute_metrics``.

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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Called while a full checkpoint dict is being built, before it is written.

        Mutate ``checkpoint`` in place to persist custom state (the counterpart
        of :meth:`on_load_checkpoint`). Not called for weights-only saves.

        Args:
            checkpoint: The checkpoint dict about to be saved.
        """

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Called after a full checkpoint has been restored, with the raw dict.

        Use it to read back anything stored via :meth:`on_save_checkpoint` or
        ``update_checkpoint_extras()``. Not called for weights-only loads.

        Args:
            checkpoint: The checkpoint dict that was just loaded.
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

    def on_epoch_start(self, epoch: int | None, loader: DataLoader, phase: str) -> None:
        """
        Called at the start of every epoch, for both training and evaluation phases.

        The step cache has been cleared before this hook fires.

        Args:
            epoch: Current epoch number, or ``None`` when called outside the training loop.
            loader: The DataLoader for this epoch.
            phase: Phase name (e.g. ``"train"``, ``"val"``, ``"test"``).
        """

    def on_epoch_end(
        self, epoch: int | None, loader: DataLoader, metrics: dict[str, float], phase: str,
    ) -> None:
        """
        Called at the end of every epoch, for both training and evaluation phases.

        Epoch metrics for the completed phase are already recorded and accessible
        via :meth:`get_epoch_metrics` when this hook fires.

        Args:
            epoch: Current epoch number, or ``None`` when called outside the training loop.
            loader: The DataLoader for this epoch.
            metrics: Aggregated metrics computed during the epoch.
            phase: Phase name (e.g. ``"train"``, ``"val"``, ``"test"``).
        """

    def on_step_start(self, step: int | None, batch: Any, phase: str) -> None:
        """
        Called at the start of every step.

        Args:
            step: 1-based step index within the current epoch, or ``None`` when called
                outside the standard epoch loop.
            batch: The batch about to be processed.
            phase: Phase name (e.g. ``"train"``, ``"val"``).
        """

    def on_step_end(
        self, step: int | None, batch: Any, metrics: dict[str, float], phase: str,
    ) -> None:
        """
        Called at the end of every step.

        Args:
            step: 1-based step index within the current epoch, or ``None`` when called
                outside the standard epoch loop.
            batch: The batch that was just processed.
            metrics: Step metrics from ``compute_metrics()`` (or
                ``compute_test_metrics()`` during the test phase), plus ``"loss"``.
            phase: Phase name (e.g. ``"train"``, ``"val"``).
        """

    # ── Main Training Workflow ────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """
        Train the model for the configured number of epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data. Required when
                early stopping (``patience``) is enabled.
        """
        self.prepare_training()

        if self.is_training_completed():
            self.print("\n⏹️  Training already completed.\n\n")
            return

        if self.should_stop_early():
            self.print("\n⏹️  Early stopping condition already met. No training will run.\n\n")
            return

        if self.patience is not None and val_loader is None:
            self.print(
                "Early stopping is enabled (patience set) but no val_loader was "
                f"provided — it can never trigger without the '{self.monitor}' "
                "metric from a validation phase.",
                level="warn",
            )

        self._init_dashboard(train_loader, val_loader)

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

                train_metrics = self._execute_epoch(
                    train_loader, phase="train", training=True, epoch=epoch,
                )
                self.print_metrics(train_metrics, phase="train")

                monitor_value: float | None = None
                if val_loader is not None:
                    val_metrics = self._execute_epoch(
                        val_loader, phase="val", training=False, epoch=epoch,
                    )
                    self.print_metrics(val_metrics, phase="val")
                    monitor_value = val_metrics.get(self.monitor)

                self.finalize_train_epoch(monitor_value)
                self.save_artifacts(phases=["train", "val"])
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
        self._finalize_dashboard()
        self.clear_cuda_cache()
        self.print(f"\n✅ Training completed. Duration: {str(duration).split('.')[0]}\n\n")

    @_require_setup
    def test(
        self,
        test_loader: DataLoader,
        use_best: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate the model on the test set.

        Per-step metrics come from ``compute_test_metrics`` (which defaults to
        ``compute_metrics``), so override it to report heavier, test-only metrics.

        Args:
            test_loader: DataLoader for test data.
            use_best: Load the best checkpoint before evaluating.

        Returns:
            Mapping of metric name to value.
        """
        if use_best:
            self._load_best_checkpoint()

        self.print("── Test Epoch\n")
        metrics = self._execute_epoch(test_loader, phase=self._TEST_PHASE, training=False)
        self.print_metrics(metrics, phase=self._TEST_PHASE)
        self.print()
        # Terminal operation, like the end of ``train()`` — release cached
        # blocks now that no further epochs depend on allocator reuse.
        self.clear_cuda_cache()
        return metrics

    @_require_setup
    def execute_epoch(
        self,
        loader: DataLoader,
        phase: str = "custom",
        epoch: int | None = None,
        print_metrics: bool = False,
    ) -> dict[str, float]:
        """
        Run one full epoch on a DataLoader.

        Args:
            loader: DataLoader to iterate.
            phase: Phase name (e.g. ``"train"``, ``"val"``).
            epoch: Epoch number, used for hook callbacks.
            print_metrics: Print aggregated metrics after the epoch.

        Returns:
            Aggregated metrics for the epoch.
        """
        training = self._is_training_phase(phase)
        metrics = self._execute_epoch(loader, phase, training, epoch=epoch)
        if print_metrics:
            self.print_metrics(metrics, phase)
        return metrics

    @_require_setup
    def execute_step(
        self,
        batch: Any,
        phase: str,
        step: int | None = None,
        print_metrics: bool = False,
    ) -> dict[str, float]:
        """
        Run one step on a single batch.

        Args:
            batch: Batch of data.
            phase: Phase name (e.g. ``"train"``).
            step: Step number, used for hook callbacks.
            print_metrics: Print computed metrics after the step.

        Returns:
            Metrics computed for the step.
        """
        training = self._is_training_phase(phase)
        metrics = self._execute_step(batch, phase, training, step=step)
        if print_metrics:
            self.print_metrics(metrics, phase)
        return metrics

    # ── Setup & State ─────────────────────────────────────────────────────────

    def prepare_training(self) -> None:
        """
        Prepare the trainer for a new run.

        Prints the environment summary, saves the config, calls ``ensure_setup()``,
        optionally resumes from the latest checkpoint, then prints model and
        optimization summaries.
        """
        self.print_env_summary()
        self.save_config()
        self.print_config()
        self.ensure_setup()

        if self.resume and self.has_latest_checkpoint():
            self._load_latest_checkpoint()

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
        Reset the trainer to a clean initial state.

        Clears setup, training state, metrics, and the step cache.
        """
        self.clear_setup()
        self.reset_training_state()
        self.clear_metrics()
        self.clear_cache()

    # ── Epoch Control ─────────────────────────────────────────────────────────

    def epoch_iterator(self) -> Iterator[tuple[int, int]]:
        """
        Yield ``(current_epoch, num_epochs)`` for each training epoch.

        Automatically increments the internal epoch counter.
        """
        while self._current_epoch < self.num_epochs:
            self._current_epoch += 1
            yield self._current_epoch, self.num_epochs

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
        self._scheduler_step(monitor_value)

    def reset_training_state(self) -> None:
        """Reset the epoch counter, best-metric tracking, and early-stopping counters."""
        self._current_epoch = 0
        self._best_metric = self._worst_metric()
        self._best_epoch = None
        self._epochs_no_improve = 0

    def is_training_completed(self) -> bool:
        """Return ``True`` if the epoch counter has reached ``num_epochs``."""
        return self._current_epoch >= self.num_epochs

    def is_best_epoch(self) -> bool:
        """Return ``True`` if the current epoch achieved the best ``monitor`` value."""
        return self._current_epoch == self._best_epoch

    def should_stop_early(self) -> bool:
        """Return ``True`` if the early-stopping patience has been exhausted."""
        return self.patience is not None and self._epochs_no_improve >= self.patience

    # ── Model / Optimizer / Scheduler ────────────────────────────────────────

    def set_models(
        self,
        models: dict[str, nn.Module],
        overwrite: bool = True,
        set_attr: bool = False,
    ) -> None:
        """
        Register multiple models, moving each to the training device.

        Args:
            models: Mapping of name to model instance.
            overwrite: Replace any existing entry with the same name.
            set_attr: Also assign each model as ``self.<name>``.
        """
        for name, model in models.items():
            self.set_model(name, model, overwrite=overwrite, set_attr=set_attr)

    def set_model(
        self,
        name: str,
        model: nn.Module,
        overwrite: bool = True,
        set_attr: bool = False,
    ) -> None:
        """
        Register a single model, moving it to the training device.

        Args:
            name: Model name.
            model: Model instance.
            overwrite: Replace an existing entry with the same name.
            set_attr: Also assign the model as ``self.<name>``.
        """
        if not overwrite and name in self._models:
            return
        self._models[name] = model.to(self.device)
        if set_attr:
            setattr(self, name, model)

    def clear_models(self) -> None:
        """Remove all registered models."""
        self._models.clear()

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
        phases: list[str] | None = None,
    ) -> None:
        """
        Save checkpoints and export all metric artifacts for the current state.

        Args:
            metric_names: Metrics to include. ``None`` includes all.
            phases: Phases to include. ``None`` includes all.
        """
        self._save_checkpoints()
        if self._epoch_metrics:
            self.save_epoch_metric_plots(metric_names=metric_names, phases=phases)
            self.export_epoch_metrics(metric_names=metric_names, phases=phases)
        if self._step_metrics:
            self.save_step_metric_plots(metric_names=metric_names, phases=phases)
            self.export_step_metrics(metric_names=metric_names, phases=phases)

    @_require_setup
    def save_checkpoints(self) -> None:
        """Save the latest, best, and periodic (if configured) checkpoints."""
        self._save_checkpoints()

    @_require_setup
    def save_checkpoint(self, path: Path | str) -> None:
        """
        Save a full checkpoint to a specific path.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        self._save_torch(path, self._build_checkpoint(), f"💾 Checkpoint saved: {path.name}")

    @_require_setup
    def save_weights(self, path: Path | str) -> None:
        """
        Save only model weights to a specific path.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        self._save_torch(
            path,
            self._build_checkpoint(weights_only=True),
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
        self.print("💾 Loading checkpoint ...")
        self.print(f" {'─' * (self.key_width + self._SEPARATOR_PAD)}")
        self._load_checkpoint(Path(path), strict=strict, key_map=key_map)

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
        self.print("💾 Loading model weights ...")
        self.print(f" {'─' * (self.key_width + self._SEPARATOR_PAD)}")
        self._load_checkpoint(Path(path), strict=strict, key_map=key_map, weights_only=True)

    @_require_setup
    def load_latest_checkpoint(self) -> None:
        """Load the most recently saved checkpoint."""
        self._load_latest_checkpoint()

    @_require_setup
    def load_best_checkpoint(self) -> None:
        """Load the checkpoint from the best validation epoch."""
        self._load_best_checkpoint()

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

        Mirrors :meth:`update_checkpoint_extras`. After a full checkpoint is
        loaded, this reflects the restored extras, so it is the symmetric way to
        read back static metadata without overriding :meth:`on_load_checkpoint`.
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
        phases: list[str] | None = None,
    ) -> MetricTable:
        """
        Return epoch-level metrics, optionally filtered.

        Args:
            metric_names: Metrics to include. ``None`` returns all.
            phases: Phases to include. ``None`` returns all.

        Returns:
            Filtered metric table.
        """
        return self._filter_metrics(self._epoch_metrics, metric_names=metric_names, phases=phases)

    def get_step_metrics(
        self,
        metric_names: list[str] | None = None,
        phases: list[str] | None = None,
    ) -> MetricTable:
        """
        Return step-level metrics, optionally filtered.

        Args:
            metric_names: Metrics to include. ``None`` returns all.
            phases: Phases to include. ``None`` returns all.

        Returns:
            Filtered metric table.
        """
        return self._filter_metrics(self._step_metrics, metric_names=metric_names, phases=phases)

    def clear_metrics(self) -> None:
        """Clear all recorded epoch and step metrics."""
        self._epoch_metrics.clear()
        self._step_metrics.clear()

    def save_epoch_metric_plots(
        self,
        metric_names: list[str] | None = None,
        phases: list[str] | None = None,
    ) -> None:
        """
        Save epoch-level metric curve plots.

        Args:
            metric_names: Metrics to plot. ``None`` plots all.
            phases: Phases to include. ``None`` includes all.
        """
        metrics = self.get_epoch_metrics(metric_names=metric_names, phases=phases)
        self._save_metric_plots(metrics, xlabel="epoch", split_phases=False)
        self.print("📈 Epoch-level metric curves saved.")

    def save_step_metric_plots(
        self,
        metric_names: list[str] | None = None,
        phases: list[str] | None = None,
    ) -> None:
        """
        Save step-level metric curve plots.

        Args:
            metric_names: Metrics to plot. ``None`` plots all.
            phases: Phases to include. ``None`` includes all.
        """
        metrics = self.get_step_metrics(metric_names=metric_names, phases=phases)
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
        phases: list[str] | None = None,
    ) -> Path:
        """
        Export epoch-level metrics to a JSON file.

        Args:
            metric_names: Metrics to export. ``None`` exports all.
            phases: Phases to include. ``None`` includes all.

        Returns:
            Path to the written JSON file.
        """
        metrics = self.get_epoch_metrics(metric_names=metric_names, phases=phases)
        path = self.get_epoch_metrics_path()
        self._export_metrics(metrics, path)
        self.print(f"📄 Epoch-level metrics exported: {path.name}")
        return path

    def export_step_metrics(
        self,
        metric_names: list[str] | None = None,
        phases: list[str] | None = None,
    ) -> Path:
        """
        Export step-level metrics to a JSON file.

        Args:
            metric_names: Metrics to export. ``None`` exports all.
            phases: Phases to include. ``None`` includes all.

        Returns:
            Path to the written JSON file.
        """
        metrics = self.get_step_metrics(metric_names=metric_names, phases=phases)
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
        phase: str | None = None,
        prefix: str | None = None,
    ) -> Path:
        """Return the output path for a metric curve plot PNG."""
        filename = get_metric_plot_filename(metric_name, phase=phase, prefix=prefix)
        return self._plots_dir / filename

    # ── Logging & Display ─────────────────────────────────────────────────────

    def get_env_info(self) -> dict[str, Any]:
        """Return the system and runtime environment summary as a dict."""
        try:
            import wmi
            cpu_name = wmi.WMI().Win32_Processor()[0].Name
        except Exception:
            cpu_name = platform.uname().processor or platform.processor() or "Unknown"

        disk = shutil.disk_usage(self.run_dir)
        info: dict[str, Any] = {
            "OS":        f"{platform.system()} {platform.release()}",
            "CPU":       cpu_name,
            "CPU cores": multiprocessing.cpu_count(),
            "RAM":       f"{psutil.virtual_memory().total / 1e9:.2f} GB",
            "Disk":      f"{disk.free / 1e9:.2f} / {disk.total / 1e9:.2f} GB free",
        }
        if torch.cuda.is_available():
            idx = self._cuda_index
            props = torch.cuda.get_device_properties(idx)
            info["GPU"]   = f"cuda:{idx} {torch.cuda.get_device_name(idx)}"
            info["VRAM"]  = f"{props.total_memory / 1e9:.2f} GB"
            info["CUDA"]  = torch.version.cuda
            info["cuDNN"] = str(torch.backends.cudnn.version())
        else:
            info |= {"GPU": "Not available", "VRAM": "-", "CUDA": "-", "cuDNN": "-"}
        info["Python"]  = platform.python_version()
        info["PyTorch"] = torch.__version__
        for pkg in ("torchvision", "torchaudio"):
            try:
                info[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pass
        return info

    def print_env_summary(self) -> None:
        """Print a system and runtime environment summary for experiment reproducibility."""
        self.print_dict_tree(self.get_env_info(), header="🖥️  Environment")

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
            if trainable == total:
                result[name] = f"{total:,} params"
            elif trainable:
                result[name] = f"{trainable:,} / {total:,} trainable"
            else:
                result[name] = "frozen"
        return result

    def print_model_summary(self) -> None:
        """Print the name and parameter counts of all registered models."""
        self.print_dict_tree(self.get_model_summary(), header="🧩 Model")

    def print_optimization_summary(self) -> None:
        """Print the optimizer and scheduler class names."""
        tree = {
            "Optimizer": self._optimizer.__class__.__name__ if self._optimizer else "-",
            "Scheduler": self._scheduler.__class__.__name__ if self._scheduler else "-",
        }
        self.print_dict_tree(tree, header="⚡ Optimization")

    def print_status(self) -> None:
        """Print the current training state (epoch, best monitored value, and recent metrics)."""
        tree: dict[str, Any] = {
            "Completed epochs":   self._current_epoch,
            f"Best val {self.monitor}": (
                f"{self._best_metric:.4f}  (epoch {self._best_epoch})"
                if self._best_epoch is not None else "-"
            ),
            "Stagnant epochs":    self._epochs_no_improve,
            "Last epoch metrics": self._format_epoch_metrics() or "-",
        }
        self.print_dict_tree(tree, header="📊 Status")

    def print_metrics(self, metrics: dict[str, float], phase: str) -> None:
        """
        Print a flat metrics table for a given phase.

        Args:
            metrics: Mapping of metric name to value.
            phase: Phase label shown in the header.
        """
        print_flat_dict_tree(
            data=metrics,
            header=f"📊 {phase.capitalize()}",
            key_width=self.key_width,
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
            key_width=self.key_width,
            trailing_newline=True,
            print_fn=self.print,
        )

    def print(self, msg: str | None = None, level: LogLevel = "info", indent: int = 0) -> None:
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
        Copy a lightweight snapshot of ``run_dir`` into ``run_snapshot_dir``.

        No-op when ``run_snapshot_dir`` is ``None``.

        Args:
            exclude: Top-level directory names to omit from the snapshot.
        """
        if self.run_snapshot_dir is None:
            return
        copy_dir(src=self.run_dir, dst=self.run_snapshot_dir, exclude=exclude)

    # ── GPU Utilities ─────────────────────────────────────────────────────────

    def print_gpu_temperature(self) -> None:
        """Print the current GPU temperature via ``nvidia-smi``."""
        if not torch.cuda.is_available():
            self.print("CUDA not available. Skipping GPU temperature check.", level="warn")
            return

        try:
            result = subprocess.run(
                [
                    "nvidia-smi", "-i", str(self._cuda_index),
                    "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            temp_str = result.stdout.strip()
            temp = int(temp_str) if temp_str.isdigit() else None
            if temp is not None:
                self.print(f"🌡️  GPU Temperature: {temp} °C")
                if temp > self._GPU_TEMP_WARN_C:
                    self.print("GPU temperature is high! Consider cooling down.", level="warn")
            else:
                self.print("GPU temperature info unavailable or invalid.", level="warn")
        except FileNotFoundError:
            self.print("'nvidia-smi' not found. Skipping GPU temperature check.", level="warn")
        except subprocess.CalledProcessError as e:
            self.print(f"'nvidia-smi' command failed: {e}", level="warn")
        except Exception as e:
            self.print(f"Failed to get GPU temperature: {e}", level="warn")

    @staticmethod
    def clear_cuda_cache() -> None:
        """Free Python-held tensor references and clear the CUDA memory cache."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    # ── Internal: Training Loop ───────────────────────────────────────────────

    def _execute_epoch(
        self,
        loader: DataLoader,
        phase: str,
        training: bool,
        epoch: int | None = None,
    ) -> dict[str, float]:
        self.clear_cache()
        self._set_training_mode(training)
        self.on_epoch_start(epoch, loader, phase)
        metrics = self._run_epoch(loader, phase, training)
        self._record_epoch_metrics(metrics, phase)
        self._dash_update()
        self.on_epoch_end(epoch, loader, metrics, phase)
        # NOTE: no per-epoch ``empty_cache()`` here — releasing cached blocks
        # back to the driver every epoch forces the allocator to re-acquire
        # them next epoch, which slows training. A single cleanup runs at the
        # end of ``train()``; call ``clear_cuda_cache()`` manually if needed.
        return metrics

    def _run_epoch(self, loader: DataLoader, phase: str, training: bool) -> dict[str, float]:
        pbar: tqdm | None = (
            tqdm(loader, desc=f"{phase.capitalize()} Epoch", leave=self.keep_progress_bar)
            if self.use_progress_bar else None
        )
        accumulated: dict[str, float] = {}
        num_samples = 0
        max_step = self._loader_len(loader)  # 0 for length-less IterableDataset loaders
        for step, batch in enumerate(pbar or loader, 1):
            batch_size = self._batch_size(batch)
            metrics = self._execute_step(batch, phase, training, step=step, max_step=max_step)
            self._accumulate_metrics(accumulated, metrics, batch_size)
            num_samples += batch_size
            if pbar is not None:
                self._update_pbar(pbar, metrics)

        return self._average_metrics(accumulated, num_samples)

    def _execute_step(
        self,
        batch: Any,
        phase: str,
        training: bool,
        step: int | None = None,
        max_step: int = 0,
    ) -> dict[str, float]:
        batch = self._to_device(batch)
        self.on_step_start(step, batch, phase)
        metrics = self._run_step(batch, training, phase)
        # Throttle intermediate steps, but always write the final step of a
        # phase so the gauge's inner ring reaches 100% before the phase resets.
        self._dash_update(
            step=step or 0, max_step=max_step, step_metrics=metrics, phase=phase,
            throttle=max_step <= 0 or step != max_step,
        )
        if training and self.record_step_metrics:
            self._record_step_metrics(metrics, phase)
        self.on_step_end(step, batch, metrics, phase)
        return metrics

    def _run_step(self, batch: Any, training: bool, phase: str) -> dict[str, float]:
        with torch.set_grad_enabled(training):
            with self._autocast():
                loss = self.compute_loss(batch)
            # Backward and the optimizer update run outside autocast — as AMP
            # requires — while still under the grad-enabled context above.
            if training:
                self._optimizer_step(loss)
        # The final test phase reports its own (possibly heavier) metrics; every
        # other phase shares the cheap per-epoch ``compute_metrics`` path.
        metric_fn = (
            self.compute_test_metrics if phase == self._TEST_PHASE else self.compute_metrics
        )
        # Metrics never need gradients. Computing them under no_grad avoids
        # building and immediately discarding a graph on every step, which
        # otherwise leaks both memory and time during training phases (where
        # grad would still be enabled by the context above).
        with torch.no_grad(), self._autocast():
            metrics = metric_fn(batch)
        metrics["loss"] = self._validated_loss(loss)
        return metrics

    def _autocast(self) -> torch.autocast:
        """Autocast context for the configured AMP device/dtype.

        A single source of truth for the two call sites in :meth:`_run_step`; a
        transparent no-op when AMP is disabled (``enabled=False``).
        """
        return torch.autocast(
            self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled,
        )

    # ── Internal: Optimizer / Scheduler ──────────────────────────────────────

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        if self._optimizer is None:
            raise RuntimeError("An optimizer is required for training.")
        # A disabled scaler (bf16 / no AMP) is a transparent passthrough, so
        # this single path is correct at every precision.
        self._optimizer.zero_grad(set_to_none=True)
        self._scaler.scale(loss).backward()
        # Fires before unscaling/clipping. Under fp16 AMP the gradients here are
        # still multiplied by the loss-scale factor (see ``on_before_optimizer_step``
        # for the unscaled, post-clip view).
        self.on_after_backward()
        if self.max_grad_norm is not None:
            # Gradients must be unscaled into "real" units before clipping;
            # unscale_ is a no-op on a disabled scaler, so this stays correct
            # at bf16 / full precision too.
            self._scaler.unscale_(self._optimizer)
            # Clip exactly the parameters the optimizer owns — the same set
            # unscale_ just rescaled and step() will update. Tying the three
            # together keeps the clipped norm correct under fp16 AMP even if
            # the optimizer's parameters differ from get_trainable_params().
            params = [p for group in self._optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.on_before_optimizer_step()
        self._scaler.step(self._optimizer)
        self._scaler.update()

    def _scheduler_step(self, monitor_value: float | None = None) -> None:
        if self._scheduler is None:
            return
        if isinstance(self._scheduler, ReduceLROnPlateau):
            if monitor_value is None:
                raise ValueError(
                    f"ReduceLROnPlateau requires the '{self.monitor}' metric, but it "
                    "was None (no validation this epoch, or the metric is missing). "
                    "Provide a val_loader exposing it, or use a different scheduler."
                )
            self._scheduler.step(monitor_value)
        else:
            self._scheduler.step()

    # ── Internal: Early Stopping / Mode ──────────────────────────────────────

    @staticmethod
    def _validate_mode(monitor_mode: str) -> str:
        if monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max'; got {monitor_mode!r}")
        return monitor_mode

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

    def _is_training_phase(self, phase: str) -> bool:
        return phase in self.training_phases

    # ── Internal: Checkpoints ─────────────────────────────────────────────────

    def _save_torch(self, path: Path, data: Any, success_msg: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(data, path)
            self.print(success_msg)
        except Exception as e:
            self.print(f"Failed to save {path.name}: {e}", level="warn")

    def _save_checkpoints(self) -> None:
        # No explicit mkdir: ``_save_torch`` creates each destination's parent.
        checkpoint = self._build_checkpoint()

        latest_path = self.get_latest_checkpoint_path()
        self._save_torch(latest_path, checkpoint, f"💾 Latest checkpoint saved: {latest_path.name}")

        if self.is_best_epoch():
            best_path = self.get_best_checkpoint_path()
            self._save_torch(best_path, checkpoint, f"🏆 Best checkpoint saved: {best_path.name}")

        if self.save_interval and self._current_epoch % self.save_interval == 0:
            epoch_path = self.get_checkpoint_path(f"epoch_{self._current_epoch}")
            self._save_torch(epoch_path, checkpoint, f"💾 Epoch {self._current_epoch} checkpoint saved: {epoch_path.name}")

    def _load_named_checkpoint(self, label: str, path: Path) -> None:
        """Print a labeled header then load a checkpoint from *path*."""
        self.print(f"{label} ...")
        self.print(f" {'─' * (self.key_width + self._SEPARATOR_PAD)}")
        self._load_checkpoint(path)

    def _load_latest_checkpoint(self) -> None:
        self._load_named_checkpoint("💾 Loading latest checkpoint", self.get_latest_checkpoint_path())

    def _load_best_checkpoint(self) -> None:
        self._load_named_checkpoint("🏆 Loading best checkpoint", self.get_best_checkpoint_path())

    def _load_checkpoint(
        self,
        path: Path | str,
        strict: bool = False,
        key_map: dict[str, str] | None = None,
        weights_only: bool = False,
    ) -> None:
        checkpoint = self._torch_load(path)
        if not checkpoint:
            return

        loaded: dict[str, str] = {}

        for name, state_dict in checkpoint.get("models", {}).items():
            status = self._load_model_state_dict(
                model=self._models.get(name),
                name=name,
                state_dict=state_dict,
                strict=strict,
                key_map=key_map,
            )
            if status is not None:
                loaded[name] = status

        if not weights_only:
            # Optimizer, scheduler, and scaler share one load-and-record path;
            # each key doubles as the checkpoint key and the status label.
            for name, obj in {
                "optimizer": self._optimizer,
                "scheduler": self._scheduler,
                "scaler":    self._scaler,
            }.items():
                status = self._load_state_dict(obj, name, checkpoint.get(name))
                if status is not None:
                    loaded[name] = status

            ts = checkpoint.get("training_state", {})
            self._current_epoch     = ts.get("current_epoch", self._current_epoch)
            # ``best_metric``/``best_epoch`` are the current keys; fall back to the
            # legacy ``best_val_loss``/``best_val_epoch`` so older checkpoints load.
            self._best_metric       = ts.get("best_metric", ts.get("best_val_loss",  self._best_metric))
            self._best_epoch        = ts.get("best_epoch",  ts.get("best_val_epoch", self._best_epoch))
            self._epochs_no_improve = ts.get("epochs_no_improve", self._epochs_no_improve)
            loaded["training_state"] = "restored"

            saved_metrics = checkpoint.get("metrics", {})
            self._epoch_metrics = saved_metrics.get("epoch_metrics", self._epoch_metrics)
            self._step_metrics  = saved_metrics.get("step_metrics",  self._step_metrics)
            loaded["metrics"] = "restored"

            # Round-trip the extras saved by ``update_checkpoint_extras()`` and
            # let subclasses restore any custom state from the raw checkpoint.
            extras = checkpoint.get("extras")
            if extras:
                self._ckpt_extras.update(extras)
                loaded["extras"] = "restored"
            self.on_load_checkpoint(checkpoint)

        print_flat_dict_tree(
            data=loaded,
            header=None,
            key_width=self.key_width,
            print_fn=self.print,
        )

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

    def _torch_load(self, path: Path | str) -> dict[str, Any] | None:
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            self.print(f"File not found: {path}", level="warn")
        except RuntimeError as e:
            self.print(f"Load error: {e}", level="warn")
        except Exception as e:
            self.print(f"Unexpected error while loading '{path}': {e}", level="warn")
        self.print()
        return None

    def _build_checkpoint(self, weights_only: bool = False) -> dict[str, Any]:
        checkpoint: dict[str, Any] = {
            "version": self._CHECKPOINT_VERSION,
            "models": {
                k: v.state_dict()
                for k, v in self._models.items()
                if k not in self._ckpt_excludes
            },
            "extras": dict(self._ckpt_extras),
        }
        if not weights_only:
            checkpoint.update({
                "optimizer": self._optimizer.state_dict() if self._optimizer else None,
                "scheduler": self._scheduler.state_dict() if self._scheduler else None,
                "scaler": self._scaler.state_dict(),
                "training_state": {
                    "current_epoch":     self._current_epoch,
                    "best_metric":       self._best_metric,
                    "best_epoch":        self._best_epoch,
                    "epochs_no_improve": self._epochs_no_improve,
                },
                "metrics": {
                    "epoch_metrics": self._epoch_metrics,
                    "step_metrics":  self._step_metrics,
                },
            })
            # Let subclasses inject custom state into the full checkpoint dict;
            # weights-only saves stay pure (models + extras) by design.
            self.on_save_checkpoint(checkpoint)
        return checkpoint

    # ── Internal: Metrics ─────────────────────────────────────────────────────

    def _record_epoch_metrics(self, metrics: dict[str, float], phase: str) -> None:
        self._record_metrics(self._epoch_metrics, metrics, phase)

    def _record_step_metrics(self, metrics: dict[str, float], phase: str) -> None:
        if self.step_metric_names is not None:
            metrics = {k: v for k, v in metrics.items() if k in self.step_metric_names}
        self._record_metrics(self._step_metrics, metrics, phase)

    @staticmethod
    def _record_metrics(target: MetricTable, metrics: dict[str, float], phase: str) -> None:
        for name, value in metrics.items():
            target.setdefault(name, {}).setdefault(phase, []).append(value)

    @staticmethod
    def _filter_metrics(
        metrics: MetricTable,
        metric_names: list[str] | None = None,
        phases: list[str] | None = None,
    ) -> MetricTable:
        result: MetricTable = {}
        for name, phase_dict in metrics.items():
            if not isinstance(phase_dict, dict):
                continue
            if metric_names is not None and name not in metric_names:
                continue
            filtered = {
                phase: values
                for phase, values in phase_dict.items()
                if (phases is None or phase in phases) and values
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
    def _average_metrics(accumulated: dict[str, float], num_samples: int) -> dict[str, float]:
        if num_samples == 0:
            return {}
        return {k: v / num_samples for k, v in accumulated.items()}

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
                for phase, values in phase_dict.items():
                    if not values:
                        continue
                    save_curves_plot(
                        curves={phase: values},
                        path=self.get_metric_plot_path(metric_name, phase=phase, prefix=path_prefix),
                        title=get_metric_plot_title(metric_name, phase=phase, prefix=title_prefix),
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
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.print(f"Failed to write {label}: {e}\n", level="warn")

    def _format_epoch_metrics(self) -> dict[str, str]:
        return {
            metric_name: "  ".join(
                f"{phase}={values[-1]:.4f}" if values else f"{phase}=N/A"
                for phase, values in phase_dict.items()
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
            used, total, _ = self._gpu_memory_mib()
            display["GPU"] = f"{used}/{total}"
        pbar.set_postfix(display)

    def _create_default_logger(self) -> UnifiedLogger:
        return UnifiedLogger(
            f"trainer_{id(self)}",
            log_path=self.run_dir / self._LOG_FILENAME,
            verbose=True,
            debug_mode=self.debug_mode,
            file_mode="a",
        )

    # ── Internal: Dashboard ───────────────────────────────────────────────────

    def _init_dashboard(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
    ) -> None:
        if self._dashboard is None:
            return
        self._dashboard.initialize(
            self._config,
            env_info=self.get_env_info(),
            model_summary=self.get_model_summary(),
            training_phases=self.training_phases,
            monitor=self.monitor,
            train_steps=self._loader_len(train_loader),
            val_steps=self._loader_len(val_loader),
        )
        self.print(f"📊 Dashboard: {self._dashboard.url}\n")

    @staticmethod
    def _loader_len(loader: DataLoader | None) -> int:
        """Return ``len(loader)`` for progress estimation, or ``0`` when unknown
        (e.g. an ``IterableDataset`` loader exposes no length)."""
        if loader is None:
            return 0
        try:
            return len(loader)
        except TypeError:
            return 0

    def _dash_update(
        self,
        *,
        step: int = 0,
        max_step: int = 0,
        step_metrics: dict[str, float] | None = None,
        phase: str = "",
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
            is_gradient_phase=self._is_training_phase(phase),
            step=step,
            max_step=max_step,
            step_metrics=step_metrics,
            phase=phase,
            learning_rate=lr,
            gpu_mem=self._dash_gpu_mem(),
        )

    def _dash_gpu_mem(self) -> tuple[float, float] | None:
        """Return ``(used_gb, total_gb)`` GPU memory for the dashboard footprint
        bar, or ``None`` when not on CUDA or no reading is available.

        Uses the same NVML/``nvidia-smi`` probe as the progress bar and reports
        decimal GB to match the VRAM total shown in the Environment panel.
        """
        if self.device.type != "cuda":
            return None
        used_mib, total_mib, _ = self._gpu_memory_mib()
        if total_mib <= 0:
            return None
        mib_to_gb = (1 << 20) / 1e9
        return (used_mib * mib_to_gb, total_mib * mib_to_gb)

    def _finalize_dashboard(self) -> None:
        if self._dashboard is None:
            return
        self._dashboard.finalize(
            self._current_epoch,
            self.num_epochs,
            self._epoch_metrics,
            self._best_metric,
            self._best_epoch,
            self._epochs_no_improve,
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

    # ── Internal: Seed & GPU ──────────────────────────────────────────────────

    @property
    def _cuda_index(self) -> int:
        """Index of the CUDA device the trainer reports on and probes."""
        if self.device.type == "cuda" and self.device.index is not None:
            return self.device.index
        return torch.cuda.current_device() if torch.cuda.is_available() else 0

    def _init_amp(self, amp: bool | str | None) -> None:
        """Initialize automatic mixed precision from the ``amp`` argument.

        Sets :attr:`_amp_enabled`, :attr:`_amp_dtype`, and :attr:`_scaler` — a
        :class:`~torch.amp.GradScaler` kept live only for fp16, since bf16's
        fp32-range exponent cannot underflow gradients and so needs no loss
        scaling. A disabled scaler is a transparent passthrough, keeping the
        optimizer step uniform across precisions.
        """
        self.amp = amp

        # Autocast dtype: an explicit "bf16"/"fp16" selects it; anything else
        # (the ``None``/bool forms) defaults to bf16.
        if isinstance(amp, str):
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp.lower())
            if dtype is None:
                raise ValueError(f"amp must be a bool, 'bf16', 'fp16', or None; got {amp!r}")
            self._amp_dtype = dtype
        else:
            self._amp_dtype = torch.bfloat16

        # Enabled unless explicitly disabled (``amp=False``), and only on CUDA;
        # an explicit request on any other device is warned about and ignored.
        on_cuda = self.device.type == "cuda"
        self._amp_enabled = amp is not False and on_cuda
        if amp and not on_cuda:
            self.print(
                f"amp={amp!r} was requested but device is '{self.device.type}'; "
                "training in full precision (AMP only applies to CUDA).",
                level="warn",
            )

        # A GradScaler matters only for fp16; bf16 and full precision use a
        # disabled, passthrough scaler.
        self._scaler = torch.amp.GradScaler(
            enabled=self._amp_enabled and self._amp_dtype is torch.float16,
        )

    def _init_tf32(self, tf32: bool | None) -> None:
        """Configure TF32 and the cuDNN autotuner from the ``tf32`` argument.

        ``None`` auto-enables both only when no ``seed`` is set, trading exact
        reproducibility for speed. Must run *after* :meth:`_set_seed`, whose
        deterministic / ``benchmark=False`` flags this may relax.
        """
        self.tf32 = tf32

        # TF32 only applies to CUDA; elsewhere it's a no-op, and an explicit
        # request that can't be honored is warned about and ignored.
        if self.device.type != "cuda":
            if tf32:
                self.print(
                    f"tf32={tf32!r} was requested but device is '{self.device.type}'; "
                    "ignored (TF32 only applies to CUDA).",
                    level="warn",
                )
            self._tf32_enabled = False
            return

        # ``None`` follows the seed (speed when not reproducing); a bool forces it.
        enabled = (self.seed is None) if tf32 is None else bool(tf32)
        self._tf32_enabled = enabled
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
        # deterministic flags _set_seed applies for a fixed seed).
        if enabled and self.seed is None:
            torch.backends.cudnn.benchmark = True

    def _customized_config(self, provided: dict[str, Any]) -> dict[str, Any]:
        """Return only the entries whose value differs from the constructor's
        default, so a saved config records exactly what the caller customized.

        ``num_epochs`` (which has no default) is always kept.
        """
        params = inspect.signature(BaseTrainer.__init__).parameters
        return {
            key: value
            for key, value in provided.items()
            if params[key].default is inspect.Parameter.empty
            or value != params[key].default
        }

    def _set_seed(self, seed: int) -> None:
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

    def _gpu_memory_mib(self) -> tuple[int, int, int]:
        """Return ``(used, total, free)`` GPU memory in MiB for the selected device.

        NVML is initialized once and the device handle is reused on every
        subsequent call, so querying memory inside the per-step progress bar
        costs a single cheap lookup rather than an init/shutdown cycle. When
        NVML is unavailable, falls back to a ``nvidia-smi`` query whose result
        is cached for :attr:`_GPU_MEM_TTL_S` seconds to avoid spawning a
        subprocess on every step.
        """
        # Reuse the live NVML handle; drop it on error so the init path retries.
        if self._nvml_handle is not None:
            try:
                return self._nvml_mib(self._nvml_handle)
            except Exception:
                self._nvml_handle = None

        if not self._nvml_failed:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._pynvml = pynvml
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._cuda_index)
                return self._nvml_mib(self._nvml_handle)
            except Exception:
                self._nvml_failed = True  # NVML unavailable — use the smi fallback

        now = time.time()
        if now - self._gpu_mem_cache_t < self._GPU_MEM_TTL_S:
            return self._gpu_mem_cache
        self._gpu_mem_cache_t = now
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i", str(self._cuda_index),
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
            )
            used, total = (int(x) for x in output.split(","))  # int() tolerates whitespace
            self._gpu_mem_cache = (used, total, total - used)
        except Exception:
            self._gpu_mem_cache = (0, 0, 0)
        return self._gpu_mem_cache

    def _nvml_mib(self, handle: Any) -> tuple[int, int, int]:
        """Query an NVML device *handle*, returning ``(used, total, free)`` in MiB."""
        mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.used >> 20, mem.total >> 20, mem.free >> 20
