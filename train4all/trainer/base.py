import abc
import json
import math
import shutil
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from train4all.utils import (
    copy_dir, exclude_none, get_metric_plot_filename, get_metric_plot_title,
    save_curves_plot, print_dict_tree, print_flat_dict_tree, replace_dict_keys, UnifiedLogger
)

__all__ = ["BaseTrainer"]


ModuleSpec = str | nn.Module | List[str | nn.Module]
MetricTable = Dict[str, Dict[str, List[float]]]


def setup_required(func):
    """
    Ensure that `self.ensure_setup()` is executed before the wrapped method.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.ensure_setup()
        return func(self, *args, **kwargs)
    return wrapper


class BaseTrainer(abc.ABC):
    """
    BaseTrainer is a generic and extensible training framework for PyTorch models.

    Subclass this and implement:
        - setup()
        - compute_loss()
        - compute_metrics()

    Args:
        num_epochs (int): Total number of epochs.

        batch_size (int | None): Batch size.

        learning_rate (float | dict[str, float]): Learning rate(s).

        device (str | None): Device for training.

        seed (int | None): Random seed.

        run_dir (str | Path):
            Output directory for this training run. Defaults to "run".

        run_snapshot_dir (str | Path | None):
            Optional directory for storing a lightweight snapshot of this run.
            If None, snapshotting is disabled.

        patience (int | None): Early stopping patience.

        resume (bool): Resume from the latest checkpoint.

        save_interval (int | None): Save a checkpoint every N epochs.

        training_phases (list[str] | None):
            Phases treated as training phases. Defaults to ["train"].

        record_step_metrics (bool):
            Whether to record metrics at each step (training only).

        metric_names_to_record (list[str] | None):
            Metric names to record at each step.  
            If None, all available metrics are recorded.

        metric_names_to_display (list[str] | None):
            Metric names to display in tqdm.  
            If None, no metrics are displayed.

        use_progress_bar (bool): Whether to display tqdm progress bars.

        keep_progress_bar (bool): Whether to keep progress bars after finishing.

        key_width (int): Display width of metric names.

        debug_mode (bool):
            Debug/verbose flag. Not used by BaseTrainer itself; for use in subclasses.

        logger (UnifiedLogger | None):
            Optional external logger instance.
            If None, a default `UnifiedLogger` is created automatically.
    """

    def __init__(
        # --- Core training parameters ---
        self,
        num_epochs: int,
        batch_size: int | None = None,
        learning_rate: float | Dict[str, float] = 1e-4,
        device: str | None = None,
        seed: int | None = None,

        # --- Output / directories ---
        run_dir: str | Path = "run",
        run_snapshot_dir: str | Path | None = None,

        # --- Trainer policies ---
        patience: int | None = None,
        resume: bool = True,
        save_interval: int | None = None,
        training_phases: List[str] | None = None,

        # --- Logging / visualization ---
        record_step_metrics: bool = False,
        metric_names_to_record: List[str] | None = None,
        metric_names_to_display: List[str] | None = None,
        use_progress_bar: bool = True,
        keep_progress_bar: bool = False,
        key_width: int = 32,
        debug_mode: bool = False,
        logger: UnifiedLogger | None = None,
    ) -> None:

        # --- Core training parameters ---
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.seed = seed

        # --- Output / directories ---
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.plots_dir = self.run_dir / "plots"

        self.run_snapshot_dir = Path(run_snapshot_dir) if run_snapshot_dir else None

        # --- Trainer policies ---
        self.patience = patience
        self.resume = resume
        self.save_interval = save_interval
        self.training_phases = training_phases or ["train"]

        # --- Logging / visualization ---
        self.record_step_metrics = record_step_metrics
        self.metric_names_to_record = metric_names_to_record
        self.metric_names_to_display = metric_names_to_display
        self.use_progress_bar = use_progress_bar
        self.keep_progress_bar = keep_progress_bar
        self.key_width = key_width
        self.debug_mode = debug_mode
        self.logger = logger or self._create_default_logger()

        # --- Models / optimizer / scheduler ---
        self.models: Dict[str, nn.Module] = {}
        self.optimizer: Optimizer | None = None
        self.scheduler: _LRScheduler | None = None

        # --- Training state variables ---
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.best_val_epoch: int | None = None
        self.epochs_no_improve: int = 0

        # --- Storage for metrics ---
        self.step_metrics: MetricTable = {}
        self.epoch_metrics: MetricTable = {}

        # --- Internal helpers ---
        self._is_setup_done: bool = False
        self._cache: Dict[str, Any] = {}
        self._checkpoint_excludes: set[str] = set()
        self._checkpoint_extras: Dict[str, Any] = {}

        self._config: Dict[str, Any] = exclude_none({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "patience": patience,
            "save_interval": save_interval,
            "training_phases": training_phases,
        })

        if self.seed is not None:
            self._set_seed(self.seed)
    
    # ----- Abstract Methods -----
    
    @abc.abstractmethod
    def setup(self) -> None:
        """
        Initialize and register all models, optimizers, and schedulers.
        This method is called once before training or evaluation starts.

        Example:
        ```python
        def setup(self):
            self.backbone = Backbone()
            self.classifier = Classifier()

            self.set_models({
                "backbone": self.backbone,
                "classifier": self.classifier,
            })

            self.freeze("backbone")  # Freeze parts if needed
            
            params = self.get_trainable_params()
            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
            self.set_optimizer(optimizer)
        ```
        """
        pass

    @abc.abstractmethod
    def compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Calculate the loss for the given batch.
        Intermediate results can be cached via set_cache() for reuse in compute_metrics().

        Args:
            batch (Any): A batch of input data.

        Returns:
            torch.Tensor: Scalar tensor representing the loss.

        Example:
        ```python
        def compute_loss(self, batch):
            x, y = batch["input"], batch["target"]
            features = self.backbone(x)
            logits = self.classifier(features)
            self.set_cache("logits", logits.detach())
            return F.cross_entropy(logits, y)
        ```
        """
        pass

    @abc.abstractmethod
    def compute_metrics(self, batch: Any) -> Dict[str, float]:
        """
        Compute evaluation metrics for the given batch.
        Cached intermediate results can be retrieved via get_cache().

        Args:
            batch (Any): A batch of input data.

        Returns:
            dict[str, float]: A dictionary mapping metric names to their computed values.

        Example:
        ```python
        def compute_metrics(self, batch):
            logits = self.get_cache("logits")
            preds = logits.argmax(dim=1)
            acc = (preds == batch["target"]).float().mean().item()
            return {"accuracy": acc}
        ```
        """
        pass

    # ----- Optional Hooks (intended to be overridden in subclass) -----

    def on_set_training_mode(self, training: bool) -> None:
        """Called when switching between training and evaluation modes."""
        pass

    def on_training_start(self) -> None:
        """Called once before the training loop starts."""
        pass

    def on_training_end(self) -> None:
        """Called once after the training loop ends."""
        pass

    def on_train_epoch_start(self, epoch: int) -> None:
        """Called at the beginning of a training loop epoch (before train/val execution)."""
        pass

    def on_train_epoch_end(self, epoch: int) -> None:
        """Called at the end of a training loop epoch (after all artifacts/checkpoints have been saved)."""
        pass

    def on_epoch_start(self, epoch: int | None, loader: DataLoader, phase: str) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int | None, loader: DataLoader, metrics: Dict[str, float], phase: str) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_start(self, step: int | None, batch: Any, phase: str) -> None:
        """Called at the beginning of each step."""
        pass

    def on_step_end(self, step: int | None, batch: Any, metrics: Dict[str, float], phase: str) -> None:
        """Called at the end of each step."""
        pass

    # ----- Main Workflow Methods -----

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """
        Train the model for a specified number of epochs.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader | None): DataLoader for validation data.
                Required if early stopping (patience) is used.
        """
        self.prepare_training()

        if self.is_training_completed():
            self.print("\n⏹️  Training already completed.\n\n")
            return

        if self.should_stop_early():
            self.print("\n⏹️  Early stopping condition already met. No training will run.\n\n")
            return

        start_time = datetime.now()
        self.print(f"\n🕛 Training started at {start_time:%Y-%m-%d %H:%M:%S}\n")
        self.on_training_start()

        for epoch, max_epoch in self.epoch_iterator():
            self.print(f"\n[Epoch {epoch}/{max_epoch}]\n")
            self.on_train_epoch_start(epoch)

            train_metrics = self._execute_epoch(train_loader, phase="train", training=True, epoch=epoch)
            self.print_metrics(train_metrics, phase="train")

            val_loss: float | None = None
            if val_loader is not None:
                val_metrics = self._execute_epoch(val_loader, phase="val", training=False, epoch=epoch)
                self.print_metrics(val_metrics, phase="val")
                val_loss = val_metrics.get("loss")

            self.finalize_train_epoch(val_loss)
            self.save_checkpoints_and_metrics(phases=["train", "val"])

            self.on_train_epoch_end(epoch)

            if self.should_stop_early():
                self.print(f"⏹️  Early stopping triggered at epoch {epoch}.\n")
                break

            self.print()

        self.on_training_end()
        self.clear_cuda_cache()

        duration = datetime.now() - start_time
        self.print(f"\n✅ Training finished. Duration: {str(duration).split('.')[0]}\n\n")

    @setup_required
    def test(
        self,
        test_loader: DataLoader,
        use_best: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test set.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            use_best (bool): If True, load the best checkpoint before testing.

        Returns:
            dict[str, float]: Dictionary mapping metric names to their float values.
        """
        if use_best:
            self._load_best_checkpoint()

        self.print("\n[Test Epoch]\n")
        metrics = self._execute_epoch(test_loader, phase="test", training=False)
        self.print_metrics(metrics, phase="test")

        self.print()
        return metrics
    
    @setup_required
    def execute_epoch(
        self,
        loader: DataLoader,
        phase: str = "custom",
        epoch: int | None = None,
        print_metrics: bool = False,
    ) -> dict[str, float]:
        """
        Run one epoch on a DataLoader, recording metrics.

        Args:
            loader (DataLoader): DataLoader to iterate.
            phase (str): Phase name (e.g., 'train', 'val', 'test').
            epoch (int | None): Optional epoch number.
            print_metrics (bool): Whether to print metrics.

        Returns:
            dict[str, float]: Aggregated metrics for the epoch.
        """
        training = self._is_training_phase(phase)
        metrics = self._execute_epoch(loader, phase, training, epoch=epoch)

        if print_metrics:
            self.print_metrics(metrics, phase)
        
        return metrics
    
    def _execute_epoch(
        self,
        loader: DataLoader,
        phase: str,
        training: bool,
        epoch: int | None = None,
    ) -> Dict[str, float]:
        self._set_training_mode(training)

        self.on_epoch_start(epoch, loader, phase)
        metrics = self._run_epoch(loader, phase, training)
        self.on_epoch_end(epoch, loader, metrics, phase)
        self.clear_cuda_cache()

        self._record_epoch_metrics(metrics, phase)

        return metrics
    
    def _run_epoch(self, loader: DataLoader, phase: str, training: bool) -> Dict[str, float]:
        iterator = loader
        if self.use_progress_bar:
            iterator = tqdm(loader, desc=f"{phase.capitalize()} Epoch", leave=self.keep_progress_bar)

        num_samples = 0
        accumulated_metrics: Dict[str, float] = {}

        for step, batch in enumerate(iterator, 1):
            metrics = self._execute_step(batch, phase, training, step=step)

            batch_size = self._get_batch_size(batch)
            self._accumulate_metrics(accumulated_metrics, metrics, batch_size)
            num_samples += batch_size

            if self.use_progress_bar:
                self._update_pbar(iterator, metrics)

        return self._compute_average_metrics(accumulated_metrics, num_samples)

    @setup_required
    def execute_step(
        self,
        batch: Any,
        phase: str,
        step: int | None = None,
        print_metrics: bool = False,
    ) -> Dict[str, float]:
        """
        Run one step on a batch, recording metrics.

        Args:
            batch (Any): Batch of data to process.
            phase (str): Phase name (e.g., 'train', 'val', 'test').
            step (int | None): Optional step number.
            print_metrics (bool): Whether to print metrics.

        Returns:
            dict[str, float]: Metrics computed for the step.
        """
        training = self._is_training_phase(phase)
        metrics = self._execute_step(batch, phase, training, step=step)

        if print_metrics:
            self.print_metrics(metrics, phase)
        
        return metrics
    
    def _execute_step(
        self,
        batch: Any,
        phase: str,
        training: bool,
        step: int | None = None,
    ) -> Dict[str, float]:
        batch = self._to_device(batch)

        self.on_step_start(step, batch, phase)
        metrics = self._run_step(batch, training)
        self.on_step_end(step, batch, metrics, phase)

        if training and self.record_step_metrics:
            self._record_step_metrics(metrics, phase)
        
        return metrics
    
    def _run_step(self, batch: Any, training: bool) -> Dict[str, float]:
        with torch.set_grad_enabled(training):
            loss = self.compute_loss(batch)
            metrics = self.compute_metrics(batch)

            if training:
                self._step_optimizer(loss)

        metrics["loss"] = self._extract_valid_loss_value(loss)
        return metrics
    
    # ----- Checkpoint Management -----

    @setup_required
    def load_latest_checkpoint(self) -> None:
        """Load the latest checkpoint."""
        self._load_latest_checkpoint()
    
    def _load_latest_checkpoint(self) -> None:
        path = self.get_latest_checkpoint_path()
        self.print(f"💾 Loading latest checkpoint ...")
        self._load_checkpoint(path)

    @setup_required
    def load_best_checkpoint(self) -> None:
        """Load the best checkpoint."""
        self._load_best_checkpoint()

    def _load_best_checkpoint(self) -> None:
        path = self.get_best_checkpoint_path()
        self.print(f"🏆 Loading best checkpoint ...")
        self._load_checkpoint(path)

    @setup_required
    def load_checkpoint(
        self,
        path: str | Path,
        strict: bool = False,
        key_map: Dict[str, str] | None = None,
    ) -> None:
        """
        Load a full checkpoint.

        Args:
            path (str | Path): Path to the checkpoint file.
            strict (bool): Enforce exact key matching when loading the state dict.
            key_map (dict[str, str] | None): Optional mapping to rename keys.
        """
        path = Path(path)
        self.print(f"💾 Loading checkpoint ...")
        self._load_checkpoint(path, strict=strict, key_map=key_map)

    @setup_required
    def load_weights(
        self,
        path: str | Path,
        strict: bool = False,
        key_map: Dict[str, str] | None = None,
    ) -> None:
        """
        Load only model weights from a checkpoint.

        Args:
            path (str | Path): Path to the checkpoint file.
            strict (bool): Enforce exact key matching when loading the state dict.
            key_map (dict[str, str] | None): Optional mapping to rename keys.
        """
        path = Path(path)
        self.print(f"💾 Loading model weights ...")
        self._load_checkpoint(path, strict=strict, key_map=key_map, weights_only=True)

    def _load_checkpoint(
        self,
        path: str | Path,
        strict: bool = False,
        key_map: Dict[str, str] | None = None,
        weights_only: bool = False,
    ) -> None:
        checkpoint = self._load_torch_file(path)
        if not checkpoint:
            return

        for name, state_dict in checkpoint.get("models", {}).items():
            self._load_state_dict(
                obj=self.models.get(name),
                name=name,
                state_dict=state_dict,
                is_model_weights=True,
                strict=strict,
                key_map=key_map,
            )

        if not weights_only:
            self._load_state_dict(
                obj=self.optimizer,
                name="optimizer",
                state_dict=checkpoint.get("optimizer"),
                is_model_weights=False,
            )

            self._load_state_dict(
                obj=self.scheduler,
                name="scheduler",
                state_dict=checkpoint.get("scheduler"),
                is_model_weights=False,
            )

            ts = checkpoint.get("training_state", {})
            self.current_epoch = ts.get("current_epoch", self.current_epoch)
            self.best_val_loss = ts.get("best_val_loss", self.best_val_loss)
            self.best_val_epoch = ts.get("best_val_epoch", self.best_val_epoch)
            self.epochs_no_improve = ts.get("epochs_no_improve", self.epochs_no_improve)
            self.print(f"  - {'training_state':<{self.key_width}}: restored")

            metrics = checkpoint.get("metrics", {})
            self.step_metrics = metrics.get("step_metrics", self.step_metrics)
            self.epoch_metrics = metrics.get("epoch_metrics", self.epoch_metrics)
            self.print(f"  - {'metrics':<{self.key_width}}: restored")

        self.print()

    def _load_state_dict(
        self,
        obj: Any | None,
        name: str,
        state_dict: Dict[str, Any] | None,
        is_model_weights: bool = False,
        strict: bool = False,
        key_map: Dict[str, str] | None = None,
    ) -> None:
        if not obj or not state_dict:
            return

        if is_model_weights and key_map:
            state_dict = replace_dict_keys(state_dict, key_map)

        try:
            if is_model_weights:
                missing, unexpected = obj.load_state_dict(state_dict, strict=strict)
                self.print(f"  - {name:<{self.key_width}}: weights loaded")
                if not strict:
                    if missing:
                        self.print(f"  ↳ {'Missing keys':<{self.key_width}}: {missing}")
                    if unexpected:
                        self.print(f"  ↳ {'Unexpected keys':<{self.key_width}}: {unexpected}")
            else:
                obj.load_state_dict(state_dict)
                self.print(f"  - {name:<{self.key_width}}: state loaded")
        except Exception as e:
            self.print(f"{name}: failed to load ({e})", level="warn", indent=2)
    
    def _load_torch_file(self, path: str | Path) -> Dict[str, Any] | None:
        try:
            return torch.load(path, map_location=self.device)
        except FileNotFoundError:
            self.print(f"File not found: {path}", level="warn")
        except RuntimeError as e:
            self.print(f"Load error: {e}", level="warn")
        except Exception as e:
            self.print(f"Unexpected error while loading '{path}': {e}", level="warn")
        self.print()
        return None
    
    def save_checkpoints_and_metrics(
        self,
        metric_names: list[str] | None = None,
        phases: list[str] | None = None,
    ) -> None:
        """
        Save all training checkpoints and associated metrics for the current training state.

        Args:
            metric_names (list[str] | None): Metric names to save. `None` saves all.
            phases (list[str] | None): Phases to include (e.g., 'train', 'val'). `None` includes all.

        Behavior:
            - Saves all training checkpoints.
            - Saves and exports epoch-wise metrics (plots and JSON) if any.
            - Saves and exports step-wise metrics (plots and JSON) if any.
        """
        self._save_checkpoints()

        if self.epoch_metrics:
            self.save_epoch_metric_plots(metric_names=metric_names, phases=phases)
            self.export_epoch_metrics(metric_names=metric_names, phases=phases)

        if self.step_metrics:
            self.save_step_metric_plots(metric_names=metric_names, phases=phases)
            self.export_step_metrics(metric_names=metric_names, phases=phases)
    
    @setup_required
    def save_checkpoints(self) -> None:
        """
        Save all training checkpoints.

        Handles:
        - Latest checkpoint
        - Best checkpoint (if current epoch is the best so far)
        - Periodic checkpoint (if `save_interval` is set and current epoch matches)
        """
        self._save_checkpoints()
    
    def _save_checkpoints(self) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._get_checkpoint()

        try:
            # Latest checkpoint
            latest_path = self.get_latest_checkpoint_path()
            torch.save(checkpoint, latest_path)
            self.print(f"💾 Latest checkpoint saved: {latest_path.name}")

            # Best checkpoint
            if self.is_best_epoch():
                best_path = self.get_best_checkpoint_path()
                torch.save(checkpoint, best_path)
                self.print(f"🏆 Best checkpoint saved: {best_path.name}")

            # Periodic checkpoint
            if self.save_interval and self.current_epoch % self.save_interval == 0:
                epoch_path = self.get_checkpoint_path(f"epoch_{self.current_epoch}")
                torch.save(checkpoint, epoch_path)
                self.print(f"💾 Epoch {self.current_epoch} checkpoint saved: {epoch_path.name}")

        except Exception as e:
            self.print(f"Failed to save checkpoint: {e}", level="warn")

    @setup_required
    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save a checkpoint to a specific path.

        Args:
            path (str | Path): Destination path to save the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(self._get_checkpoint(), path)
            self.print(f"💾 Checkpoint saved: {path.name}")
        except Exception as e:
            self.print(f"Failed to save checkpoint: {e}", level="warn")

    @setup_required
    def save_weights(self, path: str | Path) -> None:
        """
        Save only the model weights to a specific path.

        Args:
            path (str | Path): Destination path to save model weights.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(self._get_checkpoint(weights_only=True), path)
            self.print(f"💾 Model weights saved: {path.name}")
        except Exception as e:
            self.print(f"Failed to save model weights: {e}", level="warn")
    
    def backup_checkpoint(self, path: str | Path) -> None:
        """
        Create a backup of a checkpoint file by appending `.bak` to the filename.

        Args:
            path (str | Path): Path to the checkpoint file to backup.
        """
        path = Path(path)
        backup_path = path.with_name(path.name + ".bak")
        shutil.copyfile(path, backup_path)
        self.print(f"📦 Backup created: {backup_path.name}")

    def _get_checkpoint(self, weights_only: bool = False) -> Dict[str, Any]:
        checkpoint = {
            "version": "1.0",
            "models": {
                k: v.state_dict() for k, v in self.models.items()
                if k not in self._checkpoint_excludes
            },
            "extras": dict(self._checkpoint_extras),
        }

        if not weights_only:
            checkpoint.update({
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "training_state": {
                    "current_epoch": self.current_epoch,
                    "best_val_loss": self.best_val_loss,
                    "best_val_epoch": self.best_val_epoch,
                    "epochs_no_improve": self.epochs_no_improve,
                },
                "metrics": {
                    "step_metrics": self.step_metrics,
                    "epoch_metrics": self.epoch_metrics,
                },
            })

        return checkpoint
    
    def exclude_from_checkpoint(self, names: str | List[str]) -> None:
        """
        Exclude specified model(s) from checkpoint.

        Args:
            names (str | list[str]): Registered model name(s) to exclude.
        """
        if isinstance(names, str):
            names = [names]

        invalid = [n for n in names if n not in self.models]
        if invalid:
            raise ValueError(f"Unregistered model(s) cannot be excluded: {invalid}")

        self._checkpoint_excludes.update(names)

    def update_checkpoint_extras(self, extras: Dict[str, Any]) -> None:
        """
        Update the checkpoint extras with new key-value pairs.

        Args:
            extras (dict[str, Any]): Key-value pairs to add or overwrite in the checkpoint's 'extras' dictionary.
        """
        self._checkpoint_extras.update(extras)

    def has_latest_checkpoint(self) -> bool:
        """Return True if the latest checkpoint exists."""
        return self.get_latest_checkpoint_path().exists()

    def has_best_checkpoint(self) -> bool:
        """Return True if the best checkpoint exists."""
        return self.get_best_checkpoint_path().exists()

    def get_latest_checkpoint_path(self) -> Path:
        """Return the path to the latest checkpoint file."""
        return self.get_checkpoint_path("latest")

    def get_best_checkpoint_path(self) -> Path:
        """Return the path to the best checkpoint file."""
        return self.get_checkpoint_path("best")
    
    def get_checkpoint_path(self, name: str) -> Path:
        return self.checkpoints_dir / f"{name}.pth"

    # ----- Config Management -----

    def update_config(self, entries: Dict[str, Any]) -> None:
        """
        Update the trainer configuration with new entries.

        Args:
            entries (dict[str, Any]): Key-value pairs to add or overwrite in the trainer configuration.
        """
        self._config.update(entries)

    def save_config(self) -> None:
        """
        Save the trainer configuration as a JSON file.
        """
        path = self.get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            self.print(f"Failed to save config: {e}\n", level="warn")
    
    def get_config_path(self) -> Path:
        return self.run_dir / "config.json"
    
    # ----- Model / Optimizer / Scheduler Management -----
    
    def set_models(self, models: Dict[str, nn.Module], overwrite: bool = True, set_attr: bool = False) -> None:
        """
        Set multiple models.

        Args:
            models (dict[str, nn.Module]): Mapping of names to model instances.
            overwrite (bool): Whether to replace existing entries.
            set_attr (bool): Whether to set each model as an attribute.
        """
        for name, model in models.items():
            self.set_model(name, model, overwrite=overwrite, set_attr=set_attr)

    def set_model(self, name: str, model: nn.Module, overwrite: bool = True, set_attr: bool = False) -> None:
        """
        Set a single model.

        Args:
            name (str): Model name.
            model (nn.Module): Model instance.
            overwrite (bool): Whether to replace an existing entry.
            set_attr (bool): Whether to set the model as an attribute.
        """
        if not overwrite and name in self.models:
            return

        model = model.to(self.device)
        self.models[name] = model

        if set_attr:
            setattr(self, name, model)

    def clear_models(self) -> None:
        """
        Clear all registered models.
        """
        self.models.clear()

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """
        Set the optimizer.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance.
        """
        self.optimizer = optimizer

    def clear_optimizer(self) -> None:
        """
        Clear the optimizer.
        """
        self.optimizer = None

    def set_scheduler(self, scheduler: _LRScheduler) -> None:
        """
        Set the learning rate scheduler.

        Args:
            scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler instance.
        """
        self.scheduler = scheduler

    def clear_scheduler(self) -> None:
        """
        Clear the scheduler.
        """
        self.scheduler = None
    
    def _to_device(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._to_device(v) for v in x]
        return x
    
    def reset_parameters(self, targets: ModuleSpec | None = None) -> None:
        """
        Reset parameters of the specified model(s).

        This method calls `reset_parameters()` on all submodules of each target
        module that implement it. Submodules without `reset_parameters()` are
        skipped.

        Args:
            targets (ModuleSpec | None):
                Model(s) to reset. If None, all registered models are reset.
        """
        modules = self._resolve_modules(targets)
        for module in modules:
            module.apply(self._reset_parameters)

    @staticmethod
    def _reset_parameters(m: nn.Module) -> None:
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()

    def freeze(self, targets: ModuleSpec) -> None:
        """
        Freeze parameters of the specified model(s) by disabling gradients.

        Args:
            targets (ModuleSpec): Model(s) whose parameters will be frozen.
        """
        self._set_requires_grad(targets, False)

    def unfreeze(self, targets: ModuleSpec) -> None:
        """
        Unfreeze parameters of the specified model(s) by enabling gradients.

        Args:
            targets (ModuleSpec): Model(s) whose parameters will be unfrozen.
        """
        self._set_requires_grad(targets, True)

    def _set_requires_grad(self, targets: ModuleSpec, flag: bool) -> None:
        modules = self._resolve_modules(targets)
        for m in modules:
            for p in m.parameters():
                p.requires_grad = flag

    def get_trainable_params(
        self,
        targets: ModuleSpec | None = None,
        exclude_targets: ModuleSpec | None = None,
    ) -> List[nn.Parameter]:
        """
        Return trainable parameters from the specified model(s).

        Args:
            targets (ModuleSpec | None): Models to include. `None` includes all models.
            exclude_targets (ModuleSpec | None): Models to exclude.

        Returns:
            List[nn.Parameter]: Unique parameters with requires_grad=True.
        """
        modules = self._resolve_modules(targets)

        if exclude_targets is not None:
            exclude = set(self._resolve_modules(exclude_targets))
            modules = [m for m in modules if m not in exclude]

        seen = set()
        params: List[nn.Parameter] = []

        for m in modules:
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        return params
    
    def _resolve_modules(self, targets: ModuleSpec | None) -> List[nn.Module]:
        if targets is None:
            return list(self.models.values())

        if not isinstance(targets, list):
            targets = [targets]

        return [self._resolve_module(t) for t in targets]
    
    def _resolve_module(self, target: str | nn.Module) -> nn.Module:
        if isinstance(target, str):
            if target not in self.models:
                raise ValueError(f"Model '{target}' not registered")
            return self.models[target]
        if isinstance(target, nn.Module):
            return target
        raise TypeError(f"Unsupported module target: {type(target)}")
    
    def _step_optimizer(self, loss: torch.Tensor) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer required for training.")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _step_scheduler(self, val_loss: float | None = None) -> None:
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is None:
                raise ValueError(
                    "ReduceLROnPlateau scheduler requires a metric value (val_loss), "
                    "but val_loss is None. Provide val_loss or use a different scheduler."
                )
            self.scheduler.step(val_loss)
            return

        self.scheduler.step()
    
    # ----- Metrics Management -----

    def _record_epoch_metrics(self, metrics: Dict[str, float], phase: str) -> None:
        self._record_metrics(self.epoch_metrics, metrics, phase)

    def _record_step_metrics(self, metrics: Dict[str, float], phase: str) -> None:
        if self.metric_names_to_record is not None:
            metrics = {k: v for k, v in metrics.items() if k in self.metric_names_to_record}

        self._record_metrics(self.step_metrics, metrics, phase)

    @staticmethod
    def _record_metrics(
        target: MetricTable,
        metrics: Dict[str, float],
        phase: str,
    ) -> None:
        for metric_name, value in metrics.items():
            phase_values = target.setdefault(metric_name, {}).setdefault(phase, [])
            phase_values.append(value)
    
    def get_epoch_metrics(
        self,
        metric_names: List[str] | None = None,
        phases: List[str] | None = None,
    ) -> MetricTable:
        """
        Retrieve epoch-wise metrics filtered by metric names and phases.

        Args:
            metric_names (list[str] | None): Metric names to retrieve. `None` returns all available metrics.
            phases (list[str] | None): Phases to include. `None` includes all phases.

        Returns:
            MetricTable: Filtered metric table.
        """
        return self._filter_metrics(self.epoch_metrics, metric_names=metric_names, phases=phases)

    def get_step_metrics(
        self,
        metric_names: List[str] | None = None,
        phases: List[str] | None = None,
    ) -> MetricTable:
        """
        Retrieve step-wise metrics filtered by metric names and phases.

        Args:
            metric_names (list[str] | None): Metric names to retrieve. `None` returns all available metrics.
            phases (list[str] | None): Phases to include. `None` includes all phases.

        Returns:
            MetricTable: Filtered metric table.
        """
        return self._filter_metrics(self.step_metrics, metric_names=metric_names, phases=phases)
    
    @staticmethod
    def _filter_metrics(
        metrics: MetricTable,
        metric_names: List[str] | None = None,
        phases: List[str] | None = None,
    ) -> MetricTable:
        return {
            metric_name: {
                phase: values
                for phase, values in phase_dict.items()
                if (phases is None or phase in phases) and values
            }
            for metric_name, phase_dict in metrics.items()
            if isinstance(phase_dict, dict) and (metric_names is None or metric_name in metric_names)
        }
    
    def clear_all_metrics(self) -> None:
        """
        Clear all recorded metrics for both epochs and steps.

        This resets `epoch_metrics` and `step_metrics` to empty states.
        """
        self.epoch_metrics.clear()
        self.step_metrics.clear()
    
    @staticmethod
    def _accumulate_metrics(
        accumulated_metrics: Dict[str, float],
        metrics: Dict[str, float],
        weight: float,
    ) -> None:
        for metric_name, value in metrics.items():
            accumulated_metrics[metric_name] = accumulated_metrics.get(metric_name, 0.0) + value * weight

    @staticmethod
    def _compute_average_metrics(
        accumulated_metrics: Dict[str, float],
        num_samples: int,
    ) -> Dict[str, float]:
        if num_samples == 0:
            return {}
        return {k: v / num_samples for k, v in accumulated_metrics.items()}
    
    def save_epoch_metric_plots(self, metric_names: List[str] | None = None, phases: List[str] | None = None) -> None:
        """
        Save epoch-wise metric plots.

        Args:
            metric_names (list[str] | None): Metric names to plot. `None` plots all metrics.
            phases (list[str] | None): Phases to include. `None` includes all phases.
        """
        metrics = self.get_epoch_metrics(metric_names=metric_names, phases=phases)
        self._save_metric_plots(
            metrics=metrics,
            xlabel="epoch",
            #title_prefix="epoch-wise",
            #path_prefix="epoch",
            split_phases=False,
        )
        self.print("📈 Epoch-wise metric curves saved.")

    def save_step_metric_plots(self, metric_names: List[str] | None = None, phases: List[str] | None = None) -> None:
        """
        Save step-wise metric plots.

        Args:
            metric_names (list[str] | None): Metric names to plot. `None` plots all metrics.
            phases (list[str] | None): Phases to include. `None` includes all phases.
        """
        metrics = self.get_step_metrics(metric_names=metric_names, phases=phases)
        self._save_metric_plots(
            metrics=metrics,
            xlabel="step",
            title_prefix="step-wise",
            path_prefix="step",
            split_phases=True,
        )
        self.print("📈 Step-wise metric curves saved.")

    def _save_metric_plots(
        self,
        metrics: MetricTable,
        xlabel: str,
        title_prefix: str = "",
        path_prefix: str = "",
        split_phases: bool = False,
    ) -> None:
        for metric_name, phase_dict in metrics.items():
            if all(not values for values in phase_dict.values()):
                continue

            if split_phases:
                for phase, values in phase_dict.items():
                    if not values:
                        continue

                    curves = {phase: values}
                    path = self.get_metric_plot_path(metric_name, phase=phase, prefix=path_prefix)
                    title = get_metric_plot_title(metric_name, phase=phase, prefix=title_prefix)

                    save_curves_plot(
                        curves=curves,
                        path=path,
                        title=title,
                        xlabel=xlabel,
                        ylabel=metric_name,
                    )
            else:
                curves = {p: v for p, v in phase_dict.items() if v}
                path = self.get_metric_plot_path(metric_name, prefix=path_prefix)
                title = get_metric_plot_title(metric_name, prefix=title_prefix)

                save_curves_plot(
                    curves=curves,
                    path=path,
                    title=title,
                    xlabel=xlabel,
                    ylabel=metric_name,
                )
    
    def get_metric_plot_path(
        self,
        metric_name: str,
        phase: str | None = None,
        prefix: str | None = None,
    ) -> Path:
        filename = get_metric_plot_filename(metric_name, phase=phase, prefix=prefix)
        return self.plots_dir / filename

    def export_epoch_metrics(self, metric_names: List[str] | None = None, phases: List[str] | None = None) -> None:
        """
        Export epoch-wise metrics as a JSON file.

        Args:
            metric_names (list[str] | None): Metric names to export. `None` exports all metrics.
            phases (list[str] | None): Phases to include. `None` includes all phases.
        """
        metrics = self.get_epoch_metrics(metric_names=metric_names, phases=phases)
        path = self.get_epoch_metrics_path()
        self._export_metrics(metrics, path)
        self.print(f"📄 Epoch-wise metrics exported: {path.name}")
        return path
    
    def export_step_metrics(self, metric_names: List[str] | None = None, phases: List[str] | None = None) -> None:
        """
        Export step-wise metrics as a JSON file.

        Args:
            metric_names (list[str] | None): Metric names to export. `None` exports all metrics.
            phases (list[str] | None): Phases to include. `None` includes all phases.
        """
        metrics = self.get_step_metrics(metric_names=metric_names, phases=phases)
        path = self.get_step_metrics_path()
        self._export_metrics(metrics, path)
        self.print(f"📄 Step-wise metrics exported: {path.name}")
        return path

    def _export_metrics(self, metrics: MetricTable, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            self.print(f"Failed to export metrics: {e}\n", level="warn")

    def get_epoch_metrics_path(self) -> Path:
        return self.get_metrics_path("epoch_metrics")

    def get_step_metrics_path(self) -> Path:
        return self.get_metrics_path("step_metrics")

    def get_metrics_path(self, name: str) -> Path:
        return self.metrics_dir / f"{name}.json"

    # ----- Cache Management -----

    def set_cache(self, key: str, value: Any) -> None:
        """Store a value under the given key."""
        self._cache[key] = value

    def get_cache(self, key: str, default: Any = None) -> Any:
        """Return the cached value or default if missing."""
        return self._cache.get(key, default)

    def clear_cache(self) -> None:
        """Remove all cached values."""
        self._cache.clear()
    
    # ----- Logger Management -----
    
    def print_environment_summary(self) -> None:
        """Print a summary of the system and runtime environment for reproducible experiments."""
        import multiprocessing
        import platform
        import psutil

        try:
            import wmi
            w = wmi.WMI()
            cpu_name = w.Win32_Processor()[0].Name
        except Exception:
            cpu_name = platform.uname().processor or platform.processor() or "Unknown"

        tree = {}

        # OS
        tree["OS"] = f"{platform.system()} {platform.release()}"

        # CPU
        tree["CPU"] = cpu_name
        tree["CPU cores"] = multiprocessing.cpu_count()

        # RAM
        ram_total = psutil.virtual_memory().total / 1e9
        tree["RAM"] = f"{ram_total:.2f} GB"

        # Disk
        disk = shutil.disk_usage("/")
        disk_total = disk.total / 1e9
        disk_free = disk.free / 1e9
        tree["Disk"] = f"{disk_free:.2f} / {disk_total:.2f} GB free"

        # GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9

            tree["GPU"] = gpu_name
            tree["VRAM"] = f"{vram_total:.2f} GB"
            tree["CUDA"] = torch.version.cuda
            tree["cuDNN"] = torch.backends.cudnn.version()
        else:
            tree["GPU"] = "Not available"
            tree["VRAM"] = "-"
            tree["CUDA"] = "-"
            tree["cuDNN"] = "-"

        # Python / PyTorch
        tree["Python"] = platform.python_version()
        tree["PyTorch"] = torch.__version__

        # torchvision
        try:
            import torchvision
            tree["torchvision"] = torchvision.__version__
        except Exception:
            pass

        # torchaudio
        try:
            import torchaudio
            tree["torchaudio"] = torchaudio.__version__
        except Exception:
            pass

        self.print_dict_tree(tree, header="🧩 Environment Summary")
    
    def print_gpu_temperature(self) -> None:
        """Print the current GPU temperature if a CUDA-capable GPU is available."""
        if not torch.cuda.is_available():
            self.print("CUDA not available. Skipping GPU temperature check.", level="warn")
            return

        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            temp_str = result.stdout.strip()
            temp = int(temp_str) if temp_str.isdigit() else None

            if temp is not None:
                self.print(f"🌡️  GPU Temperature: {temp} °C")
                if temp > 85:
                    self.print("GPU temperature is high! Consider cooling down.", level="warn")
            else:
                self.print("GPU temperature info unavailable or invalid.", level="warn")

        except FileNotFoundError:
            self.print("'nvidia-smi' not found. Skipping GPU temperature check.", level="warn")
        except subprocess.CalledProcessError as e:
            self.print(f"'nvidia-smi' command failed: {e}", level="warn")
        except Exception as e:
            self.print(f"Failed to get GPU temperature: {e}", level="warn")
    
    def print_config(self) -> None:
        """Print the current training configuration."""
        self.print_dict_tree(self._config, header="⚙️  Configuration")
    
    def print_model_summary(self) -> None:
        """Print a summary of all models, including trainable parameters."""
        tree = {}
        for name, model in self.models.items():
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            status = f"{trainable:,}/{total:,} trainable" if trainable else "frozen"
            tree[name] = status
        
        self.print_dict_tree(tree, header="🧩 Model Summary")
    
    def print_optimization_summary(self) -> None:
        """Print a summary of the optimizer and scheduler."""
        tree = {
            "optimizer": self.optimizer.__class__.__name__ if self.optimizer else "-",
            "scheduler": self.scheduler.__class__.__name__ if self.scheduler else "-",
        }
        self.print_dict_tree(tree, header="🔧 Optimization Summary")
    
    def print_metrics(self, metrics: Dict[str, float], phase: str) -> None:
        """
        Print metrics for a given phase.

        Args:
            metrics (dict[str, float]): Dict mapping metric names to their float values.
            phase (str): Phase name (e.g., 'train', 'val', 'test').
        """
        print_flat_dict_tree(
            data=metrics,
            header=f"📊 {phase.capitalize()}",
            key_width=self.key_width,
            float_fmt=4,
            trailing_newline=True,
            print_fn=self.print,
        )
    
    def print_training_status(self) -> None:
        """Print a summary of the current training state."""

        formatted_metrics = self._get_formatted_epoch_metrics()

        tree = {
            "Completed epochs": self.current_epoch,
            "Best val loss": (
                f"{self.best_val_loss:.4f}  (epoch {self.best_val_epoch})"
                if self.best_val_loss < float('inf') else "-"
            ),
            "No improvement epochs": (
                self.epochs_no_improve
                if self.epochs_no_improve is not None else "-"
            ),
            "Last epoch metrics": formatted_metrics or "-",
        }

        self.print_dict_tree(tree, header="📊 Training Status")
    
    def _get_formatted_epoch_metrics(self) -> Dict[str, str]:
        formatted = {}
        for metric_name, phase_dict in self.epoch_metrics.items():
            parts = []
            for phase, values in phase_dict.items():
                if values:
                    parts.append(f"{phase}={values[-1]:.4f}")
                else:
                    parts.append(f"{phase}=N/A")
            formatted[metric_name] = "  ".join(parts) if parts else "N/A"
        return formatted
    
    def _update_pbar(self, pbar: tqdm, metrics: Dict[str, float]) -> None:
        display_metrics = {
            k: f"{v:.4f}" for k, v in metrics.items()
            if k in (self.metric_names_to_display or [])
        }

        if torch.cuda.is_available():
            used, total, free = self._get_gpu_memory_info()
            display_metrics["GPU"] = f"{used}/{total}"

        pbar.set_postfix(display_metrics)
    
    def print_dict_tree(
        self,
        tree: Dict[str, Any],
        header: str | None = None,
        max_depth: int | None = None,
    ) -> None:
        """
        Pretty-print nested dictionaries in a tree-like format.

        Args:
            tree (dict): Dictionary to print.
            header (str | None): Header shown at root.
            max_depth (int | None): Maximum depth to expand. `None` means unlimited.
        """
        print_dict_tree(
            tree,
            max_depth=max_depth,
            header=header,
            key_width=self.key_width,
            trailing_newline=True,
            print_fn=self.print,
            indent=0,
        )

    def print(self, msg: str | None = None, level: str = "info", indent: int = 0) -> None:
        self.logger.log(msg, level=level, indent=indent)

    def _create_default_logger(self) -> UnifiedLogger:
        return UnifiedLogger(
            f"trainer_logger_{id(self)}",
            log_path=self._get_default_log_path(),
            verbose=True,
            debug_mode=self.debug_mode,
            file_mode="a",
        )

    def _get_default_log_path(self) -> Path:
        return self.run_dir / "log.txt"

    def get_log_path(self) -> Path | None:
        return self.logger.log_path
    
    # ----- Training Preparation -----

    def reset_trainer(self) -> None:
        """
        Completely reset the trainer to a clean initial state.

        This clears any previous setup, training state, metrics, and cached
        resources, returning the trainer to a freshly-initialized state.
        """
        self.clear_setup()
        self.reset_training_state()
        self.clear_all_metrics()
        self.clear_cache()
    
    def prepare_training(self) -> None:
        """
        Prepare the trainer for a new training run.

        This:
            - Displays device and configuration information.
            - Ensures the setup is ready (models, optimizer, scheduler).
            - Loads a checkpoint if resuming.
            - Prints model and optimization summaries.
        """
        self.print_environment_summary()
        self.save_config()
        self.print_config()

        self.ensure_setup()

        if self.resume and self.has_latest_checkpoint():
            self._load_latest_checkpoint()

        self.print_model_summary()
        self.print_optimization_summary()
        self.print_training_status()
    
    def ensure_setup(self) -> None:
        """
        Ensure that the trainer is fully set up.

        Calls `setup()` once if the setup has not yet been completed.
        """
        if not self._is_setup_done:
            self.setup()
            self._is_setup_done = True

    def clear_setup(self) -> None:
        """
        Clear all resources established by `setup()`.

        Removes models, optimizer, and scheduler, and marks the setup
        as incomplete so that `ensure_setup()` will rebuild it.
        """
        self.clear_models()
        self.clear_optimizer()
        self.clear_scheduler()
        self._is_setup_done = False

    def epoch_iterator(self) -> Iterator[tuple[int, int]]:
        """
        Yield (current_epoch, max_epoch) for each training epoch.

        Automatically increments `current_epoch` and stops at `num_epochs`.
        """
        while self.current_epoch < self.num_epochs:
            self.current_epoch += 1
            yield self.current_epoch, self.num_epochs
    
    # ----- Training Phase / Mode Management -----

    def _is_training_phase(self, phase: str) -> bool:
        return phase in self.training_phases

    def _set_training_mode(self, training: bool) -> None:
        for model in self.models.values():
            model.train() if training else model.eval()
        self.on_set_training_mode(training)
    
    # ----- Batch / Loss Management -----
    
    @staticmethod
    def _get_batch_size(batch: Any) -> int:
        if isinstance(batch, dict):
            return len(next(iter(batch.values())))
        elif hasattr(batch, "__len__"):
            return len(batch)
        else:
            raise TypeError("Cannot infer batch size from batch object.")
    
    @staticmethod
    def _extract_valid_loss_value(loss: torch.Tensor) -> float:
        val = loss.item()
        if not math.isfinite(val):
            raise RuntimeError(f"Invalid loss value detected: {val}")
        return float(val)
    
    # ----- Training Initialization / State Management -----
    
    def reset_training_state(self) -> None:
        """Reset all training state variables."""
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_epoch = None
        self.epochs_no_improve = 0
    
    def is_training_completed(self) -> bool:
        """Return True if the training has reached or exceeded the maximum number of epochs."""
        return self.current_epoch >= self.num_epochs
    
    def finalize_train_epoch(self, val_loss: float | None = None) -> None:
        """
        Update early-stopping state and step the scheduler for this training epoch.

        Must be called **immediately after train/val metrics are computed** and
        **before saving checkpoints**, since checkpoint-saving depends on the
        updated internal state.

        Args:
            val_loss (float | None): Validation loss for the epoch, or None if
                validation was not performed.
        """
        self._update_early_stopping_state(val_loss)
        self._step_scheduler(val_loss)
    
    def _update_early_stopping_state(self, val_loss: float | None) -> None:
        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = self.current_epoch
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            self.print(f"No improvement for {self.epochs_no_improve} epoch(s)", level="warn")
    
    def is_best_epoch(self) -> bool:
        """Return True if the current epoch matches the best validation epoch recorded so far."""
        return self.current_epoch == self.best_val_epoch
    
    def should_stop_early(self) -> bool:
        """Return True if early stopping condition has been met."""
        return self.patience is not None and self.epochs_no_improve >= self.patience
    
    # ----- Seed & Reproducibility Management -----

    def _set_seed(self, seed: int) -> None:
        import random
        import numpy as np
        import torch.backends.cudnn as cudnn

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
    
    # ----- GPU management -----

    @staticmethod
    def _get_gpu_memory_info() -> tuple[int, int, int]:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            used = mem.used // (1024 ** 2)
            total = mem.total // (1024 ** 2)
            free = mem.free // (1024 ** 2)

            pynvml.nvmlShutdown()
            return used, total, free

        except Exception:
            pass

        try:
            import subprocess

            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
            ).strip()

            used_str, total_str = [x.strip() for x in output.split(",")]
            used = int(used_str)
            total = int(total_str)
            free = total - used

            return used, total, free

        except Exception:
            pass

        return 0, 0, 0

    @staticmethod
    def clear_cuda_cache() -> None:
        """Force clear GPU memory cache. Use with caution during training."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ----- Others -----
    
    def snapshot_run(self, exclude: List[str] | None = None) -> None:
        """
        Create a lightweight snapshot of this run in ``run_snapshot_dir``,
        excluding heavy artifacts such as checkpoints and caches.

        Args:
            exclude (list[str] | None): Paths to exclude when creating the snapshot.
        """
        if self.run_snapshot_dir is None:
            return

        copy_dir(
            src=self.run_dir,
            dst=self.run_snapshot_dir,
            exclude=exclude,
        )
