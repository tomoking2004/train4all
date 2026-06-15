<div align="center">

# train4all

![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.1-informational)

**Implement three methods. Get a complete training loop.**

</div>

---

train4all is a minimal PyTorch training framework. Subclass `BaseTrainer`, implement `setup()`, `compute_loss()`, and `compute_metrics()` — the framework handles checkpointing, early stopping, metrics, logging, and a live web dashboard automatically.

**Features at a glance**

- **Zero boilerplate** — one subclass, three methods, full training loop
- **Automatic checkpointing** — `latest.pth` and `best.pth` saved after every epoch; periodic saves every N epochs
- **Early stopping** — patience-based with automatic best-checkpoint tracking
- **Live web dashboard** — a self-contained, dependency-free panel: progress gauge, live KPIs, per-step loss graph, per-metric charts, light & dark themes
- **Flexible metrics** — epoch- and step-level recording, JSON export, matplotlib curve plots
- **Snapshot sync** — mirror `run_dir` to any path for cloud-backed storage during long runs
- **Lifecycle hooks** — 9 hook points to inject logic at any stage without subclassing the loop
- **Step cache** — share tensors between `compute_loss` and `compute_metrics` with no extra forward pass

---

## Contents

- [train4all](#train4all)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Constructor Parameters](#constructor-parameters)
  - [API Reference](#api-reference)
    - [Abstract Methods](#abstract-methods)
    - [Training \& Evaluation](#training--evaluation)
    - [Setup Helpers](#setup-helpers)
    - [Model Management](#model-management)
    - [Lifecycle Hooks](#lifecycle-hooks)
    - [Step Cache](#step-cache)
    - [Checkpointing](#checkpointing)
    - [Metrics](#metrics)
    - [Custom Training Loop](#custom-training-loop)
    - [State Inspection](#state-inspection)
    - [Configuration](#configuration)
    - [Snapshot](#snapshot)
    - [GPU Utilities](#gpu-utilities)
  - [Live Dashboard](#live-dashboard)
    - [DashboardConfig Parameters](#dashboardconfig-parameters)
  - [License](#license)

---

## Installation

```bash
pip install git+https://github.com/tomoking2004/train4all.git
```

---

## Quick Start

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from train4all import BaseTrainer


class MyTrainer(BaseTrainer):
    def setup(self):
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256,  64), nn.ReLU(),
        )
        self.head = nn.Linear(64, 10)

        self.set_models({"encoder": self.encoder, "head": self.head})
        self.set_optimizer(
            torch.optim.Adam(self.get_trainable_params(), lr=self.learning_rate)
        )

    def compute_loss(self, batch):
        x, y = batch
        logits = self.head(self.encoder(x))
        self.set_cache("logits", logits.detach())
        return F.cross_entropy(logits, y)

    def compute_metrics(self, batch):
        _, y = batch
        preds = self.get_cache("logits").argmax(dim=1)
        return {"accuracy": (preds == y).float().mean().item()}


def make_loader(n: int, batch_size: int = 64) -> DataLoader:
    x = torch.randn(n, 784)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


trainer = MyTrainer(num_epochs=5, learning_rate=1e-3, run_dir="run", use_dashboard=True)
trainer.train(make_loader(100_000), val_loader=make_loader(20_000))
trainer.test(make_loader(10_000), use_best=True)
```

---

## Constructor Parameters

| Parameter | Default | Description |
| :-- | :-- | :-- |
| `num_epochs` | — | Total training epochs *(required)*. |
| `batch_size` | `None` | Informational; accessible in `setup()` as `self.batch_size`. |
| `learning_rate` | `1e-4` | Scalar or per-group dict; available as `self.learning_rate` in `setup()`. |
| `device` | auto | `"cuda"`, `"cuda:1"`, `"mps"`, or `"cpu"`. Auto-detected when `None` — prefers CUDA, then MPS, then CPU. On a multi-GPU machine, pick a specific GPU with `"cuda:<index>"`. |
| `seed` | `None` | Global random seed for Python, NumPy, and PyTorch. |
| `run_dir` | `"run"` | Output directory for checkpoints, metrics, logs, and plots. |
| `run_snapshot_dir` | `None` | Mirror directory for a lightweight copy of `run_dir` after each epoch. |
| `patience` | `None` | Early-stopping patience in epochs. Disabled when `None`. |
| `resume` | `True` | Resume from `latest.pth` at the start of training. |
| `save_interval` | `None` | Save a periodic checkpoint every N epochs. |
| `training_phases` | `["train"]` | Phase names that trigger gradient updates. |
| `record_step_metrics` | `False` | Record per-step metrics during training phases. |
| `step_metric_names` | `None` | Subset of metric names to record at the step level. `None` records all. |
| `pbar_metric_names` | `None` | Metric names shown in the tqdm postfix. `None` hides all metrics (GPU memory still shown on CUDA). |
| `use_progress_bar` | `True` | Show tqdm progress bars during epoch iteration. |
| `keep_progress_bar` | `False` | Persist progress bars after each epoch completes. |
| `key_width` | `32` | Column width for printed metric and summary tables. |
| `debug_mode` | `False` | Enable debug-level logging. |
| `logger` | `None` | External `UnifiedLogger` instance; a default one is created if `None`. |
| `use_dashboard` | `False` | Enable the live web dashboard. |
| `dashboard_config` | `None` | Dashboard appearance and behaviour (`DashboardConfig`). |

---

## API Reference

### Abstract Methods

Implement all three in your subclass:

```python
def setup(self) -> None:
    # Initialise and register models, optimizer, and scheduler.
    # Called once before training or evaluation begins.
    ...

def compute_loss(self, batch: Any) -> torch.Tensor:
    # Compute and return a scalar loss tensor.
    # The batch is already on the training device.
    ...

def compute_metrics(self, batch: Any) -> dict[str, float]:
    # Return a flat dict of metric name → scalar value.
    # Called immediately after compute_loss; the step cache is populated.
    ...
```

---

### Training & Evaluation

```python
trainer.train(train_loader, val_loader=val_loader)
```

Run the full training loop. Calls `prepare_training()` first, then iterates epochs, runs validation after each train epoch when `val_loader` is provided, and handles early stopping, checkpointing, and dashboard updates automatically.

```python
metrics: dict[str, float] = trainer.test(test_loader, use_best=True)
```

Evaluate on a held-out test set. When `use_best=True`, loads `best.pth` before running — use this for final reporting after `train()` completes.

---

### Setup Helpers

Intended for use inside your `setup()` implementation:

```python
# Register models and move them to the training device.
self.set_models({"encoder": enc, "head": head})    # multiple at once
self.set_model("backbone", backbone)                # one at a time

# Set the optimizer.
optimizer = torch.optim.AdamW(self.get_trainable_params(), lr=self.learning_rate)
self.set_optimizer(optimizer)

# Set a learning-rate scheduler (optional).
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
self.set_scheduler(scheduler)

# Collect all trainable parameters (deduplicated) from registered models.
params = self.get_trainable_params()

# Restrict to specific models, or exclude some.
params = self.get_trainable_params(targets="head", exclude_targets="encoder")
```

---

### Model Management

```python
self.freeze("encoder")             # disable gradients
self.unfreeze("encoder")           # re-enable gradients
self.reset_parameters("head")      # re-initialise weights in place

# Targets accept a name string, an nn.Module instance, or a list of either.
self.freeze(["encoder", self.head])
```

---

### Lifecycle Hooks

Override any of these no-ops to inject logic at any stage of the loop:

```python
def on_set_training_mode(self, training: bool) -> None: ...
def on_training_start(self) -> None: ...
def on_training_end(self) -> None: ...
def on_train_epoch_start(self, epoch: int) -> None: ...
def on_train_epoch_end(self, epoch: int) -> None: ...
def on_epoch_start(self, epoch: int | None, loader: DataLoader, phase: str) -> None: ...
def on_epoch_end(self, epoch: int | None, loader: DataLoader, metrics: dict[str, float], phase: str) -> None: ...
def on_step_start(self, step: int | None, batch: Any, phase: str) -> None: ...
def on_step_end(self, step: int | None, batch: Any, metrics: dict[str, float], phase: str) -> None: ...
```

Two timing guarantees worth knowing: the step cache is cleared **before** `on_epoch_start` fires; epoch metrics for the completed phase are already recorded **before** `on_epoch_end` fires, so `get_epoch_metrics()` reflects the current epoch inside that hook.

---

### Step Cache

`set_cache` and `get_cache` share tensors between `compute_loss` and `compute_metrics` within a single step, eliminating redundant forward passes:

```python
def compute_loss(self, batch):
    logits = self.model(batch["x"])
    self.set_cache("logits", logits.detach())    # store
    return F.cross_entropy(logits, batch["y"])

def compute_metrics(self, batch):
    preds = self.get_cache("logits").argmax(1)   # retrieve
    return {"acc": (preds == batch["y"]).float().mean().item()}
```

The cache is cleared automatically at the start of each epoch and phase, before `on_epoch_start` fires. Use `get_cache(key, default=...)` to supply a fallback when the key may be absent.

---

### Checkpointing

```python
# Saving
trainer.save_checkpoint("run/my_checkpoint.pth")         # full checkpoint at any path
trainer.save_weights("run/weights_only.pth")              # model weights only
trainer.backup_checkpoint("run/checkpoints/latest.pth")   # copy with .bak suffix

# Loading
trainer.load_latest_checkpoint()                          # load checkpoints/latest.pth
trainer.load_best_checkpoint()                            # load checkpoints/best.pth
trainer.load_checkpoint("run/my_checkpoint.pth")          # full checkpoint from any path
trainer.load_weights("run/weights_only.pth")              # weights only, skip optimizer state

# Rename state-dict keys on the fly (useful when the model architecture changed).
trainer.load_checkpoint("old.pth", key_map={"old_prefix.": "new_prefix."})

# Querying
trainer.has_latest_checkpoint()
trainer.has_best_checkpoint()
path = trainer.get_latest_checkpoint_path()
path = trainer.get_best_checkpoint_path()
path = trainer.get_checkpoint_path("epoch_10")            # checkpoints/epoch_10.pth

# Extras
trainer.exclude_from_checkpoint("encoder")                # omit a model from future saves
trainer.update_checkpoint_extras({"notes": "baseline"})   # embed custom data in the file
```

---

### Metrics

```python
# Epoch-level
table = trainer.get_epoch_metrics()                       # dict[metric, dict[phase, list[float]]]
table = trainer.get_epoch_metrics(metric_names=["loss"], phases=["val"])
path  = trainer.export_epoch_metrics()                    # writes metrics/epoch_metrics.json
        trainer.save_epoch_metric_plots()                 # writes plots/*.png via matplotlib

# Step-level (requires record_step_metrics=True)
table = trainer.get_step_metrics()
table = trainer.get_step_metrics(phases=["train"])
path  = trainer.export_step_metrics()                     # writes metrics/step_metrics.json
        trainer.save_step_metric_plots()

# Resetting
trainer.clear_metrics()                                   # reset both epoch and step tables
```

---

### Custom Training Loop

When you need more control than `train()` provides, build your own loop using the building blocks:

```python
trainer.prepare_training()  # print env, save config, run setup(), optional resume

for epoch, max_epoch in trainer.epoch_iterator():
    train_metrics = trainer.execute_epoch(train_loader, phase="train")
    val_metrics   = trainer.execute_epoch(val_loader,   phase="val")

    trainer.finalize_train_epoch(val_loss=val_metrics.get("loss"))
    trainer.save_artifacts()   # checkpoints + metric plots + JSON export

    if trainer.should_stop_early():
        break
```

For step-level control:

```python
metrics = trainer.execute_epoch(loader, phase="train", print_metrics=True)
metrics = trainer.execute_step(batch,  phase="val",   print_metrics=True)
```

---

### State Inspection

```python
trainer.is_training_completed()      # True when current_epoch >= num_epochs
trainer.is_best_epoch()              # True if this epoch set the best val loss
trainer.should_stop_early()          # True if patience is exhausted

trainer.print_status()               # epoch counter, best loss, recent metrics
trainer.print_model_summary()        # model names and parameter counts
trainer.print_env_summary()          # OS, CPU, RAM, GPU, Python, PyTorch versions
trainer.print_optimization_summary() # optimizer and scheduler class names
```

---

### Configuration

```python
# Merge custom entries into the trainer config (persisted in config.json).
trainer.update_config({"experiment": "baseline", "tag": "v1"})
trainer.save_config()

path = trainer.get_config_path()     # run/config.json
```

---

### Snapshot

Copy a lightweight snapshot of `run_dir` into a mirror location at any time — useful for syncing to a cloud-backed folder during long runs:

```python
trainer = MyTrainer(
    num_epochs=50,
    run_dir="run",
    run_snapshot_dir="/mnt/gdrive/experiments/run",
)

# Or call manually:
trainer.snapshot_run(exclude=["checkpoints"])
```

---

### GPU Utilities

```python
trainer.print_gpu_temperature()  # reads temperature via nvidia-smi; warns above 85 °C
trainer.clear_cuda_cache()       # gc.collect() + torch.cuda.empty_cache()
```

---

## Live Dashboard

Pass `use_dashboard=True` for a live, dependency-free dashboard in your browser: an overall-progress gauge, a KPI grid (loss, best validation, throughput, ETA, learning rate, GPU memory), a live per-step loss graph, and a per-metric SVG chart for each metric. It follows your light/dark theme and embeds its data inline so it stays viewable offline.

> **Remote training** works headless and reports the remote GPU. From an editor (VS Code Remote-SSH, Dev Containers, WSL) it opens in your **local** browser automatically; over plain SSH, set `open_on_start=False` and forward the printed port (`ssh -L 8080:127.0.0.1:<printed-port> …`).

```python
from train4all import BaseTrainer, DashboardConfig

trainer = MyTrainer(
    num_epochs=50,
    use_dashboard=True,
    dashboard_config=DashboardConfig(
        poll_interval_ms=500,
        open_on_start=True,
    ),
)
```

### DashboardConfig Parameters

| Parameter | Default | Description |
| :-- | :-- | :-- |
| `enabled` | `True` | Master switch; `False` disables the dashboard entirely. |
| `filename` | `"dashboard.html"` | HTML shell filename written inside `run_dir`. |
| `data_filename` | `"dashboard_data.json"` | JSON data file polled by the browser. |
| `poll_interval_ms` | `500` | Browser polling interval in milliseconds. |
| `open_on_start` | `True` | Open in the system's default browser when training begins. |
| `stale_multiplier` | `12` | Mark training stale after `poll_interval_ms × stale_multiplier` ms without a JSON update. |
| `use_server` | `True` | Start a local HTTP server (required for Chrome/Edge on `file://` pages). |

---

## License

[MIT](LICENSE) © 2026 tomoking2004
