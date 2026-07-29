<div align="center">

# train4all

![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.10.0-informational)

**Implement three methods. Get a complete training loop.**

<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="assets/dashboard-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/dashboard-light.png">
  <img alt="train4all live training dashboard" src="assets/dashboard-dark.png" width="100%">
</picture>

</div>

---

train4all is a minimal PyTorch training framework. Subclass `BaseTrainer`, implement `setup()`, `compute_loss()`, and `compute_metrics()` — the framework handles checkpointing, early stopping, metrics, logging, and a live web dashboard automatically.

**Features at a glance**

- **Zero boilerplate** — one subclass, three methods, full training loop
- **Composable epochs** — an epoch is whatever sequence of [`Phase`](#phases) objects you pass to `train()`; drop in a phase that measures expensive metrics on a subset of the training data every N epochs, and everything downstream (curves, dashboard, early stopping) follows
- **Mixed precision** — automatic bf16 AMP on CUDA by default for lower VRAM and faster steps; opt into `"fp16"` for older cards or disable with `amp=False`. TF32 + cuDNN autotuner switch on automatically for unseeded runs (`tf32`)
- **Scaling on small GPUs** — gradient accumulation (`accumulation_steps`) simulates a larger effective batch at no extra memory cost, and per-model `torch.compile` (`compile=True`) unlocks graph-level speedups
- **Automatic checkpointing** — `latest.pth` after every epoch and `best.pth` whenever the monitored metric improves; periodic saves every N epochs, plus a standalone `Checkpoint` reader to inspect any file with no model or subclass
- **Early stopping** — patience-based on any `monitor` metric (`min`/`max` mode), with automatic best-checkpoint tracking
- **Live web dashboard** — a self-contained, dependency-free panel: progress gauge, live KPIs, per-step loss graph, per-metric charts, light & dark themes
- **Flexible metrics** — epoch- and step-level recording, JSON export, matplotlib curve plots
- **Snapshot sync** — mirror `run_dir` to any path for cloud-backed storage during long runs
- **Lifecycle hooks** — 14 hook points to inject logic at any stage without subclassing the loop
- **Step cache** — share tensors between `compute_loss` and `compute_metrics` with no extra forward pass

---

## Contents

- [train4all](#train4all)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Constructor Parameters](#constructor-parameters)
    - [Class Constants](#class-constants)
  - [API Reference](#api-reference)
    - [Abstract Methods](#abstract-methods)
      - [Optional: test-only metrics](#optional-test-only-metrics)
    - [Phases](#phases)
    - [Training \& Evaluation](#training--evaluation)
    - [Setup Helpers](#setup-helpers)
    - [Model Management](#model-management)
    - [Lifecycle Hooks](#lifecycle-hooks)
    - [Step Cache](#step-cache)
    - [Checkpointing](#checkpointing)
      - [Persisting custom state: `extras` vs. hooks](#persisting-custom-state-extras-vs-hooks)
      - [Inspecting a checkpoint](#inspecting-a-checkpoint)
    - [Metrics](#metrics)
      - [Weighted averaging](#weighted-averaging)
    - [Custom Training Loop](#custom-training-loop)
    - [Resetting](#resetting)
    - [State Inspection](#state-inspection)
    - [Configuration](#configuration)
    - [Snapshot](#snapshot)
    - [GPU Utilities](#gpu-utilities)
  - [Live Dashboard](#live-dashboard)
    - [DashboardConfig Parameters](#dashboardconfig-parameters)
  - [Utilities](#utilities)
  - [Development](#development)
  - [License](#license)

---

## Installation

```bash
pip install git+https://github.com/tomoking2004/train4all.git
```

```python
import train4all

train4all.__version__    # the installed version
```

---

## Quick Start

The example trains on MNIST, so it needs `torchvision` as well — `pip install torchvision`. train4all itself never imports it, and so does not depend on it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from train4all import BaseTrainer, Phase

BATCH_SIZE = 256


class MyTrainer(BaseTrainer):
    def setup(self):
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256,  64), nn.ReLU(),
        )
        self.head = nn.Linear(64, 10)

        self.set_models({"encoder": self.encoder, "head": self.head})
        self.set_optimizer(torch.optim.Adam)

    def compute_loss(self, batch):
        x, y = batch
        logits = self.head(self.encoder(x))
        self.set_cache("logits", logits.detach())
        return F.cross_entropy(logits, y)

    def compute_metrics(self, batch):
        _, y = batch
        preds = self.get_cache("logits").argmax(dim=1)
        return {"accuracy": (preds == y).float().mean().item()}


def make_loader(dataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def mnist(train: bool):
    return datasets.MNIST("data", train=train, download=True, transform=transforms.ToTensor())


train_set, val_set = random_split(
    mnist(train=True), [55_000, 5_000], generator=torch.Generator().manual_seed(0)
)

trainer = MyTrainer(
    num_epochs=5, batch_size=BATCH_SIZE, learning_rate=1e-3, seed=0,
    run_dir="run", use_dashboard=True,
)
trainer.train(
    Phase("train", make_loader(train_set, shuffle=True), training=True),
    Phase("val", make_loader(val_set)),
)
trainer.test(make_loader(mnist(train=False)), use_best=True)
```

Running it opens the [live dashboard](#live-dashboard) and streams a clean console log — a reproducibility banner (environment, resolved config, model, optimization, status), the shape of an epoch, then a per-phase metric table and automatic checkpoint saves on every epoch. This is that run, verbatim:

<div align="center">
  <img alt="train4all console output — reproducibility banner and the first epoch" src="assets/logs.png" width="62%">
</div>

---

## Constructor Parameters

Every parameter is optional, and all except `num_epochs` are **keyword-only**, so order never matters and the table can be reordered freely. The saved config records **only the reproducibility-relevant arguments you actually customized** — anything left at its default is omitted — and unpacks straight back in: [`MyTrainer.from_config("run")`](#configuration) reconstructs the trainer from the saved `config.json`. The **resolved `device`** is also pinned (e.g. `"cuda:0"`), so a reload targets the same hardware for exact reproduction and fails loudly on a host that lacks it (pass `device=` to retarget); purely operational args like `run_dir` are omitted and fall back to their defaults.

| Parameter | Default | Description |
| :-- | :-- | :-- |
| `num_epochs` | `None` | Total training epochs. Required by `train()`; leave unset to only evaluate (`test()`) or inspect checkpoints. |
| `batch_size` | `None` | Informational; accessible in `setup()` as `self.batch_size`. |
| `learning_rate` | `None` | Scalar, or a dict keyed by model name for [per-model param groups](#setup-helpers); available as `self.learning_rate` in `setup()`. Leave unset for learning-rate-free optimizers (e.g. Prodigy, D-Adaptation, Schedule-Free); **pass it explicitly for optimizers that need one** (e.g. Adam, SGD), since `self.learning_rate` is `None` until you do. |
| `max_grad_norm` | `None` | Clip the global gradient norm to this value before each optimizer step. Disabled when `None`. Correct under fp16 AMP — gradients are unscaled first. |
| `accumulation_steps` | `1` | Accumulate gradients over this many steps before each optimizer update, simulating a larger effective batch with no extra memory. The accumulation is normalized as `Σ wᵢ∇Lᵢ / Σ wᵢ` with weights from `get_batch_weight`; this is the true mean over the effective batch only when the weight matches the loss's denominator (override to the token count for per-token losses — the default sample count fits a per-sample mean). For known-length loaders the last partial cycle of each epoch is always flushed. |
| `amp` | `None` | Automatic mixed precision. `None` auto-enables bf16 on CUDA (no-op on CPU/MPS); `True`/`"bf16"`/`"fp16"` requests it explicitly (warns if the device is not CUDA); `False` forces full precision. |
| `tf32` | `None` | Allow TF32 fp32 matmuls/convolutions and the cuDNN autotuner on CUDA (Ampere+). `None` auto-enables it only when `seed` is unset (speed when not reproducing); `True`/`False` force it. CUDA-only; complementary to `amp`. |
| `patience` | `None` | Early-stopping patience in epochs. Disabled when `None`. |
| `monitor` | `"loss"` | Metric driving best-checkpoint selection and early stopping. |
| `monitor_mode` | `"min"` | `"min"` (lower is better, e.g. loss) or `"max"` (higher is better, e.g. accuracy). |
| `monitor_phase` | `"val"` | The [phase](#phases) `monitor` is read from. Just a name — any phase in the schedule can drive selection and early stopping. |
| `device` | auto | `"cuda"`, `"cuda:1"`, `"mps"`, or `"cpu"`. Auto-detected when `None` — prefers CUDA, then MPS, then CPU. On a multi-GPU machine, pick a specific GPU with `"cuda:<index>"`. |
| `seed` | `None` | Global random seed for Python, NumPy, and PyTorch. |
| `run_dir` | `"run"` | Output directory for checkpoints, metrics, logs, and plots. |
| `run_snapshot_dir` | `None` | Mirror directory for `run_dir`. When set, `train()` [snapshots](#snapshot) the run there after every epoch. Must lie outside `run_dir`. |
| `run_snapshot_exclude` | `None` | Top-level entries left out of every [snapshot](#snapshot) — e.g. `["checkpoints"]` to mirror the metrics and plots alone. `None` excludes nothing. What a mirror leaves behind belongs to the run, not to one call, so the unattended per-epoch mirror follows it too; `snapshot_run(exclude=...)` overrides it for a single call. |
| `resume` | `True` | Resume from `latest.pth` at the start of training. When `False`, `prepare_training()` first clears the run's previous artifacts (`checkpoints/`, `metrics/`, `plots/`, and dashboard files) and starts a fresh log, so a fresh run never inherits stale files — `config.json` and any user files in `run_dir` are kept, and evaluation-only flows (calling `test()` without training) are unaffected. |
| `save_interval` | `None` | Save a periodic checkpoint every N epochs. |
| `record_step_metrics` | `False` | Record per-step metrics. The master switch; each phase decides whether it takes part via `Phase.record_steps`, which defaults to the training phases. |
| `step_metric_names` | `None` | Subset of metric names to record at the step level. `None` records all. |
| `pbar_metric_names` | `None` | Metric names shown in the tqdm postfix. `None` hides all metrics (GPU memory still shown on CUDA). |
| `use_progress_bar` | `True` | Show tqdm progress bars during epoch iteration. |
| `debug_mode` | `False` | Enable debug-level logging. |
| `logger` | `None` | Any object satisfying the `TrainerLogger` protocol (a `log()` method); a default `UnifiedLogger` is created if `None`. |
| `use_dashboard` | `False` | Enable the live web dashboard. |
| `dashboard_config` | `None` | Dashboard appearance and behaviour (`DashboardConfig`). |

### Class Constants

Settings that belong to a trainer *type* rather than to a run — the run's output layout and the console/dashboard tuning — are **class constants**, not constructor arguments. They are set once per trainer, so override them in your subclass:

```python
class MyTrainer(BaseTrainer):
    _CHECKPOINTS_DIRNAME = "ckpt"   # run/ckpt/ instead of run/checkpoints/
    _KEY_WIDTH = 40
```

| Constant | Default | Description |
| :-- | :-- | :-- |
| `_CHECKPOINTS_DIRNAME` | `"checkpoints"` | Checkpoint subdirectory of `run_dir`. |
| `_METRICS_DIRNAME` | `"metrics"` | Metrics subdirectory of `run_dir`. |
| `_PLOTS_DIRNAME` | `"plots"` | Plots subdirectory of `run_dir`. |
| `_LOG_FILENAME` | `"log.txt"` | Console log written inside `run_dir`. |
| `_CONFIG_FILENAME` | `"config.json"` | The file [`from_config`](#configuration) reads back. |
| `_CHECKPOINT_LATEST` | `"latest"` | Stem of the every-epoch checkpoint (`latest.pth`). |
| `_CHECKPOINT_BEST` | `"best"` | Stem of the best-epoch checkpoint (`best.pth`). |
| `_METRICS_EPOCH` | `"epoch_metrics"` | Stem of the epoch-metrics JSON export. |
| `_METRICS_STEP` | `"step_metrics"` | Stem of the step-metrics JSON export. |
| `_TEST_PHASE` | `"test"` | Name of the [phase](#phases) the `test(loader)` shorthand builds. Pass `test()` a `Phase` to name it anything else. |
| `_KEY_WIDTH` | `32` | Column width for printed metric and summary tables. |
| `_KEEP_PROGRESS_BAR` | `False` | Keep tqdm bars on screen after each epoch completes. |
| `_GPU_TEMP_WARN_C` | `85` | `print_gpu_temperature()` warns above this, in °C. |
| `_GPU_MEM_TTL_S` | `2.0` | Seconds an `nvidia-smi` memory reading stays cached. |
| `_DASH_THROTTLE_S` | `0.5` | Minimum seconds between dashboard step writes. |
| `_DASH_EXTRA_WAIT_S` | `0.5` | Extra wait after the dashboard is finalized. |

---

## API Reference

### Abstract Methods

Implement all three in your subclass:

```python
def setup(self) -> None:
    # Initialize and register models, optimizer, and scheduler.
    # Called once before training or evaluation begins.
    ...

def compute_loss(self, batch: Any) -> torch.Tensor:
    # Compute and return a scalar loss tensor.
    # The batch is already on the training device.
    ...

def compute_metrics(self, batch: Any) -> dict[str, float]:
    # Return a flat dict of metric name → scalar value.
    # Called immediately after compute_loss; the step cache is populated.
    # The default metric function for every phase that doesn't bring its own.
    ...
```

#### Optional: test-only metrics

The final evaluation runs once, so it can afford heavier, report-only metrics that would be wasteful every epoch. `compute_test_metrics` is the metric function the `test(loader)` shorthand gives the phase it builds; the default delegates to `compute_metrics`, so test mirrors validation until you override it:

```python
def compute_test_metrics(self, batch: Any) -> dict[str, float]:
    metrics = self.compute_metrics(batch)        # reuse the shared metrics
    metrics["auc"] = roc_auc_score(...)          # plus report-only extras
    return metrics
```

Nothing routes here by phase *name*, and no entry point injects it — this is simply a default a [`Phase`](#phases) can be given, so any phase can carry it: `Phase("audit", audit_loader, metric_fn=self.compute_test_metrics)`.

---

### Phases

An epoch is a sequence of **phases**, and `train()` takes that sequence directly — the loop has no built-in notion of "train" and "val" beyond the names you choose. A `Phase` is the one place a pass over data is described, and every entry point that runs one speaks it: [`train()`](#training--evaluation), [`test()`](#training--evaluation), and the [`execute_phase()` / `execute_step()`](#custom-training-loop) building blocks.

| Field | Default | Description |
| :-- | :-- | :-- |
| `name` | — | Keys the metric tables, the plots, the dashboard legend, and `monitor_phase`. Unique within a run. |
| `loader` | — | The `DataLoader` iterated for this phase. |
| `training` | `False` | Run with gradients and step the optimizer. Most phases only measure, so evaluation is the default. |
| `metric_fn` | `None` | This phase's per-batch metric function. `None` uses the trainer's `compute_metrics`. Named for what it holds — a *function*, not the metric values `metrics` means everywhere else. |
| `every` | `1` | Run only on epochs divisible by this, so an expensive measurement need not be paid every epoch. |
| `record_steps` | `None` | Take part in `record_step_metrics`. `None` follows `training`. |

The **name** is load-bearing: a phase name *is* a metric series, and nothing else keys the tables, the plots, or the exports. `train()` rejects a schedule with duplicate names outright; across separate calls the trainer cannot tell a collision from a curriculum deliberately continuing one series on a new loader, so it warns instead — but it never lets a name change hands in silence.

A `Phase` is frozen, and two derived accessors answer the questions its raw fields only imply. The type of a metric function is exported as `MetricFn`, for annotating your own:

```python
from train4all import MetricFn, Phase     # MetricFn = Callable[[Any], dict[str, float]]

phase.records_steps      # bool  — record_steps, resolved against training
phase.runs_at(epoch)     # bool  — whether the phase runs at this 1-based epoch
```

The canonical run is two phases:

```python
trainer.train(
    Phase("train", train_loader, training=True),
    Phase("val", val_loader),
)
```

Anything else is the same expression with more phases. To keep the training pass cheap, compute only the loss there and measure the expensive metrics periodically on a subset of the same data:

```python
trainer.train(
    Phase("train", train_loader, training=True, metric_fn=lambda _: {}),
    Phase("train_eval", train_subset_loader, every=5),
    Phase("val", val_loader),
)
```

`metric_fn=lambda _: {}` suppresses only the metric *function* — `loss` is always recorded — so the training pass reports loss alone while `train_eval` reports the full metric set on a slice of the training data, every fifth epoch. The three phases then plot as three curves, each with its own ink, in every chart.

Best-checkpoint selection and early stopping read `monitor` from the phase named by `monitor_phase` (`"val"` by default), taking only the value produced **this** epoch — so a monitored phase that sits out an epoch (`every > 1`) yields no value rather than a stale one. Point `monitor_phase` at any phase you like:

```python
MyTrainer(num_epochs=40, patience=5, monitor="accuracy", monitor_mode="max", monitor_phase="val")
```

---

### Training & Evaluation

```python
trainer.train(
    Phase("train", train_loader, training=True),
    Phase("val", val_loader),
)
```

Run the full training loop. Calls `prepare_training()` first, then iterates epochs, running the given [phases](#phases) in order within each one, and handles early stopping, checkpointing, and dashboard updates automatically.

```python
metrics: dict[str, float] = trainer.test(test_loader, use_best=True)
```

Evaluate once, outside the training loop. When `use_best=True`, loads the best **weights** from `best.pth` before running — use this for final reporting after `train()` completes. Only the weights are loaded, so evaluation never rewinds the epoch counter or truncates the recorded metric history to the best epoch (call `load_best_checkpoint()` for that deliberate full rewind).

The final evaluation is one ordinary [phase](#phases), so `test()` speaks the same vocabulary `train()` does. A `DataLoader` is shorthand for the canonical test phase — these two lines are the same call:

```python
trainer.test(test_loader)
trainer.test(Phase("test", test_loader, metric_fn=trainer.compute_test_metrics))
```

Pass a `Phase` to say anything the shorthand cannot — above all a **name**, since the name is what the metrics, the plots, and the exports are filed under. Two test sets need two names, or their curves silently concatenate under one:

```python
trainer.test(Phase("test_id",  id_loader,  metric_fn=trainer.compute_test_metrics))
trainer.test(Phase("test_ood", ood_loader, metric_fn=trainer.compute_test_metrics))
```

A phase you pass means exactly what it means everywhere else — nothing is injected into it, so `metric_fn=None` is the trainer's `compute_metrics`, as always. [`compute_test_metrics`](#optional-test-only-metrics) is what the *shorthand* reaches for, not a rule `test()` applies.

---

### Setup Helpers

Intended for use inside your `setup()` implementation:

```python
# Register models and move them to the training device.
self.set_models({"encoder": enc, "head": head})    # multiple at once
self.set_model("backbone", backbone)                # one at a time

# Optionally compile a model in place for graph-level speedups (PyTorch 2.0+).
# The registered module is compiled, so your compute_loss runs the optimized
# graph and checkpoints keep their original keys.
self.set_model("decoder", decoder, compile=True)

# Set the optimizer. Given the class, the trainer supplies what it already knows:
# the trainable parameters, and `learning_rate` as `lr` (dropped when it is None,
# so learning-rate-free optimizers just work).
self.set_optimizer(torch.optim.AdamW)

# Restrict it to some models, or pass further hyperparameters — same call.
self.set_optimizer(torch.optim.AdamW, targets="head", weight_decay=0.01)

# An instance is stored untouched: the escape hatch for hand-built param groups.
self.set_optimizer(torch.optim.AdamW([
    {"params": self.encoder.parameters(), "lr": 1e-4},
    {"params": self.head.parameters(),    "lr": 1e-3},
]))

# Set a learning-rate scheduler (optional). The class form gets the registered
# optimizer, so `setup()` needs no local variable to hand it on.
self.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=self.num_epochs)

# Collect all trainable parameters (deduplicated) from registered models.
params = self.get_trainable_params()

# Restrict to specific models, or exclude some.
params = self.get_trainable_params(targets="head", exclude_targets="encoder")
```

`set_optimizer` reads the parameters at the moment you call it, so [`freeze()`](#model-management) belongs above it. A `learning_rate` dict keyed by model name is expanded into one param group per model, which is the one case where `targets` and `exclude_targets` are refused — the keys already name the models.

---

### Model Management

```python
self.freeze("encoder")             # disable gradients
self.unfreeze("encoder")           # re-enable gradients
self.reset_parameters("head")      # re-initialize weights in place

# Targets accept a name string, an nn.Module instance, or a list of either.
self.freeze(["encoder", self.head])
```

---

### Lifecycle Hooks

Override any of these no-ops to inject logic at any stage of the loop:

```python
# Training run
def on_training_start(self) -> None: ...
def on_training_end(self) -> None: ...
def on_exception(self, exc: BaseException) -> None: ...                # loop aborted; re-raised afterwards

# Epoch — the whole epoch, once per epoch
def on_train_epoch_start(self, epoch: int) -> None: ...
def on_train_epoch_end(self, epoch: int) -> None: ...

# Phase — once per phase within the epoch
def on_phase_start(self, epoch: int | None, phase: Phase) -> None: ...
def on_phase_end(self, epoch: int | None, phase: Phase, metrics: dict[str, float]) -> None: ...

# Step
def on_step_start(self, step: int | None, batch: Any, phase: Phase) -> None: ...
def on_step_end(self, step: int | None, batch: Any, metrics: dict[str, float], phase: Phase) -> None: ...

# Optimization
def on_set_training_mode(self, training: bool) -> None: ...
def on_after_backward(self) -> None: ...                              # before unscale/clip (fp16 grads still scaled)
def on_before_optimizer_step(self) -> None: ...                       # after unscale/clip, before optimizer.step()

# Checkpoint (full checkpoints only, not weights-only)
def on_save_checkpoint(self, checkpoint: Checkpoint) -> None: ...      # attach custom state (checkpoint["ema"] = ...)
def on_load_checkpoint(self, checkpoint: Checkpoint) -> None: ...      # read it back after restore
```

The phase and step hooks receive the [`Phase`](#phases) itself, not just its name, so a hook can branch on what the pass actually *is* (`phase.training`, `phase.loader`) rather than on a string.

A few timing guarantees worth knowing:

- The step cache is cleared **before** `on_phase_start` fires.
- Epoch metrics for the completed phase are already recorded **before** `on_phase_end` fires, so `get_epoch_metrics()` reflects the current epoch inside that hook.
- `on_exception` fires for any error — including `KeyboardInterrupt` (Ctrl-C) — and the exception is **re-raised** afterwards. No checkpoint is auto-saved, since a mid-epoch save would persist an incomplete state.
- `on_save_checkpoint` / `on_load_checkpoint` fire only for **full** checkpoints; weights-only saves/loads stay pure (models + extras). They pair with `update_checkpoint_extras()` for round-tripping custom state across a resume.

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

The cache is cleared automatically at the start of each phase, before `on_phase_start` fires — `clear_cache()` empties it by hand, should you ever need to. Use `get_cache(key, default=...)` to supply a fallback when the key may be absent.

---

### Checkpointing

```python
# Saving
trainer.save_checkpoints()                                # latest + best + periodic, as the loop does
trainer.save_checkpoint("run/my_checkpoint.pth")          # full checkpoint at any path
trainer.save_weights("run/weights_only.pth")              # model weights only
trainer.backup_checkpoint("run/checkpoints/latest.pth")   # copy with .bak suffix

# Loading
trainer.load_latest_checkpoint()                          # load checkpoints/latest.pth
trainer.load_best_checkpoint()                            # full restore — rewinds to the best epoch
trainer.load_best_weights()                               # best weights only — no rewind (test uses this)
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
extras = trainer.get_checkpoint_extras()                  # read it back (restored on load)
```

#### Persisting custom state: `extras` vs. hooks

Two mechanisms embed your own data in a checkpoint. They are complementary — pick by whether the data is static metadata or dynamic state:

| | `update_checkpoint_extras()` | `on_save_checkpoint()` / `on_load_checkpoint()` |
| :-- | :-- | :-- |
| Nature | **Declarative** — set the value once, when you know it | **Imperative** — runs on every save, capturing the current value |
| Scope | Rides along with **full *and* weights-only** saves | **Full checkpoints only** |
| Round-trip | Automatic (restored into `get_checkpoint_extras()`) | You write the paired save/load logic yourself |
| Best for | Static metadata: git SHA, class names, normalization constants, dataset version | Dynamic state needing custom serialization: EMA weights, RNG state, replay buffers |

```python
def setup(self):
    ...
    self.update_checkpoint_extras({"class_names": CLASSES})   # static — set once

def on_save_checkpoint(self, checkpoint):
    checkpoint["ema"] = self.ema.state_dict()                 # dynamic — captured each save

def on_load_checkpoint(self, checkpoint):
    if "ema" in checkpoint:
        self.ema.load_state_dict(checkpoint["ema"])
```

The hook receives a [`Checkpoint`](#inspecting-a-checkpoint): index it like a dict (`checkpoint["ema"]`, `"ema" in checkpoint`) or reach for its typed accessors.

#### Inspecting a checkpoint

`Checkpoint` reads a saved file and exposes its contents — **no model, no subclass, no abstract methods**. It is also the single source of truth for the on-disk format, so a trainer and an inspector never disagree about the schema.

```python
from train4all import Checkpoint

ckpt = Checkpoint.load("run/checkpoints/best.pth")   # map_location="cpu" by default
ckpt.print_summary()                  # tree: version, models + param counts, components, training state, metrics
ckpt.summary()                        # the same overview, as a plain dict

ckpt.version                          # on-disk format version (Checkpoint.VERSION is what a save stamps)
ckpt.models["encoder"]                # a raw state dict — no architecture required
ckpt.model_summary()                  # {name: {"parameters": int, "tensors": int}}
ckpt.training_state["best_epoch"]     # legacy key names normalized automatically
ckpt.extras                           # custom metadata embedded via update_checkpoint_extras()
ckpt.metrics                          # recorded {"epoch_metrics": ..., "step_metrics": ...}
ckpt.metric_names()                   # sorted union of metric names across both tables
ckpt.optimizer_state                  # None for a weights-only checkpoint
ckpt.scheduler_state                  # likewise
ckpt.scaler_state                     # likewise
ckpt.raw                              # the underlying dict, for anything not surfaced above
```

---

### Metrics

```python
# Epoch-level
table = trainer.get_epoch_metrics()                       # dict[metric, dict[phase, list[float]]]
table = trainer.get_epoch_metrics(metric_names=["loss"], phase_names=["val"])
path  = trainer.export_epoch_metrics()                    # writes metrics/epoch_metrics.json
        trainer.save_epoch_metric_plots()                 # writes plots/*.png via matplotlib

# Step-level (requires record_step_metrics=True)
table = trainer.get_step_metrics()
table = trainer.get_step_metrics(phase_names=["train"])
path  = trainer.export_step_metrics()                     # writes metrics/step_metrics.json
        trainer.save_step_metric_plots()

# Resetting
trainer.clear_metrics()                                   # reset both epoch and step tables

# Output paths
trainer.get_epoch_metrics_path()                          # run/metrics/epoch_metrics.json
trainer.get_step_metrics_path()                           # run/metrics/step_metrics.json
trainer.get_metrics_path("custom")                        # run/metrics/custom.json
trainer.get_metric_plot_path("loss", phase_name="train", prefix="step")
                                                          # run/plots/step_loss_train.png
```

#### Weighted averaging

Epoch metrics are sample-weighted averages — `Σ(metric × weight) / Σweight` across the steps in an epoch — so uneven final batches are weighted correctly. Each batch's weight defaults to its sample count; override `get_batch_weight` to weight by the loss's denominator instead, e.g. the supervised-token count for a language/vision-language model whose loss is a mean over `labels != -100`. The same weight also normalizes the accumulated gradient when `accumulation_steps > 1`, so loss and gradient stay consistently weighted:

```python
def get_batch_weight(self, batch: Any) -> int:
    # HF LM/VLM loss is a mean over labels != -100; weight must match.
    return int((batch["labels"] != -100).sum())
```

---

### Custom Training Loop

When you need more control than `train()` provides, build your own loop using the building blocks:

```python
trainer.prepare_training()  # print env, save config, run setup(), optional resume

train = Phase("train", train_loader, training=True)
val   = Phase("val", val_loader)

for epoch, max_epoch in trainer.epoch_iterator():
    train_metrics = trainer.execute_phase(train, epoch=epoch)
    val_metrics   = trainer.execute_phase(val,   epoch=epoch)

    trainer.finalize_train_epoch(val_metrics.get(trainer.monitor))
    trainer.save_artifacts()   # checkpoints + metric plots + JSON export

    if trainer.should_stop_early():
        break
```

For step-level control:

```python
metrics = trainer.execute_phase(train, print_metrics=True)
metrics = trainer.execute_step(batch, val, print_metrics=True)
```

Both building blocks honour `accumulation_steps`: `execute_phase` flushes each cycle automatically, and `execute_step` updates the optimizer on every `accumulation_steps`-th `step` you pass:

```python
for i, batch in enumerate(train_loader, 1):
    trainer.execute_step(batch, train, step=i)
```

---

### Resetting

Composable building blocks for starting a *fresh* run without recreating the trainer — the same pieces `prepare_training()` calls internally when `resume=False` (see [`resume`](#constructor-parameters)):

```python
trainer.reset_trainer()          # setup + training state + metrics + cache + RNGs + scaler
trainer.clear_artifacts()        # delete checkpoints/, metrics/, plots/, dashboard files from run_dir

# Or reset one piece at a time:
trainer.clear_setup()            # discard models/optimizer/scheduler; next ensure_setup() rebuilds them
trainer.clear_models()           # drop the model registry alone
trainer.clear_optimizer()        # drop the optimizer alone
trainer.clear_scheduler()        # drop the scheduler alone
trainer.ensure_setup()           # call setup() exactly once; a no-op on later calls
trainer.reset_training_state()   # epoch counter, best-metric tracking, early-stopping counters
trainer.reset_seed()             # reseed Python / NumPy / Torch RNGs from `seed`
trainer.reset_scaler()           # rebuild the AMP GradScaler, discarding fp16 loss-scale adaptation
```

---

### State Inspection

```python
trainer.is_training_complete()       # True when current_epoch >= num_epochs
trainer.is_best_epoch()              # True if this epoch set the best monitored value
trainer.should_stop_early()          # True if patience is exhausted

trainer.print_status()                  # epoch counter, best monitored value, recent metrics
trainer.print_config()                  # the resolved config, as written to config.json
trainer.print_model_summary()           # model names and parameter counts
trainer.print_env_summary()             # OS, CPU, RAM, GPU, Python, PyTorch versions
trainer.print_optimization_summary()    # optimizer, scheduler, and gradient accumulation
trainer.print_schedule_summary(*phases) # the shape of one epoch; train() prints it for you
```

Three of those tables are also available as plain dicts — the same values the banner and the [dashboard](#live-dashboard) are built from — and `print_dict_tree` renders any dict in the trainer's own tree style:

```python
env      = trainer.get_env_summary()   # {"OS": "Ubuntu 22.04", "GPU": "cuda:0 …", …}
model    = trainer.get_model_summary() # {"encoder": "23,508,032 params", …}
schedule = trainer.get_schedule_summary(*phases)  # {"train": "training", "audit": "eval, every 3 epochs"}

trainer.print_dict_tree(env, header="🖥️  Environment")
```

The schedule is *not* in `config.json`: [phases](#phases) are arguments to `train()`, not to the constructor, so `from_config` could not pass them back. The file holds what reconstructs the trainer; the shape of an epoch is reported where the run reports everything else.

The environment summary stops where a *training framework's* knowledge stops — the machine, the Python runtime, the PyTorch stack. Which libraries (or dataset revisions, or commits) your result actually depends on is a property of your project, so a run declares them the same way it declares [config entries](#configuration) and [checkpoint extras](#persisting-custom-state-extras-vs-hooks) — by adding them:

```python
from train4all.utils import package_versions

trainer.update_env_summary(package_versions("timm", "transformers", "scikit-learn"))
trainer.update_env_summary({"commit": git_sha(), "dataset": DATA_REVISION})
```

`package_versions` takes distribution names as on PyPI (`scikit-learn`, not `sklearn`) and leaves out what is not installed, so one list is safe across machines. Added rows close the banner in the order declared, a key that collides with a computed row replaces it, and both the printed banner and the dashboard's Environment panel read from `get_env_summary()` — so an entry added once appears in both. Add them before `train()`: the banner is printed at the start of the run, ahead of `setup()`.

---

### Configuration

```python
# Merge custom entries into the trainer config (persisted in config.json).
trainer.update_config({"experiment": "baseline", "tag": "v1"})
trainer.save_config()

path = trainer.get_config_path()     # run/config.json
```

`config.json` is the one run artifact that lives *outside* the checkpoint — everything else (metrics, plots) reconstructs from the `.pth`. `from_config` is its inverse, rebuilding an equivalently configured trainer straight from the file:

```python
trainer = MyTrainer.from_config("run")                # or "run/config.json"
trainer = MyTrainer.from_config("run", device="cpu")  # overrides replace file values
```

Only `BaseTrainer` constructor arguments are consumed, so custom metadata added via `update_config` is ignored; a subclass's own constructor arguments (model hyperparameters, …) are not in the base config and must be passed as overrides.

---

### Snapshot

Set `run_snapshot_dir` and `train()` mirrors `run_dir` there **after every epoch**, once that epoch's artifacts are on disk — so the copy is always a whole epoch, and a host that vanishes mid-run (a preemptible VM, a Colab session) leaves the checkpoints behind on durable storage:

```python
trainer = MyTrainer(
    num_epochs=50,
    run_dir="run",
    run_snapshot_dir="/mnt/gdrive/experiments/run",
)
trainer.train(...)   # every epoch is mirrored; no further wiring
```

Nothing is excluded by default — the checkpoints are exactly what a mirror exists to preserve. When the mirror should stay light, that is a standing property of the run rather than of one call — the per-epoch snapshot is unattended, and an argument only a hand-written call could reach would never touch it — so it is configured beside the directory:

```python
trainer = MyTrainer(
    num_epochs=50,
    run_dir="run",
    run_snapshot_dir="/mnt/gdrive/experiments/run",
    run_snapshot_exclude=["checkpoints"],  # every epoch: the metrics and plots alone
)
```

Call `snapshot_run()` yourself for a snapshot at any other moment. A bare call takes whatever the trainer is set to take — the same mirror the epoch loop takes — and `exclude` overrides that for the one call:

```python
trainer.snapshot_run()                          # the configured mirror
trainer.snapshot_run(exclude=["checkpoints"])   # this call only
trainer.snapshot_run(exclude=[])                # this call only: mirror everything
```

Repeating the mirror is cheap and safe by construction, which is what lets the loop take it unattended: only the files that changed are copied, each one is replaced atomically, and whatever the run no longer has is deleted **last**, once every copy is in place. So the mirror is never emptied and never holds a half-written file — interrupt the run at any moment and every file in it is whole, from this epoch or the previous one. A mirror that cleared itself before rewriting would instead be empty precisely when the host it guards against is the one that vanished.

The destination must lie outside `run_dir`; a mirror nested inside its own source would copy itself and grow on every epoch, so it is rejected.

---

### GPU Utilities

```python
trainer.print_gpu_temperature()  # reads temperature via nvidia-smi; warns above _GPU_TEMP_WARN_C
trainer.empty_cuda_cache()       # gc.collect() + torch.cuda.empty_cache()
```

---

## Live Dashboard

Pass `use_dashboard=True` for a live, dependency-free dashboard in your browser: an overall-progress gauge, a KPI grid (current metric, best monitored value, throughput, ETA, learning rate, GPU memory), a live per-step loss graph, and a per-metric SVG chart for each metric. It follows your light/dark theme and embeds its data inline so it stays viewable offline.

<div align="center">
  <img alt="train4all dashboard — a completed run" src="assets/dashboard-complete.png" width="100%">
</div>

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

Whether there is a dashboard at all is `use_dashboard`, not a setting here.

| Parameter | Default | Description |
| :-- | :-- | :-- |
| `filename` | `"dashboard.html"` | HTML shell filename written inside `run_dir`. |
| `data_filename` | `"dashboard_data.json"` | JSON data file polled by the browser. |
| `poll_interval_ms` | `500` | Browser polling interval in milliseconds. |
| `open_on_start` | `True` | Open in the system's default browser when training begins. |
| `stale_after_ms` | `30000` | Mark the run **Offline** after this many ms without a heartbeat. An absolute liveness timeout, independent of `poll_interval_ms` — size it above your slowest synchronous pause (large checkpoint saves, heavy plotting). |
| `use_server` | `True` | Start a local HTTP server (required for Chrome/Edge on `file://` pages). |

The trainer drives the dashboard for you, but the engine is exported for anyone feeding it from their own loop. `Dashboard` writes the HTML shell once on `initialize()`, overwrites a small JSON file on every `update()`, and inlines that data on `finalize()` so the page survives the process; `mark_started()` sets the elapsed-time origin, `heartbeat()` keeps a long synchronous pause from reading as *Offline*, `open_browser()` raises the page on demand, and `url` / `path` / `active` / `elapsed` / `poll_s` report where and how it is running. `PhaseSpec` is the flat projection of a [`Phase`](#phases) it renders a schedule from (`name`, `training`, `steps`, `every`).

---

## Utilities

`train4all.utils` holds the helpers the trainer is built from. Nothing here is needed to train — they are exported for reuse:

```python
from train4all.utils import TrainerLogger, print_dict_tree, remove_dir
```

| Name | Description |
| :-- | :-- |
| `MetricTable` | Type of the metric tables: `dict[metric, dict[phase, list[float]]]`. |
| `TrainerLogger` | The `log()` protocol the [`logger`](#constructor-parameters) argument accepts. |
| `UnifiedLogger` | The default logger — console, plus a file in `run_dir`. |
| `LogLevel` | A log level: `"info"`, `"debug"`, or `"warn"`. |
| `Printer` | Type of a `print_fn` callback. |
| `print_dict_tree` | Render a nested dict as the `├─`/`└─` tree the trainer prints. |
| `separator_rule` | The horizontal rule drawn under a tree header. |
| `DEFAULT_KEY_WIDTH` | The `32` behind [`_KEY_WIDTH`](#class-constants) — the one place the tree's column width is decided, so the trainer's tables and `Checkpoint.print_summary()` line up by reference rather than by coincidence. |
| `save_curves_plot` | Save labelled 1-D curves to a PNG — matplotlib, without pyplot's global state. |
| `get_metric_plot_title` | Build a plot title from metric name, phase name, and prefix. |
| `get_metric_plot_filename` | Build a plot filename from the same parts. |
| `copy_dir` | Recursive copy with an exclude list — the repeatable, atomic mirror [`snapshot_run()`](#snapshot) is built on. Refuses a destination inside the source. |
| `remove_dir` | Recursive delete that clears read-only flags first. |
| `replace_dict_keys` | Rewrite substrings in nested dict keys — what `key_map` uses. |
| `Dashboard` | The [live dashboard](#live-dashboard) engine. |
| `DashboardConfig` | Its settings (see the table above). |
| `PhaseSpec` | A [phase](#phases) as the dashboard sees it. |

Machine introspection lives in `train4all.utils.system` — the trainer delegates to it rather than knowing how to read a Windows registry key or initialize NVML, the same way it delegates the on-disk format to `Checkpoint`:

| Name | Description |
| :-- | :-- |
| `env_summary` | The reproducibility banner as a dict — OS, CPU, RAM, disk, GPU, CUDA, Python, and the PyTorch stack. Behind [`get_env_summary()`](#state-inspection). |
| `package_versions` | Installed versions of the named distributions, skipping what is absent — the dict [`update_env_summary()`](#state-inspection) merges to put project libraries in the banner. |
| `os_name` | Distro on Linux, `macOS <ver>` on Darwin — not the kernel release. |
| `cpu_name` | CPU model, from the registry / `sysctl` / `/proc/cpuinfo` rather than the bare architecture. |
| `cuda_index` | The CUDA device index a `torch.device` resolves to. |
| `gpu_temperature` | Current GPU temperature in °C via `nvidia-smi`. Behind [`print_gpu_temperature()`](#gpu-utilities). |
| `empty_cuda_cache` | `gc.collect()` + `torch.cuda.empty_cache()`. |
| `GpuProbe` | Cached GPU-memory readings for one device — NVML once, then an `nvidia-smi` fallback cached for [`_GPU_MEM_TTL_S`](#class-constants) seconds, so a per-step progress bar costs one cheap lookup. |

---

## Development

```bash
git clone https://github.com/tomoking2004/train4all.git
cd train4all
pip install -e ".[dev]"
```

| Command | Purpose |
| :-- | :-- |
| `pytest` | Run the suite. |
| `pytest --cov` | Run it under coverage, which fails below 80% — the gate before a release. |
| `ruff check` | Lint. |

Coverage is a property of the *whole* suite, so its floor is enforced only when the whole suite runs — `pytest tests/test_phase.py` stays a fast, focused loop rather than a failure that says nothing about the code under test.

The suite also holds this README to the code: every exported name, constructor argument, class constant, and dashboard setting must appear here, or `tests/test_public_api.py` fails. The API reference above cannot quietly fall behind the thing it describes.

---

## License

[MIT](LICENSE) © 2026 tomoking2004
