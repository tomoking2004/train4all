"""
The staged run behind the README's dashboard images.

A 40-epoch, 10-class image-classification job (≈159k train / 20k val images, batch
256) fine-tuning a ResNet-50 on an RTX 4090 / Ryzen 9 workstation. The hardware and
the software versions form a coherent, real stack; the curves are ordinary — loss
falls, accuracy rises, and validation crests at epoch 34, then overfits.

Why a *heavy* model, when the README's Quick Start is a small MLP? Because the hero
frame needs one. The dashboard's step-loss strip holds 96 samples at one write per
0.5 s, so filling it takes a phase lasting ≳48 s — that is ``steps ≥ 48 × it/s``. At
620 steps and 8 it/s (a 4090's real ResNet-50 @224 AMP rate) this run clears that,
its strip full at step 384; an MLP finishes an epoch in seconds and could never fill
it. The Quick Start's own image is a different thing entirely: a *real* run, shot by
``shoot.py``.

``demo_dashboard.py`` (look at it) and ``shoot.py`` (photograph it) both read from
here, so the live page and the README's images can never disagree.
"""

import math
import random
from collections.abc import Iterator

from train4all import PhaseSpec

type History = dict[str, dict[str, list[float]]]
"""Per-epoch metric history: ``metric → phase → value per epoch``."""

# ── Schedule ──────────────────────────────────────────────────────────────────
EPOCHS      = 40
BATCH_SIZE  = 256
TRAIN_STEPS = 620          # 158,720 images / batch
VAL_STEPS   = 78           #  19,968 images / batch
PEAK_EPOCH  = 34           # validation crests here, then overfits
LR_BASE     = 3e-3
LR_MIN      = 1e-5
THROUGHPUT  = 8            # it/s — a 4090's real ResNet-50 @224 AMP rate; drives speed / ETA
WRITE_S     = 0.5          # seconds between dashboard writes — the trainer's own throttle
GPU_MEM     = (14.84, 25.76)   # (used, total) GB — NVML total matches the torch VRAM below

METRICS = ("loss", "accuracy", "f1", "top2_accuracy")

# ── The epoch — the sequence of phases the dashboard lays itself out from ─────
PHASES = [
    PhaseSpec("train", training=True, steps=TRAIN_STEPS),
    PhaseSpec("val", steps=VAL_STEPS),
]

# ── The host — declared once, and the same in every README image ──────────────
# The one fact in the images that is *declared* rather than measured. Every image
# has to agree on the machine (a reader compares them), and the dashboard's GPU
# meter and it/s readout only mean anything on a GPU — so a single reference
# workstation is declared here and stamped onto the shot, both on this staged run
# and on the Quick Start's real one.
ENV = {
    "OS": "Ubuntu 22.04.4 LTS", "CPU": "AMD Ryzen 9 7950X 16-Core Processor",
    "CPU cores": 32, "RAM": "67.18 GB", "Disk": "812.44 / 1968.32 GB free",
    "GPU": "cuda:0 NVIDIA GeForce RTX 4090", "VRAM": "25.76 GB",
    "CUDA": "12.4", "cuDNN": "90100", "Python": "3.12.4", "PyTorch": "2.5.1",
    "torchvision": "0.20.1",
}
DEVICE = "cuda:0"   # what that host resolves to, and what every image's config shows
# Exactly what BaseTrainer persists: only customized, reproducibility-relevant
# args plus the resolved device, in constructor order.
CONFIG = {
    "num_epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR_BASE,
    "amp": "bf16", "patience": 8, "monitor": "accuracy", "monitor_mode": "max",
    "seed": 7, "device": DEVICE,
}
MODEL = {"encoder": "23,508,032 params", "head": "20,490 params"}   # ResNet-50 backbone + Linear(2048, 10)


def lr(epoch: int) -> float:
    """Cosine schedule, ``LR_BASE`` → ``LR_MIN`` across the run."""
    t = (epoch - 1) / (EPOCHS - 1)
    return LR_MIN + 0.5 * (LR_BASE - LR_MIN) * (1 + math.cos(math.pi * t))


def _sat(lo: float, hi: float, k: float, t: float) -> float:   # saturating lo→hi over t∈[0,1]
    return hi + (lo - hi) * math.exp(-k * t)


def _clamp01(x: float) -> float:                               # a rate in (0, 1), never a fake perfect 1.0
    return max(1e-4, min(x, 0.9999))


def series() -> History:
    """Per-epoch metric history — deterministic, so the live page and every
    image read identical numbers. Train metrics improve throughout; every val
    metric crests at ``PEAK_EPOCH`` then overfits (loss back up, accuracy / f1 /
    top-2 down), so all four charts agree on the best epoch."""
    rng = random.Random(7)
    def noise(scale: float) -> float:
        return (rng.random() - 0.5) * scale
    out: History = {m: {"train": [], "val": []} for m in METRICS}
    peak_t = (PEAK_EPOCH - 1) / (EPOCHS - 1)
    for e in range(1, EPOCHS + 1):
        t  = (e - 1) / (EPOCHS - 1)
        tv = min(t, peak_t)                          # val rise saturates at the crest
        ov = max(0, e - PEAK_EPOCH)                  # epochs of overfitting past it
        tr_acc = _clamp01(_sat(0.22, 0.995, 4.2, t) + noise(0.004))
        va_acc = _clamp01(_sat(0.18, 0.945, 3.6, tv) - 0.0026 * ov + noise(0.002))
        out["loss"]["train"].append(max(_sat(2.30, 0.12, 3.4, t) + noise(0.015), 0.02))
        out["loss"]["val"].append(_sat(2.34, 0.30, 3.0, tv) + 0.012 * ov + noise(0.015))
        out["accuracy"]["train"].append(tr_acc)
        out["accuracy"]["val"].append(va_acc)
        out["f1"]["train"].append(_clamp01(tr_acc - 0.010 + noise(0.003)))
        out["f1"]["val"].append(_clamp01(va_acc - 0.012 + noise(0.003)))
        out["top2_accuracy"]["train"].append(_clamp01(_sat(0.45, 0.999, 4.2, t) + noise(0.002)))
        out["top2_accuracy"]["val"].append(_clamp01(_sat(0.40, 0.985, 4.0, tv) - 0.0014 * ov + noise(0.0015)))
    return out


def best(val_acc: list[float], upto: int) -> tuple[float, int, int]:
    """Best val accuracy over the first ``upto`` epochs → (value, epoch, no_improve)."""
    i = max(range(upto), key=lambda j: val_acc[j])
    return val_acc[i], i + 1, upto - (i + 1)


def table(history: History, upto: int) -> History:
    """The history as of ``upto`` completed epochs — the curves the dashboard has
    already drawn when a frame opens."""
    return {m: {p: v[:upto] for p, v in history[m].items()} for m in history}


# ── The Hero Frame ────────────────────────────────────────────────────────────
# The live landing ``demo_dashboard`` freezes on: mid-training, charts nearly full,
# the best value still improving, and the step-loss strip chart running at a full
# window. It must be *played*, not replayed from a snapshot: the browser measures
# throughput and ETA itself, by watching the step counter advance across successive
# polls, so those two readouts only exist in a run that actually advances. Staged at
# ``THROUGHPUT`` it/s, which is exactly the rate the pacing below reproduces.

HERO_EPOCH = 33      # 32 epochs done; the best one sits one epoch behind the live front
HERO_FRAC  = 0.62    # fraction through the hero epoch's train phase at the freeze —
                     # exactly fills the 96-sample step-loss strip chart (96 · 4 / 620)
JITTER_SEED = 11     # per-step noise; ephemeral, so it is not part of ``series()``


def hero_steps(history: History) -> Iterator[tuple[int, dict[str, float]]]:
    """The hero epoch's train phase, step by step → ``(step, step_metrics)``.

    One sample per dashboard write: a real run throttles to one write every
    ``WRITE_S`` seconds, which at ``THROUGHPUT`` it/s is every ``stride`` steps. The
    caller sleeps that same ``WRITE_S`` between samples, so the browser sees the step
    counter advance at ``THROUGHPUT`` it/s and measures exactly that.
    """
    jitter = random.Random(JITTER_SEED)
    base = history["loss"]["train"][HERO_EPOCH - 1]
    train_now = {m: history[m]["train"][HERO_EPOCH - 1] for m in METRICS}
    stride = round(THROUGHPUT * WRITE_S)                # steps between throttled writes
    samples = round(TRAIN_STEPS * HERO_FRAC / stride)
    for s in range(1, samples + 1):
        step = s * stride
        step_metrics = {
            "loss": max(
                base * (1.05 - 0.05 * step / TRAIN_STEPS) + (jitter.random() - 0.5) * 0.045,
                1e-3,
            )
        }
        for m in ("accuracy", "f1", "top2_accuracy"):
            step_metrics[m] = train_now[m] + (jitter.random() - 0.5) * 0.01
        yield step, step_metrics
