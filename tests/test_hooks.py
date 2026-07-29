"""The lifecycle hooks, and the checkpoint dict protocol they lean on.

`on_save_checkpoint(ckpt)` / `on_load_checkpoint(ckpt)` are the documented way to
round-trip custom state (an EMA, an RNG state) across a resume, and the README
shows `checkpoint["ema"] = ...` doing it. None of that had a test.
"""

import contextlib
from typing import Any

import pytest
import torch
from conftest import TinyTrainer, make_loader

from train4all import Checkpoint, Phase


def test_the_checkpoint_behaves_like_a_dict():
    ckpt = Checkpoint({"models": {}})
    assert "models" in ckpt
    assert "ema" not in ckpt
    ckpt["ema"] = [1, 2, 3]
    assert ckpt["ema"] == [1, 2, 3]
    assert ckpt.raw["ema"] == [1, 2, 3]
    assert bool(ckpt) is True
    assert bool(Checkpoint({})) is False


class WithEMA(TinyTrainer):
    """Rides custom state across a save/load, exactly as the README shows."""

    def setup(self) -> None:
        super().setup()
        self.ema = torch.zeros(3)
        self.loaded_ema: torch.Tensor | None = None

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        checkpoint["ema"] = self.ema.clone()

    def on_load_checkpoint(self, checkpoint: Checkpoint) -> None:
        if "ema" in checkpoint:
            self.loaded_ema = checkpoint["ema"]


def test_custom_state_round_trips_through_the_hooks(run_dir):
    saver = WithEMA(num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    saver.ensure_setup()
    saver.ema = torch.tensor([1.0, 2.0, 3.0])
    saver.save_checkpoint(run_dir / "c.pth")

    loader_ = WithEMA(run_dir=run_dir, use_progress_bar=False)
    loader_.load_checkpoint(run_dir / "c.pth")
    torch.testing.assert_close(loader_.loaded_ema, torch.tensor([1.0, 2.0, 3.0]))


def test_the_save_hook_does_not_fire_for_weights_only(run_dir):
    """Weights-only saves stay pure: models + extras, nothing a hook attached."""
    trainer = WithEMA(run_dir=run_dir, use_progress_bar=False)
    trainer.ensure_setup()
    trainer.ema = torch.tensor([9.0, 9.0, 9.0])
    trainer.save_weights(run_dir / "w.pth")
    assert "ema" not in Checkpoint.load(run_dir / "w.pth")


def test_the_hooks_fire_in_the_documented_order(run_dir):
    seen: list[str] = []

    class Recorder(TinyTrainer):
        def on_training_start(self) -> None:
            seen.append("training_start")

        def on_train_epoch_start(self, epoch: int) -> None:
            seen.append("epoch_start")

        def on_phase_start(self, epoch: int | None, phase: Phase) -> None:
            seen.append(f"phase_start:{phase.name}")

        def on_step_start(self, step: int | None, batch: Any, phase: Phase) -> None:
            seen.append("step_start")

        def on_after_backward(self) -> None:
            seen.append("after_backward")

        def on_before_optimizer_step(self) -> None:
            seen.append("before_step")

        def on_step_end(self, step, batch, metrics, phase) -> None:
            seen.append("step_end")

        def on_phase_end(self, epoch, phase, metrics) -> None:
            seen.append(f"phase_end:{phase.name}")

        def on_train_epoch_end(self, epoch: int) -> None:
            seen.append("epoch_end")

        def on_training_end(self) -> None:
            seen.append("training_end")

    trainer = Recorder(num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    trainer.train(Phase("train", make_loader(4, batch_size=4), training=True))

    assert seen[0] == "training_start"
    assert seen[-1] == "training_end"
    assert seen[1] == "epoch_start"
    assert seen[2] == "phase_start:train"
    # one batch: start -> backward -> step -> end
    assert seen[3:7] == ["step_start", "after_backward", "before_step", "step_end"]
    assert "phase_end:train" in seen
    assert seen.index("phase_end:train") < seen.index("epoch_end")


def test_epoch_metrics_are_recorded_before_the_phase_end_hook(run_dir):
    """The docstring promises get_epoch_metrics() already reflects this epoch here."""
    captured: list[int] = []

    class Peek(TinyTrainer):
        def on_phase_end(self, epoch, phase, metrics) -> None:
            captured.append(len(self.get_epoch_metrics()["loss"][phase.name]))

    trainer = Peek(num_epochs=2, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    trainer.train(Phase("train", make_loader(8), training=True))
    assert captured == [1, 2], "the phase's own epoch metric was not recorded yet"


def test_the_cache_is_cleared_before_the_phase_start_hook(run_dir):
    """Also promised by the docstring — the hook sees a clean cache."""
    seen: list[Any] = []

    class Peek(TinyTrainer):
        def on_phase_start(self, epoch, phase) -> None:
            seen.append(self.get_cache("logits"))

    trainer = Peek(num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    trainer.ensure_setup()
    trainer.set_cache("logits", "stale")
    trainer.train(Phase("train", make_loader(8), training=True))
    assert seen == [None]


def test_on_exception_fires_and_the_error_is_reraised(run_dir):
    caught: list[BaseException] = []

    class Interrupted(TinyTrainer):
        def compute_loss(self, batch: Any) -> torch.Tensor:
            raise KeyboardInterrupt("ctrl-c")

        def on_exception(self, exc: BaseException) -> None:
            caught.append(exc)

    trainer = Interrupted(num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    try:
        trainer.train(Phase("train", make_loader(4), training=True))
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the exception was swallowed instead of re-raised")

    assert len(caught) == 1
    assert isinstance(caught[0], KeyboardInterrupt), "even a KeyboardInterrupt must reach the hook"


class Failing(TinyTrainer):
    """Dies mid-phase, for the paths that have to survive a run that does."""

    def compute_loss(self, batch: Any) -> torch.Tensor:
        raise RuntimeError("boom")


def test_no_checkpoint_is_written_when_the_loop_aborts(run_dir):
    trainer = Failing(num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    with contextlib.suppress(RuntimeError):
        trainer.train(Phase("train", make_loader(4), training=True))
    assert not trainer.get_latest_checkpoint_path().exists(), (
        "a mid-epoch save would persist an incomplete state"
    )


# ── The running phase ─────────────────────────────────────────────────────────
# What a callback handed no phase — `compute_loss` above all — can still learn
# about the pass it belongs to.


class Watcher(TinyTrainer):
    """Records the pass each `compute_loss` call turned out to be part of."""

    def setup(self) -> None:
        super().setup()
        self.seen: list[tuple[str, bool]] = []

    def compute_loss(self, batch: Any) -> torch.Tensor:
        self.seen.append((self.current_phase.name, self.training))
        return super().compute_loss(batch)


def test_compute_loss_can_tell_a_training_pass_from_an_evaluation_one(run_dir):
    trainer = Watcher(num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    trainer.train(
        Phase("train", make_loader(4, batch_size=4), training=True),
        Phase("val", make_loader(4, batch_size=4)),
    )
    assert trainer.seen == [("train", True), ("val", False)]


def test_a_standalone_step_marks_its_phase_too(run_dir):
    """`execute_step` is an entry point of its own, not only the epoch loop's inside."""
    trainer = Watcher(learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    trainer.ensure_setup()
    phase = Phase("audit", make_loader(4, batch_size=4))

    trainer.execute_step(next(iter(phase.loader)), phase)
    assert trainer.seen == [("audit", False)]


def test_nothing_is_running_between_passes(run_dir):
    outside: list[Phase | None] = []

    class Peek(TinyTrainer):
        def on_train_epoch_end(self, epoch: int) -> None:
            outside.append(self.current_phase)

    trainer = Peek(num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    assert trainer.current_phase is None and trainer.training is False

    trainer.train(Phase("train", make_loader(4), training=True))
    assert outside == [None], "an epoch is not a pass — the hook runs between them"
    assert trainer.current_phase is None


def test_the_mark_unwinds_when_a_pass_raises(run_dir):
    trainer = Failing(num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    with contextlib.suppress(RuntimeError):
        trainer.train(Phase("train", make_loader(4), training=True))
    assert trainer.current_phase is None, "a failed pass left itself marked as running"


def test_the_running_phase_is_read_only(trainer):
    """Assignable state would be a second copy of what the Phase already says."""
    with pytest.raises(AttributeError):
        trainer.training = True
    with pytest.raises(AttributeError):
        trainer.current_phase = Phase("nope", make_loader(4))
