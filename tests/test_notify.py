"""`notify` — one notice, once, however the run ends.

The trainer's part of a notification is knowing when a run is over and what happened;
delivery is a function the caller passes. So what is worth testing is exactly that
contract: every ending produces one call; the title alone says which run, how it ended
and on what value or error; the body carries what the reader would otherwise have to go and
look up; and a notifier that fails can neither stop a run nor stand in for its error.
"""

import json
from typing import Any

import pytest
import torch
from conftest import TinyTrainer, make_loader

from train4all import Notice, Phase


def run(trainer: TinyTrainer) -> None:
    trainer.train(Phase("train", make_loader(8), training=True), Phase("val", make_loader(4, seed=1)))


# ── Every ending, once ────────────────────────────────────────────────────────


def test_a_finished_run_is_announced_once_with_its_result(run_dir):
    said: list[Notice] = []
    trainer = TinyTrainer(num_epochs=2, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    run(trainer)

    assert len(said) == 1, "one run, one notification"
    notice = said[0]
    assert notice.title.startswith(f"[train4all] {run_dir.name} finished: best val loss ")
    assert notice.body.startswith("Finished 2 epochs in ")
    assert "Latest results:\n  train: loss " in notice.body, "each phase's latest figures"
    assert "\n  val: loss " in notice.body
    assert f"Best checkpoint: {trainer.get_best_checkpoint_path().resolve()}" in notice.body


def test_an_early_stop_says_where_it_stopped_and_why(run_dir):
    """A run that cannot learn (lr = 0) stops improving at once, so patience ends it."""
    said: list[Notice] = []
    trainer = TinyTrainer(num_epochs=9, learning_rate=0.0, patience=1, run_dir=run_dir,
                          use_progress_bar=False, notify=said.append)
    run(trainer)

    assert len(said) == 1
    assert f"{run_dir.name} stopped early at epoch 2 of 9: best val loss " in said[0].title
    assert "no improvement for 1 epoch (patience 1)" in said[0].body


class Failing(TinyTrainer):
    def compute_loss(self, batch: Any) -> torch.Tensor:
        raise RuntimeError("boom")


class FailingLater(TinyTrainer):
    """Finishes one epoch, then dies — so there is a result so far, and a checkpoint to resume from."""

    def compute_loss(self, batch: Any) -> torch.Tensor:
        if self._current_epoch == 2:
            raise RuntimeError("boom")
        return super().compute_loss(batch)


class Interrupted(TinyTrainer):
    def compute_loss(self, batch: Any) -> torch.Tensor:
        raise KeyboardInterrupt


def test_a_failure_is_announced_with_its_cause_and_still_raised(run_dir):
    said: list[Notice] = []
    trainer = Failing(num_epochs=3, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    with pytest.raises(RuntimeError, match="boom"):
        run(trainer)

    assert len(said) == 1
    notice = said[0]
    assert notice.title == f"[train4all] {run_dir.name} failed at epoch 1 of 3: RuntimeError"
    assert "RuntimeError: boom\nRaised in compute_loss (test_notify.py, line " in notice.body
    assert "No epoch finished." in notice.body
    assert "Best" not in str(notice), "no epoch finished, so there is no best value to report"
    assert f"Log: {(run_dir / 'log.txt').resolve()}" in notice.body


def test_a_failure_after_progress_reports_the_result_so_far_and_how_to_resume(run_dir):
    said: list[Notice] = []
    trainer = FailingLater(num_epochs=3, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    with pytest.raises(RuntimeError, match="boom"):
        run(trainer)

    body = said[0].body
    assert body.startswith("Failed at epoch 2 of 3 after ")
    assert "Best val loss " in body and "Latest results:\n  train: loss " in body
    latest = trainer.get_latest_checkpoint_path().resolve()
    assert f"Latest checkpoint: {latest} (run again to resume from it)" in body
    assert "Best checkpoint:" not in body, "a failed run points at what to resume, not at what to ship"


def test_ctrl_c_is_an_interruption_not_a_failure(run_dir):
    said: list[Notice] = []
    trainer = Interrupted(num_epochs=3, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    with pytest.raises(KeyboardInterrupt):
        run(trainer)

    assert len(said) == 1
    assert said[0].title == f"[train4all] {run_dir.name} interrupted at epoch 1 of 3"
    assert "KeyboardInterrupt" not in str(said[0])


# ── The notice itself ─────────────────────────────────────────────────────────


def test_the_title_is_one_ascii_line(run_dir):
    """It is sent as a mail subject and as an HTTP header, and neither takes a line
    break — nor, in a header, anything outside Latin-1."""
    said: list[Notice] = []
    trainer = Failing(num_epochs=3, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    with pytest.raises(RuntimeError):
        run(trainer)

    assert "\n" not in said[0].title
    assert said[0].title.isascii()


def test_the_body_says_when_and_where_it_ran(run_dir):
    said: list[Notice] = []
    trainer = TinyTrainer(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    run(trainer)

    assert "\nStarted " in said[0].body and ", ended " in said[0].body
    assert f", {trainer.device}." in said[0].body
    assert f"Run directory: {run_dir.resolve()}" in said[0].body


def test_str_is_the_title_and_the_body_together():
    assert str(Notice("the title", "the body")) == "the title\n\nthe body"


# ── The contract around it ────────────────────────────────────────────────────


def test_the_console_says_the_notice_went_out(run_dir, capsys):
    trainer = TinyTrainer(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=lambda _notice: None)
    run(trainer)

    assert f"📨 Notification sent — {run_dir.name} finished" in capsys.readouterr().out


def broken(_notice: Notice) -> None:
    raise ConnectionError("no route to host")


def test_a_notifier_that_fails_does_not_fail_the_run(run_dir, capsys):
    trainer = TinyTrainer(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=broken)
    run(trainer)   # must not raise

    assert trainer.is_training_complete()
    out = capsys.readouterr().out
    assert "notify failed" in out and "no route to host" in out
    assert "Notification sent" not in out, "the console must not claim a delivery that did not happen"


def test_a_notifier_that_fails_does_not_replace_the_runs_own_error(run_dir):
    trainer = Failing(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=broken)
    with pytest.raises(RuntimeError, match="boom"):
        run(trainer)


def test_ctrl_c_during_the_notifier_is_the_users_and_carries_the_runs_error(run_dir):
    """A notifier stuck on the network is exactly when someone reaches for Ctrl-C.
    That is not the notifier failing, so it is not swallowed — and the error the run
    was already leaving with is not lost: it rides along as the interrupt's context."""
    def interrupted(_notice: Notice) -> None:
        raise KeyboardInterrupt

    trainer = Failing(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        run(trainer)

    assert isinstance(caught.value.__context__, RuntimeError)
    assert str(caught.value.__context__) == "boom"


@pytest.mark.parametrize("mishap", [RuntimeError("the driver went away"), KeyboardInterrupt()])
def test_a_finished_run_is_announced_even_if_the_cleanup_after_it_is_cut_short(run_dir, mishap):
    """Closing the dashboard and releasing the GPU cache come after the last hook: the
    run has finished by then. If one of them raises — or Ctrl-C lands in the page's
    last wait — the notice must still go out, and must still say the run finished."""
    said: list[Notice] = []

    class BadCleanup(TinyTrainer):
        @staticmethod
        def empty_cuda_cache() -> None:
            raise mishap

    trainer = BadCleanup(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    with pytest.raises(type(mishap)):
        run(trainer)

    assert len(said) == 1, "the run was over; it must not end without a word"
    assert said[0].title.startswith(f"[train4all] {run_dir.name} finished")


def test_without_notify_nothing_changes(run_dir, capsys):
    trainer = TinyTrainer(num_epochs=1, run_dir=run_dir, use_progress_bar=False)
    run(trainer)

    assert trainer.notify is None
    assert "Notif" not in capsys.readouterr().out


def test_notify_must_be_callable(run_dir):
    with pytest.raises(TypeError, match="notify must be a function"):
        TinyTrainer(num_epochs=1, run_dir=run_dir, notify="https://ntfy.sh/my-topic")  # type: ignore[arg-type]


def test_notify_is_not_part_of_the_saved_config(run_dir):
    """Where a message goes belongs to a person and a run, not to the experiment."""
    trainer = TinyTrainer(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=print)
    trainer.save_config()

    assert "notify" not in json.loads(trainer.get_config_path().read_text(encoding="utf-8"))


# ── Hooks that raise ──────────────────────────────────────────────────────────


def test_a_failure_in_on_training_end_is_still_a_failure_that_gets_announced(run_dir):
    """The hook runs after the loop, but the run is not over until it returns: if it
    raises, the notice must say the run failed rather than never arrive."""
    said: list[Notice] = []

    class BadEnding(TinyTrainer):
        def on_training_end(self) -> None:
            raise RuntimeError("could not upload the weights")

    trainer = BadEnding(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    with pytest.raises(RuntimeError, match="could not upload"):
        run(trainer)

    assert len(said) == 1
    assert " failed at epoch 1 of 1: RuntimeError" in said[0].title


def test_a_raising_on_exception_does_not_cost_the_run_its_notice(run_dir):
    said: list[Notice] = []

    class BadHandler(Failing):
        def on_exception(self, exc: BaseException) -> None:
            raise ValueError("the handler itself broke")

    trainer = BadHandler(num_epochs=1, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    with pytest.raises(ValueError, match="the handler itself broke"):
        run(trainer)

    assert len(said) == 1, "the notice is sent whatever the hook does"
    assert "RuntimeError: boom" in said[0].body, "and it reports the run's own error"


def test_a_failure_before_the_first_epoch_is_worded_as_one(run_dir):
    said: list[Notice] = []

    class NeverStarts(TinyTrainer):
        def on_training_start(self) -> None:
            raise RuntimeError("no data")

    trainer = NeverStarts(num_epochs=3, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    with pytest.raises(RuntimeError, match="no data"):
        run(trainer)

    assert len(said) == 1, "the run had begun; it must not end without a word"
    assert said[0].title == f"[train4all] {run_dir.name} failed before the first epoch: RuntimeError"
    assert said[0].body.startswith("Failed before the first epoch after ")


def test_a_phase_that_runs_every_other_epoch_is_still_reported(run_dir):
    """Its series is shorter than the others', which once dropped it from the notice."""
    said: list[Notice] = []
    trainer = TinyTrainer(num_epochs=2, run_dir=run_dir, use_progress_bar=False, notify=said.append)
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(4, seed=1), every=2),
    )

    assert "\n  val: loss " in said[0].body
