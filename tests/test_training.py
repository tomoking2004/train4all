"""The training loop: an epoch is whatever sequence of phases you hand it."""

from typing import Any

import pytest
from conftest import TinyTrainer, make_loader

from train4all import Phase


def test_an_epoch_runs_its_phases_in_order(trainer):
    seen: list[str] = []

    trainer.on_phase_start = lambda _epoch, phase: seen.append(phase.name)
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(8)),
    )
    assert seen == ["train", "val"] * trainer.num_epochs


def test_every_phase_gets_its_own_metric_series(trainer):
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(8)),
    )
    loss = trainer.get_epoch_metrics()["loss"]
    assert set(loss) == {"train", "val"}
    assert len(loss["train"]) == len(loss["val"]) == trainer.num_epochs


def test_a_phase_with_every_sits_out_the_epochs_it_skips(run_dir):
    trainer = TinyTrainer(
        num_epochs=6, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False,
    )
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("audit", make_loader(8), every=3),
    )
    loss = trainer.get_epoch_metrics()["loss"]
    assert len(loss["train"]) == 6
    assert len(loss["audit"]) == 2, "audit should run only at epochs 3 and 6"


def test_metric_fn_suppresses_metrics_but_never_the_loss(trainer):
    trainer.train(
        Phase("train", make_loader(8), training=True, metric_fn=lambda _: {}),
        Phase("val", make_loader(8)),
    )
    metrics = trainer.get_epoch_metrics()
    assert "train" in metrics["loss"], "loss is always recorded"
    assert "train" not in metrics["accuracy"], "the metric function was suppressed"
    assert "val" in metrics["accuracy"]


def test_train_requires_at_least_one_phase(trainer):
    with pytest.raises(ValueError, match="at least one Phase"):
        trainer.train()


def test_phase_names_must_be_unique(trainer):
    with pytest.raises(ValueError, match="unique"):
        trainer.train(
            Phase("train", make_loader(8), training=True),
            Phase("train", make_loader(8)),
        )


def test_train_requires_num_epochs(run_dir):
    trainer = TinyTrainer(run_dir=run_dir, use_progress_bar=False)
    with pytest.raises(ValueError, match="num_epochs"):
        trainer.train(Phase("train", make_loader(8), training=True))


def test_early_stopping_reads_the_monitored_phase(run_dir):
    trainer = TinyTrainer(
        num_epochs=20, learning_rate=0.0,          # a flat run: nothing ever improves
        run_dir=run_dir, patience=2, monitor="loss", monitor_phase="val",
        use_progress_bar=False,
    )
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(8)),
    )
    assert trainer.should_stop_early()
    assert trainer._current_epoch < 20, "early stopping never triggered"


def test_test_uses_compute_test_metrics(run_dir):
    class WithTestMetrics(TinyTrainer):
        def compute_test_metrics(self, batch: Any) -> dict[str, float]:
            metrics = self.compute_metrics(batch)
            metrics["report_only"] = 1.0
            return metrics

    trainer = WithTestMetrics(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(8), training=True))
    metrics = trainer.test(make_loader(8))
    assert metrics["report_only"] == 1.0
    assert "report_only" not in trainer.get_epoch_metrics().get("report_only", {}).get("train", [])


def test_the_step_cache_bridges_loss_and_metrics(trainer):
    """compute_metrics reads the logits compute_loss stashed — no second forward."""
    trainer.train(Phase("train", make_loader(8), training=True))
    assert "accuracy" in trainer.get_epoch_metrics()


def test_the_cache_is_cleared_between_phases(trainer):
    trainer.ensure_setup()
    trainer.set_cache("stale", "value")
    trainer.execute_phase(Phase("val", make_loader(8)))
    assert trainer.get_cache("stale") is None
