"""Checkpoint owns the on-disk format, so a trainer and an inspector never disagree."""

from pathlib import Path

import pytest
import torch
from conftest import TinyTrainer, make_loader

from train4all import Checkpoint, Phase


def test_a_full_checkpoint_round_trips(trainer):
    trainer.train(Phase("train", make_loader(8), training=True))
    weights = trainer.net.weight.detach().clone()
    epoch = trainer._current_epoch

    reloaded = TinyTrainer(
        num_epochs=3, learning_rate=0.1, run_dir=trainer.run_dir, use_progress_bar=False,
    )
    reloaded.load_latest_checkpoint()

    torch.testing.assert_close(reloaded.net.weight.detach(), weights)
    assert reloaded._current_epoch == epoch
    assert reloaded.get_epoch_metrics()["loss"]["train"]


def test_a_weights_only_checkpoint_carries_no_optimizer(trainer):
    trainer.train(Phase("train", make_loader(8), training=True))
    path = trainer.run_dir / "weights.pth"
    trainer.save_weights(path)

    ckpt = Checkpoint.load(path)
    assert ckpt.models
    assert ckpt.optimizer_state is None
    assert ckpt.scheduler_state is None
    assert ckpt.scaler_state is None


def test_a_checkpoint_opens_with_no_model_and_no_subclass(trainer):
    trainer.train(Phase("train", make_loader(8), training=True))
    ckpt = Checkpoint.load(trainer.get_latest_checkpoint_path())

    assert ckpt.version == Checkpoint.VERSION
    assert set(ckpt.models) == {"net"}
    assert ckpt.model_summary()["net"]["parameters"] == 4 * 3 + 3
    assert ckpt.metric_names() == ["accuracy", "loss"]
    assert ckpt.training_state["current_epoch"] == trainer.num_epochs
    assert isinstance(ckpt.summary(), dict)
    assert "net" in repr(ckpt)


def test_extras_ride_along_with_both_kinds_of_save(trainer):
    trainer.ensure_setup()
    trainer.update_checkpoint_extras({"class_names": ["a", "b", "c"]})

    for save, name in ((trainer.save_checkpoint, "full.pth"), (trainer.save_weights, "w.pth")):
        path = trainer.run_dir / name
        save(path)
        assert Checkpoint.load(path).extras == {"class_names": ["a", "b", "c"]}


def test_extras_are_restored_on_load(trainer):
    trainer.ensure_setup()
    trainer.update_checkpoint_extras({"note": "baseline"})
    trainer.save_checkpoint(trainer.run_dir / "c.pth")

    fresh = TinyTrainer(run_dir=trainer.run_dir, use_progress_bar=False)
    fresh.load_checkpoint(trainer.run_dir / "c.pth")
    assert fresh.get_checkpoint_extras() == {"note": "baseline"}


def test_excluded_models_stay_out_of_the_file(trainer):
    trainer.ensure_setup()
    trainer.exclude_from_checkpoint("net")
    trainer.save_checkpoint(trainer.run_dir / "c.pth")
    assert Checkpoint.load(trainer.run_dir / "c.pth").models == {}


def test_legacy_training_state_keys_are_normalized():
    """Old files stored best_val_loss / best_val_epoch; readers must not care."""
    ckpt = Checkpoint({"training_state": {"best_val_loss": 0.25, "best_val_epoch": 4}})
    assert ckpt.training_state["best_metric"] == 0.25
    assert ckpt.training_state["best_epoch"] == 4


def test_a_canonical_key_wins_over_its_legacy_twin():
    ckpt = Checkpoint({"training_state": {"best_metric": 1.0, "best_val_loss": 9.0}})
    assert ckpt.training_state["best_metric"] == 1.0


def test_the_best_checkpoint_tracks_the_monitored_metric(run_dir):
    trainer = TinyTrainer(
        num_epochs=4, learning_rate=0.1, run_dir=run_dir,
        monitor="loss", monitor_mode="min", monitor_phase="val", use_progress_bar=False,
    )
    trainer.train(
        Phase("train", make_loader(16), training=True),
        Phase("val", make_loader(16)),
    )
    assert trainer.has_best_checkpoint()
    best = Checkpoint.load(trainer.get_best_checkpoint_path())
    val_losses = trainer.get_epoch_metrics()["loss"]["val"]
    assert best.training_state["best_epoch"] == val_losses.index(min(val_losses)) + 1


def test_an_interrupted_save_leaves_the_previous_file_untouched(run_dir, monkeypatch):
    path = run_dir / "ckpt.pth"
    Checkpoint.build(models={}, extras={"marker": "old"}, weights_only=True).save(path)
    before = path.read_bytes()

    def dies_midway(_obj, f):
        Path(f).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", dies_midway)
    with pytest.raises(OSError, match="disk full"):
        Checkpoint.build(models={}, extras={"marker": "new"}, weights_only=True).save(path)

    assert path.read_bytes() == before
    assert list(run_dir.iterdir()) == [path]      # and no temporary left behind
