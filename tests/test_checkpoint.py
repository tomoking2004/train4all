"""Checkpoint owns the on-disk format, so a trainer and an inspector never disagree."""

import shutil
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


def test_save_checkpoints_serializes_once_and_copies_the_rest(run_dir, monkeypatch):
    """best.pth and the periodic file are byte copies of latest.pth, not a second torch.save."""
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir,
        monitor="loss", monitor_phase="val", save_interval=1, use_progress_bar=False,
    )
    serialized: list[Path] = []
    real_save = Checkpoint.save

    def recording(self, path):
        serialized.append(Path(path))
        real_save(self, path)

    monkeypatch.setattr(Checkpoint, "save", recording)
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(8)),
    )

    latest = trainer.get_latest_checkpoint_path()
    assert serialized == [latest]
    for copy in (trainer.get_best_checkpoint_path(), trainer.get_checkpoint_path("epoch_1")):
        assert copy.read_bytes() == latest.read_bytes()


def test_a_failed_latest_write_is_not_copied_over_best(run_dir, monkeypatch):
    """The atomic save leaves last epoch's latest.pth intact; copying it would stamp a stale state as best."""
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir,
        monitor="loss", monitor_phase="val", use_progress_bar=False,
    )
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(8)),
    )
    latest, best = trainer.get_latest_checkpoint_path(), trainer.get_best_checkpoint_path()
    trainer.update_checkpoint_extras({"marker": "this epoch"})   # state the file on disk lacks
    real_save = Checkpoint.save

    def refuse(self, path):
        if Path(path) == latest:
            raise OSError("disk full")
        real_save(self, path)

    monkeypatch.setattr(Checkpoint, "save", refuse)
    trainer.save_checkpoints()                  # a save failure is a warning, never a raise

    assert "marker" not in Checkpoint.load(latest).extras
    assert Checkpoint.load(best).extras["marker"] == "this epoch"


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


def test_an_interrupted_copy_leaves_the_previous_best_untouched(run_dir, monkeypatch):
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir,
        monitor="loss", monitor_phase="val", use_progress_bar=False,
    )
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(8)),
    )
    best = trainer.get_best_checkpoint_path()
    before = best.read_bytes()
    trainer.update_checkpoint_extras({"marker": "this epoch"})

    def dies_midway(_src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", dies_midway)
    trainer.save_checkpoints()                  # latest is serialized; its copy to best fails

    assert best.read_bytes() == before
    assert not list(best.parent.glob(".*.partial"))
