"""config.json records exactly what the caller customized, and unpacks straight back in."""

import json

import pytest
from conftest import TinyTrainer, make_loader

from train4all import Phase


def test_only_customized_arguments_are_recorded(run_dir):
    trainer = TinyTrainer(num_epochs=5, run_dir=run_dir, use_progress_bar=False)
    trainer.save_config()
    config = json.loads(trainer.get_config_path().read_text(encoding="utf-8"))

    assert config["num_epochs"] == 5
    assert "monitor" not in config, "a default must not be written back"
    assert "patience" not in config
    assert "run_dir" not in config, "operational args are not reproducibility state"


def test_the_resolved_device_is_always_pinned(run_dir):
    trainer = TinyTrainer(run_dir=run_dir, use_progress_bar=False)
    trainer.save_config()
    config = json.loads(trainer.get_config_path().read_text(encoding="utf-8"))
    assert config["device"] == str(trainer.device)


def test_from_config_reconstructs_the_trainer(run_dir):
    original = TinyTrainer(
        num_epochs=7, learning_rate=0.05, run_dir=run_dir, seed=3,
        patience=2, monitor="accuracy", monitor_mode="max", accumulation_steps=2,
        use_progress_bar=False,
    )
    original.save_config()

    rebuilt = TinyTrainer.from_config(run_dir, run_dir=run_dir)
    assert rebuilt.num_epochs == 7
    assert rebuilt.learning_rate == 0.05
    assert rebuilt.seed == 3
    assert rebuilt.patience == 2
    assert rebuilt.monitor == "accuracy"
    assert rebuilt.monitor_mode == "max"
    assert rebuilt.accumulation_steps == 2


def test_overrides_beat_the_file(run_dir):
    TinyTrainer(num_epochs=7, run_dir=run_dir, use_progress_bar=False).save_config()
    rebuilt = TinyTrainer.from_config(run_dir, num_epochs=99, run_dir=run_dir)
    assert rebuilt.num_epochs == 99


def test_custom_metadata_is_kept_in_the_file_but_ignored_on_reload(run_dir):
    trainer = TinyTrainer(num_epochs=2, run_dir=run_dir, use_progress_bar=False)
    trainer.update_config({"experiment": "baseline"})
    trainer.save_config()

    config = json.loads(trainer.get_config_path().read_text(encoding="utf-8"))
    assert config["experiment"] == "baseline"
    TinyTrainer.from_config(run_dir, run_dir=run_dir)   # must not choke on the extra key


def test_the_phase_schedule_stays_out_of_the_file(trainer):
    """Phases are arguments to train(), not to the constructor: recording them would
    put something in config.json that from_config cannot pass back."""
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("audit", make_loader(8), every=3),
    )
    config = json.loads(trainer.get_config_path().read_text(encoding="utf-8"))
    assert "phases" not in config


def test_the_schedule_summarizes_how_each_phase_runs(trainer):
    summary = trainer.get_schedule_summary(
        Phase("train", make_loader(8), training=True),
        Phase("audit", make_loader(8), every=3),
    )
    assert summary == {"train": "training", "audit": "eval, every 3 epochs"}


def test_monitor_mode_is_validated(run_dir):
    with pytest.raises(ValueError, match="monitor_mode"):
        TinyTrainer(run_dir=run_dir, monitor_mode="lowest")


@pytest.mark.parametrize("bad", [0, -1])
def test_accumulation_steps_is_rejected_not_clamped(run_dir, bad):
    """Clamping would still record the *given* value, so config.json would lie."""
    with pytest.raises(ValueError, match="accumulation_steps must be >= 1"):
        TinyTrainer(run_dir=run_dir, accumulation_steps=bad)


def test_resume_false_clears_the_previous_run(run_dir):
    first = TinyTrainer(num_epochs=2, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    first.train(Phase("train", make_loader(8), training=True))
    assert first.get_latest_checkpoint_path().exists()

    fresh = TinyTrainer(
        num_epochs=2, learning_rate=0.1, run_dir=run_dir,
        resume=False, use_progress_bar=False,
    )
    fresh.prepare_training()
    assert fresh._current_epoch == 0, "a fresh run must not inherit the epoch counter"
    assert not fresh.get_latest_checkpoint_path().exists()
    assert fresh.get_config_path().exists(), "config.json is kept"
