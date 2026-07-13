"""The documented public methods a user calls but the framework never does.

`backup_checkpoint`, `load_weights`, `unfreeze`, `print_gpu_temperature` and the
rest have no caller inside train4all — they exist for the user. That is exactly why
nothing else would notice them breaking.
"""

import torch
from conftest import TinyTrainer, make_loader

from train4all import Checkpoint, Phase

# ── Model management ──────────────────────────────────────────────────────────


def test_freeze_and_unfreeze_toggle_gradients(trainer):
    trainer.ensure_setup()
    trainer.freeze("net")
    assert not any(p.requires_grad for p in trainer.net.parameters())

    trainer.unfreeze("net")
    assert all(p.requires_grad for p in trainer.net.parameters())


def test_freeze_accepts_a_module_and_a_list(trainer):
    trainer.ensure_setup()
    trainer.freeze([trainer.net])
    assert not any(p.requires_grad for p in trainer.net.parameters())


def test_a_frozen_model_is_absent_from_the_trainable_params(trainer):
    trainer.ensure_setup()
    assert trainer.get_trainable_params()
    trainer.freeze("net")
    assert trainer.get_trainable_params() == []


def test_exclude_targets_drops_a_model_from_the_params(trainer):
    trainer.ensure_setup()
    assert trainer.get_trainable_params(exclude_targets="net") == []


def test_reset_parameters_reinitializes_in_place(trainer):
    trainer.ensure_setup()
    before = trainer.net.weight.detach().clone()
    trainer.reset_parameters("net")
    after = trainer.net.weight.detach()
    assert not torch.equal(before, after), "the weights were not re-initialized"


def test_get_model_summary_reports_frozen_and_partial(trainer):
    trainer.ensure_setup()
    assert "params" in trainer.get_model_summary()["net"]
    trainer.freeze("net")
    assert trainer.get_model_summary()["net"] == "frozen"


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def test_backup_checkpoint_copies_with_a_bak_suffix(trainer):
    trainer.train(Phase("train", make_loader(8), training=True))
    latest = trainer.get_latest_checkpoint_path()

    trainer.backup_checkpoint(latest)
    backup = latest.with_name(latest.name + ".bak")
    assert backup.exists()
    assert backup.read_bytes() == latest.read_bytes()


def test_backing_up_a_missing_file_warns_rather_than_raising(trainer):
    trainer.backup_checkpoint(trainer.run_dir / "nope.pth")   # must not raise


def test_load_weights_restores_the_model_but_not_the_epoch(trainer):
    trainer.train(Phase("train", make_loader(8), training=True))
    weights = trainer.net.weight.detach().clone()
    trainer.save_weights(trainer.run_dir / "w.pth")

    fresh = TinyTrainer(num_epochs=3, learning_rate=0.1,
                        run_dir=trainer.run_dir, use_progress_bar=False)
    fresh.load_weights(trainer.run_dir / "w.pth")

    torch.testing.assert_close(fresh.net.weight.detach(), weights)
    assert fresh._current_epoch == 0, "weights-only must not rewind the training state"


def test_key_map_renames_state_dict_keys_on_the_fly(trainer):
    """The documented escape hatch when the architecture's names changed."""
    trainer.ensure_setup()
    weights = trainer.net.weight.detach().clone()
    trainer.save_weights(trainer.run_dir / "w.pth")

    # Rewrite the saved keys so they no longer match the model...
    ckpt = Checkpoint.load(trainer.run_dir / "w.pth")
    ckpt["models"] = {"net": {f"old.{k}": v for k, v in ckpt.models["net"].items()}}
    ckpt.save(trainer.run_dir / "renamed.pth")

    # ...and map them back on load.
    fresh = TinyTrainer(run_dir=trainer.run_dir, use_progress_bar=False)
    fresh.load_weights(trainer.run_dir / "renamed.pth", key_map={"old.": ""})
    torch.testing.assert_close(fresh.net.weight.detach(), weights)


def test_save_interval_writes_periodic_checkpoints(run_dir):
    trainer = TinyTrainer(
        num_epochs=4, learning_rate=0.1, run_dir=run_dir,
        save_interval=2, use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(8), training=True))

    assert trainer.get_checkpoint_path("epoch_2").exists()
    assert trainer.get_checkpoint_path("epoch_4").exists()
    assert not trainer.get_checkpoint_path("epoch_3").exists()


def test_load_best_checkpoint_rewinds_but_load_best_weights_does_not(run_dir):
    trainer = TinyTrainer(
        num_epochs=4, learning_rate=0.5, run_dir=run_dir,
        monitor="loss", monitor_phase="val", use_progress_bar=False,
    )
    trainer.train(
        Phase("train", make_loader(16), training=True),
        Phase("val", make_loader(16)),
    )
    best_epoch = trainer._best_epoch

    trainer.load_best_weights()
    assert trainer._current_epoch == 4, "weights-only must not rewind"

    trainer.load_best_checkpoint()
    assert trainer._current_epoch == best_epoch, "a full restore must rewind to best"


# ── Building blocks for a custom loop ─────────────────────────────────────────


def test_execute_step_runs_one_batch(trainer):
    trainer.ensure_setup()
    phase = Phase("train", make_loader(8), training=True)
    batch = next(iter(phase.loader))

    metrics = trainer.execute_step(batch, phase, step=1)
    assert "loss" in metrics and "accuracy" in metrics


def test_epoch_iterator_advances_the_counter(trainer):
    seen = [(e, m) for e, m in trainer.epoch_iterator()]
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert trainer.is_training_complete()


def test_a_custom_loop_can_be_assembled_from_the_blocks(run_dir):
    trainer = TinyTrainer(num_epochs=2, learning_rate=0.1,
                          run_dir=run_dir, use_progress_bar=False)
    trainer.prepare_training()

    train = Phase("train", make_loader(8), training=True)
    val = Phase("val", make_loader(8))
    for _epoch, _max_epoch in trainer.epoch_iterator():
        trainer.execute_phase(train)
        val_metrics = trainer.execute_phase(val)
        trainer.finalize_train_epoch(val_metrics.get(trainer.monitor))
        trainer.save_artifacts()
        if trainer.should_stop_early():
            break

    assert trainer.get_latest_checkpoint_path().exists()
    assert len(trainer.get_epoch_metrics()["loss"]["val"]) == 2


# ── GPU utilities ─────────────────────────────────────────────────────────────


def test_print_gpu_temperature_is_quiet_without_cuda(trainer):
    trainer.print_gpu_temperature()      # warns rather than raising, off CUDA


def test_empty_cuda_cache_is_safe_without_cuda(trainer):
    trainer.empty_cuda_cache()
