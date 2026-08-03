"""Every documented `Raises:` must actually raise, and with the type it promises.

The rest of the suite covers the exception paths reached through the loop; these are
the ones only a user hits directly, so nothing else would notice them rotting.
"""

import pytest
import torch
from conftest import TinyTrainer, make_loader

from train4all import Checkpoint, Phase


def test_an_unregistered_model_cannot_be_frozen(trainer):
    trainer.ensure_setup()
    with pytest.raises(ValueError, match="not registered"):
        trainer.freeze("nope")


def test_a_target_that_is_neither_a_name_nor_a_module_is_a_type_error(trainer):
    trainer.ensure_setup()
    with pytest.raises(TypeError, match=r"model name or nn\.Module"):
        trainer.freeze(123)


def test_an_unknown_amp_mode_is_rejected(run_dir):
    with pytest.raises(ValueError, match="amp must be a bool"):
        TinyTrainer(run_dir=run_dir, amp="fp8")


def test_epoch_iterator_needs_num_epochs(run_dir):
    trainer = TinyTrainer(run_dir=run_dir, use_progress_bar=False)
    with pytest.raises(ValueError, match="num_epochs"):
        list(trainer.epoch_iterator())


def test_training_without_an_optimizer_is_a_runtime_error(run_dir):
    class NoOptimizer(TinyTrainer):
        def setup(self) -> None:
            self.net = torch.nn.Linear(4, 3)
            self.set_models({"net": self.net})      # no set_optimizer

    trainer = NoOptimizer(num_epochs=1, run_dir=run_dir, use_progress_bar=False)
    with pytest.raises(RuntimeError, match="optimizer is required"):
        trainer.train(Phase("train", make_loader(8), training=True))


def test_reduce_lr_on_plateau_without_its_metric_says_so(run_dir):
    """It needs the monitored metric every epoch; a schedule without that phase
    cannot feed it, and the error has to name the phase rather than blow up inside torch."""

    class Plateau(TinyTrainer):
        def setup(self) -> None:
            super().setup()
            self.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    trainer = Plateau(
        num_epochs=2, learning_rate=0.1, run_dir=run_dir,
        monitor_phase="val", use_progress_bar=False,
    )
    with pytest.raises(ValueError, match="ReduceLROnPlateau requires"):
        trainer.train(Phase("train", make_loader(8), training=True))   # no 'val' phase


def test_a_scheduler_that_needs_no_metric_just_steps(run_dir):
    class Cosine(TinyTrainer):
        def setup(self) -> None:
            super().setup()
            self.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=2)

    trainer = Cosine(num_epochs=2, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    trainer.train(Phase("train", make_loader(8), training=True))
    assert trainer._optimizer is not None, "setup() bound no optimizer"
    assert trainer._optimizer.param_groups[0]["lr"] < 0.1, "the scheduler never stepped"


def test_loading_a_missing_checkpoint_is_a_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Checkpoint.load(tmp_path / "nope.pth")


def test_loading_something_that_is_not_a_checkpoint_says_so(tmp_path):
    path = tmp_path / "not-a-checkpoint.pth"
    torch.save([1, 2, 3], path)                     # a list, not a checkpoint dict
    with pytest.raises(ValueError, match="not a train4all checkpoint"):
        Checkpoint.load(path)


def test_a_missing_checkpoint_warns_rather_than_raising_through_the_trainer(trainer):
    """The trainer's own load path is forgiving: it warns and carries on."""
    trainer.ensure_setup()
    trainer.load_checkpoint(trainer.run_dir / "nope.pth")      # must not raise
    assert trainer._current_epoch == 0
