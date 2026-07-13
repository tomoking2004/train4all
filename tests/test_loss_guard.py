"""The non-finite-loss guard must fire before the model is damaged.

A NaN loss makes every gradient NaN, and one optimizer step on those writes NaN
into every parameter. The guard therefore has to precede the step: raising after
it would report a divergence over a model it had already destroyed, leaving
nothing to resume from.
"""

from typing import Any

import pytest
import torch
from conftest import TinyTrainer, make_loader

from train4all import Phase


class DivergingTrainer(TinyTrainer):
    """Its loss is non-finite from the very first batch."""

    bad_value = float("nan")

    def setup(self) -> None:
        super().setup()
        with torch.no_grad():          # start from known-finite weights
            self.net.weight.fill_(0.5)
            self.net.bias.fill_(0.0)

    def compute_loss(self, batch: Any) -> torch.Tensor:
        x, _ = batch
        return self.net(x).sum() * self.bad_value


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_loss_raises_with_the_model_intact(run_dir, bad_value):
    trainer = DivergingTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False,
    )
    trainer.bad_value = bad_value
    trainer.ensure_setup()

    before = trainer.net.weight.detach().clone()
    assert torch.isfinite(before).all()

    with pytest.raises(RuntimeError, match="Invalid loss value"):
        trainer.train(Phase("train", make_loader(8), training=True))

    after = trainer.net.weight.detach()
    assert torch.isfinite(after).all(), "the optimizer stepped on non-finite gradients"
    torch.testing.assert_close(after, before)


def test_the_guard_also_holds_under_gradient_accumulation(run_dir):
    """A mid-cycle NaN must not survive to the cycle's optimizer step either."""
    trainer = DivergingTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir,
        accumulation_steps=4, use_progress_bar=False,
    )
    trainer.ensure_setup()
    before = trainer.net.weight.detach().clone()

    with pytest.raises(RuntimeError, match="Invalid loss value"):
        trainer.train(Phase("train", make_loader(16), training=True))

    assert torch.isfinite(trainer.net.weight).all()
    torch.testing.assert_close(trainer.net.weight.detach(), before)


def test_a_finite_loss_is_untouched(trainer):
    """The guard is a guard, not a filter — normal training still records its loss."""
    trainer.train(Phase("train", make_loader(16), training=True))
    losses = trainer.get_epoch_metrics()["loss"]["train"]
    assert len(losses) == trainer.num_epochs
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
