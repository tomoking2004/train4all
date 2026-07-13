"""Gradient accumulation must reconstruct the true full-batch gradient.

The loop weights each micro-batch's loss by ``get_batch_weight`` and divides the
accumulated gradient by the cycle's total weight — Σ wᵢ∇Lᵢ / Σ wᵢ. When the weight
matches the loss's denominator, that is *exactly* the gradient of the loss over the
whole effective batch, even when the micro-batches have different sizes.

A plain ``loss / N`` would silently over-weight the short batches, and nothing in a
training curve would ever reveal it. Hence this test.
"""

from typing import Any

import torch
import torch.nn.functional as F
from conftest import TinyTrainer
from torch.utils.data import DataLoader, TensorDataset

from train4all import Phase

# One fixed dataset of 8 samples, and one fixed initialization, shared by every
# trainer below — so the only thing that varies is how the batches are cut.
_G = torch.Generator().manual_seed(7)
_X = torch.randn(8, 4, generator=_G)
_Y = torch.randint(0, 3, (8,), generator=_G)
_W0 = torch.randn(3, 4, generator=_G) * 0.1
_B0 = torch.zeros(3)


def _loader(batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(_X, _Y), batch_size=batch_size, shuffle=False)


class GradCapture(TinyTrainer):
    """Snapshots the gradient the optimizer is about to consume."""

    def setup(self) -> None:
        super().setup()
        with torch.no_grad():
            self.net.weight.copy_(_W0)
            self.net.bias.copy_(_B0)
        self.captured: list[torch.Tensor] = []
        self.steps = 0

    def on_before_optimizer_step(self) -> None:
        self.steps += 1
        self.captured = [
            p.grad.detach().clone()
            for group in self._optimizer.param_groups
            for p in group["params"]
        ]


def _run(run_dir, *, batch_size: int, accumulation_steps: int, **kw) -> GradCapture:
    trainer = GradCapture(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir,
        accumulation_steps=accumulation_steps, use_progress_bar=False, **kw,
    )
    trainer.train(Phase("train", _loader(batch_size), training=True))
    return trainer


def test_uneven_microbatches_match_the_full_batch_gradient(tmp_path):
    # One step over all 8 samples: the reference gradient.
    full = _run(tmp_path / "full", batch_size=8, accumulation_steps=1)
    # The same 8 samples, cut 5 + 3, accumulated over two micro-batches.
    split = _run(tmp_path / "split", batch_size=5, accumulation_steps=2)

    assert full.steps == 1
    assert split.steps == 1, "the accumulation cycle must fire exactly one update"

    for reference, accumulated in zip(full.captured, split.captured, strict=True):
        torch.testing.assert_close(accumulated, reference, rtol=1e-5, atol=1e-6)


def test_even_microbatches_match_too(tmp_path):
    full = _run(tmp_path / "full", batch_size=8, accumulation_steps=1)
    split = _run(tmp_path / "split", batch_size=4, accumulation_steps=2)

    for reference, accumulated in zip(full.captured, split.captured, strict=True):
        torch.testing.assert_close(accumulated, reference, rtol=1e-5, atol=1e-6)


def test_a_short_tail_cycle_is_flushed_not_dropped(tmp_path):
    """8 samples in batches of 3 -> 3+3+2, with accumulation_steps=2.

    The cycle closes at step 2, and the leftover step 3 is the epoch's last, so it
    must be flushed rather than silently discarded.
    """
    trainer = _run(tmp_path / "tail", batch_size=3, accumulation_steps=2)
    assert trainer.steps == 2, "the final partial cycle was dropped"


class TokenWeighted(GradCapture):
    """Loss is a mean over *tokens*, so the weight must be the token count."""

    def get_batch_weight(self, batch: Any) -> int:
        _, y = batch
        return int(y.numel())

    def compute_loss(self, batch: Any) -> torch.Tensor:
        x, y = batch
        logits = self.net(x)
        self.set_cache("logits", logits.detach())
        return F.cross_entropy(logits, y, reduction="mean")


def test_get_batch_weight_override_is_what_normalizes_the_gradient(tmp_path):
    """Overriding the weight must still reconstruct the full-batch gradient."""
    full = TokenWeighted(
        num_epochs=1, learning_rate=0.1, run_dir=tmp_path / "full",
        accumulation_steps=1, use_progress_bar=False,
    )
    full.train(Phase("train", _loader(8), training=True))

    split = TokenWeighted(
        num_epochs=1, learning_rate=0.1, run_dir=tmp_path / "split",
        accumulation_steps=2, use_progress_bar=False,
    )
    split.train(Phase("train", _loader(5), training=True))

    for reference, accumulated in zip(full.captured, split.captured, strict=True):
        torch.testing.assert_close(accumulated, reference, rtol=1e-5, atol=1e-6)
