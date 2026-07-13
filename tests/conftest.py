"""Shared fixtures: the smallest trainer that still exercises the real loop."""

from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train4all import BaseTrainer


class TinyTrainer(BaseTrainer):
    """A 3-class linear classifier — enough to drive every path in the loop."""

    def setup(self) -> None:
        self.net = nn.Linear(4, 3)
        self.set_models({"net": self.net})
        # `or 0.1` would be wrong: learning_rate=0.0 is falsy, and a test that means
        # "a run that cannot learn" would silently get a real learning rate instead.
        lr = 0.1 if self.learning_rate is None else self.learning_rate
        self.set_optimizer(torch.optim.SGD(self.get_trainable_params(), lr=lr))

    def compute_loss(self, batch: Any) -> torch.Tensor:
        x, y = batch
        logits = self.net(x)
        self.set_cache("logits", logits.detach())
        return F.cross_entropy(logits, y)

    def compute_metrics(self, batch: Any) -> dict[str, float]:
        _, y = batch
        preds = self.get_cache("logits").argmax(dim=1)
        return {"accuracy": (preds == y).float().mean().item()}


def make_loader(n: int, *, batch_size: int = 4, seed: int = 0) -> DataLoader:
    """A deterministic loader of ``n`` samples."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run"


@pytest.fixture
def trainer(run_dir):
    """A quiet trainer: no progress bar, no dashboard, everything under tmp_path."""
    return TinyTrainer(
        num_epochs=3,
        learning_rate=0.1,
        run_dir=run_dir,
        seed=0,
        use_progress_bar=False,
    )
