"""What `setup()` registers: given a class, the trainer supplies what it already knows."""

from typing import Any

import pytest
import torch
import torch.nn as nn
from conftest import TinyTrainer

from train4all import BaseTrainer


class TwoModels(TinyTrainer):
    """Two registered models, so that naming one of them — by `targets`, by
    `exclude_targets`, or as a per-group `learning_rate` key — says something."""

    def setup(self) -> None:
        self.net = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)
        self.set_models({"net": self.net, "head": self.head})
        self.set_optimizer(torch.optim.SGD)


def built(trainer: BaseTrainer, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Register an optimizer on an already-set-up trainer, spying on what it was handed:
    the `(params, kwargs)` the trainer supplied to the class, which is the whole of what
    the class form promises — so every test below reads it the same way."""
    calls: list[tuple[Any, dict[str, Any]]] = []

    def factory(params: Any, **passed: Any) -> torch.optim.Optimizer:
        calls.append((params, passed))
        return torch.optim.SGD(params, lr=0.1)

    trainer.set_optimizer(factory, **kwargs)
    [call] = calls
    return call


# ── The optimizer ─────────────────────────────────────────────────────────────


def test_the_class_form_supplies_the_parameters_and_the_learning_rate(trainer):
    trainer.ensure_setup()
    params, kwargs = built(trainer)

    assert params == trainer.get_trainable_params()
    assert kwargs == {"lr": trainer.learning_rate}


def test_targets_and_exclude_targets_restrict_the_parameters(run_dir):
    trainer = TwoModels(learning_rate=0.1, run_dir=run_dir, use_progress_bar=False)
    trainer.ensure_setup()
    head = trainer.get_trainable_params("head")

    assert built(trainer, targets="head")[0] == head
    assert built(trainer, exclude_targets="net")[0] == head


def test_further_arguments_reach_the_class_and_an_explicit_lr_wins(trainer):
    trainer.ensure_setup()
    _, kwargs = built(trainer, lr=0.5, momentum=0.9)

    assert kwargs == {"lr": 0.5, "momentum": 0.9}


def test_an_unset_learning_rate_is_dropped_rather_than_passed_as_none(run_dir):
    """The whole point of leaving it unset: learning-rate-free optimizers take no lr."""
    trainer = TinyTrainer(run_dir=run_dir, use_progress_bar=False)
    trainer.ensure_setup()

    _, kwargs = built(trainer)
    assert "lr" not in kwargs


def test_a_per_group_learning_rate_becomes_one_param_group_per_model(run_dir):
    trainer = TwoModels(
        learning_rate={"net": 1e-4, "head": 1e-3},
        run_dir=run_dir, use_progress_bar=False,
    )
    trainer.ensure_setup()

    groups = trainer._optimizer.param_groups
    assert [g["lr"] for g in groups] == [1e-4, 1e-3]
    assert groups[0]["params"] == trainer.get_trainable_params("net")


def test_an_instance_is_stored_untouched(trainer):
    trainer.ensure_setup()
    optimizer = torch.optim.SGD(trainer.get_trainable_params(), lr=0.3)
    trainer.set_optimizer(optimizer)

    assert trainer._optimizer is optimizer


def test_an_instance_refuses_the_arguments_that_only_build_one(trainer):
    trainer.ensure_setup()
    optimizer = torch.optim.SGD(trainer.get_trainable_params(), lr=0.3)

    with pytest.raises(TypeError, match="ready-made optimizer"):
        trainer.set_optimizer(optimizer, targets="net")
    with pytest.raises(TypeError, match="ready-made optimizer"):
        trainer.set_optimizer(optimizer, weight_decay=0.01)


def test_a_per_group_learning_rate_refuses_targets(run_dir):
    """The dict keys already name the models; a second way to say it could only disagree."""
    trainer = TwoModels(
        learning_rate={"net": 1e-4}, run_dir=run_dir, use_progress_bar=False,
    )
    trainer.ensure_setup()

    with pytest.raises(TypeError, match="per-group learning_rate"):
        trainer.set_optimizer(torch.optim.SGD, targets="head")


# ── The scheduler ─────────────────────────────────────────────────────────────


def test_the_class_form_gets_the_registered_optimizer(trainer):
    trainer.ensure_setup()
    trainer.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=2)

    assert trainer._scheduler.optimizer is trainer._optimizer
    assert trainer._scheduler.T_max == 2


def test_a_scheduler_class_before_the_optimizer_says_so(run_dir):
    class NoOptimizer(TinyTrainer):
        def setup(self) -> None:
            self.net = nn.Linear(4, 3)
            self.set_models({"net": self.net})
            self.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=2)

    trainer = NoOptimizer(run_dir=run_dir, use_progress_bar=False)
    with pytest.raises(RuntimeError, match="needs the optimizer it drives"):
        trainer.ensure_setup()


def test_a_scheduler_instance_refuses_further_arguments(trainer):
    trainer.ensure_setup()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer._optimizer, T_max=2)

    with pytest.raises(TypeError, match="ready-made scheduler"):
        trainer.set_scheduler(scheduler, T_max=3)
