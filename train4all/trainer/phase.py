"""What an epoch is made of — the declaration, and nothing that acts on one.

:class:`Phase` is frozen and :func:`schedule_summary` reads only the phases it is handed;
neither touches a trainer, a loader's contents, or a metric. That is what lets a schedule
be described before the first batch is drawn, and what keeps the loop itself free of any
built-in notion of "train" or "val" — those names are a caller's, and everything filed by
phase is filed under them.

``MetricFn`` is declared here for the same reason: a phase carries one, so the type
belongs beside the field rather than with the trainer that eventually calls it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

__all__ = ["MetricFn", "Phase", "schedule_summary"]

type MetricFn = Callable[[Any], dict[str, float]]


@dataclass(frozen=True, slots=True)
class Phase:
    """
    One pass over a :class:`~torch.utils.data.DataLoader` within an epoch.

    An epoch is a sequence of phases, and ``train()`` takes that sequence
    directly — so the schedule is data, not control flow. The canonical run is
    two phases::

        trainer.train(
            Phase("train", train_loader, training=True),
            Phase("val", val_loader),
        )

    and anything else is the same expression with more phases. To keep the
    training pass cheap, compute only the loss there and measure the expensive
    metrics periodically on a subset::

        trainer.train(
            Phase("train", train_loader, training=True, metric_fn=lambda _: {}),
            Phase("train_eval", train_subset_loader, every=5),
            Phase("val", val_loader),
        )

    ``metric_fn=lambda _: {}`` suppresses only the metric function — the trainer
    always records ``loss`` — so the training pass reports loss alone while
    ``train_eval`` reports the full metric set on a slice of the same data,
    every fifth epoch.

    Phases are compared and stored by identity of their fields, and the name is
    the key everything else is filed under: metric tables, plots, the dashboard
    legend, and the trainer's ``monitor_phase``. Names must be unique within a
    run.

    Attributes:
        name: Phase name. Used as the metric-table key, the plot legend entry,
            and the value ``monitor_phase`` selects on.
        loader: DataLoader iterated for this phase.
        training: Run the pass with gradients and step the optimizer. ``False``
            (the default) evaluates under ``torch.no_grad``, since most phases
            only measure.
        metric_fn: Per-batch metric function. ``None`` (the default) uses the
            trainer's ``compute_metrics``. Pass any callable to give this phase
            its own metrics — heavier report-only ones (as ``test()`` does with
            ``compute_test_metrics``), or ``lambda _: {}`` for none at all.
            Named for what it holds: a *function*, not the metric values every
            other ``metrics`` in the framework refers to.
        every: Run this phase only on epochs divisible by this number, so an
            expensive measurement need not be paid every epoch. Defaults to ``1``
            (every epoch). A ``monitor_phase`` with ``every > 1`` leaves the
            monitored metric absent on the epochs it skips — early stopping then
            does not advance, and ``ReduceLROnPlateau`` cannot step.
        record_steps: Record per-step metrics for this phase when the trainer's
            ``record_step_metrics`` is enabled. ``None`` (the default) follows
            ``training``: the training pass has a step curve worth watching, a
            short evaluation pass usually does not.
    """

    name: str
    loader: DataLoader
    training: bool = False
    metric_fn: MetricFn | None = None
    every: int = 1
    record_steps: bool | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A phase needs a non-empty name.")
        if self.every < 1:
            raise ValueError(f"Phase {self.name!r}: every must be >= 1; got {self.every}")

    @property
    def records_steps(self) -> bool:
        """Whether per-step metrics are recorded for this phase (subject to the
        trainer's ``record_step_metrics`` master switch)."""
        return self.training if self.record_steps is None else self.record_steps

    def runs_at(self, epoch: int) -> bool:
        """Whether this phase runs at the 1-based ``epoch``."""
        return epoch % self.every == 0


def schedule_summary(*phases: Phase) -> dict[str, str]:
    """
    The shape of one epoch as a dict: each phase name mapped to how it runs.

    A schedule is an argument to ``train()`` rather than trainer state, so it is
    not part of ``config.json`` — that file holds constructor arguments and must
    unpack straight back through ``from_config``. This is the shape's own summary,
    alongside the model's and the optimizer's, and it reads nothing but the phases
    themselves.

    Args:
        *phases: The phases of one epoch, in the order they run.
    """
    def describe(phase: Phase) -> str:
        kind = "training" if phase.training else "eval"
        return kind if phase.every == 1 else f"{kind}, every {phase.every} epochs"

    return {p.name: describe(p) for p in phases}
