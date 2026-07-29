"""
Metric recording and its artifacts for train4all.

``MetricStore`` is the single source of truth for what a run measured. It serves
both sides of the lifecycle:

  - **Record** — :meth:`record_epoch` and :meth:`record_step` append to the two
    tables as the loop runs, and :meth:`accumulate` / :meth:`average` carry the
    weighted arithmetic a phase averages its steps with, so ``BaseTrainer`` never
    reaches into a table itself.
  - **Read** — :meth:`load` wraps the JSON a finished run exported. This needs
    **no model, no subclass, and no trainer** — read a run straight off disk::

        from train4all import MetricStore

        store = MetricStore.load("run/metrics/epoch_metrics.json")
        store.print_summary()          # phases, metrics, how many points each holds
        store.epoch["loss"]["val"]     # [0.91, 0.74, 0.63, …]

Where a file goes is not decided here. The run directory's layout belongs to the
trainer that owns it, so every method that writes is handed the path (or, for the
one-plot-per-metric case, the function that builds one) rather than composing it.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from train4all.utils import (
    DEFAULT_KEY_WIDTH,
    MetricTable,
    Printer,
    get_metric_plot_title,
    print_dict_tree,
    save_curves_plot,
    write_json,
)

__all__ = ["MetricStore"]


class MetricStore:
    """
    The metrics a run recorded: two tables and everything done to them.

    Epoch-level values are appended once per phase per epoch, step-level values
    once per step. Both tables are :data:`~train4all.utils.MetricTable` —
    ``metric_name → phase_name → values`` — and both are exposed directly, so a
    caller that wants the raw numbers is never made to go through an accessor that
    only hands them back.

    What is worth recording is not decided here: whoever records applies its own
    settings first and hands over the result, the way ``BaseTrainer`` narrows a
    step's metrics by ``step_metric_names`` before calling :meth:`record_step`.

    Args:
        epoch: Initial epoch-level table. Empty when ``None``.
        step: Initial step-level table. Empty when ``None``.
        path: Source path, used only for display in :meth:`print_summary`.
    """

    def __init__(
        self,
        epoch: MetricTable | None = None,
        step: MetricTable | None = None,
        *,
        path: Path | str | None = None,
    ) -> None:
        self.epoch: MetricTable = epoch if epoch is not None else {}
        self.step: MetricTable = step if step is not None else {}
        self.path = Path(path) if path is not None else None

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, epoch: Path | str | None = None, step: Path | str | None = None) -> Self:
        """
        Read exported metric files back into a store.

        Each argument is a JSON file written by ``export_epoch_metrics`` /
        ``export_step_metrics``; either may be omitted. The epoch file comes first
        because reading one run's curves is what this is usually for.

        Args:
            epoch: Path to an exported epoch-level metrics JSON.
            step: Path to an exported step-level metrics JSON.

        Returns:
            A store holding whichever tables were given.

        Raises:
            FileNotFoundError: If a path was given and does not exist.
            ValueError: If a file does not hold a metric table.
        """
        return cls(
            cls._read(epoch),
            cls._read(step),
            path=epoch if epoch is not None else step,
        )

    @staticmethod
    def _read(path: Path | str | None) -> MetricTable | None:
        if path is None:
            return None
        table = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(table, dict):
            raise ValueError(
                f"{path} is not a train4all metric table "
                f"(expected an object, got {type(table).__name__})."
            )
        return table

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_epoch(self, metrics: dict[str, float], phase_name: str) -> None:
        """Append one epoch's values for *phase_name* to the epoch table."""
        self._append(self.epoch, metrics, phase_name)

    def record_step(self, metrics: dict[str, float], phase_name: str) -> None:
        """Append one step's values for *phase_name* to the step table."""
        self._append(self.step, metrics, phase_name)

    def clear(self) -> None:
        """Drop everything recorded so far."""
        self.epoch.clear()
        self.step.clear()

    @staticmethod
    def _append(target: MetricTable, metrics: dict[str, float], phase_name: str) -> None:
        for name, value in metrics.items():
            target.setdefault(name, {}).setdefault(phase_name, []).append(value)

    # ── Reading ───────────────────────────────────────────────────────────────

    def epoch_table(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> MetricTable:
        """The epoch table, optionally narrowed to some metrics and phases."""
        return self._filter(self.epoch, metric_names, phase_names)

    def step_table(
        self,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> MetricTable:
        """The step table, optionally narrowed to some metrics and phases."""
        return self._filter(self.step, metric_names, phase_names)

    @staticmethod
    def _filter(
        metrics: MetricTable,
        metric_names: list[str] | None,
        phase_names: list[str] | None,
    ) -> MetricTable:
        """A copy holding only the named metrics and phases, and only non-empty series."""
        result: MetricTable = {}
        for name, phase_dict in metrics.items():
            if metric_names is not None and name not in metric_names:
                continue
            filtered = {
                phase_name: values
                for phase_name, values in phase_dict.items()
                if (phase_names is None or phase_name in phase_names) and values
            }
            if filtered:
                result[name] = filtered
        return result

    # ── Weighted Averaging ────────────────────────────────────────────────────
    # The arithmetic a phase averages its steps with: `Σ wᵢxᵢ / Σ wᵢ`, split so the
    # loop can accumulate as it goes and divide once at the end.

    @staticmethod
    def accumulate(
        accumulated: dict[str, float],
        metrics: dict[str, float],
        weight: float,
    ) -> None:
        """Add *metrics* into *accumulated*, each value scaled by *weight*."""
        for name, value in metrics.items():
            accumulated[name] = accumulated.get(name, 0.0) + value * weight

    @staticmethod
    def average(accumulated: dict[str, float], total_weight: float) -> dict[str, float]:
        """Divide an accumulation by its total weight; empty when nothing was weighed."""
        if total_weight == 0:
            return {}
        return {k: v / total_weight for k, v in accumulated.items()}

    # ── Artifacts ─────────────────────────────────────────────────────────────

    def export_epoch(
        self,
        path: Path | str,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
        *,
        print_fn: Printer | None = None,
    ) -> Path:
        """Write the epoch table to *path* as JSON. See :func:`~train4all.utils.write_json`."""
        return self._export(self.epoch_table(metric_names, phase_names), path, print_fn)

    def export_step(
        self,
        path: Path | str,
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
        *,
        print_fn: Printer | None = None,
    ) -> Path:
        """Write the step table to *path* as JSON. See :func:`~train4all.utils.write_json`."""
        return self._export(self.step_table(metric_names, phase_names), path, print_fn)

    @staticmethod
    def _export(table: MetricTable, path: Path | str, print_fn: Printer | None) -> Path:
        write_json(path, table, label="metrics", print_fn=print_fn)
        return Path(path)

    def save_epoch_plots(
        self,
        path_for: Callable[..., Path],
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> None:
        """
        Save one curve plot per epoch-level metric, every phase on the same axes.

        Args:
            path_for: Builds the output path, called as ``path_for(metric_name)``.
            metric_names: Metrics to plot. ``None`` plots all.
            phase_names: Phases to include. ``None`` includes all.
        """
        self._save_plots(
            self.epoch_table(metric_names, phase_names),
            path_for=path_for,
            xlabel="epoch",
            split_phases=False,
        )

    def save_step_plots(
        self,
        path_for: Callable[..., Path],
        metric_names: list[str] | None = None,
        phase_names: list[str] | None = None,
    ) -> None:
        """
        Save one curve plot per step-level metric *per phase*.

        Steps are counted within a phase, so two phases share no x-axis and cannot
        share a plot — unlike the epoch-level curves, which are all indexed by epoch.

        Args:
            path_for: Builds the output path, called as
                ``path_for(metric_name, phase_name=...)``.
            metric_names: Metrics to plot. ``None`` plots all.
            phase_names: Phases to include. ``None`` includes all.
        """
        self._save_plots(
            self.step_table(metric_names, phase_names),
            path_for=path_for,
            xlabel="step",
            title_prefix="step-level",
            split_phases=True,
        )

    @staticmethod
    def _save_plots(
        table: MetricTable,
        *,
        path_for: Callable[..., Path],
        xlabel: str,
        title_prefix: str | None = None,
        split_phases: bool = False,
    ) -> None:
        """Plot *table*, which :meth:`_filter` has already stripped of empty series.

        That is why nothing here skips one: an empty curve cannot arrive, because both
        callers read their table through :meth:`epoch_table` / :meth:`step_table`.
        """
        for metric_name, phase_dict in table.items():
            if split_phases:
                for phase_name, values in phase_dict.items():
                    save_curves_plot(
                        curves={phase_name: values},
                        path=path_for(metric_name, phase_name=phase_name),
                        title=get_metric_plot_title(
                            metric_name, phase_name=phase_name, prefix=title_prefix,
                        ),
                        xlabel=xlabel,
                        ylabel=metric_name,
                    )
            else:
                save_curves_plot(
                    curves=phase_dict,
                    path=path_for(metric_name),
                    title=get_metric_plot_title(metric_name, prefix=title_prefix),
                    xlabel=xlabel,
                    ylabel=metric_name,
                )

    # ── Inspection ────────────────────────────────────────────────────────────

    def latest(self) -> dict[str, str]:
        """Each epoch-level metric's most recent value per phase, formatted for display."""
        return {
            metric_name: "  ".join(
                f"{phase_name}={values[-1]:.4f}" if values else f"{phase_name}=N/A"
                for phase_name, values in phase_dict.items()
            ) or "N/A"
            for metric_name, phase_dict in self.epoch.items()
        }

    def metric_names(self) -> list[str]:
        """Sorted union of metric names across both tables."""
        return sorted(set(self.epoch) | set(self.step))

    def summary(self) -> dict[str, Any]:
        """A nested, display-ready overview of what has been recorded."""
        out: dict[str, Any] = {}
        for label, table in (("epoch", self.epoch), ("step", self.step)):
            if table:
                out[label] = {
                    metric_name: ", ".join(
                        f"{phase_name} ({len(values)})"
                        for phase_name, values in phase_dict.items()
                    )
                    for metric_name, phase_dict in table.items()
                }
        return out

    def print_summary(
        self, *, key_width: int = DEFAULT_KEY_WIDTH, print_fn: Printer | None = None,
    ) -> None:
        """
        Pretty-print :meth:`summary` as a tree.

        Args:
            key_width: Column width for leaf keys.
            print_fn: Output function. Defaults to the built-in ``print``.
        """
        source = self.path.name if self.path is not None else "metrics"
        print_dict_tree(
            self.summary(),
            header=f"📈 Metrics: {source}",
            key_width=key_width,
            print_fn=print_fn,
        )

    # ── Dunder Helpers ────────────────────────────────────────────────────────

    def __bool__(self) -> bool:
        return bool(self.epoch or self.step)

    def __repr__(self) -> str:
        fields = [f"epoch={list(self.epoch)!r}", f"step={list(self.step)!r}"]
        if self.path is not None:
            fields.append(f"path={self.path.as_posix()!r}")
        return f"MetricStore({', '.join(fields)})"
