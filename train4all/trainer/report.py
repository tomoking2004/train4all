"""The console voice of a run for train4all.

``Report`` owns how a run names itself on the console: the header each banner
carries, the column its keys line up in, and the shape a metrics table takes. It
reads no trainer state — every method is handed what it prints — so what a banner
*says* stays with whoever owns the state it describes, and only how it *looks*
lives here.

That split is why two kinds of banner meet in this file. Most are handed a dict
they do not name — a public summary like ``get_env_summary()``, or the metrics of
one phase — and add a header to it. The rest are handed facts instead, because
there is no such dict to be handed: nothing returns a status or an optimization
summary, so their rows are display vocabulary exactly as their header is, and both
are written here.
"""

from typing import Any

from train4all.utils import DEFAULT_KEY_WIDTH, Printer, print_dict_tree, separator_rule

__all__ = ["Report"]

_ABSENT = "-"
"""What a row shows when it has nothing to show — no object, no best yet, nothing recorded."""


class Report:
    """
    How a run names itself on the console.

    Args:
        print_fn: Output function for every line. Defaults to the built-in
            ``print``. :class:`~train4all.BaseTrainer` passes its own ``print``,
            which reads ``self.logger`` when called, so a logger swapped in after
            construction still takes effect.
        key_width: Column width the leaf keys of every tree line up in.
    """

    def __init__(
        self,
        print_fn: Printer | None = None,
        *,
        key_width: int = DEFAULT_KEY_WIDTH,
    ) -> None:
        self.print_fn = print_fn
        self.key_width = key_width

    # ── Banners over a dict the caller brings ─────────────────────────────────

    def env(self, summary: dict[str, Any]) -> None:
        """Print the environment summary — the machine, the runtime, and the run's own rows."""
        self.tree(summary, header="🖥️  Environment")

    def config(self, config: dict[str, Any]) -> None:
        """Print the trainer's configuration."""
        self.tree(config, header="⚙️  Configuration")

    def model(self, summary: dict[str, str]) -> None:
        """Print the registered models and what each holds."""
        self.tree(summary, header="🧠 Model")

    def schedule(self, summary: dict[str, str]) -> None:
        """Print the shape of one epoch — the phases, in the order they run."""
        self.tree(summary, header="🗓️  Schedule")

    def metrics(self, metrics: dict[str, float], phase_name: str) -> None:
        """
        Print one phase's metrics as a flat table.

        Args:
            metrics: Mapping of metric name to value.
            phase_name: Phase label shown in the header.
        """
        self.tree(metrics, header=f"📊 {phase_name.capitalize()}", max_depth=0)

    # ── Banners this file names the rows of ───────────────────────────────────

    def optimization(
        self,
        optimizer: object | None,
        scheduler: object | None,
        *,
        accumulation_steps: int = 1,
    ) -> None:
        """
        Print the optimizer, the scheduler, and the gradient-accumulation setting.

        Args:
            optimizer: The registered optimizer, or ``None`` when none is set.
            scheduler: The registered scheduler, or ``None`` when none is set.
            accumulation_steps: Steps per optimizer update. Its row appears only
                above ``1``, the point at which the setting is doing something.
        """
        tree: dict[str, str] = {
            "Optimizer": self._class_name(optimizer),
            "Scheduler": self._class_name(scheduler),
        }
        if accumulation_steps > 1:
            tree["Grad accumulation"] = f"{accumulation_steps} steps"
        self.tree(tree, header="⚡ Optimization")

    def status(
        self,
        *,
        completed_epochs: int,
        monitor: str,
        best_value: float,
        best_epoch: int | None,
        stagnant_epochs: int,
        latest: dict[str, str],
    ) -> None:
        """
        Print where the run has got to.

        Args:
            completed_epochs: Epochs finished so far.
            monitor: What the best value is a value of, as ``"<phase> <metric>"``.
            best_value: The best monitored value seen.
            best_epoch: The epoch it was seen at, or ``None`` while nothing has
                been monitored yet — which is what decides whether a best exists
                to print at all, since ``best_value`` starts at a sentinel.
            stagnant_epochs: Epochs since the last improvement.
            latest: Each metric's most recent value per phase, empty when nothing
                has been recorded.
        """
        best = f"{best_value:.4f}  (epoch {best_epoch})" if best_epoch is not None else _ABSENT
        tree: dict[str, Any] = {
            "Completed epochs":   completed_epochs,
            f"Best {monitor}":    best,
            "Stagnant epochs":    stagnant_epochs,
            "Last epoch metrics": latest or _ABSENT,
        }
        self.tree(tree, header="📋 Status")

    # ── Primitives ────────────────────────────────────────────────────────────

    def tree(
        self,
        tree: dict[str, Any],
        *,
        header: str | None = None,
        max_depth: int | None = None,
    ) -> None:
        """
        Print a mapping as a tree, in this report's column.

        Args:
            tree: Mapping to display; nested mappings are expanded recursively.
            header: Title shown above the tree. Omit for a tree with no header.
            max_depth: Deepest nesting level to expand. ``None`` is unlimited.
        """
        print_dict_tree(
            tree,
            max_depth=max_depth,
            header=header,
            key_width=self.key_width,
            trailing_newline=True,
            print_fn=self.print_fn,
        )

    def rule(self) -> None:
        """Print the rule a header is underlined with, for a caller that printed its own."""
        printer = self.print_fn or print
        printer(separator_rule(self.key_width))

    @staticmethod
    def _class_name(obj: object | None) -> str:
        """An object's class name, or the absent marker where there is no object."""
        return obj.__class__.__name__ if obj is not None else _ABSENT
