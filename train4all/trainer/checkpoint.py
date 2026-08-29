"""
Checkpoint schema and inspection for train4all.

``Checkpoint`` is the single source of truth for the on-disk checkpoint format.
It serves both sides of the lifecycle:

  - **Write** — :meth:`Checkpoint.build` assembles a checkpoint and
    :meth:`Checkpoint.save` writes it, so the key names, the format version, and
    :func:`torch.save` itself all live in one place — ``BaseTrainer`` never
    touches the raw dict.
  - **Read** — :meth:`Checkpoint.load` wraps a saved file and exposes its
    contents through typed accessors. This needs **no model, no subclass, and no
    abstract methods** — load a ``.pth`` and inspect it directly::

        from train4all import Checkpoint

        ckpt = Checkpoint.load("run/checkpoints/best.pth")
        ckpt.print_summary()                 # version, models, components, training state, metrics
        weights = ckpt.models["encoder"]     # raw state dict, no architecture required
        print(ckpt.training_state["best_epoch"])
"""

from pathlib import Path
from typing import Any, Self

import torch

from train4all.utils import DEFAULT_KEY_WIDTH, MetricTable, Printer, atomic_replace, print_dict_tree

__all__ = ["Checkpoint"]


def _describe(value: Any) -> str:
    """Render a checkpoint ``extras`` value as a short, single-line preview."""
    if value is None or isinstance(value, (str, int, float, bool)):
        text = repr(value)
        return text if len(text) <= 60 else f"{text[:57]}..."
    if isinstance(value, (list, tuple, set, dict)):
        return f"{type(value).__name__}({len(value)})"
    return type(value).__name__


class Checkpoint:
    """
    A train4all checkpoint: the single owner of the on-disk format.

    Build one for saving with :meth:`build`, write it with :meth:`save`, or wrap
    a saved file with :meth:`load` (or an already-loaded dict via the
    constructor). Accessors normalize the schema — including legacy key names —
    so callers never touch raw dict keys.

    Args:
        raw: The checkpoint dict, as written by :meth:`save` / ``torch.save``.
        path: Source path, used only for display in :meth:`print_summary`.
    """

    # ── On-Disk Schema ────────────────────────────────────────────────────────
    # Bump when the saved layout changes incompatibly. Read paths stay tolerant
    # of older files (see ``training_state``), so this is informational metadata.
    VERSION: str = "1.1"

    def __init__(self, raw: dict[str, Any], *, path: Path | str | None = None) -> None:
        self._raw = raw
        self.path = Path(path) if path is not None else None

    # ── Construction & Persistence ────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path | str, *, map_location: Any = "cpu") -> Self:
        """
        Load a checkpoint file from disk.

        Args:
            path: Path to the ``.pth`` file.
            map_location: Device mapping forwarded to :func:`torch.load`.
                Defaults to ``"cpu"`` so a checkpoint inspects anywhere, even on
                a host without the GPU it was trained on.

        Returns:
            The loaded :class:`Checkpoint`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file is not a train4all checkpoint dict.
        """
        path = Path(path)
        raw = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError(
                f"{path} is not a train4all checkpoint "
                f"(expected a dict, got {type(raw).__name__})."
            )
        return cls(raw, path=path)

    @classmethod
    def build(
        cls,
        *,
        models: dict[str, Any],
        extras: dict[str, Any],
        weights_only: bool = False,
        optimizer: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
        scaler: dict[str, Any] | None = None,
        training_state: dict[str, Any] | None = None,
        metrics: dict[str, MetricTable] | None = None,
    ) -> Self:
        """Assemble a checkpoint, ready to :meth:`save`.

        A weights-only checkpoint carries just ``models`` and ``extras``; a full
        checkpoint adds optimizer, scheduler, scaler, training state, and metrics.
        Custom state can still be attached before saving via item assignment
        (``ckpt["ema"] = ...``), e.g. from ``on_save_checkpoint``.
        """
        data: dict[str, Any] = {"version": cls.VERSION, "models": models, "extras": extras}
        if not weights_only:
            data.update({
                "optimizer":      optimizer,
                "scheduler":      scheduler,
                "scaler":         scaler,
                # Stored as {}, not None: the ``training_state`` / ``metrics``
                # properties are typed as always returning a dict.
                "training_state": training_state or {},
                "metrics":        metrics or {},
            })
        return cls(data)

    def save(self, path: Path | str) -> None:
        """
        Serialize the checkpoint to *path*, creating parent directories as needed.

        Writes the underlying dict rather than the wrapper, so the on-disk format
        stays a plain ``torch.save`` payload that any reader — including an older
        train4all — can load.

        The bytes go through a temporary beside *path* (:func:`atomic_replace`),
        so an interrupted save leaves the previous file (or none) in place rather
        than a truncated one, and a failed save leaves no temporary behind.

        Args:
            path: Destination ``.pth`` file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_replace(path) as tmp:
            torch.save(self._raw, tmp)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def raw(self) -> dict[str, Any]:
        """The underlying checkpoint dict."""
        return self._raw

    @property
    def version(self) -> str | None:
        """Saved format version, or ``None`` for a file without one."""
        return self._raw.get("version")

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        """Model name → state dict. Empty when none were saved."""
        return self._raw.get("models", {})

    @property
    def extras(self) -> dict[str, Any]:
        """Custom static metadata embedded via ``update_checkpoint_extras``."""
        return self._raw.get("extras", {})

    @property
    def optimizer_state(self) -> dict[str, Any] | None:
        """Optimizer state dict, or ``None`` (absent / weights-only checkpoint)."""
        return self._raw.get("optimizer")

    @property
    def scheduler_state(self) -> dict[str, Any] | None:
        """Scheduler state dict, or ``None`` (absent / weights-only checkpoint)."""
        return self._raw.get("scheduler")

    @property
    def scaler_state(self) -> dict[str, Any] | None:
        """AMP ``GradScaler`` state dict, or ``None`` (absent / weights-only checkpoint)."""
        return self._raw.get("scaler")

    @property
    def training_state(self) -> dict[str, Any]:
        """Resume state: ``current_epoch``, ``best_metric``, ``best_epoch``, and
        ``epochs_no_improve``.

        Legacy key names (``best_val_loss`` / ``best_val_epoch`` from older files)
        are normalized to the canonical names without overwriting a canonical key
        that is already present. Missing entries are simply absent, so callers can
        fall back to their own defaults.
        """
        state = dict(self._raw.get("training_state", {}))
        if "best_metric" not in state and "best_val_loss" in state:
            state["best_metric"] = state["best_val_loss"]
        if "best_epoch" not in state and "best_val_epoch" in state:
            state["best_epoch"] = state["best_val_epoch"]
        return state

    @property
    def metrics(self) -> dict[str, MetricTable]:
        """Recorded metric tables, keyed ``"epoch_metrics"`` / ``"step_metrics"``."""
        return self._raw.get("metrics", {})

    # ── Inspection ────────────────────────────────────────────────────────────

    def model_summary(self) -> dict[str, dict[str, int]]:
        """Per-model ``{"parameters": total_numel, "tensors": count}``."""
        summary: dict[str, dict[str, int]] = {}
        for name, state_dict in self.models.items():
            params = sum(
                int(t.numel()) for t in state_dict.values() if hasattr(t, "numel")
            )
            summary[name] = {"parameters": params, "tensors": len(state_dict)}
        return summary

    def metric_names(self) -> list[str]:
        """Sorted union of metric names across the epoch and step tables."""
        names: set[str] = set()
        for table in self.metrics.values():
            if isinstance(table, dict):
                names.update(table)
        return sorted(names)

    def summary(self) -> dict[str, Any]:
        """A nested, display-ready overview of the checkpoint's contents."""
        out: dict[str, Any] = {"version": self.version or "unknown"}
        models = self.model_summary()
        if models:
            out["models"] = {
                name: f"{m['parameters']:,} params · {m['tensors']} tensors"
                for name, m in models.items()
            }
        out["components"] = {
            name: ("present" if self._raw.get(name) is not None else "absent")
            for name in ("optimizer", "scheduler", "scaler")
        }
        if self.training_state:
            out["training_state"] = self.training_state
        if self.metric_names():
            out["metrics"] = ", ".join(self.metric_names())
        if self.extras:
            out["extras"] = {k: _describe(v) for k, v in self.extras.items()}
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
        source = self.path.name if self.path is not None else "checkpoint"
        print_dict_tree(
            self.summary(),
            header=f"💾 Checkpoint: {source}",
            key_width=key_width,
            print_fn=print_fn,
        )

    # ── Dunder Helpers ────────────────────────────────────────────────────────

    def __bool__(self) -> bool:
        return bool(self._raw)

    def __contains__(self, key: str) -> bool:
        return key in self._raw

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._raw[key] = value

    def __repr__(self) -> str:
        fields = [f"version={self.version!r}", f"models={list(self.models)!r}"]
        if self.path is not None:
            fields.append(f"path={self.path.as_posix()!r}")
        return f"Checkpoint({', '.join(fields)})"
