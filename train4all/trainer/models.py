"""The models a run holds, by name, for train4all.

``ModelLedger`` is the register of what a run is training: a name for each module,
which of them were compiled, and every question asked of them as a group — collect
the trainable parameters, freeze a few, reset some, report what each holds.

Two things it deliberately does not do. It does not know the device: where a model
runs is the trainer's decision, so the trainer moves a module and hands over the
result rather than letting a second copy of that fact live here. And it does not
assign attributes — ``set_model(..., set_attr=True)`` writes into the trainer's own
namespace, which is the trainer's to write.
"""

from collections.abc import ItemsView

import torch.nn as nn

__all__ = ["ModelLedger", "ModuleSpec"]

type ModuleSpec = str | nn.Module | list[str | nn.Module]
"""A model, the name one is registered under, or a list of either."""


class ModelLedger:
    """Every model a run has registered, and what is asked of them as a group.

    A name is what the rest of the framework files a model under — the checkpoint's
    ``models`` map, the parameter groups a ``learning_rate`` dict is keyed by, the
    rows of :meth:`summary` — so a name is the one way in.
    """

    def __init__(self) -> None:
        self._models: dict[str, nn.Module] = {}
        self._compiled: set[str] = set()

    # ── Registering ───────────────────────────────────────────────────────────

    def register(self, name: str, model: nn.Module, *, compile: bool = False) -> None:
        """
        File *model* under *name*, replacing whatever that name held.

        Args:
            name: Name to file the model under.
            model: The module, already on the device it will run on.
            compile: Compile it in place with ``torch.compile()`` (PyTorch 2.0+).
                The registered module itself is compiled, so any reference to it
                runs the optimized graph and checkpoints keep their original keys.
        """
        if compile:
            model.compile()  # in place: keeps the same object and state-dict keys
            self._compiled.add(name)
        else:
            self._compiled.discard(name)
        self._models[name] = model

    def clear(self) -> None:
        """Forget every registered model."""
        self._models.clear()
        self._compiled.clear()

    # ── Reading ───────────────────────────────────────────────────────────────

    def __contains__(self, name: object) -> bool:
        """Whether *name* is registered."""
        return name in self._models

    def items(self) -> ItemsView[str, nn.Module]:
        """Every ``(name, model)`` pair, in the order they were first registered."""
        return self._models.items()

    def get(self, name: str) -> nn.Module | None:
        """The model registered under *name*, or ``None`` if there is none."""
        return self._models.get(name)

    def trainable_params(
        self,
        targets: ModuleSpec | None = None,
        exclude_targets: ModuleSpec | None = None,
    ) -> list[nn.Parameter]:
        """
        Deduplicated parameters with ``requires_grad=True``.

        Args:
            targets: Models to include. ``None`` includes every registered model.
            exclude_targets: Models to leave out of the result.

        Returns:
            List of unique trainable parameters, in the order they are reached.
        """
        modules = self._resolve(targets)
        if exclude_targets is not None:
            excluded = set(self._resolve(exclude_targets))
            modules = [m for m in modules if m not in excluded]

        seen: set[int] = set()
        params: list[nn.Parameter] = []
        for m in modules:
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def summary(self) -> dict[str, str]:
        """What each registered model holds: its parameter counts, and whether it is
        compiled or frozen."""
        result: dict[str, str] = {}
        for name, model in self._models.items():
            total = trainable = 0
            for p in model.parameters():
                n = p.numel()
                total += n
                if p.requires_grad:
                    trainable += n
            suffix = " [compiled]" if name in self._compiled else ""
            if trainable == total:
                result[name] = f"{total:,} params{suffix}"
            elif trainable:
                result[name] = f"{trainable:,} / {total:,} trainable{suffix}"
            else:
                result[name] = f"frozen{suffix}"
        return result

    # ── Acting on them ────────────────────────────────────────────────────────

    def set_requires_grad(self, targets: ModuleSpec, flag: bool) -> None:
        """Set ``requires_grad`` on every parameter of the models *targets* names."""
        for m in self._resolve(targets):
            for p in m.parameters():
                p.requires_grad = flag

    def reset_parameters(self, targets: ModuleSpec | None = None) -> None:
        """
        Re-initialize the models *targets* names.

        Calls ``reset_parameters()`` on every submodule that implements it;
        submodules without the method are silently skipped.

        Args:
            targets: Models to reset. ``None`` resets every registered model.
        """
        for module in self._resolve(targets):
            module.apply(self._reset_module_parameters)

    @staticmethod
    def _reset_module_parameters(m: nn.Module) -> None:
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()

    def set_training_mode(self, training: bool) -> None:
        """Put every registered model into training or evaluation mode."""
        for model in self._models.values():
            model.train(training)

    # ── Resolving a target ────────────────────────────────────────────────────

    def _resolve(self, targets: ModuleSpec | None) -> list[nn.Module]:
        if targets is None:
            return list(self._models.values())
        if not isinstance(targets, list):
            targets = [targets]
        return [self._resolve_one(t) for t in targets]

    def _resolve_one(self, target: str | nn.Module) -> nn.Module:
        if isinstance(target, str):
            if target not in self._models:
                raise ValueError(f"Model '{target}' is not registered.")
            return self._models[target]
        if isinstance(target, nn.Module):
            return target
        raise TypeError(f"Expected a model name or nn.Module, got {type(target)}")
