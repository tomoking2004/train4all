"""ModelLedger on its own — the register with no trainer around it.

`test_api_surface` drives freeze, unfreeze and the parameter collection through the
trainer, and `test_errors` covers what an unknown target raises. This drives the
ledger directly, for what only the class itself shows: what a name holds after a
second registration, and the rows a summary gives each shape of model.
"""

import pytest
import torch.nn as nn
from conftest import TinyTrainer

from train4all.trainer.models import ModelLedger


@pytest.fixture
def ledger() -> ModelLedger:
    return ModelLedger()


def frozen(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    return module


class Compilable(nn.Linear):
    """A model that records ``compile()`` rather than invoking torch's compiler.

    ``nn.Module.compile`` reaches into ``torch._inductor``, which is not importable
    on every machine — and what the ledger promises about it is only that it calls
    the method in place and remembers having done so.
    """

    def __init__(self) -> None:
        super().__init__(2, 2)
        self.compiled = False

    def compile(self, *args, **kwargs) -> None:
        self.compiled = True


# ── Registering ───────────────────────────────────────────────────────────────


def test_a_name_is_the_way_in_and_the_way_back(ledger):
    net = nn.Linear(2, 2)
    ledger.register("net", net)

    assert "net" in ledger
    assert ledger.get("net") is net
    assert list(ledger.items()) == [("net", net)]


def test_an_unregistered_name_holds_nothing(ledger):
    assert "net" not in ledger
    assert ledger.get("net") is None


def test_registering_a_name_twice_keeps_the_second(ledger):
    first, second = nn.Linear(2, 2), nn.Linear(2, 2)
    ledger.register("net", first)
    ledger.register("net", second)

    assert ledger.get("net") is second
    assert len(list(ledger.items())) == 1


def test_names_keep_the_order_they_were_first_registered(ledger):
    ledger.register("encoder", nn.Linear(2, 2))
    ledger.register("head", nn.Linear(2, 2))
    ledger.register("encoder", nn.Linear(2, 2))

    assert [name for name, _ in ledger.items()] == ["encoder", "head"]


def test_clearing_forgets_the_models_and_that_they_were_compiled(ledger):
    ledger.register("net", Compilable(), compile=True)
    ledger.clear()
    ledger.register("net", nn.Linear(2, 2))

    assert "[compiled]" not in ledger.summary()["net"]


# ── Summarizing ───────────────────────────────────────────────────────────────


def test_a_wholly_trainable_model_reports_its_size(ledger):
    ledger.register("net", nn.Linear(2, 2))          # 4 weights + 2 biases

    assert ledger.summary() == {"net": "6 params"}


def test_a_partly_frozen_model_reports_both_counts(ledger):
    net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    frozen(net[0])
    ledger.register("net", net)

    assert ledger.summary() == {"net": "6 / 12 trainable"}


def test_a_wholly_frozen_model_reports_no_count_at_all(ledger):
    """A size is what a run can still move; frozen, the number says nothing useful."""
    ledger.register("net", frozen(nn.Linear(2, 2)))

    assert ledger.summary() == {"net": "frozen"}


def test_a_compiled_model_is_compiled_in_place_and_says_so(ledger):
    net = Compilable()

    ledger.register("net", net, compile=True)

    assert net.compiled, "the registered module itself must be the one compiled"
    assert ledger.summary() == {"net": "6 params [compiled]"}


def test_registering_again_without_compile_drops_the_mark(ledger):
    ledger.register("net", Compilable(), compile=True)
    ledger.register("net", nn.Linear(2, 2))

    assert ledger.summary() == {"net": "6 params"}


# ── Collecting parameters ─────────────────────────────────────────────────────


def test_one_module_under_two_names_yields_its_parameters_once(ledger):
    """Deduplication is by parameter identity, so a shared module cannot be counted
    twice into an optimizer that would then step it twice."""
    shared = nn.Linear(2, 2)
    ledger.register("a", shared)
    ledger.register("b", shared)

    assert len(ledger.trainable_params()) == len(list(shared.parameters()))


def test_only_the_named_models_are_collected(ledger):
    net, head = nn.Linear(2, 2), nn.Linear(2, 2)
    ledger.register("net", net)
    ledger.register("head", head)

    assert ledger.trainable_params("head") == list(head.parameters())
    assert ledger.trainable_params(exclude_targets="head") == list(net.parameters())


def test_a_frozen_parameter_is_not_collected(ledger):
    ledger.register("net", frozen(nn.Linear(2, 2)))

    assert ledger.trainable_params() == []


# ── Acting on them ────────────────────────────────────────────────────────────


def test_requires_grad_reaches_every_parameter(ledger):
    ledger.register("net", nn.Linear(2, 2))

    ledger.set_requires_grad("net", False)
    assert all(not p.requires_grad for p in ledger.get("net").parameters())

    ledger.set_requires_grad("net", True)
    assert all(p.requires_grad for p in ledger.get("net").parameters())


def test_resetting_skips_a_submodule_that_cannot_be_reset(ledger):
    """`ReLU` has no `reset_parameters`, and a model made of both must still reset."""
    linear = nn.Linear(2, 2)
    ledger.register("net", nn.Sequential(linear, nn.ReLU()))
    before = linear.weight.clone()

    ledger.reset_parameters()

    assert not linear.weight.equal(before)


def test_the_training_mode_reaches_every_registered_model(ledger):
    ledger.register("net", nn.Linear(2, 2))
    ledger.register("head", nn.Linear(2, 2))

    ledger.set_training_mode(False)
    assert all(not model.training for _, model in ledger.items())

    ledger.set_training_mode(True)
    assert all(model.training for _, model in ledger.items())


# ── The trainer's side of the delegation ──────────────────────────────────────


def test_overwrite_false_leaves_the_incoming_model_untouched(run_dir):
    """The guard runs before anything is done to the argument, so a skipped call
    neither registers the module, moves it to the device, nor compiles it."""
    trainer = TinyTrainer(run_dir=run_dir)
    first, second = nn.Linear(2, 2), nn.Linear(2, 2)
    trainer.set_model("net", first)

    trainer.set_model("net", second, overwrite=False, compile=True)

    assert trainer.get_model_summary() == {"net": "6 params"}


def test_set_attr_binds_the_model_the_trainer_registered(run_dir):
    """The attribute is written onto the trainer, which is why the ledger has no say
    in it — and it must be the same object the ledger holds."""
    trainer = TinyTrainer(run_dir=run_dir)
    net = nn.Linear(2, 2)

    trainer.set_model("encoder", net, set_attr=True)

    # The ignore states the premise: `encoder` exists only because `set_attr=True`
    # wrote it, which is the whole of what this test checks.
    assert trainer.encoder is net  # type: ignore[attr-defined]


def test_clearing_the_setup_empties_the_ledger(trainer):
    trainer.ensure_setup()
    assert trainer.get_model_summary()

    trainer.clear_setup()

    assert trainer.get_model_summary() == {}
