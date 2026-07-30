"""The README is the API reference, so every exported name must appear in it.

This is the documentation invariant made executable: add a public method and the
suite fails until the README says what it does. Without it, "the docs are complete"
is a claim that decays silently from the first commit after anyone checks.
"""

import inspect
import pathlib
import re

import pytest

import train4all
from train4all import BaseTrainer, Checkpoint, Dashboard, DashboardConfig, Phase, PhaseSpec
from train4all import utils as t4a_utils

README = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def documented(name: str) -> bool:
    # The README writes these as ``trainer.save_checkpoint(...)`` / ``ckpt.models``,
    # so a leading dot is a mention. A trailing word character is not.
    return re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", README) is not None


def public_surface(cls: type) -> list[str]:
    """Every public name a user calls or reads on the class — methods and properties
    alike, since a property is as public as the method it replaced."""
    return sorted(
        n for n in dir(cls)
        if not n.startswith("_")
        and (callable(attr := getattr(cls, n, None)) or isinstance(attr, property))
    )


def public_members(cls: type) -> list[str]:
    return sorted(n for n in dir(cls) if not n.startswith("_"))


def class_constants() -> list[str]:
    return sorted(
        n for n, v in vars(BaseTrainer).items()
        if n.startswith("_") and n[1:2].isupper() and not callable(v)
    )


@pytest.mark.parametrize("name", sorted(train4all.__all__))
def test_the_package_exports_are_documented(name):
    assert documented(name), f"train4all.{name} is exported but absent from the README"


@pytest.mark.parametrize("name", sorted(t4a_utils.__all__))
def test_the_utils_exports_are_documented(name):
    assert documented(name), f"train4all.utils.{name} is exported but absent from the README"


@pytest.mark.parametrize("name", public_surface(BaseTrainer))
def test_the_trainer_surface_is_documented(name):
    assert documented(name), f"BaseTrainer.{name} is public but absent from the README"


@pytest.mark.parametrize("name", public_members(Checkpoint))
def test_the_checkpoint_members_are_documented(name):
    assert documented(name), f"Checkpoint.{name} is public but absent from the README"


@pytest.mark.parametrize("name", public_members(Phase))
def test_the_phase_members_are_documented(name):
    assert documented(name), f"Phase.{name} is public but absent from the README"


@pytest.mark.parametrize(
    "name", [p for p in inspect.signature(BaseTrainer.__init__).parameters if p != "self"]
)
def test_every_constructor_argument_is_documented(name):
    assert documented(name), f"BaseTrainer({name}=...) is absent from the README"


@pytest.mark.parametrize("name", class_constants())
def test_every_class_constant_is_documented(name):
    assert documented(name), f"BaseTrainer.{name} is overridable but absent from the README"


@pytest.mark.parametrize("name", sorted(DashboardConfig.__dataclass_fields__))
def test_every_dashboard_setting_is_documented(name):
    assert documented(name), f"DashboardConfig.{name} is absent from the README"


@pytest.mark.parametrize("name", sorted(PhaseSpec.__dataclass_fields__))
def test_every_phasespec_field_is_documented(name):
    assert documented(name), f"PhaseSpec.{name} is absent from the README"


@pytest.mark.parametrize("name", public_members(Dashboard))
def test_the_dashboard_members_are_documented(name):
    assert documented(name), f"Dashboard.{name} is public but absent from the README"


def test_phase_has_no_field_called_metrics():
    """`metrics` means metric *values* everywhere in the framework; the phase holds a
    *function*, and calling it `metrics` was the collision this rename removed."""
    assert "metrics" not in Phase.__dataclass_fields__
    assert "metric_fn" in Phase.__dataclass_fields__


def test_the_version_is_resolved_in_exactly_one_place():
    from train4all import _version
    from train4all.dashboard import engine

    assert train4all.__version__ is _version.__version__
    assert engine.__version__ is _version.__version__
