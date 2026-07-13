"""A Phase is the one place a pass over data is described."""

import pytest
from conftest import make_loader

from train4all import Phase


def test_a_phase_needs_a_name():
    with pytest.raises(ValueError, match="non-empty name"):
        Phase("", make_loader(4))


def test_every_must_be_at_least_one():
    with pytest.raises(ValueError, match="every must be >= 1"):
        Phase("train", make_loader(4), every=0)


def test_records_steps_follows_training_by_default():
    assert Phase("train", make_loader(4), training=True).records_steps is True
    assert Phase("val", make_loader(4)).records_steps is False


def test_records_steps_can_be_forced_either_way():
    assert Phase("val", make_loader(4), record_steps=True).records_steps is True
    assert Phase("train", make_loader(4), training=True, record_steps=False).records_steps is False


@pytest.mark.parametrize(
    ("every", "epoch", "runs"),
    [(1, 1, True), (1, 7, True), (3, 1, False), (3, 2, False), (3, 3, True), (3, 6, True)],
)
def test_runs_at(every, epoch, runs):
    assert Phase("audit", make_loader(4), every=every).runs_at(epoch) is runs


def test_a_phase_is_frozen():
    phase = Phase("train", make_loader(4))
    with pytest.raises(AttributeError):
        phase.name = "renamed"


def test_metric_fn_holds_a_function_not_values():
    """The field is named for what it holds — the collision this rename removed."""
    phase = Phase("train", make_loader(4), metric_fn=lambda _: {"custom": 1.0})
    assert callable(phase.metric_fn)
    assert phase.metric_fn(None) == {"custom": 1.0}
