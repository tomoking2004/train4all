"""MetricStore on its own — the half of the metrics that needs no trainer.

`test_metrics` drives the tables through a real loop. This drives them directly, and
covers what only the standalone class can do: read a finished run back off disk.
"""

import json
from pathlib import Path

import pytest
from conftest import TinyTrainer, make_loader

from train4all import MetricStore, Phase


@pytest.fixture
def store() -> MetricStore:
    s = MetricStore()
    s.record_epoch({"loss": 0.9, "accuracy": 0.5}, "train")
    s.record_epoch({"loss": 0.6}, "val")
    s.record_step({"loss": 1.2}, "train")
    return s


# ── Recording ─────────────────────────────────────────────────────────────────


def test_values_pile_up_per_metric_per_phase():
    s = MetricStore()
    s.record_epoch({"loss": 0.9}, "train")
    s.record_epoch({"loss": 0.7}, "train")
    s.record_epoch({"loss": 0.8}, "val")

    assert s.epoch == {"loss": {"train": [0.9, 0.7], "val": [0.8]}}


def test_the_store_records_whatever_it_is_handed(store):
    """Narrowing is the caller's business — see `test_the_trainer_narrows_step_metrics`."""
    store.record_step({"anything": 1.0}, "train")
    assert "anything" in store.step


def test_clearing_empties_both_tables(store):
    store.clear()
    assert store.epoch == {} and store.step == {}
    assert not store


# ── Reading ───────────────────────────────────────────────────────────────────


def test_a_filtered_table_keeps_only_what_was_named(store):
    assert store.epoch_table(["loss"], ["val"]) == {"loss": {"val": [0.6]}}
    assert store.step_table(["missing"]) == {}


def test_filtering_returns_a_copy_rather_than_the_table(store):
    """The raw tables are public; a filtered view must not be a way to edit them."""
    store.epoch_table()["loss"].clear()
    assert store.epoch["loss"], "filtering handed out the table it was reading"


def test_an_empty_series_is_dropped_from_a_filtered_table():
    s = MetricStore(epoch={"loss": {"train": [], "val": [0.6]}})
    assert s.epoch_table() == {"loss": {"val": [0.6]}}


# ── Weighted averaging ────────────────────────────────────────────────────────


def test_the_average_is_weighted_by_what_it_was_given():
    accumulated: dict[str, float] = {}
    MetricStore.accumulate(accumulated, {"loss": 1.0}, weight=3)
    MetricStore.accumulate(accumulated, {"loss": 2.0}, weight=1)

    assert MetricStore.average(accumulated, 4) == {"loss": 1.25}


def test_averaging_nothing_yields_nothing_rather_than_dividing_by_zero():
    assert MetricStore.average({"loss": 0.0}, 0) == {}


# ── Artifacts ─────────────────────────────────────────────────────────────────


def test_an_exported_run_reads_back_identical(store, tmp_path):
    epoch = store.export_epoch(tmp_path / "epoch.json")
    step = store.export_step(tmp_path / "step.json")

    back = MetricStore.load(epoch, step)
    assert back.epoch == store.epoch
    assert back.step == store.step


def test_load_takes_the_epoch_file_alone(store, tmp_path):
    back = MetricStore.load(store.export_epoch(tmp_path / "epoch.json"))
    assert back.epoch == store.epoch
    assert back.step == {}


def test_a_failed_export_is_reported_rather_than_raised(store, tmp_path):
    said: list[str] = []
    # A directory where the file should go: the open fails, the run does not.
    (tmp_path / "epoch.json").mkdir()
    store.export_epoch(tmp_path / "epoch.json", print_fn=said.append)

    assert said and "Failed to write metrics" in said[0]


def test_a_file_that_is_not_a_metric_table_says_so(tmp_path):
    path = tmp_path / "epoch.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="not a train4all metric table"):
        MetricStore.load(path)


def test_epoch_plots_put_every_phase_on_one_figure(store, tmp_path):
    store.save_epoch_plots(lambda metric: tmp_path / f"{metric}.png")
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["accuracy.png", "loss.png"]


def test_step_plots_split_per_phase(store, tmp_path):
    """Steps are counted within a phase, so two phases share no x-axis to share."""
    store.record_step({"loss": 1.1}, "val")
    store.save_step_plots(lambda metric, phase_name: tmp_path / f"{metric}_{phase_name}.png")

    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["loss_train.png", "loss_val.png"]


# ── Inspection ────────────────────────────────────────────────────────────────


def test_latest_shows_the_most_recent_value_of_each_metric_per_phase(store):
    assert store.latest() == {"loss": "train=0.9000  val=0.6000", "accuracy": "train=0.5000"}


def test_metric_names_unions_both_tables(store):
    store.record_step({"grad_norm": 1.0}, "train")
    assert store.metric_names() == ["accuracy", "grad_norm", "loss"]


def test_the_summary_counts_the_points_behind_each_phase(store):
    assert store.summary() == {
        "epoch": {"loss": "train (1), val (1)", "accuracy": "train (1)"},
        "step": {"loss": "train (1)"},
    }


def test_an_empty_store_summarizes_to_nothing():
    assert MetricStore().summary() == {}


def test_the_printed_summary_names_the_file_it_came_from(store, tmp_path):
    lines: list[str] = []
    MetricStore.load(store.export_epoch(tmp_path / "epoch.json")).print_summary(print_fn=lines.append)

    assert any("epoch.json" in line for line in lines), "the header does not name the source"
    assert any("train (1), val (1)" in line for line in lines)


def test_a_store_with_no_file_behind_it_still_prints(store):
    lines: list[str] = []
    store.print_summary(print_fn=lines.append)
    assert any("Metrics: metrics" in line for line in lines)


def test_the_repr_shows_what_is_held_and_where_it_came_from(store, tmp_path):
    assert repr(store) == "MetricStore(epoch=['loss', 'accuracy'], step=['loss'])"

    path = store.export_epoch(tmp_path / "epoch.json")
    assert f"path='{path.as_posix()}'" in repr(MetricStore.load(path))


def test_a_metric_with_nothing_recorded_is_not_plotted(tmp_path):
    MetricStore(epoch={"loss": {"train": []}}).save_epoch_plots(
        lambda metric: tmp_path / f"{metric}.png"
    )
    assert list(tmp_path.glob("*.png")) == []


# ── The trainer's side ────────────────────────────────────────────────────────


def test_the_trainer_narrows_step_metrics_before_the_store_sees_them(run_dir):
    """`step_metric_names` is the trainer's setting, so the trainer applies it."""
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False,
        record_step_metrics=True, step_metric_names=["loss"],
    )
    trainer.train(Phase("train", make_loader(4, batch_size=2), training=True))

    assert set(trainer.get_step_metrics()) == {"loss"}
    assert "accuracy" in trainer.get_epoch_metrics(), "only the step table is narrowed"


def test_the_trainer_reads_the_run_it_wrote(run_dir):
    """The public verbs and the standalone reader describe the same run."""
    trainer = TinyTrainer(
        num_epochs=2, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(4), training=True))
    path = trainer.export_epoch_metrics()

    assert MetricStore.load(path).epoch == trainer.get_epoch_metrics()


def test_a_setting_changed_after_construction_still_takes_effect(run_dir):
    """`step_metric_names` is a plain public attribute; reading it late is the point."""
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False,
        record_step_metrics=True,
    )
    trainer.step_metric_names = ["accuracy"]
    trainer.train(Phase("train", make_loader(4, batch_size=2), training=True))

    assert set(trainer.get_step_metrics()) == {"accuracy"}


def test_the_trainer_still_decides_where_a_plot_goes(run_dir):
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run_dir, use_progress_bar=False,
        record_step_metrics=True,
    )
    trainer.train(Phase("train", make_loader(4, batch_size=2), training=True))
    trainer.save_epoch_metric_plots()
    trainer.save_step_metric_plots()

    plots = {p.name for p in (Path(run_dir) / "plots").glob("*.png")}
    assert "loss.png" in plots, "epoch plots carry no prefix"
    assert "step_loss_train.png" in plots, "step plots carry the phase and the step prefix"
