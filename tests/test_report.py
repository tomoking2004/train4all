"""Report on its own — the console voice with no run behind it.

The banners reach a console from a real loop in `test_training`. This drives them
directly, so what a header says, and which rows the report names rather than
receives, can be read off one file instead of inferred from captured output.
"""

import pytest
from conftest import TinyTrainer, make_loader

from train4all import Phase
from train4all.trainer.report import Report


@pytest.fixture
def lines() -> list[str]:
    return []


@pytest.fixture
def report(lines) -> Report:
    """A report that collects its lines instead of printing them."""
    return Report(lines.append)


def rows(lines: list[str]) -> dict[str, str]:
    """The ``key: value`` leaves of a printed tree, keyed by leaf name."""
    found: dict[str, str] = {}
    for line in lines:
        key, sep, value = line.partition(": ")
        if sep:
            found[key.rpartition("─ ")[2].strip()] = value.strip()
    return found


# ── Banners over a dict the caller brings ─────────────────────────────────────


@pytest.mark.parametrize(("banner", "header"), [
    ("env",      "🖥️  Environment"),
    ("config",   "⚙️  Configuration"),
    ("model",    "🧠 Model"),
    ("schedule", "🗓️  Schedule"),
])
def test_a_summary_banner_adds_a_header_and_leaves_the_rows_alone(report, lines, banner, header):
    """These are handed a dict they did not build, so its rows are not theirs to
    rename — only the header above them is the report's own."""
    getattr(report, banner)({"given": "verbatim"})

    assert lines[0] == header
    assert rows(lines) == {"given": "verbatim"}


def test_metrics_heads_the_table_with_the_phase_and_stays_flat(report, lines):
    report.metrics({"loss": 0.5, "accuracy": 0.25}, "val")

    assert lines[0] == "📊 Val"
    assert rows(lines) == {"loss": "0.5000", "accuracy": "0.2500"}


# ── Banners the report names the rows of ─────────────────────────────────────


def test_optimization_reports_the_class_of_what_it_is_handed(report, lines):
    report.optimization(ValueError(), TypeError())

    assert rows(lines) == {"Optimizer": "ValueError", "Scheduler": "TypeError"}


def test_optimization_shows_a_dash_where_there_is_no_object(report, lines):
    report.optimization(None, None)

    assert rows(lines) == {"Optimizer": "-", "Scheduler": "-"}


@pytest.mark.parametrize(("steps", "shown"), [(1, False), (2, True)])
def test_the_accumulation_row_appears_only_where_it_does_something(report, lines, steps, shown):
    report.optimization(None, None, accumulation_steps=steps)

    assert ("Grad accumulation" in rows(lines)) is shown


def test_status_dates_the_best_value_by_its_epoch(report, lines):
    report.status(
        completed_epochs=7,
        monitor="val loss",
        best_value=0.125,
        best_epoch=3,
        stagnant_epochs=4,
        latest={},
    )

    found = rows(lines)
    assert found["Completed epochs"] == "7"
    assert found["Best val loss"] == "0.1250  (epoch 3)"
    assert found["Stagnant epochs"] == "4"


def test_status_reads_the_absent_best_off_the_epoch_and_not_the_value(report, lines):
    """`best_value` starts at ±inf so that any real value beats it, which makes it no
    evidence either way. The epoch is None until something has actually been monitored."""
    report.status(
        completed_epochs=0,
        monitor="val loss",
        best_value=float("inf"),
        best_epoch=None,
        stagnant_epochs=0,
        latest={},
    )

    assert rows(lines)["Best val loss"] == "-"
    assert "inf" not in "\n".join(lines)


def test_status_shows_a_dash_where_nothing_has_been_recorded(report, lines):
    report.status(
        completed_epochs=1,
        monitor="val loss",
        best_value=0.5,
        best_epoch=1,
        stagnant_epochs=0,
        latest={},
    )

    assert rows(lines)["Last epoch metrics"] == "-"


# ── Primitives ───────────────────────────────────────────────────────────────


def test_a_tree_without_a_header_prints_no_rule(report, lines):
    report.tree({"a": 1})

    assert rows(lines) == {"a": "1"}
    assert not any(set(line.strip()) == {"─"} for line in lines)


def test_the_rule_is_as_wide_as_the_column_it_underlines(lines):
    narrow, wide = Report(lines.append, key_width=10), Report(lines.append, key_width=40)

    narrow.rule()
    wide.rule()

    assert len(lines[0]) < len(lines[1])


def test_a_report_given_no_printer_falls_back_to_print(capsys):
    Report().tree({"a": 1}, header="H")

    assert "H" in capsys.readouterr().out


def test_the_column_the_report_is_built_with_is_the_column_it_prints(lines):
    Report(lines.append, key_width=48).tree({"a": 1})

    assert lines[0].index(":") > 48


# ── The trainer's side of the delegation ─────────────────────────────────────


def test_the_trainer_prints_its_banners_through_the_report(trainer, capsys):
    trainer.print_status()
    trainer.print_schedule_summary(Phase("train", make_loader(8), training=True))

    out = capsys.readouterr().out
    assert "📋 Status" in out and "🗓️  Schedule" in out


def test_a_logger_swapped_after_construction_still_receives_the_banners(trainer):
    """The report is handed `trainer.print`, which reads `self.logger` when called —
    not the logger itself, which would freeze whichever one construction happened to see."""
    collected: list[str] = []

    class Collector:
        def log(self, msg=None, level="info", *, indent=0):
            collected.append(msg or "")

    trainer.logger = Collector()
    trainer.print_status()

    assert any("📋 Status" in line for line in collected)


def test_a_subclass_widening_the_column_widens_its_banners(run_dir, capsys):
    """`_KEY_WIDTH` is documented as overridable, so the report has to be built from
    whatever the class says rather than from the shared default."""
    class Wide(TinyTrainer):
        _KEY_WIDTH = 64

    def colon_of(t) -> int:
        capsys.readouterr()
        t.print_status()
        return min(
            line.index(":") for line in capsys.readouterr().out.splitlines() if ":" in line
        )

    assert colon_of(Wide(run_dir=run_dir)) > colon_of(TinyTrainer(run_dir=run_dir))
