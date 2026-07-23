"""The dashboard engine — the one public component nothing else would notice breaking.

`Dashboard` is exported, documented, and driven by every `use_dashboard=True` run, yet the
trainer only ever reaches it through `_dash_*` wrappers that treat a dead dashboard as a
no-op. So a renamed payload key, a placeholder left unsubstituted in the HTML shell, or a
step window that never resets would all reach a user's browser before they reached a test.

Every dashboard here is hermetic: no HTTP server, no browser.
"""

import json
import re
from dataclasses import replace

import pytest

from train4all.utils.dashboard import _HTML_SHELL, Dashboard, DashboardConfig, PhaseSpec

SCHEDULE = [
    PhaseSpec("train", training=True, steps=10),
    PhaseSpec("val", steps=4),
]


@pytest.fixture
def config():
    """Never open a browser, never bind a port — the suite must stay hermetic."""
    return DashboardConfig(open_on_start=False, use_server=False)


@pytest.fixture
def dash(config, tmp_path):
    return Dashboard(config, tmp_path)


def payload(tmp_path, config) -> dict:
    """The JSON the browser polls, parsed as a browser would parse it."""
    text = (tmp_path / config.data_filename).read_text(encoding="utf-8")
    return json.loads(text, parse_constant=_reject)


def _reject(token: str):
    """`json.loads` accepts NaN / Infinity by default; `JSON.parse` does not."""
    raise AssertionError(f"payload is not valid JSON — a browser would reject {token!r}")


# ── Lifecycle ─────────────────────────────────────────────────────────────────


def test_initialize_writes_the_shell_and_the_first_snapshot(dash, config, tmp_path):
    dash.initialize({"num_epochs": 3})

    assert (tmp_path / config.filename).exists()
    assert (tmp_path / config.data_filename).exists()
    assert payload(tmp_path, config)["status"] == "training"


def test_the_shell_carries_the_config_not_the_placeholders(dash, config, tmp_path):
    """Every `__T4A_*` token must be substituted; one survivor is a dead dashboard."""
    dash.initialize({})
    html = (tmp_path / config.filename).read_text(encoding="utf-8")

    assert "__T4A_" not in html, "a placeholder survived the substitution"
    assert f'content="{config.poll_interval_ms}"' in html
    assert f'content="{config.data_filename}"' in html
    assert f'content="{config.stale_after_ms}"' in html


def test_active_is_true_only_between_initialize_and_finalize(dash):
    assert not dash.active
    dash.initialize({})
    assert dash.active
    dash.finalize(3, 3)
    assert not dash.active


def test_finalize_embeds_the_data_so_the_page_survives_the_process(dash, config, tmp_path):
    """The point of inlining: the run ends, the server dies, the file still renders."""
    dash.initialize({})
    dash.finalize(3, 3, {"loss": {"train": [0.5, 0.4, 0.3]}}, 0.3, 3)

    html = (tmp_path / config.filename).read_text(encoding="utf-8")
    _, _, tail = html.partition("window.__TRAIN4ALL_DATA__=")
    assert tail, "the final data was never inlined into the shell"

    embedded = json.loads(tail.partition(";</script>")[0], parse_constant=_reject)
    assert embedded["status"] == "completed"
    assert embedded["epoch_metrics"]["loss"]["train"] == [0.5, 0.4, 0.3]


# ── The payload ───────────────────────────────────────────────────────────────


def test_update_publishes_the_live_state(dash, config, tmp_path):
    dash.initialize({}, phases=SCHEDULE, monitor="accuracy", monitor_phase="val")
    dash.update(
        2, 5, {"loss": {"train": [0.9, 0.7]}}, 0.7, 2,
        epochs_no_improve=1, is_gradient_phase=True,
        step=3, max_step=10, step_metrics={"loss": 0.66},
        phase_name="train", learning_rate=1e-3, gpu_mem=(4.0, 8.0),
    )
    d = payload(tmp_path, config)

    assert (d["current_epoch"], d["max_epoch"]) == (2, 5)
    assert (d["current_step"], d["max_step"]) == (3, 10)
    assert d["last_phase"] == "train"
    assert d["is_gradient_phase"] is True
    assert d["last_step_metrics"] == {"loss": 0.66}
    assert (d["monitor"], d["monitor_phase"]) == ("accuracy", "val")
    assert (d["best_metric"], d["best_epoch"]) == (0.7, 2)
    assert d["epochs_no_improve"] == 1
    assert d["learning_rate"] == 1e-3
    assert (d["gpu_mem_used"], d["gpu_mem_total"]) == (4.0, 8.0)


def test_the_schedule_rides_along_in_the_payload(dash, config, tmp_path):
    """The phase list is the dashboard's whole model of an epoch — it lays out the gauge."""
    dash.initialize({}, phases=SCHEDULE)
    dash.update(1, 5)

    assert payload(tmp_path, config)["phases"] == [
        {"name": "train", "training": True, "steps": 10, "every": 1},
        {"name": "val", "training": False, "steps": 4, "every": 1},
    ]


def test_no_best_yet_is_published_as_null_not_a_sentinel(dash, config, tmp_path):
    """`best_epoch is None` is the single source of truth for "no best yet". The ±inf
    sentinel must never leak: it flips sign with monitor_mode, and `Infinity` is not JSON.
    """
    dash.initialize({})
    dash.update(1, 3, {}, float("inf"), None)

    assert payload(tmp_path, config)["best_metric"] is None


# ── The step-loss window ──────────────────────────────────────────────────────


def test_the_window_records_true_step_numbers_not_the_sample_count(dash, config, tmp_path):
    """Writes are throttled, so samples are sparse — the axis must still report the real
    step each one happened at, not how many samples have accumulated."""
    dash.initialize({}, phases=SCHEDULE)
    for step in (4, 8, 12):
        dash.update(1, 5, step=step, max_step=20,
                    step_metrics={"loss": float(step)}, phase_name="train")
    d = payload(tmp_path, config)

    assert d["step_loss"] == [4.0, 8.0, 12.0]
    assert (d["step_loss_first"], d["step_loss_last"]) == (4, 12)
    assert d["step_loss_phase"] == "train"


def test_the_window_resets_when_the_phase_changes(dash, config, tmp_path):
    dash.initialize({}, phases=SCHEDULE)
    dash.update(1, 5, step=1, max_step=10, step_metrics={"loss": 0.9}, phase_name="train")
    dash.update(1, 5, step=2, max_step=10, step_metrics={"loss": 0.8}, phase_name="train")
    dash.update(1, 5, step=1, max_step=4, step_metrics={"loss": 0.5}, phase_name="val")
    d = payload(tmp_path, config)

    assert d["step_loss"] == [0.5], "the training losses leaked into the val graph"
    assert d["step_loss_phase"] == "val"
    assert d["step_loss_first"] == 1


def test_the_window_is_bounded(dash, config, tmp_path):
    """It is a rolling window of recent activity — a long epoch must not grow the payload."""
    dash.initialize({}, phases=SCHEDULE)
    cap = payload(tmp_path, config)["step_loss_cap"]

    for step in range(1, cap + 21):
        dash.update(1, 5, step=step, max_step=cap + 50,
                    step_metrics={"loss": float(step)}, phase_name="train")
    d = payload(tmp_path, config)

    assert len(d["step_loss"]) == cap, "the rolling window grew without bound"
    assert d["step_loss_last"] == cap + 20
    assert d["step_loss_first"] == 21, "the oldest samples did not roll off"


def test_a_nonfinite_metric_never_reaches_the_browser(dash, config, tmp_path):
    """`JSON.parse` rejects NaN and Infinity, so one divergent metric would blank the whole
    dashboard rather than just its own readout — and `compute_metrics` can return NaN (a
    0/0 rate, an empty-class F1) while the loss itself stays perfectly finite.
    """
    dash.initialize({}, phases=SCHEDULE)
    dash.update(
        1, 5, {"f1": {"train": [float("nan")]}},
        step=1, max_step=10, phase_name="train",
        step_metrics={"loss": 0.5, "f1": float("nan")},
    )
    d = payload(tmp_path, config)   # raises if the browser could not parse it

    assert d["last_step_metrics"] == {"loss": 0.5, "f1": None}
    assert d["epoch_metrics"]["f1"]["train"] == [None]
    assert d["step_loss"] == [0.5], "one divergent metric blanked the finite loss too"


def test_a_nonfinite_loss_never_enters_the_step_window(dash, config, tmp_path):
    """The window is auto-scaled, so one NaN or Infinity would collapse the whole trace —
    and unlike the payload's other numbers it cannot be published as `null` instead: this
    is a list of points to draw, not a readout with an absent state.
    """
    dash.initialize({}, phases=SCHEDULE)
    for step, loss in enumerate([0.9, float("nan"), float("inf"), -float("inf"), 0.7], 1):
        dash.update(1, 5, step=step, max_step=10,
                    step_metrics={"loss": loss}, phase_name="train")
    d = payload(tmp_path, config)

    assert d["step_loss"] == [0.9, 0.7], "a non-finite loss entered the window"
    assert d["step_loss_last"] == 5, "the surviving samples kept their true step numbers"


# ── Heartbeat ─────────────────────────────────────────────────────────────────


def test_heartbeat_refreshes_liveness_without_changing_the_data(dash, config, tmp_path):
    """It exists so a long synchronous pause (a big save, heavy plotting) does not read as
    *Offline* — it must move the clock and nothing else."""
    dash.initialize({}, phases=SCHEDULE)
    dash.update(2, 5, {"loss": {"train": [0.9, 0.7]}}, 0.7, 2, step=3, max_step=10,
                phase_name="train", step_metrics={"loss": 0.66})
    before = payload(tmp_path, config)

    dash.heartbeat()
    after = payload(tmp_path, config)

    assert after["last_update_ms"] >= before["last_update_ms"]
    assert {k: v for k, v in after.items() if k not in ("last_update_ms", "updated_at")} == \
           {k: v for k, v in before.items() if k not in ("last_update_ms", "updated_at")}


def test_heartbeat_is_a_no_op_outside_the_run(dash, config, tmp_path):
    dash.heartbeat()                                     # before initialize — nothing to beat
    assert not (tmp_path / config.data_filename).exists()

    dash.initialize({})
    dash.finalize(3, 3)
    done = payload(tmp_path, config)

    dash.heartbeat()                                     # after finalize — the run is over
    assert payload(tmp_path, config)["last_update_ms"] == done["last_update_ms"]


# ── Reporting ─────────────────────────────────────────────────────────────────


def test_url_falls_back_to_a_file_uri_without_a_server(dash, config, tmp_path):
    dash.initialize({})
    assert dash.url == dash.path.as_uri()
    assert dash.path == (tmp_path / config.filename).resolve()


def test_poll_s_is_the_interval_in_seconds(config, tmp_path):
    assert Dashboard(replace(config, poll_interval_ms=250), tmp_path).poll_s == 0.25


def test_elapsed_runs_from_mark_started(dash):
    assert dash.elapsed is None, "nothing has started yet"
    dash.mark_started()
    assert dash.elapsed is not None
    assert dash.elapsed.total_seconds() >= 0


# ── The Python ↔ JavaScript seam ──────────────────────────────────────────────
# The browser half of the dashboard is ~1,300 lines of CSS and JavaScript held in a string
# literal, so no Python test executes it and coverage cannot even see it. What actually
# rots across that seam is not the algorithm — it is the *names*: rename a payload key on
# the Python side, or an element id in the shell, and the script silently reads
# `undefined` forever. The page does not crash; it just goes blank. These two tests are
# the name chain, checked statically — no browser required.

# The toast is the one element the script creates for itself, on first use.
SELF_CREATED_IDS = {"toast"}


def test_every_payload_key_the_script_reads_is_one_the_dashboard_writes(dash, config, tmp_path):
    dash.initialize({}, phases=SCHEDULE)
    written = set(payload(tmp_path, config))

    read = set(re.findall(r"\bd\.([a-z_][a-z0-9_]*)", _HTML_SHELL))
    orphans = sorted(read - written)
    assert not orphans, f"the script reads payload keys nothing writes: {orphans}"


def test_the_payload_carries_nothing_the_script_never_reads(dash, config, tmp_path):
    """The other direction: a key no one reads is weight the browser downloads every tick."""
    dash.initialize({}, phases=SCHEDULE)
    written = set(payload(tmp_path, config))

    read = set(re.findall(r"\bd\.([a-z_][a-z0-9_]*)", _HTML_SHELL))
    assert not sorted(written - read), "the payload carries keys the script never reads"


def test_every_element_the_script_drives_exists_in_the_shell():
    driven = set(re.findall(r"(?:el|setText|renderKV)\(\s*'([a-z0-9_-]+)'", _HTML_SHELL))
    present = set(re.findall(r'id="([a-z0-9_-]+)"', _HTML_SHELL))

    missing = sorted(driven - SELF_CREATED_IDS - present)
    assert not missing, f"the script drives elements the shell does not define: {missing}"
