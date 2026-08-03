"""The dashboard's edges onto the outside world: a port, a browser, a locked file.

`test_dashboard.py` is deliberately hermetic — it binds nothing and launches nothing, so
that the payload and the shell can be checked without a machine in the way. That leaves
exactly the parts a user meets first untested: the run that could not bind its port, the
SSH session with no browser to open, and the Windows write that lost a race with an
indexer. Each of those is a failure the dashboard is supposed to absorb quietly, which is
also what makes a regression in one invisible. So this file does bind a loopback port and
does call `open_browser` — against stand-ins, and never against a real browser.
"""

import http.server
import json
import subprocess
import time
import urllib.request
import webbrowser
from pathlib import Path

import pytest

from train4all.dashboard import Dashboard, DashboardConfig


@pytest.fixture
def config():
    return DashboardConfig(open_on_start=False, use_server=False)


@pytest.fixture
def dash(config, tmp_path):
    return Dashboard(config, tmp_path)


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch):
    """No test here may reach a real browser, including through a path it did not expect."""
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setattr(webbrowser, "open", lambda url: pytest.fail(f"a browser opened {url}"))


# ── The HTTP server ───────────────────────────────────────────────────────────


def test_the_server_serves_the_page_and_the_payload_it_polls(tmp_path):
    """Chrome and Edge refuse a `file://` page's fetch(), which is the whole reason
    `use_server` exists — so what matters is that the JSON is actually reachable."""
    config = DashboardConfig(open_on_start=False, use_server=True)
    dash = Dashboard(config, tmp_path)
    dash.initialize({"num_epochs": 1})

    assert dash.url.startswith("http://127.0.0.1:"), "the server never bound a port"
    with urllib.request.urlopen(dash.url, timeout=10) as response:
        assert response.status == 200
        # Without this the browser would poll its own cache and the page would freeze.
        assert response.headers["Cache-Control"] == "no-store"

    data_url = dash.url.replace(config.filename, config.data_filename)
    with urllib.request.urlopen(data_url, timeout=10) as response:
        assert json.loads(response.read().decode("utf-8"))["status"] == "training"

    dash.finalize(1, 1)


def test_a_port_that_cannot_be_bound_falls_back_to_the_file_url(tmp_path, monkeypatch):
    """A blocked or exhausted port must cost the run its server, not the run."""
    def refuse(*_args, **_kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr(http.server, "ThreadingHTTPServer", refuse)
    dash = Dashboard(DashboardConfig(open_on_start=False, use_server=True), tmp_path)

    dash.initialize({})

    assert dash.url == dash.path.as_uri()


# ── Opening a browser ─────────────────────────────────────────────────────────


def test_initialize_opens_the_browser_when_asked(tmp_path, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    dash = Dashboard(DashboardConfig(open_on_start=True, use_server=False), tmp_path)

    dash.initialize({})

    assert opened == [dash.url]


def test_the_browser_variable_names_the_helper_that_is_run(dash, monkeypatch):
    """In an editor's integrated terminal `$BROWSER` forwards the port back to the
    machine you are sitting at — so it is preferred over the platform default."""
    launched: list[list[str]] = []
    monkeypatch.setenv("BROWSER", "/usr/local/bin/code-browser")
    monkeypatch.setattr(subprocess, "Popen", lambda argv: launched.append(argv))
    dash.initialize({})

    dash.open_browser()

    assert launched == [["/usr/local/bin/code-browser", dash.url]]


def test_a_helper_that_cannot_run_falls_back_to_the_platform_browser(dash, monkeypatch):
    def missing(argv):
        raise FileNotFoundError(argv[0])

    opened: list[str] = []
    monkeypatch.setenv("BROWSER", "/usr/local/bin/gone")
    monkeypatch.setattr(subprocess, "Popen", missing)
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    dash.initialize({})

    dash.open_browser()

    assert opened == [dash.url]


def test_a_helper_written_as_a_template_is_left_to_the_platform_browser(dash, monkeypatch):
    """`firefox %s` is a `webbrowser` command line, not an argv this can run itself."""
    opened: list[str] = []
    monkeypatch.setenv("BROWSER", "firefox %s")
    monkeypatch.setattr(subprocess, "Popen", lambda argv: pytest.fail(f"ran {argv}"))
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    dash.initialize({})

    dash.open_browser()

    assert opened == [dash.url]


def test_no_display_at_all_is_survivable(dash, monkeypatch):
    """Over a plain SSH session nothing can open, and a training run must not die of it."""
    def no_display(_url):
        raise webbrowser.Error("could not locate runnable browser")

    monkeypatch.setattr(webbrowser, "open", no_display)
    dash.initialize({})

    dash.open_browser()          # must not raise — the printed URL is the fallback


# ── Writing to a filesystem that pushes back ──────────────────────────────────


def test_a_locked_file_is_retried_rather_than_dropped(tmp_path, monkeypatch):
    """On Windows an antivirus scanner or the search indexer holds a brief lock on a
    just-written file. The dashboard rewrites its payload every step, so meeting one
    is routine rather than exceptional."""
    destination = tmp_path / "data.json"
    real_replace = Path.replace
    attempts = {"n": 0}

    def flaky(self, target):
        # Scoped to this file: a keepalive thread left running by an earlier test is
        # rewriting its own payload through this very method.
        if Path(target) != destination:
            return real_replace(self, target)
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError("the file is in use by another process")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky)

    Dashboard._atomic_write(destination, '{"status": "training"}')

    assert attempts["n"] == 3, "the write was not retried"
    assert destination.read_text(encoding="utf-8") == '{"status": "training"}'


def test_a_write_that_cannot_succeed_leaves_no_rubble(tmp_path, monkeypatch):
    """A read-only volume is not a lock, so retrying it would only stall the run —
    it gives up at once, and takes its temp file with it."""
    destination = tmp_path / "data.json"
    real_replace = Path.replace

    def refuse(self, target):
        if Path(target) != destination:
            return real_replace(self, target)     # another thread's write, left alone
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "replace", refuse)

    Dashboard._atomic_write(destination, "{}")

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == [], "a temp file outlived the failed write"


# ── The keepalive thread's own view ───────────────────────────────────────────


def test_the_keepalive_refreshes_the_timestamp_without_being_asked(tmp_path):
    """The browser calls a run *Offline* once `stale_after_ms` passes with no new
    timestamp. Between steps nothing calls `update()`, so this thread is the only thing
    that keeps a slow epoch from reading as a dead process.
    """
    config = DashboardConfig(open_on_start=False, use_server=False, poll_interval_ms=20)
    dash = Dashboard(config, tmp_path)
    dash.initialize({})
    written = tmp_path / config.data_filename
    first = json.loads(written.read_text(encoding="utf-8"))["last_update_ms"]

    latest = first
    deadline = time.monotonic() + 5
    while latest == first and time.monotonic() < deadline:
        time.sleep(0.02)
        latest = json.loads(written.read_text(encoding="utf-8"))["last_update_ms"]

    assert latest > first, "the keepalive thread never refreshed the payload"


def test_the_keepalive_has_nothing_to_refresh_before_the_first_snapshot(dash, config, tmp_path):
    """The thread calls `_heartbeat` directly rather than through the public `heartbeat`,
    so it can fire in the window before `initialize` has written a payload."""
    dash._heartbeat()

    assert not (tmp_path / config.data_filename).exists()


def test_finalizing_a_dashboard_that_never_initialized_embeds_nothing(dash, config, tmp_path):
    dash.finalize(1, 1)          # must not raise

    assert not (tmp_path / config.filename).exists()
