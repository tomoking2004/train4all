"""
Shoot the four README images into ``assets/``.

    pip install playwright pillow torchvision   # drives the installed Edge; no browser download
    python scripts/shoot.py                     # → assets/  (about two minutes)

The images answer to two masters, so they are shot from different runs:

  · **The console shot sits under the README's Quick Start**, so it must be that
    code's own output — anything else is a caption lying about the snippet above it.
    So the snippet is read out of ``README.md`` (by ``quickstart``) and *run*, on real
    MNIST, and its stdout is set in a terminal face. Nothing is retyped.
  · **The dashboard shots are the dashboard's showcase**, tied to no snippet, and
    they cannot come from the Quick Start: the step-loss strip holds 96 samples at
    one write per 0.5 s, so filling it needs a phase lasting ≳48 s — that is
    ``steps ≥ 48 × it/s``, which a heavy model satisfies and a fast MLP never can.
    They are shot from the staged run in ``demo_scene``, whose 620 steps at 8 it/s
    were chosen to clear that.

Two runs, but **one machine**: ``scene``'s reference workstation is declared as the
host of both (see ``declare_host``), because a reader compares the images and they
must agree on what they were run on. That is the only fact the images declare
rather than measure.

The dashboard is **played**, not rendered from a snapshot. Throughput and ETA are
measured in the browser, from the step counter advancing across successive polls, so
a still page has nothing to measure and cannot show them. The staged run is therefore
fed to a live dashboard in real time, with a real browser watching — about 48 s.

Ratios. The dashboard is a screen, so it is shot at the screen's own ratio, 16:9.
The console is a transcript, so it is shot at that ratio's transpose, 9:16 — the two
are then the same rectangle, rotated. Both are supersampled at 2× and
resampled down, which is what keeps the hairlines and the mono text crisp.

Nothing the repository tracks is touched but the images. The dashboard's page and the
Quick Start's whole working directory live in temp folders that are removed on exit.
"""

import atexit
import contextlib
import html
import io
import logging
import shutil
import stat
import tempfile
import time
import webbrowser
from pathlib import Path

import demo_scene as scene
import quickstart
from PIL import Image
from playwright.sync_api import sync_playwright

import train4all.trainer.base as base
from train4all import BaseTrainer
from train4all.dashboard import Dashboard, DashboardConfig

ROOT   = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
# The run directory is scratch: the page and its JSON only have to exist while the
# browser is looking at them. So it lives in the system temp folder and is removed
# when the script exits, rather than accumulating beside the scripts.
RUN_DIR = Path(tempfile.mkdtemp(prefix="train4all-demo-"))
atexit.register(shutil.rmtree, RUN_DIR, ignore_errors=True)
MNIST  = ROOT / "data"         # where a reader's own Quick Start run leaves its download

SCALE = 2                      # supersample, then resample down — crisp hairlines
DASH_W, DASH_H = 2048, 1152    # 16:9 — the screen's own ratio
SETTLE_MS = 1800               # the bars ease to their widths over 0.5 s; let them land

# The README frames the dashboard on its headline, live readings and four charts.
# Cropping there would slice whatever the 16:9 cut happens to land on, so the page is
# asked to *end* there instead: the run's other charts and the closing tables are
# hidden, and a bottom gutter closes the image. The charts stand in the
# document in the order they are read, so the first four are the first two rows.
CROP_TO_FOUR_CHARTS = """
  #charts-grid .chart:nth-child(n+5) { display: none !important; }
  .epochs, .record { display: none !important; }
  .page { padding-bottom: 36px !important; }
"""
# A finished run has no live readings, which would leave the rail one panel tall
# beside four charts. Its record opens with when it ran and what it was started
# with, and those two panels fit the frame, so they are let back in.
SHOW_RUN_PANEL = """
  .record { display: grid !important; }
  .record .panel:nth-child(n+3) { display: none !important; }
"""

# The console's terminal face, sampled from the shot it replaces.
CONSOLE_BG   = "#1f2428"
CONSOLE_FG   = "#d1d5da"
CONSOLE_FONT = 14           # px
CONSOLE_LEAD = 1.55         # line-height
CONSOLE_PAD  = 30           # px of terminal background around the text


def shoot(page, path: Path, width: int, height: int) -> None:
    """Capture *page* at 2× and resample down to ``width × height``."""
    raw = page.screenshot(clip={"x": 0, "y": 0, "width": width, "height": height})
    image = Image.open(io.BytesIO(raw))
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    image.save(path)
    print(f"  {path.relative_to(ROOT)}  {width} × {height}")


def dashboard_frames(browser) -> None:
    """The three dashboard shots: the live hero in both themes, then the finish."""
    history = scene.series()
    val_acc = history["accuracy"]["val"]
    done = scene.HERO_EPOCH - 1                  # completed epochs at the freeze

    write_ms = round(scene.WRITE_S * 1000)
    dash = Dashboard(DashboardConfig(poll_interval_ms=write_ms, open_on_start=False), RUN_DIR)
    dash.initialize(scene.CONFIG, env_summary=scene.ENV, model_summary=scene.MODEL,
                    phases=scene.PHASES, monitor="accuracy", monitor_phase="val")

    page = browser.new_page(viewport={"width": DASH_W, "height": DASH_H},
                            device_scale_factor=SCALE, color_scheme="dark")
    page.goto(dash.url)
    page.add_style_tag(content=CROP_TO_FOUR_CHARTS)
    # The page follows the system preference. Both themes are shot deliberately,
    # so the theme is stamped on the page rather than left to what the host reports.
    page.evaluate("document.documentElement.dataset.theme = 'dark'")
    dash.mark_started()                          # elapsed counts from here, as in the trainer

    table = scene.table(history, done)
    bm, be, ni = scene.best(val_acc, done)

    # Played at the rate a real run writes it, with the browser watching: this is the
    # only way the throughput and ETA readouts come to exist.
    steps = list(scene.hero_steps(history))
    print(f"  playing {len(steps)} steps at {scene.THROUGHPUT} it/s "
          f"(~{len(steps) * scene.WRITE_S:.0f} s)…")
    for step, step_metrics in steps:
        dash.update(
            scene.HERO_EPOCH, scene.EPOCHS, table, bm, be,
            epochs_no_improve=ni, is_gradient_phase=True,
            step=step, max_step=scene.TRAIN_STEPS, step_metrics=step_metrics,
            phase_name="train", learning_rate=scene.lr(scene.HERO_EPOCH), gpu_mem=scene.GPU_MEM,
        )
        page.wait_for_timeout(write_ms)          # pumps the browser: it polls, and measures

    page.wait_for_timeout(SETTLE_MS)
    shoot(page, ASSETS / "dashboard-dark.png", DASH_W, DASH_H)

    # Light and dark must be the same frame, so nothing is touched but the theme.
    page.evaluate("document.documentElement.dataset.theme = 'light'")
    page.wait_for_timeout(SETTLE_MS)
    shoot(page, ASSETS / "dashboard-light.png", DASH_W, DASH_H)

    page.evaluate("document.documentElement.dataset.theme = 'dark'")
    full = scene.table(history, scene.EPOCHS)
    bm, be, ni = scene.best(val_acc, scene.EPOCHS)
    dash.finalize(scene.EPOCHS, scene.EPOCHS, full, bm, be, epochs_no_improve=ni)
    page.add_style_tag(content=SHOW_RUN_PANEL)
    page.wait_for_timeout(SETTLE_MS)
    shoot(page, ASSETS / "dashboard-complete.png", DASH_W, DASH_H)
    page.close()


def declare_host() -> None:
    """Report ``scene``'s reference workstation as the machine the run is on.

    The one declared fact in the images, and it is declared for consistency: a reader
    compares the four images, so they must agree on the machine, and the dashboard's
    GPU meter and it/s readout only mean anything on a GPU.

    Nothing else is touched. The metrics, the model's parameter counts, the schedule,
    the artifacts — everything the images actually argue about — are the run's own.
    """
    base.env_summary = lambda *_args, **_kwargs: dict(scene.ENV)

    print_config = BaseTrainer.print_config

    def print_declared_config(self: BaseTrainer) -> None:
        self._config["device"] = scene.DEVICE   # or the banner would contradict itself
        print_config(self)

    BaseTrainer.print_config = print_declared_config


def discard(directory: Path) -> None:
    """Remove a finished run's directory, log and all.

    The trainer's log file is still open when the script ends, and Windows will not
    delete an open file — so logging is shut down first, which closes it.
    """
    logging.shutdown()
    shutil.rmtree(directory, onexc=_clear_read_only)


def _clear_read_only(function, path, _exc) -> None:
    """Retry a removal that Windows refused because the entry is read-only.

    ``copytree`` carries a folder's attributes across with it, and a synced
    ``data/`` (OneDrive marks its folders read-only) therefore arrives in the temp
    directory as folders Windows will not delete. Left to ``ignore_errors`` that
    refusal was silent, and every shoot left its copy of MNIST behind.
    """
    Path(path).chmod(stat.S_IWRITE)
    function(path)


def quickstart_transcript() -> str:
    """Run the README's Quick Start and return what it printed.

    The trainer logs to the console with plain ``print``, so its own stdout is the
    transcript — no parsing, no reconstruction. tqdm draws on stderr and so stays out
    of the capture, which is exactly right: a progress bar is motion, not a record.
    """
    # The run opens the dashboard in a browser, as the README says it does. Here that
    # would fling a window open mid-shoot, so only the *opening* is stubbed — the
    # dashboard itself still starts, and still prints its URL into the transcript.
    webbrowser.open = lambda *_args, **_kwargs: True
    declare_host()
    snippet = compile(quickstart.source(), f"{quickstart.README.name} (Quick Start)", "exec")
    # The snippet writes ``run/`` and downloads MNIST into ``data/``, both relative, and
    # it resumes whatever ``run/`` already holds — while the image must be a first
    # run's. So it is given a working directory of its own: a contributor's ``run/`` is
    # never read and never cleared. Their ``data/``, if they have one, is copied in to
    # spare the download.
    cwd = Path(tempfile.mkdtemp(prefix="train4all-quickstart-"))
    atexit.register(discard, cwd)
    if MNIST.exists():
        shutil.copytree(MNIST, cwd / MNIST.name)
    with contextlib.chdir(cwd), contextlib.redirect_stdout(io.StringIO()) as buffer:
        exec(snippet, {"__name__": "__main__"})
    # The README's image runs from the banner through the first epoch's artifacts.
    return buffer.getvalue().split("\n── Epoch 2 /")[0].rstrip()


def console_frame(browser) -> None:
    """The console shot — the Quick Start's own stdout, set in a terminal face.

    The transcript's height is fixed by its content, so the ratio is reached on the
    only free axis: the terminal background around it. The text block is centred in
    a 9:16 field rather than cropped tight to its ragged right edge.
    """
    print("  running the Quick Start on MNIST…")
    transcript = html.escape(quickstart_transcript())

    page = browser.new_page(device_scale_factor=SCALE)
    page.set_content(f"""
      <style>
        @font-face {{ font-family: fallback; src: local("Segoe UI Emoji"); }}
        html, body {{ margin: 0; background: {CONSOLE_BG}; }}
        body {{ display: flex; align-items: center; justify-content: center; }}
        pre {{ margin: 0; color: {CONSOLE_FG}; background: {CONSOLE_BG};
               font-family: "Cascadia Mono", Consolas, ui-monospace, monospace, fallback;
               font-size: {CONSOLE_FONT}px; line-height: {CONSOLE_LEAD};
               font-variant-ligatures: none; white-space: pre; }}
      </style>
      <pre id="t">{transcript}</pre>
    """)
    box = page.locator("#t").bounding_box()

    # Height is the transcript's; width is whatever 9:16 asks of it.
    height = round(box["height"]) + 2 * CONSOLE_PAD
    width  = round(height * 9 / 16)
    if width < box["width"] + 2 * CONSOLE_PAD:
        raise SystemExit(
            f"the transcript is {box['width']:.0f}px wide — too wide for a 9:16 field "
            f"{width}px across. Drop CONSOLE_FONT below {CONSOLE_FONT}."
        )
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_timeout(200)
    shoot(page, ASSETS / "logs.png", width, height)
    page.close()


def main() -> None:
    print("shooting the README images…")
    started = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        dashboard_frames(browser)
        console_frame(browser)
        browser.close()
    print(f"done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
