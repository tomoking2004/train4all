"""
Open the live dashboard on the staged run in ``demo_scene`` — the eyeball-it tool for
working on the dashboard itself. No model, no training: it feeds the real
``Dashboard`` exactly the data a genuine run would, in real time.

    python scripts/demo_dashboard.py

It opens the page, plays into the hero frame — mid-training in epoch 33/40: the run
bar at 81%, the epoch bar part-way through its train segment, charts nearly full, the
live readings, and the step-loss strip chart at a full window — and holds there. Press
``T`` in the page for light / dark; press Enter in the terminal for the completed
view (the run bar full, the best epoch marked on full curves).

The README's images are *not* taken here — ``shoot.py`` drives this same staged run
through a real browser and writes them straight to ``assets/``. The page itself is
written to a temp folder that is removed on exit; no run is left beside the scripts.
"""

import atexit
import shutil
import tempfile
import time
from pathlib import Path

import demo_scene as scene

from train4all.dashboard import Dashboard, DashboardConfig

# The run directory is scratch: the page and its JSON only have to exist while the
# browser is looking at them. So it lives in the system temp folder and is removed
# when the script exits, rather than accumulating beside the scripts.
RUN_DIR = Path(tempfile.mkdtemp(prefix="train4all-demo-"))
atexit.register(shutil.rmtree, RUN_DIR, ignore_errors=True)


def main() -> None:
    history = scene.series()
    val_acc = history["accuracy"]["val"]
    done = scene.HERO_EPOCH - 1                      # 32 completed epochs at the freeze

    dash = Dashboard(DashboardConfig(poll_interval_ms=round(scene.WRITE_S * 1000)), RUN_DIR)
    dash.initialize(scene.CONFIG, env_summary=scene.ENV, model_summary=scene.MODEL,
                    phases=scene.PHASES,
                    monitor="accuracy", monitor_phase="val")
    time.sleep(1.5)   # let the browser tab come up before data flows
    dash.mark_started()   # elapsed counts from loop start, exactly like the trainer

    # the completed epochs are history → pre-fill the table (no fake stepping needed)
    table = scene.table(history, done)
    bm, be, ni = scene.best(val_acc, done)          # best so far: epoch 32, still improving (no plateau)

    # the live landing, played at the speed a real run writes it — so the speed /
    # ETA readouts, the two bars, and the strip chart build up authentically.
    # Then hold; the dashboard's keepalive keeps the frame live.
    for step, step_metrics in scene.hero_steps(history):
        dash.update(
            scene.HERO_EPOCH, scene.EPOCHS, table, bm, be,
            epochs_no_improve=ni, is_gradient_phase=True,
            step=step, max_step=scene.TRAIN_STEPS, step_metrics=step_metrics,
            phase_name="train", learning_rate=scene.lr(scene.HERO_EPOCH), gpu_mem=scene.GPU_MEM,
        )
        time.sleep(scene.WRITE_S)

    print(f"\n  ▸ LIVE hero frame is up: {dash.url}")
    print("    Press T in the page to toggle light / dark.")
    input("    Press Enter to switch to the COMPLETED view… ")

    full = scene.table(history, scene.EPOCHS)
    bm, be, ni = scene.best(val_acc, scene.EPOCHS)
    dash.finalize(scene.EPOCHS, scene.EPOCHS, full, bm, be, epochs_no_improve=ni)

    print(f"\n  ▸ COMPLETED view is up: {dash.url}")
    input("    Press Enter to exit. ")


if __name__ == "__main__":
    main()
