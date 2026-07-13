"""run_snapshot_dir must actually mirror the run.

It used to be inert: the argument was stored, `snapshot_run()` was never called by
the framework, and a 50-epoch cloud-backed run produced no mirror at all. The
README promised the sync, so these tests hold the code to it.
"""

import pytest
from conftest import TinyTrainer, make_loader

from train4all import Phase
from train4all.utils import copy_dir


def test_train_mirrors_the_run_after_every_epoch(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    trainer = TinyTrainer(
        num_epochs=2, learning_rate=0.1, run_dir=run,
        run_snapshot_dir=mirror, use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(8), training=True))

    assert mirror.is_dir(), "run_snapshot_dir was set but nothing was mirrored"
    # The checkpoints are the point of a mirror — they must not be excluded.
    assert (mirror / "checkpoints" / "latest.pth").exists()
    assert (mirror / "metrics" / "epoch_metrics.json").exists()
    assert (mirror / "config.json").exists()


def test_the_mirror_keeps_up_with_the_latest_epoch(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    trainer = TinyTrainer(
        num_epochs=3, learning_rate=0.1, run_dir=run,
        run_snapshot_dir=mirror, use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(8), training=True))

    mirrored = (mirror / "metrics" / "epoch_metrics.json").read_text(encoding="utf-8")
    live = (run / "metrics" / "epoch_metrics.json").read_text(encoding="utf-8")
    assert mirrored == live, "the mirror lags the run it mirrors"


def test_no_snapshot_dir_means_no_mirror(tmp_path):
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=tmp_path / "run", use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(8), training=True))
    assert list(tmp_path.iterdir()) == [tmp_path / "run"]


def test_exclude_leaves_the_heavy_parts_behind(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run, use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(8), training=True))

    trainer.run_snapshot_dir = mirror
    trainer.snapshot_run(exclude=["checkpoints"])
    assert (mirror / "metrics").is_dir()
    assert not (mirror / "checkpoints").exists()


def test_a_mirror_inside_the_run_is_rejected(tmp_path):
    """Nested inside its own source, each epoch's copy would contain the last."""
    run = tmp_path / "run"
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run,
        run_snapshot_dir=run / "mirror",           # inside run_dir
        use_progress_bar=False,
    )
    with pytest.raises(ValueError, match="inside the source"):
        trainer.train(Phase("train", make_loader(8), training=True))


def test_copy_dir_rejects_a_destination_inside_the_source(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the source"):
        copy_dir(src, src / "sub" / "dst")
    with pytest.raises(ValueError, match="inside the source"):
        copy_dir(src, src)


def test_copy_dir_copies_and_excludes(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "keep").mkdir(parents=True)
    (src / "drop").mkdir()
    (src / "keep" / "f.txt").write_text("hello", encoding="utf-8")
    (src / "top.txt").write_text("top", encoding="utf-8")

    result = copy_dir(src, dst, exclude=["drop"])
    assert result == dst.resolve()
    assert (dst / "keep" / "f.txt").read_text(encoding="utf-8") == "hello"
    assert (dst / "top.txt").exists()
    assert not (dst / "drop").exists()


def test_copy_dir_rejects_a_file_as_source(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        copy_dir(f, tmp_path / "dst")
