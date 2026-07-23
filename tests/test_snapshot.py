"""run_snapshot_dir must actually mirror the run.

It used to be inert: the argument was stored, `snapshot_run()` was never called by
the framework, and a 50-epoch cloud-backed run produced no mirror at all. The
README promised the sync, so these tests hold the code to it.
"""

import itertools
import os
import shutil

import pytest
from conftest import TinyTrainer, make_loader

from train4all import Phase
from train4all.utils import copy_dir, remove_dir


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


def test_the_configured_exclude_shapes_every_epochs_mirror(tmp_path):
    """The per-epoch mirror is unattended, so its exclusions can only be configuration.

    Left as a call-site argument alone, `exclude` was unreachable from the one caller
    that runs on its own — a run could mirror everything, or nothing.
    """
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    trainer = TinyTrainer(
        num_epochs=2, learning_rate=0.1, run_dir=run,
        run_snapshot_dir=mirror, run_snapshot_exclude=["checkpoints"],
        use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(8), training=True))

    assert (mirror / "metrics" / "epoch_metrics.json").exists()
    assert not (mirror / "checkpoints").exists(), "the configured exclude did not reach the loop"
    assert (run / "checkpoints" / "latest.pth").exists(), "excluded from the mirror, not the run"


def test_a_call_overrides_the_configured_exclude(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=run,
        run_snapshot_dir=mirror, run_snapshot_exclude=["checkpoints"],
        use_progress_bar=False,
    )
    trainer.train(Phase("train", make_loader(8), training=True))
    assert not (mirror / "checkpoints").exists()

    trainer.snapshot_run(exclude=[])   # this call alone mirrors everything
    assert (mirror / "checkpoints" / "latest.pth").exists()

    trainer.snapshot_run()             # a bare call is the configured mirror again
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


def test_copy_dir_leaves_the_destination_whole_when_it_fails(tmp_path, monkeypatch):
    """The invariant the whole mirror rests on.

    A copy that cleared the destination first would leave nothing at all when the
    host vanished mid-copy — empty exactly when the mirror was supposed to pay off.
    """
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("first", encoding="utf-8")
    (src / "sub" / "b.txt").write_text("first", encoding="utf-8")
    copy_dir(src, dst)

    (src / "a.txt").write_text("second", encoding="utf-8")
    (src / "sub" / "b.txt").write_text("second", encoding="utf-8")

    copies = itertools.count()
    real_copy2 = shutil.copy2

    def dies_after_the_first(*args, **kwargs):
        if next(copies):
            raise OSError("the host vanished")
        return real_copy2(*args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", dies_after_the_first)
    with pytest.raises(OSError, match="the host vanished"):
        copy_dir(src, dst)

    # Both files are still there, and each is whole — one this copy's version, one
    # the previous copy's. Which is which depends on iteration order; that neither
    # is missing or truncated does not.
    for path in (dst / "a.txt", dst / "sub" / "b.txt"):
        assert path.read_text(encoding="utf-8") in {"first", "second"}
    assert not list(dst.glob("**/*.partial")), "a half-written copy was left behind"


def test_copy_dir_copies_only_what_changed(tmp_path):
    """A 50-epoch mirror re-copying every earlier epoch's checkpoint is quadratic work."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "steady.txt").write_text("original", encoding="utf-8")
    copy_dir(src, dst)

    # Tamper with the copy without disturbing the size or mtime it is judged by, so
    # the tampering survives the next copy if and only if the file was skipped.
    target = dst / "steady.txt"
    before = target.stat()
    target.write_text("tampered", encoding="utf-8")  # same length as "original"
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

    copy_dir(src, dst)
    assert target.read_text(encoding="utf-8") == "tampered", "an unchanged file was re-copied"

    # Same length again, so only the modification time says it changed.
    (src / "steady.txt").write_text("replaced", encoding="utf-8")
    copy_dir(src, dst)
    assert target.read_text(encoding="utf-8") == "replaced", "a changed file was not copied"


def test_copy_dir_drops_what_the_source_no_longer_has(tmp_path):
    """Copying without pruning is how a mirror stops being one."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "gone").mkdir(parents=True)
    (src / "gone" / "f.txt").write_text("x", encoding="utf-8")
    (src / "stays.txt").write_text("x", encoding="utf-8")
    copy_dir(src, dst)

    remove_dir(src / "gone")
    copy_dir(src, dst)
    assert not (dst / "gone").exists(), "the mirror kept what the source dropped"
    assert (dst / "stays.txt").exists()


def test_copy_dir_rejects_a_file_as_source(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        copy_dir(f, tmp_path / "dst")
