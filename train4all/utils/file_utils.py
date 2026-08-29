"""The filesystem work a run does around the edges: its records, its mirror, its cleanup.

What the standard library does not quite give the trainer — a JSON write that holds a
run's records to be worth less than the run, a file write or copy that leaves its
destination whole or untouched, a directory copy that can be aimed safely, and a delete
that survives read-only flags.

They are here rather than on the trainer because none of them knows what a run directory
holds: every function is handed a path, so the layout stays with whoever owns it.
"""

import contextlib
import json
import shutil
import stat
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from train4all.utils.log_utils import Printer

__all__ = ["atomic_replace", "copy_dir", "copy_file", "remove_dir", "write_json"]


def _on_remove_error(func: Callable[..., None], path: str, _exc: BaseException) -> None:
    """Make a read-only path writable and retry the failed removal.

    The signature is fixed by ``shutil.rmtree(onexc=...)``; the exception itself is
    not consulted, since the retry either succeeds or the ``OSError`` below gives up.
    """
    try:
        Path(path).chmod(stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def remove_dir(path: Path | str) -> None:
    """
    Recursively delete a directory, retrying after clearing read-only flags.

    Args:
        path: Directory to remove. No-op if it does not exist.
    """
    target = Path(path)
    if target.exists():
        shutil.rmtree(target, onexc=_on_remove_error)


def write_json(
    path: Path | str,
    data: Any,
    *,
    label: str,
    print_fn: Printer | None = None,
) -> bool:
    """
    Write *data* to *path* as indented JSON, creating parent directories as needed.

    A failed write is reported and swallowed rather than raised: these files are a
    run's records, and losing one is not worth losing the run that produced it.

    Args:
        path: Destination ``.json`` file.
        data: Any JSON-serializable object.
        label: What is being written, named in the failure message (e.g. ``"config"``).
        print_fn: Where a failure is reported. Silent when ``None``.

    Returns:
        ``True`` when the file was written, ``False`` when the write failed.
    """
    path = Path(path)
    try:
        # Inside the try: the promise covers the mkdir too, and a file standing
        # where the parent directory belongs would otherwise raise past it.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        if print_fn is not None:
            print_fn(f"Failed to write {label}: {e}\n")
        return False
    return True


def _remove(path: Path) -> None:
    """Delete a file or a directory, clearing the read-only flag that blocks it.

    The flag means "do not edit this", not "do not delete this" — and a mirror that
    honoured it would be unable to drop what its source dropped.
    """
    if path.is_dir() and not path.is_symlink():
        remove_dir(path)
        return
    try:
        path.unlink()
    except PermissionError:
        path.chmod(stat.S_IWRITE)
        path.unlink()


def _is_current(src: Path, dst: Path) -> bool:
    """Return ``True`` when *dst* already holds *src*'s bytes.

    rsync's quick check: same size, same modification time. It errs only by
    re-copying a file that did not need it — for it to *skip* a changed one, a
    rewrite would have to land on the same byte count *and* the same timestamp,
    and a filesystem too coarse to distinguish two writes simply reports every
    file as stale and copies it.
    """
    try:
        source, target = src.stat(), dst.stat()
    except OSError:  # dst absent or unreadable — copy it either way
        return False
    return source.st_size == target.st_size and source.st_mtime_ns == target.st_mtime_ns


@contextlib.contextmanager
def atomic_replace(dst: Path) -> Iterator[Path]:
    """Yield a temporary beside *dst* that replaces it on a clean exit, and is removed on any other.

    The temporary sits beside its destination rather than in the system temp
    directory, which puts the two on one filesystem and so makes the final
    ``Path.replace`` atomic: at every moment *dst* is the whole old file or the whole
    new one, never the middle of a write.
    """
    tmp = dst.with_name(f".{dst.name}.partial")
    try:
        yield tmp
        with contextlib.suppress(OSError):
            tmp.chmod(tmp.stat().st_mode | stat.S_IWRITE)
        try:
            tmp.replace(dst)
        except PermissionError:
            # Windows refuses to replace a read-only file; clear the flag and retry.
            with contextlib.suppress(OSError):
                dst.chmod(stat.S_IWRITE)
            tmp.replace(dst)
    finally:
        # Still present only when the write failed — a successful replace consumed
        # it. Leaving nothing half-written behind is the point, though a directory
        # copy would sweep it up as an entry the source does not have.
        tmp.unlink(missing_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    """Copy *src* onto *dst* through :func:`atomic_replace`, so *dst* is never partial.

    Metadata travels with the bytes (``shutil.copy2``), which is what lets
    :func:`copy_dir` tell an unchanged file from a changed one by size and mtime.
    """
    with atomic_replace(dst) as tmp:
        shutil.copy2(src, tmp)


def _sync_dir(src: Path, dst: Path, exclude: frozenset[str] = frozenset()) -> None:
    """Bring *dst* into line with *src*, entry by entry.

    Recurses with no *exclude*, since the exclusions name top-level entries only.
    """
    kept: set[str] = set()
    for item in src.iterdir():
        if item.name in exclude:
            continue
        kept.add(item.name)
        target = dst / item.name

        if item.is_dir():
            if target.exists() and not target.is_dir():
                _remove(target)  # a file stands where a directory belongs
            target.mkdir(parents=True, exist_ok=True)
            _sync_dir(item, target)
        else:
            if target.is_dir():
                _remove(target)  # a directory stands where a file belongs
            if not _is_current(item, target):
                copy_file(item, target)

    # Last, once every copy is in place: what the source no longer has, and what it
    # now excludes. Deleting first would open a window in which the destination holds
    # neither version — the window a mirror exists to close.
    for stale in dst.iterdir():
        if stale.name not in kept:
            _remove(stale)


def copy_dir(
    src: Path | str,
    dst: Path | str,
    exclude: Sequence[str] | None = None,
) -> Path:
    """
    Copy a directory tree, leaving *dst* an exact copy of *src*.

    Built to be repeated over one destination — a run mirrored after every epoch
    calls this dozens of times — so it copies only the files that differ, replaces
    each of them atomically, and deletes what the source no longer has only once
    every copy is in place.

    That order is the guarantee: **the destination is never emptied and never holds
    a partial file.** Interrupt this at any moment — the preempted VM, the expired
    session — and every file in *dst* is whole, either this copy's version or the
    previous one's. A copy that cleared the destination first would instead leave
    nothing at all, exactly when the host it guards against is the one that vanished.

    Args:
        src: Source directory.
        dst: Destination directory. Created if absent, brought into line with *src*
            if it already exists.
        exclude: Top-level entry names to skip. They are also removed from *dst* if
            an earlier copy put them there: the destination becomes *src* minus
            *exclude*, not whatever it has accumulated.

    Returns:
        Path to the destination directory.

    Raises:
        NotADirectoryError: If *src* is not a directory.
        ValueError: If *dst* lies inside *src*, which would copy the destination
            into itself and grow without bound on every repeat.
    """
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()

    if not src_path.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src_path}")

    # A destination nested inside the source copies the copy. Caught here rather
    # than left to grow: a per-epoch mirror would nest one level deeper each epoch.
    if dst_path == src_path or dst_path.is_relative_to(src_path):
        raise ValueError(
            f"Destination lies inside the source, so the copy would contain itself: "
            f"{dst_path} is within {src_path}"
        )

    dst_path.mkdir(parents=True, exist_ok=True)
    _sync_dir(src_path, dst_path, frozenset(exclude or ()))
    return dst_path
