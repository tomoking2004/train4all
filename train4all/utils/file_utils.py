import contextlib
import shutil
import stat
from collections.abc import Callable, Sequence
from pathlib import Path

__all__ = ["copy_dir", "remove_dir"]


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


def copy_dir(
    src: Path | str,
    dst: Path | str,
    exclude: Sequence[str] | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Recursively copy a directory.

    Args:
        src: Source directory.
        dst: Destination directory.
        exclude: Top-level entry names to skip.
        overwrite: Remove the destination before copying if it already exists.

    Returns:
        Path to the destination directory.

    Raises:
        NotADirectoryError: If *src* is not a directory.
        ValueError: If *dst* lies inside *src*, which would copy the destination
            into itself and grow without bound on every repeat.
    """
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()
    excluded = set(exclude or ())

    if not src_path.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src_path}")

    # A destination nested inside the source copies the copy. Caught here rather
    # than left to grow: a per-epoch mirror would nest one level deeper each epoch.
    if dst_path == src_path or dst_path.is_relative_to(src_path):
        raise ValueError(
            f"Destination lies inside the source, so the copy would contain itself: "
            f"{dst_path} is within {src_path}"
        )

    if overwrite and dst_path.exists():
        shutil.rmtree(dst_path, onexc=_on_remove_error)

    dst_path.mkdir(parents=True, exist_ok=True)

    for item in src_path.iterdir():
        if item.name in excluded:
            continue

        target = dst_path / item.name

        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
            with contextlib.suppress(OSError):
                target.chmod(target.stat().st_mode | stat.S_IWRITE)

    return dst_path
