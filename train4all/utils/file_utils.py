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
    """
    src_path = Path(src)
    dst_path = Path(dst)
    excluded = set(exclude or ())

    if not src_path.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src_path}")

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
