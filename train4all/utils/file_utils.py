import os
import stat
import shutil
from pathlib import Path
from collections.abc import Sequence


def _on_remove_error(func, path: str, exc_info) -> None:
    """
    Handle read-only file removal errors during rmtree (Python 3.12+).

    Attempts to make the path writable, then retries the failed function.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        # Give up silently — rmtree will re-raise if necessary
        pass


def copy_dir(
    src: Path | str,
    dst: Path | str,
    exclude: Sequence[str] | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Copy a directory recursively (Python 3.12+ optimized version).

    Args:
        src: Source directory.
        dst: Destination directory.
        exclude: Iterable of top-level names to skip.
        overwrite: If True, remove existing destination before copying.

    Returns:
        Path to the destination directory.

    Raises:
        NotADirectoryError: If src is not a directory.
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
            shutil.copytree(
                item,
                target,
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(item, target)

            # Ensure file is writable (without destroying other permission bits)
            try:
                current_mode = target.stat().st_mode
                target.chmod(current_mode | stat.S_IWRITE)
            except OSError:
                pass

    return dst_path
