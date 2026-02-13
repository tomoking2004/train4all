import os
import stat
import shutil
from pathlib import Path
from typing import List


def _on_remove_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def copy_dir(
    src: str | Path,
    dst: str | Path,
    exclude: List[str] | None = None,
) -> None:
    src = Path(src)
    dst = Path(dst)
    exclude = set(exclude or [])

    if dst.exists():
        shutil.rmtree(dst, onexc=_on_remove_error)

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name in exclude:
            continue

        target = dst / item.name

        if item.is_dir():
            shutil.copytree(item, target)
            for p in target.rglob("*"):
                try:
                    p.chmod(stat.S_IWRITE)
                except Exception:
                    pass
        else:
            shutil.copy2(item, target)
            target.chmod(stat.S_IWRITE)
