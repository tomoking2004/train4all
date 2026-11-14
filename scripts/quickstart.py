"""
Run the README's Quick Start — the snippet itself, not a copy of it.

    pip install torchvision
    python scripts/quickstart.py

The code is read out of ``README.md`` and executed, so what runs here is what a
reader is shown, and the two cannot drift apart. ``shoot.py`` takes the snippet from
this module for the same reason.

It runs from the repository root, where the snippet's relative ``data/`` and ``run/``
land in the two places ``.gitignore`` already expects them. And it is always a first
run: the snippet resumes whatever ``run/`` holds, so that directory is deleted before
it starts.
"""

import contextlib
from pathlib import Path

from train4all.utils import remove_dir

ROOT   = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def source() -> str:
    """The Quick Start as the README prints it: that section's first Python block."""
    section = README.read_text(encoding="utf-8").split("\n## Quick Start\n", 1)[1]
    return section.split("```python\n", 1)[1].split("\n```", 1)[0]


def main() -> None:
    snippet = compile(source(), f"{README.name} (Quick Start)", "exec")
    with contextlib.chdir(ROOT):
        remove_dir("run")
        exec(snippet, {"__name__": "__main__"})


if __name__ == "__main__":
    main()
