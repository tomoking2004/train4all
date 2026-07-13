"""The installed package version — the single source both the public
``train4all.__version__`` and the dashboard's footer read.

Kept in its own module, importing nothing from the package, so the dashboard can
reach it without a cycle through ``train4all/__init__.py``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("train4all")
except PackageNotFoundError:   # a source tree that was never installed
    __version__ = "unknown"

__all__ = ["__version__"]
