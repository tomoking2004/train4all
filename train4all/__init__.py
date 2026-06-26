from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from train4all.trainer import BaseTrainer, Checkpoint
from train4all.utils.dashboard import Dashboard, DashboardConfig

try:
    __version__: str = _version("train4all")
except _PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["BaseTrainer", "Checkpoint", "Dashboard", "DashboardConfig", "__version__"]
