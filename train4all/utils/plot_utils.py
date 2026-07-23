from collections.abc import Mapping, Sequence
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

__all__ = ["get_metric_plot_filename", "get_metric_plot_title", "save_curves_plot"]


def get_metric_plot_title(
    metric_name: str,
    phase_name: str | None = None,
    prefix: str | None = None,
) -> str:
    """
    Build a human-readable plot title from its components.

    Args:
        metric_name: Name of the metric (e.g. ``"loss"``).
        phase_name: Optional phase label appended in parentheses (e.g. ``"train"``).
        prefix: Optional prefix prepended to the title (e.g. ``"step-level"``).

    Returns:
        Capitalized title string.
    """
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(metric_name)
    if phase_name:
        parts.append(f"({phase_name})")
    title = " ".join(parts).strip()
    return title[:1].upper() + title[1:] if title else ""


def get_metric_plot_filename(
    metric_name: str,
    phase_name: str | None = None,
    prefix: str | None = None,
    extension: str = "png",
) -> str:
    """
    Build a filename for a metric plot.

    Args:
        metric_name: Name of the metric (e.g. ``"loss"``).
        phase_name: Optional phase label appended to the stem (e.g. ``"train"``).
        prefix: Optional prefix prepended to the stem (e.g. ``"step"``).
        extension: File extension without the leading dot. Defaults to ``"png"``.

    Returns:
        Filename string (e.g. ``"step_loss_train.png"``).
    """
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(metric_name)
    if phase_name:
        parts.append(phase_name)
    return f"{'_'.join(parts)}.{extension}"


def save_curves_plot(
    curves: Mapping[str, Sequence[float]],
    path: Path | str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    alpha: float = 0.9,
    dpi: int = 150,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> Path:
    """
    Save a plot of one or more labelled 1-D curves to disk.

    Each curve is plotted against its 1-based index. The function is fully
    state-isolated: it creates its own ``Figure`` with an explicit Agg canvas,
    bypassing pyplot's global backend selection entirely.

    Args:
        curves: Mapping of label to sequence of float values.
        path: Destination file path (parent directories are created if needed).
        title: Optional plot title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        alpha: Line opacity.
        dpi: Output resolution in dots per inch.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        Resolved path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    has_data = False

    for label, values in curves.items():
        if not values:
            continue
        has_data = True
        ax.plot(range(1, len(values) + 1), values, label=label, alpha=alpha)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if has_data:
        ax.legend()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)

    return path
