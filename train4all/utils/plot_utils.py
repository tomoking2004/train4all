from pathlib import Path
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_metric_plot_title(
    metric_name: str,
    phase: str | None = None,
    prefix: str | None = None,
) -> str:
    parts: list[str] = []

    if prefix:
        parts.append(prefix)

    parts.append(metric_name)

    if phase:
        parts.append(f"({phase})")

    title = " ".join(parts).strip()
    return title[:1].upper() + title[1:] if title else ""


def get_metric_plot_filename(
    metric_name: str,
    phase: str | None = None,
    prefix: str | None = None,
    extension: str = "png",
) -> str:
    parts: list[str] = []

    if prefix:
        parts.append(prefix)

    parts.append(metric_name)

    if phase:
        parts.append(phase)

    filename = "_".join(parts)
    return f"{filename}.{extension}"


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
    Save a plot containing multiple labeled 1D curves.
    Fully state-isolated (no global pyplot state leakage).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    has_data = False

    for label, values in curves.items():
        if len(values) == 0:
            continue

        has_data = True

        ax.plot(
            range(1, len(values) + 1),
            values,
            label=label,
            alpha=alpha,
        )

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
    plt.close(fig)

    return path
