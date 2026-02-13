from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_metric_plot_title(metric_name: str, phase: str | None = None, prefix: str | None = None) -> str:
    parts: list[str] = []

    if prefix:
        parts.append(prefix)
    
    parts.append(metric_name)

    if phase:
        parts.append(f"({phase})")

    title = " ".join(parts)
    return title[0].upper() + title[1:]


def get_metric_plot_filename(metric_name: str, phase: str | None = None, prefix: str | None = None) -> str:
    parts: list[str] = []

    if prefix:
        parts.append(prefix)
    
    parts.append(metric_name)

    if phase:
        parts.append(phase)

    #return "_".join(parts) + ".pdf"
    return "_".join(parts) + ".png"


def save_curves_plot(
    curves: Dict[str, Sequence[float]],
    path: str | Path,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    alpha: float = 0.9,
) -> None:
    """
    Save a plot containing multiple labeled 1D curves.

    Args:
        curves (dict[str, Sequence[float]]): Mapping of label → numeric values.
        path (str | Path): Destination file path for the figure.
        title (str | None): Plot title.
        xlabel (str | None): X-axis label.
        ylabel (str | None): Y-axis label.
        alpha (float): Line transparency.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    for label, values in curves.items():
        if not values:
            continue
        plt.plot(range(1, len(values) + 1), values, label=label, alpha=alpha)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(path)
    plt.close("all")
