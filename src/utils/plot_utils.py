from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
try:
    from .color_palettes import SCI_SOFT_COLORS
except ImportError:
    from color_palettes import SCI_SOFT_COLORS

def plot_profile(
    data: pd.DataFrame,
    x: str,
    height: str = "height",
    group: str | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Height (m)",
    invert_y: bool = False,
    colors: Iterable[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a vertical profile with value on x-axis and height on y-axis."""
    required = [x, height] + ([group] if group else [])
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 7))

    plot_data = data[required].dropna(subset=[x, height]).copy()
    plot_data = plot_data.sort_values(height)

    if group:
        color_list = list(colors) if colors is not None else None
        for i, (label, part) in enumerate(plot_data.groupby(group, sort=True)):
            color = color_list[i % len(color_list)] if color_list else None
            ax.plot(part[x], part[height], marker="o", linewidth=1.5, label=str(label), color=color)
        ax.legend(title=group)
    else:
        color_list = list(colors) if colors is not None else []
        color = color_list[0] if color_list else None
        ax.plot(plot_data[x], plot_data[height], marker="o", linewidth=1.5, color=color)

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.35)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_box(
    data: pd.DataFrame,
    value: str,
    group: str | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    labels: Iterable[str] | None = None,
    colors: Iterable[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a boxplot for one variable, optionally grouped by a column."""
    required = [value] + ([group] if group else [])
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if group:
        grouped = [
            part[value].dropna().to_numpy()
            for _, part in data[[group, value]].dropna(subset=[group]).groupby(group, sort=True)
        ]
        group_labels = (
            list(labels)
            if labels is not None
            else [str(label) for label in sorted(data[group].dropna().unique())]
        )
        boxes = ax.boxplot(grouped, tick_labels=group_labels, showmeans=True, patch_artist=True)
        color_list = list(colors) if colors is not None else []
        for i, patch in enumerate(boxes["boxes"]):
            if color_list:
                patch.set_facecolor(color_list[i % len(color_list)])
            patch.set_alpha(0.85)
        ax.set_xlabel(group)
    else:
        boxes = ax.boxplot(
            data[value].dropna().to_numpy(),
            tick_labels=[value],
            showmeans=True,
            patch_artist=True,
        )
        color_list = list(colors) if colors is not None else []
        if color_list:
            boxes["boxes"][0].set_facecolor(color_list[0])
            boxes["boxes"][0].set_alpha(0.85)

    ax.set_ylabel(ylabel or value)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax



if __name__ == "__main__":
    # Example usage: run this file directly from project root.
    df = pd.read_csv(r"D:\8\Desktop\CMathc\src\test\q1\output\q1_dataset.csv")
    plot_profile(
        df,
        x="temperature",
        height="height",
        group="station",
        title="Temperature profile",
        xlabel="Temperature",
        ylabel="Height (m)",
        colors=SCI_SOFT_COLORS,
        save_path=r"D:\8\Desktop\CMathc\src\test\q1\output\profile_temperature_height.png",
    )
    plt.close()
