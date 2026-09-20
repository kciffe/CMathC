from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .color_palettes import get_sci_height_colors, SCI_HEIGHT_COLORS,get_sci_deep_colors,SCI_DEEP_COLORS
except ImportError:
    from color_palettes import get_sci_height_colors, SCI_HEIGHT_COLORS,get_sci_deep_colors,SCI_DEEP_COLORS


def set_chinese_font() -> None:
    """Configure Matplotlib fonts for Chinese labels on Windows."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


set_chinese_font()

# 阔线图
def plot_profile(
    data: pd.DataFrame,
    x: str,
    height: str = "height",
    group: str | None = None,
    legend: str | None = None,
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
        ax.legend(title=legend or group)
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
        _save_figure(ax.figure, save_path)

    return ax

# 箱线图
def plot_box(
    data: pd.DataFrame,
    value: str,
    group: str | None = None,
    legend: str | None = None,
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

    meanprops = {
        "marker": "D",
        "markerfacecolor": "#FFFFFF",
        "markeredgecolor": "#444444",
        "markersize": 4,
    }
    common_props = {
        "showmeans": True,
        "patch_artist": True,
        "meanprops": meanprops,
        "medianprops": {"color": "#D55E00", "linewidth": 1.2},
        "boxprops": {"linewidth": 1.0},
        "whiskerprops": {"linewidth": 0.9},
        "capprops": {"linewidth": 0.9},
    }

    if group:
        grouped_data = data[[group, value]].dropna(subset=[group])
        grouped = [part[value].dropna().to_numpy() for _, part in grouped_data.groupby(group, sort=True)]
        group_labels = list(labels) if labels is not None else [str(label) for label in sorted(grouped_data[group].dropna().unique())]
        boxes = ax.boxplot(grouped, tick_labels=group_labels, **common_props)
        color_list = list(colors) if colors is not None else []
        for i, patch in enumerate(boxes["boxes"]):
            if color_list:
                patch.set_facecolor(color_list[i % len(color_list)])
            patch.set_alpha(0.85)
        ax.set_xlabel(legend or group)
    else:
        boxes = ax.boxplot(data[value].dropna().to_numpy(), tick_labels=[value], **common_props)
        color_list = list(colors) if colors is not None else []
        if color_list:
            boxes["boxes"][0].set_facecolor(color_list[0])
            boxes["boxes"][0].set_alpha(0.85)

    ax.set_ylabel(ylabel or value)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax

# 剖面图
def plot_section(
    data: pd.DataFrame,
    time: str,
    height: str,
    value: str,
    *,
    title: str | None = None,
    xlabel: str = "时间",
    ylabel: str = "高度 (km)",
    colorbar_label: str | None = None,
    height_scale: float = 1000.0,
    levels: int = 18,
    contour_levels: int = 8,
    cmap: str = "Spectral_r",
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画时间-高度剖面图"""
    df = data[[time, height, value]].dropna().copy()
    df[time] = pd.to_datetime(df[time])
    df[height] = df[height] / height_scale

    table = df.pivot_table(index=height, columns=time, values=value, aggfunc="mean").sort_index().sort_index(axis=1)
    x = mdates.date2num(table.columns)
    y = table.index
    z = table.values

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    cf = ax.contourf(x, y, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
    cs = ax.contour(x, y, z, levels=contour_levels, colors="black", linewidths=0.7, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.4g")

    cbar = ax.figure.colorbar(cf, ax=ax, pad=0.025)
    cbar.set_label(colorbar_label or value)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.25, color="#555555")
    if title:
        ax.set_title(title)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


def _save_figure(fig: plt.Figure, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    df = pd.read_csv(r"D:\8\Desktop\CMathc\src\test\q1\output\q1_dataset.csv")

    plot_profile(
        df,
        x="temperature",
        height="height",
        group="station",
        legend="station",
        title="Temperature profile",
        xlabel="Temperature (C)",
        ylabel="Height (m)",
        colors=get_sci_deep_colors(df["station"].nunique()),
        save_path=r"D:\8\Desktop\CMathc\src\test\q1\output\profile_temperature_height.png",
    )

    plot_section(
        df,
        time="time",
        height="height",
        value="temperature",
        title="Temperature time-height section",
        xlabel="Time",
        ylabel="Height (km)",
        colorbar_label="Temperature (C)",
        cmap="turbo",
        save_path=r"D:\8\Desktop\CMathc\src\test\q1\output\section.png",
    )
    plt.close("all")
