from __future__ import annotations

from pathlib import Path
from typing import Iterable

from matplotlib import cm, colors as mcolors

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
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


# 热力图
def plot_heatmap(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar_label: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    df = data[[x, y, value]].dropna().copy()
    table = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(table.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(table.columns)))
    ax.set_yticks(range(len(table.index)))
    ax.set_xticklabels(table.columns, rotation=45)
    ax.set_yticklabels(table.index)

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)

    cbar = ax.figure.colorbar(im, ax=ax, pad=0.025)
    cbar.set_label(colorbar_label or value)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax

def _prepare_grid(data: pd.DataFrame, x: str, y: str, value: str):
    df = data[[x, y, value]].dropna().copy()

    if np.issubdtype(df[x].dtype, np.datetime64):
        df[x] = (df[x] - df[x].min()).dt.total_seconds() / 60
    else:
        df[x] = pd.to_numeric(df[x], errors="coerce")

    df[y] = pd.to_numeric(df[y], errors="coerce")
    table = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean").sort_index().sort_index(axis=1)
    X, Y = np.meshgrid(table.columns.to_numpy(), table.index.to_numpy())
    Z = table.to_numpy()
    return df, X, Y, Z


# 3D 曲面图
def plot_3d_surface(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    cmap: str = "jet",
    elev: float = 25,
    azim: float = -135,
    save_path: str | Path | None = None,
) -> plt.Axes:
    _, X, Y, Z = _prepare_grid(data, x, y, value)
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True)
    ax.figure.colorbar(surf, ax=ax, shrink=0.7, pad=0.08)
    _set_3d_labels(ax, title, xlabel or x, ylabel or y, zlabel or value)
    ax.view_init(elev=elev, azim=azim)

    if save_path:
        _save_figure(ax.figure, save_path)
    return ax


# 3D 网格图
def plot_3d_wireframe(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    cmap: str = "jet",
    elev: float = 25,
    azim: float = -135,
    save_path: str | Path | None = None,
) -> plt.Axes:
    _, X, Y, Z = _prepare_grid(data, x, y, value)
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

    norm = mcolors.Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(Z.shape[0]):
        ax.plot(X[i, :], Y[i, :], Z[i, :], color=mapper.to_rgba(np.nanmean(Z[i, :])), linewidth=0.9)
    for j in range(Z.shape[1]):
        ax.plot(X[:, j], Y[:, j], Z[:, j], color=mapper.to_rgba(np.nanmean(Z[:, j])), linewidth=0.9)

    ax.figure.colorbar(mapper, ax=ax, shrink=0.7, pad=0.08)
    _set_3d_labels(ax, title, xlabel or x, ylabel or y, zlabel or value)
    ax.view_init(elev=elev, azim=azim)

    if save_path:
        _save_figure(ax.figure, save_path)
    return ax


# 等高线图
def plot_contour(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "jet",
    levels: int = 16,
    save_path: str | Path | None = None,
) -> plt.Axes:
    _, X, Y, Z = _prepare_grid(data, x, y, value)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    ax.figure.colorbar(cs, ax=ax)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)

    if save_path:
        _save_figure(ax.figure, save_path)
    return ax


# 3D散点图
def plot_3d_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    cmap: str = "jet",
    elev: float = 25,
    azim: float = -135,
    save_path: str | Path | None = None,
) -> plt.Axes:
    df, _, _, _ = _prepare_grid(data, x, y, value)
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(df[x], df[y], df[value], c=df[value], cmap=cmap, s=28)
    ax.figure.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
    _set_3d_labels(ax, title, xlabel or x, ylabel or y, zlabel or value)
    ax.view_init(elev=elev, azim=azim)

    if save_path:
        _save_figure(ax.figure, save_path)
    return ax

def _set_3d_labels(ax: plt.Axes, title, xlabel, ylabel, zlabel) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)


# 保存图片
def _save_figure(fig: plt.Figure, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    df = pd.read_csv(r"D:\8\Desktop\CMathc\src\test\q1\output\q1_dataset.csv")
    plot_section(
        df,
        time="time",
        height="height",
        value="temperature",
        title="Temperature time-height section",
        xlabel="Time",
        ylabel="Height (km)",
        colorbar_label="Temperature (C)",
        save_path=r"D:\8\Desktop\CMathc\src\test\q1\output\section.png",
    )
    plt.close("all")
