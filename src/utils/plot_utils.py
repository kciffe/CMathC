from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
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

# 廓线图
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


# 折线图
def plot_line(
    data: pd.DataFrame,
    x: str,
    y: str | Iterable[str],
    group: str | None = None,
    legend: str | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: Iterable[str] | None = None,
    colors: Iterable[str] | None = None,
    marker: str | None = None,
    linestyle: str = "-",
    linestyles: Iterable[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画普通折线图，支持单指标、多指标和分组曲线。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    color_list = list(colors) if colors is not None else []
    style_list = list(linestyles) if linestyles is not None else []

    if group:
        for i, (name, part) in enumerate(data.groupby(group, sort=True)):
            part = part[[x, y]].dropna().sort_values(x)
            color = color_list[i % len(color_list)] if color_list else None
            style = style_list[i % len(style_list)] if style_list else linestyle
            ax.plot(part[x], part[y], marker=marker, linestyle=style, label=str(name), color=color)
        ax.legend(title=legend or group)
    else:
        ys = [y] if isinstance(y, str) else list(y)
        label_list = list(labels) if labels is not None else ys

        for i, col in enumerate(ys):
            part = data[[x, col]].dropna().sort_values(x)
            color = color_list[i % len(color_list)] if color_list else None
            style = style_list[i % len(style_list)] if style_list else linestyle
            ax.plot(part[x], part[col], marker=marker, linestyle=style, label=label_list[i], color=color)

        if len(ys) > 1 or labels is not None:
            ax.legend()

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or "")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 散点图
def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str | None = None,
    legend: str | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: Iterable[str] | None = None,
    alpha: float = 0.75,
    size: float = 35,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画普通散点图，可按类别分组。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    color_list = list(colors) if colors is not None else []

    if group:
        for i, (name, part) in enumerate(data.groupby(group, sort=True)):
            part = part[[x, y]].dropna()
            color = color_list[i % len(color_list)] if color_list else None
            ax.scatter(part[x], part[y], s=size, alpha=alpha, label=str(name), color=color)
        ax.legend(title=legend or group)
    else:
        df = data[[x, y]].dropna()
        color = color_list[0] if color_list else None
        ax.scatter(df[x], df[y], s=size, alpha=alpha, color=color)

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 柱状图
def plot_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: Iterable[str] | None = None,
    colors: Iterable[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画分类柱状图。"""
    df = data[[x, y]].dropna().copy()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    pos = np.arange(len(df))
    color_list = list(colors) if colors is not None else None

    ax.bar(pos, df[y], color=color_list)
    ax.set_xticks(pos)
    ax.set_xticklabels(list(labels) if labels is not None else df[x].astype(str))

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 直方图
def plot_hist(
    data: pd.DataFrame,
    value: str,
    bins: int = 20,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "频数",
    colors: Iterable[str] | None = None,
    alpha: float = 0.8,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画单变量分布直方图。"""
    values = data[value].dropna()
    color_list = list(colors) if colors is not None else []
    color = color_list[0] if color_list else None

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.hist(values, bins=bins, color=color, alpha=alpha, edgecolor="white")
    ax.set_xlabel(xlabel or value)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 联合直方图
def plot_joint_hist(
    data: pd.DataFrame,
    x: str,
    y: str,
    bins: int = 20,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "Blues",
    colors: Iterable[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画二维联合直方图，并显示 x、y 的边缘直方图。"""
    df = data[[x, y]].dropna().copy()
    color_list = list(colors) if colors is not None else []
    hist_color = color_list[0] if color_list else None

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)

    ax_top = fig.add_subplot(gs[0, :3])
    ax_main = fig.add_subplot(gs[1:, :3])
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    ax_main.hist2d(df[x], df[y], bins=bins, cmap=cmap)
    ax_top.hist(df[x], bins=bins, color=hist_color, edgecolor="black", alpha=0.85)
    ax_right.hist(df[y], bins=bins, orientation="horizontal", color=hist_color, edgecolor="black", alpha=0.85)

    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    ax_main.set_xlabel(xlabel or x)
    ax_main.set_ylabel(ylabel or y)
    if title:
        ax_top.set_title(title)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax_main


# 核密度图
def plot_kde(
    data: pd.DataFrame,
    value: str,
    group: str | None = None,
    legend: str | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "密度",
    colors: Iterable[str] | None = None,
    fill: bool = True,
    alpha: float = 0.2,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画核密度曲线，可用于比较不同分组的分布形态。"""
    df = data[[value] + ([group] if group else [])].dropna().copy()
    values_all = df[value].to_numpy(dtype=float)

    xmin, xmax = values_all.min(), values_all.max()
    pad = (xmax - xmin) * 0.08 + 1e-9
    x_grid = np.linspace(xmin - pad, xmax + pad, 400)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    color_list = list(colors) if colors is not None else []
    groups = df.groupby(group, sort=True, observed=True) if group else [(value, df)]

    for i, (name, part) in enumerate(groups):
        values = part[value].to_numpy(dtype=float)
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        bandwidth = 1.06 * std * len(values) ** (-1 / 5) + 1e-6

        diff = (x_grid[:, None] - values[None, :]) / bandwidth
        density = np.exp(-0.5 * diff ** 2).sum(axis=1)
        density /= len(values) * bandwidth * np.sqrt(2 * np.pi)

        color = color_list[i % len(color_list)] if color_list else None
        label = str(name) if group else value

        ax.plot(x_grid, density, linewidth=1.5, label=label, color=color)
        if fill:
            ax.fill_between(x_grid, density, alpha=alpha, color=color)

    if group:
        ax.legend(title=legend or group)

    ax.set_xlabel(xlabel or value)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 残差图
def plot_residual(
    data: pd.DataFrame,
    actual: str,
    predicted: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = "预测值",
    ylabel: str = "残差",
    colors: Iterable[str] | None = None,
    alpha: float = 0.75,
    size: float = 35,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画预测值与残差散点图，残差 = 实际值 - 预测值。"""
    df = data[[actual, predicted]].dropna().copy()
    residual = df[actual] - df[predicted]

    color_list = list(colors) if colors is not None else []
    color = color_list[0] if color_list else None

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(df[predicted], residual, s=size, alpha=alpha, color=color)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 置信区间带图
def plot_band(
    data: pd.DataFrame,
    x: str,
    y: str,
    lower: str,
    upper: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: Iterable[str] | None = None,
    alpha: float = 0.2,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画均值曲线和上下界阴影带，可用于均值±标准差或置信区间。"""
    df = data[[x, y, lower, upper]].dropna().sort_values(x).copy()
    color_list = list(colors) if colors is not None else []
    color = color_list[0] if color_list else None

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    line, = ax.plot(df[x], df[y], linewidth=1.6, color=color)
    ax.fill_between(df[x], df[lower], df[upper], color=line.get_color(), alpha=alpha)

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# Q-Q图
def plot_qq(
    data: pd.DataFrame,
    value: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = "理论分位数",
    ylabel: str = "样本分位数",
    colors: Iterable[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画正态 Q-Q 图，用于观察样本或残差是否近似正态分布。"""
    values = np.sort(data[value].dropna().to_numpy(dtype=float))
    n = len(values)
    p = (np.arange(n) + 0.5) / n
    theoretical = np.array([NormalDist().inv_cdf(i) for i in p])

    color_list = list(colors) if colors is not None else []
    color = color_list[0] if color_list else None

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(theoretical, values, s=28, alpha=0.75, color=color)
    k, b = np.polyfit(theoretical, values, 1)
    ax.plot(theoretical, k * theoretical + b, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# Pareto前沿图
def plot_pareto(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    minimize_x: bool = True,
    minimize_y: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: Iterable[str] | None = None,
    alpha: float = 0.65,
    size: float = 32,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画双目标散点图并标出 Pareto 前沿。"""
    df = data[[x, y]].dropna().copy()
    df = df.sort_values(x, ascending=minimize_x)

    frontier = []
    best_y = np.inf if minimize_y else -np.inf

    for idx, row in df.iterrows():
        better = row[y] < best_y if minimize_y else row[y] > best_y
        if better:
            frontier.append(idx)
            best_y = row[y]

    front = df.loc[frontier]
    color_list = list(colors) if colors is not None else []
    point_color = color_list[0] if len(color_list) > 0 else None
    front_color = color_list[1] if len(color_list) > 1 else None

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(df[x], df[y], s=size, alpha=alpha, color=point_color, label="可行解")
    ax.plot(front[x], front[y], marker="o", linewidth=1.8, color=front_color, label="Pareto前沿")

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.legend()
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


# 相关性热力图
def plot_corr_heatmap(
    data: pd.DataFrame,
    cols: list[str],
    *,
    labels: list[str] | None = None,
    title: str | None = None,
    cmap: str = "coolwarm",
    vmin: float = -1,
    vmax: float = 1,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    corr = data[cols].corr()
    names = labels or cols

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(names)):
            value = corr.iloc[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2g}", ha="center", va="center", color=color)

    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, pad=0.025)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 气泡散点图
def plot_bubble(
    data: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    color: str | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar_label: str | None = None,
    cmap: str = "viridis",
    min_size: float = 30,
    max_size: float = 500,
    alpha: float = 0.75,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画气泡散点图，点大小和颜色可表示额外变量。"""
    color = color or size
    cols = list(dict.fromkeys([x, y, size, color]))
    df = data[cols].dropna().copy()

    # 把数值缩放到合适的点大小
    s = df[size].astype(float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    s = min_size + s * (max_size - min_size)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    sc = ax.scatter(
        df[x],
        df[y],
        s=s,
        c=df[color],
        cmap=cmap,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.6,
    )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    cbar = ax.figure.colorbar(sc, ax=ax, pad=0.025)
    cbar.set_label(colorbar_label or color)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 风向风速极坐标图
def plot_wind_polar(
    data: pd.DataFrame,
    direction: str,
    speed: str,
    *,
    title: str | None = None,
    colorbar_label: str | None = None,
    cmap: str = "plasma",
    alpha: float = 0.75,
    size: float = 45,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画风向风速极坐标散点图。"""
    df = data[[direction, speed]].dropna().copy()
    theta = np.deg2rad(df[direction])
    r = df[speed]

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="polar")

    sc = ax.scatter(theta, r, c=r, s=size, cmap=cmap, alpha=alpha)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(22.5)
    ax.grid(True, linestyle="-", alpha=0.45)

    if title:
        ax.set_title(title)

    cbar = ax.figure.colorbar(sc, ax=ax, pad=0.08)
    cbar.set_label(colorbar_label or speed)

    if save_path:
        _save_figure(ax.figure, save_path)

    return ax


# 地图热力图
def plot_map_heatmap(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str | None = None,
    *,
    background: str | Path | None = None,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "jet",
    alpha: float = 0.55,
    radius: float = 0.04,
    grid_size: int = 300,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """画地图底图上的空间热力图。"""
    cols = [x, y] + ([value] if value else [])
    df = data[cols].dropna().copy()

    if extent is None:
        pad_x = (df[x].max() - df[x].min()) * 0.08
        pad_y = (df[y].max() - df[y].min()) * 0.08
        extent = (df[x].min() - pad_x, df[x].max() + pad_x, df[y].min() - pad_y, df[y].max() + pad_y)

    xs = np.linspace(extent[0], extent[1], grid_size)
    ys = np.linspace(extent[2], extent[3], grid_size)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X, dtype=float)

    weights = df[value].to_numpy() if value else np.ones(len(df))
    for xi, yi, wi in zip(df[x], df[y], weights):
        Z += wi * np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2 * radius ** 2))

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    if background:
        img = plt.imread(background)
        ax.imshow(img, extent=extent, origin="upper")

    im = ax.imshow(Z, extent=extent, origin="lower", cmap=cmap, alpha=alpha)
    ax.scatter(df[x], df[y], s=8, c="white", alpha=0.5, linewidths=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, pad=0.025)

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
    z: str,
    color: str | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    colorbar_label: str | None = None,
    cmap: str = "jet",
    elev: float = 25,
    azim: float = -135,
    save_path: str | Path | None = None,
) -> plt.Axes:
    color = color or z
    cols = list(dict.fromkeys([x, y, z, color]))
    df = data[cols].dropna().copy()

    if np.issubdtype(df[x].dtype, np.datetime64):
        df[x] = (df[x] - df[x].min()).dt.total_seconds() / 60

    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(df[x], df[y], df[z], c=df[color], cmap=cmap, s=28)
    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label(colorbar_label or color)
    _set_3d_labels(ax, title, xlabel or x, ylabel or y, zlabel or z)
    ax.view_init(elev=elev, azim=azim)

    if save_path:
        _save_figure(ax.figure, save_path)
    return ax
def _set_3d_labels(ax: plt.Axes, title, xlabel, ylabel, zlabel) -> None:
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, labelpad=14, rotation=90)
    if title:
        ax.set_title(title)


# 保存图片
def _save_figure(fig: plt.Figure, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    df = pd.read_csv(
        r"D:\8\Desktop\CMathc\src\test\q1\output\q1_dataset_features.csv"
    )
    df["time"] = pd.to_datetime(df["time"])

    output = r"D:\8\Desktop\CMathc\src\utils\output"

    # 多指标折线图：不同风场变量使用不同线型
    line_data = df[
        (df["station"] == "a")
        & (df["time"] == df["time"].min())
        & (df["height"] <= 1500)
    ].copy()

    plot_line(
        line_data,
        x="height",
        y=["wind_speed", "u", "v", "vertical_velocity"],
        labels=["风速", "u风分量", "v风分量", "垂直速度"],
        title="不同风场指标随高度变化",
        xlabel="高度 (m)",
        ylabel="速度 (m/s)",
        colors=get_sci_deep_colors(4),
        linestyles=["-", "--", "-.", ":"],
        save_path=rf"{output}\折线图-多指标不同线型.png",
    )

    # 散点图：温度与相对湿度关系，不同站点分别显示
    plot_scatter(
        df,
        x="temperature",
        y="relative_humidity",
        group="station",
        legend="站点",
        title="温度与相对湿度关系",
        xlabel="温度 (°C)",
        ylabel="相对湿度 (%)",
        colors=get_sci_deep_colors(df["station"].nunique()),
        save_path=rf"{output}\散点图-温度与湿度关系.png",
    )

    # 柱状图：比较两个站点的平均风速
    bar_data = df.groupby("station", as_index=False)["wind_speed"].mean()

    plot_bar(
        bar_data,
        x="station",
        y="wind_speed",
        labels=["A站", "B站"],
        title="不同站点平均风速对比",
        xlabel="站点",
        ylabel="平均风速 (m/s)",
        colors=get_sci_deep_colors(len(bar_data)),
        save_path=rf"{output}\柱状图-站点平均风速.png",
    )

    # 直方图：A站风速分布
    df_a = df[df["station"] == "a"].copy()

    plot_hist(
        df_a,
        value="wind_speed",
        bins=20,
        title="A站风速分布",
        xlabel="风速 (m/s)",
        ylabel="频数",
        colors=get_sci_deep_colors(1),
        save_path=rf"{output}\直方图-A站风速分布.png",
    )

    # 联合直方图：观察温度和相对湿度的联合分布及各自边缘分布
    joint_data = df[
        (df["station"] == "a")
        & (df["height"] <= 1500)
    ].copy()

    plot_joint_hist(
        joint_data,
        x="temperature",
        y="relative_humidity",
        bins=18,
        title="A站温度与相对湿度联合分布",
        xlabel="温度 (°C)",
        ylabel="相对湿度 (%)",
        cmap="Blues",
        colors=get_sci_deep_colors(1),
        save_path=rf"{output}\联合直方图-温度与湿度.png",
    )

    # 核密度图：比较不同高度层垂直速度的分布形态
    kde_data = df[
        (df["station"] == "a")
        & (df["height"] <= 2500)
    ].copy()

    kde_data["height_group"] = pd.cut(
        kde_data["height"],
        bins=[0, 500, 1000, 1500, 2000, 2500],
        labels=["0-500", "500-1000", "1000-1500", "1500-2000", "2000-2500"],
    )

    plot_kde(
        kde_data,
        value="vertical_velocity",
        group="height_group",
        legend="高度层 (m)",
        title="不同高度层垂直速度核密度图",
        xlabel="垂直速度 (m/s)",
        ylabel="密度",
        colors=get_sci_deep_colors(5),
        save_path=rf"{output}\核密度图-不同高度层垂直速度.png",
    )

    # 残差图：用一个时刻的风速-高度二次拟合作为组件测试
    fit_data = df[
        (df["station"] == "a")
        & (df["time"] == df["time"].min())
        & (df["height"] <= 1500)
    ][["height", "wind_speed"]].dropna().copy()

    coef = np.polyfit(fit_data["height"], fit_data["wind_speed"], 2)
    fit_data["predicted_wind_speed"] = np.polyval(coef, fit_data["height"])

    plot_residual(
        fit_data,
        actual="wind_speed",
        predicted="predicted_wind_speed",
        title="风速二次拟合残差图",
        xlabel="预测风速 (m/s)",
        ylabel="残差 (m/s)",
        colors=get_sci_deep_colors(1),
        save_path=rf"{output}\残差图-风速二次拟合.png",
    )


    # 置信区间带图：A站不同高度的平均风速 ± 1个标准差
    band_data = df[
        (df["station"] == "a")
        & (df["height"] <= 1500)
    ].groupby("height", as_index=False)["wind_speed"].agg(["mean", "std"]).reset_index()

    band_data["lower"] = band_data["mean"] - band_data["std"]
    band_data["upper"] = band_data["mean"] + band_data["std"]

    plot_band(
        band_data,
        x="height",
        y="mean",
        lower="lower",
        upper="upper",
        title="A站风速均值及波动范围",
        xlabel="高度 (m)",
        ylabel="风速 (m/s)",
        colors=get_sci_deep_colors(1),
        save_path=rf"{output}\置信区间带图-A站风速.png",
    )

    # Q-Q图：继续使用前面的风速二次拟合残差进行组件测试
    fit_data["residual"] = fit_data["wind_speed"] - fit_data["predicted_wind_speed"]

    plot_qq(
        fit_data,
        value="residual",
        title="风速拟合残差Q-Q图",
        xlabel="理论正态分位数",
        ylabel="残差分位数",
        colors=get_sci_deep_colors(1),
        save_path=rf"{output}\QQ图-风速拟合残差.png",
    )

    # Pareto前沿图：用现有风速和风切变演示组件，不代表正式优化目标
    pareto_data = df[
        (df["station"] == "a")
        & (df["height"] <= 1500)
    ][["wind_speed", "wind_shear"]].dropna().copy()

    plot_pareto(
        pareto_data,
        x="wind_speed",
        y="wind_shear",
        minimize_x=True,
        minimize_y=True,
        title="风速-风切变 Pareto 前沿（组件测试）",
        xlabel="风速 (m/s)",
        ylabel="风切变",
        colors=get_sci_deep_colors(2),
        save_path=rf"{output}\Pareto前沿图-组件测试.png",
    )

    plt.close("all")
