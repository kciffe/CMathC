"""Generate a paper-ready diagram of source semantics and head geometry."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[1] / "revision_v3"
sys.path.insert(0, str(REVISION))
from head_model import geometry_manifest  # noqa: E402


font_path = r"C:\Windows\Fonts\msyh.ttc"
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams.update({
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

INK = "#243746"
MUTED = "#667785"
BLUE = "#356B9A"
BLUE_LIGHT = "#EAF1F7"
GREEN = "#53836A"
GREEN_LIGHT = "#EAF3EE"
PURPLE = "#8D679A"
PURPLE_LIGHT = "#F2EBF4"
AMBER = "#C18A32"
GRAY = "#7A8790"


def _box(ax, x, y, width, height, text, edge, face, fontsize=8.8):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.07",
        linewidth=1.0, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            color=INK, fontsize=fontsize, linespacing=1.2)
    return patch


def _arrow(ax, start, end, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9,
        linewidth=1.1, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
    ))


def _draw_route_group(ax, y, title, color, light):
    ax.text(0.05, y + 1.02, title, ha="left", va="center",
            fontsize=8.2, fontweight="bold", color=INK)
    left_y, right_y = y + 0.43, y - 0.43
    _box(ax, 0.15, left_y - 0.25, 2.45, 0.50, "图像左视野输入", color, light)
    _box(ax, 0.15, right_y - 0.25, 2.45, 0.50, "图像右视野输入", color, light)
    _box(ax, 7.30, left_y - 0.25, 2.55, 0.50, "左半球源  ←  右视野", color, "white")
    _box(ax, 7.30, right_y - 0.25, 2.55, 0.50, "右半球源  ←  左视野", color, "white")
    # Deliberately crossed paths encode the contralateral visual projection.
    _arrow(ax, (2.62, left_y), (7.28, right_y), color, rad=-0.10)
    _arrow(ax, (2.62, right_y), (7.28, left_y), color, rad=0.10)


def main():
    geometry = geometry_manifest()
    fig = plt.figure(figsize=(10.5, 5.4), dpi=450, layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0), wspace=0.10)
    route_ax = fig.add_subplot(grid[0, 0])
    radius_ax = fig.add_subplot(grid[0, 1])
    fig.suptitle("revision_v3：视野、模板偏好与源几何的明确区分",
                 fontsize=14, fontweight="bold", color=INK)

    route_ax.set_xlim(0, 10.1)
    route_ax.set_ylim(0, 10)
    route_ax.axis("off")
    route_ax.text(0.05, 9.40, "A  人口通道到源代理的语义路由",
                  fontsize=10.5, fontweight="bold", color=INK)
    _draw_route_group(route_ax, 7.50, "早期视觉场群体", BLUE, BLUE_LIGHT)
    _draw_route_group(route_ax, 4.75, "空间构型群体", GREEN, GREEN_LIGHT)

    route_ax.text(0.05, 2.10, "三角模板偏好群体", fontsize=9.7,
                  fontweight="bold", color=INK)
    _box(route_ax, 0.15, 1.22, 2.30, 0.55, "偏好左三角模板  P_L", PURPLE, PURPLE_LIGHT)
    _box(route_ax, 0.15, 0.36, 2.30, 0.55, "偏好右三角模板  P_R", PURPLE, PURPLE_LIGHT)
    _box(route_ax, 3.85, 0.78, 2.20, 0.58, "有符号差  P_L − P_R", PURPLE, "white")
    _box(route_ax, 7.30, 0.78, 2.55, 0.58, "单一双侧中线形状对手源", PURPLE, "white", fontsize=8.0)
    _arrow(route_ax, (2.48, 1.48), (3.82, 1.12), PURPLE, rad=-0.05)
    _arrow(route_ax, (2.48, 0.63), (3.82, 1.02), PURPLE, rad=0.05)
    _arrow(route_ax, (6.08, 1.07), (7.28, 1.07), PURPLE)
    route_ax.text(0.15, 0.06,
                  "模板 L/R 是形状偏好标签；它们不表示视觉半区，也不指定左右半球位置。",
                  fontsize=8.2, color=MUTED, ha="left", va="bottom")

    radius_ax.set_title("B  表面传感器与内部源的径向距离", loc="left",
                        fontsize=10.5, fontweight="bold", color=INK, pad=12)
    radius_ax.set_xlim(60, 94)
    radius_ax.set_ylim(-0.7, 5.8)
    radius_ax.set_xlabel("距头中心的半径  r (mm)")
    radius_ax.set_yticks([5, 4, 3, 2, 1, 0])
    radius_ax.set_yticklabels([
        "表面电极 F3/Fz/F4", "表面参考 M1/M2", "早期视觉源 LH/RH",
        "空间构型源 LH/RH", "模板偏好中线源", "",
    ])
    radius_ax.grid(axis="x", color="#DDE3E7", linewidth=0.6)
    radius_ax.spines[["top", "right", "left"]].set_visible(False)
    radius_ax.tick_params(axis="y", length=0, pad=5, labelsize=8.5)
    radius_ax.axvline(90.0, color=AMBER, linestyle="--", linewidth=1.4, zorder=1)
    radius_ax.text(90.25, 5.55, "外表面 r=90", color=AMBER, fontsize=8,
                   ha="left", va="center")

    radii = geometry["source_radii_mm"]
    depths = geometry["source_depths_mm"]
    groups = [
        (5, [90.0, 90.0, 90.0], "#344955", "o"),
        (4, [90.0, 90.0], GRAY, "D"),
        (3, radii[:2], BLUE, "o"),
        (2, radii[2:4], GREEN, "o"),
        (1, radii[4:], PURPLE, "o"),
    ]
    for y, values, color, marker in groups:
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
        for value, offset in zip(values, offsets):
            radius_ax.scatter(value, y + offset, s=27, color=color, marker=marker,
                              edgecolor="white", linewidth=0.5, zorder=3)
    radius_ax.text(61.0, -0.15,
                   f"五个源半径 {min(radii):.2f}–{max(radii):.2f} mm；\n"
                   f"距表面深度 {min(depths):.2f}–{max(depths):.2f} mm。",
                   fontsize=8.2, color=INK, ha="left", va="top")

    fig.text(0.5, -0.01,
             "左/右视野到左右半球采用对侧路由假设；源坐标为规范化示意坐标。"
             "模板偏好群体共享 0.5/0.5 双视野前馈；偏好差→中线源是建模假设。"
             "导联为均匀无限导体近似，不是个体化源定位。",
             ha="center", fontsize=8.2, color=MUTED)

    stem = HERE / "revision_v3_源语义与内部源几何"
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "claim": "图示区分视野左右、模板偏好和源半球，标明模板偏好群体共享的双视野前馈及中线源假设，并展示源与电极几何。",
        "figure_role": "mechanism_and_geometry_audit",
        "source_paths": [
            "q2/revision_v3/model.py:map_population_to_source_channels",
            "q2/revision_v3/head_model.py:geometry_manifest",
            "q2/output/revision_v3_source_mapping_audit/source_semantics_geometry_audit.json",
        ],
        "units": "coordinates and radii in mm; source currents are relative proxies",
        "source_mapping_schema": "contralateral_visual_field_5source_midline_opponent_bilateral_ff_v2",
        "source_coordinates_mm": geometry["source_coordinates_mm"],
        "source_radii_mm": geometry["source_radii_mm"],
        "source_depths_mm": geometry["source_depths_mm"],
        "sensor_radii_mm": geometry["sensor_radii_mm"],
        "reference_radii_mm": geometry["reference_radii_mm"],
        "forward_approximation": geometry["forward_approximation"],
        "randomness": "none",
        "outputs": [str(stem.with_suffix(".png")), str(stem.with_suffix(".svg"))],
        "limitation": "The drawing audits explicit model assumptions; it does not establish anatomical source localization or EEG validity.",
    }
    stem.with_suffix(".figure.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
