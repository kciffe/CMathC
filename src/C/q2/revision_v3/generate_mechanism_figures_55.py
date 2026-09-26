"""Generate five Chinese mechanism figures weighted by the selected 55% OOF set.

The selected OOF rows supply the exact trial list, cue labels, and per-record
weights. LGN/Gabor/cortical curves remain deterministic model simulations for
those standardized cues; they are not trial-wise neural recordings or fits.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
Q2_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "math-model-agent" / "code"))

import config  # noqa: E402
from frontend import load_stimulus, mirror_stage1_frontend, simulate_frontend  # noqa: E402
from generate_final_pic_55 import (  # noqa: E402
    RECORD_LABELS,
    RECORD_ORDER,
    _read_predictions,
)
from model import ModelParams, simulate_forward  # noqa: E402
from algorithms.sci_figures import paper_figure_rc_params  # noqa: E402

RESULT_DIR = config.OUTPUT_ROOT
FIGURE_DIR = Q2_ROOT / "output" / "final_pic"
SELECTED_OUTPUTS = ("Gabor左右差异图.png", "Gabor空间响应热图.png",
                    "LGN空间峰值热图.png", "LGN平均响应.png",
                    "皮层条件差异诊断图.png", "简化导联矩阵热图.png")

COLORS = {"left": "#3569A8", "right": "#D17A35",
          "on": "#3D8B68", "off": "#8064A2"}
RECORD_SHORT = {
    "VisualCogA_Task-1": "A组记录1", "VisualCogA_Task-2": "A组记录2",
    "VisualCogB_Task-1": "B组记录1", "VisualCogB_Task-2": "B组记录2",
}
POPULATIONS = ("早期视觉群", "空间构型群", "三角模板偏好群")
POPULATION_CHANNELS = (("图像左半区", "图像右半区"), ("图像左半区", "图像右半区"),
                       ("左指模板偏好", "右指模板偏好"))

plt.rcParams.update(paper_figure_rc_params())
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def _boxed_legend(fig, handles, *, y=0.925, ncol=4):
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
                        ncol=ncol, frameon=True, fancybox=False, framealpha=1,
                        borderpad=0.5, handlelength=2.0, columnspacing=1.3,
                        labelspacing=0.4)
    legend.get_frame().set_edgecolor("#777777")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")


def _weighted_mean(items: list[tuple[np.ndarray, int]]) -> np.ndarray:
    total_n = sum(n for _, n in items)
    if total_n <= 0:
        raise ValueError("cannot average an empty selected condition")
    return sum(values * n for values, n in items) / total_n


def _paired_record_weight(counts, record):
    """Use the same record weight on both cue sides for paired contrasts."""
    return counts[record]["left"] + counts[record]["right"]


def _selected_counts(rows):
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        record = next(key for key, label in RECORD_LABELS.items()
                      if label == row["记录"])
        condition = "left" if row["真实方向"] == "左" else "right"
        counts[record][condition] += 1
    if set(counts) != set(RECORD_ORDER):
        raise ValueError("the selected OOF data must contain all four records")
    return counts


def _load_model_inputs(rows):
    manifest = json.loads((RESULT_DIR / "manifest.json").read_text(encoding="utf-8"))
    fit_payloads = json.loads((RESULT_DIR / "heldout_fit_details.json").read_text(encoding="utf-8"))
    resolution = int(manifest["resolution"])
    stride = float(manifest["frontend_feature_stride_ms"])
    time_ms = np.asarray(config.TIME_MS, dtype=float)
    counts = _selected_counts(rows)

    scales_by_population = defaultdict(dict)
    with (RESULT_DIR / "drive_scales.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            side = "left" if row["channel"].startswith("left_") else "right"
            scales_by_population[row["population"]][side] = float(row["reference_rms_scale"])
    drive_scales = np.asarray([
        [scales_by_population[population][side] for side in ("left", "right")]
        for population in ("early", "configuration", "shape_preference")
    ], dtype=np.float32)

    stimuli = {side: load_stimulus("Stage1", side) for side in ("left", "right")}
    frontend_by_tau = {}
    frontends = {}
    cortical = {}
    for record in RECORD_ORDER:
        payload = fit_payloads[record]
        parameters = payload["parameters"]
        tau_a = float(parameters["tau_a"])
        if tau_a not in frontend_by_tau:
            left_front = simulate_frontend(
                stimuli["left"], params={"tau_a": tau_a}, time_ms=time_ms,
                resolution=resolution, include_offset=True,
                feature_stride_ms=stride)
            frontend_by_tau[tau_a] = {
                "left": left_front,
                "right": mirror_stage1_frontend(left_front, stimuli["right"]),
            }
        fit_pair = frontend_by_tau[tau_a]
        frontends[record] = fit_pair
        params = ModelParams.from_any(parameters)
        amplitude = float(payload["amplitude"])
        cortical[record] = {
            side: simulate_forward(front, params=params, amplitude=amplitude,
                                   drive_scales=drive_scales)
            for side, front in fit_pair.items()
        }
    return time_ms, counts, frontends, cortical


def _weighted_front(frontends, counts, side, key):
    return _weighted_mean([(frontends[record][side][key], counts[record][side])
                           for record in RECORD_ORDER if counts[record][side] > 0])


def _plot_gabor_difference(time_ms, frontends, counts, n_total):
    mask = (time_ms >= 0) & (time_ms < 200)
    left = _weighted_mean([
        (frontends[record]["left"]["gabor"][..., mask].mean(axis=-1), _paired_record_weight(counts, record))
        for record in RECORD_ORDER if _paired_record_weight(counts, record) > 0])
    right = _weighted_mean([
        (frontends[record]["right"]["gabor"][..., mask].mean(axis=-1), _paired_record_weight(counts, record))
        for record in RECORD_ORDER if _paired_record_weight(counts, record) > 0])
    difference = right - left
    limit = max(float(np.quantile(np.abs(difference), 0.99)), 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.0), sharex=True, sharey=True)
    fig.suptitle("左右提示的 Gabor 方向响应差异", fontsize=15, y=0.98)
    fig.text(0.5, 0.875,
             f"8×8 空间通道响应的 0–200 ms 均值；右提示 − 左提示；左右使用相同记录权重（n={n_total}）",
             ha="center", va="center", fontsize=9.5,
             bbox={"boxstyle": "square,pad=0.45", "facecolor": "white",
                   "edgecolor": "#777777", "linewidth": 0.8})
    fig.subplots_adjust(left=0.035, right=0.93, bottom=0.08, top=0.75, wspace=0.28)
    for orientation, ax in enumerate(axes):
        im = ax.imshow(difference[orientation], origin="upper", interpolation="nearest",
                       cmap="seismic", vmin=-limit, vmax=limit)
        ax.set_title(f"{config.GABOR_ANGLES_DEG[orientation]:g}° 方向")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035, shrink=0.88)
    fig.savefig(FIGURE_DIR / SELECTED_OUTPUTS[0], dpi=300, facecolor="white")
    plt.close(fig)


def _plot_gabor_spatial(time_ms, frontends, counts, n_total):
    mask = (time_ms >= 0) & (time_ms < 200)
    left_orient = _weighted_mean([
        (frontends[record]["left"]["gabor"][..., mask].mean(axis=-1), _paired_record_weight(counts, record))
        for record in RECORD_ORDER if _paired_record_weight(counts, record) > 0])
    right_orient = _weighted_mean([
        (frontends[record]["right"]["gabor"][..., mask].mean(axis=-1), _paired_record_weight(counts, record))
        for record in RECORD_ORDER if _paired_record_weight(counts, record) > 0])
    # Display 4x4 block means of the native 8x8 spatial channels, matching the
    # compact matrix style of the legacy figure without changing model data.
    left_display = left_orient.reshape(4, 4, 2, 4, 2).mean(axis=(2, 4))
    right_display = right_orient.reshape(4, 4, 2, 4, 2).mean(axis=(2, 4))
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.2), sharex=True, sharey=True)
    fig.suptitle("左右提示的 Gabor 空间方向响应", fontsize=15, y=0.98)
    fig.text(0.5, 0.92,
             f"左右使用相同记录权重；显示期 0–200 ms；数值为 8×8 通道的 2×2 块平均；n={n_total}",
             ha="center", va="center", fontsize=10,
             bbox={"boxstyle": "square,pad=0.45", "facecolor": "white",
                   "edgecolor": "#777777", "linewidth": 0.8})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.84,
                        wspace=0.24, hspace=0.30)
    for orientation in range(4):
        vmax = max(float(left_display[orientation].max()),
                   float(right_display[orientation].max()), 1e-8)
        for row, (values, cue_name) in enumerate(((left_display, "左提示"),
                                                   (right_display, "右提示"))):
            ax = axes[row, orientation]
            grid = values[orientation]
            ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{cue_name} {config.GABOR_ANGLES_DEG[orientation]:g}°")
            ax.set_xticks(range(4), labels=range(1, 5))
            ax.set_yticks(range(4), labels=range(1, 5))
            for r in range(4):
                for c in range(4):
                    value = grid[r, c]
                    ax.text(c, r, f"{value:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if value < vmax * 0.35 else "black")
    fig.supxlabel("水平空间区域")
    fig.supylabel("垂直空间区域")
    fig.savefig(FIGURE_DIR / SELECTED_OUTPUTS[1], dpi=300, facecolor="white")
    plt.close(fig)


def _plot_leadfield_matrix():
    """Plot the fixed source-to-sensor mapping as a transparent method figure."""
    table = np.genfromtxt(RESULT_DIR / "leadfield.csv", delimiter=",", skip_header=1)
    matrix = np.asarray(table[:, 1:], dtype=float)
    if matrix.shape != (3, 5):
        raise ValueError(f"expected a 3x5 leadfield matrix, got {matrix.shape}")
    source_labels = ("早期视觉源\n右视野→左半球", "早期视觉源\n左视野→右半球",
                     "构型编码源\n右视野→左半球", "构型编码源\n左视野→右半球",
                     "中线形状偏好\n对手源")
    limit = float(np.max(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    fig.suptitle("固定源到电极映射矩阵", fontsize=15, y=0.98)
    fig.text(0.5, 0.90,
             "均匀无限导体点偶极近似；假设乳突平均参考；源代理未按物理偶极矩标定",
             ha="center", va="center", fontsize=10,
             bbox={"boxstyle": "square,pad=0.45", "facecolor": "white",
                   "edgecolor": "#777777", "linewidth": 0.8})
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(5), labels=source_labels, rotation=0, fontsize=9)
    ax.set_yticks(range(3), labels=("F3", "Fz", "F4"))
    ax.set_xlabel("功能源代理")
    ax.set_ylabel("观测电极")
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    threshold = limit * 0.48
    for r in range(3):
        for c in range(5):
            value = matrix[r, c]
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(value) > threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025,
                 label="导联系数（V/(A·m)）")
    fig.subplots_adjust(left=0.105, right=0.92, bottom=0.22, top=0.78)
    fig.savefig(FIGURE_DIR / SELECTED_OUTPUTS[5], dpi=300, facecolor="white")
    plt.close(fig)


def _plot_lgn_spatial_peaks(time_ms, frontends, counts, n_total):
    mask = (time_ms >= 0) & (time_ms <= 800)
    maps = {}
    for state in ("on", "off"):
        key = f"lgn_{state}"
        for side in ("left", "right"):
            maps[(state, side)] = _weighted_mean([
                (frontends[record][side][key][..., mask].max(axis=-1), _paired_record_weight(counts, record))
                for record in RECORD_ORDER if _paired_record_weight(counts, record) > 0])
        maps[(state, "difference")] = maps[(state, "right")] - maps[(state, "left")]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0), sharex=True, sharey=True)
    fig.suptitle("LGN 空间峰值热图\n0–800 ms 内各空间通道的 ON/OFF 峰值", fontsize=13, y=0.99)
    handles = [Line2D([0], [0], color="#555555", label=f"按入选试次记录构成加权；左右同权，n={n_total}")]
    _boxed_legend(fig, handles, y=0.925, ncol=1)
    fig.subplots_adjust(left=0.075, right=0.88, bottom=0.12, top=0.84, wspace=0.22, hspace=0.26)
    activity_images = []
    difference_images = []
    all_peaks = np.concatenate([maps[(state, side)].ravel()
                                for state in ("on", "off") for side in ("left", "right")])
    shared_vmax = max(float(np.quantile(all_peaks, 0.99)), 1e-8)
    all_differences = np.concatenate([maps[(state, "difference")].ravel()
                                      for state in ("on", "off")])
    shared_diff_limit = max(float(np.quantile(np.abs(all_differences), 0.99)), 1e-8)
    for row, state in enumerate(("on", "off")):
        for col, side, title in ((0, "left", "左提示"), (1, "right", "右提示"), (2, "difference", "右减左")):
            values = maps[(state, side)]
            lim = shared_diff_limit if side == "difference" else None
            im = axes[row, col].imshow(values, origin="upper", interpolation="nearest",
                                       cmap="coolwarm" if lim else "magma",
                                       vmin=-lim if lim else 0, vmax=lim if lim else shared_vmax,
                                       extent=(0.5, 8.5, 8.5, 0.5))
            (difference_images if lim else activity_images).append(im)
            axes[row, col].set_title(title if row == 0 else "")
            axes[row, col].set_ylabel("ON 通路" if state == "on" and col == 0 else
                                      "OFF 通路" if state == "off" and col == 0 else "")
            axes[row, col].set_xticks(range(1, 9)); axes[row, col].set_yticks(range(1, 9))
    fig.supxlabel("水平空间通道")
    fig.colorbar(activity_images[0], ax=axes[:, :2].ravel().tolist(),
                 fraction=0.035, pad=0.025, label="TCR 峰值活动（相对值）")
    fig.colorbar(difference_images[0], ax=axes[:, 2].ravel().tolist(),
                 fraction=0.05, pad=0.08, label="右减左峰值差")
    fig.savefig(FIGURE_DIR / SELECTED_OUTPUTS[2], dpi=300, facecolor="white")
    plt.close(fig)


def _plot_lgn_mean(time_ms, frontends, counts, n_total):
    tmask = (time_ms >= 0) & (time_ms <= 800)
    times = time_ms[tmask]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.suptitle(f"LGN 群体平均 ON/OFF 响应\n按入选试次的记录构成加权，左右同权；n={n_total}", fontsize=13, y=0.99)
    handles = [
        Line2D([0], [0], color=COLORS["on"], linestyle="-", label="左提示 ON 通路"),
        Line2D([0], [0], color=COLORS["on"], linestyle="--", label="右提示 ON 通路"),
        Line2D([0], [0], color=COLORS["off"], linestyle="-", label="左提示 OFF 通路"),
        Line2D([0], [0], color=COLORS["off"], linestyle="--", label="右提示 OFF 通路"),
        Line2D([0], [0], color="#555555", linestyle=":", label="提示出现"),
        Line2D([0], [0], color="#555555", linestyle="--", label="提示消失（200 ms）"),
    ]
    _boxed_legend(fig, handles, y=0.91, ncol=3)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.13, top=0.78)
    for side, side_name in (("left", "左提示"), ("right", "右提示")):
        for state, color in (("on", COLORS["on"]), ("off", COLORS["off"])):
            curve = _weighted_mean([
                (frontends[record][side][f"lgn_{state}_mean"][tmask], _paired_record_weight(counts, record))
                for record in RECORD_ORDER if _paired_record_weight(counts, record) > 0])
            ax.plot(times, curve, color=color, linewidth=1.5,
                    linestyle="-" if side == "left" else "--",
                    label=f"{side_name} · {'ON通路' if state == 'on' else 'OFF通路'}")
    ax.axvspan(0, 200, color="#91AFC4", alpha=0.12)
    ax.axvline(0, color="#555555", linestyle=":", linewidth=0.9)
    ax.axvline(200, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, 800)
    ax.set_xlabel("提示出现后时间（ms）")
    ax.set_ylabel("群体活动（相对值）")
    ax.grid(alpha=0.22); ax.set_axisbelow(True)
    fig.savefig(FIGURE_DIR / SELECTED_OUTPUTS[3], dpi=300, facecolor="white")
    plt.close(fig)


def _weighted_cortical(cortical, counts, condition, field):
    entries = [(getattr(cortical[record][condition], field), _paired_record_weight(counts, record))
               for record in RECORD_ORDER if _paired_record_weight(counts, record) > 0]
    return _weighted_mean(entries)


def _plot_cortical_difference(time_ms, cortical, counts, n_total):
    left_e = _weighted_cortical(cortical, counts, "left", "excitatory")
    right_e = _weighted_cortical(cortical, counts, "right", "excitatory")
    left_i = _weighted_cortical(cortical, counts, "left", "inhibitory")
    right_i = _weighted_cortical(cortical, counts, "right", "inhibitory")
    diff_e, diff_i = right_e - left_e, right_i - left_i
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 7.2), sharex=True)
    fig.suptitle("皮层群体条件差异：右指三角减左指三角", fontsize=13, y=0.99)
    fig.text(0.5, 0.955,
             f"前两组为图像左/右半区；第三组为左/右模板偏好；同一记录拟合参数，按入选试次加权，n={n_total}",
             ha="center", va="center", fontsize=9.2)
    handles = [
        Line2D([0], [0], color="#3265A8", linestyle="-", label="兴奋性 · 通道1"),
        Line2D([0], [0], color="#3265A8", linestyle="--", label="兴奋性 · 通道2"),
        Line2D([0], [0], color="#D17A35", linestyle="-", label="抑制性 · 通道1"),
        Line2D([0], [0], color="#D17A35", linestyle="--", label="抑制性 · 通道2"),
    ]
    _boxed_legend(fig, handles, y=0.91, ncol=4)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.10, top=0.82, hspace=0.28)
    mask = (time_ms >= 0) & (time_ms <= 800)
    for group, ax in enumerate(axes):
        ax.plot(time_ms[mask], diff_e[group, 0, mask], color="#3265A8", lw=1.35)
        ax.plot(time_ms[mask], diff_e[group, 1, mask], color="#3265A8", lw=1.35, ls="--")
        ax.plot(time_ms[mask], diff_i[group, 0, mask], color="#D17A35", lw=1.35)
        ax.plot(time_ms[mask], diff_i[group, 1, mask], color="#D17A35", lw=1.35, ls="--")
        ax.axhline(0, color="#555555", lw=0.75)
        ax.axvline(0, color="#555555", ls=":", lw=0.75)
        ax.axvline(200, color="#555555", ls="--", lw=0.75)
        ax.set_ylabel("差值（相对活动）")
        ax.set_title(f"{POPULATIONS[group]}：第1通道={POPULATION_CHANNELS[group][0]}，第2通道={POPULATION_CHANNELS[group][1]}",
                     loc="left", fontsize=9, pad=2)
        ax.set_xlim(0, 800)
        ax.grid(alpha=0.22); ax.set_axisbelow(True)
    axes[-1].set_xlabel("提示出现后时间（ms）")
    fig.savefig(FIGURE_DIR / SELECTED_OUTPUTS[4], dpi=300, facecolor="white")
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_predictions()
    if not rows:
        raise ValueError("the selected 55% OOF dataset is empty")
    time_ms, counts, frontends, cortical = _load_model_inputs(rows)
    total_left = sum(counts[record]["left"] for record in RECORD_ORDER)
    total_right = sum(counts[record]["right"] for record in RECORD_ORDER)
    n_total = total_left + total_right
    _plot_gabor_difference(time_ms, frontends, counts, n_total)
    _plot_gabor_spatial(time_ms, frontends, counts, n_total)
    _plot_lgn_spatial_peaks(time_ms, frontends, counts, n_total)
    _plot_lgn_mean(time_ms, frontends, counts, n_total)
    _plot_cortical_difference(time_ms, cortical, counts, n_total)
    _plot_leadfield_matrix()
    print(f"Selected OOF trials: {n_total} (left={total_left}, right={total_right})")
    print(f"Generated {len(SELECTED_OUTPUTS)} PNG files in {FIGURE_DIR}")
    for name in SELECTED_OUTPUTS:
        print(FIGURE_DIR / name)


if __name__ == "__main__":
    main()
