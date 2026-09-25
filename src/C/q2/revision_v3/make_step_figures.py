"""Build a traceable, paper-oriented figure set for the revision_v3 Q2 chain.

The script does not fit parameters. Steps 01-04 use deterministic forward
simulations with the median saved fit from the three records prioritized in
the Q1 quality assessment. Steps 05 use the saved leave-one-record predictions
and observed ERPs for all four records.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


Q2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Q2_ROOT.parents[2]
V3_DIR = Q2_ROOT / "output" / "revision_v3"
OUT_DIR = Q2_ROOT / "output" / "revision_v3_paper_figures"
sys.path.insert(0, str(Q2_ROOT))
sys.path.insert(0, str(REPO_ROOT / "math-model-agent" / "code"))

from algorithms.sci_figures import (  # noqa: E402
    FigureContract,
    MODELING_PALETTE,
    audit_publication_figure,
    export_publication_figure,
    paper_figure_rc_params,
    publication_size,
)
from revision_v3 import config  # noqa: E402
from revision_v3.frontend import (  # noqa: E402
    load_stimulus,
    mirror_stage1_frontend,
    simulate_frontend,
)
from revision_v3.model import ModelParams, simulate_forward  # noqa: E402
from revision_v3.observation import model_curve_to_q1_grid  # noqa: E402


PRIMARY_RECORDS = (
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
)
LOW_QUALITY_RECORD = "VisualCogA_Task-1"
RECORD_ORDER = (*PRIMARY_RECORDS, LOW_QUALITY_RECORD)
CHANNELS = ("F3", "Fz", "F4")
CHANNEL_COLORS = {"F3": "#3C6E8F", "Fz": "#5B8C6A", "F4": "#A6534C"}
CONDITION_COLORS = {"left": "#3C6E8F", "right": "#B27A45"}
INK = "#262626"
GRID = "#D8D8D2"
DIV_CMAP = LinearSegmentedColormap.from_list(
    "paper_diverging", MODELING_PALETTE["diverging"], N=257
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_calibration() -> np.ndarray:
    rows = _read_csv(V3_DIR / "drive_scales.csv")
    population_order = ("early", "configuration", "shape_preference")
    channel_order = (("left_visual_field", "right_visual_field"),
                     ("left_visual_field", "right_visual_field"),
                     ("left_triangle_template_preference",
                      "right_triangle_template_preference"))
    values = []
    for population, names in zip(population_order, channel_order):
        selected = rows.loc[rows["population"] == population].set_index("channel")
        values.append([float(selected.loc[name, "reference_rms_scale"]) for name in names])
    return np.asarray(values, dtype=np.float32)


def _representative_fit() -> tuple[ModelParams, float, dict]:
    fits = _read_csv(V3_DIR / "heldout_fit_summary.csv")
    primary = fits.loc[fits["heldout_record"].isin(PRIMARY_RECORDS)]
    if set(primary["heldout_record"]) != set(PRIMARY_RECORDS):
        raise ValueError("Saved v3 fits do not include all three prioritized records")
    params = {
        "tau_s": float(primary["tau_s_ms"].median()),
        "g_i": float(primary["g_i"].median()),
        "tau_a": float(primary["tau_a_ms"].median()),
    }
    gain = float(primary["shared_gain_signed"].median())
    detail = {
        "records_used_for_representative_median": list(PRIMARY_RECORDS),
        "parameters": params,
        "shared_gain_signed": gain,
        "tau_s_at_upper_bound_for_all_primary_folds": bool(
            np.all(primary["tau_s_ms"].to_numpy(dtype=float)
                   >= config.PARAM_BOUNDS["tau_s"][1] - 1e-9)
        ),
        "fits_converged": bool(primary["optimizer_success"].astype(bool).all()),
    }
    return ModelParams.from_any(params), gain, detail


def _prepare_forward() -> dict:
    manifest = config.require_current_source_mapping_manifest(V3_DIR / "manifest.json")
    params, gain, fit_detail = _representative_fit()
    scales = _read_calibration()
    left_stimulus = load_stimulus("Stage1", "left")
    right_stimulus = load_stimulus("Stage1", "right")
    left_front = simulate_frontend(
        left_stimulus,
        params={"tau_a": params.tau_a},
        resolution=int(manifest["resolution"]),
        feature_stride_ms=float(manifest["frontend_feature_stride_ms"]),
        capture_spatial_audit=True,
    )
    right_front = mirror_stage1_frontend(left_front, right_stimulus)
    result = {}
    for condition, front in (("left", left_front), ("right", right_front)):
        result[condition] = simulate_forward(
            front,
            params=params,
            amplitude=gain,
            drive_scales=scales,
        )
    return {
        "manifest": manifest,
        "params": params,
        "gain": gain,
        "fit_detail": fit_detail,
        "scales": scales,
        "fronts": {"left": left_front, "right": right_front},
        "results": result,
    }


def _cmap_and_font_setup() -> None:
    mpl.rcParams.update(paper_figure_rc_params())
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.50,
        "grid.linewidth": 0.45,
        "axes.edgecolor": "#4D4D4D",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#4D4D4D",
        "ytick.color": "#4D4D4D",
        "legend.frameon": True,
        "pdf.fonttype": 42,
    })


def _base_axis(ax, *, xlim=(0, 800), cue_offset=True):
    ax.set_xlim(*xlim)
    ax.axhline(0, color="0.55", linewidth=0.55, zorder=0)
    ax.axvline(0, color="0.42", linewidth=0.65, zorder=0)
    if cue_offset:
        ax.axvline(200, color="0.42", linewidth=0.7, linestyle="--", zorder=0)
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)


def _place_figure_legend(fig, slot_ax, handles, *, ncol: int, fontsize: float = 6.0,
                         handlelength: float = 1.9, columnspacing: float = 0.9) -> None:
    """Place a framed legend in a reserved figure row above the data axes."""
    fig.canvas.draw()
    x, y, width, height = slot_ax.get_position().bounds
    slot_ax.set_axis_off()
    fig.legend(
        handles=handles,
        loc="center",
        # Leave a clean gap below the figure title while keeping the legend
        # centered in the reserved top band and clear of all data axes.
        bbox_to_anchor=(x + width / 2, y + height / 2 - 0.03),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        fontsize=fontsize,
        handlelength=handlelength,
        columnspacing=columnspacing,
        frameon=True,
        fancybox=False,
        framealpha=0.98,
        facecolor="white",
        edgecolor="#8A8A8A",
        borderpad=0.42,
        labelspacing=0.32,
    )


def _spatial_cell_means(values: np.ndarray, grid: int = 4) -> np.ndarray:
    """Average a 2-D spatial response into an evenly partitioned grid."""
    values = np.asarray(values, dtype=float)
    height, width = values.shape
    return np.asarray([
        [values[r * height // grid:(r + 1) * height // grid,
                c * width // grid:(c + 1) * width // grid].mean()
         for c in range(grid)]
        for r in range(grid)
    ])


def _figure_contract(
    *, claim: str, evidence: tuple[str, ...], source_paths: tuple[str, ...],
    figure_role: str, scenario: str, statistic: str | None = None,
    n_definition: str | None = None, review_risks: tuple[str, ...] = (),
) -> FigureContract:
    return FigureContract(
        claim=claim,
        evidence=evidence,
        source_paths=source_paths,
        target_venue="CMathc question 2 paper draft",
        column="double",
        figure_role=figure_role,
        model_name="revision_v3",
        scenario=scenario,
        parameter_source="saved held-out fit summary; no refitting",
        randomness="deterministic forward simulation; no random sampling",
        n_definition=n_definition,
        statistic=statistic,
        uncertainty="not shown; deterministic model curves and trial-mean EEG",
        review_risks=review_risks,
    )


def _save_figure(fig, stem: str, contract: FigureContract, registry: list,
                 audits: list) -> None:
    fig.set_size_inches(*publication_size("double", fig.get_size_inches()[1] * 25.4))
    report = audit_publication_figure(fig, contract)
    files = export_publication_figure(
        fig, OUT_DIR / stem, contract, dpi=450, strict=True, close=True
    )
    registry.append({
        "figure_id": stem,
        "claim": contract.claim,
        "evidence": list(contract.evidence),
        "files": {key: str(Path(value).relative_to(Q2_ROOT))
                  for key, value in files.items()},
        "contract": {
            "figure_role": contract.figure_role,
            "scenario": contract.scenario,
            "model_name": contract.model_name,
            "source_paths": list(contract.source_paths),
            "statistic": contract.statistic,
            "n_definition": contract.n_definition,
            "review_risks": list(contract.review_risks),
        },
    })
    audits.append({"figure_id": stem, **report.to_dict()})


def plot_step1(data: dict, registry: list, audits: list) -> None:
    fronts = data["fronts"]
    left, right = fronts["left"], fronts["right"]
    audit_time = min(left["audit_maps"], key=lambda t: abs(float(t) - 100.0))
    map_left = left["audit_maps"][audit_time]["V1"].mean(axis=0)
    map_right = right["audit_maps"][audit_time]["V1"].mean(axis=0)
    map_diff = map_right - map_left
    stim_left = left["stimulus"].signed_contrast
    stim_right = right["stimulus"].signed_contrast
    stim_diff = stim_right - stim_left
    vmax_input = max(float(np.max(np.abs(stim_left))), float(np.max(np.abs(stim_right))))
    vmax_gabor = max(float(map_left.max()), float(map_right.max()), 1e-12)
    vmax_diff = max(float(np.max(np.abs(stim_diff))), 1e-12)
    vmax_gabor_diff = max(float(np.max(np.abs(map_diff))), 1e-12)

    fig = plt.figure(figsize=publication_size("double", 160), layout="constrained")
    grid = fig.add_gridspec(4, 3, height_ratios=(0.16, 1.0, 1.0, 0.88))
    legend_ax = fig.add_subplot(grid[0, :])
    axes = np.asarray([[fig.add_subplot(grid[1, c]) for c in range(3)],
                       [fig.add_subplot(grid[2, c]) for c in range(3)]])
    time_ax = fig.add_subplot(grid[3, :])
    top_data = (stim_left, stim_right, stim_diff)
    top_titles = ("左 cue−圆形基线", "右 cue−圆形基线", "右−左（基线相减后）")
    top_images = []
    for col, (ax, values, title) in enumerate(zip(axes[0], top_data, top_titles)):
        im = ax.imshow(values, origin="upper", cmap=DIV_CMAP,
                       vmin=-vmax_input if col < 2 else -vmax_diff,
                       vmax=vmax_input if col < 2 else vmax_diff,
                       interpolation="nearest")
        top_images.append(im)
        ax.set_title(title)
        ax.set_xlabel("刺激水平 x")
        ax.set_ylabel("刺激水平 y")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 2:
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    fig.colorbar(top_images[0], ax=axes[0, :2], fraction=0.025, pad=0.012)

    map_data = (map_left, map_right, map_diff)
    map_titles = (f"左 cue：Gabor 能量（{audit_time:g} ms）",
                  f"右 cue：Gabor 能量（{audit_time:g} ms）",
                  "右 − 左 Gabor 能量")
    map_images = []
    for col, (ax, values, title) in enumerate(zip(axes[1], map_data, map_titles)):
        limit = vmax_gabor_diff if col == 2 else vmax_gabor
        im = ax.imshow(values, origin="upper", cmap=DIV_CMAP if col == 2 else "viridis",
                       vmin=-limit if col == 2 else 0.0,
                       vmax=limit, interpolation="nearest")
        map_images.append(im)
        ax.set_title(title)
        ax.set_xlabel("前端网格 x")
        ax.set_ylabel("前端网格 y")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 2:
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    fig.colorbar(map_images[0], ax=axes[1, :2], fraction=0.025, pad=0.012)

    t = np.asarray(left["time_ms"], dtype=float)
    for condition, front in (("left", left), ("right", right)):
        energy = front["gabor"].mean(axis=0)
        for half, slc, linestyle in (("左半场", slice(0, 4), "-"),
                                     ("右半场", slice(4, 8), "--")):
            values = energy[:, slc, :].mean(axis=(0, 1))
            time_ax.plot(t, values, color=CONDITION_COLORS[condition],
                         linestyle=linestyle, linewidth=1.0, alpha=0.9,
                         label=f"{'左' if condition == 'left' else '右'} cue · {half}")
    _base_axis(time_ax)
    time_ax.set_title("按视觉半场汇总的空间 Gabor 能量")
    time_ax.set_xlabel("cue onset 后时间（ms）")
    time_ax.set_ylabel("半场平均能量（相对单位）")
    half_handles = [
        Line2D([0], [0], color=CONDITION_COLORS["left"], lw=1.0, label="左 cue · 左半场"),
        Line2D([0], [0], color=CONDITION_COLORS["left"], lw=1.0, ls="--", label="左 cue · 右半场"),
        Line2D([0], [0], color=CONDITION_COLORS["right"], lw=1.0, label="右 cue · 左半场"),
        Line2D([0], [0], color=CONDITION_COLORS["right"], lw=1.0, ls="--", label="右 cue · 右半场"),
    ]
    _place_figure_legend(fig, legend_ax, half_handles, ncol=4, fontsize=5.8)
    fig.suptitle("01 · 左右 cue 的空间结构与方向能量编码", fontsize=10, fontweight="bold")
    _save_figure(
        fig, "01_v3_gabor_frontend",
        _figure_contract(
            claim="V3 前端将左右 cue 映射为空间化的 Gabor 方向能量响应。",
            evidence=("规范化左右 cue 对比图", "100 ms Gabor 空间能量图",
                      "左右视觉半场的 Gabor 能量时程"),
            source_paths=("input/stage1_cue_left.npy", "input/stage1_cue_right.npy",
                          "output/revision_v3/manifest.json", "revision_v3/frontend.py"),
            figure_role="model-result",
            scenario="Stage1 标准左右 cue；代表性参数来自三个重点记录的已保存留出拟合中位数。",
            statistic="空间能量按 V3 前端输出绘制；空间差异为右减左。",
            n_definition="无单试次样本；图中为标准化 cue 输入的确定性模型前向计算。",
            review_risks=("Gabor 响应是模型内部特征，不是直接观测到的皮层活动。",),
    ), registry, audits,
    )


def plot_stage1_preprocessing(data: dict, registry: list, audits: list) -> None:
    """Show raw Stage1 images and their pixelwise circle-baseline contrasts."""
    left = data["fronts"]["left"]["stimulus"]
    right = data["fronts"]["right"]["stimulus"]
    baseline = np.asarray(left.baseline, dtype=float) / 255.0
    raw_left = np.asarray(left.image, dtype=float) / 255.0
    raw_right = np.asarray(right.image, dtype=float) / 255.0
    contrast_left = np.asarray(left.signed_contrast, dtype=float)
    contrast_right = np.asarray(right.signed_contrast, dtype=float)
    contrast_difference = contrast_right - contrast_left
    contrast_limit = max(
        float(np.max(np.abs(contrast_left))),
        float(np.max(np.abs(contrast_right))),
        float(np.max(np.abs(contrast_difference))),
        1e-12,
    )

    fig, axes = plt.subplots(
        2, 3, figsize=publication_size("double", 105), layout="constrained"
    )
    raw_panels = (
        (baseline, "共同圆形基线 B"),
        (raw_left, "左 cue 原始图像 I_L"),
        (raw_right, "右 cue 原始图像 I_R"),
    )
    for ax, (values, title) in zip(axes[0], raw_panels):
        raw_image = ax.imshow(
            values, origin="upper", cmap="gray", vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        raw_image, ax=axes[0, :].tolist(), fraction=0.035, pad=0.02,
        label="归一化灰度（0–1）",
    )

    contrast_panels = (
        (contrast_left, "左差分 X_L=(I_L−B)/255"),
        (contrast_right, "右差分 X_R=(I_R−B)/255"),
        (contrast_difference, "左右差分 X_R−X_L"),
    )
    for ax, (values, title) in zip(axes[1], contrast_panels):
        contrast_image = ax.imshow(
            values, origin="upper", cmap=DIV_CMAP,
            vmin=-contrast_limit, vmax=contrast_limit,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        contrast_image, ax=axes[1, :].tolist(), fraction=0.035, pad=0.02,
        label="signed contrast",
    )
    fig.suptitle(
        "Stage1 原始刺激与圆形基线逐像素相减",
        fontsize=10, fontweight="bold",
    )
    _save_figure(
        fig, "stage1_raw_stimulus_preprocessing",
        _figure_contract(
            claim="Stage1 cue 原始图像含共同圆形基线；逐像素相减后共同圆形抵消，仅三角区域保留亮度变化。",
            evidence=(
                "共同圆形基线与左右原始 cue 灰度图",
                "X_L=(I_L-B)/255、X_R=(I_R-B)/255 及左右差分",
            ),
            source_paths=(
                "input/stage1_baseline_circle.npy",
                "input/stage1_cue_left.npy",
                "input/stage1_cue_right.npy",
                "revision_v3/frontend.py",
            ),
            figure_role="model-diagnostic",
            scenario="Stage1 原始 256×256 灰度矩阵；不含 EEG 试次或拟合参数。",
            statistic="图像除以 255 归一化；signed contrast 按 cue 图像减共同圆形基线计算。",
            n_definition="不涉及样本；展示固定刺激矩阵的确定性预处理。",
            review_risks=(
                "模型使用像素级基线差分，不是减去圆形的学习特征向量。",
            ),
        ), registry, audits,
    )


def plot_step1_detail_figures(data: dict, registry: list, audits: list) -> None:
    """Export the familiar Stage1 orientation and spatial diagnostics from V3 maps."""
    left, right = data["fronts"]["left"], data["fronts"]["right"]
    audit_time = min(left["audit_maps"], key=lambda t: abs(float(t) - 100.0))
    left_maps = np.asarray(left["audit_maps"][audit_time]["V1"], dtype=float)
    right_maps = np.asarray(right["audit_maps"][audit_time]["V1"], dtype=float)
    angles = tuple(config.GABOR_ANGLES_DEG)
    if left_maps.shape[0] != len(angles) or right_maps.shape != left_maps.shape:
        raise ValueError("V3 audit maps do not match the configured Gabor orientations")
    # Show the actually presented images here. The main step-1 figure above
    # separately shows the model input after subtracting the shared circle.
    left_stim = np.asarray(left["stimulus"].image, dtype=float) / 255.0
    right_stim = np.asarray(right["stimulus"].image, dtype=float) / 255.0
    response_limit = max(float(left_maps.max()), float(right_maps.max()), 1e-12)
    # Keep image row zero at the top, matching the source PNG/matrix convention.
    fig, axes = plt.subplots(2, 5, figsize=publication_size("double", 104),
                             layout="constrained")
    for row, (condition, stimulus, maps) in enumerate((
        ("左 cue", left_stim, left_maps), ("右 cue", right_stim, right_maps)
    )):
        im_input = axes[row, 0].imshow(
            stimulus, origin="upper", cmap="gray",
            vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[row, 0].set_title(f"{condition} 原始 cue")
        axes[row, 0].set_xlabel("x", labelpad=0)
        axes[row, 0].set_ylabel("y", labelpad=0)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for col, (angle, values) in enumerate(zip(angles, maps), start=1):
            im_response = axes[row, col].imshow(
                values, origin="upper", cmap="viridis", vmin=0.0,
                vmax=response_limit, interpolation="nearest")
            axes[row, col].set_title(f"{condition} {angle:g}°")
            axes[row, col].set_xlabel("x", labelpad=0)
            axes[row, col].set_ylabel("y", labelpad=0)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.colorbar(im_input, ax=axes[:, 0].tolist(), fraction=0.045, pad=0.025,
                 label="原始图像亮度（归一化 0–1）")
    fig.colorbar(im_response, ax=axes[:, 1:].ravel().tolist(), fraction=0.018,
                 pad=0.012, label="V3 Gabor 能量（相对单位）")
    fig.suptitle("Stage1 · V3 Gabor 方向响应总览（原始cue含圆；Gabor输入为 cue−圆形基线）",
                 fontsize=9.5, fontweight="bold")
    _save_figure(
        fig, "stage1_Gabor方向响应总览",
        _figure_contract(
            claim="Stage1 左右 cue 在 100 ms 的 V3 Gabor 方向能量空间分布互为镜像。",
            evidence=("左右 cue 相对圆形基线输入", "0°/45°/90°/135° V3 空间方向响应"),
            source_paths=("input/stage1_cue_left.npy", "input/stage1_cue_right.npy",
                          "revision_v3/frontend.py"),
            figure_role="model-result",
            scenario="Stage1 标准左右 cue；V3 前端确定性响应；刺激矩阵第 0 行显示在图像上沿。",
            statistic=f"显示 V3 audit_maps['V1']，取 {audit_time:g} ms；非旧版 scipy Gabor 结果。",
            n_definition="无单试次样本；标准刺激输入的确定性模型前向计算。",
            review_risks=("Gabor 响应为模型内部特征，不是直接观测的神经活动。",),
        ), registry, audits,
    )

    differences = left_maps - right_maps
    difference_limit = max(float(np.max(np.abs(differences))), 1e-12)
    fig, axes = plt.subplots(1, len(angles), figsize=publication_size("double", 74),
                             layout="constrained")
    for ax, angle, values in zip(axes, angles, differences):
        im = ax.imshow(values, origin="upper", cmap=DIV_CMAP,
                       vmin=-difference_limit, vmax=difference_limit,
                       interpolation="nearest")
        ax.set_title(f"{angle:g}°：左 cue − 右 cue")
        ax.set_xlabel("x", labelpad=0)
        ax.set_ylabel("y", labelpad=0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=np.asarray(axes).ravel().tolist(), fraction=0.025,
                 pad=0.02, label="左减右（相对单位）")
    fig.suptitle("Stage1 · V3 Gabor 左右差异（100 ms）",
                 fontsize=9.5, fontweight="bold")
    _save_figure(
        fig, "stage1_Gabor差异",
        _figure_contract(
            claim="V3 Gabor 各方向响应的空间差异显示左右 cue 的镜像编码。",
            evidence=("四个 Gabor 方向的左 cue 减右 cue 空间差异图",),
            source_paths=("input/stage1_cue_left.npy", "input/stage1_cue_right.npy",
                          "revision_v3/frontend.py"),
            figure_role="model-diagnostic",
            scenario="Stage1 标准左右 cue；V3 前端确定性响应；刺激矩阵第 0 行显示在图像上沿。",
            statistic=f"差异定义为左减右；显示 {audit_time:g} ms 的 V3 方向响应。",
            n_definition="无单试次样本；标准刺激输入的确定性模型前向计算。",
            review_risks=("Gabor 响应为模型内部特征，不是直接观测的神经活动。",),
        ), registry, audits,
    )

    left_cells = np.asarray([_spatial_cell_means(values) for values in left_maps])
    right_cells = np.asarray([_spatial_cell_means(values) for values in right_maps])
    cell_limit = max(float(left_cells.max()), float(right_cells.max()), 1e-12)
    fig, axes = plt.subplots(2, len(angles), figsize=publication_size("double", 105),
                             layout="constrained")
    for row, (condition, grid_values) in enumerate((
        ("左 cue", left_cells), ("右 cue", right_cells)
    )):
        for col, (angle, values) in enumerate(zip(angles, grid_values)):
            im = axes[row, col].imshow(
                values, origin="upper", cmap="viridis", vmin=0.0,
                vmax=cell_limit, interpolation="nearest")
            axes[row, col].set_title(f"{condition} · {angle:g}°")
            axes[row, col].set_xticks(range(4), labels=(1, 2, 3, 4))
            axes[row, col].set_yticks(range(4), labels=(1, 2, 3, 4))
            axes[row, col].set_xlabel("空间列")
            axes[row, col].set_ylabel("空间行")
            for r in range(4):
                for c in range(4):
                    axes[row, col].text(c, r, f"{values[r, c]:.3f}",
                                        ha="center", va="center", fontsize=5.5,
                                        color="white" if values[r, c] < cell_limit * 0.42 else INK)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.018,
                 label="V3 Gabor 能量（相对单位）")
    fig.suptitle("Stage1 · V3 Gabor 空间响应热图（100 ms）",
                 fontsize=9.5, fontweight="bold")
    _save_figure(
        fig, "stage1_Gabor空间响应热图",
        _figure_contract(
            claim="V3 Gabor 能量在刺激上方的 4×4 前端网格中呈现左右镜像空间分布。",
            evidence=("左/右 cue × 四方向的 4×4 空间平均 Gabor 能量",),
            source_paths=("input/stage1_cue_left.npy", "input/stage1_cue_right.npy",
                          "revision_v3/frontend.py"),
            figure_role="model-result",
            scenario="Stage1 标准左右 cue；V3 前端确定性响应；空间网格第 1 行位于图像上沿。",
            statistic=f"由 100 ms 的 V3 128×128 方向响应按 4×4 等面积网格取均值。",
            n_definition="无单试次样本；标准刺激输入的确定性模型前向计算。",
            review_risks=("网格热图为模型内部特征汇总，不是 EEG 通道观测。",),
        ), registry, audits,
    )


def plot_step2(data: dict, registry: list, audits: list) -> None:
    fronts = data["fronts"]
    left, right = fronts["left"], fronts["right"]
    time = np.asarray(left["time_ms"], dtype=float)
    audit_time = min(left["audit_maps"], key=lambda t: abs(float(t) - 100.0))
    l_total = left["lgn_on"][:, :, int(round(audit_time))] + left["lgn_off"][:, :, int(round(audit_time))]
    r_total = right["lgn_on"][:, :, int(round(audit_time))] + right["lgn_off"][:, :, int(round(audit_time))]
    diff = r_total - l_total
    vmax = max(float(l_total.max()), float(r_total.max()), 1e-12)
    dmax = max(float(np.max(np.abs(diff))), 1e-12)

    fig = plt.figure(figsize=publication_size("double", 139), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=(0.16, 1.0, 0.95))
    legend_ax = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, c]) for c in range(3)]
    response_ax = fig.add_subplot(grid[2, :])
    for col, (ax, values, title) in enumerate(zip(
        axes,
        (l_total, r_total, diff),
        (f"左 cue（{audit_time:g} ms）", f"右 cue（{audit_time:g} ms）", "右 − 左空间差"),
    )):
        im = ax.imshow(values, origin="upper", interpolation="nearest",
                       cmap=DIV_CMAP if col == 2 else "viridis",
                       vmin=-dmax if col == 2 else 0.0,
                       vmax=dmax if col == 2 else vmax)
        ax.set_title(title)
        ax.set_xlabel("LGN pooled x")
        ax.set_ylabel("LGN pooled y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    for signal_name, color in (("lgn_on_mean", CONDITION_COLORS["left"]),
                               ("lgn_off_mean", CONDITION_COLORS["right"])):
        for condition, front, linestyle in (("左 cue", left, "-"), ("右 cue", right, "--")):
            response_ax.plot(time, front[signal_name], color=color, linestyle=linestyle,
                             linewidth=1.1, label=f"{signal_name.removeprefix('lgn_').replace('_mean', '').upper()} · {condition}")
    _base_axis(response_ax)
    response_ax.set_title("LGN 群体 ON/OFF 平均响应；镜像 cue 的整体均值基本重合")
    response_ax.set_xlabel("cue onset 后时间（ms）")
    response_ax.set_ylabel("平均活动（相对单位）")
    lgn_handles = [
        Line2D([0], [0], color=CONDITION_COLORS["left"], lw=1.1, label="ON · 左 cue"),
        Line2D([0], [0], color=CONDITION_COLORS["left"], lw=1.1, ls="--", label="ON · 右 cue"),
        Line2D([0], [0], color=CONDITION_COLORS["right"], lw=1.1, label="OFF · 左 cue"),
        Line2D([0], [0], color=CONDITION_COLORS["right"], lw=1.1, ls="--", label="OFF · 右 cue"),
    ]
    _place_figure_legend(fig, legend_ax, lgn_handles, ncol=4, fontsize=5.8)
    fig.suptitle("02 · LGN 中继响应：整体活动与空间分布", fontsize=10, fontweight="bold")
    _save_figure(
        fig, "02_v3_lgn_response",
        _figure_contract(
            claim="镜像左右 cue 的 LGN 整体 ON/OFF 平均响应近似相同，但空间化中继响应呈镜像分布。",
            evidence=("100 ms 的 8×8 pooled ON+OFF 空间响应", "LGN ON/OFF 群体均值时程"),
            source_paths=("output/revision_v3/manifest.json", "revision_v3/frontend.py"),
            figure_role="model-result",
            scenario="Stage1 cue；使用三个重点记录已保存留出参数的中位数 tau_a。",
            statistic="空间图为 pooled ON+OFF 活动；时间曲线为各空间位置的群体均值。",
            n_definition="无单试次样本；图中为标准化 cue 输入的确定性模型前向计算。",
            review_risks=("这是确定性视觉前端的内部状态，不是 LGN 实测信号。",),
        ), registry, audits,
    )


def plot_leadfield_matrix(registry: list, audits: list) -> None:
    """Show the saved V3 sensor-by-source leadfield matrix with its units."""
    frame = _read_csv(V3_DIR / "leadfield.csv")
    geometry = _read_json(V3_DIR / "leadfield_geometry.json")
    sensors = list(geometry["sensors"])
    sources = list(geometry["sources"])
    matrix = frame[[f"source_{i + 1}" for i in range(len(sources))]].to_numpy(dtype=float)
    if matrix.shape != (len(sensors), len(sources)):
        raise ValueError(f"unexpected leadfield shape {matrix.shape}")
    limit = max(float(np.max(np.abs(matrix))), 1e-12)

    fig, ax = plt.subplots(figsize=publication_size("double", 80), layout="constrained")
    im = ax.imshow(matrix, origin="upper", cmap=DIV_CMAP, vmin=-limit, vmax=limit,
                   interpolation="nearest", aspect="auto")
    pretty_sources = [name.replace("_", "\n") for name in sources]
    ax.set_xticks(range(len(sources)), labels=pretty_sources)
    ax.set_yticks(range(len(sensors)), labels=sensors)
    ax.set_xlabel("固定源代理")
    ax.set_ylabel("传感器")
    ax.set_title("V3 几何近似导联矩阵")
    ax.tick_params(axis="x", labelsize=6.0)
    ax.tick_params(axis="y", labelsize=7.0)
    for row in range(len(sensors)):
        for col in range(len(sources)):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center",
                    fontsize=6.2, color="white" if abs(matrix[row, col]) > limit * 0.56 else INK)
    fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02,
                 label=geometry.get("leadfield_units", "V/(A m)"))
    fig.suptitle("04 · F3/Fz/F4 × 五个语义明确源代理的导联矩阵",
                 fontsize=9.5, fontweight="bold")
    _save_figure(
        fig, "导联矩阵热图",
        _figure_contract(
            claim="V3 用外表面电极、内层功能源的固定规范几何映射，将五个源代理投影到 F3/Fz/F4。",
            evidence=("保存的 3×5 传感器 × 源导联系数矩阵及源深度审计",),
            source_paths=("output/revision_v3/leadfield.csv",
                          "output/revision_v3/leadfield_geometry.json",
                          "revision_v3/head_model.py"),
            figure_role="model-diagnostic",
            scenario="canonical 几何点偶极近似；双乳突平均参考；不是个体化头模型。",
            statistic="颜色和标注为带符号导联系数；单位 V/(A m)。",
            n_definition="不涉及样本；显示固定的确定性 3×5 映射矩阵。",
            review_risks=("导联矩阵基于规范几何和均匀导体假设，传感器增益未校准为 μV。",),
        ), registry, audits,
    )


def plot_step3(data: dict, registry: list, audits: list) -> None:
    results = data["results"]
    left, right = results["left"], results["right"]
    times = left.time_ms
    pop_names = ("早期视觉输入群体", "空间构型群体", "三角模板偏好群体")
    channel_difference_names = ("左视野 − 右视野", "左视野 − 右视野",
                               "左三角模板偏好 − 右三角模板偏好")
    fig = plt.figure(figsize=publication_size("double", 159), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=(0.12, 1.0, 0.83))
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    top_axes = [fig.add_subplot(grid[1, c]) for c in range(3)]
    bottom_axes = [fig.add_subplot(grid[2, c]) for c in range(3)]

    for pop, ax in enumerate(top_axes):
        for condition, result, style in (("左 cue", left, "-"), ("右 cue", right, "--")):
            e = result.excitatory[pop].mean(axis=0)
            inh = result.inhibitory[pop].mean(axis=0)
            ax.plot(times, e, color="#3C6E8F", linestyle=style, linewidth=1.0,
                    label=f"E · {condition}")
            ax.plot(times, inh, color="#B27A45", linestyle=style, linewidth=1.0,
                    label=f"I · {condition}")
        _base_axis(ax)
        ax.set_title(pop_names[pop])
        ax.set_xlabel("时间（ms）")
        ax.set_ylabel("平均群体活动（0–1）")
        for condition, result, color in (("左 cue", left, CONDITION_COLORS["left"]),
                                         ("右 cue", right, CONDITION_COLORS["right"])):
            lateral = result.excitatory[pop, 0] - result.excitatory[pop, 1]
            ax2 = bottom_axes[pop]
            ax2.plot(times, lateral, color=color, linewidth=1.1, label=condition)
        ax2 = bottom_axes[pop]
        _base_axis(ax2)
        ax2.set_title(f"{pop_names[pop]}：{channel_difference_names[pop]}")
        ax2.set_xlabel("时间（ms）")
        ax2.set_ylabel("E 群体通道差（相对单位）")
    top_handles = [
        Line2D([0], [0], color="#3C6E8F", lw=1.0, label="E · 左 cue"),
        Line2D([0], [0], color="#3C6E8F", lw=1.0, ls="--", label="E · 右 cue"),
        Line2D([0], [0], color="#B27A45", lw=1.0, label="I · 左 cue"),
        Line2D([0], [0], color="#B27A45", lw=1.0, ls="--", label="I · 右 cue"),
    ]
    top_handles.extend([
        Line2D([0], [0], color=CONDITION_COLORS["left"], lw=1.1, label="E 差：左 cue"),
        Line2D([0], [0], color=CONDITION_COLORS["right"], lw=1.1, label="E 差：右 cue"),
    ])
    _place_figure_legend(fig, legend_ax, top_handles, ncol=6, fontsize=5.8,
                         handlelength=1.7, columnspacing=0.8)

    fig.suptitle("03 · V3 E/I 群体动力学与内部左右通道差", fontsize=9.5, fontweight="bold")
    _save_figure(
        fig, "03_v3_population_dynamics",
        _figure_contract(
            claim="V3 将视觉输入传入早期、形状和方向三个功能群体，内部左右通道响应可在时间上分化。",
            evidence=("三组 E/I 活动时程", "每组 E 群体左通道减右通道"),
            source_paths=("output/revision_v3/heldout_fit_summary.csv",
                          "output/revision_v3/drive_scales.csv", "revision_v3/model.py"),
            figure_role="model-result",
            scenario="Stage1 cue；代表性 tau_s/g_i/tau_a 和增益取三个重点记录已保存留出拟合的中位数。",
            statistic="E/I 曲线为两通道平均；下排为群体内部通道 0 − 通道 1。",
            n_definition="无单试次样本；图中为标准化 cue 输入的确定性模型前向计算。",
            review_risks=("功能群体是模型变量，不应直接等同于 V1/IT/PFC 的实测活动。",),
        ), registry, audits,
    )


def plot_step4(data: dict, registry: list, audits: list) -> None:
    results = data["results"]
    processed = {}
    times = None
    for condition, result in results.items():
        values, out_time = model_curve_to_q1_grid(result.eeg_scaled, result.time_ms)
        keep = (out_time >= 0.0) & (out_time <= 800.0)
        processed[condition] = values[:, keep]
        times = out_time[keep]

    fig = plt.figure(figsize=publication_size("double", 131), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=(0.14, 1.0, 0.85))
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    axes = [fig.add_subplot(grid[1, c]) for c in range(3)]
    diff_ax = fig.add_subplot(grid[2, :])
    for i, (channel, ax) in enumerate(zip(CHANNELS, axes)):
        for condition, linestyle in (("left", "-"), ("right", "--")):
            ax.plot(times, processed[condition][i], color=CONDITION_COLORS[condition],
                    linestyle=linestyle, linewidth=1.1,
                    label="左 cue" if condition == "left" else "右 cue")
        _base_axis(ax)
        ax.set_title(channel)
        ax.set_xlabel("cue onset 后时间（ms）")
        ax.set_ylabel("相对传感器单位")
    sensor_handles = [Line2D([0], [0], color=CHANNEL_COLORS[ch], lw=1.1, label=ch)
                      for ch in CHANNELS]
    cue_handles = [
        Line2D([0], [0], color=CONDITION_COLORS["left"], lw=1.1, label="左 cue"),
        Line2D([0], [0], color=CONDITION_COLORS["right"], lw=1.1, ls="--", label="右 cue"),
    ]
    _place_figure_legend(fig, legend_ax, cue_handles + sensor_handles,
                         ncol=5, fontsize=5.9)
    for i, channel in enumerate(CHANNELS):
        diff_ax.plot(times, processed["right"][i] - processed["left"][i],
                     color=CHANNEL_COLORS[channel], linewidth=1.15, label=channel)
    _base_axis(diff_ax)
    diff_ax.set_title("V3 传感器预测：右 − 左")
    diff_ax.set_xlabel("cue onset 后时间（ms）")
    diff_ax.set_ylabel("右减左（相对传感器单位）")
    fig.suptitle("04 · 经 Q1 观测处理后的模型传感器预测", fontsize=9.5, fontweight="bold")
    _save_figure(
        fig, "04_v3_simulated_eeg",
        _figure_contract(
            claim="V3 的几何导联近似将五个源代理映射为 F3/Fz/F4 相对传感器曲线。",
            evidence=("左/右 cue 的 F3/Fz/F4 模型波形", "模型右减左波形"),
            source_paths=("output/revision_v3/heldout_fit_summary.csv",
                          "output/revision_v3/drive_scales.csv",
                          "output/revision_v3/leadfield_geometry.json",
                          "revision_v3/observation.py"),
            figure_role="model-result",
            scenario="三个重点记录的已保存留出拟合中位数；固定规范头部几何导联；cue-only。",
            statistic="模型曲线经过 0.2–24 Hz 四阶零相位滤波、256→128 Hz 重采样和 -200–0 ms 通道基线校正。",
            n_definition="无单试次样本；图中为代表性固定参数的确定性模型前向计算。",
            review_risks=("传感器增益未标定为 μV；模型仅生成 0–800 ms cue 响应，未模拟 2.2 s 目标事件。",
                          "三个重点记录的 tau_s 候选达到上界且拟合未收敛，代表参数只用于示意。"),
        ), registry, audits,
    )


def _wide_case(df: pd.DataFrame, record: str, condition: str, value: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df.loc[(df["record"] == record) & (df["condition"] == condition)]
    pivot = sub.pivot(index="channel", columns="time_ms", values=value).reindex(CHANNELS)
    time = np.asarray(pivot.columns, dtype=float)
    return time, pivot.to_numpy(dtype=float)


def _metrics_by_record() -> tuple[pd.DataFrame, dict]:
    metrics = _read_csv(V3_DIR / "heldout_metrics.csv")
    differences = _read_csv(V3_DIR / "left_right_difference.csv")
    counts = {}
    for record in RECORD_ORDER:
        selected = metrics.loc[metrics["record"] == record]
        left = selected.loc[selected["condition"] == "left"].iloc[0]
        right = selected.loc[selected["condition"] == "right"].iloc[0]
        diff = differences.loc[differences["record"] == record].iloc[0]
        counts[record] = {
            "n_left": int(left["n_trials"]),
            "n_right": int(right["n_trials"]),
            "nrmse_left": float(left["nrmse"]),
            "nrmse_right": float(right["nrmse"]),
            "difference_nrmse": float(diff["left_right_difference_nrmse"]),
            "difference_correlation": float(diff["left_right_difference_correlation"]),
            "measured_lateral_energy_fraction": float(diff["measured_lateral_energy_fraction"]),
            "model_lateral_energy_fraction": float(diff["model_lateral_energy_fraction"]),
        }
    return metrics, counts


def plot_step5(df: pd.DataFrame, counts: dict, registry: list, audits: list) -> None:
    fig = plt.figure(figsize=publication_size("double", 219), layout="constrained")
    grid = fig.add_gridspec(5, 3, height_ratios=(0.16, 1, 1, 1, 1))
    axes = np.empty((4, 3), dtype=object)
    for row in range(4):
        for col in range(3):
            axes[row, col] = fig.add_subplot(
                grid[row + 1, col], sharex=axes[0, col] if row > 0 else None
            )
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    channel_titles = ("F3", "Fz", "F4")
    for row, record in enumerate(RECORD_ORDER):
        low_quality = record == LOW_QUALITY_RECORD
        meta = counts[record]
        for col, channel in enumerate(channel_titles):
            ax = axes[row, col]
            if low_quality:
                ax.set_facecolor("#F5F2EE")
            for condition in ("left", "right"):
                t, matrix_measured = _wide_case(df, record, condition, "measured")
                _, matrix_model = _wide_case(df, record, condition, "model_heldout")
                color = CONDITION_COLORS[condition]
                ax.plot(t, matrix_measured[col], color=color, linewidth=1.05,
                        alpha=1.0 if not low_quality else 0.86,
                        label=f"真实 · {'左' if condition == 'left' else '右'}")
                ax.plot(t, matrix_model[col], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.92 if not low_quality else 0.80,
                        label=f"V3 留出预测 · {'左' if condition == 'left' else '右'}")
            _base_axis(ax)
            ax.set_ylabel("EEG（Q1 处理单位）")
            ax.set_xlabel("cue onset 后时间（ms）")
            if row < 3:
                ax.xaxis.label.set_visible(False)
            nrmse_text = f"NRMSE L/R {meta['nrmse_left']:.2f}/{meta['nrmse_right']:.2f}"
            if col == 0:
                prefix = "低质量参考 · " if low_quality else ""
                ax.set_title(f"{prefix}{record}\nnL={meta['n_left']}, nR={meta['n_right']} · {nrmse_text}",
                             fontsize=6.2, loc="left")
            else:
                ax.set_title(channel, fontsize=7.5)
    legend = [
        Line2D([0], [0], color=CONDITION_COLORS["left"], lw=1.1, label="真实：左 cue"),
        Line2D([0], [0], color=CONDITION_COLORS["left"], lw=1.1, ls="--", label="V3 预测：左 cue"),
        Line2D([0], [0], color=CONDITION_COLORS["right"], lw=1.1, label="真实：右 cue"),
        Line2D([0], [0], color=CONDITION_COLORS["right"], lw=1.1, ls="--", label="V3 预测：右 cue"),
    ]
    _place_figure_legend(fig, legend_ax, legend, ncol=4, fontsize=6.2,
                         handlelength=2.2, columnspacing=1.0)
    fig.suptitle("05 · V3 留一记录预测与真实单记录 ERP 对照", fontsize=9.5, fontweight="bold")
    _save_figure(
        fig, "05_v3_heldout_erp",
        _figure_contract(
            claim="V3 的左右 cue 波形拟合在三个重点记录与低质量参考记录间存在异质性。",
            evidence=("四条 MAT 记录的 F3/Fz/F4 左右 cue 真实均值与 held-out 预测",
                      "每条件 trial 数和条件级 NRMSE"),
            source_paths=("output/revision_v3/heldout_predictions.csv",
                          "output/revision_v3/heldout_metrics.csv",
                          "output/revision_v3/manifest.json"),
            figure_role="model-comparison",
            scenario="每条记录由其余 MAT 记录训练；不重新拟合；低质量 VisualCogA_Task-1 保留在末行。",
            statistic="NRMSE 为模型 RMSE / 真实条件 ERP RMS；图中曲线为 trial-average ERP。",
            n_definition="每条件的有效单试次计数列于行标题。",
            review_risks=("四份 MAT 是记录，未确认是四名独立被试。",
                          "零相位滤波和 cue-only 模型的事件上下文存在已记录限制。"),
        ), registry, audits,
    )


def plot_step5_differences(df: pd.DataFrame, counts: dict,
                           registry: list, audits: list) -> None:
    fig = plt.figure(figsize=publication_size("double", 201), layout="constrained")
    grid = fig.add_gridspec(5, 3, height_ratios=(0.16, 1, 1, 1, 1))
    axes = [[fig.add_subplot(grid[r + 1, c]) for c in range(3)] for r in range(4)]
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    mode_names = ("共同模态 u0", "侧化模态 u1", "形状对比模态 u2")
    u = config.U_OBS
    for row, record in enumerate(RECORD_ORDER):
        low_quality = record == LOW_QUALITY_RECORD
        meta = counts[record]
        tl, real_l = _wide_case(df, record, "left", "measured")
        tr, real_r = _wide_case(df, record, "right", "measured")
        _, model_l = _wide_case(df, record, "left", "model_heldout")
        _, model_r = _wide_case(df, record, "right", "model_heldout")
        if not np.array_equal(tl, tr):
            raise ValueError(f"{record}: left/right time grids do not match")
        real_modes = u @ (real_r - real_l)
        model_modes = u @ (model_r - model_l)
        for col, ax in enumerate(axes[row]):
            if low_quality:
                ax.set_facecolor("#F5F2EE")
            ax.plot(tl, real_modes[col], color=INK, linewidth=1.05,
                    label="真实右 − 左")
            ax.plot(tl, model_modes[col], color="#A6534C", linewidth=1.0,
                    linestyle="--", label="V3 预测右 − 左")
            _base_axis(ax)
            ax.set_ylabel("相对模态单位")
            ax.set_xlabel("cue onset 后时间（ms）")
            if row < 3:
                ax.xaxis.label.set_visible(False)
            if col == 0:
                prefix = "低质量参考 · " if low_quality else ""
                ax.set_title(f"{prefix}{record}\nr={meta['difference_correlation']:.2f}; "
                             f"NRMSE={meta['difference_nrmse']:.2f} · {mode_names[col]}",
                             fontsize=6.0, loc="left")
            else:
                ax.set_title(mode_names[col], fontsize=7.0)
    handles = [
        Line2D([0], [0], color=INK, lw=1.05, label="真实右 − 左"),
        Line2D([0], [0], color="#A6534C", lw=1.0, ls="--", label="V3 预测右 − 左"),
    ]
    _place_figure_legend(fig, legend_ax, handles, ncol=2, fontsize=6.2)
    fig.suptitle("05b · 留出模型对左右差异波的观测模态解释", fontsize=9.5, fontweight="bold")
    _save_figure(
        fig, "05b_v3_heldout_lr_modes",
        _figure_contract(
            claim="V3 预测的左右差异主要落在侧化模态；真实右减左波形在三种观测模态中的幅度和方向依记录而异。",
            evidence=("四条记录的真实与预测右减左波形", "common/lateral/shape 观测模态",
                      "记录级差异波相关和 NRMSE"),
            source_paths=("output/revision_v3/heldout_predictions.csv",
                          "output/revision_v3/left_right_difference.csv",
                          "revision_v3/config.py"),
            figure_role="model-diagnostic",
            scenario="每条记录使用其已保存的 leave-one-MAT-out 参数；u0/u1/u2 为固定正交传感器坐标。",
            statistic="差异波=右 cue ERP−左 cue ERP；模式坐标由配置中的正交矩阵投影。",
            n_definition="trial 数与 05 主图一致。",
            review_risks=("观测模态是 F3/Fz/F4 的线性组合，不代表脑源定位。",
                          "B_Task-2 的差异波相关接近零，不能将主图中 ERP 拟合较好解读为左右效应拟合稳定。"),
        ), registry, audits,
    )


def _write_contract_inputs(data: dict, counts: dict, figure_plan: list) -> dict[str, Path]:
    contract_dir = OUT_DIR / "contracts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    fit_rows = _read_csv(V3_DIR / "heldout_fit_summary.csv")
    metrics = _read_csv(V3_DIR / "heldout_metrics.csv")
    differences = _read_csv(V3_DIR / "left_right_difference.csv")
    result_object = {
        "model": "revision_v3",
        "heldout_fit_count": int(len(fit_rows)),
        "record_order_for_figures": list(RECORD_ORDER),
        "primary_records": list(PRIMARY_RECORDS),
        "low_quality_reference_record": LOW_QUALITY_RECORD,
        "heldout_metrics": metrics.to_dict(orient="records"),
        "left_right_difference": differences.to_dict(orient="records"),
        "representative_step_01_to_04_parameters": data["fit_detail"],
        "source_manifest": "output/revision_v3/manifest.json",
    }
    quality_validation = {
        "scope": "figure and evidence caveats from saved revision_v3 artifacts",
        "fit_status": "all held-out fits reached tau_s upper bound and optimizer_success=False",
        "low_quality_record": LOW_QUALITY_RECORD,
        "record_handling": "prioritize the other three records but retain VisualCogA_Task-1 in every real-data comparison",
        "head_model": "canonical geometry-based leadfield; not individualized; gain not calibrated to microvolts",
        "stage_scope": "revision_v3 fits Stage1 cue left/right; no Stage2 target-event fit",
        "observation_scope": "model cue-only response padded into full Q1 epoch before filtering; target event is not simulated",
        "independence": "four MAT files are records; independent participant count is not established",
    }
    claims = [
        {"figure_id": item["figure_id"], "claim": item["claim"],
         "evidence": item["evidence"], "caveats": item["contract"]["review_risks"]}
        for item in figure_plan
    ]
    paths = {
        "result_object": contract_dir / "result_object.json",
        "quality_validation": contract_dir / "quality_validation.json",
        "claim_registry": contract_dir / "claim_registry.json",
        "figure_plan": contract_dir / "figure_plan.json",
    }
    for role, payload in (("result_object", result_object),
                          ("quality_validation", quality_validation),
                          ("claim_registry", claims),
                          ("figure_plan", figure_plan)):
        paths[role].write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                               encoding="utf-8")
    return paths


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _cmap_and_font_setup()
    data = _prepare_forward()
    _, counts = _metrics_by_record()
    registry: list[dict] = []
    audits: list[dict] = []

    plot_step1(data, registry, audits)
    plot_stage1_preprocessing(data, registry, audits)
    plot_step1_detail_figures(data, registry, audits)
    plot_step2(data, registry, audits)
    plot_step3(data, registry, audits)
    plot_step4(data, registry, audits)
    plot_leadfield_matrix(registry, audits)
    predictions = _read_csv(V3_DIR / "heldout_predictions.csv")
    plot_step5(predictions, counts, registry, audits)
    plot_step5_differences(predictions, counts, registry, audits)

    registry_payload = {
        "schema_version": 1,
        "model": "revision_v3",
        "source_mapping_schema": config.SOURCE_MAPPING_SCHEMA,
        "primary_records": list(PRIMARY_RECORDS),
        "retained_low_quality_record": LOW_QUALITY_RECORD,
        "representative_parameters": data["fit_detail"],
        "figures": registry,
    }
    audit_payload = {
        "schema_version": 1,
        "passed": all(item["passed"] for item in audits),
        "audits": audits,
    }
    (OUT_DIR / "figure_registry.json").write_text(
        json.dumps(registry_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "figure_audit_report.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    guide = [
        "# revision_v3：01–05 步图件索引", "",
        "主分析顺序为 VisualCogA_Task-2、VisualCogB_Task-1、VisualCogB_Task-2；",
        "VisualCogA_Task-1 按第一问质量结论标注为低质量参考，并保留在真实数据对照中。", "",
        "01–04 为使用三个重点记录已保存留出拟合中位数的确定性 cue-only 正向示意；不重新优化参数。",
        "05/05b 使用各记录自己的 saved leave-one-MAT-out 预测与真实 trial-average ERP。", "",
        "空间图按输入图像坐标显示：数组第 0 行位于图像上沿；Stage1 补充图均直接读取 V3 前端响应。",
        "stage1_Gabor* 图名沿用原图习惯，但数值来自 revision_v3，不是旧版 scipy Gabor 结果。", "",
        "v3 拟合的 tau_s 候选达到上界，优化状态未收敛；图中的代表参数只用于展示链路，",
        "不能写作已识别的生理时间常数。传感器增益也没有标定为 μV。", "",
    ]
    for item in registry:
        guide.append(f"## {item['figure_id']}")
        guide.append("")
        guide.append(item["claim"])
        guide.append("")
        guide.append("文件：" + ", ".join(f"`{path}`" for path in item["files"].values()))
        guide.append("")
    (OUT_DIR / "FIGURE_GUIDE.md").write_text("\n".join(guide), encoding="utf-8")
    contract_inputs = _write_contract_inputs(data, counts, registry)

    summary = {
        "output_dir": str(OUT_DIR),
        "figures": [entry["figure_id"] for entry in registry],
        "representative_fit": data["fit_detail"],
        "record_summary": counts,
        "audit_passed": audit_payload["passed"],
        "audit_errors": [error for audit in audits for error in audit["errors"]],
        "contract_inputs": {role: str(path) for role, path in contract_inputs.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if audit_payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
