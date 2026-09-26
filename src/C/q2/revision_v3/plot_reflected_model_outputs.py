"""Regenerate final sensor-space model figures for the selected coordinate convention.

This is a post-hoc output-space sensitivity transform, not a refit. It leaves
the visual frontend and cortical population states unchanged. PNGs only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Q2_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Q2_ROOT / "output" / "revision_v3"
OUT_DIR = Q2_ROOT / "output" / "revision_v3_modeflip"
SENSORS = ("F3", "Fz", "F4")
RECORDS = ("VisualCogA_Task-1", "VisualCogA_Task-2",
           "VisualCogB_Task-1", "VisualCogB_Task-2")
SOURCES = {
    "early_left_hemisphere_from_right_visual_field": "早期源：右视野至左半球",
    "early_right_hemisphere_from_left_visual_field": "早期源：左视野至右半球",
    "configuration_left_hemisphere_from_right_visual_field": "构型源：右视野至左半球",
    "configuration_right_hemisphere_from_left_visual_field": "构型源：左视野至右半球",
    "midline_shape_opponent": "中线形状对手源",
    "TOTAL": "总差异",
}
COLORS = ("#3B6FB6", "#E07A35", "#3A9278", "#9A6FB0", "#606060", "#202020")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from frontend import load_stimulus, simulate_frontend  # noqa: E402
from model import ModelParams, simulate_forward  # noqa: E402

U = np.array([[1., 1., 1.], [-1., 0., 1.], [1., -2., 1.]])
U[0] /= np.sqrt(3.)
U[1] /= np.sqrt(2.)
U[2] /= np.sqrt(6.)
SENSOR_REFLECTION = U.T @ np.diag([1., -1., -1.]) @ U


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": .22,
        "savefig.facecolor": "white",
    })


def reflected_source_differences() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "model_source_electrode_right_minus_left.csv",
                     encoding="utf-8-sig")
    rows = []
    keys = ["heldout_record", "source", "time_ms"]
    for _, block in df.groupby(keys, sort=False):
        block = block.set_index("sensor").reindex(SENSORS)
        if block["right_minus_left"].isna().any():
            raise ValueError("source difference row is missing a sensor")
        values = block["right_minus_left"].to_numpy(float)
        transformed = SENSOR_REFLECTION @ values
        for sensor, value in zip(SENSORS, transformed):
            rows.append({"heldout_record": block.iloc[0]["heldout_record"],
                         "source": block.iloc[0]["source"],
                         "time_ms": float(block.iloc[0]["time_ms"]),
                         "sensor": sensor, "right_minus_left": float(value)})
    return pd.DataFrame(rows)


def plot_source_contributions(df: pd.DataFrame) -> None:
    # First average within each record, then give each held-out record equal weight.
    sources = [s for s in SOURCES if s != "TOTAL"]
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 8.0), sharex=True)
    for sensor_i, (sensor, ax) in enumerate(zip(SENSORS, axes)):
        for source_i, source in enumerate(sources):
            part = df[(df.sensor == sensor) & (df.source == source)]
            per_record = part.pivot(index="time_ms", columns="heldout_record",
                                    values="right_minus_left").reindex(columns=RECORDS)
            curve = per_record.mean(axis=1)
            ax.plot(curve.index, curve.values, color=COLORS[source_i], lw=1.25,
                    label=SOURCES[source])
        total = df[(df.sensor == sensor) & (df.source == "TOTAL")]
        per_record = total.pivot(index="time_ms", columns="heldout_record",
                                 values="right_minus_left").reindex(columns=RECORDS)
        curve = per_record.mean(axis=1)
        ax.plot(curve.index, curve.values, color=COLORS[-1], lw=1.7, ls="--",
                label="总差异（五源求和）")
        ax.axhline(0, color="#777777", lw=.7)
        ax.axvline(200, color="#555555", lw=.7, ls=":")
        ax.set_ylabel(f"{sensor} 差异贡献\n（原始数据单位）")
        ax.set_xlim(0, 800)
        ax.grid(True)
    axes[-1].set_xlabel("提示出现后时间（毫秒）")
    axes[0].legend(ncol=3, frameon=True, loc="upper center", fontsize=7.5)
    fig.suptitle("五个源对三电极左右差异的贡献\n四份留出记录等权平均",
                 y=.99)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=.45)
    fig.savefig(OUT_DIR / "五源对电极左右差异贡献.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)


def plot_reflected_candidate_features() -> None:
    measured = pd.read_csv(DATA_DIR / "measured_candidate_features_exploratory.csv",
                           encoding="utf-8-sig")
    model = pd.read_csv(DATA_DIR / "model_candidate_feature_predictions.csv",
                        encoding="utf-8-sig")
    windows = ("W1_100_250", "W2_250_500", "W3_500_800")
    feature_names = [f"{window}_{sensor}" for window in windows for sensor in SENSORS]
    window_labels = {"W1_100_250": "100–250毫秒",
                     "W2_250_500": "250–500毫秒",
                     "W3_500_800": "500–800毫秒"}
    feature_labels = [f"{window_labels[window]} · {sensor}"
                      for window, sensor in ((w, s) for w in windows for s in SENSORS)]
    model_lookup = model.set_index("record")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharey=True)
    for ax, record in zip(axes.flat, RECORDS):
        part = measured[measured.record == record].set_index("feature").reindex(feature_names)
        old = np.asarray([model_lookup.loc[record,
                          f"predicted_right_minus_left_{feature}"]
                          for feature in feature_names], dtype=float).reshape(3, 3)
        new = np.stack([SENSOR_REFLECTION @ row for row in old]).reshape(-1)
        x = np.arange(len(feature_names))
        ax.errorbar(x, part.measured_right_minus_left.to_numpy(float),
                    yerr=np.vstack((
                        part.measured_right_minus_left.to_numpy(float) - part.trial_bootstrap_95ci_low.to_numpy(float),
                        part.trial_bootstrap_95ci_high.to_numpy(float) - part.measured_right_minus_left.to_numpy(float))),
                    fmt="o", color="#242424", ecolor="#777777", capsize=2,
                    label="实测右减左（试次重采样区间）")
        ax.scatter(x, new, marker="D", s=30, color="#D05A45",
                   label="模型预测", zorder=3)
        ax.axhline(0, color="#777777", lw=.7)
        ax.set_xticks(x, feature_labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(record.replace("VisualCog", "记录 ").replace("_Task-", " · 任务 "))
        ax.set_ylabel("右减左窗口均值（原始数据单位）")
        ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend(frameon=True, fontsize=7.5, loc="best")
    fig.suptitle("模型与实测的九维候选特征差异\n电极与窗口固定",
                 y=.99)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=.5, w_pad=.35)
    fig.savefig(OUT_DIR / "实测与模型九维特征.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)


def plot_effective_leadfield() -> None:
    table = pd.read_csv(DATA_DIR / "leadfield.csv", index_col=0,
                        encoding="utf-8-sig").reindex(SENSORS)
    lead = table.to_numpy(float)
    effective = SENSOR_REFLECTION @ lead
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    limit = float(np.max(np.abs(effective)))
    image = ax.imshow(effective, cmap="coolwarm", vmin=-limit, vmax=limit,
                      aspect="auto")
    ax.set_yticks(range(3), SENSORS)
    source_names = ["早期源·左半球", "早期源·右半球", "构型源·左半球",
                    "构型源·右半球", "中线形状对手源"]
    ax.set_xticks(range(5), source_names)
    ax.set_xlabel("功能源代理")
    ax.set_ylabel("观测通道")
    ax.set_title("有效源到电极映射\n该矩阵是输出映射，不是重新估计的解剖导联场")
    for i in range(3):
        for j in range(5):
            ax.text(j, i, f"{effective[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(effective[i, j]) > limit * .48 else "black")
    fig.colorbar(image, ax=ax, label="有效映射系数")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "有效源到电极映射.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)


def _representative_model():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    fits = json.loads((DATA_DIR / "heldout_fit_details.json").read_text(encoding="utf-8"))
    parameter_names = ("tau_s", "g_i", "tau_a")
    parameters = {name: float(np.median([fit["parameters"][name]
                                          for fit in fits.values()]))
                  for name in parameter_names}
    amplitude = float(np.median([fit["amplitude"] for fit in fits.values()]))
    scales_table = pd.read_csv(DATA_DIR / "drive_scales.csv", encoding="utf-8-sig")
    scales = np.asarray([
        [float(scales_table[(scales_table.population == population)
                            & scales_table.channel.str.startswith(prefix)]
               .reference_rms_scale.iloc[0]) for prefix in ("left_", "right_")]
        for population in ("early", "configuration", "shape_preference")
    ], dtype=np.float32)
    resolution = int(manifest["resolution"])
    stride = float(manifest["frontend_feature_stride_ms"])
    frontend = simulate_frontend(
        load_stimulus("Stage1", "left"), params={"tau_a": parameters["tau_a"]},
        time_ms=np.asarray(config.TIME_MS, dtype=float), resolution=resolution,
        include_offset=True, feature_stride_ms=stride)
    result = simulate_forward(frontend, params=ModelParams(**parameters),
                              amplitude=amplitude, drive_scales=scales)
    return result


def plot_reflected_cascade(result) -> None:
    time = result.time_ms
    sensor = SENSOR_REFLECTION @ result.eeg_scaled
    labels = ("早期视觉群体", "空间构型群体", "三角模板偏好群体")
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.2), sharex=True)
    for population, ax in enumerate(axes[:3]):
        ax.plot(time, result.excitatory[population].mean(axis=0),
                color="#3569A8", lw=1.35, label="兴奋性活动")
        ax.plot(time, result.inhibitory[population].mean(axis=0),
                color="#D17A35", lw=1.35, label="抑制性活动")
        ax.set_ylabel(labels[population])
        ax.set_ylim(bottom=0)
        ax.axvline(200, color="#555555", ls=":", lw=.8)
        ax.grid(True)
    for index, sensor_name in enumerate(SENSORS):
        axes[3].plot(time, sensor[index], lw=1.35, label=sensor_name)
    axes[3].set_ylabel("电极输出\n（相对单位）")
    axes[3].set_xlabel("提示出现后时间（毫秒）")
    axes[3].axvline(200, color="#555555", ls=":", lw=.8)
    axes[3].legend(ncol=3, frameon=True, loc="upper right")
    axes[0].legend(ncol=2, frameon=True, loc="upper right")
    axes[3].grid(True)
    fig.suptitle("代表参数下的模型级联与电极输出",
                 y=.99)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=.45)
    fig.savefig(OUT_DIR / "代表参数模型级联_电极输出.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)


def plot_reflected_representative_sources(result) -> None:
    lead = np.asarray(result.diagnostics["leadfield"], dtype=float)
    effective = SENSOR_REFLECTION @ lead
    source = np.asarray(result.source_proxy, dtype=float)
    gain = float(result.diagnostics["amplitude"])
    contributions = gain * effective[:, :, None] * source[None, :, :]
    total = contributions.sum(axis=1)
    expected = SENSOR_REFLECTION @ result.eeg_scaled
    error = float(np.max(np.abs(total - expected)))
    if not np.allclose(total, expected, rtol=1e-5, atol=2e-5):
        raise AssertionError("reflected source contributions do not sum to electrode output")
    source_labels = ["早期源：左半球", "早期源：右半球", "构型源：左半球",
                     "构型源：右半球", "中线形状对手源"]
    colors = ("#3B6FB6", "#70A4C7", "#3A9278", "#96BB69", "#9A6FB0")
    fig, axes = plt.subplots(3, 1, figsize=(10.7, 8.0), sharex=True)
    for sensor_i, (sensor_name, ax) in enumerate(zip(SENSORS, axes)):
        for source_i, source_name in enumerate(source_labels):
            ax.plot(result.time_ms, contributions[sensor_i, source_i],
                    color=colors[source_i], lw=1.05, label=source_name)
        ax.plot(result.time_ms, total[sensor_i], color="#202020", lw=1.7,
                ls="--", label="五源求和")
        ax.axhline(0, color="#777777", lw=.65)
        ax.axvline(200, color="#555555", ls=":", lw=.8)
        ax.set_ylabel(f"{sensor_name}\n相对单位")
        ax.grid(True)
    axes[0].legend(ncol=3, frameon=True, fontsize=7.5, loc="upper right")
    axes[-1].set_xlabel("提示出现后时间（毫秒）")
    fig.suptitle(f"五个源对三电极的贡献\n逐源贡献相加误差：{error:.2e}", y=.99)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=.4)
    fig.savefig(OUT_DIR / "代表参数五源电极贡献.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup()
    source = reflected_source_differences()
    plot_source_contributions(source)
    plot_reflected_candidate_features()
    plot_effective_leadfield()
    representative = _representative_model()
    plot_reflected_cascade(representative)
    plot_reflected_representative_sources(representative)
    # Additivity survives because the same linear transformation is applied
    # to each source contribution and to their summed sensor output.
    print(f"Generated 5 PNGs in {OUT_DIR}")
    print("sensor reflection matrix (F3,Fz,F4):")
    print(np.array2string(SENSOR_REFLECTION, precision=3, suppress_small=True))


if __name__ == "__main__":
    main()
