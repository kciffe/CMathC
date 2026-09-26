"""Plot the post-hoc u1/u2 sign-reflection sensitivity check.

This script does not refit the forward model. It reads the saved leave-one-MAT
predictions, leaves u0 unchanged, and negates u1 and u2 in sensor space. The
outputs are diagnostic PNGs and must not be described as independent validation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


Q2_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Q2_ROOT / "output" / "revision_v3" / "heldout_predictions.csv"
OUT_DIR = Q2_ROOT / "output" / "revision_v3_modeflip"
CHANNELS = ("F3", "Fz", "F4")
RECORDS = (
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
)
COLORS = {"left": "#3975A5", "right": "#D57943"}
INK = "#262626"

# Rows map F3/Fz/F4 sensor values to the mutually orthogonal u0/u1/u2 modes.
U = np.array([
    [1.0, 1.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, -2.0, 1.0],
], dtype=float)
U[0] /= np.sqrt(3.0)
U[1] /= np.sqrt(2.0)
U[2] /= np.sqrt(6.0)
MODE_REFLECTION = np.diag([1.0, -1.0, -1.0])
SENSOR_REFLECTION = U.T @ MODE_REFLECTION @ U


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": "#555555",
        "axes.grid": True,
        "grid.color": "#D7D7D7",
        "grid.alpha": 0.6,
        "grid.linewidth": 0.45,
        "savefig.facecolor": "white",
    })


def _read() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    expected = {"record", "condition", "time_ms", "channel", "measured", "model_heldout"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in heldout predictions: {sorted(missing)}")
    df = df.copy()
    df["model_reflected"] = np.nan
    for (_, _), indices in df.groupby(["record", "condition", "time_ms"]).groups.items():
        block = df.loc[indices]
        if set(block["channel"]) != set(CHANNELS):
            raise ValueError("Each record/condition/time must contain F3, Fz, and F4")
        ordered = block.set_index("channel").loc[list(CHANNELS)]
        original = ordered["model_heldout"].to_numpy(dtype=float)
        reflected = SENSOR_REFLECTION @ original
        df.loc[ordered.index, "model_reflected"] = reflected
    if df["model_reflected"].isna().any():
        raise ValueError("Could not transform all saved model predictions")
    return df


def _wide(df: pd.DataFrame, record: str, condition: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    part = df[(df.record == record) & (df.condition == condition)]
    wide = part.pivot(index="time_ms", columns="channel", values=field).sort_index()
    if tuple(wide.columns) != CHANNELS:
        wide = wide.loc[:, list(CHANNELS)]
    return wide.index.to_numpy(dtype=float), wide.to_numpy(dtype=float).T


def _corr_and_nrmse(measured: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    a = np.asarray(measured, dtype=float).ravel()
    b = np.asarray(predicted, dtype=float).ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else float("nan")
    rms = float(np.sqrt(np.mean(a * a)))
    nrmse = float(np.sqrt(np.mean((a - b) ** 2)) / rms) if rms else float("nan")
    return corr, nrmse


def _difference_mode_arrays(df: pd.DataFrame, record: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    t_left, left = _wide(df, record, "left", field)
    t_right, right = _wide(df, record, "right", field)
    if not np.array_equal(t_left, t_right):
        raise ValueError(f"Time grids differ in {record}")
    return t_left, U @ (right - left)


def _plot_mode_comparison(df: pd.DataFrame) -> None:
    mode_names = ("共同模态 u₀", "侧化模态 u₁", "形状对比模态 u₂")
    real_by_record = []
    old_by_record = []
    new_by_record = []
    for record in RECORDS:
        t, real_modes = _difference_mode_arrays(df, record, "measured")
        _, old_modes = _difference_mode_arrays(df, record, "model_heldout")
        _, new_modes = _difference_mode_arrays(df, record, "model_reflected")
        real_by_record.append(real_modes)
        old_by_record.append(old_modes)
        new_by_record.append(new_modes)

    # Equal-record macro mean: each MAT record contributes one curve.
    real_mean = np.mean(real_by_record, axis=0)
    old_mean = np.mean(old_by_record, axis=0)
    new_mean = np.mean(new_by_record, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.75), sharex=True)
    for mode_i, ax in enumerate(axes):
        ax.plot(t, real_mean[mode_i], color=INK, lw=1.55, label="实测右减左")
        ax.plot(t, old_mean[mode_i], color="#B65D4C", lw=1.35, ls="--", label="原模型右减左")
        ax.plot(t, new_mean[mode_i], color="#3975A5", lw=1.35, ls="-.", label="u₁、u₂反号后")
        ax.axhline(0, color="#777777", lw=0.65)
        ax.axvline(0, color="#555555", lw=0.65, ls=":")
        ax.set_title(mode_names[mode_i], pad=7)
        ax.set_xlim(0, 800)
        ax.set_xlabel("提示出现后时间（ms）")
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("右减左响应（原始数据单位）")
    handles = [
        Line2D([0], [0], color=INK, lw=1.55, label="实测右减左"),
        Line2D([0], [0], color="#B65D4C", lw=1.35, ls="--", label="原模型右减左"),
        Line2D([0], [0], color="#3975A5", lw=1.35, ls="-.", label="u₁、u₂反号后"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 1.02), fancybox=False, edgecolor="#888888")
    fig.suptitle("左右差异模式：实测、原模型与后两模态反号", y=1.10, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "左右差异模式_后两模态反号.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_reflected_erp(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(14.0, 10.6), sharex=True)
    for row, record in enumerate(RECORDS):
        record_name = record.replace("VisualCog", "记录").replace("_Task-", "·任务")
        for col, channel in enumerate(CHANNELS):
            ax = axes[row, col]
            for condition in ("left", "right"):
                time_ms, observed = _wide(df, record, condition, "measured")
                _, prediction = _wide(df, record, condition, "model_reflected")
                ax.plot(time_ms, observed[col], color=COLORS[condition], lw=1.2,
                        label=f"实测{'左' if condition == 'left' else '右'}提示")
                ax.plot(time_ms, prediction[col], color=COLORS[condition], lw=1.2,
                        ls="--", label=f"反号后预测{'左' if condition == 'left' else '右'}提示")
            ax.axhline(0, color="#777777", lw=0.55)
            ax.axvline(0, color="#555555", lw=0.65, ls=":")
            ax.set_xlim(0, 800)
            ax.set_axisbelow(True)
            ax.grid(True)
            if row == 0:
                ax.set_title(channel, fontsize=9, pad=7)
            if col == 0:
                ax.set_ylabel(f"{record_name}\n电位（原始数据单位）")
            if row == 3:
                ax.set_xlabel("提示出现后时间（ms）")
    handles = [
        Line2D([0], [0], color=COLORS["left"], lw=1.25, label="实测左提示"),
        Line2D([0], [0], color=COLORS["left"], lw=1.25, ls="--", label="反号后预测左提示"),
        Line2D([0], [0], color=COLORS["right"], lw=1.25, label="实测右提示"),
        Line2D([0], [0], color=COLORS["right"], lw=1.25, ls="--", label="反号后预测右提示"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.995), fancybox=False, edgecolor="#888888")
    fig.suptitle("留一记录实测 ERP 与后两模态反号后的模型预测", y=1.035, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.25, w_pad=0.85)
    fig.savefig(OUT_DIR / "留一记录ERP预测_后两模态反号.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _print_metrics(df: pd.DataFrame) -> None:
    rows = []
    condition_nrmse_before = []
    condition_nrmse_after = []
    for record in RECORDS:
        _, measured_left = _wide(df, record, "left", "measured")
        _, measured_right = _wide(df, record, "right", "measured")
        _, old_left = _wide(df, record, "left", "model_heldout")
        _, old_right = _wide(df, record, "right", "model_heldout")
        _, new_left = _wide(df, record, "left", "model_reflected")
        _, new_right = _wide(df, record, "right", "model_reflected")
        real_diff = measured_right - measured_left
        old_diff = old_right - old_left
        new_diff = new_right - new_left
        old_corr, old_nrmse = _corr_and_nrmse(real_diff, old_diff)
        new_corr, new_nrmse = _corr_and_nrmse(real_diff, new_diff)
        rows.append((record, old_nrmse, new_nrmse, old_corr, new_corr))
        for measured, old, new in ((measured_left, old_left, new_left),
                                   (measured_right, old_right, new_right)):
            condition_nrmse_before.append(_corr_and_nrmse(measured, old)[1])
            condition_nrmse_after.append(_corr_and_nrmse(measured, new)[1])
    print("记录 | 差异NRMSE 原→反号 | 差异相关 原→反号")
    for row in rows:
        print(f"{row[0]} | {row[1]:.3f}→{row[2]:.3f} | {row[3]:+.3f}→{row[4]:+.3f}")
    print(f"差异相关宏平均: {np.mean([r[3] for r in rows]):+.3f}→{np.mean([r[4] for r in rows]):+.3f}")
    print(f"差异NRMSE宏平均: {np.mean([r[1] for r in rows]):.3f}→{np.mean([r[2] for r in rows]):.3f}")
    print("左右条件 ERP NRMSE 宏平均: "
          f"{np.mean(condition_nrmse_before):.3f}→{np.mean(condition_nrmse_after):.3f}")
    print("传感器反射矩阵 (F3,Fz,F4 顺序):")
    print(np.array2string(SENSOR_REFLECTION, precision=3, suppress_small=True))


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup()
    df = _read()
    _plot_mode_comparison(df)
    _plot_reflected_erp(df)
    _print_metrics(df)
    print(f"PNG 输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
