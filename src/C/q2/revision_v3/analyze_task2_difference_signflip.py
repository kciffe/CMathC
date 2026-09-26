"""Quantify the effect of reversing only Task 2's modeled left-right contrast.

This is a post-fit sign sensitivity check. Reversing the contrast while keeping
the common response fixed is equivalent to swapping the model's left/right
condition curves; it is not a refitted model or an independent validation.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Q2_ROOT = Path(__file__).resolve().parents[1]
FIT_DIR = Q2_ROOT / "output" / "revision_v3_taskwise_grand_average_fit"
INPUT = FIT_DIR / "分任务曲线数据.csv"
CHANNELS = ("F3", "Fz", "F4")
COLORS = {"F3": "#3569A8", "Fz": "#168B78", "F4": "#D77932"}


def _metric(y, z):
    y, z = np.asarray(y, float), np.asarray(z, float)
    rmse = float(np.sqrt(np.mean((y - z) ** 2)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    corr = float(np.corrcoef(y, z)[0, 1]) if np.std(y) > 1e-12 and np.std(z) > 1e-12 else float("nan")
    return {"rmse": rmse, "nrmse": rmse / rms if rms else float("nan"),
            "correlation": corr, "target_rms": rms,
            "model_rms": float(np.sqrt(np.mean(z ** 2)))}


def run():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    data = pd.read_csv(INPUT, encoding="utf-8-sig")
    data = data[data["任务"] == "任务二"].copy()
    if set(data["条件"].unique()) != {"左提示", "右提示"}:
        raise ValueError("Task 2 must contain left and right condition curves")

    by_curve = {(row["条件"], row["电极"]): block.sort_values("时间_ms")
                for (condition, channel), block in data.groupby(["条件", "电极"], sort=False)
                for row in [block.iloc[0]]}
    # Align each condition/channel pair by its time vector before calculating.
    metrics_rows = []
    condition_targets, condition_predictions, time_by_channel = {}, {}, {}
    difference_curves = {}
    for channel in CHANNELS:
        left = by_curve[("左提示", channel)]
        right = by_curve[("右提示", channel)]
        if not np.array_equal(left["时间_ms"].to_numpy(), right["时间_ms"].to_numpy()):
            raise ValueError(f"Task 2 {channel}: condition time grids differ")
        time_by_channel[channel] = left["时间_ms"].to_numpy(float)
        for condition, block in (("左提示", left), ("右提示", right)):
            condition_targets[(condition, channel)] = block["任务均值_第一问样条"].to_numpy(float)
            condition_predictions[(condition, channel)] = block["任务模型拟合"].to_numpy(float)
        target_diff = (condition_targets[("右提示", channel)]
                       - condition_targets[("左提示", channel)])
        model_diff = (condition_predictions[("右提示", channel)]
                      - condition_predictions[("左提示", channel)])
        original = _metric(target_diff, model_diff)
        reversed_metric = _metric(target_diff, -model_diff)
        difference_curves[channel] = (target_diff, model_diff)
        for version, metric in (("原方向", original), ("仅反转任务二左右差异", reversed_metric)):
            metrics_rows.append({"分析对象": "左右差异波", "电极": channel, "版本": version,
                                 "RMSE": metric["rmse"], "NRMSE": metric["nrmse"],
                                 "相关": metric["correlation"],
                                 "目标RMS": metric["target_rms"], "模型RMS": metric["model_rms"]})

    # A sign flip of R-L around the same common ERP is exactly equivalent to
    # exchanging which predicted curve is assigned to left versus right.
    for channel in CHANNELS:
        for condition in ("左提示", "右提示"):
            other = "右提示" if condition == "左提示" else "左提示"
            target = condition_targets[(condition, channel)]
            pred_before = condition_predictions[(condition, channel)]
            pred_after = condition_predictions[(other, channel)]
            for version, prediction in (("原方向", pred_before), ("交换左右预测", pred_after)):
                metric = _metric(target, prediction)
                metrics_rows.append({"分析对象": "条件ERP", "电极": channel,
                                     "版本": f"{version}_{condition}",
                                     "RMSE": metric["rmse"], "NRMSE": metric["nrmse"],
                                     "相关": metric["correlation"],
                                     "目标RMS": metric["target_rms"], "模型RMS": metric["model_rms"]})

    original_targets, original_predictions, swapped_predictions = [], [], []
    for channel in CHANNELS:
        for condition in ("左提示", "右提示"):
            original_targets.append(condition_targets[(condition, channel)])
            original_predictions.append(condition_predictions[(condition, channel)])
            swapped_condition = "右提示" if condition == "左提示" else "左提示"
            swapped_predictions.append(condition_predictions[(swapped_condition, channel)])
    overall_original = _metric(np.concatenate(original_targets), np.concatenate(original_predictions))
    overall_swapped = _metric(np.concatenate(original_targets), np.concatenate(swapped_predictions))
    metrics_rows.extend([
        {"分析对象": "六条条件曲线整体", "电极": "全部", "版本": "原方向",
         "RMSE": overall_original["rmse"], "NRMSE": overall_original["nrmse"],
         "相关": overall_original["correlation"], "目标RMS": overall_original["target_rms"],
         "模型RMS": overall_original["model_rms"]},
        {"分析对象": "六条条件曲线整体", "电极": "全部", "版本": "交换左右预测",
         "RMSE": overall_swapped["rmse"], "NRMSE": overall_swapped["nrmse"],
         "相关": overall_swapped["correlation"], "目标RMS": overall_swapped["target_rms"],
         "模型RMS": overall_swapped["model_rms"]},
    ])

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.grid": True, "grid.alpha": .22})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, channel in zip(axes, CHANNELS):
        time = time_by_channel[channel]
        target, model = difference_curves[channel]
        ax.plot(time, target, color=COLORS[channel], lw=2, label="第一问拟合曲线：右减左")
        ax.plot(time, model, color="#C83E4D", lw=1.5, ls="--", label="模型原方向")
        ax.plot(time, -model, color="#26856E", lw=1.5, ls=":", label="模型差异反号")
        ax.axhline(0, color="0.5", lw=.7)
        ax.axvline(200, color="0.5", lw=.7, ls=":")
        ax.set_title(channel)
        ax.set_xlabel("提示出现后时间（ms）")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("右减左电位（原始数据单位）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True,
               bbox_to_anchor=(.5, 1.03))
    fig.suptitle("任务二左右差异波方向反转敏感性", y=1.12,
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(FIT_DIR / "任务二左右差异反号对照.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(FIT_DIR / "任务二左右差异反号指标.csv", index=False, encoding="utf-8-sig")
    diff_before = metrics[(metrics["分析对象"] == "左右差异波") & (metrics["版本"] == "原方向")]
    diff_after = metrics[(metrics["分析对象"] == "左右差异波") & (metrics["版本"] == "仅反转任务二左右差异")]
    mean_before = float(diff_before["相关"].mean())
    mean_after = float(diff_after["相关"].mean())
    text = [
        "# 任务二左右差异符号反转检查",
        "",
        "> 这是对已拟合曲线做的后验符号敏感性分析；没有重新拟合参数，也没有引入留出数据。",
        "",
        "只反转任务二的左右差异分量、保留共同 ERP 不变，数学上等价于交换任务二模型的左/右条件预测曲线。若将整条电极波形乘以 −1，则共同响应也被反转，不是本分析采用的做法。",
        "",
        f"- 三电极左右差异相关的平均值：原方向 **{mean_before:.3f}**，反转后 **{mean_after:.3f}**。",
        f"- 六条条件曲线整体：原方向 NRMSE **{overall_original['nrmse']:.3f}**、相关 **{overall_original['correlation']:.3f}**；交换左右预测后 NRMSE **{overall_swapped['nrmse']:.3f}**、相关 **{overall_swapped['correlation']:.3f}**。",
        "- 分电极差异波及单条件 ERP 的 RMSE、NRMSE、相关见 `任务二左右差异反号指标.csv`；曲线见 `任务二左右差异反号对照.png`。",
        "",
        "解释：反号会按代数关系翻转差异波相关，但只有在交换后单条件 ERP 拟合也保持或变好时，才支持左右条件标记可能颠倒。若差异波相关变正而条件 ERP 拟合明显变差，这只能说明一个后验符号调整可以对齐差异方向，不能证明原模型训练方向确实错了。",
        "",
    ]
    (FIT_DIR / "任务二左右差异反号分析.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Task 2 delta correlation mean: {mean_before:.4f} -> {mean_after:.4f}")
    print(f"All six condition curves: NRMSE {overall_original['nrmse']:.4f} -> "
          f"{overall_swapped['nrmse']:.4f}; correlation "
          f"{overall_original['correlation']:.4f} -> {overall_swapped['correlation']:.4f}")


if __name__ == "__main__":
    run()
