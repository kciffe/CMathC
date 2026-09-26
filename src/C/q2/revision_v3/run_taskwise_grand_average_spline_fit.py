"""Fit separate task-level revision_v3 models to Q1 ERP spline grand means.

Task 1 uses the two VisualCogA records; Task 2 uses the two VisualCogB records.
Each task's two records are averaged with equal record weights. F3/Fz/F4 are
calibrated independently, with left/right conditions sharing parameters within
an electrode. All included records contribute to their task mean, so reported
scores are in-sample curve-fit summaries, not independent test performance.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config
    from .fit import _cached_left_frontend
    from .frontend import load_stimulus, mirror_stage1_frontend
    from .model import calibrate_drive_scales
    from .run_grand_average_spline_fit import (
        COLORS, CONDITIONS, CONDITION_CN, _curve_metrics, _fit_channel,
        _fit_old_shared_baseline, _interpolate_channels, _load_macro_targets,
        _make_reflected_predictions, _setup_plotting,
    )
except ImportError:
    import config
    from fit import _cached_left_frontend
    from frontend import load_stimulus, mirror_stage1_frontend
    from model import calibrate_drive_scales
    from run_grand_average_spline_fit import (
        COLORS, CONDITIONS, CONDITION_CN, _curve_metrics, _fit_channel,
        _fit_old_shared_baseline, _interpolate_channels, _load_macro_targets,
        _make_reflected_predictions, _setup_plotting,
    )


OUT_DIR = config.Q2_ROOT / "output" / "revision_v3_taskwise_grand_average_fit"
TASKS = {task: tuple(records) for task, records in config.DATASETS.items()}
TASK_CN = {"Task1": "任务一", "Task2": "任务二"}


def _make_task_means(record_values):
    task_means = {}
    for task, records in TASKS.items():
        task_means[task] = {}
        for condition in CONDITIONS:
            for channel in config.CHANNELS:
                task_means[task][(condition, channel)] = np.mean(
                    [record_values[(record, condition, channel)] for record in records], axis=0)
    return task_means


def _plot_task_curves(time_ms, task_spline, task_observed, task_predictions,
                      task_metrics, path):
    _setup_plotting()
    panels = [(task, condition) for task in TASKS for condition in CONDITIONS]
    fig, axes = plt.subplots(len(panels), 3, figsize=(15, 11), sharex=True,
                             squeeze=False)
    for row, (task, condition) in enumerate(panels):
        for col, channel in enumerate(config.CHANNELS):
            ax = axes[row, col]
            ax.plot(time_ms, task_observed[task][(condition, channel)], color="#A0A0A0",
                    lw=.95, alpha=.72, label="该任务两记录实测均值")
            ax.plot(time_ms, task_spline[task][(condition, channel)],
                    color=COLORS[channel], lw=1.85, label="第一问拟合曲线均值")
            ax.plot(time_ms, task_predictions[task][condition][col], color="#C83E4D",
                    lw=1.55, ls="--", label="该任务模型拟合")
            ax.axvline(0, color="0.4", lw=.75)
            ax.axvline(200, color="0.5", lw=.75, ls=":")
            ax.axhline(0, color="0.6", lw=.65)
            metric = task_metrics[task][(condition, channel)]
            ax.set_title(f"{TASK_CN[task]} · {CONDITION_CN[condition]} · {channel}  "
                         f"(NRMSE={metric['nrmse_by_target_rms']:.2f}, "
                         f"r={metric['correlation']:.2f})")
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("电位（原始数据单位）")
            if row == len(panels) - 1:
                ax.set_xlabel("提示出现后时间（ms）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True,
               bbox_to_anchor=(.5, 1.005))
    fig.suptitle("按任务分别拟合：第一问 ERP 曲线均值与模型输出", y=1.035,
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .98), h_pad=.8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_task_differences(time_ms, task_spline, task_predictions, path):
    _setup_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, squeeze=False)
    for row, task in enumerate(TASKS):
        for col, channel in enumerate(config.CHANNELS):
            ax = axes[row, col]
            index = col
            target = (task_spline[task][("right", channel)]
                      - task_spline[task][("left", channel)])
            predicted = (task_predictions[task]["right"][index]
                         - task_predictions[task]["left"][index])
            ax.plot(time_ms, target, color=COLORS[channel], lw=1.8,
                    label="第一问拟合曲线：右减左")
            ax.plot(time_ms, predicted, color="#C83E4D", lw=1.5, ls="--",
                    label="任务模型：右减左")
            ax.axvline(0, color="0.4", lw=.75)
            ax.axvline(200, color="0.5", lw=.75, ls=":")
            ax.axhline(0, color="0.55", lw=.65)
            ax.set_title(f"{TASK_CN[task]} · {channel}")
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("右减左电位（原始数据单位）")
            if row == 1:
                ax.set_xlabel("提示出现后时间（ms）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True,
               bbox_to_anchor=(.5, 1.02))
    fig.suptitle("分任务左右差异波拟合", y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=.8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(max_nfev=38):
    config.require_current_source_mapping_manifest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    time_ms, _, _, spline_by_record, observed_by_record, trial_counts = _load_macro_targets()
    fit_mask = (time_ms >= config.FIT_WINDOW_MS[0]) & (time_ms <= config.FIT_WINDOW_MS[1])
    time_ms = time_ms[fit_mask]
    spline_by_record = {key: value[fit_mask] for key, value in spline_by_record.items()}
    observed_by_record = {key: value[fit_mask] for key, value in observed_by_record.items()}
    task_spline = _make_task_means(spline_by_record)
    task_observed = _make_task_means(observed_by_record)

    tau_a = float(config.PARAM_DEFAULTS["tau_a"])
    left_front = _cached_left_frontend(tau_a, 64)
    fronts = {"left": left_front,
              "right": mirror_stage1_frontend(left_front, load_stimulus("Stage1", "right"))}
    drive_scales = calibrate_drive_scales(fronts, feature_route=config.FRONTEND_ROUTE)

    print("分任务拟合目标：任务一=A 两记录均值；任务二=B 两记录均值；范围 0–800 ms。",
          flush=True)
    task_fits, task_predictions, task_metrics = {}, {}, {}
    parameter_rows, task_metric_rows, record_metric_rows, difference_rows = [], [], [], []
    for task, records in TASKS.items():
        print(f"开始拟合{TASK_CN[task]}：{', '.join(records)}", flush=True)
        task_fits[task] = {}
        task_predictions[task] = {condition: np.zeros((3, len(time_ms),), dtype=float)
                                  for condition in CONDITIONS}
        task_metrics[task] = {}
        for channel_index, channel in enumerate(config.CHANNELS):
            fit = _fit_channel(channel, channel_index, task_spline[task], time_ms,
                               fronts, drive_scales, tau_a=tau_a, max_nfev=max_nfev)
            task_fits[task][channel] = fit
            for condition in CONDITIONS:
                task_predictions[task][condition][channel_index] = fit["predictions"][condition][channel_index]
            parameter_rows.append({"任务": TASK_CN[task], "纳入记录": "；".join(records),
                                   "电极": channel, "tau_s_ms": fit["parameters"]["tau_s_ms"],
                                   "g_i": fit["parameters"]["g_i"], "tau_a_ms": tau_a,
                                   "左右共享增益": fit["gain"], "优化收敛": fit["success"],
                                   "函数评估次数": fit["evaluations"],
                                   "唯一参数点数": fit["n_unique_simulations"]})
            print(f"  {channel}: tau_s={fit['parameters']['tau_s_ms']:.2f} ms, "
                  f"g_i={fit['parameters']['g_i']:.3f}, gain={fit['gain']:.5g}, "
                  f"success={fit['success']}", flush=True)

        for condition in CONDITIONS:
            for channel_index, channel in enumerate(config.CHANNELS):
                key = (condition, channel)
                metric = _curve_metrics(task_spline[task][key],
                                        task_predictions[task][condition][channel_index])
                metric["against_observed_mean"] = _curve_metrics(
                    task_observed[task][key], task_predictions[task][condition][channel_index])
                task_metrics[task][key] = metric
                task_metric_rows.append({"任务": TASK_CN[task], "条件": CONDITION_CN[condition],
                                         "电极": channel,
                                         "NRMSE_任务均值样条RMS": metric["nrmse_by_target_rms"],
                                         "相关_任务均值样条": metric["correlation"],
                                         "RMSE_原始数据单位": metric["rmse"],
                                         "NRMSE_任务实测均值RMS": metric["against_observed_mean"]["nrmse_by_target_rms"],
                                         "相关_任务实测均值": metric["against_observed_mean"]["correlation"]})
                for record in records:
                    target = spline_by_record[(record, condition, channel)]
                    observed = observed_by_record[(record, condition, channel)]
                    pred = task_predictions[task][condition][channel_index]
                    m = _curve_metrics(target, pred)
                    mo = _curve_metrics(observed, pred)
                    record_metric_rows.append({"任务": TASK_CN[task], "记录": record,
                                               "条件": CONDITION_CN[condition], "电极": channel,
                                               "试次数": trial_counts[(record, condition, channel)],
                                               "NRMSE_记录样条RMS": m["nrmse_by_target_rms"],
                                               "相关_记录样条": m["correlation"],
                                               "RMSE_原始数据单位": m["rmse"],
                                               "NRMSE_记录实测均值RMS": mo["nrmse_by_target_rms"],
                                               "相关_记录实测均值": mo["correlation"]})
        for channel_index, channel in enumerate(config.CHANNELS):
            target = task_spline[task][("right", channel)] - task_spline[task][("left", channel)]
            pred = task_predictions[task]["right"][channel_index] - task_predictions[task]["left"][channel_index]
            metric = _curve_metrics(target, pred)
            difference_rows.append({"任务": TASK_CN[task], "电极": channel,
                                    "右减左_RMSE": metric["rmse"],
                                    "右减左_NRMSE": metric["nrmse_by_target_rms"],
                                    "右减左_相关": metric["correlation"],
                                    "实测差异RMS": metric["target_rms"],
                                    "模型差异RMS": metric["prediction_rms"]})

    _plot_task_curves(time_ms, task_spline, task_observed, task_predictions,
                      task_metrics, OUT_DIR / "分任务总体曲线拟合.png")
    _plot_task_differences(time_ms, task_spline, task_predictions,
                           OUT_DIR / "分任务左右差异波拟合.png")
    _write_csv(OUT_DIR / "分任务分电极拟合参数.csv", parameter_rows)
    _write_csv(OUT_DIR / "任务均值拟合指标.csv", task_metric_rows)
    _write_csv(OUT_DIR / "逐记录总体对照指标.csv", record_metric_rows)
    _write_csv(OUT_DIR / "左右差异拟合指标.csv", difference_rows)

    curve_rows = []
    for task in TASKS:
        for index, t in enumerate(time_ms):
            for condition in CONDITIONS:
                for channel_index, channel in enumerate(config.CHANNELS):
                    key = (condition, channel)
                    curve_rows.append({"任务": TASK_CN[task], "时间_ms": t,
                                       "条件": CONDITION_CN[condition], "电极": channel,
                                       "任务均值_第一问样条": task_spline[task][key][index],
                                       "任务均值_实测ERP": task_observed[task][key][index],
                                       "任务模型拟合": task_predictions[task][condition][channel_index, index]})
    _write_csv(OUT_DIR / "分任务曲线数据.csv", curve_rows)

    average_targets = np.concatenate([task_spline[t][(c, ch)]
                                      for t in TASKS for c in CONDITIONS for ch in config.CHANNELS])
    average_predictions = np.concatenate([task_predictions[t][c][i]
                                          for t in TASKS for c in CONDITIONS
                                          for i in range(len(config.CHANNELS))])
    overall_metric = _curve_metrics(average_targets, average_predictions)
    record_targets, record_predictions = [], []
    for task, records in TASKS.items():
        for record in records:
            for condition in CONDITIONS:
                for channel_index, channel in enumerate(config.CHANNELS):
                    record_targets.append(spline_by_record[(record, condition, channel)])
                    record_predictions.append(task_predictions[task][condition][channel_index])
    record_metric = _curve_metrics(np.concatenate(record_targets), np.concatenate(record_predictions))
    old_baseline = _fit_old_shared_baseline(time_ms,
                                            {key: value for key, value in
                                             _make_task_means(spline_by_record).get("Task1", {}).items()},
                                            fronts, drive_scales)
    # Evaluate the previously fitted, all-data shared parameter set over the
    # same task-specific grand means as a coarse same-target reference.
    old_baseline_by_task = {}
    if old_baseline is not None:
        old_model = old_baseline["predictions"]
        old_targets = np.concatenate([task_spline[t][(c, ch)]
                                      for t in TASKS for c in CONDITIONS for ch in config.CHANNELS])
        old_predictions = np.concatenate([old_model[c][i]
                                          for t in TASKS for c in CONDITIONS
                                          for i, ch in enumerate(config.CHANNELS)])
        old_baseline_by_task = _curve_metrics(old_targets, old_predictions)

    lines = [
        "# 按任务分开的第一问 ERP 总体曲线拟合",
        "",
        "> 评估口径：任务一用两份 A 记录，任务二用两份 B 记录；每个任务先对两份记录等权平均后拟合。随后又将任务模型与组成该任务均值的各记录逐一对照。记录参与了任务均值拟合，因此这些是总体拟合结果，不是独立留出测试。",
        "",
        "## 模型设置",
        "",
        "- 任务一：VisualCogA_Task-1 与 VisualCogA_Task-2；任务二：VisualCogB_Task-1 与 VisualCogB_Task-2。",
        "- 各任务分别使用一套前端驱动和任务级曲线目标；模型架构、视觉输入、观测滤波和固定源到电极映射相同。",
        "- F3、Fz、F4 分别拟合 `tau_s`、`g_i` 和左右条件共享的电极增益；每个电极内，左右提示使用同一组参数。`tau_a=80 ms` 固定。",
        "- 目标为第一问每记录的 ERP 样条曲线；每个任务内两记录等权。拟合区间为提示后的 0–800 ms。",
        "",
        "## 拟合结果",
        "",
        f"- 两个任务的 12 条任务均值曲线合并：NRMSE **{overall_metric['nrmse_by_target_rms']:.3f}**，相关 **{overall_metric['correlation']:.3f}**。",
        f"- 将任务级模型分别与所属的四份记录曲线对照：NRMSE **{record_metric['nrmse_by_target_rms']:.3f}**，相关 **{record_metric['correlation']:.3f}**。",
        "- 逐任务、逐电极、逐记录指标见对应 CSV；参数及优化状态见 `分任务分电极拟合参数.csv`。",
    ]
    if old_baseline_by_task:
        lines += [f"- 既有全数据共享参数模型在相同任务均值目标上的参考 NRMSE：**{old_baseline_by_task['nrmse_by_target_rms']:.3f}**，相关：**{old_baseline_by_task['correlation']:.3f}**。"]
    lines += [
        "",
        "## 左右差异波",
        "",
        "任务级模型预测的右减左差异波与第一问样条差异波的对应相关见 `左右差异拟合指标.csv`。该指标与条件 ERP 总曲线拟合分开报告，因为两条件的共同波形可以拟合较好，而较小的左右差异波仍可能拟合较差。",
        "",
        "## 解读边界",
        "",
        "按任务分开能避免把任务一与任务二的总体波形差异压进同一平均曲线。由于每个任务只有两份记录，而且两份都参与了任务曲线与参数估计，记录级对照不是未见数据上的预测。当前每个任务也各有三套电极参数，这是一种分电极拟合诊断；不能把它表述为一套完全共享的神经动力学生理参数。参数优化若未收敛，表中仍是预算内最佳候选点，不能称为最优收敛解。",
        "",
        "图中灰线为任务内等权实测 ERP 均值，彩色实线为第一问样条均值，红色虚线为任务模型曲线。",
        "",
    ]
    (OUT_DIR / "结果说明.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"两任务总体曲线：NRMSE={overall_metric['nrmse_by_target_rms']:.4f}, "
          f"r={overall_metric['correlation']:.4f}", flush=True)
    print(f"所属记录总体对照：NRMSE={record_metric['nrmse_by_target_rms']:.4f}, "
          f"r={record_metric['correlation']:.4f}", flush=True)
    for row in difference_rows:
        print(f"{row['任务']} {row['电极']} 右减左: r={row['右减左_相关']:.3f}, "
              f"NRMSE={row['右减左_NRMSE']:.3f}", flush=True)
    print(f"输出目录：{OUT_DIR}", flush=True)
    return {"overall_task_means": overall_metric, "all_records": record_metric,
            "difference_metrics": difference_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=38)
    args = parser.parse_args()
    run(max_nfev=args.max_nfev)
