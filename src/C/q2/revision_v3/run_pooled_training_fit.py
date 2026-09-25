"""Fit one shared revision_v3 model to all A/B Stage1 record ERPs.

Every included record contributes to parameter estimation. All resulting
metrics and plots are therefore in-sample training-fit summaries, not
validation or test performance.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config
    from .evaluate import erp_metrics
    from .fit import _cached_left_frontend, _interpolate_prediction, fit_model
    from .frontend import load_stimulus, mirror_stage1_frontend
    from .model import ModelParams, calibrate_drive_scales, simulate_forward
    from .observation import model_curve_to_q1_grid
    from .real_data import load_cases_with_audit
except ImportError:
    import config
    from evaluate import erp_metrics
    from fit import _cached_left_frontend, _interpolate_prediction, fit_model
    from frontend import load_stimulus, mirror_stage1_frontend
    from model import ModelParams, calibrate_drive_scales, simulate_forward
    from observation import model_curve_to_q1_grid
    from real_data import load_cases_with_audit


OUTPUT_DIR = config.Q2_ROOT / "output" / "revision_v3_pooled_training_fit"
CHANNEL_COLORS = {"F3": "#3b6fb6", "Fz": "#26856e", "F4": "#c15d3b"}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(value), ensure_ascii=False)
                             if isinstance(value, (dict, list, tuple, np.ndarray))
                             else value for key, value in row.items()})


def _corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _make_predictions(cases, fit, drive_scales):
    params = ModelParams.from_any(fit.parameters)
    left_front = _cached_left_frontend(float(params.tau_a), int(fit.resolution))
    fronts = {
        "left": left_front,
        "right": mirror_stage1_frontend(left_front, load_stimulus("Stage1", "right")),
    }
    model_curves = {}
    for condition, front in fronts.items():
        result = simulate_forward(front, params=params, amplitude=fit.amplitude,
                                  feature_route=fit.feature_route,
                                  drive_scales=drive_scales)
        processed, processed_time = model_curve_to_q1_grid(result.eeg_scaled,
                                                            result.time_ms)
        model_curves[condition] = (processed, processed_time)

    predictions = {}
    for case in cases:
        curve, curve_time = model_curves[case["condition"]]
        predictions[(case["dataset"], case["condition"])] = _interpolate_prediction(
            curve, curve_time, case["time_ms"])
    return predictions


def _metric_rows(cases, predictions):
    rows = []
    for case in cases:
        key = (case["dataset"], case["condition"])
        pred = predictions[key]
        metric = erp_metrics(case["real"], pred, case["time_ms"], "Stage1")
        row = {
            "evaluation_scope": "IN_SAMPLE_TRAINING_FIT_NOT_VALIDATION",
            "dataset": case["dataset"], "task": case["task"],
            "condition": case["condition"], "n_trials": case["n_trials"],
            "rmse": metric["full_sensor_rmse"],
            "nrmse_by_observed_rms": metric["full_sensor_nrmse_by_real_rms"],
            "fit_skill_vs_zero": 1.0 - metric["full_sensor_nrmse_by_real_rms"] ** 2,
            "mean_channel_correlation": metric["channel_corr_mean"],
            "F3_correlation": metric["channel_correlations"][0],
            "Fz_correlation": metric["channel_correlations"][1],
            "F4_correlation": metric["channel_correlations"][2],
            "mean_observable_mode_correlation": metric["mode_corr_mean"],
            "fit_time_start_ms": float(case["time_ms"][0]),
            "fit_time_end_ms": float(case["time_ms"][-1]),
        }
        rows.append(row)
    return rows


def _difference_rows(cases, predictions):
    groups = {}
    for case in cases:
        groups.setdefault(case["dataset"], {})[case["condition"]] = case
    rows = []
    for dataset, pair in groups.items():
        if set(pair) != {"left", "right"}:
            continue
        measured = pair["right"]["real"] - pair["left"]["real"]
        predicted = (predictions[(dataset, "right")]
                     - predictions[(dataset, "left")])
        error = measured - predicted
        measured_rms = float(np.sqrt(np.mean(measured ** 2)))
        row = {
            "evaluation_scope": "IN_SAMPLE_TRAINING_FIT_NOT_VALIDATION",
            "dataset": dataset, "task": pair["left"]["task"],
            "n_left_trials": pair["left"]["n_trials"],
            "n_right_trials": pair["right"]["n_trials"],
            "difference_rmse": float(np.sqrt(np.mean(error ** 2))),
            "difference_nrmse_by_observed_rms": (
                float(np.sqrt(np.mean(error ** 2)) / measured_rms)
                if measured_rms > 0 else float("nan")),
            "difference_waveform_correlation": _corr(measured.ravel(), predicted.ravel()),
            "observed_difference_rms": measured_rms,
            "predicted_difference_rms": float(np.sqrt(np.mean(predicted ** 2))),
        }
        for index, channel in enumerate(config.CHANNELS):
            row[f"{channel}_difference_correlation"] = _corr(measured[index], predicted[index])
        rows.append(row)
    return rows


def _plot_fit(cases, predictions, path):
    by_record = {record: {} for record in sorted({case["dataset"] for case in cases})}
    for case in cases:
        by_record[case["dataset"]][case["condition"]] = case
    fig, axes = plt.subplots(len(by_record), 2, figsize=(13, 3.0 * len(by_record)),
                             sharex=True, squeeze=False)
    for row_index, (record, conditions) in enumerate(by_record.items()):
        for col_index, condition in enumerate(("left", "right")):
            case = conditions[condition]
            ax = axes[row_index, col_index]
            measured = case["real"]
            predicted = predictions[(record, condition)]
            for channel_index, channel in enumerate(config.CHANNELS):
                color = CHANNEL_COLORS[channel]
                ax.plot(case["time_ms"], measured[channel_index], color=color, lw=1.3,
                        label=f"{channel} observed")
                ax.plot(case["time_ms"], predicted[channel_index], color=color, lw=1.3,
                        ls="--", label=f"{channel} model")
            ax.axvline(0, color="0.45", lw=0.8)
            ax.axvline(200, color="0.45", lw=0.8, ls=":")
            ax.axhline(0, color="0.65", lw=0.7)
            ax.set_title(f"{record} · {condition} cue · n={case['n_trials']}")
            ax.set_ylabel("Q1-clean EEG (input units)")
            ax.grid(alpha=0.18)
    for ax in axes[-1, :]:
        ax.set_xlabel("Cue-relative time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=8)
    fig.suptitle("ALL A/B RECORDS USED FOR FIT — in-sample training fit (not validation)",
                 y=1.035, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_differences(cases, predictions, path):
    groups = {}
    for case in cases:
        groups.setdefault(case["dataset"], {})[case["condition"]] = case
    records = [record for record in sorted(groups)
               if set(groups[record]) == {"left", "right"}]
    fig, axes = plt.subplots(len(records), len(config.CHANNELS),
                             figsize=(14, 2.7 * len(records)), sharex=True, squeeze=False)
    for row, record in enumerate(records):
        left, right = groups[record]["left"], groups[record]["right"]
        if not np.array_equal(left["time_ms"], right["time_ms"]):
            raise ValueError(f"{record}: left/right EEG time grids do not match")
        measured = right["real"] - left["real"]
        predicted = predictions[(record, "right")] - predictions[(record, "left")]
        for col, channel in enumerate(config.CHANNELS):
            ax = axes[row, col]
            ax.plot(left["time_ms"], measured[col], color="#222222", lw=1.25,
                    label="observed right − left")
            ax.plot(left["time_ms"], predicted[col], color="#bd4b3b", lw=1.25,
                    ls="--", label="model right − left")
            ax.axvline(0, color="0.45", lw=0.8)
            ax.axvline(200, color="0.45", lw=0.8, ls=":")
            ax.axhline(0, color="0.65", lw=0.7)
            ax.set_title(f"{record} · {channel}")
            ax.grid(alpha=0.18)
            if col == 0:
                ax.set_ylabel("right − left (input units)")
            if row == len(records) - 1:
                ax.set_xlabel("Cue-relative time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Observed and modelled left–right ERP differences — in-sample fit",
                 y=1.035, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_report(path, fit, cases, metric_rows, difference_rows):
    nrmse = np.asarray([row["nrmse_by_observed_rms"] for row in metric_rows], dtype=float)
    corr = np.asarray([row["mean_channel_correlation"] for row in metric_rows], dtype=float)
    diff_nrmse = np.asarray([row["difference_nrmse_by_observed_rms"]
                             for row in difference_rows], dtype=float)
    diff_corr = np.asarray([row["difference_waveform_correlation"]
                            for row in difference_rows], dtype=float)
    old_heldout_path = config.OUTPUT_ROOT / "heldout_metrics.csv"
    old_heldout_nrmse = []
    if old_heldout_path.exists():
        with old_heldout_path.open(newline="", encoding="utf-8-sig") as stream:
            old_heldout_nrmse = [float(row["nrmse"]) for row in csv.DictReader(stream)
                                 if row.get("nrmse") not in (None, "")]
    trial_counts = {}
    for case in cases:
        trial_counts.setdefault(case["dataset"], {})[case["condition"]] = case["n_trials"]
    lines = [
        "# revision_v3：A/B 四记录联合训练内拟合",
        "",
        "> **评估口径：训练内拟合，不是验证或测试。** 四份记录的左右 ERP 均参与参数估计；以下拟合分数只能描述模型对参与拟合数据的贴合程度，不能作为泛化性能或独立验证成绩。",
        "",
        "## 联合拟合设置",
        "",
        f"- 记录数：{len(fit.train_records)}；记录：{', '.join(fit.train_records)}。",
        f"- 纳入条件：每份记录的 Stage1 左提示与右提示 ERP，共 {len(cases)} 个条件均值。",
        f"- 条件试次数：`{json.dumps(trial_counts, ensure_ascii=False)}`。",
        "- 共享参数：一套 `tau_s`、`g_i`、`tau_a` 与一个有符号共享增益；记录与左右条件按相同权重参与拟合。",
        f"- 参数结果：`tau_s={fit.parameters['tau_s']:.3f} ms`，`g_i={fit.parameters['g_i']:.4f}`，`tau_a={fit.parameters['tau_a']:.3f} ms`，共享增益 `{fit.amplitude:.6g}`。",
        f"- 优化状态：`{fit.status}`；边界标记：`{json.dumps(fit.boundary_flags, ensure_ascii=False)}`；目标函数评估次数：{fit.n_evaluations}。",
        "- 模型输出已经过与第一问相同的完整 epoch 滤波、重采样和基线流程，再截取提示后的 0–800 ms 与观测 ERP 对比。",
        "",
        "## 训练内拟合汇总",
        "",
        f"- 8 个记录×方向条件的平均 NRMSE：**{np.nanmean(nrmse):.3f}**（中位数 {np.nanmedian(nrmse):.3f}）。",
        f"- 平均通道相关：**{np.nanmean(corr):.3f}**（中位数 {np.nanmedian(corr):.3f}）。",
        f"- 4 份记录右减左差异波形的平均 NRMSE：**{np.nanmean(diff_nrmse):.3f}**；平均相关：**{np.nanmean(diff_corr):.3f}**。",
        (f"- 作为口径参照，既有留一记录结果的平均 NRMSE 为 **{np.mean(old_heldout_nrmse):.3f}**；本轮训练内数值虽较低，但因四份记录均参与拟合，不能解释成泛化性能提升。"
         if old_heldout_nrmse else "- 未找到既有留一记录指标文件，未做跨口径参照。"),
        f"- `tau_s` 命中允许上界 {config.PARAM_BOUNDS['tau_s'][1]:g} ms；即使优化状态收敛，该参数仍是边界解，不能视为已由数据识别出的内部最优值。",
        f"- {int(np.count_nonzero(nrmse > 1.0))}/{len(nrmse)} 个条件的 NRMSE 大于 1，表示这些条件下模型误差超过零预测基准对应的观测 RMS。",
        "- 分记录、分条件和分左右差异指标见 `训练内拟合指标.csv` 与 `左右差异训练内拟合.csv`；对应曲线见两张 PNG。",
        "",
        "## 解释限制",
        "",
        "1. 这里拟合的是四份记录的条件平均 ERP，而不是对未见试次或未见记录的预测。共享参数由所有纳入记录共同估计，所以不能把这些同一批数据上的误差、相关或 `fit_skill_vs_zero` 称为验证分数。",
        "2. 多记录共享参数可减少每份记录单独拟合的自由度，并回答‘一套共享动力学能否描述这些记录的平均响应’；它本身不能证明受试者间参数同质，也不能替代留出评估。当前一个共享增益也无法反映各记录响应幅度差异。",
        "3. 要报告泛化表现，必须另行保留完整记录或连续试次块作为完全未参与参数估计的留出集；不能把已参与拟合的数据再次命名为验证集。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(max_nfev=None, progress=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases, audit = load_cases_with_audit()
    eligible = [case for case in cases
                if case["stage"] == "Stage1" and case["eligible_fit"]]
    expected_records = sorted(record for records in config.DATASETS.values()
                              for record in records)
    actual_records = sorted({case["dataset"] for case in eligible})
    if actual_records != expected_records:
        raise ValueError(f"expected all A/B records {expected_records}, found {actual_records}")

    left_front = _cached_left_frontend(float(config.TAU_ADAPT_DEFAULT), 64)
    right_front = mirror_stage1_frontend(left_front, load_stimulus("Stage1", "right"))
    drive_scales = calibrate_drive_scales({"left": left_front, "right": right_front},
                                          feature_route=config.FRONTEND_ROUTE)
    fit_kwargs = {"resolution": 64, "feature_route": config.FRONTEND_ROUTE,
                  "observation_rank": 3, "drive_scales": drive_scales,
                  "progress": progress}
    if max_nfev is not None:
        fit_kwargs["max_nfev"] = int(max_nfev)

    print("Pooled fit scope: all four A/B Stage1 records; metrics will be in-sample.",
          flush=True)
    for record in actual_records:
        record_cases = [case for case in eligible if case["dataset"] == record]
        counts = {case["condition"]: case["n_trials"] for case in record_cases}
        print(f"  {record}: {counts}", flush=True)
    fit = fit_model(eligible, **fit_kwargs)
    print("Shared fit complete:", fit.parameters, "gain=", fit.amplitude,
          "status=", fit.status, flush=True)

    predictions = _make_predictions(eligible, fit, drive_scales)
    metric_rows = _metric_rows(eligible, predictions)
    difference_rows = _difference_rows(eligible, predictions)
    summary = {
        "evaluation_scope": "IN_SAMPLE_TRAINING_FIT_NOT_VALIDATION_OR_TEST",
        "records": fit.train_records,
        "n_records": len(fit.train_records),
        "n_condition_erps": len(eligible),
        "n_trials_by_record_condition": {
            f"{case['dataset']}:{case['condition']}": case["n_trials"] for case in eligible},
        "parameters": fit.parameters,
        "shared_signed_gain": fit.amplitude,
        "loss": fit.loss,
        "fit_status": fit.status,
        "optimizer_success": fit.success,
        "boundary_flags": fit.boundary_flags,
        "n_objective_evaluations": fit.n_evaluations,
        "resolution": fit.resolution,
        "feature_route": fit.feature_route,
        "observation_rank": fit.observation_rank,
        "metrics_are_in_sample": True,
        "mean_condition_nrmse": float(np.nanmean([r["nrmse_by_observed_rms"]
                                                   for r in metric_rows])),
        "median_condition_nrmse": float(np.nanmedian([r["nrmse_by_observed_rms"]
                                                      for r in metric_rows])),
        "mean_condition_channel_correlation": float(np.nanmean(
            [r["mean_channel_correlation"] for r in metric_rows])),
        "mean_lr_difference_nrmse": float(np.nanmean(
            [r["difference_nrmse_by_observed_rms"] for r in difference_rows])),
        "mean_lr_difference_correlation": float(np.nanmean(
            [r["difference_waveform_correlation"] for r in difference_rows])),
        "event_audit": audit,
    }
    _write_json(OUTPUT_DIR / "联合拟合结果.json", {"fit": fit.to_dict(), "summary": summary,
                                                  "drive_scales": drive_scales})
    _write_csv(OUTPUT_DIR / "训练内拟合指标.csv", metric_rows)
    _write_csv(OUTPUT_DIR / "左右差异训练内拟合.csv", difference_rows)
    _plot_fit(eligible, predictions, OUTPUT_DIR / "全数据训练内拟合.png")
    _plot_differences(eligible, predictions, OUTPUT_DIR / "左右差异训练内拟合.png")
    _write_report(OUTPUT_DIR / "结果说明.md", fit, eligible, metric_rows, difference_rows)
    print(f"Saved pooled in-sample fit outputs to: {OUTPUT_DIR}", flush=True)
    print(f"Mean condition NRMSE={summary['mean_condition_nrmse']:.4f}; "
          f"mean channel correlation={summary['mean_condition_channel_correlation']:.4f}; "
          f"left-right difference correlation={summary['mean_lr_difference_correlation']:.4f}",
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=None,
                        help="optional optimizer evaluation cap per start")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    run(max_nfev=args.max_nfev, progress=not args.quiet_progress)


if __name__ == "__main__":
    main()
