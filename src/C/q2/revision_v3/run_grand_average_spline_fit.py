"""Fit the revision_v3 forward model to Q1's four-record grand-average ERP splines.

This is an in-sample descriptive fit. Each electrode is calibrated separately,
while left/right conditions share that electrode's parameters and gain. The
output is therefore a channelwise fit diagnostic, not an independent test or a
single globally shared physiological parameter estimate.
"""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from . import config
    from .fit import _cached_left_frontend
    from .frontend import load_stimulus, mirror_stage1_frontend
    from .model import ModelParams, calibrate_drive_scales, simulate_forward
    from .observation import model_curve_to_q1_grid
except ImportError:
    import config
    from fit import _cached_left_frontend
    from frontend import load_stimulus, mirror_stage1_frontend
    from model import ModelParams, calibrate_drive_scales, simulate_forward
    from observation import model_curve_to_q1_grid


Q1_CURVES = config.Q2_ROOT.parent / "q1" / "output" / "14erp_spline_fit" / "ERP样条拟合波形.csv"
OUT_DIR = config.Q2_ROOT / "output" / "revision_v3_grand_average_fit"
RECORDS = tuple(record for group in config.DATASETS.values() for record in group)
CONDITIONS = ("left", "right")
CONDITION_CN = {"left": "左提示", "right": "右提示"}
COLORS = {"F3": "#3569A8", "Fz": "#168B78", "F4": "#D77932"}
U = np.array([[1., 1., 1.], [-1., 0., 1.], [1., -2., 1.]])
U[0] /= np.sqrt(3.)
U[1] /= np.sqrt(2.)
U[2] /= np.sqrt(6.)
# The project has selected the sign convention that reverses the lateral and
# shape sensor modes. Apply that same fixed convention to every model output.
SENSOR_REFLECTION = U.T @ np.diag([1., -1., -1.]) @ U


def _setup_plotting():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": .22,
        "savefig.facecolor": "white",
    })


def _load_macro_targets():
    if not Q1_CURVES.exists():
        raise FileNotFoundError(f"Q1 spline waveform file not found: {Q1_CURVES}")
    table = pd.read_csv(Q1_CURVES, encoding="utf-8-sig")
    required = {"Dataset", "Condition", "Channel", "TimeFromCue", "ObservedERP",
                "SplineFit", "SplineFitValid"}
    if not required.issubset(table.columns):
        raise ValueError(f"Q1 spline file is missing columns: {sorted(required - set(table.columns))}")

    condition_map = {"左条件": "left", "右条件": "right", "左提示": "left", "右提示": "right"}
    table = table[table["Dataset"].isin(RECORDS)].copy()
    table["condition"] = table["Condition"].map(condition_map)
    table = table[table["condition"].isin(CONDITIONS)]
    if set(table["Dataset"].unique()) != set(RECORDS):
        raise ValueError("Q1 spline file does not contain all four expected records")
    table["TimeMs"] = table["TimeFromCue"].astype(float) * 1000.0
    table = table[table["SplineFitValid"].astype(int) == 1]

    time_grids = []
    for (record, condition, channel), block in table.groupby(
            ["Dataset", "condition", "Channel"], sort=False):
        block = block.sort_values("TimeMs")
        time = block["TimeMs"].to_numpy(float)
        if time.size == 0 or not np.all(np.diff(time) > 0):
            raise ValueError(f"Invalid time grid in {record}/{condition}/{channel}")
        time_grids.append(time)
    reference_time = time_grids[0]
    if any(not np.array_equal(reference_time, time) for time in time_grids[1:]):
        raise ValueError("Q1 curves have inconsistent time grids")

    time_mask = (reference_time >= -1000.0) & (reference_time <= 800.0 + 1e-8)
    time_ms = reference_time[time_mask]
    spline_by_record = {}
    observed_by_record = {}
    trials_by_record = {}
    for record in RECORDS:
        for condition in CONDITIONS:
            for channel in config.CHANNELS:
                block = table[(table["Dataset"] == record)
                              & (table["condition"] == condition)
                              & (table["Channel"] == channel)].sort_values("TimeMs")
                if len(block) != len(reference_time):
                    raise ValueError(f"Missing spline curve: {record}/{condition}/{channel}")
                spline = block["SplineFit"].to_numpy(float)[time_mask]
                observed = block["ObservedERP"].to_numpy(float)[time_mask]
                if not np.isfinite(spline).all() or not np.isfinite(observed).all():
                    raise ValueError(f"Non-finite Q1 curve: {record}/{condition}/{channel}")
                key = (record, condition, channel)
                spline_by_record[key] = spline
                observed_by_record[key] = observed
                trials_by_record[key] = int(block["N_trials"].iloc[0])

    # Equal weighting across records avoids letting the largest MAT dominate.
    macro_spline = {}
    macro_observed = {}
    for condition in CONDITIONS:
        for channel in config.CHANNELS:
            macro_spline[(condition, channel)] = np.mean(
                [spline_by_record[(record, condition, channel)] for record in RECORDS], axis=0)
            macro_observed[(condition, channel)] = np.mean(
                [observed_by_record[(record, condition, channel)] for record in RECORDS], axis=0)
    return time_ms, macro_spline, macro_observed, spline_by_record, observed_by_record, trials_by_record


def _make_reflected_predictions(fronts, params, amplitude, drive_scales):
    predictions = {}
    for condition in CONDITIONS:
        result = simulate_forward(fronts[condition], params=params, amplitude=1.0,
                                  feature_route=config.FRONTEND_ROUTE,
                                  drive_scales=drive_scales)
        observed, observed_time = model_curve_to_q1_grid(result.eeg, result.time_ms)
        reflected = SENSOR_REFLECTION @ observed
        predictions[condition] = (float(amplitude) * reflected, observed_time)
    return predictions


def _interpolate_channels(curve, source_time, target_time):
    return np.vstack([np.interp(target_time, source_time, row) for row in curve])


def _score_pair(targets, predictions, time_ms, channel_index):
    channel = config.CHANNELS[channel_index]
    actual = np.concatenate([targets[(condition, channel)] for condition in CONDITIONS])
    pred = np.concatenate([predictions[condition][channel_index] for condition in CONDITIONS])
    denominator = float(np.dot(pred, pred))
    gain = float(np.dot(actual, pred) / denominator) if denominator > 1e-20 else 0.0
    residual = actual - gain * pred
    return float(np.mean(residual ** 2)), gain


def _fit_channel(channel, channel_index, targets, time_ms, fronts, drive_scales,
                 tau_a=80.0, max_nfev=38):
    """Fit one channel's entire left/right curve pair with a shared channel gain."""
    cache = {}

    def evaluate(x):
        tau_s = float(np.clip(x[0], *config.PARAM_BOUNDS["tau_s"]))
        g_i = float(np.clip(x[1], *config.PARAM_BOUNDS["g_i"]))
        key = (round(tau_s, 6), round(g_i, 8))
        if key not in cache:
            model_curves = _make_reflected_predictions(
                fronts, ModelParams(tau_s=tau_s, g_i=g_i, tau_a=tau_a), 1.0, drive_scales)
            pred = {condition: _interpolate_channels(curve, model_time, time_ms)
                    for condition, (curve, model_time) in model_curves.items()}
            loss, gain = _score_pair(targets, pred, time_ms, channel_index)
            cache[key] = (loss, gain, pred)
        return cache[key]

    starts = (np.array([40.0, 1.0]), np.array([80.0, 0.8]), np.array([95.0, 1.3]))
    bounds = (config.PARAM_BOUNDS["tau_s"], config.PARAM_BOUNDS["g_i"])
    results = []
    for start in starts:
        result = minimize(lambda x: evaluate(x)[0], start, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxfun": int(max_nfev), "maxiter": int(max_nfev),
                                    "ftol": 1e-9, "maxls": 12,
                                    "eps": np.array([0.5, 0.01])})
        loss, gain, predictions = evaluate(result.x)
        results.append((loss, gain, result, predictions))
    best_loss, best_gain, best_result, best_predictions = min(results, key=lambda item: item[0])
    best_predictions = {condition: curve * best_gain
                        for condition, curve in best_predictions.items()}
    return {
        "channel": channel,
        "parameters": {"tau_s_ms": float(best_result.x[0]), "g_i": float(best_result.x[1]),
                       "tau_a_ms": float(tau_a)},
        "gain": float(best_gain),
        "loss_mse": float(best_loss),
        "success": bool(best_result.success),
        "status": int(best_result.status),
        "message": str(best_result.message),
        "evaluations": int(sum(int(item[2].nfev) for item in results)),
        "n_unique_simulations": len(cache),
        "predictions": best_predictions,
    }


def _curve_metrics(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = target - prediction
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    correlation = float(np.corrcoef(target, prediction)[0, 1]) if (
        np.std(target) > 1e-12 and np.std(prediction) > 1e-12) else float("nan")
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"rmse": rmse, "nrmse_by_target_rms": rmse / target_rms if target_rms else float("nan"),
            "correlation": correlation, "target_rms": target_rms,
            "prediction_rms": float(np.sqrt(np.mean(prediction ** 2)))}


def _fit_old_shared_baseline(time_ms, targets, fronts, drive_scales):
    old_path = config.Q2_ROOT / "output" / "revision_v3_pooled_training_fit" / "联合拟合结果.json"
    if not old_path.exists():
        return None
    old = json.loads(old_path.read_text(encoding="utf-8"))["summary"]
    params = ModelParams.from_any(old["parameters"])
    model = _make_reflected_predictions(fronts, params, old["shared_signed_gain"], drive_scales)
    pred = {condition: _interpolate_channels(curve, model_time, time_ms)
            for condition, (curve, model_time) in model.items()}
    return {"parameters": old["parameters"], "gain": float(old["shared_signed_gain"]),
            "predictions": pred}


def _write_plot(time_ms, spline, observed, predictions, metrics, path):
    _setup_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True)
    for row, condition in enumerate(CONDITIONS):
        for col, channel in enumerate(config.CHANNELS):
            ax = axes[row, col]
            color = COLORS[channel]
            ax.plot(time_ms, observed[(condition, channel)], color="#A0A0A0", lw=1.0,
                    alpha=.75, label="四记录等权实测均值")
            ax.plot(time_ms, spline[(condition, channel)], color=color, lw=2.0,
                    label="第一问样条拟合均值")
            ax.plot(time_ms, predictions[condition][col], color="#C83E4D", lw=1.6,
                    ls="--", label="模型拟合")
            ax.axvline(0, color="0.4", lw=.8)
            ax.axvline(200, color="0.5", lw=.8, ls=":")
            ax.axhline(0, color="0.6", lw=.7)
            m = metrics[(condition, channel)]
            ax.set_title(f"{CONDITION_CN[condition]} · {channel}  (NRMSE={m['nrmse_by_target_rms']:.2f}, r={m['correlation']:.2f})")
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("电位（原始数据单位）")
            if row == 1:
                ax.set_xlabel("提示出现后时间（ms）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True,
               bbox_to_anchor=(.5, 1.01))
    fig.suptitle("第一问四记录等权 ERP 样条曲线与逐电极模型拟合", y=1.065,
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_difference_plot(time_ms, spline, predictions, path):
    _setup_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), sharex=True)
    for i, (channel, ax) in enumerate(zip(config.CHANNELS, axes)):
        target_diff = spline[("right", channel)] - spline[("left", channel)]
        model_diff = predictions["right"][i] - predictions["left"][i]
        ax.plot(time_ms, target_diff, color=COLORS[channel], lw=1.9, label="第一问拟合曲线：右减左")
        ax.plot(time_ms, model_diff, color="#C83E4D", lw=1.6, ls="--", label="模型输出：右减左")
        ax.axvline(0, color="0.4", lw=.8)
        ax.axvline(200, color="0.5", lw=.8, ls=":")
        ax.axhline(0, color="0.55", lw=.7)
        ax.set_title(channel)
        ax.set_xlabel("提示出现后时间（ms）")
        ax.grid(alpha=.2)
        if i == 0:
            ax.set_ylabel("右减左电位（原始数据单位）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True,
               bbox_to_anchor=(.5, 1.04))
    fig.suptitle("左右 ERP 差异波：第一问拟合曲线与逐电极模型", y=1.13,
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(max_nfev=38, reuse_fit=False):
    config.require_current_source_mapping_manifest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    time_ms, target_spline, target_observed, _, _, trial_counts = _load_macro_targets()
    fit_mask = (time_ms >= config.FIT_WINDOW_MS[0]) & (time_ms <= config.FIT_WINDOW_MS[1])
    time_ms = time_ms[fit_mask]
    target_spline = {key: value[fit_mask] for key, value in target_spline.items()}
    target_observed = {key: value[fit_mask] for key, value in target_observed.items()}

    tau_a = float(config.PARAM_DEFAULTS["tau_a"])
    left_front = _cached_left_frontend(tau_a, 64)
    fronts = {"left": left_front,
              "right": mirror_stage1_frontend(left_front, load_stimulus("Stage1", "right"))}
    drive_scales = calibrate_drive_scales(fronts, feature_route=config.FRONTEND_ROUTE)

    print("拟合目标：第一问四记录等权 Stage1 ERP 样条均值；范围 0–800 ms。", flush=True)
    print("模型：每个电极单独拟合 tau_s、g_i 和共享左右增益；tau_a 固定为 80 ms。", flush=True)
    fitted = {}
    predictions = {condition: np.zeros((3, len(time_ms)), dtype=float) for condition in CONDITIONS}
    parameter_rows = []
    saved_parameters = None
    if reuse_fit:
        parameter_path = OUT_DIR / "分电极拟合参数.csv"
        if not parameter_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing fit parameters: {parameter_path}")
        saved_parameters = pd.read_csv(parameter_path, encoding="utf-8-sig").set_index("电极")
    for channel_index, channel in enumerate(config.CHANNELS):
        if reuse_fit:
            row = saved_parameters.loc[channel]
            channel_params = {"tau_s": float(row["tau_s_ms"]),
                              "g_i": float(row["g_i"]), "tau_a": float(row["tau_a_ms"])}
            channel_predictions = _make_reflected_predictions(
                fronts, ModelParams.from_any(channel_params), float(row["左右共享增益"]), drive_scales)
            pred = {condition: _interpolate_channels(curve, model_time, time_ms)
                    for condition, (curve, model_time) in channel_predictions.items()}
            fit = {"channel": channel, "parameters": {"tau_s_ms": channel_params["tau_s"],
                    "g_i": channel_params["g_i"], "tau_a_ms": channel_params["tau_a"]},
                    "gain": float(row["左右共享增益"]), "success": bool(row["优化收敛"]),
                    "evaluations": int(row["函数评估次数"]),
                    "n_unique_simulations": int(row["唯一参数点数"]), "predictions": pred}
        else:
            fit = _fit_channel(channel, channel_index, target_spline, time_ms,
                               fronts, drive_scales, tau_a=tau_a, max_nfev=max_nfev)
        fitted[channel] = fit
        for condition in CONDITIONS:
            predictions[condition][channel_index] = fit["predictions"][condition][channel_index]
        parameter_rows.append({"电极": channel, "tau_s_ms": fit["parameters"]["tau_s_ms"],
                               "g_i": fit["parameters"]["g_i"], "tau_a_ms": tau_a,
                               "左右共享增益": fit["gain"], "优化收敛": fit["success"],
                               "函数评估次数": fit["evaluations"],
                               "唯一参数点数": fit["n_unique_simulations"]})
        print(f"  {channel}: tau_s={fit['parameters']['tau_s_ms']:.2f} ms, "
              f"g_i={fit['parameters']['g_i']:.3f}, gain={fit['gain']:.5g}, "
              f"success={fit['success']}", flush=True)

    metrics = {}
    metric_rows = []
    for condition in CONDITIONS:
        for channel_index, channel in enumerate(config.CHANNELS):
            key = (condition, channel)
            metric = _curve_metrics(target_spline[key], predictions[condition][channel_index])
            metric["against_observed_mean"] = _curve_metrics(
                target_observed[key], predictions[condition][channel_index])
            metrics[key] = metric
            metric_rows.append({"评估对象": f"{CONDITION_CN[condition]}_{channel}",
                                "NRMSE_相对第一问样条均值RMS": metric["nrmse_by_target_rms"],
                                "相关_第一问样条均值": metric["correlation"],
                                "RMSE_原始数据单位": metric["rmse"],
                                "NRMSE_相对实测均值RMS": metric["against_observed_mean"]["nrmse_by_target_rms"],
                                "相关_实测均值": metric["against_observed_mean"]["correlation"]})
    all_target = np.concatenate([target_spline[(condition, channel)]
                                 for condition in CONDITIONS for channel in config.CHANNELS])
    all_pred = np.concatenate([predictions[condition][i]
                               for condition in CONDITIONS for i in range(3)])
    overall = _curve_metrics(all_target, all_pred)
    differences = {}
    for channel_index, channel in enumerate(config.CHANNELS):
        target = target_spline[("right", channel)] - target_spline[("left", channel)]
        prediction = predictions["right"][channel_index] - predictions["left"][channel_index]
        differences[channel] = _curve_metrics(target, prediction)
    old_shared = _fit_old_shared_baseline(time_ms, target_spline, fronts, drive_scales)
    old_metrics = None
    if old_shared is not None:
        old_metrics = _curve_metrics(
            np.concatenate([target_spline[(condition, channel)]
                            for condition in CONDITIONS for channel in config.CHANNELS]),
            np.concatenate([old_shared["predictions"][condition][i]
                            for condition in CONDITIONS for i in range(3)]))

    _write_plot(time_ms, target_spline, target_observed, predictions, metrics,
                OUT_DIR / "总体平均曲线拟合.png")
    _write_difference_plot(time_ms, target_spline, predictions,
                           OUT_DIR / "左右差异波拟合.png")
    _write_csv(OUT_DIR / "分电极拟合参数.csv", parameter_rows)
    _write_csv(OUT_DIR / "总体拟合指标.csv", metric_rows)

    curve_rows = []
    for index, t in enumerate(time_ms):
        for condition in CONDITIONS:
            for channel_index, channel in enumerate(config.CHANNELS):
                curve_rows.append({"时间_ms": t, "条件": CONDITION_CN[condition], "电极": channel,
                                   "第一问样条均值": target_spline[(condition, channel)][index],
                                   "实测ERP均值": target_observed[(condition, channel)][index],
                                   "模型拟合": predictions[condition][channel_index, index]})
    _write_csv(OUT_DIR / "曲线数据.csv", curve_rows)

    mean_channel_corr = float(np.mean([row["相关_第一问样条均值"] for row in metric_rows]))
    mean_diff_corr = float(np.mean([differences[channel]["correlation"] for channel in config.CHANNELS]))
    lines = [
        "# 第一问样条 ERP 的逐电极总体曲线拟合",
        "",
        "> 口径：四份记录先各自形成条件平均 ERP，再对记录等权平均；所有这些记录同时用于参数拟合与效果计算。这是总体曲线的训练内贴合，不是留出预测或独立验证。",
        "",
        "## 实现方式",
        "",
        "- 目标曲线：第一问 `14erp_spline_fit/ERP样条拟合波形.csv` 中四份记录的左、右条件 F3/Fz/F4 P 样条曲线。每份记录等权，不按试次数加权。",
        "- 拟合区间：提示后 0–800 ms。模型输出先经过 revision_v3 的完整 epoch 观测流程，再进入拟合区间。",
        "- 分电极：F3、Fz、F4 分别估计 `tau_s`、`g_i` 和一个左右共用的有符号增益；每个电极内左右提示共用参数。`tau_a` 固定为 80 ms，LGN/前端和导联映射固定。",
        "- 电极模式符号按当前采用的左右/形状模式约定转换；左右输入使用同一固定的前端驱动尺度。",
        "- 原始条件均值由第一问样条文件中的 `ObservedERP` 列按记录等权平均得到；用于检查平滑目标附近的实测走势。",
        "",
        "## 效果",
        "",
        f"- 六条条件×电极曲线合并的样条目标 NRMSE：**{overall['nrmse_by_target_rms']:.3f}**；相关：**{overall['correlation']:.3f}**。",
        f"- 六条曲线平均相关：**{mean_channel_corr:.3f}**。",
        f"- 左右差异波平均相关：**{mean_diff_corr:.3f}**。",
    ]
    if old_metrics is not None:
        lines += [f"- 既有共享参数模型（`tau_a=80 ms`）在同一六条目标曲线上的 NRMSE：**{old_metrics['nrmse_by_target_rms']:.3f}**，相关：**{old_metrics['correlation']:.3f}**。",
                  f"- 逐电极拟合相对该固定共享参数输出的 NRMSE 变化：**{overall['nrmse_by_target_rms'] - old_metrics['nrmse_by_target_rms']:+.3f}**（负值代表误差下降）。"]
    lines += [
        "",
        "## 解读边界",
        "",
        "逐电极参数拟合用于回答‘每条电极曲线能否被该模型形状贴近’，不是一个具有单一共享动力学参数集的严格正向解。若逐电极拟合明显优于共享参数模型，只能说明电极间存在当前共享模型未刻画的波形/增益差异；不能把三套参数并称为同一套生理参数。拟合高低也不能证明模型机制正确，后续若要主张跨试次或跨记录预测，仍需独立留出。",
        "",
        "相关与 NRMSE 分电极数值见 `总体拟合指标.csv`，参数见 `分电极拟合参数.csv`，可复画曲线数据见 `曲线数据.csv`。",
        "",
    ]
    (OUT_DIR / "结果说明.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"总目标 NRMSE={overall['nrmse_by_target_rms']:.4f}; "
          f"总相关={overall['correlation']:.4f}; "
          f"平均电极曲线相关={mean_channel_corr:.4f}; "
          f"左右差异平均相关={mean_diff_corr:.4f}", flush=True)
    print(f"输出目录：{OUT_DIR}", flush=True)
    return {"overall": overall, "mean_channel_correlation": mean_channel_corr,
            "mean_difference_correlation": mean_diff_corr, "old_shared": old_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-fit", action="store_true",
                        help="regenerate metrics and figures from saved per-channel parameters")
    parser.add_argument("--max-nfev", type=int, default=38)
    args = parser.parse_args()
    run(max_nfev=args.max_nfev, reuse_fit=args.reuse_fit)
