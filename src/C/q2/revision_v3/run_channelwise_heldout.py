"""Leave-one-record comparison with independent F3/Fz/F4 response fits.

This is a diagnostic model variant. Each sensor gets its own cortical response
parameters, while the left/right conditions share that sensor's parameters.
The shared visual-adaptation time constant is taken from the training-only
joint fit for the same fold. No held-out ERP is used for parameter fitting.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

try:
    from . import config
    from .frontend import load_stimulus, mirror_stage1_frontend, simulate_frontend
    from .model import ModelParams, simulate_forward
    from .observation import model_curve_to_q1_grid
    from .real_data import load_cases
except ImportError:
    import config
    from frontend import load_stimulus, mirror_stage1_frontend, simulate_frontend
    from model import ModelParams, simulate_forward
    from observation import model_curve_to_q1_grid
    from real_data import load_cases


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "output" / "revision_v3"
OUT = ROOT / "output" / "revision_v3_channelwise"
CHANNELS = tuple(config.CHANNELS)
CONDITIONS = ("left", "right")
MAX_EVALUATIONS = 32
RECORD_LABELS = {
    "VisualCogA_Task-1": "A组·任务1",
    "VisualCogA_Task-2": "A组·任务2",
    "VisualCogB_Task-1": "B组·任务1",
    "VisualCogB_Task-2": "B组·任务2",
}
COLORS = {"left": "#3265a8", "right": "#d36f3d"}


def write_csv(path: Path, rows: list[dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def interpolate_rows(values: np.ndarray, source_time: np.ndarray,
                     target_time: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(target_time, source_time, row) for row in values])


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def curve_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.asarray(predicted) - np.asarray(actual)
    actual_rms = float(np.sqrt(np.mean(np.asarray(actual) ** 2)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {
        "rmse": rmse,
        "nrmse": rmse / actual_rms if actual_rms > 1e-12 else float("nan"),
        "correlation": correlation(np.asarray(actual), np.asarray(predicted)),
    }


def load_baseline_predictions(path: Path) -> dict:
    grouped = {}
    for row in read_csv(path):
        key = (row["record"], row["condition"])
        grouped.setdefault(key, {"time": [], "measured": [], "model": []})
        grouped[key]["time"].append(float(row["time_ms"]))
        grouped[key]["measured"].append((row["channel"], float(row["measured"])))
        grouped[key]["model"].append((row["channel"], float(row["model_heldout"])))
    output = {}
    for key, values in grouped.items():
        times = np.asarray(values["time"], dtype=float)
        unique_time = np.unique(times)
        measured = np.zeros((3, unique_time.size), dtype=float)
        model = np.zeros_like(measured)
        for j, t in enumerate(unique_time):
            at_time = [i for i, value in enumerate(times) if value == t]
            for i in at_time:
                channel = values["measured"][i][0]
                channel_i = CHANNELS.index(channel)
                measured[channel_i, j] = values["measured"][i][1]
                model[channel_i, j] = values["model"][i][1]
        output[key] = {"time": unique_time, "measured": measured, "model": model}
    return output


def load_drive_scales(path: Path) -> np.ndarray:
    rows = read_csv(path)
    values = np.asarray([float(row["reference_rms_scale"]) for row in rows], dtype=float)
    if values.size != 6:
        raise ValueError(f"expected six saved drive scales, found {values.size}")
    return values.reshape(3, 2)


def make_fronts(tau_a: float, resolution: int, stride_ms: float,
                left_stimulus, right_stimulus) -> dict:
    left = simulate_frontend(left_stimulus, params={"tau_a": float(tau_a)},
                             resolution=resolution, time_ms=config.TIME_MS,
                             feature_stride_ms=stride_ms)
    right = mirror_stage1_frontend(left, right_stimulus)
    return {"left": left, "right": right}


def model_curve(front: dict, condition: str, params: ModelParams,
                drive_scales: np.ndarray, channel_i: int,
                target_time: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    result = simulate_forward(front[condition], params=params,
                              time_ms=front[condition]["time_ms"],
                              drive_scales=drive_scales)
    processed, processed_time = model_curve_to_q1_grid(result.eeg, result.time_ms)
    processed = interpolate_rows(processed, processed_time, target_time)
    return float(amplitude) * processed[channel_i]


def fit_one_channel(train_cases: list[dict], channel_i: int, tau_a: float,
                    shared_start: dict, fronts: dict,
                    drive_scales: np.ndarray) -> dict:
    """Fit tau_s, g_i, and one gain for one sensor across both cue sides."""
    n_records = len({case["dataset"] for case in train_cases})
    weights = {case["dataset"]: 1.0 / (2.0 * n_records) for case in train_cases}
    lower = np.asarray([config.PARAM_BOUNDS["tau_s"][0],
                        config.PARAM_BOUNDS["g_i"][0]], dtype=float)
    upper = np.asarray([config.PARAM_BOUNDS["tau_s"][1],
                        config.PARAM_BOUNDS["g_i"][1]], dtype=float)
    def evaluate(x):
        params = ModelParams(float(x[0]), float(x[1]), float(tau_a))
        predicted_by_condition = {}
        for condition in CONDITIONS:
            target_time = next(case["time_ms"] for case in train_cases
                               if case["condition"] == condition)
            predicted_by_condition[condition] = model_curve(
                fronts, condition, params, drive_scales, channel_i, target_time)
        numerator = 0.0
        denominator = 0.0
        for case in train_cases:
            prediction = predicted_by_condition[case["condition"]]
            target = np.asarray(case["real"][channel_i], dtype=float)
            weight = weights[case["dataset"]]
            numerator += weight * float(np.mean(prediction * target))
            denominator += weight * float(np.mean(prediction * prediction))
        if denominator <= 1e-18 or not np.isfinite(denominator):
            return float("inf"), 0.0
        gain = numerator / denominator
        loss = 0.0
        for case in train_cases:
            prediction = predicted_by_condition[case["condition"]]
            target = np.asarray(case["real"][channel_i], dtype=float)
            loss += weights[case["dataset"]] * float(
                np.mean((target - gain * prediction) ** 2))
        return float(loss), float(gain)

    best = {"loss": float("inf"), "gain": 0.0, "x": None}
    evaluations = 0

    def objective(x):
        nonlocal evaluations
        evaluations += 1
        try:
            loss, gain = evaluate(x)
        except (FloatingPointError, ValueError, OverflowError):
            loss, gain = float("inf"), 0.0
        if np.isfinite(loss) and loss < best["loss"]:
            best.update({"loss": loss, "gain": gain,
                         "x": np.asarray(x, dtype=float).copy()})
        return loss if np.isfinite(loss) else 1e30

    start = np.asarray([shared_start["tau_s"], shared_start["g_i"]], dtype=float)
    start = np.clip(start, lower, upper)
    start_loss, start_gain = evaluate(start)
    best.update({"loss": start_loss, "gain": start_gain, "x": start.copy()})
    opt = minimize(objective, start, method="L-BFGS-B", bounds=list(zip(lower, upper)),
                   options={"maxfun": MAX_EVALUATIONS, "maxiter": MAX_EVALUATIONS,
                            "ftol": 1e-8, "maxls": 12,
                            "eps": config.FIT_FINITE_DIFF_STEPS.copy()})
    xbest = best["x"] if best["x"] is not None else start
    return {
        "parameters": {"tau_s": float(xbest[0]), "g_i": float(xbest[1]),
                       "tau_a": float(tau_a)},
        "gain": float(best["gain"]), "training_loss": float(best["loss"]),
        "optimizer_success": bool(opt.success), "optimizer_message": str(opt.message),
        "objective_evaluations": int(evaluations),
    }


def fit_training_gains(train_cases: list[dict], shared_params: dict,
                       fronts: dict, drive_scales: np.ndarray) -> np.ndarray:
    """Fit one gain per sensor while holding shared dynamics fixed."""
    n_records = len({case["dataset"] for case in train_cases})
    gains = np.zeros(3, dtype=float)
    for channel_i in range(3):
        numerator = denominator = 0.0
        for case in train_cases:
            raw = model_curve(fronts, case["condition"],
                              ModelParams.from_any(shared_params), drive_scales,
                              channel_i, case["time_ms"])
            target = np.asarray(case["real"][channel_i], dtype=float)
            weight = 1.0 / (2.0 * n_records)
            numerator += weight * float(np.mean(raw * target))
            denominator += weight * float(np.mean(raw * raw))
        gains[channel_i] = numerator / denominator if denominator > 1e-18 else 0.0
    return gains


def plot_predictions(rows: list[dict], predictions: dict, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
    })
    records = list(RECORD_LABELS)
    fig, axes = plt.subplots(4, 3, figsize=(15.2, 10.4), sharex=True)
    legend_handles = None
    for ri, record in enumerate(records):
        for ci, channel in enumerate(CHANNELS):
            ax = axes[ri, ci]
            for condition in CONDITIONS:
                curve = predictions[(record, condition)]
                t = curve["time"]
                color = COLORS[condition]
                label = "左向提示" if condition == "left" else "右向提示"
                ax.plot(t, curve["measured"][ci], color=color, lw=1.35,
                        linestyle="-", label=f"实测{label}")
                ax.plot(t, curve["shared"][ci], color=color, lw=1.0,
                        linestyle="--", alpha=.72, label=f"共享模型{label}")
                ax.plot(t, curve["channelwise"][ci], color=color, lw=1.35,
                        linestyle=":", label=f"逐电极拟合{label}")
            ax.axhline(0, color="0.6", lw=.6)
            ax.axvline(200, color="0.55", lw=.7, linestyle="--")
            ax.grid(alpha=.2, linewidth=.55)
            ax.set_title(f"{RECORD_LABELS[record]} · {channel}", fontsize=9.5)
            if ci == 0:
                ax.set_ylabel("脑电幅值（原始数据单位）")
            if ri == 3:
                ax.set_xlabel("提示出现后时间（ms）")
            if ri == 0 and ci == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=6,
               frameon=True, bbox_to_anchor=(.5, .966), fontsize=8.2,
               columnspacing=1.15, handlelength=1.5)
    fig.suptitle("留一记录：逐电极 ERP 拟合与留出预测", y=.996, fontsize=14)
    fig.text(.5, .927, "实线：实测 ERP　虚线：共享模型　点线：逐电极拟合模型",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.015, 0.015, .99, .9), h_pad=1.2, w_pad=.8)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def refresh_outputs_from_saved() -> None:
    """Rebuild the figure and report from saved predictions without refitting."""
    csv_files = [(path, read_csv(path)) for path in OUT.glob("*.csv")]
    prediction_rows = next(rows for _, rows in csv_files if len(rows) == 2472)
    metric_rows = next(rows for _, rows in csv_files if len(rows) == 24)
    parameter_rows = next(rows for _, rows in csv_files if len(rows) == 12)
    predictions = {}
    for row in prediction_rows:
        values = list(row.values())
        record, condition = values[0], values[1]
        key = (record, condition)
        entry = predictions.setdefault(key, {"rows": []})
        entry["rows"].append(values)
    for entry in predictions.values():
        entry["rows"].sort(key=lambda values: (float(values[2]), CHANNELS.index(values[3])))
        times = sorted({float(values[2]) for values in entry["rows"]})
        time_index = {time: index for index, time in enumerate(times)}
        arrays = {name: np.zeros((3, len(times)), dtype=float)
                  for name in ("measured", "shared", "gain_only", "channelwise")}
        for values in entry["rows"]:
            channel_i = CHANNELS.index(values[3])
            ti = time_index[float(values[2])]
            for name, vi in (("measured", 4), ("shared", 5),
                             ("gain_only", 6), ("channelwise", 7)):
                arrays[name][channel_i, ti] = float(values[vi])
        entry.update({"time": np.asarray(times, dtype=float), **arrays})
        del entry["rows"]
    records = [record for record in RECORD_LABELS if any(key[0] == record for key in predictions)]
    plot_predictions(records, predictions, OUT / "逐电极拟合留出ERP.png")

    names = ("共享模型", "逐电极增益", "逐电极曲线拟合")
    prediction_keys = ("shared", "gain_only", "channelwise")
    metric_nrmse_columns = (4, 5, 6)
    metric_corr_columns = (7, 8, 9)
    summary = {}
    for mi, (name, prediction_key) in enumerate(zip(names, prediction_keys)):
        condition_scores, actual_all, predicted_all = [], [], []
        for data in predictions.values():
            actual = data["measured"]
            predicted = data[prediction_key]
            condition_scores.append(float(
                np.sqrt(np.mean((predicted - actual) ** 2)) /
                max(np.sqrt(np.mean(actual ** 2)), 1e-12)))
            actual_all.extend(actual.ravel())
            predicted_all.extend(predicted.ravel())
        actual_all = np.asarray(actual_all)
        predicted_all = np.asarray(predicted_all)
        summary[name] = {
            "record_condition_nrmse": float(np.mean(condition_scores)),
            "curve_macro_nrmse": float(np.mean([float(list(row.values())[metric_nrmse_columns[mi]])
                                                 for row in metric_rows])),
            "pooled_nrmse": float(np.sqrt(np.mean((predicted_all - actual_all) ** 2)) /
                                  max(np.sqrt(np.mean(actual_all ** 2)), 1e-12)),
            "curve_correlation": float(np.nanmean([float(list(row.values())[metric_corr_columns[mi]])
                                                   for row in metric_rows])),
        }

    difference_corr = {name: [] for name in names}
    difference_nrmse = {name: [] for name in names}
    inter_channel_corr = {name: [] for name in names}
    diff_model_index = 0
    sensor_model_index = 0
    for record in records:
        for channel_i in range(3):
            left, right = predictions[(record, "left")], predictions[(record, "right")]
            actual = right["measured"][channel_i] - left["measured"][channel_i]
            for name, prediction_key in zip(names, prediction_keys):
                predicted = right[prediction_key][channel_i] - left[prediction_key][channel_i]
                difference_corr[name].append(correlation(actual, predicted))
                difference_nrmse[name].append(curve_metrics(actual, predicted)["nrmse"])
                diff_model_index += 1
        for condition in CONDITIONS:
            data = predictions[(record, condition)]
            for name, prediction_key in zip(names, prediction_keys):
                corr = np.corrcoef(data[prediction_key])
                inter_channel_corr[name].extend((corr[0, 1], corr[0, 2], corr[1, 2]))
                sensor_model_index += 1

    per_channel = {}
    for channel in CHANNELS:
        rows = [row for row in metric_rows if list(row.values())[2] == channel]
        per_channel[channel] = []
        for mi in range(3):
            nrmse = float(np.mean([float(list(row.values())[metric_nrmse_columns[mi]])
                                   for row in rows]))
            corr = float(np.nanmean([float(list(row.values())[metric_corr_columns[mi]])
                                     for row in rows]))
            per_channel[channel].append((nrmse, corr))
    converged = sum(str(list(row.values())[8]).lower() == "true" for row in parameter_rows)
    max_evaluations = max(int(float(list(row.values())[9])) for row in parameter_rows)

    report = [
        "# 逐电极拟合的留一记录验证", "",
        "## 拟合与验证口径", "",
        "- 共 4 折，每折留出 1 份记录，其余 3 份用于拟合。训练目标为记录内左右条件 ERP 叠加平均，各记录与方向等权。",
        "- 每个电极单独拟合 `tau_s`、`g_i` 和增益；同一电极的左右条件共享参数。`tau_a` 固定为该折共享模型仅使用训练记录选出的值。",
        "- 留出 ERP 只用于评分。模拟响应延长到 3 s，再按原流程滤波、重采样、基线校正并截取 0–800 ms。没有事后反号。", "",
        "## 留出预测结果", "",
        "主指标为每个留出记录×提示方向先合并三电极波形计算 NRMSE，再对 8 个条件等权平均；这一口径可与之前的共享模型 NRMSE=1.810 比较。",
        "| 模型 | 记录×方向宏平均 NRMSE | 电极×方向曲线 NRMSE 均值 | 全部样本合并 NRMSE | 单曲线相关均值 |",
        "|---|---:|---:|---:|---:|"]
    for name in names:
        values = summary[name]
        report.append(f"| {name} | {values['record_condition_nrmse']:.3f} | {values['curve_macro_nrmse']:.3f} | {values['pooled_nrmse']:.3f} | {values['curve_correlation']:.3f} |")
    report += ["", "## 分电极结果", "",
               "| 电极 | 共享模型 NRMSE / 相关 | 逐电极增益 NRMSE / 相关 | 逐电极拟合 NRMSE / 相关 |",
               "|---|---:|---:|---:|"]
    for channel in CHANNELS:
        cells = [f"{nrmse:.3f} / {corr:.3f}" for nrmse, corr in per_channel[channel]]
        report.append(f"| {channel} | " + " | ".join(cells) + " |")
    report += ["", "## 左右差异与通道相似度", "",
               "| 模型 | 留出右减左 NRMSE 均值 | 留出右减左相关均值 | 预测跨电极时序相关均值 |",
               "|---|---:|---:|---:|"]
    for name in names:
        report.append(f"| {name} | {np.nanmean(difference_nrmse[name]):.3f} | {np.nanmean(difference_corr[name]):.3f} | {np.mean(inter_channel_corr[name]):.4f} |")
    report += ["", f"逐电极拟合的 12 组优化中收敛状态为真者 {converged}/12，最多实际评估 {max_evaluations} 次目标函数。",
               "", "## 结论", "",
               f"逐电极拟合的记录×方向宏平均 NRMSE 为 {summary['逐电极曲线拟合']['record_condition_nrmse']:.3f}，共享模型为 {summary['共享模型']['record_condition_nrmse']:.3f}。左右差异相关也没有改善。本轮结果不支持“每个电极独立拟合动力学参数能提高跨记录预测”的判断；同时只有少数组合报告收敛，因此逐电极拟合的参数还不能作稳定估计。",
               "", "图中每行是留出记录，每列是电极；实线为实测 ERP，虚线为共享模型，点线为逐电极拟合模型。",
               "", "![逐电极拟合与留出 ERP](逐电极拟合留出ERP.png)"]
    (OUT / "逐电极拟合留出验证.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = config.require_current_source_mapping_manifest(BASELINE_DIR / "manifest.json")
    resolution = int(manifest["resolution"])
    stride_ms = float(manifest["frontend_feature_stride_ms"])
    drive_scales = load_drive_scales(BASELINE_DIR / "drive_scales.csv")
    with (BASELINE_DIR / "heldout_fit_details.json").open("r", encoding="utf-8") as stream:
        shared_fits = json.load(stream)
    baseline_predictions = load_baseline_predictions(BASELINE_DIR / "heldout_predictions.csv")
    cases = [case for case in load_cases()
             if case["stage"] == "Stage1" and case["eligible_fit"]]
    records = list(RECORD_LABELS)
    if set(shared_fits) != set(records):
        raise ValueError("shared baseline does not contain the four expected held-out folds")
    by_fold = {}
    for heldout in records:
        train = [case for case in cases if case["dataset"] != heldout]
        grouped = {condition: [case for case in train if case["condition"] == condition]
                   for condition in CONDITIONS}
        if any(len(grouped[condition]) != 3 for condition in CONDITIONS):
            raise ValueError(f"{heldout}: expected three training-record ERP averages per side")
        by_fold[heldout] = train

    frontend_cache = {}
    fitted = {}
    parameter_rows = []
    predictions = {}
    metric_rows = []
    diagnostics = []
    for fold_i, heldout in enumerate(records, start=1):
        start_time = perf_counter()
        base = shared_fits[heldout]
        shared_params = base["parameters"]
        tau_a = float(shared_params["tau_a"])
        if tau_a not in frontend_cache:
            print(f"构造共享前端，tau_a={tau_a:g} ms", flush=True)
            frontend_cache[tau_a] = make_fronts(
                tau_a, resolution, stride_ms,
                load_stimulus("Stage1", "left"), load_stimulus("Stage1", "right"))
        fronts = frontend_cache[tau_a]
        train = by_fold[heldout]
        gains_only = fit_training_gains(train, shared_params, fronts, drive_scales)
        fold_parameters = {}
        print(f"第 {fold_i}/4 折：留出 {heldout}", flush=True)
        for channel_i, channel in enumerate(CHANNELS):
            print(f"  拟合 {channel}", flush=True)
            result = fit_one_channel(train, channel_i, tau_a, shared_params,
                                     fronts, drive_scales)
            fold_parameters[channel] = result
            fitted[(heldout, channel)] = result
            parameter_rows.append({
                "留出记录": heldout, "训练记录": ";".join(sorted({c["dataset"] for c in train})),
                "电极": channel, "共享tau_a_ms": tau_a,
                "独立tau_s_ms": result["parameters"]["tau_s"],
                "独立g_i": result["parameters"]["g_i"],
                "独立增益": result["gain"], "训练损失": result["training_loss"],
                "优化器收敛": result["optimizer_success"],
                "目标函数评估次数": result["objective_evaluations"],
            })

        channel_fit_curves = {}
        gain_curves = {}
        for condition in CONDITIONS:
            case = next(case for case in cases
                        if case["dataset"] == heldout and case["condition"] == condition)
            channel_fit_curves[condition] = np.vstack([
                model_curve(fronts, condition,
                            ModelParams.from_any(fold_parameters[channel]["parameters"]),
                            drive_scales, channel_i, case["time_ms"],
                            fold_parameters[channel]["gain"])
                for channel_i, channel in enumerate(CHANNELS)
            ])
            # Generate all sensor rows once for shared and gain-only predictions.
            shared_result = simulate_forward(fronts[condition],
                params=ModelParams.from_any(shared_params), time_ms=fronts[condition]["time_ms"],
                drive_scales=drive_scales)
            processed, processed_time = model_curve_to_q1_grid(shared_result.eeg,
                                                                shared_result.time_ms)
            raw_matrix = interpolate_rows(processed, processed_time, case["time_ms"])
            gain_matrix = raw_matrix.copy()
            gain_matrix *= gains_only[:, None]
            saved_baseline = baseline_predictions[(heldout, condition)]
            if not np.allclose(saved_baseline["time"], case["time_ms"], rtol=0, atol=1e-8):
                shared_matrix = interpolate_rows(saved_baseline["model"],
                                                 saved_baseline["time"], case["time_ms"])
            else:
                shared_matrix = saved_baseline["model"]
            if not np.allclose(saved_baseline["measured"], case["real"],
                               rtol=1e-6, atol=1e-6):
                raise ValueError(f"{heldout}/{condition}: saved ERP differs from reloaded data")
            predictions[(heldout, condition)] = {
                "time": case["time_ms"].copy(), "measured": case["real"].copy(),
                "shared": shared_matrix, "gain_only": gain_matrix,
                "channelwise": channel_fit_curves[condition],
            }

            for channel_i, channel in enumerate(CHANNELS):
                actual = case["real"][channel_i]
                model_map = {
                    "共享模型": shared_matrix[channel_i],
                    "逐电极增益": gain_matrix[channel_i],
                    "逐电极曲线拟合": channel_fit_curves[condition][channel_i],
                }
                metrics = {name: curve_metrics(actual, curve)
                           for name, curve in model_map.items()}
                metric_rows.append({
                    "留出记录": heldout, "条件": condition, "电极": channel,
                    "试次数": int(case["n_trials"]),
                    **{f"{name}_NRMSE": value["nrmse"] for name, value in metrics.items()},
                    **{f"{name}_相关": value["correlation"] for name, value in metrics.items()},
                })

        for channel_i, channel in enumerate(CHANNELS):
            left = predictions[(heldout, "left")]
            right = predictions[(heldout, "right")]
            observed_diff = right["measured"][channel_i] - left["measured"][channel_i]
            for model_name, key in (("共享模型", "shared"),
                                    ("逐电极增益", "gain_only"),
                                    ("逐电极曲线拟合", "channelwise")):
                pred_diff = right[key][channel_i] - left[key][channel_i]
                result = curve_metrics(observed_diff, pred_diff)
                diagnostics.append({"留出记录": heldout, "电极": channel,
                                    "模型": model_name,
                                    "右减左_NRMSE": result["nrmse"],
                                    "右减左_相关": result["correlation"]})
        print(f"  本折完成，用时 {perf_counter() - start_time:.1f} 秒", flush=True)

    # Inter-electrode similarity diagnoses whether separate fits broke the
    # almost-collinear predicted waveforms observed in the shared model.
    for heldout in records:
        for condition in CONDITIONS:
            values = predictions[(heldout, condition)]
            for model_name, key in (("共享模型", "shared"),
                                    ("逐电极增益", "gain_only"),
                                    ("逐电极曲线拟合", "channelwise")):
                matrix = values[key]
                corr = np.corrcoef(matrix)
                diagnostics.append({
                    "留出记录": heldout, "电极": "三电极间",
                    "模型": f"{model_name}跨电极时序相关",
                    "F3_Fz": float(corr[0, 1]), "F3_F4": float(corr[0, 2]),
                    "Fz_F4": float(corr[1, 2]),
                })

    write_csv(OUT / "逐电极参数.csv", parameter_rows)
    write_csv(OUT / "留出曲线指标.csv", metric_rows)
    write_csv(OUT / "左右差异与通道相似度.csv", diagnostics)
    prediction_rows = []
    for (record, condition), values in predictions.items():
        for ti, time_ms in enumerate(values["time"]):
            for channel_i, channel in enumerate(CHANNELS):
                prediction_rows.append({
                    "记录": record, "条件": condition, "时间_ms": float(time_ms),
                    "电极": channel, "实测": float(values["measured"][channel_i, ti]),
                    "共享模型": float(values["shared"][channel_i, ti]),
                    "逐电极增益": float(values["gain_only"][channel_i, ti]),
                    "逐电极曲线拟合": float(values["channelwise"][channel_i, ti]),
                })
    write_csv(OUT / "留出预测曲线.csv", prediction_rows)
    plot_predictions(records, predictions, OUT / "逐电极拟合留出ERP.png")

    model_names = ("共享模型", "逐电极增益", "逐电极曲线拟合")
    summary = {}
    for model_name in model_names:
        nrmse_values = np.asarray([row[f"{model_name}_NRMSE"] for row in metric_rows], dtype=float)
        corr_values = np.asarray([row[f"{model_name}_相关"] for row in metric_rows], dtype=float)
        diff_rows = [row for row in diagnostics if row.get("模型") == model_name]
        summary[model_name] = {
            "平均逐曲线NRMSE": float(np.nanmean(nrmse_values)),
            "平均逐曲线相关": float(np.nanmean(corr_values)),
            "右减左平均相关": float(np.nanmean([r["右减左_相关"] for r in diff_rows])),
            "右减左平均NRMSE": float(np.nanmean([r["右减左_NRMSE"] for r in diff_rows])),
        }
    per_channel = {}
    for channel in CHANNELS:
        per_channel[channel] = {}
        for model_name in model_names:
            rows = [row for row in metric_rows if row["电极"] == channel]
            per_channel[channel][model_name] = {
                "平均NRMSE": float(np.mean([row[f"{model_name}_NRMSE"] for row in rows])),
                "平均相关": float(np.nanmean([row[f"{model_name}_相关"] for row in rows])),
            }

    report = [
        "# 逐电极拟合的留一记录验证",
        "",
        "## 实验目的",
        "",
        "当前共享模型在 F3/Fz/F4 上生成的预测曲线时序相关约为 0.993–0.999。本实验检验允许各电极独立拟合动力学参数后，能否在未参与拟合的记录上改善 ERP 预测。",
        "",
        "## 拟合与验证口径",
        "",
        "- 共 4 折，每折留出 1 份记录，另外 3 份记录用于拟合。",
        "- 每份记录先按左右提示分别叠加平均；训练目标为记录×条件×电极 ERP 均值，各训练记录等权。",
        "- 每个电极分别拟合 `tau_s`、`g_i` 与一个增益；同一电极的左右提示共用参数。`tau_a` 固定为该折共享模型仅用训练记录选出的值。",
        "- 留出记录的曲线只用于评分。模拟响应自然延长到完整 3 s，再按 revision_v3 的滤波、重采样与基线流程处理并截取 0–800 ms。",
        "- 同时报告原共享模型与仅增加逐电极增益的对照；未做事后反号。",
        "",
        "## 总体结果",
        "",
        "| 模型 | 平均逐曲线 NRMSE | 平均逐曲线相关 | 右减左平均相关 | 右减左平均 NRMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in model_names:
        row = summary[name]
        report.append(f"| {name} | {row['平均逐曲线NRMSE']:.3f} | {row['平均逐曲线相关']:.3f} | {row['右减左平均相关']:.3f} | {row['右减左平均NRMSE']:.3f} |")
    report += ["", "## 分电极结果", "",
               "| 电极 | 共享模型 NRMSE / 相关 | 逐电极增益 NRMSE / 相关 | 逐电极拟合 NRMSE / 相关 |",
               "|---|---:|---:|---:|"]
    for channel in CHANNELS:
        vals = per_channel[channel]
        cells = [f"{vals[name]['平均NRMSE']:.3f} / {vals[name]['平均相关']:.3f}"
                 for name in model_names]
        report.append(f"| {channel} | " + " | ".join(cells) + " |")
    report += [
        "",
        "## 解释边界",
        "",
        "逐电极拟合若改善留出表现，说明电极特异的曲线参数有助于跨记录预测；不代表三个电极各自具有一套独立的真实皮层动力学。该诊断变体把每个电极的参数分别用于完整级联后只取该电极输出，因此三路结果不能合并解释为同一组神经源的一次物理头皮投影。源定位仍应以共享动力学和有约束的观测映射模型为依据。",
        "",
        "图中每行是留出记录，每列是电极；实线为留出 ERP，虚线为共享模型，点线为逐电极拟合。指标和参数见同目录 CSV。",
        "",
        "![逐电极拟合与留出 ERP](逐电极拟合留出ERP.png)",
    ]
    (OUT / "逐电极拟合留出验证.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("总体指标:")
    for name, row in summary.items():
        print(name, row)
    print(f"结果目录：{OUT}", flush=True)


if __name__ == "__main__":
    import sys
    if "--refresh" in sys.argv:
        refresh_outputs_from_saved()
    else:
        main()
