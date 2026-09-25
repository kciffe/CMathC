"""Run full-context stimulus and chronological observation-map comparison.

This script does not overwrite the frozen revision_v3 outputs. It uses the
already fitted leave-one-record dynamics, builds a full Stage1 scene sequence,
and compares the legacy fixed sensor map with maps estimated on one time block
and evaluated on the other.
"""
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from frontend import (_simulate_lgn, load_stimulus, mirror_stage1_frontend,
                      resize_area, simulate_frontend)
from head_model import build_sensor_leadfield
from model import ModelParams, calibrate_drive_scales, observable_modes_from_real, simulate_forward
from observation import filter_resample_baseline
from real_data import load_dataset
from round1_validation import (build_scene_timeline, fit_effective_mapping,
                               fit_shared_gain, model_sources_to_q1_grid,
                               source_channel_labels)


OUT = config.Q2_ROOT / "output" / "revision_v3_round1"
FIT_JSON = config.OUTPUT_ROOT / "heldout_fit_details.json"
TIME_MS = np.arange(-1000.0, 3001.0, 1.0)
TARGET_ONSET_MS = 2200.0
RESPONSE_MS = (0.0, 796.875)
ALPHAS = (0.1, 1.0, 10.0)
PRIMARY_ALPHA = 1.0
TARGETS_BY_TASK = {
    "Task1": ("dots",),
    "Task2": ("inward", "outward"),
}

_CJK_FONT = Path("C:/Windows/Fonts/msyh.ttc")
if _CJK_FONT.exists():
    font_manager.fontManager.addfont(str(_CJK_FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_CJK_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scene_frontend(target_name, tau_a):
    cue = load_stimulus("Stage1", "left")
    target = load_stimulus("Stage2", target_name)
    if not np.array_equal(target.image, target.image[:, ::-1]):
        raise ValueError(f"target {target_name} is not an exact horizontal mirror; cannot mirror cue frontend")
    circle_contrast = (cue.baseline - 255.0) / 255.0
    cue_contrast = (cue.image - 255.0) / 255.0
    target_contrast = (target.image - 255.0) / 255.0
    scenes = np.stack([circle_contrast, cue_contrast, target_contrast]).astype(np.float32)
    scene_index = build_scene_timeline(TIME_MS, target_onset_ms=TARGET_ONSET_MS)

    # Warm LGN with the circle-only scene before the recorded -1 s epoch.
    warm_contrast = resize_area(circle_contrast, 64)
    warm_time = np.arange(0.0, 1201.0, 1.0)
    *_, state = _simulate_lgn(warm_contrast, "Stage1", warm_time, float(tau_a),
                              include_offset=False, return_state=True)
    left = simulate_frontend(
        cue, params={"tau_a": float(tau_a)}, time_ms=TIME_MS,
        resolution=64, scene_contrasts=scenes, scene_index=scene_index,
        initial_lgn_state=state, feature_stride_ms=4.0)
    right = mirror_stage1_frontend(left, load_stimulus("Stage1", "right"))
    return left, right


def _load_time_blocks(dataset):
    data = load_dataset(config.REAL_ROOT / f"{dataset}_clean.mat")
    time_ms = data["time_s"] * 1000.0
    baseline = (time_ms >= -200.0) & (time_ms < 0.0)
    response = (time_ms >= RESPONSE_MS[0]) & (time_ms <= RESPONSE_MS[1] + 1e-9)
    if baseline.sum() < 2 or response.sum() < 50:
        raise ValueError(f"{dataset}: incomplete baseline or Stage1 response interval")
    corrected = data["eeg"] - data["eeg"][:, :, baseline].mean(axis=2, keepdims=True)
    n_trials = corrected.shape[0]
    split = n_trials // 2
    groups = {
        "first": np.arange(0, split),
        "second": np.arange(split, n_trials),
    }
    blocks = {}
    for block, indices in groups.items():
        blocks[block] = {}
        for condition, label in (("left", -1.0), ("right", 1.0)):
            selected = indices[data["cue_type"][indices] == label]
            if selected.size < 8:
                raise ValueError(f"{dataset}/{block}/{condition}: only {selected.size} trials")
            blocks[block][condition] = {
                "erp": corrected[selected][:, :, response].mean(axis=0),
                "n": int(selected.size),
            }
    return time_ms[response], blocks, n_trials, split


def _predict_fixed(source, lead, gain):
    return float(gain) * np.einsum("rs,cst->crt", lead, source)


def _metrics(prediction, observed):
    pred, obs = np.asarray(prediction, dtype=float), np.asarray(observed, dtype=float)
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    denom = float(np.sqrt(np.mean(obs ** 2)))
    common_pred = pred.mean(axis=0)
    common_obs = obs.mean(axis=0)
    difference_pred = pred[1] - pred[0]
    difference_obs = obs[1] - obs[0]
    difference_rms = float(np.sqrt(np.mean(difference_obs ** 2)))
    difference_rmse = float(np.sqrt(np.mean((difference_pred - difference_obs) ** 2)))
    a, b = difference_pred.ravel(), difference_obs.ravel()
    difference_corr = (float(np.corrcoef(a, b)[0, 1])
                       if np.std(a) > 1e-12 and np.std(b) > 1e-12 else float("nan"))
    common_rmse = float(np.sqrt(np.mean((common_pred - common_obs) ** 2)))
    common_scale = float(np.sqrt(np.mean(common_obs ** 2)))
    return {
        "sensor_nrmse": rmse / denom if denom else float("nan"),
        "common_mode_nrmse": common_rmse / common_scale if common_scale else float("nan"),
        "condition_difference_nrmse": difference_rmse / difference_rms if difference_rms else float("nan"),
        "condition_difference_rms_real": difference_rms,
        "condition_difference_rms_model": float(np.sqrt(np.mean(difference_pred ** 2))),
        "condition_difference_correlation": difference_corr,
    }


def _prepare_source_predictions(dataset, task, params, target_fronts):
    all_sources = {"left": [], "right": []}
    scales = None
    for target_name, fronts in target_fronts.items():
        left_front, right_front = fronts
        if scales is None:
            scales = calibrate_drive_scales({"left": left_front, "right": right_front}, "opponent")
        for condition, front in (("left", left_front), ("right", right_front)):
            model = simulate_forward(front, params=params, feature_route="opponent",
                                     drive_scales=scales)
            processed, model_time = model_sources_to_q1_grid(model.source_proxy, model.time_ms)
            all_sources[condition].append(processed)
    # Task2 has no trial-certified target layout. The two documented target
    # layouts are treated as equally weighted nuisance scenarios, not labels.
    return {condition: np.mean(np.stack(values), axis=0)
            for condition, values in all_sources.items()}, scales


def _write_figure(plot_data, path, figure_title):
    records = list(config.DATASETS["Task1"] + config.DATASETS["Task2"])
    fig, axes = plt.subplots(len(records), 3, figsize=(14, 12), sharex=True)
    modes = ("u0 common", "u1 lateral", "u2 sensor contrast")
    colors = {"real": "#202020", "fixed": "#d17c00", "effective": "#2377b4"}
    for ri, record in enumerate(records):
        datum = plot_data[record]
        time_ms = datum["time_ms"]
        real = datum["real"].mean(axis=0)
        fixed = datum["fixed"].mean(axis=0)
        effective = datum["effective"].mean(axis=0)
        for mi in range(3):
            ax = axes[ri, mi]
            real_diff = real[1] - real[0]
            fixed_diff = fixed[1] - fixed[0]
            effective_diff = effective[1] - effective[0]
            for label, matrix in (("real", real_diff), ("fixed", fixed_diff), ("effective", effective_diff)):
                value = observable_modes_from_real(matrix)[mi]
                ax.plot(time_ms, value, color=colors[label], lw=1.45,
                        label={"real": "实测留出", "fixed": "固定映射", "effective": "训练估计映射 α=1"}[label])
            ax.axhline(0.0, color="0.6", lw=0.6)
            ax.axvline(200.0, color="0.45", ls="--", lw=0.7)
            ax.grid(alpha=0.2)
            if ri == 0:
                ax.set_title(modes[mi])
            if mi == 0:
                short_record = record.replace("VisualCogA_", "A/").replace("VisualCogB_", "B/")
                ax.set_ylabel(short_record + "\n相对幅值")
            if ri == len(records) - 1:
                ax.set_xlabel("提示出现后时间 (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(figure_title, y=0.995, fontsize=14)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.967),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if not FIT_JSON.exists():
        raise FileNotFoundError(f"frozen v3 fit file not found: {FIT_JSON}")
    fitted = json.loads(FIT_JSON.read_text(encoding="utf-8"))
    target_front_cache = {}
    results, coefficient_rows, plot_data = [], [], {}
    effective_maps = {}
    lead = build_sensor_leadfield()
    source_labels = source_channel_labels()
    records = [(task, name) for task, names in config.DATASETS.items() for name in names]

    for task, dataset in records:
        fit = fitted[dataset]
        params = ModelParams.from_any(fit["parameters"])
        tau_a = float(params.tau_a)
        target_names = TARGETS_BY_TASK[task]
        for target_name in target_names:
            cache_key = (tau_a, target_name)
            if cache_key not in target_front_cache:
                print(f"  building full-scene frontend tau_a={tau_a:g} ms, target={target_name}", flush=True)
                target_front_cache[cache_key] = _scene_frontend(target_name, tau_a)
        front_pairs = {name: target_front_cache[(tau_a, name)] for name in target_names}
        source_by_condition, drive_scales = _prepare_source_predictions(dataset, task, params, front_pairs)
        # Model source currents after the same Q1 filter/resample/baseline operator.
        model_time = -1000.0 + np.arange(512) * (1000.0 / 128.0)
        model_slice = (model_time >= RESPONSE_MS[0]) & (model_time <= RESPONSE_MS[1] + 1e-9)
        source = np.stack([source_by_condition[name][:, model_slice]
                           for name in ("left", "right")])
        response_time = model_time[model_slice]

        real_time, blocks, n_trials, split = _load_time_blocks(dataset)
        if not np.allclose(real_time, response_time, atol=1e-8, rtol=0):
            raise ValueError(f"{dataset}: modeled and real Q1 response time grids differ")
        fold_outputs = []
        for train_name, test_name in (("first", "second"), ("second", "first")):
            train_real = np.stack([blocks[train_name][name]["erp"] for name in ("left", "right")])
            test_real = np.stack([blocks[test_name][name]["erp"] for name in ("left", "right")])
            source_train = source
            fixed_unscaled = _predict_fixed(source_train, lead, 1.0)
            fixed_gain = fit_shared_gain(fixed_unscaled, train_real)
            fixed_map = fixed_gain * lead
            fixed_train = _predict_fixed(source_train, lead, fixed_gain)
            fixed_test = fixed_train
            fixed_metrics_train = _metrics(fixed_train, train_real)
            fixed_metrics_test = _metrics(fixed_test, test_real)
            results.append({"record": dataset, "task": task, "train_block": train_name,
                            "test_block": test_name, "mapping": "legacy_fixed_G_train_scalar",
                            "alpha": "", "gain": fixed_gain,
                            "train_trials_left": blocks[train_name]["left"]["n"],
                            "train_trials_right": blocks[train_name]["right"]["n"],
                            "test_trials_left": blocks[test_name]["left"]["n"],
                            "test_trials_right": blocks[test_name]["right"]["n"],
                            **{f"train_{k}": v for k, v in fixed_metrics_train.items()},
                            **{f"test_{k}": v for k, v in fixed_metrics_test.items()}})
            coefficient_rows.extend({"record": dataset, "train_block": train_name,
                                     "mapping": "legacy_fixed_G_train_scalar", "alpha": "",
                                     "sensor": sensor, "source": source_name,
                                     "weight": float(fixed_map[si, sj])}
                                    for si, sensor in enumerate(config.CHANNELS)
                                    for sj, source_name in enumerate(source_labels))

            train_map_predictions = {}
            test_map_predictions = {}
            for alpha in ALPHAS:
                effective_map = fit_effective_mapping(source_train, train_real,
                                                       fixed_map, alpha=alpha)
                effective_maps[(dataset, train_name, float(alpha))] = effective_map
                train_pred = np.einsum("rs,cst->crt", effective_map, source_train)
                test_pred = train_pred
                mtrain, mtest = _metrics(train_pred, train_real), _metrics(test_pred, test_real)
                map_name = f"effective_ridge_alpha_{alpha:g}"
                results.append({"record": dataset, "task": task, "train_block": train_name,
                                "test_block": test_name, "mapping": map_name, "alpha": alpha,
                                "gain": "fit_in_map",
                                "train_trials_left": blocks[train_name]["left"]["n"],
                                "train_trials_right": blocks[train_name]["right"]["n"],
                                "test_trials_left": blocks[test_name]["left"]["n"],
                                "test_trials_right": blocks[test_name]["right"]["n"],
                                **{f"train_{k}": v for k, v in mtrain.items()},
                                **{f"test_{k}": v for k, v in mtest.items()}})
                train_map_predictions[alpha] = train_pred
                test_map_predictions[alpha] = test_pred
                coefficient_rows.extend({"record": dataset, "train_block": train_name,
                                         "mapping": map_name, "alpha": alpha,
                                         "sensor": sensor, "source": source_name,
                                         "weight": float(effective_map[si, sj])}
                                        for si, sensor in enumerate(config.CHANNELS)
                                        for sj, source_name in enumerate(source_labels))

            # Trial-average persistence is a direct non-neural ERP baseline.
            erp_base = np.broadcast_to(train_real, test_real.shape)
            erp_metrics = _metrics(erp_base, test_real)
            results.append({"record": dataset, "task": task, "train_block": train_name,
                            "test_block": test_name, "mapping": "train_ERP_persistence_baseline",
                            "alpha": "", "gain": "none",
                            "train_trials_left": blocks[train_name]["left"]["n"],
                            "train_trials_right": blocks[train_name]["right"]["n"],
                            "test_trials_left": blocks[test_name]["left"]["n"],
                            "test_trials_right": blocks[test_name]["right"]["n"],
                            **{f"test_{k}": v for k, v in erp_metrics.items()}})

            fold_outputs.append({"train": train_name, "test": test_name,
                                 "real": test_real, "fixed": fixed_test,
                                 "effective": test_map_predictions[PRIMARY_ALPHA]})

        # For display only, average the two chronological held-out directions.
        plot_data[dataset] = {"time_ms": response_time,
                              "real": np.stack([x["real"] for x in fold_outputs]),
                              "fixed": np.stack([x["fixed"] for x in fold_outputs]),
                              "effective": np.stack([x["effective"] for x in fold_outputs])}

    _write_csv(OUT / "mapping_holdout_metrics.csv", results)
    _write_csv(OUT / "mapping_coefficients.csv", coefficient_rows)
    _write_figure(plot_data, OUT / "lr_difference_holdout.png",
                  "左右条件差异：同记录时间块留出（提示 offset=200 ms）")

    # Summarize descriptive record-level means; records are not independent subjects.
    summary_rows = []
    for mapping in ["legacy_fixed_G_train_scalar", "effective_ridge_alpha_0.1",
                    "effective_ridge_alpha_1", "effective_ridge_alpha_10",
                    "train_ERP_persistence_baseline"]:
        rows = [r for r in results if r["mapping"] == mapping]
        summary_rows.append({"mapping": mapping,
                             "n_record_directions": len(rows),
                             "mean_test_sensor_nrmse": float(np.nanmean([r["test_sensor_nrmse"] for r in rows])),
                             "mean_test_common_mode_nrmse": float(np.nanmean([r["test_common_mode_nrmse"] for r in rows])),
                             "mean_test_condition_difference_nrmse": float(np.nanmean([r["test_condition_difference_nrmse"] for r in rows])),
                             "mean_test_condition_difference_correlation": float(np.nanmean([r["test_condition_difference_correlation"] for r in rows]))})
    _write_csv(OUT / "mapping_summary.csv", summary_rows)

    by_record_rows = []
    for dataset in sorted(name for _, name in records):
        for mapping in ["legacy_fixed_G_train_scalar", "effective_ridge_alpha_0.1",
                        "effective_ridge_alpha_1", "effective_ridge_alpha_10",
                        "train_ERP_persistence_baseline"]:
            rows = [r for r in results if r["record"] == dataset and r["mapping"] == mapping]
            by_record_rows.append({"record": dataset, "mapping": mapping,
                                   "n_directions": len(rows),
                                   "mean_test_sensor_nrmse": float(np.nanmean([r["test_sensor_nrmse"] for r in rows])),
                                   "mean_test_condition_difference_nrmse": float(np.nanmean([r["test_condition_difference_nrmse"] for r in rows])),
                                   "mean_test_condition_difference_correlation": float(np.nanmean([r["test_condition_difference_correlation"] for r in rows]))})
    _write_csv(OUT / "mapping_by_record.csv", by_record_rows)

    stability_rows = []
    for dataset in sorted(name for _, name in records):
        for alpha in ALPHAS:
            first = effective_maps[(dataset, "first", float(alpha))].ravel()
            second = effective_maps[(dataset, "second", float(alpha))].ravel()
            denom = float(np.linalg.norm(first) * np.linalg.norm(second))
            stability_rows.append({"record": dataset, "alpha": alpha,
                                   "map_cosine_first_vs_second": float(first @ second / denom) if denom else float("nan"),
                                   "relative_map_difference": float(np.linalg.norm(first - second) /
                                                                    (0.5 * (np.linalg.norm(first) + np.linalg.norm(second))))})
    _write_csv(OUT / "mapping_stability.csv", stability_rows)

    def _mean(rows, key):
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "", "nan")]
        return float(np.mean(values)) if values else float("nan")

    selected_mean = {row["mapping"]: row for row in summary_rows}
    stability_primary = [row for row in stability_rows if float(row["alpha"]) == PRIMARY_ALPHA]
    by_record_primary = [row for row in by_record_rows if row["mapping"] in
                         ("legacy_fixed_G_train_scalar", "effective_ridge_alpha_1",
                          "train_ERP_persistence_baseline")]
    lines = [
        "# 第一轮：完整提示场景与观测映射留出比较",
        "",
        "## 本轮做了什么",
        "",
        "冻结 revision_v3 的 LGN/Wilson–Cowan 参数，未重新搜索动力学参数。视觉输入改为完整场景序列：模型先用 1200 ms 圆圈场景预热 LGN 状态，再在记录 epoch 的 -1000–0 ms 继续呈现圆圈背景；圆圈+三角提示在 0–200 ms 显示，提示消失后恢复圆圈；2.2 s 加入目标场景，以覆盖 Q1 对完整四秒 epoch 的零相位滤波输入。圆圈开始时间未由 EEG 事件通道标记，因此这是稳态预热假设，等效于提示前至少约 2.2 s 已有圆圈场景，不是已确认的实验显示时长。Task1 使用双圆点目标；Task2 的真实目标布局没有试次级认证，因此将 inward/outward 两种前向预测等权平均作为后续目标的干扰场景，不赋予真实标签。",
        "",
        "每份 MAT 按保留试次顺序分前、后两块，分别做每试次 [-200,0) ms 基线校正；做 first→second 与 second→first 两个方向。旧固定 G 只在训练块估一个共享标量；受约束有效 G 在同一训练块估计，惩罚向已拟合标量后的固定 G 收缩，报告 α=0.1、1、10 三档，主读数为 α=1。两种映射用同一组模型源曲线，并在另一时间块评价。另加训练块 ERP 原样预测留出块的基线。",
        "",
        "## 汇总结果",
        "",
        "以下是 4 份记录、双向时间块的 8 个描述性留出折均值；四份 MAT 不是四名独立被试。NRMSE=1 表示与零预测的误差处于实测 RMS 同量级。",
        "",
        "| 方法 | 全通道留出 NRMSE | 共同模式 NRMSE | 左右差异 NRMSE | 左右差异相关 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mapping, title in [("legacy_fixed_G_train_scalar", "固定 G + 训练标量"),
                           ("effective_ridge_alpha_0.1", "有效 G，α=0.1"),
                           ("effective_ridge_alpha_1", "有效 G，α=1（主读数）"),
                           ("effective_ridge_alpha_10", "有效 G，α=10"),
                           ("train_ERP_persistence_baseline", "训练块 ERP 持续基线")]:
        row = selected_mean[mapping]
        lines.append(f"| {title} | {row['mean_test_sensor_nrmse']:.3f} | {row['mean_test_common_mode_nrmse']:.3f} | {row['mean_test_condition_difference_nrmse']:.3f} | {row['mean_test_condition_difference_correlation']:.3f} |")
    lines += ["", "α=1 的有效 G 平均只把全通道 NRMSE 从固定 G 的 1.031 降至 0.998；左右差异 NRMSE 从 1.183 降至 1.012，差异相关从 -0.157 升至 0.068。较松的 α=0.1 在左右差异上更好（NRMSE 0.904，相关 0.204），但该提升没有在记录间稳定出现，不能据此把 α=0.1 选成最终模型。直接 ERP 持续基线的全通道 NRMSE 为 0.811，整体优于两个神经源映射。",
            "", "## 记录内差异与映射稳定性", "",
            "| 记录 | 固定 G 全通道 NRMSE | 有效 G α=1 | ERP 持续基线 | α=1 两块映射余弦 |", "|---|---:|---:|---:|---:|"]
    stability_by_record = {row["record"]: row for row in stability_primary}
    for dataset in sorted(name for _, name in records):
        rows = {row["mapping"]: row for row in by_record_primary if row["record"] == dataset}
        stab = stability_by_record[dataset]
        lines.append(f"| {dataset} | {rows['legacy_fixed_G_train_scalar']['mean_test_sensor_nrmse']:.3f} | {rows['effective_ridge_alpha_1']['mean_test_sensor_nrmse']:.3f} | {rows['train_ERP_persistence_baseline']['mean_test_sensor_nrmse']:.3f} | {stab['map_cosine_first_vs_second']:.3f} |")
    lines += ["", "A_Task-1 对有效映射的改善最大；A_Task-2 改善较小；B_Task-1/B_Task-2 的全通道误差仍约为或高于零预测。有效 G 在两份 B 记录的双向训练结果不稳定：α=1 的两块映射余弦分别为 -0.425 和 0.151，表明用一块数据估到的通道权重不能稳定迁移到另一块。高自由度映射存在过拟合风险。",
              "", "## 结论", "",
              "时间上下文的修正和训练估计观测映射带来了一次可量化诊断，但没有解决整体拟合。固定 G 确实是部分记录的限制（尤其 A_Task-1），然而它不是唯一主因：有效映射的留出增益较小且记录依赖，B 记录权重不稳定，而简单 ERP 持续基线总体更好。现有证据不支持把下一步定为继续放开导联矩阵或挑选 α；应优先处理跨块 ERP 的非平稳性/记录差异，并在训练数据内部检查共同响应可预测性，再决定是否修改神经动力学。模型的方向偏好通道是功能性三角模板偏好，不解释为左右脑半球。",
              "", "## 限制", "",
              "1. 圆圈背景开始时刻未被 VisCue 事件标记；1200 ms 预热及整个 -1000–0 ms 的圆圈持续显示是状态初始化假设，未做预热时长敏感性分析。",
              "2. 2.2 s 目标时间沿用 Q1 分析时序假设；Task2 后续目标按两个标准化布局等权平均，真实逐试次布局未确认。目标输入只用于匹配完整 epoch 过滤，不作为 cue 分类标签。",
              "3. 观测映射只在记录内时间块训练，因此结果是映射可迁移性诊断，不是跨记录/跨被试验证，也不是生理导联场估计。",
              "4. 动力学参数冻结自旧 leave-one-record 拟合；本轮不把映射变化与动力学重拟合混在一起。",
              "5. v3 原结果和原始 EEG 均未覆盖或修改。",
              "", "图：`lr_difference_holdout.png`。数值：`mapping_holdout_metrics.csv`、`mapping_summary.csv`、`mapping_by_record.csv`、`mapping_stability.csv`、`mapping_coefficients.csv`。", ""]
    (OUT / "第一轮结果与结论.md").write_text("\n".join(lines), encoding="utf-8")

    generated = {
        "analysis": "full-scene LGN warmup, cue onset/offset, shared WC, blocked train-estimated observation map",
        "cue_timeline_ms": {"circle_pre_cue": "at least -2200..0 assumed; 1200 ms warmup plus recorded -1000..0",
                             "lgn_warmup": "1200 ms stationary circle-only numerical initialization, not an observed display interval",
                             "cue_scene": "0..199",
                             "circle_after_offset": "200..2199", "target_nuisance_onset": 2200},
        "target_policy": {"Task1": "dots target at 2.2 s",
                          "Task2": "equal average of inward/outward predictions; not inferred trial labels"},
        "source_labels": list(source_labels),
        "mapping_alpha_sensitivity": list(ALPHAS),
        "primary_alpha": PRIMARY_ALPHA,
        "filtering": "same 256-to-128 Hz Q1 zero-phase filter and [-200,0) per-source baseline",
        "evaluation": "within-record chronological first/second trial blocks, both directions",
        "records_are_participants": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote first-round results to {OUT}", flush=True)
    for row in summary_rows:
        print(row, flush=True)


if __name__ == "__main__":
    started = time.perf_counter()
    run()
    print(f"Elapsed: {time.perf_counter() - started:.1f} s", flush=True)
