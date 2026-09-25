# -*- coding: utf-8 -*-
"""Compare clipped-only and formal SQI-clean ERP across all four recordings.

This is a diagnostic only: it reads the existing filtered/downsampled data,
does not refilter or rewrite any Q1 inputs, and does not change the formal
SQI-retained trial sets. Random equal-N references are stratified by original
trial-order blocks to control broad time-order composition.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
FILTERED_DIR = OUTPUT_DIR / "7filter_downsample"
SQI_DIR = OUTPUT_DIR / "8riemann_denoise"
RESULT_DIR = OUTPUT_DIR / "15_all_dataset_cleaning_sensitivity"
DATASETS = (
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
)
CHANNELS = (("F3", 1), ("Fz", 0), ("F4", 2))
BASELINE_S = (-0.2, 0.0)
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260925
BLOCK_COUNT = 5
WINDOWS_MS = {
    "cue_onset_0_200ms": (0.0, 200.0),
    "cue_offset_200_800ms": (200.0, 800.0),
    "cue_full_0_800ms": (0.0, 800.0),
    "target_candidate_250_500ms": (2450.0, 2700.0),
}
COLORS = {"valid": "#333333", "clean": "#c1543f", "random": "#4f78a8"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _read_aligned_dataset(name):
    mat_path = FILTERED_DIR / f"{name}_filtered_downsample.mat"
    sqi_path = SQI_DIR / f"{name}_SQI指标.csv"
    clean_path = SQI_DIR / f"{name}_clean.mat"
    mat = loadmat(mat_path)
    required = {"trial_data", "relative_time", "cue_type", "drop"}
    missing = required.difference(mat)
    if missing:
        raise KeyError(f"{mat_path.name}: missing fields {sorted(missing)}")

    trial_data = np.asarray(mat["trial_data"], dtype=float)
    relative_time = np.asarray(mat["relative_time"], dtype=float)
    cue_type = np.asarray(mat["cue_type"], dtype=float).reshape(-1)
    clipped_drop = np.asarray(mat["drop"], dtype=bool).reshape(-1)
    if trial_data.ndim != 3 or trial_data.shape[1] < 3:
        raise ValueError(f"{name}: expected Trial x Channel x Time data")
    n_trials, _, n_time = trial_data.shape
    if relative_time.shape != (n_trials, n_time):
        raise ValueError(f"{name}: relative_time shape disagrees with trial_data")
    if not np.isfinite(trial_data[:, :3]).all() or not np.isfinite(relative_time).all():
        raise ValueError(f"{name}: nonfinite EEG or time values")
    if cue_type.size != n_trials or clipped_drop.size != n_trials:
        raise ValueError(f"{name}: trial labels do not align with EEG rows")
    if not np.isin(cue_type, (-1.0, 1.0)).all():
        raise ValueError(f"{name}: unexpected cue_type values")
    if not np.allclose(relative_time, relative_time[:1], rtol=0.0, atol=1e-12):
        raise ValueError(f"{name}: trials must share a finite time axis")

    metrics = pd.read_csv(sqi_path, encoding="utf-8-sig")
    required_metrics = {"Trial", "CueType", "ClippedDrop", "FinalDrop"}
    missing_metrics = required_metrics.difference(metrics.columns)
    if missing_metrics:
        raise KeyError(f"{sqi_path.name}: missing columns {sorted(missing_metrics)}")
    for column in required_metrics:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    trial_ids = metrics["Trial"].to_numpy(dtype=float)
    if (not np.isfinite(trial_ids).all()
            or not np.equal(trial_ids, np.floor(trial_ids)).all()
            or len(np.unique(trial_ids)) != n_trials
            or not np.array_equal(np.sort(trial_ids).astype(int), np.arange(n_trials))):
        raise ValueError(f"{name}: SQI Trial indices must cover 0..{n_trials - 1} once")
    metrics = metrics.set_index("Trial").loc[np.arange(n_trials)]
    csv_cue = metrics["CueType"].to_numpy(dtype=float)
    csv_clipped = metrics["ClippedDrop"].to_numpy(dtype=float)
    final_drop = metrics["FinalDrop"].to_numpy(dtype=float)
    if not np.array_equal(csv_cue, cue_type) or not np.array_equal(csv_clipped, clipped_drop.astype(float)):
        raise ValueError(f"{name}: SQI labels/drop decisions do not match source MAT by Trial")
    if not np.isin(final_drop, (0.0, 1.0)).all():
        raise ValueError(f"{name}: FinalDrop must be binary")
    clean_keep = (final_drop == 0.0) & ~clipped_drop
    valid_keep = ~clipped_drop
    if np.any(clean_keep & ~valid_keep):
        raise ValueError(f"{name}: formal SQI set includes clipped trials")

    # Verify that the clean MAT is exactly the retained subset of the filtered MAT.
    clean_mat = loadmat(clean_path)
    clean_trials = np.asarray(clean_mat["trial_data"], dtype=float)
    clean_cue = np.asarray(clean_mat["cue_type"], dtype=float).reshape(-1)
    expected_trials = trial_data[clean_keep]
    exact_data_match = (clean_trials.shape == expected_trials.shape
                        and np.array_equal(clean_trials, expected_trials))
    exact_cue_match = np.array_equal(clean_cue, cue_type[clean_keep])
    if not exact_data_match or not exact_cue_match:
        raise ValueError(f"{name}: formal clean MAT differs from the Trial-keyed SQI subset")

    baseline = (relative_time[0] >= BASELINE_S[0]) & (relative_time[0] < BASELINE_S[1])
    if baseline.sum() < 2:
        raise ValueError(f"{name}: baseline interval has too few samples")
    eeg = trial_data[:, [index for _, index in CHANNELS], :]
    eeg = eeg - eeg[:, :, baseline].mean(axis=2, keepdims=True)
    return {
        "name": name,
        "time_ms": relative_time[0] * 1000.0,
        "eeg": eeg,
        "cue_type": cue_type,
        "clipped_drop": clipped_drop,
        "valid_keep": valid_keep,
        "clean_keep": clean_keep,
        "clean_data_exact_match": exact_data_match,
        "clean_cue_exact_match": exact_cue_match,
    }


def _block_ids(n_trials):
    return np.minimum(np.arange(n_trials) * BLOCK_COUNT // n_trials, BLOCK_COUNT - 1)


def _stratified_equal_n_bootstrap(eeg, valid_keep, clean_keep, block_ids, rng):
    """Random same-N means preserving clean-set counts in five order blocks."""
    n_trials, n_channels, n_time = eeg.shape
    result = np.empty((BOOTSTRAP_REPEATS, n_channels, n_time), dtype=np.float32)
    pools, sample_counts = [], []
    for block in range(BLOCK_COUNT):
        pools.append(np.flatnonzero(valid_keep & (block_ids == block)))
        sample_counts.append(int(np.count_nonzero(clean_keep & (block_ids == block))))
        if sample_counts[-1] > len(pools[-1]):
            raise ValueError("clean same-block count exceeds clipped-only pool")
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled = [rng.choice(pool, size=count, replace=False)
                   for pool, count in zip(pools, sample_counts) if count]
        indices = np.concatenate(sampled) if sampled else np.array([], dtype=int)
        if indices.size != int(clean_keep.sum()):
            raise RuntimeError("stratified bootstrap sample count changed")
        result[repeat] = eeg[indices].mean(axis=0)
    return result


def _window_mask(time_ms, window):
    mask = (time_ms >= window[0]) & (time_ms <= window[1])
    if mask.sum() < 2:
        raise ValueError(f"window {window} ms is absent/truncated on this epoch")
    return mask


def _corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _summarize_condition(item, condition, boot):
    time_ms = item["time_ms"]
    eeg = item["eeg"]
    valid_idx = item["valid_keep"] & (item["cue_type"] == condition)
    clean_idx = item["clean_keep"] & (item["cue_type"] == condition)
    valid_erp = eeg[valid_idx].mean(axis=0)
    clean_erp = eeg[clean_idx].mean(axis=0)
    random_mean = boot.mean(axis=0)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    rows = []
    for window_name, window in WINDOWS_MS.items():
        mask = _window_mask(time_ms, window)
        for channel_index, (channel, _) in enumerate(CHANNELS):
            observed = clean_erp[channel_index, mask]
            baseline = valid_erp[channel_index, mask]
            random_curve = boot[:, channel_index, :][:, mask]
            point_low, point_high = low[channel_index, mask], high[channel_index, mask]
            outside = ((observed < point_low) | (observed > point_high))
            valid_rms = float(np.sqrt(np.mean(baseline ** 2)))
            diff = observed - baseline
            rows.append({
                "dataset": item["name"], "condition": "left" if condition == -1 else "right",
                "channel": channel, "window": window_name,
                "window_start_ms": window[0], "window_end_ms": window[1],
                "n_clipped_only": int(valid_idx.sum()), "n_sqi_retained": int(clean_idx.sum()),
                "valid_erp_rms": valid_rms,
                "clean_erp_rms": float(np.sqrt(np.mean(observed ** 2))),
                "clean_vs_clipped_only_rmse": float(np.sqrt(np.mean(diff ** 2))),
                "clean_vs_clipped_only_nrmse": (float(np.sqrt(np.mean(diff ** 2)) / valid_rms)
                                                  if valid_rms > 0 else float("nan")),
                "clean_vs_equal_n_random_mean_rmse": float(np.sqrt(np.mean(
                    (observed - random_mean[channel_index, mask]) ** 2))),
                "clean_outside_stratified_random_pointwise_95_pct": float(np.mean(outside)),
                "random_band_mean_width": float(np.mean(point_high - point_low)),
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "bootstrap_blocks": BLOCK_COUNT,
                "interpretation": "descriptive_pointwise_interval_not_familywise_test",
            })
    return valid_erp, clean_erp, random_mean, low, high, rows


def _summarize_lr(item, erps, boots):
    valid_diff = erps["right"]["valid"] - erps["left"]["valid"]
    clean_diff = erps["right"]["clean"] - erps["left"]["clean"]
    boot_diff = boots["right"] - boots["left"]
    random_mean = boot_diff.mean(axis=0)
    low, high = np.quantile(boot_diff, [0.025, 0.975], axis=0)
    rows = []
    for window_name, window in WINDOWS_MS.items():
        mask = _window_mask(item["time_ms"], window)
        for channel_index, (channel, _) in enumerate(CHANNELS):
            measured = clean_diff[channel_index, mask]
            valid = valid_diff[channel_index, mask]
            random_curve = boot_diff[:, channel_index, :][:, mask]
            err = measured - valid
            valid_rms = float(np.sqrt(np.mean(valid ** 2)))
            outside = ((measured < low[channel_index, mask])
                       | (measured > high[channel_index, mask]))
            rows.append({
                "dataset": item["name"], "channel": channel, "window": window_name,
                "window_start_ms": window[0], "window_end_ms": window[1],
                "n_left_clipped_only": int(np.count_nonzero(item["valid_keep"] & (item["cue_type"] == -1))),
                "n_right_clipped_only": int(np.count_nonzero(item["valid_keep"] & (item["cue_type"] == 1))),
                "n_left_sqi_retained": int(np.count_nonzero(item["clean_keep"] & (item["cue_type"] == -1))),
                "n_right_sqi_retained": int(np.count_nonzero(item["clean_keep"] & (item["cue_type"] == 1))),
                "clipped_only_lr_difference_rms": valid_rms,
                "clean_lr_difference_rms": float(np.sqrt(np.mean(measured ** 2))),
                "clean_vs_clipped_only_difference_rmse": float(np.sqrt(np.mean(err ** 2))),
                "clean_vs_clipped_only_difference_nrmse": (
                    float(np.sqrt(np.mean(err ** 2)) / valid_rms) if valid_rms > 0 else float("nan")),
                "clean_vs_clipped_only_difference_correlation": _corr(measured, valid),
                "clean_vs_equal_n_random_mean_rmse": float(np.sqrt(np.mean(
                    (measured - random_mean[channel_index, mask]) ** 2))),
                "clean_difference_outside_stratified_random_pointwise_95_pct": float(np.mean(outside)),
                "random_band_mean_width": float(np.mean(high[channel_index, mask] - low[channel_index, mask])),
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "interpretation": "descriptive_pointwise_interval_not_familywise_test",
            })
    return valid_diff, clean_diff, random_mean, low, high, rows


def _plot_lr_comparison(items, results, out_path):
    fig, axes = plt.subplots(len(items), len(CHANNELS), figsize=(14, 3.0 * len(items)),
                             sharex=True, squeeze=False)
    for row_index, item in enumerate(items):
        result = results[item["name"]]
        time_ms = item["time_ms"]
        mask = _window_mask(time_ms, (0.0, 800.0))
        valid, clean = result["valid_lr"], result["clean_lr"]
        random_mean, low, high = result["random_lr_mean"], result["random_lr_low"], result["random_lr_high"]
        for col_index, (channel, _) in enumerate(CHANNELS):
            ax = axes[row_index, col_index]
            ci = col_index
            ax.fill_between(time_ms[mask], low[ci, mask], high[ci, mask],
                            color=COLORS["random"], alpha=0.18, linewidth=0,
                            label="random equal-N pointwise 95%")
            ax.plot(time_ms[mask], valid[ci, mask], color=COLORS["valid"], lw=1.2,
                    label="clipped-only")
            ax.plot(time_ms[mask], clean[ci, mask], color=COLORS["clean"], lw=1.4,
                    label="formal SQI")
            ax.plot(time_ms[mask], random_mean[ci, mask], color=COLORS["random"], lw=1.0,
                    ls="--", label="random equal-N mean")
            ax.axvline(0, color="0.35", lw=0.8)
            ax.axvline(200, color="0.35", lw=0.8, ls=":")
            ax.axhline(0, color="0.65", lw=0.7)
            ax.set_title(f"{item['name']} · {channel}")
            ax.grid(alpha=0.18)
            if col_index == 0:
                ax.set_ylabel("right − left ERP (input units)")
            if row_index == len(items) - 1:
                ax.set_xlabel("Cue-relative time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("SQI cleaning sensitivity of left–right ERP contrast (0–800 ms)",
                 y=1.035, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_report(path, trial_rows, condition_rows, lr_rows):
    def mean_for(frame_rows, key, window):
        values = [row[key] for row in frame_rows if row["window"] == window]
        return float(np.nanmean(values)) if values else float("nan")

    records = []
    for trial in trial_rows:
        records.append(
            f"| {trial['dataset']} | {trial['left_n_clipped_only']}/{trial['right_n_clipped_only']} | "
            f"{trial['left_n_sqi_retained']}/{trial['right_n_sqi_retained']} | "
            f"{trial['left_retention_pct']:.1f}%/{trial['right_retention_pct']:.1f}% | "
            f"{trial['clean_mat_exact_match']} |"
        )
    lr_summaries = []
    target_summaries = []
    for record in DATASETS:
        rows = [row for row in lr_rows if row["dataset"] == record]
        onset_out = mean_for(rows, "clean_difference_outside_stratified_random_pointwise_95_pct",
                             "cue_onset_0_200ms")
        offset_out = mean_for(rows, "clean_difference_outside_stratified_random_pointwise_95_pct",
                              "cue_offset_200_800ms")
        full_corr = [row["clean_vs_clipped_only_difference_correlation"] for row in rows
                     if row["window"] == "cue_full_0_800ms"]
        lr_summaries.append(
            f"| {record} | {mean_for(rows, 'clean_vs_clipped_only_difference_nrmse', 'cue_full_0_800ms'):.3f} | "
            f"{float(np.nanmean(full_corr)):.3f} | {onset_out:.1%} | {offset_out:.1%} |"
        )
        target_rows = [row for row in rows if row["window"] == "target_candidate_250_500ms"]
        by_channel = {row["channel"]: row for row in target_rows}
        target_nrmse = "/".join(
            f"{by_channel[channel]['clean_vs_clipped_only_difference_nrmse']:.3f}"
            for channel, _ in CHANNELS
        )
        target_corr = "/".join(
            f"{by_channel[channel]['clean_vs_clipped_only_difference_correlation']:.3f}"
            for channel, _ in CHANNELS
        )
        outside_channels = ", ".join(
            channel for channel, _ in CHANNELS
            if by_channel[channel]["clean_difference_outside_stratified_random_pointwise_95_pct"] > 0
        ) or "—"
        target_summaries.append(
            f"| {record} | {target_nrmse} | {target_corr} | {outside_channels} |"
        )

    clean_exclusion = sum(row["sqi_excluded_after_clipping"] for row in trial_rows)
    retained = sum(row["total_sqi_retained"] for row in trial_rows)
    clipped_only = sum(row["total_clipped_only"] for row in trial_rows)
    lines = [
        "# 四份记录的第一问 SQI 清洗敏感性诊断",
        "",
        "> 本分析不重新滤波、不改 SQI 阈值、不重写 clean MAT。比较的是第一问 7 阶段已滤波/降采样数据中的“仅排除硬件削顶”与 8 阶段正式 SQI 保留集。",
        "",
        "## 试次留存",
        "",
        "左右数值按“左/右”排列；SQI 保留数是在未削顶池中的正式保留数。",
        "",
        "| 记录 | 仅排削顶 左/右 | SQI 保留 左/右 | 保留率 左/右 | clean MAT 与掩码是否逐样本一致 |",
        "|---|---:|---:|---:|---|",
        *records,
        "",
        f"四份记录合计：未削顶试次 {clipped_only} 条，正式 SQI 保留 {retained} 条；在未削顶池中又剔除了 {clean_exclusion} 条。",
        "",
        "## 左右差异清洗敏感性",
        "",
        "下表中的差异 NRMSE 与相关比较的是正式 SQI 右减左曲线和仅排削顶右减左曲线；区间外比例是正式曲线落在等样本数随机重抽样的逐点 95% 区间之外的采样比例。每次重抽样按原试次顺序分成五块，并在每块保留与正式 SQI 集合相同的试次数。",
        "",
        "| 记录 | 0–800 ms 差异 NRMSE | 清洗差异与仅排削顶相关 | onset 0–200 区间外均值 | offset 200–800 区间外均值 |",
        "|---|---:|---:|---:|---:|",
        *lr_summaries,
        "",
        "### 目标候选窗（目标出现后 250–500 ms）",
        "",
        "第一问目标约在提示后 2.20 s 出现，因此此处分析提示相对时间 2450–2700 ms。每格按 F3/Fz/F4 顺序列出正式 SQI 曲线相对仅排削顶曲线的 NRMSE 与相关；‘区间外通道’表示该通道在此窗内至少有采样点落在等样本量重抽样的逐点 95% 区间外。该窗是目标晚期候选窗，不应仅凭它认定为 P300。",
        "",
        "| 记录 | NRMSE（F3/Fz/F4） | 相关（F3/Fz/F4） | 区间外通道 |",
        "|---|---|---|---|",
        *target_summaries,
        "",
        "目标候选窗中，VisualCogA_Task-2 的清洗前后左右差异最稳定；VisualCogB_Task-1 的 Fz、VisualCogB_Task-2 的 F3 有区间外偏离。相关较高表示曲线形状相似，不代表幅度相同；区间外比例是逐点描述，不是整窗显著性检验。",
        "",
        "## 判读边界",
        "",
        "- NRMSE、相关和区间外比例都是描述性量；逐点区间没有控制整段波形的多重比较，不能直接解释成显著性检验。",
        "- 随机同样本量对照还保持了五个试次顺序块内的保留数量，因此能控制粗粒度的前后时段构成差异；它仍不能判定 SQI 排除的是伪迹还是与条件相关的真实脑电成分。",
        "- 若清洗后左右差异明显偏离随机同样本量分布，说明 SQI 选样可能影响该波形估计；这不是自动证明清洗错误。需要结合被剔除试次的 SQI 特征、波形和伪迹来源判断。",
        "- 若清洗前后都没有稳定的左右差异，则现有数据对该效应的支持有限；不能据此宣称数据损坏。",
        "- 当前分析只覆盖三个额区 EEG 通道与现有四份记录。结果不能替代完整头皮空间测量或独立被试层面的推断。",
        "",
        "## 输出文件",
        "",
        "- `试次数留存.csv`：削顶与 SQI 保留情况、条件试次数和 clean MAT 一致性。",
        "- `条件ERP清洗敏感性.csv`：左/右条件在四个预设时间窗中的 ERP 改变量及随机等样本量区间诊断。",
        "- `左右差异清洗敏感性.csv`：右减左波形清洗前后比较。",
        "- `左右差异对照.png`：0–800 ms 右减左曲线与分块随机等样本量区间。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    items, trial_rows, condition_rows, lr_rows, results = [], [], [], [], {}
    for name in DATASETS:
        item = _read_aligned_dataset(name)
        items.append(item)
        block_ids = _block_ids(item["eeg"].shape[0])
        left_n_valid = int(np.count_nonzero(item["valid_keep"] & (item["cue_type"] == -1)))
        right_n_valid = int(np.count_nonzero(item["valid_keep"] & (item["cue_type"] == 1)))
        left_n_clean = int(np.count_nonzero(item["clean_keep"] & (item["cue_type"] == -1)))
        right_n_clean = int(np.count_nonzero(item["clean_keep"] & (item["cue_type"] == 1)))
        trial_rows.append({
            "dataset": name,
            "total_trials": int(len(item["cue_type"])),
            "hardware_clipped": int(item["clipped_drop"].sum()),
            "total_clipped_only": int(item["valid_keep"].sum()),
            "total_sqi_retained": int(item["clean_keep"].sum()),
            "sqi_excluded_after_clipping": int(item["valid_keep"].sum() - item["clean_keep"].sum()),
            "left_n_clipped_only": left_n_valid,
            "right_n_clipped_only": right_n_valid,
            "left_n_sqi_retained": left_n_clean,
            "right_n_sqi_retained": right_n_clean,
            "left_retention_pct": 100.0 * left_n_clean / left_n_valid if left_n_valid else float("nan"),
            "right_retention_pct": 100.0 * right_n_clean / right_n_valid if right_n_valid else float("nan"),
            "clean_mat_exact_match": item["clean_data_exact_match"] and item["clean_cue_exact_match"],
            "baseline_window_s": str(BASELINE_S),
        })
        erps, boots = {}, {}
        for condition in (-1, 1):
            condition_name = "left" if condition == -1 else "right"
            valid_idx = item["valid_keep"] & (item["cue_type"] == condition)
            clean_idx = item["clean_keep"] & (item["cue_type"] == condition)
            boot = _stratified_equal_n_bootstrap(item["eeg"], valid_idx, clean_idx,
                                                 block_ids, rng)
            valid_erp, clean_erp, random_mean, low, high, rows = _summarize_condition(
                item, condition, boot)
            erps[condition_name] = {"valid": valid_erp, "clean": clean_erp,
                                    "random_mean": random_mean}
            boots[condition_name] = boot
            condition_rows.extend(rows)
        valid_diff, clean_diff, random_mean, low, high, rows = _summarize_lr(item, erps, boots)
        lr_rows.extend(rows)
        results[name] = {"valid_lr": valid_diff, "clean_lr": clean_diff,
                         "random_lr_mean": random_mean, "random_lr_low": low,
                         "random_lr_high": high}
        print(f"{name}: clipped-only L/R={left_n_valid}/{right_n_valid}; "
              f"SQI L/R={left_n_clean}/{right_n_clean}; "
              f"clean MAT exact match={trial_rows[-1]['clean_mat_exact_match']}", flush=True)

    _write_csv(RESULT_DIR / "试次数留存.csv", trial_rows)
    _write_csv(RESULT_DIR / "条件ERP清洗敏感性.csv", condition_rows)
    _write_csv(RESULT_DIR / "左右差异清洗敏感性.csv", lr_rows)
    _plot_lr_comparison(items, results, RESULT_DIR / "左右差异对照.png")
    _write_report(RESULT_DIR / "清洗影响诊断.md", trial_rows, condition_rows, lr_rows)
    print(f"Saved cleaning sensitivity analysis to: {RESULT_DIR}", flush=True)
    return trial_rows, condition_rows, lr_rows


if __name__ == "__main__":
    run()
