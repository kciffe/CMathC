# -*- coding: utf-8 -*-
"""Exploratory 2x2 sensitivity analysis for clipping repair and SQI screening.

The imputed arm is deliberately a stress test, not a recovery of the unknown
EEG hidden by ADC saturation. Short clipped runs are PCHIP-interpolated and
long runs are linearly bridged so the existing Q1 filtering pipeline can run.
No source MAT or formal clean MAT is overwritten.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
SOURCE_DIR = OUTPUT_DIR
RESULT_DIR = OUTPUT_DIR / "16_clip_imputation_factorial"
DATASETS = (
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
)
CHANNELS = (("F3", 1), ("Fz", 0), ("F4", 2))
EEG_CHANNELS = (0, 1, 2)  # Q1/SQI native order: Fz, F3, F4
DISPLAY_ORDER = (1, 0, 2)  # figures and summaries: F3, Fz, F4
CLIP_THRESHOLD = 999.0
MAX_SHORT_RUN_S = 0.05
CONTEXT_SAMPLES = 12
FILTER_LOW_HZ = 0.2
FILTER_HIGH_HZ = 24.0
FILTER_ORDER = 4
SOURCE_SAMPLE_RATE = 256
OUTPUT_SAMPLE_RATE = 128
BASELINE_S = (-0.2, 0.0)
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260926
ORDER_BLOCKS = 5
WINDOWS_MS = {
    "cue_onset_0_200ms": (0.0, 200.0),
    "cue_offset_200_800ms": (200.0, 800.0),
    "cue_full_0_800ms": (0.0, 800.0),
    "target_candidate_250_500ms": (2450.0, 2700.0),
}
SCENARIOS = (
    ("剔除削顶_不做SQI", True, False),
    ("剔除削顶_执行SQI", True, True),
    ("插补削顶_不做SQI", False, False),
    ("插补削顶_执行SQI", False, True),
)
COLORS = {
    "剔除削顶_不做SQI": "#4C78A8",
    "剔除削顶_执行SQI": "#F58518",
    "插补削顶_不做SQI": "#54A24B",
    "插补削顶_执行SQI": "#E45756",
}
PLOT_LABELS = {
    "剔除削顶_不做SQI": "剔除 + 无 SQI",
    "剔除削顶_执行SQI": "剔除 + SQI",
    "插补削顶_不做SQI": "插补 + 无 SQI",
    "插补削顶_执行SQI": "插补 + SQI",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_local_module(filename, name):
    path = PROJECT_DIR / filename
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contiguous_runs(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts, ends))


def impute_clipped_signal(signal):
    """PCHIP-fill <=50 ms runs; linearly bridge longer clipped runs."""
    source = np.asarray(signal, dtype=float)
    if not np.isfinite(source).all():
        raise ValueError("Source EEG contains nonfinite values")
    clipped = np.abs(source) >= CLIP_THRESHOLD
    if not clipped.any():
        return source.copy(), clipped

    good = ~clipped
    if good.sum() < 2:
        raise ValueError("Cannot interpolate a channel with fewer than two valid samples")
    indices = np.arange(source.size)
    # This is the conservative bridge used to make long invalid intervals
    # filterable; its output is synthetic and does not recover the lost peak.
    result = source.copy()
    result[clipped] = np.interp(indices[clipped], indices[good], source[good])

    max_short_samples = int(round(MAX_SHORT_RUN_S * SOURCE_SAMPLE_RATE))
    for start, end in contiguous_runs(clipped):
        length = end - start
        if length > max_short_samples:
            continue
        lo = max(0, start - CONTEXT_SAMPLES)
        hi = min(source.size, end + CONTEXT_SAMPLES)
        local = np.arange(lo, hi)
        local_good = local[~clipped[lo:hi]]
        if local_good.size >= 4:
            result[start:end] = PchipInterpolator(
                local_good, source[local_good], extrapolate=True
            )(np.arange(start, end))
        elif local_good.size >= 2:
            result[start:end] = np.interp(
                np.arange(start, end), local_good, source[local_good]
            )
    return result, clipped


def filter_and_downsample(eeg, time_s):
    """Reproduce Q1 stage 7: 0.2–24 Hz zero-phase filtering, 256→128 Hz."""
    b, a = butter(
        FILTER_ORDER,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
        fs=SOURCE_SAMPLE_RATE,
    )
    filtered = np.asarray(eeg, dtype=np.float64).copy()
    for channel_index in range(filtered.shape[1]):
        filtered[:, channel_index] = filtfilt(b, a, filtered[:, channel_index], axis=-1)
    n_output = filtered.shape[-1] // (SOURCE_SAMPLE_RATE // OUTPUT_SAMPLE_RATE)
    downsampled = resample(filtered, n_output, axis=-1)
    return downsampled, np.asarray(time_s)[:: SOURCE_SAMPLE_RATE // OUTPUT_SAMPLE_RATE]


def load_dataset(name):
    path = SOURCE_DIR / f"{name}_sliced_with_drop.mat"
    data = loadmat(path)
    raw = np.asarray(data["trial_data"], dtype=np.float64)
    time = np.asarray(data["relative_time"], dtype=float)[0]
    cue = np.asarray(data["cue_type"], dtype=float).reshape(-1)
    clip_flag = np.asarray(data["drop"], dtype=bool).reshape(-1)
    fs = int(np.asarray(data["SampleRate"]).squeeze())
    if fs != SOURCE_SAMPLE_RATE or raw.shape[1] < 3:
        raise ValueError(f"{name}: unexpected sample rate or channel layout")
    detected = np.any(np.abs(raw[:, :3, :]) >= CLIP_THRESHOLD, axis=(1, 2))
    if not np.array_equal(detected, clip_flag):
        raise ValueError(f"{name}: clipping mask differs from the recorded Q1 drop flags")
    if cue.size != raw.shape[0] or not np.isin(cue, (-1, 1)).all():
        raise ValueError(f"{name}: invalid cue labels")

    raw_eeg = raw[:, EEG_CHANNELS, :]
    imputed_eeg = raw_eeg.copy()
    imputed_mask = np.zeros_like(raw_eeg, dtype=bool)
    run_lengths = []
    for trial in range(raw_eeg.shape[0]):
        for channel in range(raw_eeg.shape[1]):
            repaired, clip_samples = impute_clipped_signal(raw_eeg[trial, channel])
            imputed_eeg[trial, channel] = repaired
            imputed_mask[trial, channel] = clip_samples
            run_lengths.extend(end - start for start, end in contiguous_runs(clip_samples))

    # Verify this implementation reproduces the project's filtered data for
    # every unclipped trial before relying on it for the sensitivity arms.
    stage7_path = OUTPUT_DIR / "7filter_downsample" / f"{name}_filtered_downsample.mat"
    stage7 = loadmat(stage7_path)
    stage7_trials = np.asarray(stage7["trial_data"], dtype=np.float64)
    expected = stage7_trials[:, EEG_CHANNELS, :]
    valid_indices = np.flatnonzero(~clip_flag)
    rebuilt_valid, output_time = filter_and_downsample(raw_eeg[valid_indices], time)
    if rebuilt_valid.shape != expected[valid_indices].shape or not np.allclose(
        rebuilt_valid, expected[valid_indices], rtol=1e-10, atol=1e-8
    ):
        max_error = float(np.max(np.abs(rebuilt_valid - expected[valid_indices])))
        raise ValueError(f"{name}: stage-7 reproduction failed, max abs error={max_error}")

    return {
        "name": name,
        "raw": raw_eeg,
        "imputed_raw": imputed_eeg,
        "imputed_mask": imputed_mask,
        "clip_flag": clip_flag,
        "cue": cue,
        "time_s": time,
        "output_time_ms": output_time * 1000,
        "run_count": len(run_lengths),
        "run_gt_50ms": int(sum(length > round(MAX_SHORT_RUN_S * SOURCE_SAMPLE_RATE)
                                for length in run_lengths)),
        "max_run_ms": max(run_lengths, default=0) * 1000 / SOURCE_SAMPLE_RATE,
        "clipped_samples": int(imputed_mask.sum()),
        "valid_stage7_match": True,
    }


def run_sqi(eeg_ds, time_s, eligible, is_group_a, sqi_module):
    indices = np.flatnonzero(eligible)
    if indices.size < 4:
        raise ValueError("Too few trials to estimate SQI")
    _, _, sqi_values, threshold = sqi_module.calculate_sqi(
        eeg_ds, OUTPUT_SAMPLE_RATE, time_s, indices, is_group_a
    )
    keep = np.zeros(len(eligible), dtype=bool)
    keep[indices] = sqi_values >= threshold
    return keep, float(threshold), int(np.count_nonzero(~keep & eligible))


def baseline_correct(eeg, time_ms):
    mask = (time_ms >= BASELINE_S[0] * 1000) & (time_ms < BASELINE_S[1] * 1000)
    if mask.sum() < 2:
        raise ValueError("Baseline interval has too few samples")
    return eeg - eeg[:, :, mask].mean(axis=2, keepdims=True)


def window_mask(time_ms, window):
    mask = (time_ms >= window[0]) & (time_ms <= window[1])
    if mask.sum() < 2:
        raise ValueError(f"Window {window} ms is absent")
    return mask


def bootstrap_difference(eeg, left_indices, right_indices, trial_blocks, repeats, rng):
    n_channels, n_time = eeg.shape[1:]
    draws = np.empty((repeats, n_channels, n_time), dtype=np.float32)
    for repeat in range(repeats):
        sampled_left, sampled_right = [], []
        for block in range(ORDER_BLOCKS):
            left_pool = left_indices[trial_blocks[left_indices] == block]
            right_pool = right_indices[trial_blocks[right_indices] == block]
            if left_pool.size:
                sampled_left.append(rng.choice(left_pool, size=left_pool.size, replace=True))
            if right_pool.size:
                sampled_right.append(rng.choice(right_pool, size=right_pool.size, replace=True))
        left_draw = np.concatenate(sampled_left)
        right_draw = np.concatenate(sampled_right)
        draws[repeat] = eeg[right_draw].mean(axis=0) - eeg[left_draw].mean(axis=0)
    return np.quantile(draws, [0.025, 0.975], axis=0)


def analyze_scenario(item, scenario_name, exclude_clipped, use_sqi, sqi_module, rng):
    raw = item["raw"] if exclude_clipped else item["imputed_raw"]
    eeg_ds_native, time_s = filter_and_downsample(raw, item["time_s"])
    if not np.allclose(time_s * 1000, item["output_time_ms"], atol=1e-10):
        raise ValueError(f"{item['name']}: output time grid mismatch")

    eligible = ~item["clip_flag"] if exclude_clipped else np.ones(len(item["cue"]), dtype=bool)
    if use_sqi:
        keep, threshold, sqi_drop_n = run_sqi(
            eeg_ds_native, time_s, eligible,
            item["name"].startswith("VisualCogA"), sqi_module
        )
        # The exclude-clipped + SQI arm must reproduce the official Q1 trial set.
        if exclude_clipped:
            official = pd.read_csv(
                OUTPUT_DIR / "8riemann_denoise" / f"{item['name']}_SQI指标.csv",
                encoding="utf-8-sig",
            ).sort_values("Trial")
            expected_keep = (official["FinalDrop"].to_numpy() == 0)
            if not np.array_equal(keep, expected_keep):
                differing = np.flatnonzero(keep != expected_keep).tolist()
                raise ValueError(
                    f"{item['name']}: reproduced SQI mask differs from official mask; "
                    f"threshold={threshold:.8f}, differing_trials={differing}"
                )
    else:
        keep, threshold, sqi_drop_n = eligible.copy(), float("nan"), 0
    keep &= eligible
    if not np.any(keep & (item["cue"] == -1)) or not np.any(keep & (item["cue"] == 1)):
        raise ValueError(f"{item['name']} {scenario_name}: one cue condition has no trials")

    eeg_ds = eeg_ds_native[:, DISPLAY_ORDER, :]
    baseline_eeg = baseline_correct(eeg_ds, item["output_time_ms"])
    left_idx = keep & (item["cue"] == -1)
    right_idx = keep & (item["cue"] == 1)
    trial_indices = np.arange(len(item["cue"]))
    trial_blocks = np.minimum(
        trial_indices * ORDER_BLOCKS // len(trial_indices), ORDER_BLOCKS - 1
    )
    lr = baseline_eeg[right_idx].mean(axis=0) - baseline_eeg[left_idx].mean(axis=0)
    left_erp = baseline_eeg[left_idx].mean(axis=0)
    right_erp = baseline_eeg[right_idx].mean(axis=0)
    boot_low_high = bootstrap_difference(
        baseline_eeg, np.flatnonzero(left_idx), np.flatnonzero(right_idx),
        trial_blocks, BOOTSTRAP_REPEATS, rng
    )

    rows = []
    for window_name, window in WINDOWS_MS.items():
        mask = window_mask(item["output_time_ms"], window)
        for ci, (channel, _) in enumerate(CHANNELS):
            curve = lr[ci, mask]
            low, high = boot_low_high[0, ci, mask], boot_low_high[1, ci, mask]
            rows.append({
                "dataset": item["name"],
                "scenario": scenario_name,
                "clip_policy": "exclude" if exclude_clipped else "impute",
                "sqi": "on" if use_sqi else "off",
                "channel": channel,
                "window": window_name,
                "n_left": int(left_idx.sum()),
                "n_right": int(right_idx.sum()),
                "left_right_difference_rms": float(np.sqrt(np.mean(curve ** 2))),
                "left_right_difference_mean": float(np.mean(curve)),
                "max_abs_difference": float(np.max(np.abs(curve))),
                "max_abs_difference_time_ms": float(item["output_time_ms"][mask][np.argmax(np.abs(curve))]),
                "bootstrap_pointwise_ci_excludes_zero_pct": float(
                    np.mean((low > 0) | (high < 0))
                ),
                "imputed_sample_fraction": (
                    float(item["imputed_mask"][keep][:, DISPLAY_ORDER, :][
                        :, ci, :
                    ][:, window_mask(item["time_s"] * 1000, window)].mean())
                    if not exclude_clipped else 0.0
                ),
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "interpretation": "descriptive_pointwise_CI_not_familywise_test",
            })

    return {
        "name": item["name"],
        "scenario": scenario_name,
        "eeg": baseline_eeg,
        "left_erp": left_erp,
        "right_erp": right_erp,
        "lr": lr,
        "keep": keep,
        "n_left": int(left_idx.sum()),
        "n_right": int(right_idx.sum()),
        "sqi_threshold": threshold,
        "sqi_drop_n": sqi_drop_n,
        "rows": rows,
    }


def plot_waveforms(items, results, out_path, window):
    fig, axes = plt.subplots(len(items), len(CHANNELS), figsize=(15, 11), sharex=True, squeeze=False)
    first = next(iter(results.values()))
    for row, item in enumerate(items):
        for col, (channel, _) in enumerate(CHANNELS):
            ax = axes[row, col]
            for scenario_name, _, _ in SCENARIOS:
                result = results[(item["name"], scenario_name)]
                mask = window_mask(item["output_time_ms"], window)
                ax.plot(
                    item["output_time_ms"][mask], result["lr"][col, mask],
                    color=COLORS[scenario_name], lw=1.3,
                    label=PLOT_LABELS[scenario_name],
                )
            ax.axhline(0, color="0.35", lw=0.8)
            if window[0] <= 0 <= window[1]:
                ax.axvline(0, color="0.4", lw=0.8)
            if window[0] <= 200 <= window[1]:
                ax.axvline(200, color="0.5", lw=0.8, ls=":")
            ax.set_title(f"{item['name']} · {channel}")
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("右 − 左 ERP（输入单位）")
            if row == len(items) - 1:
                ax.set_xlabel("相对提示时间（ms）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        f"削顶处理与 SQI 对左右 ERP 差异的影响（{window[0]:g}–{window[1]:g} ms）",
        y=1.015, fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(items, summaries, metrics, result_dir):
    metric_frame = pd.DataFrame(metrics)
    summary_lines = []
    for item in items:
        clip_percent = 100 * item["clipped_samples"] / item["raw"].size
        summary_lines.append(
            f"| {item['name']} | {int(item['clip_flag'].sum())}/100 | {item['run_count']} | "
            f"{item['run_gt_50ms']} | {item['max_run_ms']:.1f} | {clip_percent:.3f}% |"
        )

    full = metric_frame[metric_frame["window"] == "cue_full_0_800ms"]
    effect_lines = []
    for name in DATASETS:
        for scenario_name, _, _ in SCENARIOS:
            subset = full[(full.dataset == name) & (full.scenario == scenario_name)]
            vals = "/".join(
                f"{subset.loc[subset.channel == channel, 'left_right_difference_rms'].iloc[0]:.2f}"
                for channel, _ in CHANNELS
            )
            ci = "/".join(
                f"{subset.loc[subset.channel == channel, 'bootstrap_pointwise_ci_excludes_zero_pct'].iloc[0]:.0%}"
                for channel, _ in CHANNELS
            )
            result = summaries[(name, scenario_name)]
            effect_lines.append(
                f"| {name} | {scenario_name} | {result['n_left']}/{result['n_right']} | "
                f"{vals} | {ci} |"
            )

    text = [
        "# 削顶插补 × SQI 四组合敏感性分析",
        "",
        "> 本报告是探索性敏感性分析。饱和区真实 EEG 峰值已被采集系统截断，以下插补不能恢复真实观测，也不能作为正式 ERP 数据。",
        "",
        "## 四种组合",
        "",
        "| 组合 | 削顶试次 | SQI |",
        "|---|---|---|",
        "| 剔除削顶_不做SQI | 整条试次剔除 | 否 |",
        "| 剔除削顶_执行SQI | 整条试次剔除 | 是；该组合复现第一问正式 SQI 掩码 |",
        "| 插补削顶_不做SQI | 样本级插补后保留 | 否 |",
        "| 插补削顶_执行SQI | 样本级插补后保留 | 是；在插补后的全部试次上重新计算 SQI |",
        "",
        "## 插补边界与数据质量",
        "",
        "饱和判据沿用第一问 `abs(EEG) >= 999`。连续削顶段不超过 50 ms 时用局部 PCHIP；更长的段用两侧有效样本线性桥接，若削顶碰到试次边界则使用最近有效值延拓。插补在原始 256 Hz 分段信号上进行，随后统一重跑第一问 0.2–24 Hz 零相位滤波和 128 Hz 降采样；所有组合都用每试次提示前 [-200, 0) ms 均值做基线校正。插补削顶组合的 SQI 在插补后的数据上重算。目标候选窗 2450–2700 ms 仍使用提示前基线，没有改为目标前 2.0–2.2 s 基线，因此只用于处理方案敏感性对照，不等同于标准目标锁时 P300 ERP。",
        "",
        "| 记录 | 含削顶试次数 | 削顶连续段数 | >50 ms 段数 | 最长段（ms） | 削顶样本占 EEG 样本比例 |",
        "|---|---:|---:|---:|---:|---:|",
        *summary_lines,
        "",
        "长削顶段不能可靠地由邻点恢复。本数据最长段远超 50 ms，因此长段线性桥接仅用于测试‘保留这些试次会怎样’，不能称为有效重建。",
        "",
        "## 0–800 ms 左右差异",
        "",
        "下表按 F3/Fz/F4 顺序列出右减左 ERP 的 RMS 差异幅度；后列是按原试次顺序分五块、条件内分层重抽样得到的逐点 95% 区间不含 0 的时间点比例。它是描述性指标，未做全时间窗多重比较校正，不等于显著性检验。",
        "",
        "| 记录 | 组合 | 左/右保留数 | 差异 RMS（F3/Fz/F4） | 逐点区间不含0比例（F3/Fz/F4） |",
        "|---|---|---:|---|---|",
        *effect_lines,
        "",
        "## 结论边界",
        "",
        "- 只有当差异在正式未削顶数据中存在、且清洗/试次数变化后仍保持相近时间形状，才可作为稳健的左右差异线索。",
        "- 若某条趋势只在长段插补后出现，说明它可能由插补假设制造，不能作为刺激效应证据。",
        "- SQI 对插补试次重算后，保留数与阈值可能改变；因此四种组合是处理方案敏感性比较，不是随机化实验。",
        "- 左右差异是否真实，还需要看跨记录方向一致性、事件顺序/试次顺序以及被试级重复；本分析不做因果结论。",
        "",
        "## 输出",
        "",
        "- `四组合波形指标.csv`：记录 × 组合 × 通道 × 时间窗的差异 RMS、峰值和逐点 bootstrap 描述。",
        "- `四组合样本量.csv`：每组合的左右样本数、SQI 阈值、SQI 剔除数。",
        "- `cue左右ERP四组合.png`：0–800 ms 左右差异曲线。",
        "- `target候选窗四组合.png`：目标后 250–500 ms 候选窗的左右差异曲线。",
        "- `补值方法说明与结论.md`：本说明。",
        "",
    ]
    (result_dir / "补值方法说明与结论.md").write_text("\n".join(text), encoding="utf-8")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sqi_module = load_local_module("8riemann_denoise.py", "q1_sqi_stage8")
    items = [load_dataset(name) for name in DATASETS]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries, metric_rows, sample_rows = {}, [], []

    for item in items:
        for scenario_name, exclude_clipped, use_sqi in SCENARIOS:
            result = analyze_scenario(
                item, scenario_name, exclude_clipped, use_sqi, sqi_module, rng
            )
            summaries[(item["name"], scenario_name)] = result
            metric_rows.extend(result["rows"])
            sample_rows.append({
                "dataset": item["name"],
                "scenario": scenario_name,
                "clip_policy": "exclude" if exclude_clipped else "impute",
                "sqi": "on" if use_sqi else "off",
                "n_left": result["n_left"],
                "n_right": result["n_right"],
                "sqi_threshold": result["sqi_threshold"],
                "sqi_dropped_within_eligible": result["sqi_drop_n"],
            })
            print(
                f"{item['name']} | {scenario_name}: left/right="
                f"{result['n_left']}/{result['n_right']}, SQI threshold="
                f"{result['sqi_threshold'] if np.isfinite(result['sqi_threshold']) else 'n/a'}",
                flush=True,
            )

    pd.DataFrame(metric_rows).to_csv(
        RESULT_DIR / "四组合波形指标.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(sample_rows).to_csv(
        RESULT_DIR / "四组合样本量.csv", index=False, encoding="utf-8-sig"
    )
    plot_waveforms(items, summaries, RESULT_DIR / "cue左右ERP四组合.png", (0, 800))
    plot_waveforms(
        items, summaries, RESULT_DIR / "target候选窗四组合.png", WINDOWS_MS["target_candidate_250_500ms"]
    )
    write_report(items, summaries, metric_rows, RESULT_DIR)
    print(f"Saved four-condition clipping/SQI sensitivity analysis to {RESULT_DIR}", flush=True)


if __name__ == "__main__":
    main()
