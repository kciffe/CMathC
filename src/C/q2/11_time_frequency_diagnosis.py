# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import ttest_ind


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent / "q1" / "output" / "8riemann_denoise"
OUTPUT_DIR = SCRIPT_DIR / "output" / "11_time_frequency_diagnosis"

DATASETS = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]

# clean.mat 前三通道顺序是 Fz、F3、F4，这里统一成 F3、Fz、F4
CHANNELS = ["F3", "Fz", "F4"]
CHANNEL_INDEX = [1, 0, 2]

BASELINE = (-0.2, 0.0)
ANALYSIS_WINDOW = (0.0, 2.2)
P300_WINDOW = (0.25, 0.50)

# 第一问已经做 1-24 Hz 滤波，因此最高只分析到 24 Hz
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 24.0),
}

PLOT_SMOOTH_MS = 100
POINT_P = 0.05
N_PERMUTATIONS = 1000
RANDOM_STATE = 2026

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def band_power_envelope(trials, fs, low, high):
    """带通 + Hilbert，得到每个 Trial 未平滑的瞬时功率。"""
    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        trials,
        axis=-1,
    )

    power = np.abs(
        hilbert(filtered, axis=-1)
    ) ** 2

    return power


def baseline_db(power, time):
    """每个 Trial、每个通道用刺激前基线转换为 dB 功率变化。"""
    baseline = (
        (time >= BASELINE[0])
        & (time < BASELINE[1])
    )

    base = power[..., baseline].mean(
        axis=-1,
        keepdims=True,
    )

    return 10.0 * np.log10(
        (power + 1e-12)
        / (base + 1e-12)
    )


def cohen_d_time(left, right):
    """逐时间点 Cohen's d，正值表示左刺激更大。"""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    n1 = len(left)
    n2 = len(right)

    mean_diff = left.mean(axis=0) - right.mean(axis=0)
    var1 = left.var(axis=0, ddof=1)
    var2 = right.var(axis=0, ddof=1)

    pooled = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2)
        / (n1 + n2 - 2)
    )

    return np.divide(
        mean_diff,
        pooled,
        out=np.zeros_like(mean_diff),
        where=pooled > 1e-12,
    )


def find_clusters(stat, significant):
    """寻找同号、连续的显著时间点，并计算 cluster mass。"""
    clusters = []
    start = None
    sign = 0

    for i in range(len(stat) + 1):
        if i < len(stat):
            current_sign = np.sign(stat[i]) if significant[i] else 0
        else:
            current_sign = 0

        if current_sign != 0 and start is None:
            start = i
            sign = current_sign

        elif start is not None and current_sign != sign:
            end = i - 1
            mass = float(np.sum(np.abs(stat[start:end + 1])))
            clusters.append((start, end, mass, int(sign)))

            if current_sign != 0:
                start = i
                sign = current_sign
            else:
                start = None
                sign = 0

    return clusters


def t_stat_and_clusters(left, right):
    t_stat, p_value = ttest_ind(
        left,
        right,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )

    t_stat = np.nan_to_num(t_stat)
    p_value = np.nan_to_num(p_value, nan=1.0)
    significant = p_value < POINT_P

    return t_stat, find_clusters(t_stat, significant)


def load_dataset(dataset):
    mat = loadmat(
        INPUT_DIR / f"{dataset}_clean.mat"
    )

    trials = np.asarray(
        mat["trial_data"],
        dtype=float,
    )[:, CHANNEL_INDEX, :]

    times = np.asarray(
        mat["relative_time"],
        dtype=float,
    )

    labels = np.asarray(
        mat["cue_type"]
    ).reshape(-1).astype(int)

    fs = int(
        np.asarray(
            mat["SampleRate"]
        ).squeeze()
    )

    return trials, times[0], labels, fs


def prepare_band_data(trials, time, fs):
    """返回未时间平滑的单 Trial dB 功率，供统计检验使用。"""
    band_data = {}

    for band, (low, high) in BANDS.items():
        power = band_power_envelope(
            trials,
            fs,
            low,
            high,
        )

        power_db = baseline_db(
            power,
            time,
        )

        band_data[band] = power_db

    return band_data


def permutation_max_masses(band_data, labels):
    rng = np.random.default_rng(RANDOM_STATE)
    max_masses = np.zeros(N_PERMUTATIONS, dtype=float)

    for p in range(N_PERMUTATIONS):
        perm = rng.permutation(labels)
        left_mask = perm == -1
        right_mask = perm == 1
        current_max = 0.0

        for band in BANDS:
            data = band_data[band]

            for ch in range(3):
                _, clusters = t_stat_and_clusters(
                    data[left_mask, ch],
                    data[right_mask, ch],
                )

                if clusters:
                    current_max = max(
                        current_max,
                        max(c[2] for c in clusters),
                    )

        max_masses[p] = current_max

    return max_masses


def analyze_dataset(dataset):
    trials, time, labels, fs = load_dataset(dataset)

    analysis = (
        (time >= ANALYSIS_WINDOW[0])
        & (time < ANALYSIS_WINDOW[1])
    )

    analysis_time = time[analysis]

    band_data_full = prepare_band_data(
        trials,
        time,
        fs,
    )

    band_data = {
        band: data[..., analysis]
        for band, data in band_data_full.items()
    }

    left_mask = labels == -1
    right_mask = labels == 1

    null_max = permutation_max_masses(
        band_data,
        labels,
    )

    cluster_rows = []
    summary_rows = []
    effect = {}

    for band in BANDS:
        effect[band] = {}

        for ch, channel in enumerate(CHANNELS):
            left = band_data[band][left_mask, ch]
            right = band_data[band][right_mask, ch]

            d = cohen_d_time(
                left,
                right,
            )
            effect[band][channel] = d

            t_stat, clusters = t_stat_and_clusters(
                left,
                right,
            )

            min_cluster_p = 1.0
            significant_count = 0

            for start, end, mass, sign in clusters:
                corrected_p = (
                    1
                    + np.sum(null_max >= mass)
                ) / (
                    N_PERMUTATIONS + 1
                )

                min_cluster_p = min(
                    min_cluster_p,
                    corrected_p,
                )

                if corrected_p < 0.05:
                    significant_count += 1

                segment_d = d[start:end + 1]
                local_index = int(
                    np.argmax(np.abs(segment_d))
                )
                peak_index = start + local_index

                cluster_rows.append({
                    "Dataset": dataset,
                    "Band": band,
                    "Channel": channel,
                    "Start_ms": float(analysis_time[start] * 1000),
                    "End_ms": float(analysis_time[end] * 1000),
                    "Duration_ms": float((end - start + 1) / fs * 1000),
                    "Direction": "左>右" if sign > 0 else "左<右",
                    "ClusterMass": mass,
                    "CorrectedP": corrected_p,
                    "Significant": int(corrected_p < 0.05),
                    "PeakAbsD": float(abs(d[peak_index])),
                    "PeakD": float(d[peak_index]),
                    "PeakTime_ms": float(analysis_time[peak_index] * 1000),
                })

            peak_index = int(
                np.argmax(np.abs(d))
            )

            summary_rows.append({
                "Dataset": dataset,
                "Band": band,
                "Channel": channel,
                "LeftTrials": int(left_mask.sum()),
                "RightTrials": int(right_mask.sum()),
                "MaxAbsD": float(abs(d[peak_index])),
                "D_at_Max": float(d[peak_index]),
                "Time_at_Max_ms": float(analysis_time[peak_index] * 1000),
                "MinCorrectedClusterP": float(min_cluster_p),
                "SignificantClusterCount": significant_count,
            })

    return analysis_time, effect, cluster_rows, summary_rows


def plot_effects(results, significant_clusters):
    fig, axes = plt.subplots(
        len(DATASETS),
        len(BANDS),
        figsize=(16, 13),
        sharex=True,
    )

    for row, dataset in enumerate(DATASETS):
        time, effect = results[dataset]
        time_ms = time * 1000
        fs = 1.0 / np.median(np.diff(time))
        smooth_samples = max(
            1,
            int(round(PLOT_SMOOTH_MS / 1000 * fs)),
        )

        for col, band in enumerate(BANDS):
            ax = axes[row, col]

            for channel in CHANNELS:
                # 平滑仅用于显示；置换、簇检验与汇总效应量都使用原始 dB 功率。
                effect_for_plot = uniform_filter1d(
                    effect[band][channel],
                    size=smooth_samples,
                    mode="nearest",
                )
                ax.plot(
                    time_ms,
                    effect_for_plot,
                    label=channel,
                    linewidth=1.2,
                )

            ax.axhline(0, linewidth=0.8)
            ax.axvspan(
                P300_WINDOW[0] * 1000,
                P300_WINDOW[1] * 1000,
                alpha=0.08,
                label="P300参考窗" if row == 0 and col == 0 else None,
            )

            clusters = significant_clusters[
                (significant_clusters["Dataset"] == dataset)
                & (significant_clusters["Band"] == band)
                & (significant_clusters["Significant"] == 1)
            ]

            for _, cluster in clusters.iterrows():
                ax.axvspan(
                    cluster["Start_ms"],
                    cluster["End_ms"],
                    alpha=0.15,
                )

            ax.set_title(f"{dataset} - {band}")
            ax.set_ylabel("Cohen's d（左 - 右）")
            ax.grid(alpha=0.25)

            if row == len(DATASETS) - 1:
                ax.set_xlabel("相对三角刺激时间 (ms)")

            if row == 0 and col == 0:
                ax.legend(ncol=2, fontsize=8)

    fig.suptitle(
        "全时程非相位锁定频带功率差异：theta / alpha / beta"
    )
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "时频效应量全时程.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_clusters = []
    all_summary = []
    results = {}

    for dataset in DATASETS:
        print("处理：", dataset)

        time, effect, clusters, summary = analyze_dataset(
            dataset
        )

        results[dataset] = (time, effect)
        all_clusters.extend(clusters)
        all_summary.extend(summary)

    cluster_df = pd.DataFrame(all_clusters)
    summary_df = pd.DataFrame(all_summary)

    cluster_df.to_csv(
        OUTPUT_DIR / "时频cluster置换检验.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        OUTPUT_DIR / "时频诊断汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_effects(
        results,
        cluster_df,
    )

    print()
    print("11 时频诊断完成：", OUTPUT_DIR)

    significant = cluster_df[
        cluster_df["Significant"] == 1
    ]

    if len(significant) == 0:
        print("没有通过全时程×三通道×三频带校正的显著 cluster。")
    else:
        print("显著 cluster：")
        print(
            significant[
                [
                    "Dataset",
                    "Band",
                    "Channel",
                    "Start_ms",
                    "End_ms",
                    "Direction",
                    "CorrectedP",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
