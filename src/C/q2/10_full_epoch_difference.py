# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent / "q1" / "output" / "8riemann_denoise"
OUTPUT_DIR = SCRIPT_DIR / "output" / "10_full_epoch_difference"

DATASETS = {
    "Task1": [
        "VisualCogA_Task-1",
        "VisualCogA_Task-2",
    ],
    "Task2": [
        "VisualCogB_Task-1",
        "VisualCogB_Task-2",
    ],
}

BASELINE = (-0.2, 0.0)
ANALYSIS_START = 0.0
ANALYSIS_END = 2.20  # 下一次目标信息约在2.20 s出现，因此不包含该时刻
P300_WINDOW_MS = (250, 500)

N_PERMUTATIONS = 1000
CLUSTER_ALPHA = 0.05
RANDOM_STATE = 2026

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def matlab_labels(raw):
    arr = np.asarray(raw).squeeze()
    labels = []

    for item in np.atleast_1d(arr):
        if isinstance(item, bytes):
            labels.append(item.decode("utf-8"))
        elif isinstance(item, np.ndarray):
            value = item.squeeze()
            if value.ndim == 0:
                labels.append(str(value.item()))
            else:
                labels.append("".join(str(x) for x in value.tolist()).strip())
        else:
            labels.append(str(item).strip())

    return labels


def channel_indices(mat):
    labels = matlab_labels(mat["DataLabel"])
    return [
        labels.index("F3"),
        labels.index("Fz"),
        labels.index("F4"),
    ]


def baseline_correct(eeg, time):
    mask = (time >= BASELINE[0]) & (time < BASELINE[1])
    baseline = np.nanmean(eeg[:, :, mask], axis=2, keepdims=True)
    return eeg - baseline


def student_t_time(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    n1 = left.shape[0]
    n2 = right.shape[0]

    mean1 = np.mean(left, axis=0)
    mean2 = np.mean(right, axis=0)
    var1 = np.var(left, axis=0, ddof=1)
    var2 = np.var(right, axis=0, ddof=1)

    pooled_var = (
        (n1 - 1) * var1
        + (n2 - 1) * var2
    ) / (n1 + n2 - 2)

    se = np.sqrt(
        pooled_var
        * (1.0 / n1 + 1.0 / n2)
    )

    return np.divide(
        mean1 - mean2,
        se,
        out=np.zeros_like(mean1),
        where=se > 1e-12,
    )


def cohen_d_time(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    n1 = left.shape[0]
    n2 = right.shape[0]

    var1 = np.var(left, axis=0, ddof=1)
    var2 = np.var(right, axis=0, ddof=1)

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * var1
            + (n2 - 1) * var2
        ) / (n1 + n2 - 2)
    )

    diff = np.mean(left, axis=0) - np.mean(right, axis=0)

    return np.divide(
        diff,
        pooled_sd,
        out=np.zeros_like(diff),
        where=pooled_sd > 1e-12,
    )


def t_critical(n_left, n_right, alpha=0.05):
    df = n_left + n_right - 2
    return float(student_t.ppf(1.0 - alpha / 2.0, df))


def find_clusters(t_stat, critical):
    clusters = []
    start = None
    sign = 0

    for i, value in enumerate(np.r_[t_stat, 0.0]):
        current_sign = 1 if value >= critical else -1 if value <= -critical else 0

        if current_sign != 0 and start is None:
            start = i
            sign = current_sign

        elif start is not None and current_sign != sign:
            end = i - 1
            clusters.append({
                "start": start,
                "end": end,
                "sign": sign,
                "mass": float(np.sum(np.abs(t_stat[start:end + 1]))),
            })

            if current_sign != 0:
                start = i
                sign = current_sign
            else:
                start = None
                sign = 0

    return clusters


def load_record(dataset):
    mat = loadmat(INPUT_DIR / f"{dataset}_clean.mat")

    trials = np.asarray(mat["trial_data"], dtype=float)
    relative_time = np.asarray(mat["relative_time"], dtype=float)
    cue_type = np.asarray(mat["cue_type"]).reshape(-1).astype(int)

    time = relative_time[0]
    indices = channel_indices(mat)

    eeg = trials[:, indices, :]
    eeg = baseline_correct(eeg, time)

    mask = (time >= ANALYSIS_START) & (time < ANALYSIS_END)

    eeg = eeg[:, :, mask]
    time = time[mask]

    valid = np.isin(cue_type, [-1, 1])

    return (
        eeg[valid],
        cue_type[valid],
        time,
    )


def cluster_permutation(eeg, labels, time, task, dataset):
    left = eeg[labels == -1]
    right = eeg[labels == 1]

    critical = t_critical(
        len(left),
        len(right),
        CLUSTER_ALPHA,
    )

    observed = []

    for ch, channel in enumerate(["F3", "Fz", "F4"]):
        t_stat = student_t_time(left[:, ch], right[:, ch])
        d_stat = cohen_d_time(left[:, ch], right[:, ch])
        diff = left[:, ch].mean(axis=0) - right[:, ch].mean(axis=0)

        for cluster_id, cluster in enumerate(
            find_clusters(t_stat, critical),
            start=1,
        ):
            start = cluster["start"]
            end = cluster["end"]
            local_d = d_stat[start:end + 1]
            peak_local = int(np.argmax(np.abs(local_d)))
            peak_index = start + peak_local

            observed.append({
                "Task": task,
                "Dataset": dataset,
                "Channel": channel,
                "ClusterID": cluster_id,
                "Start_ms": time[start] * 1000.0,
                "End_ms": time[end] * 1000.0,
                "Duration_ms": (time[end] - time[start]) * 1000.0,
                "Direction": "左>右" if cluster["sign"] > 0 else "左<右",
                "ClusterMass": cluster["mass"],
                "MeanDifference": float(np.mean(diff[start:end + 1])),
                "PeakAbsCohens_d": float(np.abs(d_stat[peak_index])),
                "PeakDTime_ms": time[peak_index] * 1000.0,
            })

    rng = np.random.default_rng(RANDOM_STATE)
    n_left = int(np.sum(labels == -1))
    all_indices = np.arange(len(labels))
    max_null_mass = np.zeros(N_PERMUTATIONS, dtype=float)

    for p in range(N_PERMUTATIONS):
        perm = rng.permutation(all_indices)
        left_idx = perm[:n_left]
        right_idx = perm[n_left:]

        max_mass = 0.0

        for ch in range(3):
            t_perm = student_t_time(
                eeg[left_idx, ch],
                eeg[right_idx, ch],
            )

            clusters = find_clusters(
                t_perm,
                critical,
            )

            if clusters:
                max_mass = max(
                    max_mass,
                    max(cluster["mass"] for cluster in clusters),
                )

        max_null_mass[p] = max_mass

    for row in observed:
        row["ClusterP_corrected"] = float(
            (
                1
                + np.sum(max_null_mass >= row["ClusterMass"])
            )
            / (N_PERMUTATIONS + 1)
        )
        row["Significant_0.05"] = int(
            row["ClusterP_corrected"] < 0.05
        )
        row["N_Left"] = int(np.sum(labels == -1))
        row["N_Right"] = int(np.sum(labels == 1))
        row["N_Permutations"] = N_PERMUTATIONS

    return observed


def prepare_all_records():
    records = []
    cluster_rows = []

    for task, datasets in DATASETS.items():
        for dataset in datasets:
            eeg, labels, time = load_record(dataset)

            records.append({
                "Task": task,
                "Dataset": dataset,
                "eeg": eeg,
                "labels": labels,
                "time": time,
            })

            cluster_rows.extend(
                cluster_permutation(
                    eeg,
                    labels,
                    time,
                    task,
                    dataset,
                )
            )

    return records, pd.DataFrame(cluster_rows)


def plot_full_erp(records):
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(16, 14),
        sharex=True,
    )

    channels = ["F3", "Fz", "F4"]

    for row, record in enumerate(records):
        eeg = record["eeg"]
        labels = record["labels"]
        time_ms = record["time"] * 1000.0

        left = eeg[labels == -1]
        right = eeg[labels == 1]

        for ch, channel in enumerate(channels):
            ax = axes[row, ch]

            left_mean = left[:, ch].mean(axis=0)
            right_mean = right[:, ch].mean(axis=0)
            left_sem = left[:, ch].std(axis=0, ddof=1) / np.sqrt(len(left))
            right_sem = right[:, ch].std(axis=0, ddof=1) / np.sqrt(len(right))

            line_left = ax.plot(
                time_ms,
                left_mean,
                label="左刺激 ERP",
                linewidth=1.5,
            )[0]
            line_right = ax.plot(
                time_ms,
                right_mean,
                label="右刺激 ERP",
                linewidth=1.5,
            )[0]

            ax.fill_between(
                time_ms,
                left_mean - left_sem,
                left_mean + left_sem,
                alpha=0.12,
            )
            ax.fill_between(
                time_ms,
                right_mean - right_sem,
                right_mean + right_sem,
                alpha=0.12,
            )

            ax.axvspan(
                P300_WINDOW_MS[0],
                P300_WINDOW_MS[1],
                alpha=0.07,
            )
            ax.axhline(0, linewidth=0.8)
            ax.grid(alpha=0.2)
            ax.set_title(
                f"{record['Dataset']} - {channel}"
            )

            if ch == 0:
                ax.set_ylabel("基线校正电位")

            if row == 3:
                ax.set_xlabel("相对三角刺激时间 (ms)")

            if row == 0 and ch == 0:
                ax.legend(
                    handles=[line_left, line_right],
                    loc="best",
                )

    fig.suptitle(
        "三角刺激出现后至下一事件前：左/右刺激全时程 ERP"
    )
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "左右ERP全时程对比.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_difference_effect(records, cluster_df):
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(16, 14),
        sharex=True,
    )

    channels = ["F3", "Fz", "F4"]

    for row, record in enumerate(records):
        eeg = record["eeg"]
        labels = record["labels"]
        time_ms = record["time"] * 1000.0

        left = eeg[labels == -1]
        right = eeg[labels == 1]

        for ch, channel in enumerate(channels):
            ax = axes[row, ch]
            ax2 = ax.twinx()

            diff = (
                left[:, ch].mean(axis=0)
                - right[:, ch].mean(axis=0)
            )
            d_stat = cohen_d_time(
                left[:, ch],
                right[:, ch],
            )

            line_diff = ax.plot(
                time_ms,
                diff,
                linewidth=1.4,
                label="ERP差（左-右）",
            )[0]
            line_d = ax2.plot(
                time_ms,
                d_stat,
                linestyle="--",
                linewidth=1.2,
                label="Cohen's d",
            )[0]

            ax.axvspan(
                P300_WINDOW_MS[0],
                P300_WINDOW_MS[1],
                alpha=0.06,
            )

            significant = cluster_df[
                (cluster_df["Dataset"] == record["Dataset"])
                & (cluster_df["Channel"] == channel)
                & (cluster_df["Significant_0.05"] == 1)
            ]

            for _, cluster in significant.iterrows():
                ax.axvspan(
                    cluster["Start_ms"],
                    cluster["End_ms"],
                    alpha=0.16,
                )

            ax.axhline(0, linewidth=0.8)
            ax2.axhline(0, linewidth=0.6, linestyle=":")
            ax.grid(alpha=0.2)

            ax.set_title(
                f"{record['Dataset']} - {channel}"
            )

            if ch == 0:
                ax.set_ylabel("ERP差")

            if ch == 2:
                ax2.set_ylabel("Cohen's d")

            if row == 3:
                ax.set_xlabel("相对三角刺激时间 (ms)")

            if row == 0 and ch == 0:
                ax.legend(
                    handles=[line_diff, line_d],
                    loc="best",
                )

    fig.suptitle(
        "左/右刺激全时程差异：虚线为效应量，深色时间段为显著 cluster"
    )
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "左右差异及效应量.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records, cluster_df = prepare_all_records()

    columns = [
        "Task",
        "Dataset",
        "Channel",
        "ClusterID",
        "Start_ms",
        "End_ms",
        "Duration_ms",
        "Direction",
        "ClusterMass",
        "ClusterP_corrected",
        "Significant_0.05",
        "MeanDifference",
        "PeakAbsCohens_d",
        "PeakDTime_ms",
        "N_Left",
        "N_Right",
        "N_Permutations",
    ]

    if len(cluster_df):
        cluster_df = cluster_df[columns]
    else:
        cluster_df = pd.DataFrame(columns=columns)

    cluster_df.to_csv(
        OUTPUT_DIR / "cluster置换检验.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_full_erp(records)
    plot_difference_effect(records, cluster_df)

    significant = cluster_df[
        cluster_df["Significant_0.05"] == 1
    ]

    print("10 全时程左右差异诊断完成：", OUTPUT_DIR)
    print("分析区间：0 ms 到下一事件 2200 ms 之前")
    print("置换次数：", N_PERMUTATIONS)
    print("显著 cluster 数：", len(significant))

    if len(significant):
        print()
        print(
            significant[
                [
                    "Dataset",
                    "Channel",
                    "Start_ms",
                    "End_ms",
                    "Direction",
                    "ClusterP_corrected",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
