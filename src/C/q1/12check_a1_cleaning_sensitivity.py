# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_MAT = OUTPUT_DIR / "7filter_downsample" / "VisualCogA_Task-1_filtered_downsample.mat"
INPUT_CSV = OUTPUT_DIR / "8riemann_denoise" / "VisualCogA_Task-1_SQI指标.csv"
RESULT_DIR = OUTPUT_DIR / "12check_a1_cleaning_sensitivity"

CHANNELS = [("F3", 1), ("Fz", 0), ("F4", 2)]
BASELINE_START, BASELINE_END = -0.2, 0.0
PLOT_START, PLOT_END = -0.2, 3.0
N_REPEAT = 500
RANDOM_SEED = 2026

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_comparison_inputs(mat_path, metrics_path):
    """Load and align filtered trials with SQI decisions by the explicit Trial key."""
    source = loadmat(mat_path)
    required_mat_fields = {"trial_data", "relative_time", "cue_type", "drop"}
    missing_mat_fields = required_mat_fields - source.keys()
    if missing_mat_fields:
        raise KeyError(f"Filtered MAT is missing fields: {sorted(missing_mat_fields)}")

    trial_data = np.asarray(source["trial_data"], dtype=float)
    relative_time = np.asarray(source["relative_time"], dtype=float)
    cue_type = np.asarray(source["cue_type"]).reshape(-1)
    drop = np.asarray(source["drop"]).reshape(-1)

    if trial_data.ndim != 3 or trial_data.shape[1] < 3:
        raise ValueError(
            f"Expected trials x at least 3 channels x samples, got {trial_data.shape}"
        )
    trial_count, _, sample_count = trial_data.shape
    if relative_time.shape != (trial_count, sample_count):
        raise ValueError("relative_time shape does not match trial_data")
    if not np.isfinite(relative_time).all() or not np.allclose(
        relative_time, relative_time[:1], rtol=0.0, atol=1e-12
    ):
        raise ValueError("All trials must share the same finite relative_time axis")
    if cue_type.size != trial_count or drop.size != trial_count:
        raise ValueError("cue_type/drop length does not match trial_data")
    if not np.isin(cue_type, (-1, 1)).all():
        raise ValueError("cue_type must contain only -1 and +1")
    if not np.isin(drop, (0, 1)).all():
        raise ValueError("drop must contain only 0 and 1")

    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig")
    required_metrics = {"Trial", "CueType", "ClippedDrop", "FinalDrop"}
    missing_metrics = required_metrics - set(metrics.columns)
    if missing_metrics:
        raise ValueError(f"SQI CSV is missing columns: {sorted(missing_metrics)}")

    for column in required_metrics:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    trial_index = metrics["Trial"].to_numpy(dtype=float)
    if (
        not np.isfinite(trial_index).all()
        or not np.equal(trial_index, np.floor(trial_index)).all()
        or len(np.unique(trial_index)) != trial_count
        or not np.array_equal(np.sort(trial_index).astype(int), np.arange(trial_count))
    ):
        raise ValueError("SQI CSV Trial indices must uniquely cover every MAT trial")

    # CSV row order is not a data key. Reindex by Trial before pairing masks with MAT rows.
    metrics = metrics.set_index("Trial").loc[np.arange(trial_count)]
    csv_cue_type = metrics["CueType"].to_numpy(dtype=float)
    csv_clipped_drop = metrics["ClippedDrop"].to_numpy(dtype=float)
    final_drop = metrics["FinalDrop"].to_numpy(dtype=float)
    drop = drop.astype(bool)

    if not np.array_equal(csv_cue_type, cue_type.astype(float)):
        raise ValueError("SQI CSV CueType values do not match MAT cue_type by Trial")
    if not np.array_equal(csv_clipped_drop, drop.astype(float)):
        raise ValueError("SQI CSV ClippedDrop values do not match MAT drop by Trial")
    if not np.isin(final_drop, (0, 1)).all():
        raise ValueError("SQI CSV FinalDrop must contain only 0 and 1")
    if np.any((final_drop == 0) & drop):
        raise ValueError("SQI CSV FinalDrop cannot retain a clipped MAT trial")

    return (
        trial_data[:, :3, :],
        relative_time[0],
        cue_type,
        ~drop,
        final_drop == 0,
    )


def baseline_correct(trials, time):
    baseline = (time >= BASELINE_START) & (time < BASELINE_END)
    return trials - trials[:, :, baseline].mean(axis=2, keepdims=True)


def random_same_n_erp(trials, sample_n, repeat, rng):
    if trials.ndim != 3 or len(trials) == 0:
        raise ValueError("At least one trial is required for random same-N ERP")
    if not 1 <= sample_n <= len(trials):
        raise ValueError("sample_n must be between 1 and the available trial count")
    if repeat < 1:
        raise ValueError("repeat must be positive")

    erps = []
    for _ in range(repeat):
        index = rng.choice(len(trials), sample_n, replace=False)
        erps.append(trials[index].mean(axis=0))

    erps = np.asarray(erps)
    return (
        erps.mean(axis=0),
        np.percentile(erps, 2.5, axis=0),
        np.percentile(erps, 97.5, axis=0),
    )


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    trials, time, cue_type, clipped_keep, current_keep = load_comparison_inputs(
        INPUT_MAT, INPUT_CSV
    )
    eeg = baseline_correct(trials, time)

    plot_mask = (time >= PLOT_START) & (time <= PLOT_END)
    plot_time = time[plot_mask]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for row, (condition_name, condition_value) in enumerate([
        ("左条件", -1),
        ("右条件", 1),
    ]):
        clipped_trials = eeg[clipped_keep & (cue_type == condition_value)]
        current_trials = eeg[current_keep & (cue_type == condition_value)]

        clipped_erp = clipped_trials.mean(axis=0)
        current_erp = current_trials.mean(axis=0)

        random_mean, random_low, random_high = random_same_n_erp(
            clipped_trials,
            len(current_trials),
            N_REPEAT,
            rng,
        )

        for col, (channel_name, channel_index) in enumerate(CHANNELS):
            ax = axes[row, col]

            ax.plot(
                plot_time,
                clipped_erp[channel_index, plot_mask],
                linestyle="--",
                linewidth=1.2,
                label=f"仅削顶 n={len(clipped_trials)}",
            )
            ax.plot(
                plot_time,
                random_mean[channel_index, plot_mask],
                linewidth=1.2,
                label=f"随机同样本量均值 n={len(current_trials)}",
            )
            ax.fill_between(
                plot_time,
                random_low[channel_index, plot_mask],
                random_high[channel_index, plot_mask],
                alpha=0.15,
                label="随机同样本量逐点 95% 区间",
            )
            ax.plot(
                plot_time,
                current_erp[channel_index, plot_mask],
                linewidth=1.6,
                label=f"当前SQI清洗 n={len(current_trials)}",
            )

            ax.axvline(0, linestyle="--", linewidth=1)
            ax.set_title(f"{condition_name} · {channel_name}")
            ax.set_ylabel("基线校正后 ERP 幅值")
            ax.grid(alpha=0.22)

            if row == 1:
                ax.set_xlabel("相对提示 onset 的时间 (s)")

            if row == 0 and col == 0:
                ax.legend(fontsize=9)

            current_curve = current_erp[channel_index, plot_mask]
            clipped_curve = clipped_erp[channel_index, plot_mask]
            random_curve = random_mean[channel_index, plot_mask]
            low_curve = random_low[channel_index, plot_mask]
            high_curve = random_high[channel_index, plot_mask]

            rows.append({
                "Condition": condition_name,
                "Channel": channel_name,
                "N_ClippedOnly": len(clipped_trials),
                "N_CurrentSQI": len(current_trials),
                "RMSE_Current_vs_ClippedOnly": float(np.sqrt(np.mean((current_curve - clipped_curve) ** 2))),
                "RMSE_Current_vs_RandomSameN": float(np.sqrt(np.mean((current_curve - random_curve) ** 2))),
                "CurrentOutsideRandom95Ratio": float(np.mean((current_curve < low_curve) | (current_curve > high_curve))),
            })

    fig.suptitle(
        "VisualCogA_Task-1 清洗强度与等样本量诊断",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(
        RESULT_DIR / "VisualCogA_Task-1_清洗强度与等样本量诊断.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.DataFrame(rows).to_csv(
        RESULT_DIR / "VisualCogA_Task-1_诊断指标.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
