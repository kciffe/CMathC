# -*- coding: utf-8 -*-
"""Riemannian quality screening for the four Q1 EEG datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt


DATASETS = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "output"
INPUT_DIR = OUTPUT_ROOT / "7filter_downsample"
RESULT_DIR = OUTPUT_ROOT / "8riemann_denoise"
EEG_CHANNELS = (0, 1, 2)
EPOCH_START, EPOCH_END = -0.2, 1.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def matrix_function(matrix, function, positive=False):
    values, vectors = eigh((matrix + matrix.T) / 2)
    if positive:
        values = np.maximum(values, 1e-12)
    return (vectors * function(values)) @ vectors.T


def covariance_matrix(signals):
    centered = signals - signals.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T / (centered.shape[1] - 1)
    return covariance + 1e-6 * np.eye(centered.shape[0])


def riemann_distance(first, second):
    inverse_root = matrix_function(first, lambda x: 1 / np.sqrt(x), positive=True)
    relative = inverse_root @ second @ inverse_root
    eigenvalues = eigh((relative + relative.T) / 2, eigvals_only=True)
    return np.linalg.norm(np.log(np.maximum(eigenvalues, 1e-12)))


def riemann_mean(covariances):
    log_mean = np.mean(
        [matrix_function(cov, np.log, positive=True) for cov in covariances], axis=0
    )
    center = matrix_function(log_mean, np.exp)
    for _ in range(30):
        root = matrix_function(center, np.sqrt, positive=True)
        inverse_root = matrix_function(center, lambda x: 1 / np.sqrt(x), positive=True)
        delta = np.mean(
            [matrix_function(inverse_root @ cov @ inverse_root, np.log, positive=True)
             for cov in covariances],
            axis=0,
        )
        if np.linalg.norm(delta, "fro") < 1e-8:
            break
        center = root @ matrix_function(delta, np.exp) @ root
        center = (center + center.T) / 2
    return center


def robust_riemann_center(covariances):
    keep = np.ones(len(covariances), dtype=bool)
    for _ in range(4):
        center = riemann_mean(covariances[keep])
        distances = np.array([riemann_distance(cov, center) for cov in covariances])
        reference = distances[keep]
        median = np.median(reference)
        mad = np.median(np.abs(reference - median))
        z_score = (distances - median) / (1.4826 * mad + 1e-12)
        updated = z_score <= 3
        if np.array_equal(updated, keep):
            break
        keep = updated
    return riemann_mean(covariances[keep]), keep


def bandpass(data, sample_rate, low_hz, high_hz):
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def positive_robust_z(feature, inliers):
    reference = feature[inliers]
    median = np.median(reference)
    mad = np.median(np.abs(reference - median))
    z_score = (feature - median) / (1.4826 * mad + 1e-12)
    return np.maximum(0, np.clip(z_score, -3, 6))


def sqi_knee_threshold(sqi):
    ordered = np.sort(sqi)
    if len(ordered) < 2:
        raise ValueError("At least two non-clipped trials are required to estimate SQI threshold")
    search_end = min(len(ordered) - 1, max(3, int(len(ordered) * 0.4)))
    return ordered[np.argmax(np.diff(ordered[:search_end + 1])) + 1]


def feature_specs(is_group_a):
    if is_group_a:
        return [
            ("RP1_F3F4_1_7_Frobenius", "low", (1, 2), "frobenius"),
            ("RP2_Fz_1_7_Variance", "low", (0,), "variance"),
            ("RP3_All_16_24_Trace", "high", (0, 1, 2), "trace"),
            ("RP4_All_1_24_Riemann", "full", (0, 1, 2), "riemann"),
        ]
    return [
        ("RP1_All_1_7_Riemann", "low", (0, 1, 2), "riemann"),
        ("RP2_F3F4_16_24_Trace", "high", (1, 2), "trace"),
        ("RP3_All_1_24_Riemann", "full", (0, 1, 2), "riemann"),
    ]


def calculate_sqi(data, sample_rate, time, valid_indices, is_group_a):
    epoch = (time >= EPOCH_START) & (time <= EPOCH_END)
    if not epoch.any():
        raise ValueError(f"No samples in the analysis window {EPOCH_START}–{EPOCH_END} s")

    bands = {
        "low": bandpass(data, sample_rate, 1, 7),
        "high": bandpass(data, sample_rate, 16, 24),
        "full": data,
    }
    feature_names, feature_values, normalized = [], [], []

    for feature_name, band, channels, metric in feature_specs(is_group_a):
        channel_indices = np.asarray(channels)
        covariances = np.stack([
            covariance_matrix(bands[band][trial, channel_indices][:, epoch])
            for trial in valid_indices
        ])
        center, inliers = robust_riemann_center(covariances)

        if metric == "frobenius":
            values = np.linalg.norm(covariances, axis=(1, 2))
        elif metric == "variance":
            values = covariances[:, 0, 0]
        elif metric == "trace":
            values = np.trace(covariances, axis1=1, axis2=2)
        else:
            values = np.array([riemann_distance(cov, center) for cov in covariances])

        feature_names.append(feature_name)
        feature_values.append(values)
        normalized.append(positive_robust_z(values, inliers))

    features = np.column_stack(feature_values)
    anomaly_score = np.mean(normalized, axis=0)
    sqi_valid = np.exp(-anomaly_score)
    knee = sqi_knee_threshold(sqi_valid)
    threshold = knee if is_group_a else max(0.05, 0.6 * knee)
    return feature_names, features, sqi_valid, threshold


def load_dataset(dataset_name):
    path = INPUT_DIR / f"{dataset_name}_filtered_downsample.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing input MAT: {path}")

    source = loadmat(path)
    required = {"trial_data", "relative_time", "cue_type", "drop", "SampleRate", "DataLabel"}
    missing = required - source.keys()
    if missing:
        raise KeyError(f"{path.name} is missing fields: {sorted(missing)}")

    trials = np.asarray(source["trial_data"], dtype=float)
    relative_time = np.asarray(source["relative_time"], dtype=float)
    cue_type = np.asarray(source["cue_type"]).reshape(-1)
    drop_values = np.asarray(source["drop"]).reshape(-1)
    sample_rate = int(np.asarray(source["SampleRate"]).squeeze())

    if trials.ndim != 3 or trials.shape[1] < 3:
        raise ValueError(f"Expected trials x channels x samples, got {trials.shape}")
    if relative_time.shape != (trials.shape[0], trials.shape[2]):
        raise ValueError(f"relative_time shape {relative_time.shape} does not match {trials.shape}")
    if cue_type.size != trials.shape[0] or drop_values.size != trials.shape[0]:
        raise ValueError(f"cue_type/drop length does not match {trials.shape[0]} trials")

    return source, trials, relative_time, cue_type, drop_values, sample_rate


def save_diagnostic_plots(dataset_name, sqi_valid, threshold, sqi, valid_indices, riemann_drop):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = np.sort(sqi_valid)
    ax.plot(np.arange(len(ordered)), ordered, marker="o", markersize=3)
    ax.axhline(threshold, linestyle="--", label=f"阈值 = {threshold:.4f}")
    ax.set(xlabel="排序后的 Trial 编号", ylabel="SQI", title=f"{dataset_name} SQI 排序与阈值")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{dataset_name}_01_SQI排序与阈值.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    keep = ~riemann_drop[valid_indices]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(valid_indices[keep], sqi[valid_indices[keep]], s=30, label="保留 Trial")
    ax.scatter(valid_indices[~keep], sqi[valid_indices[~keep]], s=35, label="黎曼剔除 Trial")
    ax.axhline(threshold, linestyle="--", label=f"阈值 = {threshold:.4f}")
    ax.set(xlabel="原始 Trial 编号", ylabel="SQI", title=f"{dataset_name} 每个 Trial 的 SQI")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{dataset_name}_02_各Trial_SQI.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_dataset(dataset_name):
    source, trials, relative_time, cue_type, drop_values, sample_rate = load_dataset(dataset_name)
    eeg = trials[:, EEG_CHANNELS, :]
    time = relative_time[0]
    valid_indices = np.flatnonzero(~drop_values.astype(bool))
    if not len(valid_indices):
        raise ValueError(f"{dataset_name} has no trials left after clipped-trial exclusion")

    feature_names, features, sqi_valid, threshold = calculate_sqi(
        eeg, sample_rate, time, valid_indices, dataset_name.startswith("VisualCogA")
    )
    sqi = np.full(len(trials), np.nan)
    sqi[valid_indices] = sqi_valid
    riemann_drop = np.zeros(len(trials), dtype=bool)
    riemann_drop[valid_indices] = sqi_valid < threshold
    final_drop = drop_values.astype(bool) | riemann_drop
    clean_indices = np.flatnonzero(~final_drop)

    metrics = pd.DataFrame({
        "Trial": np.arange(len(trials)),
        "CueType": cue_type,
        "ClippedDrop": drop_values.astype(int),
        "SQI": sqi,
        "RiemannDrop": riemann_drop.astype(int),
        "FinalDrop": final_drop.astype(int),
    })
    for column, feature_name in enumerate(feature_names):
        values = np.full(len(trials), np.nan)
        values[valid_indices] = features[:, column]
        metrics[feature_name] = values

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULT_DIR / f"{dataset_name}_SQI指标.csv", index=False, encoding="utf-8-sig")

    # Keep the input schema and all channels; only replace trial-aligned fields.
    clean_mat = {key: value for key, value in source.items() if not key.startswith("__")}
    clean_mat.update({
        "trial_data": trials[clean_indices],
        "relative_time": relative_time[clean_indices],
        "cue_type": cue_type[clean_indices],
        "drop": drop_values[clean_indices],
    })
    savemat(RESULT_DIR / f"{dataset_name}_clean.mat", clean_mat, do_compression=True)
    save_diagnostic_plots(dataset_name, sqi_valid, threshold, sqi, valid_indices, riemann_drop)

    print(
        f"{dataset_name}: trials={len(trials)}, clipped={int(drop_values.sum())}, "
        f"riemann_drop={int(riemann_drop.sum())}, kept={len(clean_indices)}, "
        f"threshold={threshold:.6f}"
    )


def main():
    for dataset_name in DATASETS:
        process_dataset(dataset_name)


if __name__ == "__main__":
    main()
