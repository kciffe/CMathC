# -*- coding: utf-8 -*-
"""Time-resolved, single-trial decoding of left/right cue EEG.

This is a diagnostic branch. It reads the existing Q1-cleaned MAT files and
writes only to output/11_full_epoch_temporal_decoding; it does not alter the
01-04 forward model or any source EEG files.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revision.real_data import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent / "q1" / "output" / "8riemann_denoise"
OUTPUT_DIR = SCRIPT_DIR / "output" / "11_full_epoch_temporal_decoding"

DATASETS = {
    "Task1": ["VisualCogA_Task-1", "VisualCogA_Task-2"],
    "Task2": ["VisualCogB_Task-1", "VisualCogB_Task-2"],
}

BASELINE_S = (-0.2, 0.0)
EPOCH_S = (0.0, 2.2)  # excludes the next target onset at 2.20 s
WINDOW_MS = 100.0
N_FOLDS = 5
N_PERMUTATIONS = 199
RANDOM_STATE = 20260925
PERMUTATION_BLOCKING = "permute labels within each contiguous held-out trial block"
CHANNELS = ("F3", "Fz", "F4")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_windows(eeg, time_s, window_ms=WINDOW_MS):
    """Return non-overlapping local mean+slope features and window metadata."""
    duration_s = window_ms / 1000.0
    starts = np.arange(EPOCH_S[0], EPOCH_S[1] - duration_s + 1e-9, duration_s)
    features = []
    windows = []

    for start_s in starts:
        end_s = start_s + duration_s
        mask = (time_s >= start_s - 1e-10) & (time_s < end_s - 1e-10)
        local_time = time_s[mask]
        if local_time.size < 4:
            continue

        segment = eeg[:, :, mask]
        centered_time = local_time - local_time.mean()
        slope_denominator = float(np.sum(centered_time**2))
        means = np.mean(segment, axis=2)
        slopes = np.einsum("nct,t->nc", segment, centered_time) / slope_denominator
        features.append(np.concatenate([means, slopes], axis=1))
        windows.append({
            "window_start_ms": float(start_s * 1000.0),
            "window_end_ms": float(end_s * 1000.0),
            "window_center_ms": float((start_s + end_s) * 500.0),
            "window_samples": int(local_time.size),
            "offset_relative_center_ms": float(((start_s + end_s) / 2.0 - 0.2) * 1000.0),
        })

    return features, windows


def load_record(task, dataset):
    data = load_dataset(INPUT_DIR / f"{dataset}_clean.mat")
    time_s = np.asarray(data["time_s"], dtype=float)
    eeg = np.asarray(data["eeg"], dtype=float)
    labels = np.asarray(data["cue_type"], dtype=int)

    baseline = (time_s >= BASELINE_S[0]) & (time_s < BASELINE_S[1])
    epoch = (time_s >= EPOCH_S[0]) & (time_s < EPOCH_S[1])
    if baseline.sum() < 4 or epoch.sum() < 4:
        raise ValueError(f"{dataset}: missing baseline or full cue epoch")
    if not np.all(np.isin(labels, (-1, 1))):
        raise ValueError(f"{dataset}: cue_type must contain only -1/+1")

    corrected = eeg - eeg[:, :, baseline].mean(axis=2, keepdims=True)
    corrected = corrected[:, :, epoch]
    epoch_time = time_s[epoch]
    features, windows = make_windows(corrected, epoch_time)
    if not features:
        raise ValueError(f"{dataset}: no valid time windows")

    folds = [x.astype(int) for x in np.array_split(np.arange(labels.size), N_FOLDS)]
    fold_rows = []
    for fold_id, test_idx in enumerate(folds, start=1):
        counts = {int(label): int(np.sum(labels[test_idx] == label)) for label in (-1, 1)}
        if min(counts.values()) == 0:
            raise ValueError(f"{dataset}: chronological fold {fold_id} lacks one cue class")
        fold_rows.append({
            "fold": fold_id,
            "first_trial_1based": int(test_idx[0] + 1),
            "last_trial_1based": int(test_idx[-1] + 1),
            "n_left": counts[-1],
            "n_right": counts[1],
        })

    return {
        "task": task,
        "dataset": dataset,
        "features": features,
        "windows": windows,
        "labels": labels,
        "folds": folds,
        "n_trials": int(labels.size),
        "n_left": int(np.sum(labels == -1)),
        "n_right": int(np.sum(labels == 1)),
        "sample_rate_hz": float(data["sample_rate"]),
        "fold_rows": fold_rows,
    }


def classifier():
    # All scaling is fitted separately inside each training fold.
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def auc_curve(record, labels):
    fold_auc = np.empty((len(record["features"]), len(record["folds"])), dtype=float)
    all_indices = np.arange(labels.size)

    for window_index, x in enumerate(record["features"]):
        for fold_index, test_idx in enumerate(record["folds"]):
            train_mask = np.ones(labels.size, dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_indices[train_mask]
            model = classifier()
            model.fit(x[train_idx], labels[train_idx])
            scores = model.decision_function(x[test_idx])
            fold_auc[window_index, fold_index] = roc_auc_score(labels[test_idx], scores)

    signed_auc = fold_auc.mean(axis=1)
    # Absolute deviation from chance allows either consistent polarity while
    # preserving cancellations when fold directions disagree.
    discriminability = 0.5 + np.abs(signed_auc - 0.5)
    return signed_auc, discriminability, fold_auc.std(axis=1, ddof=1)


def permute_within_folds(labels, folds, rng):
    """Keep each chronological test block's left/right count fixed."""
    permuted = labels.copy()
    for indices in folds:
        permuted[indices] = rng.permutation(labels[indices])
    return permuted


def plot_results(windows_df):
    global_95_threshold = float(windows_df["Global_null_95_threshold"].iloc[0])
    order = [dataset for datasets in DATASETS.values() for dataset in datasets]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, dataset in zip(axes, order):
        subset = windows_df.loc[windows_df["Dataset"] == dataset].sort_values("window_center_ms")
        if subset.empty:
            continue
        row = subset.iloc[0]
        centers = subset["window_center_ms"].to_numpy(dtype=float)
        scores = subset["AUC_discriminability"].to_numpy(dtype=float)
        ax.axvspan(250, 500, color="#6baed6", alpha=0.12, label="onset参照窗 250–500 ms")
        ax.axvspan(450, 700, color="#fdae6b", alpha=0.12, label="offset后参照窗 250–500 ms")
        ax.plot(centers, scores, marker="o", markersize=3, linewidth=1.5, label="交叉验证可分性")
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="机会水平")
        ax.axhline(global_95_threshold, color="#a50f15", linewidth=1.0, linestyle=":", label="全局置换95%阈值")
        sig = subset["FWER_significant_0.05"].to_numpy(dtype=bool)
        if np.any(sig):
            ax.scatter(centers[sig], scores[sig], color="red", zorder=5, label="全局校正显著")
        ax.set_title(f"{dataset}  (n={int(row['N_trials'])}, 左/右={int(row['N_left'])}/{int(row['N_right'])})")
        ax.set_xlim(0, 2200)
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.22)
        ax.set_xlabel("相对提示出现时间 (ms)")
        ax.set_ylabel("交叉验证 AUC 可分性")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("左右提示全时段单试次解码", y=0.99, fontsize=14)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.83, bottom=0.10, hspace=0.34, wspace=0.14)
    fig.savefig(OUTPUT_DIR / "全时段左右解码.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_saved_results():
    """Regenerate the figure from saved results without rerunning permutations."""
    results_path = OUTPUT_DIR / "逐窗左右解码.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing saved decoding table: {results_path}")
    plot_results(pd.read_csv(results_path, encoding="utf-8-sig"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for task, datasets in DATASETS.items():
        for dataset in datasets:
            records.append(load_record(task, dataset))

    observed = {}
    for record in records:
        signed, discrim, fold_sd = auc_curve(record, record["labels"])
        observed[record["dataset"]] = {
            "signed_auc": signed,
            "discriminability": discrim,
            "fold_auc_sd": fold_sd,
        }
        print(
            f"Observed {record['dataset']}: {record['n_trials']} trials, "
            f"{len(record['features'])} windows",
            flush=True,
        )

    rng = np.random.default_rng(RANDOM_STATE)
    null_max = np.empty(N_PERMUTATIONS, dtype=float)
    started = time.perf_counter()
    for permutation in range(N_PERMUTATIONS):
        maximum = 0.0
        for record in records:
            permuted = permute_within_folds(record["labels"], record["folds"], rng)
            _, discrim, _ = auc_curve(record, permuted)
            maximum = max(maximum, float(np.max(discrim)))
        null_max[permutation] = maximum
        if (permutation + 1) % 10 == 0 or permutation + 1 == N_PERMUTATIONS:
            elapsed = time.perf_counter() - started
            print(
                f"Permutation {permutation + 1}/{N_PERMUTATIONS}; "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )

    sorted_null = np.sort(null_max)
    critical_index = min(len(sorted_null) - 1, math.ceil(0.95 * len(sorted_null)) - 1)
    global_95_threshold = float(sorted_null[critical_index])
    window_rows = []
    summary_rows = []

    for record in records:
        dataset = record["dataset"]
        result = observed[dataset]
        for index, window in enumerate(record["windows"]):
            score = float(result["discriminability"][index])
            p_global = float((1 + np.sum(null_max >= score)) / (N_PERMUTATIONS + 1))
            window_rows.append({
                "Task": record["task"],
                "Dataset": dataset,
                **window,
                "AUC_signed_mean_folds": float(result["signed_auc"][index]),
                "AUC_discriminability": score,
                "AUC_fold_SD_descriptive": float(result["fold_auc_sd"][index]),
                "Global_permutation_p": p_global,
                "FWER_significant_0.05": int(p_global < 0.05),
                "Global_null_95_threshold": global_95_threshold,
                "N_trials": record["n_trials"],
                "N_left": record["n_left"],
                "N_right": record["n_right"],
                "N_permutations": N_PERMUTATIONS,
            })

        scores = result["discriminability"]
        best = int(np.argmax(scores))
        onset_ref = np.array([250.0 <= w["window_center_ms"] < 500.0 for w in record["windows"]])
        offset_ref = np.array([450.0 <= w["window_center_ms"] < 700.0 for w in record["windows"]])
        outside_onset = ~onset_ref
        summary_rows.append({
            "Task": record["task"],
            "Dataset": dataset,
            "N_trials": record["n_trials"],
            "N_left": record["n_left"],
            "N_right": record["n_right"],
            "best_window_start_ms": record["windows"][best]["window_start_ms"],
            "best_window_end_ms": record["windows"][best]["window_end_ms"],
            "best_window_center_ms_from_onset": record["windows"][best]["window_center_ms"],
            "best_window_center_ms_from_offset": record["windows"][best]["offset_relative_center_ms"],
            "best_signed_AUC": float(result["signed_auc"][best]),
            "best_AUC_discriminability": float(scores[best]),
            "best_window_global_p": float((1 + np.sum(null_max >= scores[best])) / (N_PERMUTATIONS + 1)),
            "max_AUC_onset_ref_250_500": float(np.max(scores[onset_ref])) if onset_ref.any() else float("nan"),
            "max_AUC_offset_ref_450_700": float(np.max(scores[offset_ref])) if offset_ref.any() else float("nan"),
            "max_AUC_outside_onset_ref": float(np.max(scores[outside_onset])) if outside_onset.any() else float("nan"),
            "any_FWER_significant_window": int(any(
                row["Dataset"] == dataset and row["FWER_significant_0.05"] == 1
                for row in window_rows
            )),
        })

    windows_df = pd.DataFrame(window_rows)
    summary_df = pd.DataFrame(summary_rows)
    windows_df.to_csv(OUTPUT_DIR / "逐窗左右解码.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "记录汇总.csv", index=False, encoding="utf-8-sig")

    plot_results(windows_df)

    manifest = {
        "input": str(INPUT_DIR),
        "output": str(OUTPUT_DIR),
        "records": [{
            "task": r["task"],
            "dataset": r["dataset"],
            "n_trials": r["n_trials"],
            "n_left": r["n_left"],
            "n_right": r["n_right"],
            "sample_rate_hz": r["sample_rate_hz"],
            "folds": r["fold_rows"],
        } for r in records],
        "channels": CHANNELS,
        "baseline_s": BASELINE_S,
        "epoch_s_exclusive_end": EPOCH_S,
        "window_ms": WINDOW_MS,
        "features_per_window": "three channel means plus three within-window linear slopes",
        "classifier": "StandardScaler + shrinkage LDA (shrinkage='auto')",
        "cross_validation": "five contiguous held-out trial blocks; transformations fit within training fold",
        "permutation": PERMUTATION_BLOCKING,
        "familywise_correction": "permutation maximum over all windows and all four records",
        "n_permutations": N_PERMUTATIONS,
        "random_state": RANDOM_STATE,
        "cue_offset_reference_ms": 200.0,
        "interpretation_limit": "Decoding indicates predictive information in these three cleaned EEG channels; it does not identify a neural generator or prove causal mechanism.",
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    significant = windows_df[windows_df["FWER_significant_0.05"] == 1]
    print(f"完成：{OUTPUT_DIR}", flush=True)
    print(f"置换最大统计量95%阈值：{global_95_threshold:.3f}", flush=True)
    print(f"全局校正显著时间窗：{len(significant)}", flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        plot_saved_results()
    else:
        main()
