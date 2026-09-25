"""Evaluate record-specific left/right ERP template matching on held-out trials.

Each recording is evaluated independently. Its trials are split into five
chronological blocks; each test block is classified against left/right mean
ERP templates built only from the other four blocks. The primary distance is
the mean absolute error over F3/Fz/F4 and the full 0-800 ms cue interval.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from revision_v3.config import CHANNELS, DATASETS, REAL_ROOT
from revision_v3.real_data import load_dataset


Q2_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Q2_ROOT / "output" / "15_record_specific_template_decoder"
N_BLOCKS = 5
BASELINE_S = (-0.2, 0.0)
RESPONSE_S = (0.0, 0.8)
WINDOW_DURATIONS_MS = (25, 50, 75, 100, 150, 200)
WINDOW_STEP_SAMPLES = 2


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    recalls = [float(np.mean(y_pred[y_true == label] == label))
               if np.any(y_true == label) else float("nan")
               for label in (0, 1)]
    return float(np.mean(recalls)), recalls[0], recalls[1]


def _best_threshold_balanced_accuracy(values: np.ndarray, labels: np.ndarray) -> float:
    """Training-only separability score for selecting one channel and window."""
    order = np.argsort(values, kind="mergesort")
    x = values[order]
    y = labels[order]
    cuts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
    positive = (y == 1).astype(int)
    negative = (y == 0).astype(int)
    cum_positive = np.r_[0, np.cumsum(positive)][cuts]
    cum_negative = np.r_[0, np.cumsum(negative)][cuts]
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    scores = []
    for index in range(len(cuts)):
        scores.append(0.5 * ((n_positive - cum_positive[index]) / n_positive
                             + cum_negative[index] / n_negative))
        scores.append(0.5 * (cum_positive[index] / n_positive
                             + (n_negative - cum_negative[index]) / n_negative))
    return float(max(scores))


def _select_local_template(trials: np.ndarray, labels: np.ndarray,
                           train_idx: np.ndarray, time_ms: np.ndarray) -> dict:
    """Select a univariate training feature, then form its left/right means."""
    sample_step_ms = float(np.median(np.diff(time_ms)))
    best = None
    for channel_index, channel in enumerate(CHANNELS):
        for duration_ms in WINDOW_DURATIONS_MS:
            n_samples = max(2, int(round(duration_ms / sample_step_ms)))
            last_start = len(time_ms) - n_samples
            for start in range(0, last_start + 1, WINDOW_STEP_SAMPLES):
                stop = start + n_samples
                values = trials[:, channel_index, start:stop].mean(axis=1)
                score = _best_threshold_balanced_accuracy(values[train_idx], labels[train_idx])
                # Deterministic ties: shorter window, then earlier channel/time.
                key = (score, -duration_ms, -channel_index, -start)
                if best is None or key > best["key"]:
                    left_mean = float(values[train_idx][labels[train_idx] == 0].mean())
                    right_mean = float(values[train_idx][labels[train_idx] == 1].mean())
                    best = {
                        "key": key,
                        "channel": channel,
                        "duration_ms": duration_ms,
                        "start_index": start,
                        "stop_index": stop,
                        "start_ms": float(time_ms[start]),
                        "end_ms": float(time_ms[stop - 1]),
                        "training_selection_ba": score,
                        "values": values,
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                    }
    if best is None:
        raise RuntimeError("No local feature window could be selected")
    return best


def evaluate_record(task: str, record: str) -> tuple[list[dict], dict]:
    data = load_dataset(REAL_ROOT / f"{record}_clean.mat")
    time = np.asarray(data["time_s"], dtype=float)
    cue = np.asarray(data["cue_type"], dtype=int)
    eeg = np.asarray(data["eeg"], dtype=float)

    baseline = (time >= BASELINE_S[0]) & (time < BASELINE_S[1])
    response = (time >= RESPONSE_S[0]) & (time <= RESPONSE_S[1])
    if baseline.sum() < 2 or response.sum() < 2:
        raise ValueError(f"{record}: baseline or response interval is unavailable")
    if not np.all(np.isin(cue, (-1, 1))):
        raise ValueError(f"{record}: cue labels must be -1/+1")

    # Per-trial prestimulus correction; no cross-trial fitted transform.
    trials = eeg - eeg[:, :, baseline].mean(axis=2, keepdims=True)
    trials = trials[:, :, response]
    time_ms = time[response] * 1000.0
    y = (cue == 1).astype(int)  # 0=left, 1=right
    block_indices = np.array_split(np.arange(len(y)), N_BLOCKS)
    full_curve_pred = np.full(len(y), -1, dtype=int)
    local_window_pred = np.full(len(y), -1, dtype=int)
    trial_rows = []

    for fold, test_idx in enumerate(block_indices, start=1):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
        if set(np.unique(y[train_idx])) != {0, 1}:
            raise ValueError(f"{record}, fold {fold}: training split lacks a cue class")
        left_template = trials[train_idx[y[train_idx] == 0]].mean(axis=0)
        right_template = trials[train_idx[y[train_idx] == 1]].mean(axis=0)

        test_trials = trials[test_idx]
        # The requested nearest-curve rule: average pointwise absolute error.
        d_left = np.mean(np.abs(test_trials - left_template[None, :, :]), axis=(1, 2))
        d_right = np.mean(np.abs(test_trials - right_template[None, :, :]), axis=(1, 2))
        full_pred = (d_right < d_left).astype(int)
        full_curve_pred[test_idx] = full_pred

        local = _select_local_template(trials, y, train_idx, time_ms)
        local_values = local["values"][test_idx]
        local_d_left = np.abs(local_values - local["left_mean"])
        local_d_right = np.abs(local_values - local["right_mean"])
        local_pred = (local_d_right < local_d_left).astype(int)
        local_window_pred[test_idx] = local_pred

        for local_i, trial_i in enumerate(test_idx):
            trial_rows.append({
                "task": task,
                "record": record,
                "fold": fold,
                "trial_index_1based": int(trial_i + 1),
                "cue_true": "right" if y[trial_i] else "left",
                "full_curve_prediction": "right" if full_pred[local_i] else "left",
                "full_curve_distance_left": float(d_left[local_i]),
                "full_curve_distance_right": float(d_right[local_i]),
                "full_curve_correct": int(full_pred[local_i] == y[trial_i]),
                "local_window_prediction": "right" if local_pred[local_i] else "left",
                "local_window_distance_left": float(local_d_left[local_i]),
                "local_window_distance_right": float(local_d_right[local_i]),
                "local_window_correct": int(local_pred[local_i] == y[trial_i]),
                "selected_channel": local["channel"],
                "selected_window_start_ms": local["start_ms"],
                "selected_window_last_sample_ms": local["end_ms"],
                "selected_window_nominal_ms": local["duration_ms"],
                "training_selection_balanced_accuracy": local["training_selection_ba"],
                "training_left_feature_mean": local["left_mean"],
                "training_right_feature_mean": local["right_mean"],
                "n_train_left": int(np.count_nonzero(y[train_idx] == 0)),
                "n_train_right": int(np.count_nonzero(y[train_idx] == 1)),
                "n_test": int(len(test_idx)),
                "template_scope": "same_record_training_trials_only",
                "full_curve_scope": "F3_Fz_F4_all_samples_0_800ms_mean_absolute_error",
                "local_window_scope": "training_selected_channel_window_mean_absolute_error_to_class_means",
            })

    if np.any(full_curve_pred < 0) or np.any(local_window_pred < 0):
        raise RuntimeError(f"{record}: not every trial received one held-out prediction")
    summaries = []
    for method, prediction in (("full_curve", full_curve_pred),
                               ("local_window", local_window_pred)):
        ba, recall_left, recall_right = _balanced_accuracy(y, prediction)
        correct = int(np.count_nonzero(y == prediction))
        summaries.append({
            "task": task,
            "record": record,
            "method": method,
            "n_trials": int(len(y)),
            "n_left": int(np.count_nonzero(y == 0)),
            "n_right": int(np.count_nonzero(y == 1)),
            "correct": correct,
            "accuracy": correct / len(y),
            "balanced_accuracy": ba,
            "macro_record_accuracy": correct / len(y),
            "macro_record_balanced_accuracy": ba,
            "recall_left": recall_left,
            "recall_right": recall_right,
            "cv": f"{N_BLOCKS}-fold leave-one-chronological-block-out",
            "channels": ";".join(CHANNELS),
            "response_ms": f"{time_ms[0]:g}-{time_ms[-1]:g}",
            "baseline_ms": f"{BASELINE_S[0]*1000:g}-{BASELINE_S[1]*1000:g}",
        })
    return trial_rows, summaries


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_trials, record_summaries = [], []
    for task, records in DATASETS.items():
        for record in records:
            trial_rows, summaries = evaluate_record(task, record)
            all_trials.extend(trial_rows)
            record_summaries.extend(summaries)
            for summary in summaries:
                print(f"{record} {summary['method']}: {summary['correct']}/"
                      f"{summary['n_trials']} acc={summary['accuracy']:.3f}, "
                      f"BA={summary['balanced_accuracy']:.3f}", flush=True)

    pooled_summaries = []
    for method in ("full_curve", "local_window"):
        method_rows = [row for row in all_trials]
        y = np.asarray([row["cue_true"] == "right" for row in method_rows], dtype=int)
        pred_name = f"{method}_prediction"
        prediction = np.asarray([row[pred_name] == "right" for row in method_rows], dtype=int)
        pooled_ba, pooled_recall_left, pooled_recall_right = _balanced_accuracy(y, prediction)
        method_record_summaries = [row for row in record_summaries if row["method"] == method]
        total_correct = int(np.count_nonzero(y == prediction))
        pooled_summaries.append({
            "task": "All",
            "record": "pooled_trials",
            "method": method,
            "n_trials": int(len(y)),
            "n_left": int(np.count_nonzero(y == 0)),
            "n_right": int(np.count_nonzero(y == 1)),
            "correct": total_correct,
            "accuracy": total_correct / len(y),
            "balanced_accuracy": pooled_ba,
            "macro_record_accuracy": float(np.mean([row["accuracy"] for row in method_record_summaries])),
            "macro_record_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in method_record_summaries])),
            "recall_left": pooled_recall_left,
            "recall_right": pooled_recall_right,
            "cv": f"{N_BLOCKS}-fold leave-one-chronological-block-out",
            "channels": ";".join(CHANNELS),
            "response_ms": "0-800",
            "baseline_ms": "-200-0",
        })
    summaries = record_summaries + pooled_summaries
    _write_csv(OUTPUT_DIR / "逐试次留出预测.csv", all_trials)
    _write_csv(OUTPUT_DIR / "逐记录准确率.csv", summaries)
    for row in pooled_summaries:
        print(f"{row['method']} pooled: {row['correct']}/{row['n_trials']} "
              f"acc={row['accuracy']:.3f}, BA={row['balanced_accuracy']:.3f}, "
              f"macro BA={row['macro_record_balanced_accuracy']:.3f}", flush=True)
    print(f"Saved outputs to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
