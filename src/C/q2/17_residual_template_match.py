"""Test the user's model-residual-template matching rule on Q1-cleaned EEG.

For each record, a training-fold ERP and the leave-one-MAT-out model curve
form a reference residual. A held-out trial's model residual is compared with
that reference in the previously selected local channel/window. The script
also verifies the algebraic equivalence with direct ERP-template matching.

Important: the four windows were selected by the exploratory all-trial scan
in output/14_exploratory_local_window_decoder. Therefore these scores are
diagnostic only; window selection is not independent of the held-out trials.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from revision_v3.config import CHANNELS, DATASETS, OUTPUT_ROOT, REAL_ROOT
from revision_v3.real_data import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = OUTPUT_ROOT / "17_residual_template_match"
MODEL_CURVES_FILE = OUTPUT_ROOT / "heldout_predictions.csv"
WINDOWS_FILE = SCRIPT_DIR / "output" / "14_exploratory_local_window_decoder" / "局部时间窗最佳训练内结果.csv"
N_BLOCKS = 5
BASELINE_S = (-0.2, 0.0)
RESPONSE_S = (0.0, 0.8)


def read_windows() -> dict[str, dict]:
    result = {}
    with WINDOWS_FILE.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            result[row["dataset"]] = {
                "channel": row["channel"],
                "start_ms": float(row["start_ms"]),
                "end_exclusive_ms": float(row["end_exclusive_ms"]),
            }
    expected = {record for records in DATASETS.values() for record in records}
    if set(result) != expected:
        raise ValueError(f"Expected windows for all four records; found {sorted(result)}")
    return result


def read_model_curves() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with MODEL_CURVES_FILE.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["condition"] in ("left", "right"):
                grouped[(row["record"], row["condition"])][row["channel"]].append(
                    (float(row["time_ms"]), float(row["model_heldout"]))
                )
    result = {}
    for record in (record for records in DATASETS.values() for record in records):
        for condition in ("left", "right"):
            channels = grouped[(record, condition)]
            if set(channels) != set(CHANNELS):
                raise ValueError(f"Missing generated curve for {record}/{condition}")
            times, values = [], []
            for channel in CHANNELS:
                points = sorted(channels[channel])
                t = np.asarray([point[0] for point in points], dtype=float)
                y = np.asarray([point[1] for point in points], dtype=float)
                if times and not np.array_equal(times[0], t):
                    raise ValueError(f"Model channel grids disagree for {record}/{condition}")
                if not times:
                    times.append(t)
                values.append(y)
            result[(record, condition)] = (times[0], np.stack(values))
    return result


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    left = float(np.mean(pred[y == 0] == 0))
    right = float(np.mean(pred[y == 1] == 1))
    return 0.5 * (left + right), left, right


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def global_four_region_match(windows: dict[str, dict], model_curves: dict) -> tuple[list[dict], list[dict]]:
    """Choose the highest weighted similarity among all 4 x 2 templates.

    Each candidate is normalized by its training-fold within-class absolute
    deviation before applying the requested region-point fraction. The
    candidate-specific model residual comparison is kept explicit; it is
    algebraically equivalent to empirical ERP-template matching.
    """
    prepared = {}
    for task, records in DATASETS.items():
        for record in records:
            data = load_dataset(REAL_ROOT / f"{record}_clean.mat")
            time_s = np.asarray(data["time_s"], dtype=float)
            eeg = np.asarray(data["eeg"], dtype=float)
            labels = (np.asarray(data["cue_type"], dtype=int) == 1).astype(int)
            baseline = (time_s >= BASELINE_S[0]) & (time_s < BASELINE_S[1])
            response = (time_s >= RESPONSE_S[0]) & (time_s <= RESPONSE_S[1] + 1e-12)
            corrected = eeg - eeg[:, :, baseline].mean(axis=2, keepdims=True)
            prepared[record] = {
                "task": task,
                "trials": corrected[:, :, response],
                "labels": labels,
                "time_ms": time_s[response] * 1000.0,
            }

    global_trial_rows = []
    block_map = {
        record: np.array_split(np.arange(len(item["labels"])), N_BLOCKS)
        for record, item in prepared.items()
    }
    all_records = list(prepared)
    for fold in range(N_BLOCKS):
        templates = {}
        for record in all_records:
            item = prepared[record]
            y, trials, time_ms = item["labels"], item["trials"], item["time_ms"]
            test_idx = block_map[record][fold]
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
            win = windows[record]
            channel_index = CHANNELS.index(win["channel"])
            region = ((time_ms >= win["start_ms"] - 1e-8)
                      & (time_ms < win["end_exclusive_ms"] - 1e-8))
            region_values = {}
            model_local = {}
            train_curve = {}
            within_class_deviation = []
            for label, condition in ((0, "left"), (1, "right")):
                train_class = trials[train_idx[y[train_idx] == label]]
                erp = train_class.mean(axis=0)
                model_time, model_signal = model_curves[(record, condition)]
                model = np.interp(time_ms[region], model_time, model_signal[channel_index])
                train_curve[label] = erp[channel_index, region]
                model_local[label] = model
                # R = empirical ERP - generated curve.
                region_values[label] = train_curve[label] - model
                within_class_deviation.extend(
                    np.abs(train_class[:, channel_index, :][:, region] - train_curve[label]).ravel()
                )
            scale = float(np.median(within_class_deviation))
            scale = max(scale, 1e-9)
            weight = int(region.sum()) / (len(CHANNELS) * len(time_ms))
            templates[record] = {
                "channel_index": channel_index,
                "region": region,
                "weight": weight,
                "scale": scale,
                "reference_residual": region_values,
                "model": model_local,
                "empirical_erp": train_curve,
            }

        for source_record in all_records:
            item = prepared[source_record]
            y, trials, time_ms = item["labels"], item["trials"], item["time_ms"]
            for trial_index in block_map[source_record][fold]:
                candidate_rows = []
                for candidate_record in all_records:
                    template = templates[candidate_record]
                    ch = template["channel_index"]
                    region = template["region"]
                    observed_local = trials[trial_index, ch, region]
                    for label in (0, 1):
                        # r = observed trial - candidate model curve.
                        observed_residual = observed_local - template["model"][label]
                        # Compare observed and training reference residuals.
                        point_loss = float(np.mean(np.abs(
                            observed_residual - template["reference_residual"][label]
                        )))
                        normalized_loss = point_loss / template["scale"]
                        similarity = float(np.exp(-normalized_loss))
                        weighted_similarity = template["weight"] * similarity
                        empirical_loss = float(np.mean(np.abs(
                            observed_local - template["empirical_erp"][label]
                        )))
                        candidate_rows.append({
                            "candidate_record": candidate_record,
                            "candidate_label": label,
                            "point_loss": point_loss,
                            "empirical_equivalent_loss": empirical_loss,
                            "normalized_loss": normalized_loss,
                            "similarity": similarity,
                            "region_weight": template["weight"],
                            "weighted_similarity": weighted_similarity,
                            "channel": CHANNELS[ch],
                            "start_ms": windows[candidate_record]["start_ms"],
                            "end_exclusive_ms": windows[candidate_record]["end_exclusive_ms"],
                            "n_region_points": int(region.sum()),
                        })
                best = max(candidate_rows, key=lambda row: row["weighted_similarity"])
                best_unweighted = max(candidate_rows, key=lambda row: row["similarity"])
                global_trial_rows.append({
                    "task_true": item["task"],
                    "record_true": source_record,
                    "fold": fold + 1,
                    "clean_trial_index_1based": int(trial_index + 1),
                    "cue_true": "right" if y[trial_index] else "left",
                    "prediction": "right" if best["candidate_label"] else "left",
                    "unweighted_prediction": "right" if best_unweighted["candidate_label"] else "left",
                    "predicted_template_record": best["candidate_record"],
                    "unweighted_template_record": best_unweighted["candidate_record"],
                    "predicted_region_channel": best["channel"],
                    "predicted_region_start_ms": best["start_ms"],
                    "predicted_region_end_exclusive_ms": best["end_exclusive_ms"],
                    "region_weight": best["region_weight"],
                    "weighted_similarity": best["weighted_similarity"],
                    "best_candidate_loss": best["point_loss"],
                    "best_candidate_empirical_equivalent_loss": best["empirical_equivalent_loss"],
                    "correct": int(best["candidate_label"] == y[trial_index]),
                    "unweighted_correct": int(best_unweighted["candidate_label"] == y[trial_index]),
                    "candidate_scope": "all four record windows x left/right training templates",
                    "window_selection": "fixed from all-trial exploratory scan; selection leakage",
                })

    summaries = []
    for record in all_records:
        rows = [row for row in global_trial_rows if row["record_true"] == record]
        y = np.asarray([row["cue_true"] == "right" for row in rows], dtype=int)
        pred = np.asarray([row["prediction"] == "right" for row in rows], dtype=int)
        ba, recall_left, recall_right = balanced_accuracy(y, pred)
        correct = int(np.count_nonzero(y == pred))
        unweighted_pred = np.asarray([row["unweighted_prediction"] == "right" for row in rows], dtype=int)
        unweighted_ba, unweighted_recall_left, unweighted_recall_right = balanced_accuracy(y, unweighted_pred)
        unweighted_correct = int(np.count_nonzero(y == unweighted_pred))
        summaries.append({
            "record": record,
            "n_trials": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "balanced_accuracy": ba,
            "recall_left": recall_left,
            "recall_right": recall_right,
            "unweighted_correct": unweighted_correct,
            "unweighted_accuracy": unweighted_correct / len(rows),
            "unweighted_balanced_accuracy": unweighted_ba,
            "unweighted_recall_left": unweighted_recall_left,
            "unweighted_recall_right": unweighted_recall_right,
            "most_selected_template_record": max(
                {candidate: sum(row["predicted_template_record"] == candidate for row in rows)
                 for candidate in all_records},
                key=lambda candidate: sum(row["predicted_template_record"] == candidate for row in rows),
            ),
            "scope": "highest weighted similarity among four region x two cue templates",
        })
    y = np.asarray([row["cue_true"] == "right" for row in global_trial_rows], dtype=int)
    pred = np.asarray([row["prediction"] == "right" for row in global_trial_rows], dtype=int)
    ba, recall_left, recall_right = balanced_accuracy(y, pred)
    correct = int(np.count_nonzero(y == pred))
    unweighted_pred = np.asarray([row["unweighted_prediction"] == "right" for row in global_trial_rows], dtype=int)
    unweighted_ba, unweighted_recall_left, unweighted_recall_right = balanced_accuracy(y, unweighted_pred)
    unweighted_correct = int(np.count_nonzero(y == unweighted_pred))
    summaries.append({
        "record": "pooled_trials",
        "n_trials": len(y),
        "correct": correct,
        "accuracy": correct / len(y),
        "balanced_accuracy": ba,
        "recall_left": recall_left,
        "recall_right": recall_right,
        "unweighted_correct": unweighted_correct,
        "unweighted_accuracy": unweighted_correct / len(y),
        "unweighted_balanced_accuracy": unweighted_ba,
        "unweighted_recall_left": unweighted_recall_left,
        "unweighted_recall_right": unweighted_recall_right,
        "most_selected_template_record": "see trial scores",
        "scope": "pooled highest weighted similarity; all-trial window-selection leakage",
    })
    return global_trial_rows, summaries


def main() -> None:
    windows = read_windows()
    model_curves = read_model_curves()
    trial_rows, summary_rows = [], []

    for task, records in DATASETS.items():
        for record in records:
            data = load_dataset(REAL_ROOT / f"{record}_clean.mat")
            time_s = np.asarray(data["time_s"], dtype=float)
            eeg = np.asarray(data["eeg"], dtype=float)
            y = (np.asarray(data["cue_type"], dtype=int) == 1).astype(int)
            baseline = (time_s >= BASELINE_S[0]) & (time_s < BASELINE_S[1])
            response = (time_s >= RESPONSE_S[0]) & (time_s <= RESPONSE_S[1] + 1e-12)
            if baseline.sum() < 2 or response.sum() < 2:
                raise ValueError(f"{record}: incomplete baseline or response epoch")
            time_ms = time_s[response] * 1000.0
            corrected = eeg - eeg[:, :, baseline].mean(axis=2, keepdims=True)
            trials = corrected[:, :, response]

            window = windows[record]
            channel_index = CHANNELS.index(window["channel"])
            region = ((time_ms >= window["start_ms"] - 1e-8)
                      & (time_ms < window["end_exclusive_ms"] - 1e-8))
            if region.sum() < 1:
                raise ValueError(f"{record}: selected local window has no samples")
            # A selected region is one electrode x its time samples; the full
            # input curve has three electrodes x all response samples.
            region_points = int(region.sum())
            total_points = int(len(CHANNELS) * response.sum())
            weight = region_points / total_points

            test_blocks = np.array_split(np.arange(len(y)), N_BLOCKS)
            pred = np.full(len(y), -1, dtype=int)
            for fold, test_idx in enumerate(test_blocks, start=1):
                train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
                if set(np.unique(y[train_idx])) != {0, 1}:
                    raise ValueError(f"{record}, fold {fold}: training set lacks a class")
                # The reference residuals use only the training-fold empirical
                # ERP and the corresponding held-out model output.
                residual_templates = {}
                for label, condition in ((0, "left"), (1, "right")):
                    empirical_erp = trials[train_idx[y[train_idx] == label]].mean(axis=0)
                    model_time, model_signal = model_curves[(record, condition)]
                    model_on_grid = np.stack([
                        np.interp(time_ms, model_time, row) for row in model_signal
                    ])
                    residual_templates[label] = (
                        empirical_erp[channel_index, region]
                        - model_on_grid[channel_index, region]
                    )

                for trial_index in test_idx:
                    raw_curve = trials[trial_index, channel_index, region]
                    residual_losses = {}
                    empirical_losses = {}
                    for label, condition in ((0, "left"), (1, "right")):
                        model_time, model_signal = model_curves[(record, condition)]
                        model_local = np.interp(
                            time_ms[region], model_time, model_signal[channel_index]
                        )
                        observed_residual = raw_curve - model_local
                        residual_loss = float(np.mean(np.abs(
                            observed_residual - residual_templates[label]
                        )))
                        empirical_template = trials[
                            train_idx[y[train_idx] == label], channel_index
                        ][:, region].mean(axis=0)
                        empirical_loss = float(np.mean(np.abs(raw_curve - empirical_template)))
                        residual_losses[label] = residual_loss
                        empirical_losses[label] = empirical_loss
                    # Equal region weight is applied to each pointwise MAE
                    # contribution; it cancels between left/right for a trial.
                    weighted = {label: weight * loss for label, loss in residual_losses.items()}
                    pred[trial_index] = int(weighted[1] < weighted[0])
                    trial_rows.append({
                        "task": task,
                        "record": record,
                        "fold": fold,
                        "clean_trial_index_1based": int(trial_index + 1),
                        "cue_true": "right" if y[trial_index] else "left",
                        "prediction": "right" if pred[trial_index] else "left",
                        "region_channel": window["channel"],
                        "region_start_ms": window["start_ms"],
                        "region_end_exclusive_ms": window["end_exclusive_ms"],
                        "region_valid_points": region_points,
                        "total_curve_points": total_points,
                        "region_weight": weight,
                        "weighted_loss_left": weighted[0],
                        "weighted_loss_right": weighted[1],
                        "residual_loss_left": residual_losses[0],
                        "residual_loss_right": residual_losses[1],
                        "empirical_curve_loss_left": empirical_losses[0],
                        "empirical_curve_loss_right": empirical_losses[1],
                        "algebraic_cancellation_max_abs_error": max(
                            abs(residual_losses[label] - empirical_losses[label])
                            for label in (0, 1)
                        ),
                        "correct": int(pred[trial_index] == y[trial_index]),
                        "window_source": "14 exploratory all-trial search; window-selection leakage",
                    })

            ba, recall_left, recall_right = balanced_accuracy(y, pred)
            n_correct = int(np.count_nonzero(y == pred))
            summary_rows.append({
                "task": task,
                "record": record,
                "n_trials": int(len(y)),
                "n_left": int(np.count_nonzero(y == 0)),
                "n_right": int(np.count_nonzero(y == 1)),
                "correct": n_correct,
                "accuracy": n_correct / len(y),
                "balanced_accuracy": ba,
                "recall_left": recall_left,
                "recall_right": recall_right,
                "region_channel": window["channel"],
                "region_start_ms": window["start_ms"],
                "region_end_exclusive_ms": window["end_exclusive_ms"],
                "region_valid_points": region_points,
                "total_curve_points": total_points,
                "region_weight": weight,
                "cv": "5-fold chronological test blocks; training-fold ERP residual templates",
                "window_selection": "fixed from all-trial exploratory analysis; not an unbiased estimate",
            })

    total_n = sum(row["n_trials"] for row in summary_rows)
    total_correct = sum(row["correct"] for row in summary_rows)
    summary_rows.append({
        "task": "All",
        "record": "pooled_trials",
        "n_trials": total_n,
        "n_left": sum(row["n_left"] for row in summary_rows),
        "n_right": sum(row["n_right"] for row in summary_rows),
        "correct": total_correct,
        "accuracy": total_correct / total_n,
        "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in summary_rows])),
        "recall_left": float("nan"),
        "recall_right": float("nan"),
        "region_channel": "record-specific",
        "region_start_ms": float("nan"),
        "region_end_exclusive_ms": float("nan"),
        "region_valid_points": float("nan"),
        "total_curve_points": float("nan"),
        "region_weight": float("nan"),
        "cv": "5-fold chronological test blocks",
        "window_selection": "macro balanced accuracy across four records",
    })

    save_csv(RESULT_DIR / "summary.csv", summary_rows)
    save_csv(RESULT_DIR / "trial_scores.csv", trial_rows)
    global_trial_rows, global_summaries = global_four_region_match(windows, model_curves)
    save_csv(RESULT_DIR / "global_four_region_summary.csv", global_summaries)
    save_csv(RESULT_DIR / "global_four_region_trial_scores.csv", global_trial_rows)
    for row in summary_rows:
        print(f"{row['record']}: {row['correct']}/{row['n_trials']} "
              f"accuracy={row['accuracy']:.3f}, BA={row['balanced_accuracy']:.3f}")
    print("Global four-region weighted similarity:")
    for row in global_summaries:
        print(f"  {row['record']}: {row['correct']}/{row['n_trials']} "
              f"accuracy={row['accuracy']:.3f}, BA={row['balanced_accuracy']:.3f}; "
              f"unweighted={row['unweighted_accuracy']:.3f}, "
              f"BA={row['unweighted_balanced_accuracy']:.3f}")
    print(f"Saved residual-template results to {RESULT_DIR}")


if __name__ == "__main__":
    main()
