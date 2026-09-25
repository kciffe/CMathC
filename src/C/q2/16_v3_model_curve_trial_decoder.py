"""Compare cleaned Q1 EEG curves with revision_v3 generated cue curves.

The revision_v3 model is fitted leave-one-MAT-record-out. For each held-out
record, a trial is assigned to the generated left or right model curve with
the smaller mean absolute error over F3/Fz/F4 and the 0-800 ms cue epoch.
The same comparison is also reported for each cleaned condition-average ERP.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from revision_v3.config import CHANNELS, DATASETS, OUTPUT_ROOT, REAL_ROOT
from revision_v3.real_data import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = OUTPUT_ROOT / "model_curve_trial_decoder"
PREDICTION_FILE = OUTPUT_ROOT / "heldout_predictions.csv"
BASELINE_S = (-0.2, 0.0)
RESPONSE_S = (0.0, 0.8)


def _read_model_curves() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with PREDICTION_FILE.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["condition"] not in ("left", "right"):
                continue
            key = (row["record"], row["condition"])
            grouped[key][row["channel"]].append(
                (float(row["time_ms"]), float(row["model_heldout"]))
            )

    curves = {}
    for record in DATASETS["Task1"] + DATASETS["Task2"]:
        for condition in ("left", "right"):
            key = (record, condition)
            channels = grouped.get(key)
            if channels is None or set(channels) != set(CHANNELS):
                raise ValueError(f"Missing held-out model curve for {key}")
            times = []
            values = []
            for channel in CHANNELS:
                points = sorted(channels[channel])
                times_channel = np.asarray([point[0] for point in points], dtype=float)
                values_channel = np.asarray([point[1] for point in points], dtype=float)
                if times and not np.array_equal(times[0], times_channel):
                    raise ValueError(f"Channel time grids disagree for {key}")
                if not times:
                    times.append(times_channel)
                values.append(values_channel)
            curves[key] = (times[0], np.stack(values))
    return curves


def _balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    recall_left = float(np.mean(prediction[y == 0] == 0))
    recall_right = float(np.mean(prediction[y == 1] == 1))
    return (recall_left + recall_right) / 2.0, recall_left, recall_right


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    model_curves = _read_model_curves()
    trial_rows, average_rows, summary_rows = [], [], []

    for task, records in DATASETS.items():
        for record in records:
            data = load_dataset(REAL_ROOT / f"{record}_clean.mat")
            time_s = np.asarray(data["time_s"], dtype=float)
            eeg = np.asarray(data["eeg"], dtype=float)
            cue = np.asarray(data["cue_type"], dtype=int)
            baseline = (time_s >= BASELINE_S[0]) & (time_s < BASELINE_S[1])
            response = (time_s >= RESPONSE_S[0]) & (time_s <= RESPONSE_S[1] + 1e-12)
            if baseline.sum() < 2 or response.sum() < 2:
                raise ValueError(f"{record}: baseline or response interval is unavailable")

            # *_clean.mat contains Q1-filtered, 128-Hz, SQI-retained trials.
            # Apply the same per-trial prestimulus baseline subtraction used by
            # revision_v3, then compare on its measured cue-response grid.
            corrected = eeg - eeg[:, :, baseline].mean(axis=2, keepdims=True)
            observed = corrected[:, :, response]
            time_ms = time_s[response] * 1000.0
            y = (cue == 1).astype(int)  # 0=left, 1=right

            candidates = {}
            for condition in ("left", "right"):
                model_time, model_signal = model_curves[(record, condition)]
                candidates[condition] = np.stack(
                    [np.interp(time_ms, model_time, row) for row in model_signal]
                )

            distance_left = np.mean(
                np.abs(observed - candidates["left"][None, :, :]), axis=(1, 2)
            )
            distance_right = np.mean(
                np.abs(observed - candidates["right"][None, :, :]), axis=(1, 2)
            )
            prediction = (distance_right < distance_left).astype(int)
            ba, recall_left, recall_right = _balanced_accuracy(y, prediction)
            n_correct = int(np.count_nonzero(y == prediction))
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
                "mean_distance_to_true_model_curve": float(np.mean(
                    np.where(y == 0, distance_left, distance_right)
                )),
                "mean_distance_to_other_model_curve": float(np.mean(
                    np.where(y == 0, distance_right, distance_left)
                )),
                "mean_model_left_right_curve_distance": float(np.mean(
                    np.abs(candidates["left"] - candidates["right"])
                )),
                "comparison": "same_record_leave_one_MAT_out_model_curves; trial MAE over F3_Fz_F4 and 0-800ms",
                "preprocessing": "Q1 0.2-24Hz filtered and 128Hz clean.mat; per-trial -200_to_0ms baseline",
            })

            for index in range(len(y)):
                trial_rows.append({
                    "task": task,
                    "record": record,
                    "clean_trial_index_1based": index + 1,
                    "cue_true": "right" if y[index] else "left",
                    "prediction": "right" if prediction[index] else "left",
                    "distance_to_left_model_curve": float(distance_left[index]),
                    "distance_to_right_model_curve": float(distance_right[index]),
                    "correct": int(prediction[index] == y[index]),
                })

            # Also score condition-average cleaned input ERPs against both
            # generated alternatives, so ERP-level fit and trial decoding are
            # kept as separate results.
            for condition, label in (("left", 0), ("right", 1)):
                class_trials = observed[y == label]
                erp = class_trials.mean(axis=0)
                d_left = float(np.mean(np.abs(erp - candidates["left"])))
                d_right = float(np.mean(np.abs(erp - candidates["right"])))
                average_rows.append({
                    "task": task,
                    "record": record,
                    "observed_condition": condition,
                    "n_trials": int(len(class_trials)),
                    "distance_to_left_model_curve": d_left,
                    "distance_to_right_model_curve": d_right,
                    "nearest_model_condition": "right" if d_right < d_left else "left",
                    "correct_nearest_condition": int((d_right < d_left) == (label == 1)),
                })

    for task, records in [("All", tuple(record for values in DATASETS.values() for record in values))]:
        rows = [row for row in summary_rows if row["record"] in records]
        total_n = sum(row["n_trials"] for row in rows)
        total_correct = sum(row["correct"] for row in rows)
        summary_rows.append({
            "task": task,
            "record": "pooled_trials",
            "n_trials": total_n,
            "n_left": sum(row["n_left"] for row in rows),
            "n_right": sum(row["n_right"] for row in rows),
            "correct": total_correct,
            "accuracy": total_correct / total_n,
            "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in rows])),
            "recall_left": float("nan"),
            "recall_right": float("nan"),
            "mean_distance_to_true_model_curve": float("nan"),
            "mean_distance_to_other_model_curve": float("nan"),
            "mean_model_left_right_curve_distance": float("nan"),
            "comparison": "pooled across four held-out MAT records; balanced_accuracy is macro mean",
            "preprocessing": "see record rows",
        })

    _write_csv(RESULT_DIR / "record_summary.csv", summary_rows)
    _write_csv(RESULT_DIR / "trial_predictions.csv", trial_rows)
    _write_csv(RESULT_DIR / "condition_average_curve_match.csv", average_rows)

    for row in summary_rows:
        print(f"{row['record']}: {row['correct']}/{row['n_trials']} "
              f"accuracy={row['accuracy']:.3f}, BA={row['balanced_accuracy']:.3f}")
    print(f"Saved model-curve comparison to {RESULT_DIR}")


if __name__ == "__main__":
    main()
