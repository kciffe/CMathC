"""Nested inner-fold probability threshold selection over the joint candidate set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import joint_search as joint
import model_search as previous
import temporal_search as temporal

Q3_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Q3_DIR / "output" / "experiments" / "20260925_threshold_search_v1"
PROBABILITY_THRESHOLDS = (0.50, 0.45, 0.55, 0.40, 0.60, 0.35, 0.65)


def select_threshold_candidate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the highest inner BA; ties prefer earlier candidates and native threshold."""
    if not scored:
        raise ValueError("At least one candidate threshold score is required")
    return max(
        scored,
        key=lambda row: (
            row["inner_mean_ba"],
            -row["candidate_order"],
            -abs(row["threshold"] - row.get("native_threshold", 0.5)),
            -row["threshold"],
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _continuous_scores(
    item: joint.JointCandidate,
    model_frame: pd.DataFrame,
    temporal_frame: pd.DataFrame,
    fixed: dict[str, np.ndarray],
    csp: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[np.ndarray, float]:
    if item.source == "round1_classifier":
        candidate = item.candidate
        columns = previous.FEATURE_SETS[candidate.feature_set]
        train_raw = model_frame.iloc[train_idx].loc[:, columns].to_numpy(dtype=float)
        eval_raw = model_frame.iloc[eval_idx].loc[:, columns].to_numpy(dtype=float)
        x_train, x_eval, _, _ = previous.scale_train_test(train_raw, eval_raw, candidate.scaler)
        labels = model_frame.iloc[train_idx]["cue_side"].to_numpy(dtype=int)
        scores, native = previous.predict_scores(x_train, labels, x_eval, candidate)
        return np.asarray(scores, dtype=float), float(native)
    candidate = item.candidate
    labels = temporal_frame["cue_side"].to_numpy(dtype=int)
    train_raw, eval_raw = temporal._features_for_split(
        candidate.representation, candidate, train_idx, eval_idx, fixed, csp, labels
    )
    x_train, x_eval = temporal.scale_train_test(train_raw, eval_raw)
    scores = temporal.fit_logistic_scores(
        x_train, labels[train_idx], x_eval, candidate.l2
    )
    return scores, 0.5


def _thresholds(item: joint.JointCandidate, native: float) -> tuple[float, ...]:
    if item.source == "temporal_or_csp" or item.candidate.model in {"logistic", "lda"}:
        return PROBABILITY_THRESHOLDS
    return (native,)


def _prediction_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    predicted = np.where(scores >= threshold, 1, -1)
    return {
        "accuracy": float(np.mean(predicted == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "auc": float(roc_auc_score((y_true > 0).astype(int), scores)),
        "macro_f1": float(f1_score(y_true, predicted, labels=[-1, 1], average="macro", zero_division=0)),
    }, predicted


def _prepare(stage: str, sample: str, features_path: Path, epochs_path: Path):
    features = pd.read_csv(features_path, encoding="utf-8-sig")
    model_frame = previous.prepare_scope(features, stage, sample).reset_index(drop=True)
    temporal_full, fixed_full, csp_full = temporal.build_stage_data(features_path, epochs_path, stage)
    indices = temporal.prepare_scope(temporal_full, sample)
    temporal_frame = temporal_full.iloc[indices].reset_index(drop=True)
    fixed = {name: values[indices] for name, values in fixed_full.items()}
    csp = csp_full[indices]
    joint._align_frames(model_frame, temporal_frame)
    return model_frame, temporal_frame, fixed, csp


def run_experiment(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = previous.DEFAULT_INPUT
    epochs_path = temporal.DEFAULT_EPOCHS
    candidates = joint._candidate_union()
    inner_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    baseline_predictions: list[pd.DataFrame] = []
    selected_predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for stage in ("cue_locked", "target_locked"):
        for sample in ("raw_qc_pass", "q1_quality_matched"):
            model_frame, temporal_frame, fixed, csp = _prepare(
                stage, sample, features_path, epochs_path
            )
            groups = model_frame["record"].to_numpy()
            labels = model_frame["cue_side"].to_numpy(dtype=int)
            outer_folds = previous.grouped_leave_one_group_out(groups)
            baseline_candidate = candidates[0]
            for outer_train, outer_test, held_out in outer_folds:
                train_records = sorted(model_frame.iloc[outer_train]["record"].unique().tolist())
                if held_out in train_records:
                    raise AssertionError("Outer held-out record leaked into threshold training")
                baseline_scores, _ = _continuous_scores(
                    baseline_candidate, model_frame, temporal_frame, fixed, csp,
                    outer_train, outer_test,
                )
                baseline_metrics, baseline_pred = _prediction_metrics(
                    labels[outer_test], baseline_scores, 0.5
                )
                baseline_predictions.append(pd.DataFrame({
                    "record": model_frame.iloc[outer_test]["record"].astype(str).to_numpy(),
                    "original_trial_index": model_frame.iloc[outer_test]["original_trial_index"].to_numpy(dtype=int),
                    "true_cue_side": labels[outer_test], "predicted_cue_side": baseline_pred,
                    "score_right": baseline_scores, "sample": sample, "stage": stage,
                    "candidate": baseline_candidate.name, "threshold": 0.5,
                    "held_out_record": held_out,
                }))
                baseline_rows.append({
                    "sample": sample, "stage": stage, "held_out_record": held_out,
                    "balanced_accuracy": baseline_metrics["balanced_accuracy"],
                })

                inner_folds = previous.grouped_leave_one_group_out(groups[outer_train])
                threshold_scores: list[dict[str, Any]] = []
                for candidate in candidates:
                    threshold_values: tuple[float, ...] | None = None
                    per_threshold: dict[float, list[float]] = {}
                    for inner_train_local, inner_valid_local, inner_held in inner_folds:
                        inner_train = outer_train[inner_train_local]
                        inner_valid = outer_train[inner_valid_local]
                        scores, native = _continuous_scores(
                            candidate, model_frame, temporal_frame, fixed, csp,
                            inner_train, inner_valid,
                        )
                        if threshold_values is None:
                            threshold_values = _thresholds(candidate, native)
                            per_threshold = {value: [] for value in threshold_values}
                        inner_y = labels[inner_valid]
                        for threshold in threshold_values:
                            predicted = np.where(scores >= threshold, 1, -1)
                            score = float(balanced_accuracy_score(inner_y, predicted))
                            per_threshold[threshold].append(score)
                            inner_rows.append({
                                "sample": sample, "stage": stage,
                                "outer_held_out_record": held_out,
                                "inner_held_out_record": inner_held,
                                "candidate": candidate.name,
                                "source": candidate.source,
                                "candidate_order": candidate.order,
                                "threshold": threshold,
                                "native_threshold": native,
                                "balanced_accuracy": score,
                            })
                    for threshold, values in per_threshold.items():
                        threshold_scores.append({
                            "inner_mean_ba": float(np.mean(values)),
                            "candidate_order": candidate.order,
                            "threshold": float(threshold),
                            "native_threshold": float(native),
                            "candidate": candidate,
                            "candidate_name": candidate.name,
                            "source": candidate.source,
                        })
                chosen_row = select_threshold_candidate(threshold_scores)
                chosen = chosen_row["candidate"]
                chosen_scores, _ = _continuous_scores(
                    chosen, model_frame, temporal_frame, fixed, csp, outer_train, outer_test
                )
                chosen_threshold = float(chosen_row["threshold"])
                metrics, predicted = _prediction_metrics(labels[outer_test], chosen_scores, chosen_threshold)
                selected_rows.append({
                    "sample": sample, "stage": stage, "held_out_record": held_out,
                    "n_train": int(len(outer_train)), "n_test": int(len(outer_test)),
                    "training_records": ";".join(train_records),
                    "selected_candidate": chosen.name, "source": chosen.source,
                    "threshold": chosen_threshold,
                    "selected_by_inner_balanced_accuracy": chosen_row["inner_mean_ba"],
                    **metrics,
                    "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                    "balanced_accuracy_delta": metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
                })
                selected_predictions.append(pd.DataFrame({
                    "record": model_frame.iloc[outer_test]["record"].astype(str).to_numpy(),
                    "original_trial_index": model_frame.iloc[outer_test]["original_trial_index"].to_numpy(dtype=int),
                    "true_cue_side": labels[outer_test], "predicted_cue_side": predicted,
                    "score_right": chosen_scores, "sample": sample, "stage": stage,
                    "candidate": chosen.name, "threshold": chosen_threshold,
                    "held_out_record": held_out,
                }))

            selected = pd.DataFrame([r for r in selected_rows if r["sample"] == sample and r["stage"] == stage])
            baseline_ba = float(np.mean([
                row["balanced_accuracy"] for row in baseline_rows
                if row["sample"] == sample and row["stage"] == stage
            ]))
            summaries.append({
                "sample": sample, "stage": stage, "n_trials": int(len(model_frame)),
                "valid_record_folds": int(selected["held_out_record"].nunique()),
                "baseline_mean_balanced_accuracy": baseline_ba,
                "threshold_selected_mean_balanced_accuracy": float(selected["balanced_accuracy"].mean()),
                "delta_percentage_points": float(selected["balanced_accuracy_delta"].mean() * 100),
                "selected_fold_balanced_accuracy": selected.set_index("held_out_record")["balanced_accuracy"].to_dict(),
                "selected_candidates_by_fold": selected.set_index("held_out_record")["selected_candidate"].to_dict(),
                "selected_thresholds_by_fold": selected.set_index("held_out_record")["threshold"].to_dict(),
            })

    inner_df = pd.DataFrame(inner_rows)
    selected_df = pd.DataFrame(selected_rows)
    inner_df.to_csv(output_dir / "inner_threshold_scores.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(output_dir / "nested_selected_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(baseline_predictions, ignore_index=True).to_csv(output_dir / "baseline_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(selected_predictions, ignore_index=True).to_csv(output_dir / "nested_selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "experiment": "nested inner-record probability threshold calibration over the round-one and temporal candidate union",
        "development_only": True, "base_candidate_count": len(candidates),
        "probability_threshold_grid": list(PROBABILITY_THRESHOLDS),
        "primary_metric": "balanced_accuracy",
        "outer_validation": "leave one complete recording out",
        "inner_selection": "candidate and probability threshold selected on inner leave-one-record-out folds only",
        "results": summaries,
        "limitations": [
            "This threshold search follows several exploratory rounds on the same four outer records; it is a development result and is prone to selection optimism.",
            "Only probability-producing Logistic/LDA models use the threshold grid; SVM margins retain their fixed native zero threshold.",
            "Only four recording-level folds are available, so results are uncertain and need an independent recording-level replication.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_findings(output_dir, summary)
    artifact_paths = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "experiment": summary["experiment"], "created_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "inputs": {"trial_features_sha256": _sha256(features_path), "raw_event_epochs_sha256": _sha256(epochs_path)},
        "sources": {name: _sha256(Path(__file__).with_name(name)) for name in ("threshold_search.py", "joint_search.py", "model_search.py", "temporal_search.py")},
        "artifacts": [{"path": p.name, "sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifact_paths],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_findings(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Nested threshold calibration findings", "",
        "Thresholds were selected from inner held-record scores; outer records were scored after selection.", "",
        "| Sample | Stage | Baseline BA | Nested threshold BA | Delta |", "|---|---|---:|---:|---:|",
    ]
    for item in summary["results"]:
        lines.append(
            f"| {item['sample']} | {item['stage']} | {item['baseline_mean_balanced_accuracy']:.4f} | "
            f"{item['threshold_selected_mean_balanced_accuracy']:.4f} | {item['delta_percentage_points']:+.2f} pp |"
        )
    lines.extend(["", "See fold-level selections and thresholds in `nested_selected_fold_metrics.csv`.", "", "## Limits", "", *[f"- {x}" for x in summary["limitations"]]])
    (output_dir / "experiment_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_experiment(args.output_dir)
    for item in summary["results"]:
        print(f"{item['sample']} {item['stage']}: {item['threshold_selected_mean_balanced_accuracy']:.3f} ({item['delta_percentage_points']:+.2f} pp)")
    print(f"Wrote threshold search results to {args.output_dir}")


if __name__ == "__main__":
    main()
