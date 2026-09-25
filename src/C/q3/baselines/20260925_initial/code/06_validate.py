"""Validate EEG cue-condition information with held-out-record prediction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit

from common import ensure_output_dir, write_csv, write_json
from config import EEG_CHANNELS, RANDOM_SEED


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    design = np.column_stack([np.ones(x.shape[0]), x])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ parameters
        loss = np.logaddexp(0.0, logits).sum() - np.dot(y, logits)
        loss += 0.5 * l2 * np.dot(parameters[1:], parameters[1:])
        residual = expit(logits) - y
        gradient = design.T @ residual
        gradient[1:] += l2 * parameters[1:]
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = []
    for label in (-1, 1):
        mask = actual == label
        if mask.any():
            recalls.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(recalls)) if recalls else np.nan


def area_under_curve(actual: np.ndarray, probability_positive: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=int)
    scores = np.asarray(probability_positive, dtype=float)
    positive = actual == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if not n_positive or not n_negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in (-1, 1):
        tp = int(np.sum((actual == label) & (predicted == label)))
        fp = int(np.sum((actual != label) & (predicted == label)))
        fn = int(np.sum((actual == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def validate_stage(
    frame: pd.DataFrame,
    stage: str,
    feature_columns: list[str],
    variant: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    stage_frame = frame.loc[frame["stage"] == stage].copy()
    if variant is not None:
        stage_frame = stage_frame.loc[stage_frame["analysis_variant"] == variant].copy()
    if "qc_valid" in stage_frame:
        stage_frame = stage_frame.loc[stage_frame["qc_valid"].fillna(False).astype(bool)].copy()
    stage_frame["cue_side"] = pd.to_numeric(stage_frame["cue_side"], errors="coerce")
    stage_frame[feature_columns] = stage_frame[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    stage_frame = stage_frame.dropna(subset=["cue_side", *feature_columns])
    predictions: list[dict] = []
    folds: list[dict] = []

    for held_out_record in sorted(stage_frame["record"].unique()):
        train = stage_frame.loc[stage_frame["record"] != held_out_record]
        test = stage_frame.loc[stage_frame["record"] == held_out_record]
        y_train_side = train["cue_side"].to_numpy(dtype=int)
        y_test_side = test["cue_side"].to_numpy(dtype=int)
        if np.unique(y_train_side).size < 2 or np.unique(y_test_side).size < 2:
            folds.append(
                {"stage": stage, "analysis_variant": variant or "nominal", "held_out_record": held_out_record, "status": "skipped_single_class"}
            )
            continue

        x_train_raw = train[feature_columns].to_numpy(dtype=float)
        x_test_raw = test[feature_columns].to_numpy(dtype=float)
        center = np.mean(x_train_raw, axis=0)
        scale = np.std(x_train_raw, axis=0)
        scale[~np.isfinite(scale) | (scale <= np.finfo(float).eps)] = 1.0
        x_train = (x_train_raw - center) / scale
        x_test = (x_test_raw - center) / scale
        y_train = (y_train_side > 0).astype(float)
        coefficients = fit_logistic(x_train, y_train)
        probabilities = expit(np.column_stack([np.ones(len(test)), x_test]) @ coefficients)
        predicted_side = np.where(probabilities >= 0.5, 1, -1)
        accuracy = float(np.mean(predicted_side == y_test_side))
        bal_acc = balanced_accuracy(y_test_side, predicted_side)
        majority_baseline = float(
            max(np.mean(y_train_side == -1), np.mean(y_train_side == 1))
        )
        fold = {
            "stage": stage,
            "analysis_variant": variant or "nominal",
            "held_out_record": held_out_record,
            "status": "ok",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "left_test_n": int(np.sum(y_test_side == -1)),
            "right_test_n": int(np.sum(y_test_side == 1)),
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "auc": area_under_curve((y_test_side > 0).astype(int), probabilities),
            "macro_f1": macro_f1(y_test_side, predicted_side),
            "confusion_true_left_pred_left": int(np.sum((y_test_side == -1) & (predicted_side == -1))),
            "confusion_true_left_pred_right": int(np.sum((y_test_side == -1) & (predicted_side == 1))),
            "confusion_true_right_pred_left": int(np.sum((y_test_side == 1) & (predicted_side == -1))),
            "confusion_true_right_pred_right": int(np.sum((y_test_side == 1) & (predicted_side == 1))),
            "training_majority_baseline_accuracy": majority_baseline,
        }
        folds.append(fold)
        predictions.extend(
            {
                "record": row.record,
                "original_trial_index": int(row.original_trial_index),
                "stage": stage,
                "analysis_variant": variant or "nominal",
                "true_cue_side": int(row.cue_side),
                "predicted_cue_side": int(predicted_side[index]),
                "probability_right": float(probabilities[index]),
                "held_out_record": held_out_record,
            }
            for index, row in enumerate(test.itertuples(index=False))
        )
    return pd.DataFrame(predictions), folds


def within_record_diagnostic(
    frame: pd.DataFrame,
    stage: str,
    feature_columns: list[str],
    repeats: int = 10,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Check within-record cue-side signal; this is secondary and optimistic."""
    stage_frame = frame.loc[frame["stage"] == stage].copy()
    if "qc_valid" in stage_frame:
        stage_frame = stage_frame.loc[stage_frame["qc_valid"].fillna(False).astype(bool)].copy()
    stage_frame["cue_side"] = pd.to_numeric(stage_frame["cue_side"], errors="coerce")
    stage_frame[feature_columns] = stage_frame[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    stage_frame = stage_frame.dropna(subset=["cue_side", *feature_columns])
    rows: list[dict] = []
    random = np.random.default_rng(RANDOM_SEED)

    for record in sorted(stage_frame["record"].unique()):
        subset = stage_frame.loc[stage_frame["record"] == record].reset_index(drop=True)
        y_side = subset["cue_side"].to_numpy(dtype=int)
        class_sizes = [int(np.sum(y_side == side)) for side in (-1, 1)]
        folds = min(n_splits, min(class_sizes)) if class_sizes else 0
        scores: list[float] = []
        if folds < 2:
            rows.append(
                {
                    "stage": stage,
                    "record": record,
                    "status": "insufficient_trials_per_cue_side",
                    "left_n": class_sizes[0] if class_sizes else 0,
                    "right_n": class_sizes[1] if len(class_sizes) > 1 else 0,
                }
            )
            continue

        x_raw = subset[feature_columns].to_numpy(dtype=float)
        for _ in range(repeats):
            fold_indices: list[list[int]] = [[] for _ in range(folds)]
            for side in (-1, 1):
                indices = np.flatnonzero(y_side == side)
                random.shuffle(indices)
                for fold_index, split in enumerate(np.array_split(indices, folds)):
                    fold_indices[fold_index].extend(split.tolist())

            observed_all: list[int] = []
            predicted_all: list[int] = []
            for test_indices_list in fold_indices:
                test_indices = np.asarray(test_indices_list, dtype=int)
                train_mask = np.ones(len(subset), dtype=bool)
                train_mask[test_indices] = False
                train_indices = np.flatnonzero(train_mask)
                center = np.mean(x_raw[train_indices], axis=0)
                scale = np.std(x_raw[train_indices], axis=0)
                scale[~np.isfinite(scale) | (scale <= np.finfo(float).eps)] = 1.0
                x_train = (x_raw[train_indices] - center) / scale
                x_test = (x_raw[test_indices] - center) / scale
                coefficients = fit_logistic(x_train, (y_side[train_indices] > 0).astype(float))
                probability = expit(np.column_stack([np.ones(len(test_indices)), x_test]) @ coefficients)
                predicted_all.extend(np.where(probability >= 0.5, 1, -1).tolist())
                observed_all.extend(y_side[test_indices].tolist())
            scores.append(
                balanced_accuracy(np.asarray(observed_all), np.asarray(predicted_all))
            )

        rows.append(
            {
                "stage": stage,
                "record": record,
                "status": "ok",
                "left_n": class_sizes[0],
                "right_n": class_sizes[1],
                "folds": folds,
                "repeats": repeats,
                "mean_balanced_accuracy": float(np.mean(scores)),
                "sd_across_repeats": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "interpretation": "within-record diagnostic only; not a cross-record generalization estimate",
            }
        )
    return pd.DataFrame(rows)


def descriptive_effects(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    records = [*sorted(frame["record"].unique()), "ALL_RECORDS"]
    for record in records:
        subset = frame if record == "ALL_RECORDS" else frame.loc[frame["record"] == record]
        for stage in sorted(subset["stage"].unique()):
            stage_data = subset.loc[subset["stage"] == stage]
            left = stage_data.loc[stage_data["cue_side"] == -1]
            right = stage_data.loc[stage_data["cue_side"] == 1]
            for feature in feature_columns:
                left_values = pd.to_numeric(left[feature], errors="coerce").dropna()
                right_values = pd.to_numeric(right[feature], errors="coerce").dropna()
                if not len(left_values) or not len(right_values):
                    continue
                left_sd = float(left_values.std(ddof=1)) if len(left_values) > 1 else np.nan
                right_sd = float(right_values.std(ddof=1)) if len(right_values) > 1 else np.nan
                pooled_denominator = np.sqrt(
                    ((len(left_values) - 1) * left_sd**2 + (len(right_values) - 1) * right_sd**2)
                    / max(len(left_values) + len(right_values) - 2, 1)
                ) if np.isfinite(left_sd) and np.isfinite(right_sd) else np.nan
                delta = float(right_values.mean() - left_values.mean())
                rows.append(
                    {
                        "record": record,
                        "stage": stage,
                        "feature": feature,
                        "left_n": int(len(left_values)),
                        "right_n": int(len(right_values)),
                        "left_mean": float(left_values.mean()),
                        "right_mean": float(right_values.mean()),
                        "right_minus_left": delta,
                        "cohens_d_descriptive": delta / pooled_denominator
                        if pooled_denominator > 0
                        else np.nan,
                        "inference": "descriptive only; no significance claim",
                    }
                )
    return pd.DataFrame(rows)


def write_interpretation(
    output_dir,
    features: pd.DataFrame,
    effects: pd.DataFrame,
    stage_summaries: dict,
    within_record: pd.DataFrame,
    behavior_status: dict,
) -> None:
    trial_table = pd.read_csv(output_dir / "trial_table.csv", encoding="utf-8-sig")
    core = features.loc[
        features["stage"].isin(("cue_locked", "target_locked"))
        & features["qc_valid"].fillna(False)
    ]
    left_n = int((core.loc[core["stage"] == "cue_locked", "cue_side"] == -1).sum())
    right_n = int((core.loc[core["stage"] == "cue_locked", "cue_side"] == 1).sum())
    missing_values = int(core["erp_mean_F3"].isna().sum())
    q1_matched_n = int(
        core.loc[core["stage"] == "cue_locked", "q1_quality_pass"].fillna(False).sum()
    )
    q1_cue = core.loc[
        (core["stage"] == "cue_locked") & core["q1_quality_pass"].fillna(False)
    ]
    q1_left_n = int((q1_cue["cue_side"] == -1).sum())
    q1_right_n = int((q1_cue["cue_side"] == 1).sum())
    response_n = int(trial_table["choice_side"].notna().sum())
    response_rt = pd.to_numeric(trial_table["reaction_time_s"], errors="coerce")
    response_rt_median = float(response_rt.median()) if response_rt.notna().any() else np.nan
    rt_short_n = int((response_rt.notna() & (response_rt < 0.10)).sum())
    behavior_balanced_accuracy = behavior_status.get("mean_balanced_accuracy")
    behavior_comparison = behavior_status.get("model_comparison", {})
    behavior_comparison_ba = behavior_comparison.get("mean_balanced_accuracy", {})
    cue_choice_alignment = behavior_comparison.get(
        "cue_choice_alignment_by_filename_task_code", []
    )
    cue_choice_alignment_text = "; ".join(
        f"code {int(row['filename_task_code_candidate'])}: "
        f"{row['n_cue_choice_same_side']}/{row['n_trials']} same-side "
        f"({row['cue_choice_same_side_fraction']:.1%})"
        for row in cue_choice_alignment
        if pd.notna(row.get("filename_task_code_candidate"))
    )
    quality_agreement_text = "not available"
    quality_agreement_path = output_dir / "quality_agreement.csv"
    if quality_agreement_path.exists():
        quality_agreement = pd.read_csv(quality_agreement_path, encoding="utf-8-sig")
        quality_agreement_text = "; ".join(
            f"{row.stage}: Q1/raw-QC both pass {int(row.q1_pass_raw_qc_pass)}, "
            f"Q1 fail/raw-QC pass {int(row.q1_fail_raw_qc_pass)}, "
            f"both fail {int(row.q1_fail_raw_qc_fail)}"
            for row in quality_agreement.itertuples(index=False)
        )
    validation_summary = json.loads(
        (output_dir / "validation_summary.json").read_text(encoding="utf-8")
    )
    matched_delta = validation_summary.get("q1_matched_delta_vs_legacy", {})
    timing_rows = validation_summary.get("target_timing_sensitivity", [])
    timing_values = [row["mean_balanced_accuracy"] for row in timing_rows if row.get("mean_balanced_accuracy") is not None]
    sign_consistency: dict[str, str] = {}
    for stage in ("cue_locked", "target_locked"):
        per_record = effects.loc[(effects["stage"] == stage) & (effects["record"] != "ALL_RECORDS")]
        consistent = 0
        n_features = 0
        for _, values in per_record.groupby("feature"):
            signs = np.sign(values["right_minus_left"].to_numpy(dtype=float))
            if len(signs) == 4:
                n_features += 1
                consistent += int(np.all(signs != 0) and np.all(signs == signs[0]))
        sign_consistency[stage] = f"{consistent}/{n_features}"

    def score_text(stage: str) -> str:
        result = stage_summaries[stage]
        return (
            f"balanced accuracy {result['mean_balanced_accuracy']:.3f}, "
            f"accuracy {result['mean_accuracy']:.3f}, AUC {result['mean_auc']:.3f}, "
            f"macro-F1 {result['mean_macro_f1']:.3f}"
            if result["mean_balanced_accuracy"] is not None
            else "no valid held-out folds"
        )

    lines = [
        "# Q3 raw EEG validation results and possible explanations",
        "",
        "## Observed result",
        "",
        f"- Raw VisCue trials: {len(trial_table)}; cue epochs passing raw artifact QC: {stage_summaries['cue_locked']['n_qc_pass']}; Q1-clean matched sensitivity sample: {q1_matched_n} (left {q1_left_n}, right {q1_right_n}).",
        f"- Q1/raw-QC agreement: {quality_agreement_text}.",
        f"- Missing primary ERP amplitude values after QC: {missing_values}.",
        f"- Leave-one-record-out cue-side prediction: cue-locked {score_text('cue_locked')}; target-locked {score_text('target_locked')}.",
        f"- On the Q1-matched {q1_cue.shape[0]}-trial sample, the new raw-filter balanced accuracy was {matched_delta.get('cue_locked', {}).get('new_raw_filter_q1_matched_balanced_accuracy'):.3f} for cue and {matched_delta.get('target_locked', {}).get('new_raw_filter_q1_matched_balanced_accuracy'):.3f} for target; changes from the old Q1-clean baseline were {matched_delta.get('cue_locked', {}).get('delta'):+.3f} and {matched_delta.get('target_locked', {}).get('delta'):+.3f}.",
        f"- Target-offset sensitivity (2.0-2.4 s) balanced accuracy ranged from {min(timing_values):.3f} to {max(timing_values):.3f}; this is a timing robustness range, not an offset-selection procedure." if timing_values else "- Target-offset sensitivity: no valid grouped folds.",
        f"- Across-record feature-effect direction agrees in all four records for {sign_consistency['cue_locked']} cue-locked features and {sign_consistency['target_locked']} target-locked features.",
        f"- Channel 9 markers: {response_n}; median time from the assumed cue+2.2 s target anchor is {response_rt_median:.3f} s, with {rt_short_n} under 100 ms. This is not treated as validated RT.",
        f"- Channel 9 choice Logistic, leave-one-record-out balanced accuracy: {behavior_balanced_accuracy:.3f}."
        if behavior_balanced_accuracy is not None
        else f"- Behavior model: {behavior_status.get('status', 'not run')}.",
        f"- Choice baseline comparison: cue/task-code BA {behavior_comparison_ba.get('cue_plus_task_code'):.3f}; cue/task-code+EEG BA {behavior_comparison_ba.get('cue_task_plus_EEG'):.3f}; EEG increment {behavior_comparison.get('EEG_increment_over_cue_task'):+.3f}."
        if behavior_comparison_ba.get("cue_plus_task_code") is not None
        and behavior_comparison_ba.get("cue_task_plus_EEG") is not None
        else "- Choice baseline comparison: not available.",
        f"- Cue/choice same-side fraction by filename task-code candidate: {cue_choice_alignment_text}. This is not correctness; task mapping is unverified."
        if cue_choice_alignment_text
        else "- Cue/choice same-side fraction: not available.",
        "",
        "The main results use raw continuous EEG filtered in separate 0.5-30 Hz ERP and 1-80 Hz time-frequency branches. The Q1-clean epochs are used only for mapping and a matched-sample comparison. Leave-one-record-out is the main validation; it is not leave-one-participant-out because file-to-participant identity is unverified.",
        "",
        "## Plausible explanations if performance is weak (hypotheses)",
        "",
        "1. **Recording shift.** Four held-out records provide only four independent validation groups; electrode offsets, session state, or participant differences can overwhelm cue-locked effects.",
        "2. **Short windows and single-trial noise.** Cue (0.5 s) and target (0.8 s) windows contain few cycles at theta/alpha frequencies, while single-trial frontal ERP peaks are noisy.",
        "3. **Frontal-only coverage.** F3/Fz/F4 cannot measure posterior visual topography or support reliable localization of LGN, hippocampal, or PFC generators.",
        "4. **Target timing assumption.** The target event is not independently marked; the offset sensitivity table should be read as a robustness check, not as a search for the best event time.",
        "5. **Artifact exclusions.** The raw QC removes event windows with hard clipping, non-finite samples, or flatline; remaining unflagged movement or muscle artifacts may still reduce signal quality.",
        "6. **Unverified file/task mapping.** Task-1/Task-2 is retained as a filename code candidate only; project semantics and participant identity are not assumed.",
        "7. **No added choice information from EEG in this split.** The cue/task-code baseline and cue/task-code-plus-EEG balanced accuracies should be compared directly. If they are similar, this may mean the three-channel proxies add little across recordings, or that the provisional channel-9 label is strongly tied to cue/task coding; it does not prove a cognitive mechanism.",
        "",
        "These are hypotheses, not established causes. PLV and theta-gamma PAC are marked unavailable because the short windows do not support reliable estimates without surrogate validation. ICA is not claimed: the available signal has three EEG electrodes and no separate EOG reference.",
        "",
        "## Interpretation boundary",
        "",
        "Channel 9 supplies response direction and event time only; it is never an EEG feature or V/H/P input. Correctness and omission rate remain unverified because the response event semantics, target-side truth, and deadline are not independently established. The DDM is skipped because the assumed target-time anchor yields implausibly short RTs. V/H/P are anchored functional proxies, not localized brain sources.",
        "",
    ]
    (output_dir / "validation_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = ensure_output_dir()
    feature_path = output_dir / "trial_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError("Run 03_extract_features.py first to create trial_features.csv")
    features = pd.read_csv(feature_path, encoding="utf-8-sig")

    feature_columns = [
        *[f"erp_mean_{channel}" for channel in EEG_CHANNELS],
        *[f"log_theta_power_{channel}" for channel in EEG_CHANNELS],
        *[f"log_alpha_power_{channel}" for channel in EEG_CHANNELS],
        *[f"log_beta_power_{channel}" for channel in EEG_CHANNELS],
        "AI_alpha",
    ]
    analysis_features = features.loc[
        features["stage"].isin(("cue_locked", "target_locked"))
        & features["qc_valid"].fillna(False).astype(bool)
    ].copy()
    q1_matched_features = analysis_features.loc[
        analysis_features["q1_quality_pass"].fillna(False).astype(bool)
    ].copy()
    effects = descriptive_effects(analysis_features, feature_columns)
    write_csv(effects, output_dir / "eeg_condition_effects.csv")

    all_predictions: list[pd.DataFrame] = []
    all_folds: list[dict] = []
    q1_matched_stage_summaries: dict[str, dict] = {}
    for stage in ("cue_locked", "target_locked"):
        predictions, folds = validate_stage(analysis_features, stage, feature_columns)
        if not predictions.empty:
            all_predictions.append(predictions)
        all_folds.extend(folds)
        _, matched_folds = validate_stage(q1_matched_features, stage, feature_columns)
        valid_matched = [row for row in matched_folds if row.get("status") == "ok"]
        q1_matched_stage_summaries[stage] = {
            "n_qc_pass_q1_trials": int((q1_matched_features["stage"] == stage).sum()),
            "valid_record_folds": len(valid_matched),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in valid_matched])) if valid_matched else None,
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in valid_matched])) if valid_matched else None,
            "mean_auc": float(np.mean([row["auc"] for row in valid_matched])) if valid_matched else None,
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in valid_matched])) if valid_matched else None,
            "folds": matched_folds,
        }
    predictions_table = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame(
            columns=["record", "original_trial_index", "stage", "true_cue_side", "predicted_cue_side"]
        )
    )
    write_csv(predictions_table, output_dir / "eeg_heldout_record_predictions.csv")
    write_csv(pd.DataFrame(all_folds), output_dir / "eeg_heldout_record_metrics.csv")

    within_record_tables = [
        within_record_diagnostic(analysis_features, stage, feature_columns)
        for stage in ("cue_locked", "target_locked")
    ]
    within_record = pd.concat(within_record_tables, ignore_index=True)
    write_csv(within_record, output_dir / "eeg_within_record_diagnostic.csv")

    stage_summaries: dict[str, dict] = {}
    for stage in ("cue_locked", "target_locked"):
        valid = [row for row in all_folds if row.get("stage") == stage and row.get("status") == "ok"]
        stage_summaries[stage] = {
            "n_qc_pass": int((analysis_features["stage"] == stage).sum()),
            "n_q1_matched": int((q1_matched_features["stage"] == stage).sum()),
            "valid_record_folds": len(valid),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in valid])) if valid else None,
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in valid]))
            if valid
            else None,
            "mean_auc": float(np.mean([row["auc"] for row in valid])) if valid else None,
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in valid])) if valid else None,
            "folds": valid,
        }

    sensitivity_rows: list[dict] = []
    sensitivity_variants = sorted(
        features.loc[features["stage"] == "target_offset_sensitivity", "analysis_variant"].dropna().unique()
    )
    for variant in sensitivity_variants:
        variant_predictions, variant_folds = validate_stage(
            features, "target_offset_sensitivity", feature_columns, variant=variant
        )
        valid_variant = [row for row in variant_folds if row.get("status") == "ok"]
        sensitivity_rows.append(
            {
                "analysis_variant": variant,
                "target_offset_s": float(variant.rsplit("_", 1)[-1].replace("s", "")),
                "n_qc_pass_trials": int(len(variant_predictions)),
                "valid_record_folds": len(valid_variant),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in valid_variant])) if valid_variant else None,
                "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in valid_variant])) if valid_variant else None,
                "mean_auc": float(np.mean([row["auc"] for row in valid_variant])) if valid_variant else None,
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in valid_variant])) if valid_variant else None,
                "folds": variant_folds,
            }
        )
    nominal_target = stage_summaries["target_locked"]
    sensitivity_rows.append(
        {
            "analysis_variant": "target_offset_2.2s",
            "target_offset_s": 2.2,
            "n_qc_pass_trials": int(nominal_target["n_qc_pass"]),
            "valid_record_folds": int(nominal_target["valid_record_folds"]),
            "mean_accuracy": nominal_target["mean_accuracy"],
            "mean_balanced_accuracy": nominal_target["mean_balanced_accuracy"],
            "mean_auc": nominal_target["mean_auc"],
            "mean_macro_f1": nominal_target["mean_macro_f1"],
            "folds": nominal_target["folds"],
        }
    )
    sensitivity_rows.sort(key=lambda row: row["target_offset_s"])
    sensitivity_frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "folds"}
            for row in sensitivity_rows
        ]
    )
    write_csv(sensitivity_frame, output_dir / "target_timing_sensitivity.csv")

    behavior_status_path = output_dir / "behavior_model_status.json"
    if behavior_status_path.exists():
        with behavior_status_path.open("r", encoding="utf-8") as stream:
            behavior_status = json.load(stream)
    else:
        behavior_status = {"status": "not_run; run 05_behavior_model.py"}

    summary = {
        "EEG_validation_target": "VisCue direction (-1/+1), not choice, correctness, RT, or omission",
        "feature_source": "raw continuous F3/Fz/F4; ERP 0.5-30 Hz branch and TF 1-80 Hz branch; Q1 clean is mapping/quality reference only",
        "artifact_qc": "event epochs flagged for non-finite samples, a flat channel, or absolute amplitude >= 999.5 raw data units; branch and joint pass counts are in raw_feature_qc_summary.json",
        "CV": "leave one recording file out; all scaling estimated from training records only",
        "record_identity": "the four recording files are held out individually; independent participant grouping is unverified",
        "feature_count": len(feature_columns),
        "stage_results": stage_summaries,
        "q1_quality_matched_sensitivity": q1_matched_stage_summaries,
        "legacy_q1_baseline_reference": json.loads((output_dir / "legacy_q1_baseline_reference.json").read_text(encoding="utf-8")),
        "target_timing_sensitivity": sensitivity_rows,
        "within_record_diagnostic": within_record.to_dict(orient="records"),
        "within_record_diagnostic_caveat": "random trial folds share each recording/session across train and test and can be optimistic; do not use as the main generalization result",
        "behavior_validation": behavior_status,
        "target_locking": "nominal event is fixed cue + 2.2 s by schedule assumption; sensitivity offsets span 2.0-2.4 s",
        "channel_9_scope": "used for event/response direction and endpoint metadata only; excluded from EEG feature arrays and predictors",
        "interpretation_limit": "EEG classification estimates cue-side decodability; behavior logistic predicts channel-9 choice and is not a correctness or clinical diagnosis result",
    }
    baseline = summary["legacy_q1_baseline_reference"].get("stage_results", {})
    summary["q1_matched_delta_vs_legacy"] = {
        stage: {
            "new_raw_filter_q1_matched_balanced_accuracy": q1_matched_stage_summaries[stage]["mean_balanced_accuracy"],
            "legacy_q1_clean_balanced_accuracy": baseline.get(stage, {}).get("mean_balanced_accuracy"),
            "delta": (
                q1_matched_stage_summaries[stage]["mean_balanced_accuracy"]
                - baseline[stage]["mean_balanced_accuracy"]
            )
            if q1_matched_stage_summaries[stage]["mean_balanced_accuracy"] is not None
            and stage in baseline
            and baseline[stage].get("mean_balanced_accuracy") is not None
            else None,
            "n_q1_matched_qc_trials": q1_matched_stage_summaries[stage]["n_qc_pass_q1_trials"],
        }
        for stage in ("cue_locked", "target_locked")
    }
    write_json(summary, output_dir / "validation_summary.json")
    write_interpretation(
        output_dir,
        features,
        effects,
        stage_summaries,
        within_record,
        behavior_status,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, stage in zip(axes, ("cue_locked", "target_locked")):
        subset = predictions_table.loc[predictions_table["stage"] == stage]
        if subset.empty:
            ax.text(0.5, 0.5, "No valid held-out folds", ha="center", va="center")
            ax.set_axis_off()
            continue
        actual = subset["true_cue_side"].to_numpy(dtype=int)
        predicted = subset["predicted_cue_side"].to_numpy(dtype=int)
        matrix = np.asarray(
            [[np.sum((actual == a) & (predicted == p)) for p in (-1, 1)] for a in (-1, 1)]
        )
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(int(value)), ha="center", va="center")
        ax.set_xticks([0, 1], ["Left cue", "Right cue"])
        ax.set_yticks([0, 1], ["Left cue", "Right cue"])
        ax.set_xlabel("Predicted cue direction")
        ax.set_ylabel("Observed VisCue direction")
        ax.set_title(stage.replace("_", " "))
    fig.suptitle("Held-out-record EEG cue-side prediction")
    fig.savefig(output_dir / "eeg_heldout_record_confusion.png", dpi=180)
    plt.close(fig)

    print(f"Wrote EEG condition effects for {len(features)} trial-stage rows.")
    for stage, result in stage_summaries.items():
        print(
            f"{stage}: balanced accuracy={result['mean_balanced_accuracy']}, "
            f"valid held-out records={result['valid_record_folds']}"
        )


if __name__ == "__main__":
    main()
