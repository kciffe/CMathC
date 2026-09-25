"""Model channel-9 choices and report whether RTs support a DDM."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from common import ensure_output_dir, write_csv, write_json
from config import MIN_DDM_RT_S


STATE_FEATURES = ["V_visual_like_raw", "H_memory_like_raw", "P_control_like_raw"]


def parse_choice(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric in (-2.0, -1.0, 1.0, 2.0):
            return float(np.sign(numeric))
    text = str(value).strip().lower()
    if text in {"left", "l", "-1"}:
        return -1.0
    if text in {"right", "r", "+1", "1"}:
        return 1.0
    return np.nan


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
        raise RuntimeError(f"Choice logistic optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def grouped_choice_cv(
    frame: pd.DataFrame,
    predictor_columns: list[str] | None = None,
    model_name: str = "EEG_state_only",
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    predictor_columns = predictor_columns or STATE_FEATURES
    predictions: list[dict] = []
    folds: list[dict] = []
    parameter_rows: list[dict] = []
    for held_out_record in sorted(frame["record"].unique()):
        train = frame.loc[frame["record"] != held_out_record]
        test = frame.loc[frame["record"] == held_out_record]
        if train["choice_side"].nunique() < 2 or test["choice_side"].nunique() < 2:
            folds.append(
                {"held_out_record": held_out_record, "status": "skipped_single_class"}
            )
            continue
        x_train_raw = train[predictor_columns].to_numpy(dtype=float)
        x_test_raw = test[predictor_columns].to_numpy(dtype=float)
        center = np.mean(x_train_raw, axis=0)
        scale = np.std(x_train_raw, axis=0)
        scale[~np.isfinite(scale) | (scale <= np.finfo(float).eps)] = 1.0
        x_train = (x_train_raw - center) / scale
        x_test = (x_test_raw - center) / scale
        y_train = (train["choice_side"].to_numpy(dtype=float) > 0).astype(float)
        y_test = (test["choice_side"].to_numpy(dtype=float) > 0).astype(int)
        coefficients = fit_logistic(x_train, y_train)
        parameter_rows.append(
            {
                "held_out_record": held_out_record,
                "parameter": "intercept",
                "coefficient": float(coefficients[0]),
                "training_center": np.nan,
                "training_scale": np.nan,
                "model": model_name,
            }
        )
        for feature_index, feature_name in enumerate(predictor_columns):
            parameter_rows.append(
                {
                    "held_out_record": held_out_record,
                    "parameter": feature_name,
                    "coefficient": float(coefficients[feature_index + 1]),
                    "training_center": float(center[feature_index]),
                    "training_scale": float(scale[feature_index]),
                    "model": model_name,
                }
            )
        probability = expit(np.column_stack([np.ones(len(test)), x_test]) @ coefficients)
        predicted = (probability >= 0.5).astype(int)
        actual_side = np.where(y_test > 0, 1, -1)
        predicted_side = np.where(predicted > 0, 1, -1)
        predictions.extend(
            {
                "record": row.record,
                "original_trial_index": int(row.original_trial_index),
                "choice_side": int(row.choice_side),
                "predicted_choice_side": int(predicted_side[index]),
                "probability_right": float(probability[index]),
                "held_out_record": held_out_record,
                "model": model_name,
            }
            for index, row in enumerate(test.itertuples(index=False))
        )
        true_positive = int(np.sum((actual_side == 1) & (predicted_side == 1)))
        true_negative = int(np.sum((actual_side == -1) & (predicted_side == -1)))
        positive_n = int(np.sum(actual_side == 1))
        negative_n = int(np.sum(actual_side == -1))
        accuracy = float(np.mean(actual_side == predicted_side))
        balanced_accuracy = 0.5 * (
            true_positive / positive_n + true_negative / negative_n
        )
        folds.append(
            {
                "held_out_record": held_out_record,
                "status": "ok",
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "accuracy": accuracy,
                "balanced_accuracy": float(balanced_accuracy),
                "model": model_name,
            }
        )
    return pd.DataFrame(predictions), folds, pd.DataFrame(parameter_rows)


def main() -> None:
    output_dir = ensure_output_dir()
    status_path = output_dir / "behavior_model_status.json"
    trial_path = output_dir / "trial_table.csv"
    state_path = output_dir / "state_scores.csv"
    if not trial_path.exists() or not state_path.exists():
        raise FileNotFoundError("Run scripts 02_extract_trials.py and 04_cognitive_state.py first")

    trials = pd.read_csv(trial_path, encoding="utf-8-sig")
    state = pd.read_csv(state_path, encoding="utf-8-sig")
    cue_state = state.loc[state["stage"] == "cue_locked"].copy()
    key_columns = ["record", "original_trial_index"]
    model_data = trials.merge(
        cue_state[
            key_columns
            + STATE_FEATURES
            + ["qc_valid", "filename_task_code_candidate"]
        ],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    model_data["choice_side"] = model_data["choice_side"].map(parse_choice)
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data["cue_by_task_code_candidate"] = model_data["cue_side"] * (
        pd.to_numeric(model_data["filename_task_code_candidate"], errors="coerce") - 1.5
    )
    baseline_predictors = [
        "cue_side",
        "filename_task_code_candidate",
        "cue_by_task_code_candidate",
    ]
    combined_predictors = [*baseline_predictors, *STATE_FEATURES]
    model_data = model_data.dropna(subset=["choice_side", *combined_predictors]).copy()
    model_data = model_data.loc[model_data["qc_valid"].fillna(False).astype(bool)]
    q1_matched_data = model_data.loc[model_data["q1_quality_pass"].fillna(False).astype(bool)].copy()

    rt_values = pd.to_numeric(trials["reaction_time_s"], errors="coerce")
    plausible_rt = rt_values.ge(MIN_DDM_RT_S) & np.isfinite(rt_values)
    rt_counts = {
        "response_trials": int(trials["choice_side"].notna().sum()),
        "no_response_marker_trials": int(trials["is_omission"].fillna(False).sum()),
        "rt_observed_trials": int(rt_values.notna().sum()),
        "rt_at_least_minimum_for_ddm": int(plausible_rt.sum()),
        "rt_below_minimum_for_ddm": int((rt_values.notna() & ~plausible_rt).sum()),
        "rt_median_s": float(rt_values.median()) if rt_values.notna().any() else None,
        "minimum_rt_gate_s": MIN_DDM_RT_S,
        "deadline_known": bool(pd.to_numeric(trials.get("deadline_s"), errors="coerce").notna().any())
        if "deadline_s" in trials
        else False,
    }
    cue_choice_alignment = []
    for task_code, group in model_data.groupby("filename_task_code_candidate", dropna=False):
        cue_choice_alignment.append(
            {
                "filename_task_code_candidate": task_code,
                "n_trials": int(len(group)),
                "n_cue_choice_same_side": int((group["cue_side"] == group["choice_side"]).sum()),
                "cue_choice_same_side_fraction": float(
                    (group["cue_side"] == group["choice_side"]).mean()
                ),
                "interpretation": "descriptive only; not response correctness; filename task mapping unverified",
            }
        )
    write_csv(
        pd.DataFrame(cue_choice_alignment),
        output_dir / "behavior_cue_choice_alignment.csv",
    )

    if (
        len(model_data) < 12
        or model_data["choice_side"].nunique() < 2
        or model_data["record"].nunique() < 2
    ):
        status = {
            "status": "insufficient_raw_qc_passed_choice_labels",
            "n_joined_trials": int(len(model_data)),
            "n_choice_classes": int(model_data["choice_side"].nunique()),
            "n_records": int(model_data["record"].nunique()),
            "reason": "Need at least 12 raw-QC-passed trials, both response choices, and at least two recordings for grouped validation.",
            "rt_quality": rt_counts,
        }
        write_json(status, status_path)
        print("Behavior model skipped: too few matched response labels or only one choice class.")
        return

    baseline_predictions, baseline_folds, baseline_parameters = grouped_choice_cv(
        model_data, baseline_predictors, "cue_plus_task_code"
    )
    state_predictions, state_folds, state_parameters = grouped_choice_cv(
        model_data, STATE_FEATURES, "EEG_state_only"
    )
    predictions, folds, parameters = grouped_choice_cv(
        model_data, combined_predictors, "cue_task_plus_EEG"
    )
    write_csv(predictions, output_dir / "behavior_choice_predictions.csv")
    write_csv(
        pd.concat([baseline_parameters, state_parameters, parameters], ignore_index=True),
        output_dir / "behavior_model_fold_parameters.csv",
    )
    comparison_folds = [*baseline_folds, *state_folds, *folds]
    write_csv(pd.DataFrame(comparison_folds), output_dir / "behavior_model_comparison.csv")
    write_csv(
        pd.concat([baseline_predictions, state_predictions, predictions], ignore_index=True),
        output_dir / "behavior_model_comparison_predictions.csv",
    )
    valid_folds = [fold for fold in folds if fold["status"] == "ok"]
    valid_baseline = [fold for fold in baseline_folds if fold["status"] == "ok"]
    valid_state = [fold for fold in state_folds if fold["status"] == "ok"]
    q1_match_results: dict = {}
    if (
        len(q1_matched_data) >= 12
        and q1_matched_data["choice_side"].nunique() >= 2
        and q1_matched_data["record"].nunique() >= 2
    ):
        for name, predictors in (
            ("cue_plus_task_code", baseline_predictors),
            ("EEG_state_only", STATE_FEATURES),
            ("cue_task_plus_EEG", combined_predictors),
        ):
            _, q1_match_folds, _ = grouped_choice_cv(q1_matched_data, predictors, name)
            q1_valid = [fold for fold in q1_match_folds if fold["status"] == "ok"]
            q1_match_results[name] = {
                "n_trials": int(len(q1_matched_data)),
                "valid_record_folds": len(q1_valid),
                "mean_balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in q1_valid]))
                if q1_valid
                else None,
                "folds": q1_match_folds,
            }
    status = {
        "status": "choice_logistic_comparison_fitted_ddm_skipped_rt_quality_gate",
        "model": "L2-regularized binary logistic regression",
        "n_joined_trials": int(len(model_data)),
        "raw_qc_passed_trials": int(len(model_data)),
        "q1_quality_matched_sensitivity": q1_match_results,
        "n_records": int(model_data["record"].nunique()),
        "predictors": {
            "cue_plus_task_baseline": baseline_predictors,
            "EEG_state_only": STATE_FEATURES,
            "cue_task_plus_EEG": combined_predictors,
        },
        "target": "channel 9 response direction normalized to choice side -1/+1",
        "response_code_normalization": "raw negative -> -2; raw zero -> 0; raw positive -> +2; raw values retained separately",
        "rt_quality": rt_counts,
        "ddm_status": "not_fitted: RT quality gate fails and no verified deadline is available",
        "ddm_reason": "This implementation uses a choice-only logistic fallback. RTs relative to the scheduled cue+2.2 s target are mostly below 100 ms; the target-time anchor is not separately marked.",
        "cross_validation": "leave-one-record-out; standardization fitted on training records only; primary set uses raw-signal QC, with Q1-retained sensitivity subset",
        "model_comparison": {
            "folds": comparison_folds,
            "cue_choice_alignment_by_filename_task_code": cue_choice_alignment,
            "mean_balanced_accuracy": {
                "cue_plus_task_code": float(np.mean([fold["balanced_accuracy"] for fold in valid_baseline])) if valid_baseline else None,
                "EEG_state_only": float(np.mean([fold["balanced_accuracy"] for fold in valid_state])) if valid_state else None,
                "cue_task_plus_EEG": float(np.mean([fold["balanced_accuracy"] for fold in valid_folds])) if valid_folds else None,
            },
            "EEG_increment_over_cue_task": (
                float(np.mean([fold["balanced_accuracy"] for fold in valid_folds]))
                - float(np.mean([fold["balanced_accuracy"] for fold in valid_baseline]))
            )
            if valid_folds and valid_baseline
            else None,
            "task_code_source": "filename suffix only; formal Project-1/Project-2 mapping unverified",
        },
        "folds": folds,
        "mean_balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in valid_folds]))
        if valid_folds
        else None,
        "correctness": "left unassigned; target-side truth is not available for every task",
        "omissions": "all trials have a channel-9 marker; this does not establish that the protocol had no behavioral omissions",
        "pre_response_EEG": "not used as a predictor because its endpoint depends on the response/event time",
    }
    write_json(status, status_path)
    print(
        f"Choice models used {len(model_data)} raw-QC channel-9-labeled trials; "
        f"cue/task BA={status['model_comparison']['mean_balanced_accuracy']['cue_plus_task_code']:.3f}, "
        f"cue/task+EEG BA={status['model_comparison']['mean_balanced_accuracy']['cue_task_plus_EEG']:.3f}."
    )


if __name__ == "__main__":
    main()
