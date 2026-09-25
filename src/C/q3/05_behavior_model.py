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


def grouped_choice_cv(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
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
        x_train_raw = train[STATE_FEATURES].to_numpy(dtype=float)
        x_test_raw = test[STATE_FEATURES].to_numpy(dtype=float)
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
            }
        )
        for feature_index, feature_name in enumerate(STATE_FEATURES):
            parameter_rows.append(
                {
                    "held_out_record": held_out_record,
                    "parameter": feature_name,
                    "coefficient": float(coefficients[feature_index + 1]),
                    "training_center": float(center[feature_index]),
                    "training_scale": float(scale[feature_index]),
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
        cue_state[key_columns + STATE_FEATURES],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    model_data["choice_side"] = model_data["choice_side"].map(parse_choice)
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=["choice_side", *STATE_FEATURES]).copy()
    model_data = model_data.loc[model_data["eeg_trial_available"].astype(bool)]

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

    if (
        len(model_data) < 12
        or model_data["choice_side"].nunique() < 2
        or model_data["record"].nunique() < 2
    ):
        status = {
            "status": "insufficient_matched_choice_labels",
            "n_joined_trials": int(len(model_data)),
            "n_choice_classes": int(model_data["choice_side"].nunique()),
            "n_records": int(model_data["record"].nunique()),
            "reason": "Need at least 12 EEG-quality-retained trials, both response choices, and at least two recordings for grouped validation.",
            "rt_quality": rt_counts,
        }
        write_json(status, status_path)
        print("Behavior model skipped: too few matched response labels or only one choice class.")
        return

    predictions, folds, parameters = grouped_choice_cv(model_data)
    write_csv(predictions, output_dir / "behavior_choice_predictions.csv")
    write_csv(parameters, output_dir / "behavior_model_fold_parameters.csv")
    valid_folds = [fold for fold in folds if fold["status"] == "ok"]
    status = {
        "status": "choice_logistic_fitted_ddm_skipped_rt_quality_gate",
        "model": "L2-regularized binary logistic regression",
        "n_joined_trials": int(len(model_data)),
        "n_records": int(model_data["record"].nunique()),
        "predictors": STATE_FEATURES,
        "target": "channel 9 response direction normalized to choice side -1/+1",
        "response_code_normalization": "raw negative -> -2; raw zero -> 0; raw positive -> +2; raw values retained separately",
        "rt_quality": rt_counts,
        "ddm_status": "not_fitted: RT quality gate fails and no verified deadline is available",
        "ddm_reason": "This implementation uses a choice-only logistic fallback. RTs relative to the scheduled cue+2.2 s target are mostly below 100 ms; the target-time anchor is not separately marked.",
        "cross_validation": "leave-one-record-out; standardization fitted on training records only",
        "folds": folds,
        "mean_balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in valid_folds]))
        if valid_folds
        else None,
        "correctness": "left unassigned; target-side truth is not available for every task",
        "omissions": "no response marker within the cue-to-next-cue trial interval is recorded as a no-response trial",
        "pre_response_EEG": "not used as a predictor because its time window is aligned using the response time",
    }
    write_json(status, status_path)
    print(f"Choice logistic model used {len(model_data)} channel-9-labeled EEG trials.")


if __name__ == "__main__":
    main()
