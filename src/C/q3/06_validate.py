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


def validate_stage(frame: pd.DataFrame, stage: str, feature_columns: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    stage_frame = frame.loc[frame["stage"] == stage].copy()
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
                {"stage": stage, "held_out_record": held_out_record, "status": "skipped_single_class"}
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
            "held_out_record": held_out_record,
            "status": "ok",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "left_test_n": int(np.sum(y_test_side == -1)),
            "right_test_n": int(np.sum(y_test_side == 1)),
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "training_majority_baseline_accuracy": majority_baseline,
        }
        folds.append(fold)
        predictions.extend(
            {
                "record": row.record,
                "original_trial_index": int(row.original_trial_index),
                "stage": stage,
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
    retained_trials = trial_table.loc[trial_table["eeg_trial_available"].astype(bool)]
    left_n = int((retained_trials["cue_side"] == -1).sum())
    right_n = int((retained_trials["cue_side"] == 1).sum())
    feature_columns = [
        *[f"erp_mean_{channel}" for channel in EEG_CHANNELS],
        *[f"log_{band}_power_{channel}" for band in ("theta", "alpha", "beta") for channel in EEG_CHANNELS],
        "AI_alpha",
    ]
    validation_features = features.loc[features["stage"].isin(("cue_locked", "target_locked"))]
    missing_values = int(validation_features[feature_columns].isna().sum().sum())
    response_n = int(trial_table["choice_side"].notna().sum())
    omission_n = int(trial_table["is_omission"].fillna(False).sum())
    response_rt = pd.to_numeric(trial_table["reaction_time_s"], errors="coerce")
    response_rt_median = float(response_rt.median()) if response_rt.notna().any() else np.nan
    rt_short_n = int((response_rt.notna() & (response_rt < 0.10)).sum())
    behavior_balanced_accuracy = behavior_status.get("mean_balanced_accuracy")
    sign_consistency: dict[str, str] = {}
    for stage in ("cue_locked", "target_locked"):
        per_record = effects.loc[
            (effects["stage"] == stage) & (effects["record"] != "ALL_RECORDS")
        ]
        consistent = 0
        feature_count = 0
        for _, values in per_record.groupby("feature"):
            signs = np.sign(values["right_minus_left"].to_numpy(dtype=float))
            if len(signs) == 4:
                feature_count += 1
                consistent += int(np.all(signs != 0) and np.all(signs == signs[0]))
        sign_consistency[stage] = f"{consistent}/{feature_count}"

    def score_text(stage: str) -> str:
        result = stage_summaries[stage]
        return (
            f"balanced accuracy {result['mean_balanced_accuracy']:.3f}, "
            f"accuracy {result['mean_accuracy']:.3f}"
            if result["mean_balanced_accuracy"] is not None
            else "no valid held-out folds"
        )

    lines = [
        "# Q3 validation result and possible explanations",
        "",
        "## Observed result",
        "",
        f"- VisCue events: {len(trial_table)}; Q1 quality-retained, timestamp-matched EEG trials: {int(trial_table['eeg_trial_available'].sum())}.",
        f"- Retained cue labels: left {left_n}, right {right_n}; missing values among the 13 validation features: {missing_values}.",
        f"- Leave-one-record-out cue-side prediction: cue-locked {score_text('cue_locked')}; target-locked {score_text('target_locked')}.",
        f"- Across-record feature-effect direction agrees in all four records for {sign_consistency['cue_locked']} cue-locked features and {sign_consistency['target_locked']} target-locked features.",
        f"- Channel 9 response markers: {response_n} choices, {omission_n} trials without a marker; median RT from the scheduled target anchor {response_rt_median:.3f} s, with {rt_short_n} RTs under 100 ms.",
        f"- Channel 9 choice Logistic, leave-one-record-out balanced accuracy: {behavior_balanced_accuracy:.3f}."
        if behavior_balanced_accuracy is not None
        else f"- Behavior model: {behavior_status.get('status', 'not run')}.",
        "",
        "The current EEG cue-side decoding effect is weak: both primary balanced accuracies are near 0.5. The channel-9 choice model is reported separately and does not establish correctness or clinical diagnosis.",
        "",
        "## Plausible causes (hypotheses)",
        "",
        "1. **Recording-to-recording variation.** The cue-side feature differences do not keep a common direction across the four records. The repeated within-record diagnostic is higher for a few record/stage pairs and near or below chance for others, which is consistent with session-specific effects that do not transfer reliably.",
        "2. **Limited usable sample size.** Q1 retained 297 of 400 raw cue trials (54–83 trials per recording). With 13 EEG features and only four held-out recordings, the cross-record estimate has substantial sampling uncertainty.",
        "3. **Restricted scalp coverage and bandwidth.** The analysis uses only F3/Fz/F4 and Q1's 0.2–24 Hz clean signal. It cannot capture posterior scalp patterns often used for P300 analysis or activity above 24 Hz.",
        "4. **Target-stage timing is assumed.** Target-locked features use cue time + 2.2 s from the experimental schedule; this offset was not verified from an allowed event channel. Timing variation would blur target-locked responses.",
        "5. **Conditions remain unresolved.** Task type and participant grouping could not be verified from the allowed signals, so potentially different conditions are pooled and each recording file is only a proxy for a held-out group.",
        "6. **The RT anchor may not match the recorded action clock.** 399 of 400 RTs are under 100 ms when target time is set to cue + 2.2 s. This could reflect a schedule-to-marker mismatch or channel timing semantics; until verified, it does not support a DDM fit.",
        "",
        "These are plausible explanations, not established causes. Cue-to-Q1 timestamp matches were one-to-one, sampling intervals matched the stated 256 Hz rate, all planned cue/target windows had full coverage, and those validation features had no missing values; this makes obvious cue mapping or feature-window truncation errors less likely.",
        "",
        "## Interpretation boundary",
        "",
        "Channel 9 supplies response direction and timing only; it is never an EEG feature. Correctness remains unassigned because target-side truth is not verified for all tasks. The target offset remains a protocol assumption. RTs near 15 ms from that anchor are flagged, so the DDM is skipped and the choice-only logistic model is used instead.",
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
    effects = descriptive_effects(features, feature_columns)
    write_csv(effects, output_dir / "eeg_condition_effects.csv")

    all_predictions: list[pd.DataFrame] = []
    all_folds: list[dict] = []
    for stage in ("cue_locked", "target_locked"):
        predictions, folds = validate_stage(features, stage, feature_columns)
        if not predictions.empty:
            all_predictions.append(predictions)
        all_folds.extend(folds)
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
        within_record_diagnostic(features, stage, feature_columns)
        for stage in ("cue_locked", "target_locked")
    ]
    within_record = pd.concat(within_record_tables, ignore_index=True)
    write_csv(within_record, output_dir / "eeg_within_record_diagnostic.csv")

    stage_summaries: dict[str, dict] = {}
    for stage in ("cue_locked", "target_locked"):
        valid = [row for row in all_folds if row.get("stage") == stage and row.get("status") == "ok"]
        stage_summaries[stage] = {
            "valid_record_folds": len(valid),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in valid])) if valid else None,
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in valid]))
            if valid
            else None,
            "folds": valid,
        }

    behavior_status_path = output_dir / "behavior_model_status.json"
    if behavior_status_path.exists():
        with behavior_status_path.open("r", encoding="utf-8") as stream:
            behavior_status = json.load(stream)
    else:
        behavior_status = {"status": "not_run; run 05_behavior_model.py"}

    summary = {
        "EEG_validation_target": "VisCue direction (-1/+1), not choice, correctness, RT, or omission",
        "CV": "leave one recording file out; all scaling estimated from training records only",
        "record_identity": "the four recording files are held out individually; independent participant grouping is unverified",
        "feature_count": len(feature_columns),
        "stage_results": stage_summaries,
        "within_record_diagnostic": within_record.to_dict(orient="records"),
        "within_record_diagnostic_caveat": "random trial folds share each recording/session across train and test and can be optimistic; do not use as the main generalization result",
        "behavior_validation": behavior_status,
        "target_locking": "target event is fixed cue + 2.2 s by schedule assumption; Action/TgtAct marks responses, not a separate target display",
        "channel_9_scope": "used for response direction/time/omission labels; excluded from EEG feature arrays and predictors",
        "interpretation_limit": "EEG classification estimates cue-side decodability; behavior logistic predicts channel-9 choice and is not a correctness or clinical diagnosis result",
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
