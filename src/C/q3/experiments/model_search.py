"""Exploratory nested grouped-CV model search for Q3 EEG features.

This script is isolated from the primary Q3 pipeline. It evaluates a small,
predeclared set of classifiers and feature representations. Hyperparameters and
the selected candidate are chosen only in inner leave-one-record-out folds.
The outer held-out record is used once for the final per-fold score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC, LinearSVC


Q3_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Q3_DIR / "baselines" / "20260925_initial" / "output" / "trial_features.csv"
DEFAULT_OUTPUT = Q3_DIR / "output" / "experiments" / "20260925_model_search_v1"
RANDOM_SEED = 20260925

CHANNELS = ("F3", "Fz", "F4")
ERP_FEATURES = tuple(f"erp_mean_{channel}" for channel in CHANNELS)
TF_FEATURES = tuple(
    f"log_{band}_power_{channel}"
    for band in ("theta", "alpha", "beta")
    for channel in CHANNELS
)
BASELINE_FEATURES = ERP_FEATURES + TF_FEATURES + ("AI_alpha",)

FEATURE_SETS = {
    "full_baseline_13": BASELINE_FEATURES,
    "drop_redundant_alpha_asym_12": ERP_FEATURES + TF_FEATURES,
    "erp_only_3": ERP_FEATURES,
    "bandpower_only_9": TF_FEATURES,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    feature_set: str
    scaler: str
    parameter: float | str | None = None
    gamma: float | str | None = None

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["candidate"] = self.name
        row["parameters"] = json.dumps(
            {"parameter": self.parameter, "gamma": self.gamma}, ensure_ascii=False
        )
        return row


def grouped_leave_one_group_out(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Indices for leave-one-record-out folds, keeping every group intact."""
    group_values = np.asarray(groups).astype(str)
    folds = []
    for held_out in sorted(np.unique(group_values)):
        test_index = np.flatnonzero(group_values == held_out)
        train_index = np.flatnonzero(group_values != held_out)
        if not train_index.size or not test_index.size:
            continue
        folds.append((train_index, test_index, held_out))
    return folds


def scale_train_test(
    x_train: np.ndarray, x_test: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale from training rows only; apply unchanged parameters to test rows."""
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train/test arrays must be 2D with matching feature counts")
    if method == "standard":
        center = np.mean(x_train, axis=0)
        scale = np.std(x_train, axis=0)
    elif method == "robust":
        center = np.median(x_train, axis=0)
        scale = np.quantile(x_train, 0.75, axis=0) - np.quantile(x_train, 0.25, axis=0)
    elif method == "none":
        center = np.zeros(x_train.shape[1], dtype=float)
        scale = np.ones(x_train.shape[1], dtype=float)
    else:
        raise ValueError(f"Unknown scaler: {method}")
    invalid = ~np.isfinite(scale) | (scale <= np.finfo(float).eps)
    scale[invalid] = 1.0
    return (x_train - center) / scale, (x_test - center) / scale, center, scale


def fit_logistic_scores(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, l2: float
) -> np.ndarray:
    """Match the current pipeline's regularized logistic objective."""
    design = np.column_stack([np.ones(len(x_train)), x_train])
    y_binary = (np.asarray(y_train) > 0).astype(float)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ parameters
        loss = np.logaddexp(0.0, logits).sum() - np.dot(y_binary, logits)
        loss += 0.5 * l2 * np.dot(parameters[1:], parameters[1:])
        residual = expit(logits) - y_binary
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
    eval_design = np.column_stack([np.ones(len(x_eval)), x_eval])
    return expit(eval_design @ result.x)


def candidate_grid() -> list[Candidate]:
    """Small, predeclared model set; list order breaks exact inner-score ties."""
    candidates = [
        Candidate("baseline_logistic_l2_1", "logistic", "full_baseline_13", "standard", 1.0)
    ]
    for feature_set in FEATURE_SETS:
        for l2 in (0.1, 10.0, 100.0):
            if feature_set == "full_baseline_13":
                # The fixed-lambda baseline above already covers l2=1 only.
                pass
            candidates.append(
                Candidate(f"logistic_{feature_set}_l2_{l2:g}", "logistic", feature_set, "standard", l2)
            )
        candidates.append(
            Candidate(f"shrinkage_lda_{feature_set}", "lda", feature_set, "standard")
        )

    for feature_set in ("full_baseline_13", "drop_redundant_alpha_asym_12"):
        for l2 in (0.1, 1.0, 10.0):
            candidates.append(
                Candidate(f"robust_logistic_{feature_set}_l2_{l2:g}", "logistic", feature_set, "robust", l2)
            )
        for c_value in (0.01, 0.1, 1.0, 10.0):
            candidates.append(
                Candidate(f"linear_svm_{feature_set}_C_{c_value:g}", "linear_svm", feature_set, "standard", c_value)
            )
        for c_value in (0.1, 1.0, 10.0):
            for gamma in ("scale", 0.03, 0.1):
                candidates.append(
                    Candidate(
                        f"rbf_svm_{feature_set}_C_{c_value:g}_gamma_{gamma}",
                        "rbf_svm",
                        feature_set,
                        "standard",
                        c_value,
                        gamma,
                    )
                )
    return candidates


def predict_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    candidate: Candidate,
) -> tuple[np.ndarray, float]:
    """Return a continuous score and the candidate's fixed decision threshold."""
    if candidate.model == "logistic":
        return fit_logistic_scores(x_train, y_train, x_eval, float(candidate.parameter)), 0.5
    if candidate.model == "lda":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        model.fit(x_train, y_train)
        positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(x_eval)[:, positive_index], 0.5
    if candidate.model == "linear_svm":
        model = LinearSVC(C=float(candidate.parameter), dual="auto", random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        return model.decision_function(x_eval), 0.0
    if candidate.model == "rbf_svm":
        model = SVC(
            C=float(candidate.parameter),
            kernel="rbf",
            gamma=candidate.gamma,
            decision_function_shape="ovr",
        )
        model.fit(x_train, y_train)
        return model.decision_function(x_eval), 0.0
    raise ValueError(f"Unknown model: {candidate.model}")


def score_fold(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    candidate: Candidate,
) -> tuple[dict[str, float], pd.DataFrame]:
    feature_columns = FEATURE_SETS[candidate.feature_set]
    x_train_raw = train_frame.loc[:, feature_columns].to_numpy(dtype=float)
    x_eval_raw = eval_frame.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train_frame["cue_side"].to_numpy(dtype=int)
    y_eval = eval_frame["cue_side"].to_numpy(dtype=int)
    x_train, x_eval, _, _ = scale_train_test(x_train_raw, x_eval_raw, candidate.scaler)
    scores, threshold = predict_scores(x_train, y_train, x_eval, candidate)
    predicted = np.where(scores >= threshold, 1, -1)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, predicted)),
        "auc": float(roc_auc_score((y_eval > 0).astype(int), scores)),
        "macro_f1": float(f1_score(y_eval, predicted, labels=[-1, 1], average="macro", zero_division=0)),
    }
    predictions = pd.DataFrame(
        {
            "original_trial_index": eval_frame["original_trial_index"].to_numpy(dtype=int),
            "record": eval_frame["record"].astype(str).to_numpy(),
            "true_cue_side": y_eval,
            "predicted_cue_side": predicted,
            "score_right": scores,
        }
    )
    return metrics, predictions


def prepare_scope(features: pd.DataFrame, stage: str, sample: str) -> pd.DataFrame:
    frame = features.loc[
        (features["stage"] == stage) & features["qc_valid"].fillna(False).astype(bool)
    ].copy()
    if sample == "q1_quality_matched":
        frame = frame.loc[frame["q1_quality_pass"].fillna(False).astype(bool)].copy()
    elif sample != "raw_qc_pass":
        raise ValueError(f"Unknown sample scope: {sample}")
    columns = sorted({name for feature_set in FEATURE_SETS.values() for name in feature_set})
    frame["cue_side"] = pd.to_numeric(frame["cue_side"], errors="coerce")
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["cue_side", *columns])
    frame["cue_side"] = frame["cue_side"].astype(int)
    return frame.sort_values(["record", "original_trial_index"]).reset_index(drop=True)


def _candidate_inner_score(
    outer_train: pd.DataFrame,
    candidate: Candidate,
) -> tuple[float, list[dict[str, Any]]]:
    scores = []
    details = []
    for train_index, valid_index, held_group in grouped_leave_one_group_out(
        outer_train["record"].to_numpy()
    ):
        train = outer_train.iloc[train_index]
        valid = outer_train.iloc[valid_index]
        metrics, _ = score_fold(train, valid, candidate)
        scores.append(metrics["balanced_accuracy"])
        details.append(
            {
                "inner_held_out_record": held_group,
                "n_inner_train": int(len(train)),
                "n_inner_validation": int(len(valid)),
                **metrics,
            }
        )
    return (float(np.mean(scores)) if scores else float("nan")), details


def run_experiment(input_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(input_path, encoding="utf-8-sig")
    candidates = candidate_grid()
    all_outer_metrics: list[dict[str, Any]] = []
    all_inner_metrics: list[dict[str, Any]] = []
    selected_metrics: list[dict[str, Any]] = []
    selected_predictions: list[pd.DataFrame] = []
    baseline_predictions: list[pd.DataFrame] = []
    baseline_name = "baseline_logistic_l2_1"

    for sample in ("raw_qc_pass", "q1_quality_matched"):
        for stage in ("cue_locked", "target_locked"):
            frame = prepare_scope(features, stage, sample)
            groups = frame["record"].to_numpy()
            outer_folds = grouped_leave_one_group_out(groups)
            baseline_candidate = next(item for item in candidates if item.name == baseline_name)
            for outer_train_index, outer_test_index, held_out_record in outer_folds:
                outer_train = frame.iloc[outer_train_index].reset_index(drop=True)
                outer_test = frame.iloc[outer_test_index].reset_index(drop=True)
                train_records = sorted(outer_train["record"].unique().tolist())
                if held_out_record in train_records:
                    raise AssertionError("Outer held-out record leaked into training data")

                baseline_fold, baseline_pred = score_fold(outer_train, outer_test, baseline_candidate)
                baseline_metrics_row = {
                    "sample": sample,
                    "stage": stage,
                    "candidate": baseline_name,
                    "held_out_record": held_out_record,
                    "n_train": int(len(outer_train)),
                    "n_test": int(len(outer_test)),
                    "training_records": ";".join(train_records),
                    **baseline_fold,
                }
                baseline_predictions.append(
                    baseline_pred.assign(
                        sample=sample,
                        stage=stage,
                        candidate=baseline_name,
                        held_out_record=held_out_record,
                    )
                )

                candidate_scores = []
                for candidate_index, candidate in enumerate(candidates):
                    inner_mean, inner_details = _candidate_inner_score(outer_train, candidate)
                    all_inner_metrics.extend(
                        {
                            "sample": sample,
                            "stage": stage,
                            "outer_held_out_record": held_out_record,
                            "candidate_order": candidate_index,
                            **candidate.as_row(),
                            "inner_mean_balanced_accuracy": inner_mean,
                            **detail,
                        }
                        for detail in inner_details
                    )
                    candidate_scores.append((inner_mean, candidate_index, candidate))

                    # Every row's test score is produced once from the other three records.
                    outer_metrics, _ = score_fold(outer_train, outer_test, candidate)
                    all_outer_metrics.append(
                        {
                            "sample": sample,
                            "stage": stage,
                            **candidate.as_row(),
                            "held_out_record": held_out_record,
                            "n_train": int(len(outer_train)),
                            "n_test": int(len(outer_test)),
                            "training_records": ";".join(train_records),
                            **outer_metrics,
                        }
                    )

                # Stable tie break favors the earlier, simpler predeclared model.
                best_inner, _, chosen = max(candidate_scores, key=lambda item: (item[0], -item[1]))
                chosen_metrics, chosen_pred = score_fold(outer_train, outer_test, chosen)
                chosen_row = {
                    "sample": sample,
                    "stage": stage,
                    "held_out_record": held_out_record,
                    "n_train": int(len(outer_train)),
                    "n_test": int(len(outer_test)),
                    "training_records": ";".join(train_records),
                    "selected_by_inner_balanced_accuracy": best_inner,
                    **chosen.as_row(),
                    **chosen_metrics,
                    "baseline_balanced_accuracy": baseline_fold["balanced_accuracy"],
                    "balanced_accuracy_delta": chosen_metrics["balanced_accuracy"] - baseline_fold["balanced_accuracy"],
                }
                selected_metrics.append(chosen_row)
                selected_predictions.append(
                    chosen_pred.assign(
                        sample=sample,
                        stage=stage,
                        candidate=chosen.name,
                        held_out_record=held_out_record,
                    )
                )

    outer_df = pd.DataFrame(all_outer_metrics)
    inner_df = pd.DataFrame(all_inner_metrics)
    selected_df = pd.DataFrame(selected_metrics)
    baseline_df = outer_df.loc[outer_df["candidate"] == baseline_name].copy()
    candidate_summary = (
        outer_df.groupby(["sample", "stage", "candidate", "model", "feature_set", "scaler", "parameter", "gamma"], dropna=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            mean_auc=("auc", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            valid_record_folds=("held_out_record", "nunique"),
        )
        .reset_index()
    )
    selected_summary_rows = []
    for (sample, stage), subset in selected_df.groupby(["sample", "stage"]):
        baseline_subset = baseline_df.loc[
            (baseline_df["sample"] == sample) & (baseline_df["stage"] == stage)
        ]
        baseline_ba = float(baseline_subset["balanced_accuracy"].mean())
        selected_ba = float(subset["balanced_accuracy"].mean())
        selected_summary_rows.append(
            {
                "sample": sample,
                "stage": stage,
                "n_trials": int(len(prepare_scope(features, stage, sample))),
                "valid_record_folds": int(subset["held_out_record"].nunique()),
                "baseline_mean_balanced_accuracy": baseline_ba,
                "nested_selected_mean_balanced_accuracy": selected_ba,
                "delta_balanced_accuracy": selected_ba - baseline_ba,
                "delta_percentage_points": 100.0 * (selected_ba - baseline_ba),
                "baseline_fold_balanced_accuracy": baseline_subset.set_index("held_out_record")["balanced_accuracy"].to_dict(),
                "selected_fold_balanced_accuracy": subset.set_index("held_out_record")["balanced_accuracy"].to_dict(),
                "selected_candidates_by_fold": subset.set_index("held_out_record")["name"].to_dict(),
                "delta_by_fold_percentage_points": (
                    subset.set_index("held_out_record")["balanced_accuracy"]
                    - baseline_subset.set_index("held_out_record")["balanced_accuracy"]
                ).mul(100.0).to_dict(),
            }
        )

    outer_df.to_csv(output_dir / "outer_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(output_dir / "outer_candidate_summary.csv", index=False, encoding="utf-8-sig")
    inner_df.to_csv(output_dir / "inner_cv_scores.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(output_dir / "nested_selected_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(baseline_predictions, ignore_index=True).to_csv(
        output_dir / "baseline_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(selected_predictions, ignore_index=True).to_csv(
        output_dir / "nested_selected_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "experiment": "exploratory nested leave-one-record-out model search",
        "input": str(input_path.resolve()),
        "primary_metric": "balanced_accuracy",
        "outer_validation": "leave one complete recording file out",
        "inner_selection": "leave one of the three outer-training records out; training-only scaling",
        "decision_thresholds": "fixed native thresholds; no threshold tuning",
        "candidate_count": len(candidates),
        "candidate_families": sorted({candidate.model for candidate in candidates}),
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "selections_and_deltas": selected_summary_rows,
        "limitations": [
            "Only four record-level outer folds are available; the point estimates are noisy.",
            "Candidate families were prespecified, but exploratory comparisons still motivate an independent replication.",
            "The nominal target anchor remains cue+2.2 s by schedule assumption.",
            "The snapshot baseline inputs and outputs are preserved under baselines/20260925_initial.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(output_dir, summary)
    return summary


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Exploratory Q3 model search (2026-09-25)",
        "",
        "This search is separate from the primary Q3 pipeline. The frozen reference is `src/C/q3/baselines/20260925_initial/`.",
        "",
        "## Protocol",
        "",
        "- Primary metric: balanced accuracy; all results also include accuracy, AUC, macro-F1, and fold-level values.",
        "- Outer split: leave one entire recording file out. Inner selection: leave one of the three outer-training records out.",
        "- Scaling is fit only on each training partition. Candidate choice is made only from inner-fold balanced accuracy; the outer record is scored once after selection.",
        "- The fixed baseline is the same 13-feature regularized Logistic model used in the original Q3 validation. Candidate families are Logistic with alternative regularization/feature sets, robust scaling, shrinkage LDA, linear SVM, and RBF SVM.",
        "- `AI_alpha` is a linear contrast of the three alpha log-power columns. The candidate set includes a version that drops this redundant column.",
        "- No threshold tuning or target-offset selection is performed.",
        "",
        "## Interpretation",
        "",
        "The nested-selected model is the main estimate of the model-selection procedure. The fixed-candidate table is exploratory; do not select a model based on its outer-fold maximum and claim that maximum as an unbiased score. Four outer records are too few to establish stable generalization.",
        "",
        "See `summary.json`, `nested_selected_fold_metrics.csv`, `outer_candidate_summary.csv`, and the out-of-fold prediction files.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_experiment(args.input, args.output_dir)
    for item in summary["selections_and_deltas"]:
        print(
            f"{item['sample']} {item['stage']}: "
            f"baseline BA={item['baseline_mean_balanced_accuracy']:.3f}, "
            f"nested selected BA={item['nested_selected_mean_balanced_accuracy']:.3f}, "
            f"delta={item['delta_percentage_points']:+.2f} pp"
        )
    print(f"Wrote exploratory results to {args.output_dir}")


if __name__ == "__main__":
    main()
