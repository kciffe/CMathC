"""Nested record-wise search over additional response-independent EEG features."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

import model_search as baseline_search


HERE = Path(__file__).resolve().parent
Q3_DIR = HERE.parent
BASELINE_FEATURES_PATH = Q3_DIR / "baselines" / "20260925_initial" / "output" / "trial_features.csv"
PROFILE_PATH = Q3_DIR / "output" / "experiments" / "20260925_expanded_search_profile.json"
DEFAULT_OUTPUT = Q3_DIR / "output" / "experiments" / "20260925_expanded_search_v1"
RANDOM_SEED = 20260925
CHANNELS = ("F3", "Fz", "F4")
ERP_MEAN = tuple(f"erp_mean_{channel}" for channel in CHANNELS)
ERP_PEAK = tuple(f"erp_peak_{channel}" for channel in CHANNELS)
ERP_LATENCY = tuple(f"erp_peak_latency_s_{channel}" for channel in CHANNELS)
BANDPOWER = tuple(
    f"log_{band}_power_{channel}"
    for band in ("theta", "alpha", "beta", "gamma")
    for channel in CHANNELS
)
BASELINE_13 = baseline_search.BASELINE_FEATURES
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "baseline_13": BASELINE_13,
    "gamma_extended_16": BASELINE_13 + tuple(f"log_gamma_power_{c}" for c in CHANNELS),
    "erp_shape_10": ERP_MEAN + ERP_PEAK + ERP_LATENCY + ("AI_erp_F4_minus_F3",),
    "static_no_gamma_19": ERP_MEAN + tuple(
        f"log_{band}_power_{channel}"
        for band in ("theta", "alpha", "beta") for channel in CHANNELS
    ) + ERP_PEAK + ERP_LATENCY + ("AI_erp_F4_minus_F3",),
    "static_safe_22": ERP_MEAN + BANDPOWER + ERP_PEAK + ERP_LATENCY + ("AI_erp_F4_minus_F3",),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    feature_set: str
    scaler: str = "standard"
    k: int | None = None
    parameters: tuple[tuple[str, Any], ...] = ()

    def params(self) -> dict[str, Any]:
        return dict(self.parameters)

    def as_row(self) -> dict[str, Any]:
        return {
            "candidate": self.name,
            "model": self.model,
            "feature_set": self.feature_set,
            "scaler": self.scaler,
            "selector_k": self.k,
            "parameters": json.dumps(self.params(), ensure_ascii=False, sort_keys=True),
        }


def fit_selector_train_only(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    k: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit univariate feature selection on training labels only, then transform eval."""
    x_train = np.asarray(x_train, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    if x_train.ndim != 2 or x_eval.ndim != 2 or x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("Train/evaluation arrays must be 2D with matching feature counts")
    if k is None or k >= x_train.shape[1]:
        return x_train.copy(), x_eval.copy(), np.arange(x_train.shape[1], dtype=int)
    if k < 1:
        raise ValueError("k must be positive or None")
    selector = SelectKBest(score_func=f_classif, k=min(k, x_train.shape[1]))
    selected_train = selector.fit_transform(x_train, y_train)
    selected_eval = selector.transform(x_eval)
    return selected_train, selected_eval, selector.get_support(indices=True)


def candidate_grid() -> list[Candidate]:
    """Predeclared feature/model grid; list order breaks exact inner-score ties."""
    result = [Candidate("baseline_logistic_l2_1", "logistic", "baseline_13")]
    for feature_set in FEATURE_SETS:
        for l2 in (0.1, 10.0, 100.0) if feature_set == "baseline_13" else (0.1, 1.0, 10.0, 100.0):
            if feature_set == "baseline_13" and l2 == 1.0:
                continue
            result.append(Candidate(f"logistic_{feature_set}_l2_{l2:g}", "logistic", feature_set,
                                    parameters=(("l2", l2),)))
    for feature_set in FEATURE_SETS:
        result.append(Candidate(f"shrinkage_lda_{feature_set}", "lda", feature_set))
    for feature_set in ("baseline_13", "erp_shape_10", "static_no_gamma_19"):
        for regularization in (0.1, 0.5, 0.9):
            result.append(Candidate(f"qda_{feature_set}_reg_{regularization:g}", "qda", feature_set,
                                    parameters=(("reg_param", regularization),)))
    for feature_set in ("baseline_13", "gamma_extended_16", "erp_shape_10", "static_no_gamma_19", "static_safe_22"):
        for c_value in (0.1, 1.0, 10.0):
            result.append(Candidate(f"linear_svm_{feature_set}_C_{c_value:g}", "linear_svm", feature_set,
                                    parameters=(("C", c_value),)))
    for feature_set in ("baseline_13", "gamma_extended_16", "erp_shape_10", "static_no_gamma_19", "static_safe_22"):
        for c_value in (0.3, 3.0, 30.0):
            for gamma in ("scale", 0.01, 0.03, 0.1):
                result.append(Candidate(
                    f"rbf_svm_{feature_set}_C_{c_value:g}_gamma_{gamma}", "rbf_svm", feature_set,
                    parameters=(("C", c_value), ("gamma", gamma)),
                ))
    for k in (3, 6, 10, 16):
        for c_value in (0.3, 3.0, 30.0):
            for gamma in (0.003, 0.03, 0.1):
                result.append(Candidate(
                    f"rbf_svm_static_safe_22_top{k}_C_{c_value:g}_gamma_{gamma}",
                    "rbf_svm", "static_safe_22", k=k,
                    parameters=(("C", c_value), ("gamma", gamma)),
                ))
    for feature_set in ("erp_shape_10", "static_safe_22"):
        for max_features in ("sqrt", 0.7):
            for min_leaf in (3, 8):
                for model in ("random_forest", "extra_trees"):
                    result.append(Candidate(
                        f"{model}_{feature_set}_maxfeat_{max_features}_leaf_{min_leaf}",
                        model, feature_set,
                        parameters=(("max_features", max_features), ("min_samples_leaf", min_leaf)),
                    ))
    for feature_set in ("erp_shape_10", "static_safe_22"):
        for leaves in (3, 7):
            for learning_rate in (0.03, 0.1):
                result.append(Candidate(
                    f"hist_gradient_{feature_set}_leaves_{leaves}_lr_{learning_rate:g}",
                    "hist_gradient", feature_set,
                    parameters=(("max_leaf_nodes", leaves), ("learning_rate", learning_rate),
                                ("max_iter", 100), ("l2_regularization", 1.0)),
                ))
    for feature_set in ("baseline_13", "erp_shape_10", "static_safe_22"):
        for neighbors in (5, 15, 31):
            result.append(Candidate(
                f"knn_{feature_set}_k_{neighbors}", "knn", feature_set,
                parameters=(("n_neighbors", neighbors), ("weights", "distance"), ("p", 2)),
            ))
    return result


def _scale(train: np.ndarray, evaluate: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_eval, _, _ = baseline_search.scale_train_test(train, evaluate, method)
    return x_train, x_eval


def _predict_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    candidate: Candidate,
) -> tuple[np.ndarray, float]:
    params = candidate.params()
    if candidate.model == "logistic":
        return baseline_search.fit_logistic_scores(x_train, y_train, x_eval, float(params.get("l2", 1.0))), 0.5
    if candidate.model == "lda":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        model.fit(x_train, y_train)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(x_eval)[:, positive], 0.5
    if candidate.model == "qda":
        model = QuadraticDiscriminantAnalysis(reg_param=float(params["reg_param"]))
        model.fit(x_train, y_train)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(x_eval)[:, positive], 0.5
    if candidate.model == "linear_svm":
        model = LinearSVC(C=float(params["C"]), dual="auto", random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        return model.decision_function(x_eval), 0.0
    if candidate.model == "rbf_svm":
        model = SVC(C=float(params["C"]), kernel="rbf", gamma=params["gamma"], random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        return model.decision_function(x_eval), 0.0
    if candidate.model in {"random_forest", "extra_trees"}:
        model_cls = RandomForestClassifier if candidate.model == "random_forest" else ExtraTreesClassifier
        model = model_cls(
            n_estimators=300,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(x_eval)[:, positive], 0.5
    if candidate.model == "hist_gradient":
        model = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **params)
        model.fit(x_train, y_train)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(x_eval)[:, positive], 0.5
    if candidate.model == "knn":
        model = KNeighborsClassifier(**params)
        model.fit(x_train, y_train)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(x_eval)[:, positive], 0.5
    raise ValueError(f"Unknown model: {candidate.model}")


def score_fold(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    candidate: Candidate,
) -> tuple[dict[str, float], pd.DataFrame]:
    columns = FEATURE_SETS[candidate.feature_set]
    train_raw = frame.iloc[train_idx].loc[:, columns].to_numpy(dtype=float)
    eval_raw = frame.iloc[eval_idx].loc[:, columns].to_numpy(dtype=float)
    y_train = frame.iloc[train_idx]["cue_side"].to_numpy(dtype=int)
    y_eval = frame.iloc[eval_idx]["cue_side"].to_numpy(dtype=int)
    train_raw, eval_raw, selected = fit_selector_train_only(train_raw, y_train, eval_raw, candidate.k)
    x_train, x_eval = _scale(train_raw, eval_raw, candidate.scaler)
    scores, threshold = _predict_scores(x_train, y_train, x_eval, candidate)
    predicted = np.where(scores >= threshold, 1, -1)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, predicted)),
        "auc": float(roc_auc_score((y_eval > 0).astype(int), scores)),
        "macro_f1": float(f1_score(y_eval, predicted, labels=[-1, 1], average="macro", zero_division=0)),
    }
    predictions = pd.DataFrame({
        "record": frame.iloc[eval_idx]["record"].astype(str).to_numpy(),
        "original_trial_index": frame.iloc[eval_idx]["original_trial_index"].to_numpy(dtype=int),
        "true_cue_side": y_eval,
        "predicted_cue_side": predicted,
        "score_right": scores,
        "decision_threshold": threshold,
    })
    return metrics, predictions


def prepare_scope(features: pd.DataFrame, stage: str, sample: str) -> pd.DataFrame:
    frame = baseline_search.prepare_scope(features, stage, sample)
    safe_columns = sorted({column for values in FEATURE_SETS.values() for column in values})
    frame[safe_columns] = frame[safe_columns].apply(pd.to_numeric, errors="coerce")
    frame[safe_columns] = frame[safe_columns].replace([np.inf, -np.inf], np.nan)
    before = len(frame)
    frame = frame.dropna(subset=safe_columns).copy()
    frame.attrs["rows_dropped_for_safe_feature_missingness"] = before - len(frame)
    return frame.sort_values(["record", "original_trial_index"]).reset_index(drop=True)


def _inner_score(frame: pd.DataFrame, outer_train: np.ndarray, candidate: Candidate):
    groups = frame.iloc[outer_train]["record"].to_numpy()
    details: list[dict[str, Any]] = []
    for inner_train, inner_valid, held_out in baseline_search.grouped_leave_one_group_out(groups):
        train_idx, valid_idx = outer_train[inner_train], outer_train[inner_valid]
        metrics, _ = score_fold(frame, train_idx, valid_idx, candidate)
        details.append({
            "inner_held_out_record": held_out,
            "n_inner_train": int(len(train_idx)),
            "n_inner_validation": int(len(valid_idx)),
            **metrics,
        })
    return float(np.mean([row["balanced_accuracy"] for row in details])), details


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(feature_path: Path = BASELINE_FEATURES_PATH, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(feature_path, encoding="utf-8-sig")
    candidates = candidate_grid()
    inner_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_predictions: list[pd.DataFrame] = []
    baseline_predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    baseline_candidate = candidates[0]

    for sample in ("raw_qc_pass", "q1_quality_matched"):
        for stage in ("cue_locked", "target_locked"):
            frame = prepare_scope(source, stage, sample)
            folds = baseline_search.grouped_leave_one_group_out(frame["record"].to_numpy())
            scope_selected: list[dict[str, Any]] = []
            scope_baseline: list[dict[str, Any]] = []
            for outer_train, outer_test, held_out in folds:
                training_records = sorted(frame.iloc[outer_train]["record"].unique().tolist())
                if held_out in training_records:
                    raise AssertionError("Outer held-out record leaked into training")
                base_metrics, base_pred = score_fold(frame, outer_train, outer_test, baseline_candidate)
                base_row = {
                    "sample": sample, "stage": stage, "held_out_record": held_out,
                    "n_train": int(len(outer_train)), "n_test": int(len(outer_test)), **base_metrics,
                }
                scope_baseline.append(base_row)
                baseline_predictions.append(base_pred.assign(
                    sample=sample, stage=stage, candidate=baseline_candidate.name,
                    held_out_record=held_out,
                ))

                scored: list[tuple[float, int, Candidate]] = []
                for order, candidate in enumerate(candidates):
                    inner_mean, details = _inner_score(frame, outer_train, candidate)
                    scored.append((inner_mean, order, candidate))
                    inner_rows.extend({
                        "sample": sample,
                        "stage": stage,
                        "outer_held_out_record": held_out,
                        "candidate_order": order,
                        **candidate.as_row(),
                        "inner_mean_balanced_accuracy": inner_mean,
                        **detail,
                    } for detail in details)
                best_inner, _, chosen = max(scored, key=lambda row: (row[0], -row[1]))
                chosen_metrics, chosen_pred = score_fold(frame, outer_train, outer_test, chosen)
                row = {
                    "sample": sample, "stage": stage, "held_out_record": held_out,
                    "n_train": int(len(outer_train)), "n_test": int(len(outer_test)),
                    "training_records": ";".join(training_records),
                    "selected_by_inner_balanced_accuracy": best_inner,
                    **chosen.as_row(), **chosen_metrics,
                    "baseline_balanced_accuracy": base_metrics["balanced_accuracy"],
                    "balanced_accuracy_delta": chosen_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"],
                }
                scope_selected.append(row)
                selected_rows.append(row)
                selected_predictions.append(chosen_pred.assign(
                    sample=sample, stage=stage, candidate=chosen.name, held_out_record=held_out,
                ))
            base_mean = float(np.mean([row["balanced_accuracy"] for row in scope_baseline]))
            selected_mean = float(np.mean([row["balanced_accuracy"] for row in scope_selected]))
            summaries.append({
                "sample": sample, "stage": stage, "n_trials": int(len(frame)),
                "rows_dropped_for_safe_feature_missingness": int(frame.attrs.get("rows_dropped_for_safe_feature_missingness", 0)),
                "valid_record_folds": len(scope_selected),
                "baseline_mean_balanced_accuracy": base_mean,
                "nested_selected_mean_balanced_accuracy": selected_mean,
                "delta_percentage_points": (selected_mean - base_mean) * 100,
                "baseline_fold_balanced_accuracy": {r["held_out_record"]: r["balanced_accuracy"] for r in scope_baseline},
                "selected_fold_balanced_accuracy": {r["held_out_record"]: r["balanced_accuracy"] for r in scope_selected},
                "selected_candidates_by_fold": {r["held_out_record"]: r["candidate"] for r in scope_selected},
                "delta_by_fold_percentage_points": {r["held_out_record"]: r["balanced_accuracy_delta"] * 100 for r in scope_selected},
            })
            print(f"Finished {sample} {stage}: baseline BA={base_mean:.3f}, selected BA={selected_mean:.3f}", flush=True)

    inner_df = pd.DataFrame(inner_rows)
    selected_df = pd.DataFrame(selected_rows)
    inner_df.to_csv(output_dir / "inner_candidate_scores.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(output_dir / "nested_selected_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(baseline_predictions, ignore_index=True).to_csv(output_dir / "baseline_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(selected_predictions, ignore_index=True).to_csv(output_dir / "nested_selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary: dict[str, Any] = {
        "experiment": "nested leave-one-record-out search over expanded response-independent static EEG features",
        "development_only": True,
        "input": str(feature_path.resolve()),
        "primary_metric": "balanced_accuracy",
        "outer_validation": "leave one complete recording out",
        "inner_selection": "leave one of the other three records out; feature selector and scaler fit on inner training rows only",
        "candidate_count": len(candidates),
        "candidate_families": sorted({item.model for item in candidates}),
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "excluded_columns": [
            "response_code, choice_side, cue/target timing, task code, QC flags",
            "pre_response_cumulative_* and pre_response_terminal_500ms_* because these windows end relative to the channel-9 action marker",
            "erp_candidate_mean_* duplicates erp_mean_*; AI_alpha duplicates AI_alpha_log_F4_minus_F3",
        ],
        "results": summaries,
        "limitations": [
            "The same four records were inspected in previous exploration rounds; this remains development-only and needs independent record-level replication.",
            "Candidate expansion itself creates selection optimism even though every candidate is evaluated in nested folds.",
            "Gamma power is exploratory; the feature set remains limited to frontal F3/Fz/F4 channels.",
            "Results measure cue-side prediction and do not establish correctness, omission rate, or clinical diagnosis.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_findings(output_dir, summary)
    inputs = [feature_path]
    if PROFILE_PATH.exists():
        inputs.append(PROFILE_PATH)
    artifacts = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "experiment": summary["experiment"],
        "created_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sources": {"expanded_search.py": _sha256(Path(__file__)), "model_search.py": _sha256(HERE / "model_search.py")},
        "inputs": [{"path": str(p.resolve()), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in inputs],
        "artifacts": [{"path": p.name, "sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifacts],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_findings(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Expanded feature and model search", "",
        "This is an exploratory, development-only nested leave-one-record-out search. The frozen baseline remains unchanged.", "",
        "| Sample | Stage | Baseline BA | Nested selected BA | Delta |", "|---|---|---:|---:|---:|",
    ]
    for item in summary["results"]:
        lines.append(
            f"| {item['sample']} | {item['stage']} | {item['baseline_mean_balanced_accuracy']:.4f} | "
            f"{item['nested_selected_mean_balanced_accuracy']:.4f} | {item['delta_percentage_points']:+.2f} pp |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "The first-round outer records have already been viewed in previous searches. Treat every score as development evidence; do not report the maximum outer-candidate score as an unbiased estimate.",
        "Feature selection and scaling are re-fitted inside each inner and outer training split. Response-conditioned pre-response windows and behavioral labels are excluded.",
        "", "## Limitations", "", *[f"- {limitation}" for limitation in summary["limitations"]],
    ])
    (output_dir / "experiment_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=BASELINE_FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_experiment(args.features, args.output_dir)
    for item in summary["results"]:
        print(
            f"{item['sample']} {item['stage']}: baseline BA="
            f"{item['baseline_mean_balanced_accuracy']:.3f}, selected BA="
            f"{item['nested_selected_mean_balanced_accuracy']:.3f}, delta={item['delta_percentage_points']:+.2f} pp"
        )
    print(f"Wrote expanded search results to {args.output_dir}")


if __name__ == "__main__":
    main()
