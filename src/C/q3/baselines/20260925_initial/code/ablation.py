"""Grouped validation and ablation helpers for the anchored V/H/P model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit

from common import ensure_output_dir, write_csv, write_json
from config import RANDOM_SEED


CLASSIFICATION_MODELS = {
    "Model-0_V": ["V_visual_like_raw"],
    "Model-0_VplusGamma": ["V_visual_like_raw", "Gamma_visual_like_raw"],
    "Model-A1_no_H": ["V_visual_like_raw", "P_control_like_raw"],
    "Model-A2_no_P": ["V_visual_like_raw", "H_memory_like_raw"],
    "Model-A3_no_T": ["V_visual_like_raw", "H_memory_like_raw", "P_control_like_raw"],
    "Model-F_VHP_plus_Tcandidate": [
        "V_visual_like_raw",
        "H_memory_like_raw",
        "P_control_like_raw",
        "filename_task_code_candidate",
    ],
    "Model-F_VHP_gamma_plus_Tcandidate": [
        "V_visual_like_raw",
        "H_memory_like_raw",
        "P_control_like_raw",
        "Gamma_visual_like_raw",
        "filename_task_code_candidate",
    ],
}

RECONSTRUCTION_MODELS = {
    "Model-0_V": ["V"],
    "Model-0_VplusGamma": ["V", "G"],
    "Model-A1_no_H": ["V", "P"],
    "Model-A2_no_P": ["V", "H"],
    "Model-A3_no_T": ["V", "H", "P"],
    "Model-F_VHP_plus_Tcandidate": ["V", "H", "P", "T"],
    "Model-F_VHP_gamma_plus_Tcandidate": ["V", "H", "P", "G", "T"],
}

CORE_FEATURES = [
    *[f"erp_mean_{channel}" for channel in ("F3", "Fz", "F4")],
    *[
        f"log_{band}_power_{channel}"
        for band in ("theta", "alpha", "beta")
        for channel in ("F3", "Fz", "F4")
    ],
    "AI_alpha",
]

PROXY_ANCHORS = {
    "V": [
        *[f"erp_mean_{channel}" for channel in ("F3", "Fz", "F4")],
    ],
    "G": [
        *[f"log_gamma_power_{channel}" for channel in ("F3", "Fz", "F4")],
    ],
    "H": [f"log_theta_power_{channel}" for channel in ("F3", "Fz", "F4")],
    "P": [
        *[f"log_beta_power_{channel}" for channel in ("F3", "Fz", "F4")],
        "AI_alpha",
    ],
}


def make_proxy_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    excluded_feature: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fold-local standardized proxies; optionally leave one indicator out."""
    train_proxy = pd.DataFrame(index=train.index)
    test_proxy = pd.DataFrame(index=test.index)
    for state, anchors in PROXY_ANCHORS.items():
        selected = [name for name in anchors if name != excluded_feature]
        if excluded_feature and excluded_feature.startswith("log_alpha_power_"):
            selected = [name for name in selected if name != "AI_alpha"]
        selected = [name for name in selected if name in train.columns and name in test.columns]
        if not selected:
            train_proxy[state] = np.nan
            test_proxy[state] = np.nan
            continue
        train_values = train[selected].apply(pd.to_numeric, errors="coerce")
        test_values = test[selected].apply(pd.to_numeric, errors="coerce")
        center = train_values.mean(axis=0)
        scale = train_values.std(axis=0, ddof=0)
        scale = scale.where(np.isfinite(scale) & (scale > np.finfo(float).eps), 1.0)
        train_proxy[state] = ((train_values - center) / scale).mean(axis=1)
        test_proxy[state] = ((test_values - center) / scale).mean(axis=1)
    return train_proxy, test_proxy


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
    if not np.isfinite(result.fun):
        raise RuntimeError(f"Ablation logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = []
    for label in (0, 1):
        mask = actual == label
        if mask.any():
            recalls.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def auc_score(actual: np.ndarray, scores: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = actual == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if not n_positive or not n_negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def _aic_bic(log_likelihood: float, parameter_count: int, n: int) -> tuple[float, float]:
    return (
        float(2 * parameter_count - 2 * log_likelihood),
        float(parameter_count * np.log(max(n, 1)) - 2 * log_likelihood),
    )


def classification_ablation(frame: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    stages = ("cue_locked", "target_locked")
    for stage in stages:
        stage_frame = frame.loc[
            (frame["stage"] == stage) & frame["qc_valid"].fillna(False).astype(bool)
        ].copy()
        for held_out_record in sorted(stage_frame["record"].unique()):
            train = stage_frame.loc[stage_frame["record"] != held_out_record]
            test = stage_frame.loc[stage_frame["record"] == held_out_record]
            if train["cue_side"].nunique() < 2 or test["cue_side"].nunique() < 2:
                continue
            for model_name, predictors in CLASSIFICATION_MODELS.items():
                train_data = train.dropna(subset=[*predictors, "cue_side"])
                test_data = test.dropna(subset=[*predictors, "cue_side"])
                if len(train_data) < len(predictors) + 5 or test_data["cue_side"].nunique() < 2:
                    continue
                x_train_raw = train_data[predictors].to_numpy(dtype=float)
                x_test_raw = test_data[predictors].to_numpy(dtype=float)
                center = np.mean(x_train_raw, axis=0)
                scale = np.std(x_train_raw, axis=0)
                scale[~np.isfinite(scale) | (scale <= np.finfo(float).eps)] = 1.0
                x_train = (x_train_raw - center) / scale
                x_test = (x_test_raw - center) / scale
                y_train = (train_data["cue_side"].to_numpy(dtype=int) > 0).astype(float)
                y_test = (test_data["cue_side"].to_numpy(dtype=int) > 0).astype(int)
                coefficients = fit_logistic(x_train, y_train)
                train_probability = np.clip(
                    expit(np.column_stack([np.ones(len(x_train)), x_train]) @ coefficients),
                    1e-12,
                    1 - 1e-12,
                )
                test_probability = expit(
                    np.column_stack([np.ones(len(x_test)), x_test]) @ coefficients
                )
                test_predicted = (test_probability >= 0.5).astype(int)
                log_likelihood = float(
                    np.sum(y_train * np.log(train_probability) + (1 - y_train) * np.log(1 - train_probability))
                )
                aic, bic = _aic_bic(log_likelihood, len(predictors) + 1, len(y_train))
                rows.append(
                    {
                        "sample": sample_name,
                        "stage": stage,
                        "model": model_name,
                        "held_out_record": held_out_record,
                        "n_train": int(len(train_data)),
                        "n_test": int(len(test_data)),
                        "accuracy": float(np.mean(y_test == test_predicted)),
                        "balanced_accuracy": balanced_accuracy(y_test, test_predicted),
                        "auc": auc_score(y_test, test_probability),
                        "training_aic_approx": aic,
                        "training_bic_approx": bic,
                        "predictors": ";".join(predictors),
                    }
                )
    return pd.DataFrame(rows)


def reconstruction_ablation(frame: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for stage in ("cue_locked", "target_locked"):
        stage_frame = frame.loc[
            (frame["stage"] == stage) & frame["qc_valid"].fillna(False).astype(bool)
        ].copy()
        for held_out_record in sorted(stage_frame["record"].unique()):
            train = stage_frame.loc[stage_frame["record"] != held_out_record].copy()
            test = stage_frame.loc[stage_frame["record"] == held_out_record].copy()
            for target in CORE_FEATURES:
                required = [target]
                train_target = train.dropna(subset=required)
                test_target = test.dropna(subset=required)
                if len(train_target) < 20 or not len(test_target):
                    continue
                proxy_train, proxy_test = make_proxy_features(train_target, test_target, target)
                y_center = float(train_target[target].mean())
                y_scale = float(train_target[target].std(ddof=0))
                if not np.isfinite(y_scale) or y_scale <= np.finfo(float).eps:
                    y_scale = 1.0
                y_train = (train_target[target].to_numpy(dtype=float) - y_center) / y_scale
                y_test = (test_target[target].to_numpy(dtype=float) - y_center) / y_scale
                train_task = pd.to_numeric(train_target["filename_task_code_candidate"], errors="coerce").to_numpy(dtype=float)
                test_task = pd.to_numeric(test_target["filename_task_code_candidate"], errors="coerce").to_numpy(dtype=float)
                for model_name, state_names in RECONSTRUCTION_MODELS.items():
                    train_parts = [proxy_train[name].to_numpy(dtype=float) for name in state_names if name in proxy_train]
                    test_parts = [proxy_test[name].to_numpy(dtype=float) for name in state_names if name in proxy_test]
                    if "T" in state_names:
                        train_parts.append(train_task)
                        test_parts.append(test_task)
                    if not train_parts:
                        continue
                    x_train_raw = np.column_stack(train_parts)
                    x_test_raw = np.column_stack(test_parts)
                    finite_train = np.isfinite(y_train) & np.isfinite(x_train_raw).all(axis=1)
                    finite_test = np.isfinite(y_test) & np.isfinite(x_test_raw).all(axis=1)
                    if finite_train.sum() < x_train_raw.shape[1] + 5 or not finite_test.any():
                        continue
                    x_train_clean = x_train_raw[finite_train]
                    x_test_clean = x_test_raw[finite_test]
                    target_train = y_train[finite_train]
                    target_test = y_test[finite_test]
                    design_train = np.column_stack([np.ones(len(x_train_clean)), x_train_clean])
                    design_test = np.column_stack([np.ones(len(x_test_clean)), x_test_clean])
                    coefficients, _, rank, _ = np.linalg.lstsq(design_train, target_train, rcond=None)
                    residual_train = target_train - design_train @ coefficients
                    residual_test = target_test - design_test @ coefficients
                    rss = float(np.dot(residual_train, residual_train))
                    variance = max(rss / len(residual_train), 1e-12)
                    log_likelihood = float(
                        -0.5 * len(residual_train) * (np.log(2 * np.pi * variance) + 1.0)
                    )
                    parameter_count = int(rank + 1)  # model coefficients + residual variance
                    aic, bic = _aic_bic(log_likelihood, parameter_count, len(residual_train))
                    rows.append(
                        {
                            "sample": sample_name,
                            "stage": stage,
                            "held_out_record": held_out_record,
                            "target_feature": target,
                            "model": model_name,
                            "n_train": int(len(target_train)),
                            "n_test": int(len(target_test)),
                            "heldout_rmse_standardized": float(np.sqrt(np.mean(residual_test**2))),
                            "training_aic": aic,
                            "training_bic": bic,
                            "predictors": ";".join(state_names),
                        }
                    )
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> list[dict]:
    if frame.empty:
        return []
    return (
        frame.groupby(group_columns, dropna=False)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )


def main() -> None:
    output_dir = ensure_output_dir()
    path = output_dir / "state_scores.csv"
    if not path.exists():
        raise FileNotFoundError("Run 04_cognitive_state.py before 07_ablation.py")
    state = pd.read_csv(path, encoding="utf-8-sig")
    analysis = state.loc[state["stage"].isin(("cue_locked", "target_locked"))].copy()
    analysis["qc_valid"] = analysis["qc_valid"].fillna(False).astype(bool)
    primary = analysis.loc[analysis["qc_valid"]].copy()
    matched = primary.loc[primary["q1_quality_pass"].fillna(False).astype(bool)].copy()

    classification = pd.concat(
        [
            classification_ablation(primary, "raw_qc_pass"),
            classification_ablation(matched, "q1_quality_matched"),
        ],
        ignore_index=True,
    )
    reconstruction = pd.concat(
        [
            reconstruction_ablation(primary, "raw_qc_pass"),
            reconstruction_ablation(matched, "q1_quality_matched"),
        ],
        ignore_index=True,
    )
    write_csv(classification, output_dir / "ablation_cv_metrics.csv")
    write_csv(reconstruction, output_dir / "ablation_reconstruction_metrics.csv")

    summary = {
        "validation": "leave-one-record-out; training-only standardization; four recording files are the holdout groups",
        "samples": {
            "raw_qc_pass": int(len(primary)),
            "q1_quality_matched_and_raw_qc_pass": int(len(matched)),
        },
        "classification_models": CLASSIFICATION_MODELS,
        "classification_results": _aggregate(
            classification,
            ["sample", "stage", "model"],
            ["accuracy", "balanced_accuracy", "auc", "training_aic_approx", "training_bic_approx"],
        ),
        "feature_reconstruction_results": _aggregate(
            reconstruction,
            ["sample", "stage", "model"],
            ["heldout_rmse_standardized", "training_aic", "training_bic"],
        ),
        "aic_bic_limit": "AIC/BIC are auxiliary within-fold likelihood summaries; task-code candidate is filename-derived, the trial units may be clustered, and no significance claim is made.",
        "task_code_limit": "T_candidate is the numeric Task-1/Task-2 filename suffix only; formal project mapping is not confirmed.",
        "identifiability_limit": "V/H/P are fixed observed feature proxies; no free latent loading matrix C or source localization is fitted.",
        "interpretation_limit": "The reconstruction procedure excludes each target EEG feature from the corresponding proxy anchor before estimating that held-out target; it assesses cross-record association, not latent-source recovery.",
    }
    write_json(summary, output_dir / "ablation_summary.json")

    classification_summary = pd.DataFrame(summary["classification_results"])
    reconstruction_summary = pd.DataFrame(summary["feature_reconstruction_results"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if not classification_summary.empty:
        subset = classification_summary.loc[classification_summary["sample"] == "raw_qc_pass"]
        models = list(CLASSIFICATION_MODELS)
        x = np.arange(len(models))
        width = 0.35
        for index, stage in enumerate(("cue_locked", "target_locked")):
            values = subset.loc[subset["stage"] == stage].set_index("model")["balanced_accuracy"]
            axes[0].bar(
                x + (index - 0.5) * width,
                [values.get(model, np.nan) for model in models],
                width,
                label=stage.replace("_", " "),
            )
        axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axes[0].set_xticks(x, models, rotation=25, ha="right")
        axes[0].set_ylabel("Balanced accuracy")
        axes[0].set_title("Cue-side classification, raw QC sample")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No valid classifier folds", ha="center", va="center")
    if not reconstruction_summary.empty:
        subset = reconstruction_summary.loc[reconstruction_summary["sample"] == "raw_qc_pass"]
        models = list(RECONSTRUCTION_MODELS)
        x = np.arange(len(models))
        width = 0.35
        for index, stage in enumerate(("cue_locked", "target_locked")):
            values = subset.loc[subset["stage"] == stage].set_index("model")["heldout_rmse_standardized"]
            axes[1].bar(
                x + (index - 0.5) * width,
                [values.get(model, np.nan) for model in models],
                width,
                label=stage.replace("_", " "),
            )
        axes[1].set_xticks(x, models, rotation=25, ha="right")
        axes[1].set_ylabel("Held-out standardized RMSE (lower is better)")
        axes[1].set_title("Leave-one-feature-out proxy reconstruction")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No valid reconstruction folds", ha="center", va="center")
    fig.savefig(output_dir / "ablation_effects.png", dpi=180)
    plt.close(fig)

    lines = [
        "# V/H/P ablation results",
        "",
        "Evaluation uses leave-one-record-out folds, with all scaling estimated from the three training records. The Q1-retained subset is reported as a matched-sample sensitivity check.",
        "",
        "## Cue-side prediction",
        "",
        "Balanced accuracy and AUC are in `ablation_cv_metrics.csv`. Models use channel-8 cue direction as the target; cue direction is not included in the EEG proxy predictors. The V-only versus V-plus-gamma models provide the gamma inclusion sensitivity check; PAC remains unavailable.",
        "",
        "## Feature reconstruction",
        "",
        "`ablation_reconstruction_metrics.csv` reports held-out standardized RMSE by target feature. Each target is removed from its proxy anchor before the model is fitted. AIC/BIC are auxiliary likelihood summaries, not significance tests.",
        "",
        "## Interpretation limits",
        "",
        "V/H/P are observed feature composites used because three electrodes do not identify a free sparse measurement matrix. Task dependence is exploratory and uses the Task-1/Task-2 filename suffix, whose formal project meaning remains unverified. Four recording files do not establish participant-level generalization.",
        "",
    ]
    (output_dir / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")

    validation_path = output_dir / "validation_summary.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["VHP_ablation"] = {
            "classification": summary["classification_results"],
            "feature_reconstruction": summary["feature_reconstruction_results"],
            "limitations": summary["interpretation_limit"],
        }
        write_json(validation, validation_path)
        raw_class = pd.DataFrame(summary["classification_results"])
        raw_recon = pd.DataFrame(summary["feature_reconstruction_results"])
        report_path = output_dir / "validation_interpretation.md"
        if report_path.exists():
            report_lines = [
                "",
                "## V/H/P ablation check",
                "",
                "The following are grouped cross-validation summaries, not significance tests:",
                "",
            ]
            for stage in ("cue_locked", "target_locked"):
                cls = raw_class.loc[
                    (raw_class["sample"] == "raw_qc_pass")
                    & (raw_class["stage"] == stage)
                ].set_index("model")
                rec = raw_recon.loc[
                    (raw_recon["sample"] == "raw_qc_pass")
                    & (raw_recon["stage"] == stage)
                ].set_index("model")
                if "Model-0_V" in cls.index and "Model-F_VHP_plus_Tcandidate" in cls.index:
                    base_ba = float(cls.loc["Model-0_V", "balanced_accuracy"])
                    full_ba = float(cls.loc["Model-F_VHP_plus_Tcandidate", "balanced_accuracy"])
                    report_lines.append(
                        f"- {stage}: V-only balanced accuracy {base_ba:.3f}; V/H/P plus filename task-code candidate {full_ba:.3f} (change {full_ba - base_ba:+.3f})."
                    )
                if "Model-0_V" in rec.index and "Model-F_VHP_plus_Tcandidate" in rec.index:
                    base_rmse = float(rec.loc["Model-0_V", "heldout_rmse_standardized"])
                    full_rmse = float(rec.loc["Model-F_VHP_plus_Tcandidate", "heldout_rmse_standardized"])
                    report_lines.append(
                        f"- {stage}: leave-one-feature-out standardized reconstruction RMSE was {base_rmse:.3f} for V-only and {full_rmse:.3f} for the full candidate (change {full_rmse - base_rmse:+.3f})."
                    )
            report_lines.extend(
                [
                    "",
                    "The incremental classification effect is small and differs by event stage. The reconstruction result measures cross-record feature association under fixed proxies; it does not identify hidden brain sources. See `ablation_report.md` and `ablation_cv_metrics.csv` for all models.",
                    "",
                ]
            )
            with report_path.open("a", encoding="utf-8") as stream:
                stream.write("\n".join(report_lines))
    print(
        f"Wrote {len(classification)} grouped classification rows and "
        f"{len(reconstruction)} grouped feature-reconstruction rows."
    )


if __name__ == "__main__":
    main()
