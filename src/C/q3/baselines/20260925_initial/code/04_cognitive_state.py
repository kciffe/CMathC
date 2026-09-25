"""Construct anchored trial-level V/H/P proxies and estimate descriptive paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from common import ensure_output_dir, write_csv, write_json
from config import EEG_CHANNELS


def _fit_ols(y: np.ndarray, predictors: np.ndarray) -> tuple[np.ndarray, int, float]:
    design = np.column_stack([np.ones(len(y)), predictors])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    rmse = float(np.sqrt(np.mean(residual**2)))
    return coefficients, int(rank), rmse


def fit_path_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit the proposed within-trial paths on fixed, observed EEG proxies.

    This is a path regression on anchored functional proxies, not an estimate
    of latent sources or a state-transition model.
    """
    rows: list[dict] = []
    equation_definitions = {
        "V_visual_forward_proxy": (["cue_side"], ["intercept", "cue_side"]),
        "H_memory_proxy": (
            ["V_visual_like_raw", "task_filename_code_centered"],
            ["intercept", "V_visual_like_raw", "task_filename_code_centered"],
        ),
        "P_control_proxy": (
            ["H_memory_like_raw", "V_visual_like_raw", "task_filename_code_centered"],
            ["intercept", "H_memory_like_raw", "V_visual_like_raw", "task_filename_code_centered"],
        ),
    }
    summaries: dict = {}
    for stage in ("cue_locked", "target_locked"):
        subset = frame.loc[
            (frame["stage"] == stage)
            & frame["qc_valid"].fillna(False)
            & frame["V_visual_like_raw"].notna()
            & frame["H_memory_like_raw"].notna()
            & frame["P_control_like_raw"].notna()
            & frame["task_filename_code_centered"].notna()
        ].copy()
        stage_summary = {"n_trials": int(len(subset)), "equations": {}}
        if not len(subset):
            summaries[stage] = stage_summary
            continue
        for target, (predictor_columns, parameter_names) in equation_definitions.items():
            outcome_column = {
                "V_visual_forward_proxy": "V_visual_like_raw",
                "H_memory_proxy": "H_memory_like_raw",
                "P_control_proxy": "P_control_like_raw",
            }[target]
            if target == "V_visual_forward_proxy":
                y = subset["V_visual_like_raw"].to_numpy(dtype=float)
                x = subset["cue_side"].to_numpy(dtype=float).reshape(-1, 1)
            elif target == "H_memory_proxy":
                y = subset["H_memory_like_raw"].to_numpy(dtype=float)
                x = subset[predictor_columns].to_numpy(dtype=float)
            else:
                y = subset["P_control_like_raw"].to_numpy(dtype=float)
                x = subset[predictor_columns].to_numpy(dtype=float)
            if len(y) < x.shape[1] + 5:
                stage_summary["equations"][target] = {
                    "status": "insufficient_rows",
                    "n": int(len(y)),
                }
                continue
            coefficients, rank, rmse = _fit_ols(y, x)
            equation_rows = []
            for name, coefficient in zip(parameter_names, coefficients):
                rows.append(
                    {
                        "stage": stage,
                        "equation": target,
                        "parameter": name,
                        "estimate": float(coefficient),
                        "n_trials": int(len(y)),
                        "design_rank": rank,
                        "rmse_in_sample": rmse,
                        "inference": "descriptive OLS; four recording files, participant/task mapping not verified",
                    }
                )
                equation_rows.append({"parameter": name, "estimate": float(coefficient)})
            stage_summary["equations"][target] = {
                "status": "fit_descriptive_only",
                "n": int(len(y)),
                "rank": rank,
                "rmse_in_sample": rmse,
                "coefficients": equation_rows,
                "outcome_column": outcome_column,
            }
        summaries[stage] = stage_summary
    return pd.DataFrame(rows), summaries


def main() -> None:
    output_dir = ensure_output_dir()
    feature_path = output_dir / "trial_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError("Run 03_extract_features.py first to create trial_features.csv")
    features = pd.read_csv(feature_path, encoding="utf-8-sig")

    erp_columns = [f"erp_mean_{channel}" for channel in EEG_CHANNELS]
    theta_columns = [f"log_theta_power_{channel}" for channel in EEG_CHANNELS]
    beta_columns = [f"log_beta_power_{channel}" for channel in EEG_CHANNELS]
    gamma_columns = [f"log_gamma_power_{channel}" for channel in EEG_CHANNELS]
    features["V_visual_like_raw"] = features[erp_columns].mean(axis=1)
    features["H_memory_like_raw"] = features[theta_columns].mean(axis=1)
    features["P_control_like_raw"] = features[beta_columns].mean(axis=1)
    features["Gamma_visual_like_raw"] = features[gamma_columns].mean(axis=1)
    features["task_filename_code_centered"] = (
        pd.to_numeric(features["filename_task_code_candidate"], errors="coerce") - 1.5
    )

    state_sources = {
        "V_visual_like": ("V_visual_like_raw", "mean frontal baseline-corrected ERP candidate amplitude"),
        "H_memory_like": ("H_memory_like_raw", "mean frontal log theta power"),
        "P_control_like": ("P_control_like_raw", "mean frontal log beta power"),
    }
    scale_rows: list[dict] = []
    for stage, row_indices in features.groupby("stage").groups.items():
        for score_name, (source_name, definition) in state_sources.items():
            values = pd.to_numeric(features.loc[row_indices, source_name], errors="coerce")
            center = float(values.mean()) if values.notna().any() else np.nan
            scale = float(values.std(ddof=0)) if values.notna().any() else np.nan
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                scale = 1.0
            score_column = f"{score_name}_score_z"
            features.loc[row_indices, score_column] = (values - center) / scale
            scale_rows.append(
                {
                    "stage": stage,
                    "state_proxy": score_name,
                    "source_feature": source_name,
                    "definition": definition,
                    "pooled_center": center,
                    "pooled_scale_sd": scale,
                    "scope": "descriptive plot only; held-out predictive models re-standardize within training folds",
                }
            )

    # This stimulus-to-state equation is used only for path description. The
    # cue label is never included in an EEG-only cue classifier feature vector.
    path_coefficients, path_summary = fit_path_models(features)
    write_csv(features, output_dir / "state_scores.csv")
    write_csv(pd.DataFrame(scale_rows), output_dir / "state_proxy_definitions.csv")
    write_csv(path_coefficients, output_dir / "cognitive_path_coefficients.csv")
    condition_frame = features.loc[
        features["stage"].isin(("cue_locked", "target_locked"))
        & features["qc_valid"].fillna(False)
    ]
    condition_rows: list[dict] = []
    for (stage, task_code, cue_side), group in condition_frame.groupby(
        ["stage", "filename_task_code_candidate", "cue_side"], dropna=False
    ):
        for state_name, source_name in (
            ("V_visual_like", "V_visual_like_raw"),
            ("H_memory_like", "H_memory_like_raw"),
            ("P_control_like", "P_control_like_raw"),
        ):
            values = pd.to_numeric(group[source_name], errors="coerce").dropna()
            condition_rows.append(
                {
                    "stage": stage,
                    "filename_task_code_candidate": task_code,
                    "cue_side": int(cue_side),
                    "state_proxy": state_name,
                    "n_trials": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "inference": "descriptive condition means; task suffix is not yet verified as formal project type",
                }
            )
    write_csv(pd.DataFrame(condition_rows), output_dir / "state_condition_effects.csv")

    model_description = {
        "observation_unit": "one raw EEG trial at a fixed cue/target event-locked stage",
        "proxy_vector": ["V_visual_like", "H_memory_like", "P_control_like"],
        "proxy_definitions": {
            "V_visual_like": "mean frontal ERP candidate amplitude (250-500 ms)",
            "H_memory_like": "mean frontal log theta power",
            "P_control_like": "mean frontal log beta power",
        },
        "additional_feature_families": ["gamma power proxy (exploratory; evaluated in ablation sensitivity)", "ERP frontal asymmetry", "alpha asymmetry"],
        "within_trial_path": {
            "equations": [
                "V = aV + bV*u + eV",
                "H = aH + bHV*V + bHT*T_candidate + eH",
                "P = aP + bPH*H + bPV*V + bPT*T_candidate + eP",
            ],
            "estimates": path_summary,
        },
        "task_type": "formal Project-1/Project-2 mapping remains unverified; exploratory T_candidate uses only the filename Task-1/Task-2 suffix",
        "condition_summary": "state_condition_effects.csv reports descriptive cue-side means by filename task-code candidate; no task or participant significance claim is made",
        "latent_factor_model_fitted": False,
        "reason": "Three electrodes and a small number of recording groups do not identify an unconstrained sparse C matrix. Fixed observed proxies are used as a transparent fallback; coefficients are not interpreted as source activity.",
        "state_transition_model_fitted": False,
        "state_scores_for_prediction": "raw proxy values are used; descriptive pooled z scores are not used by held-out predictive models",
        "cue_label_leakage_guard": "cue_side is a predictor only in the descriptive V path equation; it is absent from EEG feature-only cue classifiers",
        "path_inference": "coefficients are descriptive, without trial-independent p-values; four records do not verify independent participants",
        "interpretation_limit": "V/H/P are macro cognitive functional proxies, not localized LGN, hippocampal or PFC sources",
    }
    write_json(model_description, output_dir / "cognitive_state_model.json")

    target = features.loc[
        (features["stage"] == "target_locked") & features["qc_valid"].fillna(False)
    ]
    score_columns = [f"{name}_score_z" for name in state_sources]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, column in zip(axes, score_columns):
        values = [
            target.loc[target["cue_side"] == side, column].dropna().to_numpy()
            for side in (-1, 1)
        ]
        if all(len(group) for group in values):
            ax.boxplot(values, tick_labels=["Left cue", "Right cue"], showfliers=False)
        else:
            ax.text(0.5, 0.5, "Insufficient QC-passed trials", ha="center", va="center")
        ax.set_title(column.replace("_score_z", ""))
        ax.set_ylabel("Descriptive z score")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Raw EEG trial-level anchored V/H/P functional proxies")
    fig.savefig(output_dir / "state_proxy_scores.png", dpi=180)
    plt.close(fig)

    print(
        f"Wrote {len(features)} state rows and {len(path_coefficients)} descriptive path coefficients; "
        "unidentified latent source model was not fitted."
    )


if __name__ == "__main__":
    main()
