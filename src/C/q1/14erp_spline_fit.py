# -*- coding: utf-8 -*-
"""Whole-epoch smoothing-spline fits for condition-averaged ERP waveforms.

This stage reads the cleaned trials from stage 8 and follows the ERP
baseline-correction and averaging convention established in stage 11. It fits
one penalized cubic B-spline (P-spline) to each complete condition/channel ERP.

The spline describes the full ERP waveform; it does not identify or validate a
P300 component.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.io import loadmat
from scipy.optimize import minimize_scalar


DATASETS = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]

PROJECT_DIR = Path(__file__).resolve().parent
CLEAN_DIR = PROJECT_DIR / "output" / "8riemann_denoise"
RESULT_DIR = PROJECT_DIR / "output" / "14erp_spline_fit"

CHANNELS = [
    ("F3", 1),
    ("Fz", 0),
    ("F4", 2),
]

CONDITIONS = [
    ("左条件", -1),
    ("右条件", 1),
]

BASELINE_START, BASELINE_END = -0.2, 0.0
TARGET_ONSET = 2.20
CUE_CANDIDATE_WINDOW = (0.25, 0.50)
TARGET_CANDIDATE_WINDOW = (2.45, 2.70)
DEFAULT_BASIS_COUNT = 48
P300_CUE_COLOR = "#AFC6E9"
P300_TARGET_COLOR = "#C8B6E2"

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def baseline_correct(trials, time):
    """Apply the same per-trial pre-cue baseline correction as stage 11."""
    baseline = (time >= BASELINE_START) & (time < BASELINE_END)
    if not np.any(baseline):
        raise ValueError(
            f"No samples in the baseline interval "
            f"[{BASELINE_START}, {BASELINE_END}) s"
        )
    baseline_mean = trials[:, :, baseline].mean(axis=2, keepdims=True)
    return trials - baseline_mean


def validate_inputs(trials, relative_time, cue_type):
    if trials.ndim != 3 or trials.shape[1] < 3:
        raise ValueError("trial_data must have shape (trials, >=3 channels, time)")
    if relative_time.shape != (trials.shape[0], trials.shape[2]):
        raise ValueError("relative_time must match trial and time dimensions")
    if cue_type.size != trials.shape[0]:
        raise ValueError("cue_type length must equal the number of trials")
    if not np.allclose(relative_time, relative_time[0], rtol=0, atol=1e-9):
        raise ValueError("relative_time differs across trials")
    time = relative_time[0]
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("relative_time must be finite and strictly increasing")
    if time.size < 5:
        raise ValueError("At least five time samples are required for a cubic spline")
    return time


def make_bspline_basis(time, basis_count=DEFAULT_BASIS_COUNT):
    """Create an open cubic B-spline basis over the complete observed epoch."""
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 5 or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing vector of length >= 5")

    count = min(max(int(basis_count), 4), time.size - 1)
    interior_count = count - 4
    interior_knots = (
        np.linspace(time[0], time[-1], interior_count + 2)[1:-1]
        if interior_count > 0
        else np.empty(0, dtype=float)
    )
    knot_vector = np.r_[
        np.repeat(time[0], 4),
        interior_knots,
        np.repeat(time[-1], 4),
    ]
    basis = BSpline.design_matrix(
        time,
        knot_vector,
        k=3,
        extrapolate=False,
    ).toarray()
    return basis, count


def select_smoothing_lambda(basis, observed_erp):
    """Choose a P-spline penalty by GCV for one observed ERP mean curve."""
    basis = np.asarray(basis, dtype=float)
    y = np.asarray(observed_erp, dtype=float)
    n_samples, n_basis = basis.shape
    if y.shape != (n_samples,) or not np.all(np.isfinite(y)):
        raise ValueError("observed_erp must be finite and match the basis rows")
    if n_basis < 4 or n_basis >= n_samples:
        raise ValueError("The basis must have 4 or more columns and fewer than samples")

    second_difference = np.diff(np.eye(n_basis), n=2, axis=0)
    penalty = second_difference.T @ second_difference
    gram = basis.T @ basis
    cross = basis.T @ y
    scale = float(np.trace(gram) / np.trace(penalty))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Could not scale the P-spline smoothness penalty")

    def evaluate(log_lambda, return_fit=False):
        lam = float(np.exp(log_lambda))
        system = gram + lam * penalty
        coefficients = np.linalg.solve(system, cross)
        fitted = basis @ coefficients
        df = float(np.trace(np.linalg.solve(system, gram)))
        residual = y - fitted
        denominator = max(1.0 - df / n_samples, np.finfo(float).eps)
        gcv = float(np.mean(residual**2) / denominator**2)
        if return_fit:
            return lam, coefficients, fitted, df, gcv
        return gcv

    log_grid = np.log(scale) + np.linspace(-8.0, 8.0, 65)
    grid_scores = np.asarray([evaluate(value) for value in log_grid])
    best_index = int(np.nanargmin(grid_scores))
    lower_index = max(0, best_index - 1)
    upper_index = min(log_grid.size - 1, best_index + 1)

    if lower_index == upper_index:
        best_log_lambda = float(log_grid[best_index])
    else:
        result = minimize_scalar(
            evaluate,
            bounds=(float(log_grid[lower_index]), float(log_grid[upper_index])),
            method="bounded",
            options={"xatol": 1e-5},
        )
        best_log_lambda = float(result.x)

    lam, coefficients, fitted, df, gcv = evaluate(
        best_log_lambda,
        return_fit=True,
    )
    return {
        "Lambda": lam,
        "Coefficients": coefficients,
        "Fitted": fitted,
        "EffectiveDegreesFreedom": df,
        "GCV": gcv,
        "Penalty": penalty,
        "Gram": gram,
    }


def fit_pspline(basis, observed_erp):
    """Fit one condition-average ERP with a cubic P-spline selected by GCV."""
    observed_erp = np.asarray(observed_erp, dtype=float)
    if observed_erp.shape != (basis.shape[0],):
        raise ValueError("ERP curve must have the same length as the basis")

    selected = select_smoothing_lambda(basis, observed_erp)
    if not np.all(np.isfinite(observed_erp)):
        raise ValueError("ERP curve contains non-finite values")

    fitted_erp = selected["Fitted"]

    residual = observed_erp - fitted_erp
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_total = float(np.sum((observed_erp - observed_erp.mean()) ** 2))
    r2 = (
        np.nan
        if np.isclose(ss_total, 0.0)
        else float(1.0 - np.sum(residual**2) / ss_total)
    )

    return {
        "Fitted": fitted_erp,
        "RMSE": rmse,
        "R2": r2,
        "Lambda": selected["Lambda"],
        "EffectiveDegreesFreedom": selected["EffectiveDegreesFreedom"],
        "GCV": selected["GCV"],
        "Coefficients": selected["Coefficients"],
    }


def make_failed_result(time, reason):
    nan_curve = np.full(time.shape, np.nan, dtype=float)
    return {
        "Fitted": nan_curve.copy(),
        "RMSE": np.nan,
        "R2": np.nan,
        "SplineFitValid": 0,
        "FailureReason": reason,
    }


def process_dataset(
    dataset_name,
    result_dir=None,
):
    """Fit whole-epoch ERP splines and return summary and waveform records."""
    output_dir = Path(result_dir) if result_dir is not None else RESULT_DIR
    mat_path = CLEAN_DIR / f"{dataset_name}_clean.mat"
    mat = loadmat(mat_path)

    trials = np.asarray(mat["trial_data"], dtype=float)
    relative_time = np.asarray(mat["relative_time"], dtype=float)
    cue_type = np.asarray(mat["cue_type"]).reshape(-1)
    time = validate_inputs(trials, relative_time, cue_type)
    basis, basis_count = make_bspline_basis(time)

    eeg = baseline_correct(trials[:, :3, :], time)
    if not np.all(np.isfinite(eeg)):
        raise ValueError(f"{dataset_name}: baseline-corrected data contains NaN/Inf")

    summary_rows = []
    waveform_frames = []
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)

    for row_index, (condition_name, condition_value) in enumerate(CONDITIONS):
        condition_trials = eeg[cue_type == condition_value]
        n_trials = int(condition_trials.shape[0])

        for col_index, (channel_name, channel_index) in enumerate(CHANNELS):
            ax = axes[row_index, col_index]
            base_row = {
                "Dataset": dataset_name,
                "Condition": condition_name,
                "Channel": channel_name,
                "N_trials": n_trials,
                "SplineMethod": "Cubic P-spline; second-difference penalty; GCV",
                "GCVSelectionScope": "Observed condition-average ERP",
                "BasisCount": basis_count,
                "SmoothingLambda": np.nan,
                "EffectiveDegreesFreedom": np.nan,
                "GCV": np.nan,
                "TimeStart": float(time[0]),
                "TimeEnd": float(time[-1]),
                "SplineFitValid": 0,
                "R2": np.nan,
                "RMSE": np.nan,
                "FailureReason": "",
            }

            if n_trials == 0:
                erp = np.full(time.shape, np.nan)
                fit_result = make_failed_result(time, "no_trials_for_condition")
            else:
                channel_trials = condition_trials[:, channel_index, :]
                erp = channel_trials.mean(axis=0)

                try:
                    fit_result = fit_pspline(basis, erp)
                    fit_result["SplineFitValid"] = 1
                    fit_result["FailureReason"] = ""
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    fit_result = make_failed_result(time, str(exc))

            base_row["SplineFitValid"] = int(fit_result["SplineFitValid"])
            base_row["R2"] = fit_result["R2"]
            base_row["RMSE"] = fit_result["RMSE"]
            base_row["FailureReason"] = fit_result["FailureReason"]
            base_row["SmoothingLambda"] = fit_result.get("Lambda", np.nan)
            base_row["EffectiveDegreesFreedom"] = fit_result.get(
                "EffectiveDegreesFreedom",
                np.nan,
            )
            base_row["GCV"] = fit_result.get("GCV", np.nan)
            summary_rows.append(base_row)

            if n_trials:
                ax.plot(
                    time,
                    erp,
                    color="tab:blue",
                    alpha=0.95,
                    linewidth=3.0,
                    label="条件平均 ERP",
                )
            if fit_result["SplineFitValid"]:
                ax.plot(
                    time,
                    fit_result["Fitted"],
                    color="tab:orange",
                    linestyle="-",
                    linewidth=1.2,
                    zorder=4,
                    label="拟合曲线",
                )
            elif not n_trials:
                ax.text(
                    0.5,
                    0.5,
                    "该条件没有 Trial",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

            ax.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1)
            ax.axvline(TARGET_ONSET, color="tab:blue", linestyle=":", linewidth=1)
            ax.axvspan(
                *CUE_CANDIDATE_WINDOW,
                color=P300_CUE_COLOR,
                alpha=0.55,
                label="_nolegend_",
            )
            ax.axvspan(
                *TARGET_CANDIDATE_WINDOW,
                color=P300_TARGET_COLOR,
                alpha=0.55,
                label="_nolegend_",
            )
            ax.set_title(f"{condition_name} · {channel_name} (n={n_trials})")
            ax.set_xlabel("相对提示 onset 的时间 (s)")
            ax.set_ylabel("基线校正后 ERP 幅值（原始数据单位）")
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(alpha=0.2)

            waveform_frames.append(pd.DataFrame({
                "Dataset": dataset_name,
                "Condition": condition_name,
                "Channel": channel_name,
                "N_trials": n_trials,
                "TimeFromCue": time,
                "ObservedERP": erp,
                "SplineFit": fit_result["Fitted"],
                "SplineFitValid": int(fit_result["SplineFitValid"]),
            }))

    fig.suptitle(f"{dataset_name}：ERP 拟合曲线", fontsize=15, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", linewidth=3.0, label="条件平均 ERP"),
            Line2D(
                [0], [0], color="tab:orange", linewidth=1.2, linestyle="-", label="拟合曲线"
            ),
            Patch(
                facecolor=P300_CUE_COLOR,
                edgecolor="none",
                alpha=0.55,
                label="提示后 P300 候选时窗（250–500 ms）",
            ),
            Patch(
                facecolor=P300_TARGET_COLOR,
                edgecolor="none",
                alpha=0.55,
                label="目标后 P300 候选时窗（250–500 ms）",
            ),
            Line2D(
                [0], [0], color="tab:blue", linewidth=1, linestyle="--",
                label="提示开始（0 s）",
            ),
            Line2D(
                [0], [0], color="tab:blue", linewidth=1, linestyle=":",
                label="目标显示开始（2.20 s）",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        fontsize=9,
        frameon=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{dataset_name}_ERP样条拟合.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    waveform_table = (
        pd.concat(waveform_frames, ignore_index=True)
        if waveform_frames
        else pd.DataFrame()
    )
    return summary_rows, waveform_table


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary_rows = []
    all_waveforms = []

    for dataset_name in DATASETS:
        summary_rows, waveform_table = process_dataset(
            dataset_name,
        )
        all_summary_rows.extend(summary_rows)
        all_waveforms.append(waveform_table)

    pd.DataFrame(all_summary_rows).to_csv(
        RESULT_DIR / "ERP样条拟合摘要.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_waveforms, ignore_index=True).to_csv(
        RESULT_DIR / "ERP样条拟合波形.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Results written to: {RESULT_DIR}")
    print(f"Curves: {len(all_summary_rows)}")


if __name__ == "__main__":
    main()
