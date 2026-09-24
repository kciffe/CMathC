# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import curve_fit


DATASETS = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]

PROJECT_DIR = Path(__file__).resolve().parent
CLEAN_DIR = PROJECT_DIR / "output" / "8riemann_denoise"
ERP_DIR = PROJECT_DIR / "output" / "11erp_extract"
RESULT_DIR = PROJECT_DIR / "output" / "13erp_fit"

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
MIN_SHAPE_SIDE = 0.08

FIT_WINDOWS = [
    {
        "FitType": "FixedCandidate",
        "Event": "Cue",
        "Start": 0.25,
        "End": 0.50,
        "StatusColumn": "CuePeakStatus",
    },
    {
        "FitType": "FixedCandidate",
        "Event": "Target",
        "Start": 2.45,
        "End": 2.70,
        "StatusColumn": "TargetPeakStatus",
    },
]

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def gaussian(t, a, mu, sigma):
    return a * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def baseline_correct(trials, time):
    baseline = (time >= BASELINE_START) & (time < BASELINE_END)
    baseline_mean = trials[:, :, baseline].mean(axis=2, keepdims=True)
    return trials - baseline_mean


def extract_peak(signal, time):
    peak_index = int(np.argmax(signal))
    amplitude = float(signal[peak_index])

    if amplitude <= 0:
        status = "no_positive_peak"
    elif peak_index == 0:
        status = "left_boundary"
    elif peak_index == len(signal) - 1:
        status = "right_boundary"
    else:
        status = "valid"

    return {
        "index": peak_index,
        "amplitude": amplitude,
        "latency": float(time[peak_index]),
        "status": status,
    }


def peak_shape_status(latency, start, end):
    if end - latency < MIN_SHAPE_SIDE:
        return "truncated_right"
    if latency - start < MIN_SHAPE_SIDE:
        return "insufficient_shape"
    return "complete"


def fit_gaussian(time, signal, start, end):
    peak = extract_peak(signal, time)

    p0 = [
        max(peak["amplitude"], 1e-6),
        peak["latency"],
        max((end - start) / 6, 0.01),
    ]

    popt, _ = curve_fit(
        gaussian,
        time,
        signal,
        p0=p0,
        bounds=(
            [1e-8, start, 1e-4],
            [np.inf, end, np.inf],
        ),
        maxfev=20000,
    )

    fitted = gaussian(time, *popt)

    residual = signal - fitted
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((signal - signal.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    fit_valid = int(
        np.all(np.isfinite(popt))
        and popt[0] > 0
        and start <= popt[1] <= end
        and popt[2] > 0
    )

    return {
        "A": float(popt[0]),
        "Mu": float(popt[1]),
        "Sigma": float(popt[2]),
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "RMSE": rmse,
        "FitValid": fit_valid,
        "Fitted": fitted,
    }


def process_dataset(dataset_name, peak_table):
    mat = loadmat(CLEAN_DIR / f"{dataset_name}_clean.mat")

    trials = np.asarray(mat["trial_data"], dtype=float)
    relative_time = np.asarray(mat["relative_time"], dtype=float)
    cue_type = np.asarray(mat["cue_type"]).reshape(-1)

    time = relative_time[0]
    eeg = baseline_correct(trials[:, :3, :], time)

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    for row_index, (condition_name, condition_value) in enumerate(CONDITIONS):
        condition_trials = eeg[cue_type == condition_value]
        erp = condition_trials.mean(axis=0)

        for col_index, (channel_name, channel_index) in enumerate(CHANNELS):
            ax = axes[row_index, col_index]
            channel_erp = erp[channel_index]

            ax.plot(time, channel_erp, linewidth=1.3, label="ERP")

            for spec in FIT_WINDOWS:
                start = spec["Start"]
                end = spec["End"]
                mask = (time >= start) & (time <= end)

                window_time = time[mask]
                window_erp = channel_erp[mask]

                peak = extract_peak(window_erp, window_time)
                shape_status = peak_shape_status(
                    peak["latency"],
                    start,
                    end,
                )

                attempted = 1
                skip_reason = ""

                fixed_row = peak_table[
                    (peak_table["Dataset"] == dataset_name)
                    & (peak_table["Condition"] == condition_name)
                    & (peak_table["Channel"] == channel_name)
                ].iloc[0]

                old_status = fixed_row[spec["StatusColumn"]]

                if old_status != "valid":
                    attempted = 0
                    skip_reason = f"fixed_status_{old_status}"

                fit_result = {
                    "A": np.nan,
                    "Mu": np.nan,
                    "Sigma": np.nan,
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "FitValid": 0,
                }

                if attempted:
                    try:
                        fit_result = fit_gaussian(
                            window_time,
                            window_erp,
                            start,
                            end,
                        )

                        if fit_result["FitValid"]:
                            ax.plot(
                                window_time,
                                fit_result["Fitted"],
                                color={
                                    "Cue": "tab:orange",
                                    "Target": "tab:green",
                                }[spec["Event"]],
                                linestyle="--",
                                linewidth=1.5,
                                label={
                                    "Cue": "P300候选窗·提示后",
                                    "Target": "P300候选窗·目标后",
                                }[spec["Event"]],
                            )
                        else:
                            skip_reason = "fit_parameter_invalid"

                    except (RuntimeError, ValueError, FloatingPointError):
                        fit_result["FitValid"] = 0
                        skip_reason = "curve_fit_failed"

                observed_target_latency = (
                    peak["latency"] - TARGET_ONSET
                    if spec["Event"] == "Target"
                    else np.nan
                )

                gaussian_target_latency = (
                    fit_result["Mu"] - TARGET_ONSET
                    if spec["Event"] == "Target"
                    and np.isfinite(fit_result["Mu"])
                    else np.nan
                )

                gaussian_mu_at_boundary = (
                    int(
                        np.isclose(fit_result["Mu"], start, atol=1e-6)
                        or np.isclose(fit_result["Mu"], end, atol=1e-6)
                    )
                    if np.isfinite(fit_result["Mu"])
                    else np.nan
                )

                rows.append({
                    "Dataset": dataset_name,
                    "Condition": condition_name,
                    "Channel": channel_name,
                    "N_trials": len(condition_trials),
                    "FitType": spec["FitType"],
                    "Event": spec["Event"],
                    "WindowStart": start,
                    "WindowEnd": end,
                    "ObservedPeakAmplitude": peak["amplitude"],
                    "ObservedPeakLatencyFromCue": peak["latency"],
                    "ObservedPeakLatencyFromTarget": observed_target_latency,
                    "ObservedPeakStatus": peak["status"],
                    "PeakShapeStatus": shape_status,
                    "Attempted": attempted,
                    "SkipReason": skip_reason,
                    "GaussianA": fit_result["A"],
                    "GaussianMuFromCue": fit_result["Mu"],
                    "GaussianMuFromTarget": gaussian_target_latency,
                    "GaussianSigma": fit_result["Sigma"],
                    "GaussianMuAtBoundary": gaussian_mu_at_boundary,
                    "FitValid": fit_result["FitValid"],
                    "R2": fit_result["R2"],
                    "RMSE": fit_result["RMSE"],
                })

            ax.axvline(0, linestyle="--", linewidth=1)
            ax.axvline(TARGET_ONSET, linestyle=":", linewidth=1)

            ax.axvspan(0.25, 0.50, alpha=0.08)
            ax.axvspan(2.45, 2.70, alpha=0.08)

            ax.set_title(f"{condition_name} · {channel_name}")
            ax.set_xlabel("相对提示 onset 的时间 (s)")
            ax.set_ylabel("基线校正后 ERP 幅值（原始数据单位）")
            ax.grid(alpha=0.22)
            ax.legend(fontsize=8)

    fig.suptitle(
        f"{dataset_name} P300 候选窗高斯拟合",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / f"{dataset_name}_ERP高斯拟合.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return rows


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    peak_table = pd.read_csv(
        ERP_DIR / "ERP候选峰指标.csv"
    )

    all_rows = []

    for dataset_name in DATASETS:
        all_rows.extend(
            process_dataset(
                dataset_name,
                peak_table,
            )
        )

    result = pd.DataFrame(all_rows)

    result.to_csv(
        RESULT_DIR / "ERP高斯拟合参数.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(result)


if __name__ == "__main__":
    main()
