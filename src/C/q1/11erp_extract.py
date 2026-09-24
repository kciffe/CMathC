# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.io import loadmat


DATASETS = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "output" / "8riemann_denoise"
RESULT_DIR = PROJECT_DIR / "output" / "11erp_extract"

CHANNELS = [
    ("F3", 1),
    ("Fz", 0),
    ("F4", 2),
]

BASELINE_START, BASELINE_END = -0.2, 0.0
CUE_WINDOW = (0.25, 0.50)
TARGET_ONSET = 2.20
TARGET_WINDOW = (2.45, 2.70)
PLOT_START, PLOT_END = -0.2, 3.0
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
    baseline = (time >= BASELINE_START) & (time < BASELINE_END)
    baseline_mean = trials[:, :, baseline].mean(axis=2, keepdims=True)
    return trials - baseline_mean


def extract_peak(erp, time, start, end):
    mask = (time >= start) & (time <= end)
    window_time = time[mask]
    window_erp = erp[mask]
    peak_index = np.argmax(window_erp)
    amplitude = float(window_erp[peak_index])

    if amplitude <= 0:
        status = "no_positive_peak"
    elif peak_index == 0:
        status = "left_boundary"
    elif peak_index == len(window_erp) - 1:
        status = "right_boundary"
    else:
        status = "valid"

    return {
        "amplitude": amplitude,
        "latency": float(window_time[peak_index]),
        "left_boundary": int(peak_index == 0),
        "right_boundary": int(peak_index == len(window_time) - 1),
        "has_positive_peak": int(amplitude > 0),
        "status": status,
    }


def validate_clean_data(trials, relative_time, cue_type, drop):
    if np.any(drop.astype(bool)):
        raise ValueError("clean.mat 中仍存在 drop=1 的 Trial")
    if relative_time.shape != (trials.shape[0], trials.shape[2]):
        raise ValueError("relative_time 与 trial_data 的时间维度不一致")


def process_dataset(dataset_name):
    mat = loadmat(INPUT_DIR / f"{dataset_name}_clean.mat")

    trials = np.asarray(mat["trial_data"], dtype=float)
    relative_time = np.asarray(mat["relative_time"], dtype=float)
    cue_type = np.asarray(mat["cue_type"]).reshape(-1)
    drop = np.asarray(mat["drop"]).reshape(-1)

    validate_clean_data(trials, relative_time, cue_type, drop)

    time = relative_time[0]
    eeg = baseline_correct(trials[:, :3, :], time)

    plot_mask = (time >= PLOT_START) & (time <= PLOT_END)
    plot_time = time[plot_mask]

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)

    for ax, (condition_name, condition_value) in zip(
        axes,
        [("左条件", -1), ("右条件", 1)],
    ):
        condition_trials = eeg[cue_type == condition_value]
        erp = condition_trials.mean(axis=0)

        for channel_name, channel_index in CHANNELS:
            channel_erp = erp[channel_index]

            ax.plot(
                plot_time,
                channel_erp[plot_mask],
                linewidth=1.5,
                label=channel_name,
            )

            cue_peak = extract_peak(
                channel_erp,
                time,
                CUE_WINDOW[0],
                CUE_WINDOW[1],
            )

            target_peak = extract_peak(
                channel_erp,
                time,
                TARGET_WINDOW[0],
                TARGET_WINDOW[1],
            )

            rows.append({
                "Dataset": dataset_name,
                "Condition": condition_name,
                "Channel": channel_name,
                "N_trials": len(condition_trials),
                "CuePeakAmplitude": cue_peak["amplitude"],
                "CuePeakLatencyFromCue": cue_peak["latency"],
                "CuePeakAtLeftBoundary": cue_peak["left_boundary"],
                "CuePeakAtRightBoundary": cue_peak["right_boundary"],
                "CueHasPositivePeak": cue_peak["has_positive_peak"],
                "CuePeakStatus": cue_peak["status"],
                "TargetPeakAmplitude": target_peak["amplitude"],
                "TargetPeakLatencyFromCue": target_peak["latency"],
                "TargetPeakLatencyFromTarget": target_peak["latency"] - TARGET_ONSET,
                "TargetPeakAtLeftBoundary": target_peak["left_boundary"],
                "TargetPeakAtRightBoundary": target_peak["right_boundary"],
                "TargetHasPositivePeak": target_peak["has_positive_peak"],
                "TargetPeakStatus": target_peak["status"],
            })

        ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1)
        ax.axvline(TARGET_ONSET, color="tab:blue", linestyle=":", linewidth=1)
        ax.axvspan(
            CUE_WINDOW[0],
            CUE_WINDOW[1],
            color=P300_CUE_COLOR,
            alpha=0.55,
            zorder=0,
        )
        ax.axvspan(
            TARGET_WINDOW[0],
            TARGET_WINDOW[1],
            color=P300_TARGET_COLOR,
            alpha=0.55,
            zorder=0,
        )

        ax.set_title(f"{condition_name} ERP (n={len(condition_trials)})")
        ax.set_xlabel("相对提示 onset 的时间 (s)")
        ax.set_ylabel("基线校正后 ERP 幅值（原始数据单位）")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.22)

    fig.suptitle(
        f"{dataset_name} 正式 ERP：左右条件叠加平均",
        fontsize=16,
        y=0.985,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    window_handles = [
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
    ]
    stage_handles = [
        Line2D(
            [0], [0], color="tab:blue", linewidth=1, linestyle="--",
            label="提示开始（0 s）",
        ),
        Line2D(
            [0], [0], color="tab:blue", linewidth=1, linestyle=":",
            label="目标显示开始（2.20 s）",
        ),
    ]
    handles.extend(window_handles + stage_handles)
    labels.extend(handle.get_label() for handle in window_handles + stage_handles)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.84))

    fig.savefig(
        RESULT_DIR / f"{dataset_name}_ERP.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    return rows


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset_name in DATASETS:
        all_rows.extend(process_dataset(dataset_name))

    result = pd.DataFrame(all_rows)
    result.to_csv(
        RESULT_DIR / "ERP候选峰指标.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(result)


if __name__ == "__main__":
    main()
