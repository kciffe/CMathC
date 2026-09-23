
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

PRE_FILL = "#DCEAF7"
CUE_FILL = "#FBE5D6"
WAIT_FILL = "#DFF0E3"
P300_CUE_FILL = "#AFC6E9"
P300_TARGET_FILL = "#C8B6E2"

PRE_LINE = "#4C78A8"
CUE_LINE = "#F58518"
WAIT_LINE = "#54A24B"
TARGET_LINE = "#9C755F"
ACTION_LINE = "#E64B35"


def plot_trials(
    trial_data,
    relative_time,
    cue_type,
    action_onsets,
    save_path,
    title,
    event_label="事件开始（通道9）",
):
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))

    left = np.where(cue_type == -1)[0][:5]
    right = np.where(cue_type == 1)[0][:5]
    line_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row in range(5):
        for col in range(2):
            ax = axes[row, col]

            idx = left[row] if col == 0 else right[row]

            eeg = trial_data[idx]
            t = relative_time[idx]

            # Fz,F3,F4三个通道
            for ch, name, color in zip([1, 0, 2], ["F3", "Fz", "F4"], line_colors):
                ax.plot(
                    t,
                    eeg[ch],
                    linewidth=1,
                    color=color,
                    label=name
                )

            # 阶段区间与1clear_data.py保持一致，时间均相对提示开始。
            ax.axvspan(-0.5, 0.0, color=PRE_FILL, alpha=0.45)
            ax.axvspan(0.0, 0.2, color=CUE_FILL, alpha=0.60)
            ax.axvspan(0.2, 2.2, color=WAIT_FILL, alpha=0.45)
            ax.axvspan(0.25, 0.50, color=P300_CUE_FILL, alpha=0.55)
            ax.axvspan(2.45, 2.70, color=P300_TARGET_FILL, alpha=0.55)

            ax.axvline(-0.5, color=PRE_LINE, linestyle="--", linewidth=1.0)
            ax.axvline(0.0, color=CUE_LINE, linestyle="--", linewidth=1.0)
            ax.axvline(0.2, color=WAIT_LINE, linestyle="--", linewidth=1.0)
            ax.axvline(2.2, color=TARGET_LINE, linestyle="--", linewidth=1.0)
            if np.isfinite(action_onsets[idx]):
                ax.axvline(action_onsets[idx], color=ACTION_LINE, linestyle="--", linewidth=1.1)

            ax.set_title(
                f"Trial-{idx+1} {'左刺激' if cue_type[idx]==-1 else '右刺激'}"
            )

            ax.set_xlabel("Time(s)")
            ax.set_ylabel("EEG")
            ax.grid(alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=line_colors[0], lw=1.5, label="F3"),
        Line2D([0], [0], color=line_colors[1], lw=1.5, label="Fz"),
        Line2D([0], [0], color=line_colors[2], lw=1.5, label="F4"),
        Patch(facecolor=PRE_FILL, edgecolor="none", alpha=0.45, label="提示前 0.5 s"),
        Patch(facecolor=CUE_FILL, edgecolor="none", alpha=0.60, label="三角形显示 0.2 s"),
        Patch(facecolor=WAIT_FILL, edgecolor="none", alpha=0.45, label="显示后等待 2 s"),
        Patch(facecolor=P300_CUE_FILL, edgecolor="none", alpha=0.55, label="提示后 P300 候选时窗 (250–500 ms)"),
        Patch(facecolor=P300_TARGET_FILL, edgecolor="none", alpha=0.55, label="目标后 P300 候选时窗 (250–500 ms)"),
        Line2D([0], [0], color=PRE_LINE, linestyle="--", lw=1.2, label="提示前阶段开始"),
        Line2D([0], [0], color=CUE_LINE, linestyle="--", lw=1.2, label="三角形显示开始"),
        Line2D([0], [0], color=WAIT_LINE, linestyle="--", lw=1.2, label="2 s 等待开始"),
        Line2D([0], [0], color=TARGET_LINE, linestyle="--", lw=1.2, label="目标显示开始"),
        Line2D(
            [0],
            [0],
            color=ACTION_LINE,
            linestyle="--",
            lw=1.2,
            label=event_label,
        ),
    ]

    fig.suptitle(title, fontsize=18, y=1.04)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
