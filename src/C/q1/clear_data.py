import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys
from pathlib import Path

# ============================================================
# 中文字体
# ============================================================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.color_palettes import get_sci_deep_colors

# ============================================================
# 参数
# ============================================================
mat_path = (
    r"D:\8\Desktop\CMathc\data"
    r"\第二十三届中国研究生数学建模竞赛+-+中文题目"
    r"\中文题目\C题"
    r"\VisualCogA_Task-1.mat"
)

output_path = (
    r"D:\8\Desktop\CMathc\src\C\q1\output"
    r"\A组_Task1_左5右5_阶段与P300候选时窗.png"
)

# ============================================================
# 读取数据
# ============================================================
mat = loadmat(mat_path)

fs = int(np.asarray(mat["SampleRate"]).squeeze())
data = np.asarray(mat["data"])

df = pd.DataFrame(
    data.T,
    columns=[
        "Fz",
        "F3",
        "F4",
        "FzDecon",
        "F3Decon",
        "F4Decon",
        "ECG",
        "VisCue",
        "Action",
        "TimeStamp",
    ],
)

# ============================================================
# 找刺激开始与动作开始
# ============================================================
cue = df["VisCue"].to_numpy()
action = df["Action"].to_numpy()

cue_onsets = np.where((cue != 0) & (np.r_[0, cue[:-1]] == 0))[0]
action_onsets = np.where((action != 0) & (np.r_[0, action[:-1]] == 0))[0]

# ============================================================
# 分出左刺激 / 右刺激
# ============================================================
left_cue_onsets = [idx for idx in cue_onsets if int(df.loc[idx, "VisCue"]) == -1]
right_cue_onsets = [idx for idx in cue_onsets if int(df.loc[idx, "VisCue"]) == 1]

left_cue_onsets = left_cue_onsets[:5]
right_cue_onsets = right_cue_onsets[:5]

# ============================================================
# 提取 trial 信息
# ============================================================
def build_trial_info(cue_idx, trial_name, direction):
    next_cue_candidates = cue_onsets[cue_onsets > cue_idx]
    next_cue_idx = next_cue_candidates[0] if len(next_cue_candidates) > 0 else len(df) - 1

    t_cue = df.loc[cue_idx, "TimeStamp"]          # 三角形提示开始
    t_pre = t_cue - 0.5                           # 提示前0.5s开始
    t_cue_end = t_cue + 0.2                       # 三角形提示结束
    t_wait = t_cue_end                            # 2s等待开始
    t_target = t_cue_end + 2.0                    # 目标显示开始

    # 第一个动作开始
    cand = action_onsets[(action_onsets > cue_idx) & (action_onsets < next_cue_idx)]
    if len(cand) > 0:
        action_idx = cand[0]
        t_action = df.loc[action_idx, "TimeStamp"]
    else:
        t_action = np.nan

    # 两段 P300 候选时窗
    p300_1_start = t_cue + 0.25
    p300_1_end = t_cue + 0.50

    p300_2_start = t_target + 0.25
    p300_2_end = t_target + 0.50

    # 画图范围：提示前1s，到目标后0.8s；若动作更晚，延长到动作后0.3s
    seg_start = t_cue - 1.0
    seg_end = t_target + 0.8
    if not np.isnan(t_action):
        seg_end = max(seg_end, t_action + 0.3)

    seg = df[(df["TimeStamp"] >= seg_start) & (df["TimeStamp"] <= seg_end)].copy()

    return {
        "name": trial_name,
        "direction": direction,
        "segment": seg,
        "t_pre": t_pre,
        "t_cue": t_cue,
        "t_wait": t_wait,
        "t_target": t_target,
        "t_action": t_action,
        "p300_1_start": p300_1_start,
        "p300_1_end": p300_1_end,
        "p300_2_start": p300_2_start,
        "p300_2_end": p300_2_end,
    }

left_trials = [
    build_trial_info(idx, f"Trial-L{i+1}", "左刺激")
    for i, idx in enumerate(left_cue_onsets)
]

right_trials = [
    build_trial_info(idx, f"Trial-R{i+1}", "右刺激")
    for i, idx in enumerate(right_cue_onsets)
]

# ============================================================
# 配色
# ============================================================
ch_colors = get_sci_deep_colors(3)  # F3 / Fz / F4

# 阶段阴影色（浅）
pre_fill = "#DCEAF7"       # 提示前0.5s
cue_fill = "#FBE5D6"       # 三角形显示0.2s
wait_fill = "#DFF0E3"      # 等待2s

# P300 候选时窗（稍深）
p300_fill_1 = "#AFC6E9"    # 提示后P300候选时窗
p300_fill_2 = "#C8B6E2"    # 目标后P300候选时窗

# 虚线颜色
pre_line = "#4C78A8"
cue_line = "#F58518"
wait_line = "#54A24B"
target_line = "#9C755F"
action_line = "#E64B35"

# ============================================================
# 绘图：5行 × 2列
# 左列=前5个左刺激；右列=前5个右刺激
# ============================================================
fig, axes = plt.subplots(5, 2, figsize=(16, 20))

channels = ["F3", "Fz", "F4"]

for row in range(5):
    for col in range(2):
        ax = axes[row, col]

        trial = left_trials[row] if col == 0 else right_trials[row]
        seg = trial["segment"]

        # EEG三通道
        for color, ch in zip(ch_colors, channels):
            ax.plot(
                seg["TimeStamp"],
                seg[ch],
                color=color,
                linewidth=1.2,
                label=ch
            )

        # 阶段阴影
        ax.axvspan(trial["t_pre"], trial["t_cue"], color=pre_fill, alpha=0.45)
        ax.axvspan(trial["t_cue"], trial["t_wait"], color=cue_fill, alpha=0.60)
        ax.axvspan(trial["t_wait"], trial["t_target"], color=wait_fill, alpha=0.45)

        # P300 候选时窗（覆盖在上层，用稍深颜色）
        ax.axvspan(trial["p300_1_start"], trial["p300_1_end"], color=p300_fill_1, alpha=0.55)
        ax.axvspan(trial["p300_2_start"], trial["p300_2_end"], color=p300_fill_2, alpha=0.55)

        # 各阶段开始虚线
        ax.axvline(trial["t_pre"], color=pre_line, linestyle="--", linewidth=1.0)
        ax.axvline(trial["t_cue"], color=cue_line, linestyle="--", linewidth=1.0)
        ax.axvline(trial["t_wait"], color=wait_line, linestyle="--", linewidth=1.0)
        ax.axvline(trial["t_target"], color=target_line, linestyle="--", linewidth=1.0)

        # 动作开始虚线
        if not np.isnan(trial["t_action"]):
            ax.axvline(trial["t_action"], color=action_line, linestyle="--", linewidth=1.1)

        # 标题
        ax.set_title(
            f"{trial['name']}  {trial['direction']}",
            fontsize=12,
            pad=4
        )

        ax.set_ylabel("EEG", fontsize=11)
        ax.grid(True, alpha=0.22)

        # 自动y范围
        y_min = seg[channels].min().min()
        y_max = seg[channels].max().max()
        pad = 0.08 * (y_max - y_min + 1e-9)
        ax.set_ylim(y_min - pad, y_max + pad)

# 底部横轴
for ax in axes[-1, :]:
    ax.set_xlabel("TimeStamp (s)", fontsize=11)

# 列标题（更清楚说明左右分组）
axes[0, 0].text(
    0.5, 1.18, "前5个左刺激 Trial",
    transform=axes[0, 0].transAxes,
    ha="center", va="bottom", fontsize=14, fontweight="bold"
)

axes[0, 1].text(
    0.5, 1.18, "前5个右刺激 Trial",
    transform=axes[0, 1].transAxes,
    ha="center", va="bottom", fontsize=14, fontweight="bold"
)

# 全局图例
legend_handles = [
    Line2D([0], [0], color=ch_colors[0], lw=1.5, label="F3"),
    Line2D([0], [0], color=ch_colors[1], lw=1.5, label="Fz"),
    Line2D([0], [0], color=ch_colors[2], lw=1.5, label="F4"),

    Patch(facecolor=pre_fill, edgecolor="none", alpha=0.45, label="提示前 0.5 s"),
    Patch(facecolor=cue_fill, edgecolor="none", alpha=0.60, label="三角形显示 0.2 s"),
    Patch(facecolor=wait_fill, edgecolor="none", alpha=0.45, label="显示后等待 2 s"),

    Patch(facecolor=p300_fill_1, edgecolor="none", alpha=0.55, label="提示后 P300 候选时窗 (250–500 ms)"),
    Patch(facecolor=p300_fill_2, edgecolor="none", alpha=0.55, label="目标后 P300 候选时窗 (250–500 ms)"),

    Line2D([0], [0], color=pre_line, linestyle="--", lw=1.2, label="提示前阶段开始"),
    Line2D([0], [0], color=cue_line, linestyle="--", lw=1.2, label="三角形显示开始"),
    Line2D([0], [0], color=wait_line, linestyle="--", lw=1.2, label="2 s 等待开始"),
    Line2D([0], [0], color=target_line, linestyle="--", lw=1.2, label="目标显示开始"),
    Line2D([0], [0], color=action_line, linestyle="--", lw=1.2, label="动作开始（通道9）"),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=4,
    fontsize=10,
    frameon=True
)

fig.suptitle(
    "A组 Task-1 左右刺激 Trial 原始 EEG 及阶段/P300候选时窗标注",
    fontsize=20,
    y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()