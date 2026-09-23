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
    r"\A组_Task1_10个Trial_阶段标注原始EEG.png"
)

num_trials = 10

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
# 事件检测
# ============================================================
cue = df["VisCue"].to_numpy()
action = df["Action"].to_numpy()

cue_onsets = np.where((cue != 0) & (np.r_[0, cue[:-1]] == 0))[0]
action_onsets = np.where((action != 0) & (np.r_[0, action[:-1]] == 0))[0]

# ============================================================
# 提取前10个 trial
# ============================================================
trials = []

for i in range(num_trials):
    cue_idx = cue_onsets[i]
    next_cue_idx = cue_onsets[i + 1] if i < len(cue_onsets) - 1 else len(df) - 1

    t_cue = df.loc[cue_idx, "TimeStamp"]          # 三角形显示开始（通道8）
    t_pre = t_cue - 0.5                           # 提示前0.5s开始
    t_cue_end = t_cue + 0.2                       # 三角形显示结束
    t_wait = t_cue_end                            # 2s等待开始
    t_target = t_cue_end + 2.0                    # 目标显示开始

    cue_type = int(df.loc[cue_idx, "VisCue"])
    direction = "左刺激" if cue_type == -1 else "右刺激"

    # 当前 trial 内的第一个动作开始（通道9）
    cand = action_onsets[(action_onsets > cue_idx) & (action_onsets < next_cue_idx)]
    if len(cand) > 0:
        action_idx = cand[0]
        t_action = df.loc[action_idx, "TimeStamp"]
    else:
        t_action = np.nan

    # 截取绘图窗口：提示前1s，到目标显示后/动作后稍微延长一点
    seg_start = t_cue - 1.0
    seg_end = t_target + 0.6
    if not np.isnan(t_action):
        seg_end = max(seg_end, t_action + 0.4)

    seg = df[(df["TimeStamp"] >= seg_start) & (df["TimeStamp"] <= seg_end)].copy()

    trials.append(
        {
            "name": f"Trial-{i+1}",
            "direction": direction,
            "segment": seg,
            "t_pre": t_pre,
            "t_cue": t_cue,
            "t_wait": t_wait,
            "t_target": t_target,
            "t_action": t_action,
        }
    )

# ============================================================
# 配色
# ============================================================
ch_colors = get_sci_deep_colors(3)   # F3 / Fz / F4

# 阴影色（SCI风格偏淡）
pre_fill = "#DCEAF7"      # 提示前0.5s
cue_fill = "#FBE5D6"      # 三角形显示0.2s
wait_fill = "#DFF0E3"     # 显示后等待2s

# 阶段起始虚线颜色
pre_line = "#4C78A8"
cue_line = "#F58518"
wait_line = "#54A24B"
target_line = "#7570B3"
action_line = "#E64B35"

# ============================================================
# 绘图
# ============================================================
fig, axes = plt.subplots(5, 2, figsize=(18, 20))
axes = axes.flatten()

channels = ["F3", "Fz", "F4"]

for ax, trial in zip(axes, trials):
    seg = trial["segment"]

    # 三个通道
    for c, ch in zip(ch_colors, channels):
        ax.plot(seg["TimeStamp"], seg[ch], color=c, linewidth=1.2, label=ch)

    # 阶段阴影
    ax.axvspan(trial["t_pre"], trial["t_cue"], color=pre_fill, alpha=0.45)
    ax.axvspan(trial["t_cue"], trial["t_wait"], color=cue_fill, alpha=0.55)
    ax.axvspan(trial["t_wait"], trial["t_target"], color=wait_fill, alpha=0.45)

    # 各阶段开始虚线
    ax.axvline(trial["t_pre"], color=pre_line, linestyle="--", linewidth=1.2)
    ax.axvline(trial["t_cue"], color=cue_line, linestyle="--", linewidth=1.2)
    ax.axvline(trial["t_wait"], color=wait_line, linestyle="--", linewidth=1.2)
    ax.axvline(trial["t_target"], color=target_line, linestyle="--", linewidth=1.2)

    # 动作开始虚线（通道9）
    if not np.isnan(trial["t_action"]):
        ax.axvline(trial["t_action"], color=action_line, linestyle="--", linewidth=1.3)

    # 子图标题
    ax.set_title(
        f"{trial['name']}  {trial['direction']}",
        fontsize=12,
        pad=6
    )

    ax.set_ylabel("EEG", fontsize=11)
    ax.grid(True, alpha=0.25)

    # 自动y范围
    y_min = seg[channels].min().min()
    y_max = seg[channels].max().max()
    pad = 0.08 * (y_max - y_min + 1e-9)
    ax.set_ylim(y_min - pad, y_max + pad)

# 底部x轴
for ax in axes[-2:]:
    ax.set_xlabel("TimeStamp (s)", fontsize=11)

# 总标题
fig.suptitle("A组 Task-1 连续10个Trial原始 EEG 信号", fontsize=22, y=0.992)

# 图例（全局）
legend_handles = [
    Line2D([0], [0], color=ch_colors[0], lw=1.5, label="F3"),
    Line2D([0], [0], color=ch_colors[1], lw=1.5, label="Fz"),
    Line2D([0], [0], color=ch_colors[2], lw=1.5, label="F4"),
    Patch(facecolor=pre_fill, edgecolor="none", alpha=0.45, label="提示前 0.5 s"),
    Patch(facecolor=cue_fill, edgecolor="none", alpha=0.55, label="三角形显示 0.2 s"),
    Patch(facecolor=wait_fill, edgecolor="none", alpha=0.45, label="显示后等待 2 s"),
    Line2D([0], [0], color=pre_line, linestyle="--", lw=1.3, label="提示前阶段开始"),
    Line2D([0], [0], color=cue_line, linestyle="--", lw=1.3, label="三角形显示开始"),
    Line2D([0], [0], color=wait_line, linestyle="--", lw=1.3, label="2 s 等待开始"),
    Line2D([0], [0], color=target_line, linestyle="--", lw=1.3, label="目标显示开始"),
    Line2D([0], [0], color=action_line, linestyle="--", lw=1.3, label="动作开始（通道9）"),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.968),
    ncol=4,
    fontsize=10,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 1, 0.955])

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()