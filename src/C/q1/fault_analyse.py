import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.color_palettes import get_sci_deep_colors


# ============================================================
# 中文字体
# ============================================================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 路径
# ============================================================
csv_path = (
    Path(r"D:\8\Desktop\CMathc\src\C\q1\output")
    / "EEG异常检测指标.csv"
)

output_path = (
    Path(r"D:\8\Desktop\CMathc\src\C\q1\output")
    / "04_EEG异常情况及严重程度统计图_修正版.png"
)


# ============================================================
# 读取数据
# ============================================================
metrics = pd.read_csv(csv_path)


# ============================================================
# 确定“基线不稳定”列名
# ============================================================
if "baseline_instability_z" in metrics.columns:
    baseline_col = "baseline_instability_z"
    baseline_label = "基线不稳定"
elif "baseline_z" in metrics.columns:
    baseline_col = "baseline_z"
    baseline_label = "基线不稳定"
else:
    baseline_col = "drift_z"
    baseline_label = "基线不稳定"


# ============================================================
# Trial级汇总
# ============================================================
trial_summary = (
    metrics
    .groupby("trial")
    .agg(
        clipping_score=("longest_clip_s", "max"),
        jump_score=("jump_z", lambda x: np.max(np.abs(x))),
        baseline_score=(baseline_col, lambda x: np.max(np.abs(x))),
    )
    .reset_index()
)


# ============================================================
# 严重程度分级
# ============================================================
def level_clipping(x):
    if x <= 0:
        return "无"
    elif x <= 0.05:
        return "轻度"
    elif x <= 0.20:
        return "中度"
    else:
        return "重度"


def level_zscore(x):
    if x <= 2.0:
        return "无"
    elif x <= 3.5:
        return "轻度"
    elif x <= 6.0:
        return "中度"
    else:
        return "重度"


trial_summary["削顶饱和"] = trial_summary["clipping_score"].apply(level_clipping)
trial_summary["瞬时跳变"] = trial_summary["jump_score"].apply(level_zscore)
trial_summary[baseline_label] = trial_summary["baseline_score"].apply(level_zscore)


anomaly_types = ["削顶饱和", "瞬时跳变", baseline_label]


# ============================================================
# 左图：各类异常出现的 Trial 数
# ============================================================
count_records = []
for col in anomaly_types:
    n = (trial_summary[col] != "无").sum()
    count_records.append({
        "异常类型": col,
        "Trial数": n
    })

count_df = pd.DataFrame(count_records)


# ============================================================
# 右图：严重程度分布
# ============================================================
severity_order = ["轻度", "中度", "重度"]

severity_records = []
for col in anomaly_types:
    for level in severity_order:
        n = (trial_summary[col] == level).sum()
        severity_records.append({
            "异常类型": col,
            "严重程度": level,
            "Trial数": n
        })

severity_df = pd.DataFrame(severity_records)

severity_pivot = severity_df.pivot(
    index="异常类型",
    columns="严重程度",
    values="Trial数"
).fillna(0)

severity_pivot = severity_pivot.reindex(anomaly_types)


# ============================================================
# SCI 配色
# ============================================================
# 左图三根柱子
left_colors = [
    "#B7C9E2",
    "#E8C6A8",
    "#BFD3B4",
]

# 右图严重程度三种颜色（浅-中-深）
severity_colors = {
    "轻度": "#9ecae1",   # 浅蓝
    "中度": "#fdb863",   # 橙黄
    "重度": "#d73027",   # 深红
}


# ============================================================
# 绘图
# ============================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6.5)
)

# ------------------------------------------------------------
# 左图：各类异常出现的 Trial 数
# ------------------------------------------------------------
ax = axes[0]

bars = ax.bar(
    count_df["异常类型"],
    count_df["Trial数"],
    color=left_colors,
    edgecolor="black",
    linewidth=0.8
)

ax.set_title("各类异常出现的 Trial 数", fontsize=16)
ax.set_xlabel("异常类型", fontsize=13)
ax.set_ylabel("Trial 数", fontsize=13)
ax.grid(axis="y", alpha=0.25)

# 给顶部留白
ymax_left = count_df["Trial数"].max()
ax.set_ylim(0, ymax_left + 6)

# 柱顶数字
for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.6,
        f"{int(h)}",
        ha="center",
        va="bottom",
        fontsize=12
    )


# ------------------------------------------------------------
# 右图：严重程度分布（堆叠柱状图）
# ------------------------------------------------------------
ax = axes[1]

bottom = np.zeros(len(severity_pivot))

for level in severity_order:
    values = severity_pivot[level].values
    ax.bar(
        severity_pivot.index,
        values,
        bottom=bottom,
        color=severity_colors[level],
        edgecolor="black",
        linewidth=0.8,
        label=level
    )
    bottom += values

# 顶部留白
totals = severity_pivot.sum(axis=1).values
ymax_right = totals.max()
ax.set_ylim(0, ymax_right + 6)

# 总数标注
for i, total in enumerate(totals):
    ax.text(
        i,
        total + 0.6,
        f"{int(total)}",
        ha="center",
        va="bottom",
        fontsize=12
    )

ax.set_title("各类异常的严重程度分布", fontsize=16)
ax.set_xlabel("异常类型", fontsize=13)
ax.set_ylabel("Trial 数", fontsize=13)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="严重程度", fontsize=11, title_fontsize=11)


# ============================================================
# 总标题与保存
# ============================================================
fig.suptitle(
    "A组 Task-1 EEG 异常情况及严重程度统计图",
    fontsize=20,
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()