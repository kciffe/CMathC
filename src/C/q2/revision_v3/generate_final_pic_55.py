"""Generate Chinese PNG figures from the selected 55% within-record CV run.

The script combines saved out-of-fold predictions with the matching Q1-cleaned
Stage1 EEG trials. It does not infer or recreate hidden LGN/cortical states.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
Q2_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "math-model-agent" / "code"))

from algorithms.sci_figures import paper_figure_rc_params  # noqa: E402
from config import DATASETS, FEATURE_WINDOWS_MS, REAL_ROOT  # noqa: E402
from evaluate import extract_features  # noqa: E402
from real_data import load_cases_with_audit, load_dataset  # noqa: E402


SOURCE_CSV = Q2_ROOT / "output" / "revision_v3_random_split_search" / "命中轮次预测.csv"
OUTPUT_DIR = Q2_ROOT / "output" / "final_pic"
RECORD_ORDER = tuple(name for records in DATASETS.values() for name in records)
RECORD_LABELS = {
    "VisualCogA_Task-1": "A组记录1",
    "VisualCogA_Task-2": "A组记录2",
    "VisualCogB_Task-1": "B组记录1",
    "VisualCogB_Task-2": "B组记录2",
}
MODE_NAMES = ("共同模式", "左右侧化模式", "中央－两侧模式")
MODE_COLORS = ("#3569A8", "#4B8D62", "#C77C35")
LEFT_COLOR = "#3569A8"
RIGHT_COLOR = "#D17A35"
CORRECT_COLOR = "#39805A"
ERROR_COLOR = "#B64A4A"
CHANCE = 0.5

plt.rcParams.update(paper_figure_rc_params())
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def _read_predictions() -> list[dict[str, str]]:
    with SOURCE_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"记录", "试次序号", "真实方向", "预测方向", "右向判别分数",
                "是否正确", "交叉验证折", "搜索轮次", "随机种子"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected prediction CSV schema: {SOURCE_CSV}")
    return rows


def _summaries(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result = {}
    for record in RECORD_ORDER:
        local = [row for row in rows if row["记录"] == RECORD_LABELS[record]]
        if not local:
            raise ValueError(f"Missing rows for {record}")
        truth = np.array([row["真实方向"] for row in local])
        pred = np.array([row["预测方向"] for row in local])
        recall_left = float(np.mean(pred[truth == "左"] == "左"))
        recall_right = float(np.mean(pred[truth == "右"] == "右"))
        result[record] = {
            "n": len(local),
            "correct": int(np.count_nonzero(truth == pred)),
            "accuracy": float(np.mean(truth == pred)),
            "balanced_accuracy": 0.5 * (recall_left + recall_right),
            "recall_left": recall_left,
            "recall_right": recall_right,
        }
    return result


def _boxed_legend(fig, handles, *, y: float = 0.925, ncol: int = 3) -> None:
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
                        ncol=ncol, frameon=True, fancybox=False, framealpha=1.0,
                        borderpad=0.55, handlelength=2.2, columnspacing=1.5,
                        labelspacing=0.45)
    legend.get_frame().set_edgecolor("#7A7A7A")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")


def _plot_performance(rows: list[dict[str, str]], summary: dict[str, dict[str, float]]) -> None:
    counts = sum(item["n"] for item in summary.values())
    correct = sum(item["correct"] for item in summary.values())
    macro_ba = float(np.mean([item["balanced_accuracy"] for item in summary.values()]))
    pooled_acc = correct / counts
    round_no = rows[0]["搜索轮次"]
    seed = rows[0]["随机种子"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [0.9, 1.25]})
    fig.suptitle(f"左右三角刺激分类结果\n记录内分层随机五折：第{round_no}轮，随机种子{seed}，共{counts}个试次",
                 fontsize=14, y=0.99)
    handles = [
        Patch(facecolor="#557A95", label=f"记录宏平均平衡准确率  {macro_ba:.1%}"),
        Patch(facecolor="#D17A35", label=f"合并试次准确率  {pooled_acc:.1%}"),
        Line2D([0], [0], color="#555555", linestyle="--", label="随机参考水平  50%"),
    ]
    _boxed_legend(fig, handles, y=0.91, ncol=3)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.77, wspace=0.32)

    ax = axes[0]
    labels = ["记录宏平均\n平衡准确率", "合并试次\n准确率"]
    values = [macro_ba * 100, pooled_acc * 100]
    bars = ax.barh([1, 0], values, color=["#557A95", RIGHT_COLOR], height=0.55)
    ax.axvline(CHANCE * 100, color="#555555", linestyle="--", linewidth=1.1)
    ax.set_yticks([1, 0], labels)
    ax.set_xlim(0, 80)
    ax.set_xlabel("准确率（%）")
    ax.set_title("总体表现")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", ha="left", fontsize=10)

    ax = axes[1]
    local_values = [summary[record]["balanced_accuracy"] * 100 for record in RECORD_ORDER]
    local_labels = [f"{RECORD_LABELS[record]}（n={summary[record]['n']}）" for record in RECORD_ORDER]
    bars = ax.barh(np.arange(len(RECORD_ORDER)), local_values,
                   color=["#557A95", "#91A9B8", "#4B8D62", "#8BB96D"], height=0.58)
    ax.axvline(CHANCE * 100, color="#555555", linestyle="--", linewidth=1.1)
    ax.set_yticks(np.arange(len(RECORD_ORDER)), local_labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("平衡准确率（%）")
    ax.set_title("各记录表现")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, local_values):
        ax.text(min(value + 1.5, 94), bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", ha="left", fontsize=9)
    ax.invert_yaxis()
    fig.savefig(OUTPUT_DIR / "01_随机五折分类结果.png", dpi=300, facecolor="white")
    plt.close(fig)


def _plot_trial_scores(rows: list[dict[str, str]], summary: dict[str, dict[str, float]]) -> None:
    rng = np.random.default_rng(20260928)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey=True)
    fig.suptitle("左右判别分数的逐试次分布\n每个试次均为其所在记录五折交叉验证的折外预测",
                 fontsize=14, y=0.99)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=CORRECT_COLOR,
               markeredgecolor=CORRECT_COLOR, label="预测正确"),
        Line2D([0], [0], marker="x", linestyle="None", color=ERROR_COLOR,
               markersize=6, label="预测错误"),
        Line2D([0], [0], color="#333333", linestyle="--", label="左右判别阈值  0"),
    ]
    _boxed_legend(fig, handles, y=0.925, ncol=3)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.83,
                        wspace=0.15, hspace=0.30)

    axes_flat = axes.ravel()
    for ax, record in zip(axes_flat, RECORD_ORDER):
        name = RECORD_LABELS[record]
        local = [row for row in rows if row["记录"] == name]
        score = np.array([float(row["右向判别分数"]) for row in local])
        truth = np.array([row["真实方向"] for row in local])
        predicted = np.array([row["预测方向"] for row in local])
        trial_id = np.array([int(row["试次序号"]) for row in local])
        for class_x, condition in enumerate(("左", "右")):
            select = truth == condition
            jitter = rng.uniform(-0.12, 0.12, int(select.sum()))
            correct_mask = predicted[select] == truth[select]
            y = score[select]
            x = class_x + jitter
            ax.scatter(x[correct_mask], y[correct_mask], s=22,
                       facecolors=CORRECT_COLOR, edgecolors="white", linewidths=0.35,
                       alpha=0.85, zorder=3)
            ax.scatter(x[~correct_mask], y[~correct_mask], s=34, marker="x",
                       color=ERROR_COLOR, linewidths=1.15, alpha=0.95, zorder=4)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1], ["左三角", "右三角"])
        ax.set_title(f"{name}（平衡准确率 {summary[record]['balanced_accuracy']:.1%}）")
        ax.grid(axis="y", alpha=0.23)
        ax.set_axisbelow(True)
        ax.set_ylabel("右向判别分数")
    fig.savefig(OUTPUT_DIR / "02_逐试次判别分数.png", dpi=300, facecolor="white")
    plt.close(fig)


def _case_map(cases: list[dict]) -> dict[tuple[str, str], dict]:
    return {(case["dataset"], case["condition"]): case for case in cases
            if case["stage"] == "Stage1" and case["condition"] in ("left", "right")}


def _plot_erp(cases: list[dict], rows: list[dict[str, str]]) -> None:
    lookup = _case_map(cases)
    for record in RECORD_ORDER:
        raw = load_dataset(REAL_ROOT / f"{record}_clean.mat")
        local = [row for row in rows if row["记录"] == RECORD_LABELS[record]]
        raw_cue = np.asarray(raw["cue_type"], dtype=int)
        saved_indices = set()
        for row in local:
            trial_index = int(row["试次序号"]) - 1
            saved_indices.add(trial_index)
            expected = "左" if raw_cue[trial_index] == -1 else "右"
            if row["真实方向"] != expected:
                raise AssertionError(f"Trial label mismatch in {record}, trial {trial_index + 1}")
        expected_indices = set(np.flatnonzero(np.isin(raw_cue, (-1, 1))).tolist())
        if saved_indices != expected_indices:
            raise AssertionError(f"Saved predictions do not cover exactly the classified trials in {record}")
        counts = {condition: sum(row["真实方向"] == label for row in local)
                  for condition, label in (("left", "左"), ("right", "右"))}
        for condition, label in (("left", "左"), ("right", "右")):
            if counts[condition] != len(lookup[(record, condition)]["trials"]):
                raise AssertionError(f"Prediction trials do not cover the matching {record}/{label} EEG trials")

    fig, axes = plt.subplots(4, 3, figsize=(13.2, 9.0), sharex=True, sharey="col")
    fig.suptitle("同一批交叉验证试次的第一问清洗脑电\n按真实三角方向计算的事件相关电位",
                 fontsize=14, y=0.99)
    handles = [
        Line2D([0], [0], color=LEFT_COLOR, linewidth=1.7, label="左三角真实条件"),
        Line2D([0], [0], color=RIGHT_COLOR, linewidth=1.7, label="右三角真实条件"),
        Patch(facecolor="#A7B8C7", alpha=0.20, label="阴影：均值的95%近似置信带"),
        Line2D([0], [0], color="#555555", linestyle=":", label="提示出现  0 ms"),
        Line2D([0], [0], color="#555555", linestyle="--", label="提示消失  200 ms"),
    ]
    _boxed_legend(fig, handles, y=0.925, ncol=5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.84,
                        wspace=0.16, hspace=0.25)
    channels = ("F3", "Fz", "F4")

    for row_index, record in enumerate(RECORD_ORDER):
        for col_index, channel in enumerate(channels):
            ax = axes[row_index, col_index]
            for condition, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
                case = lookup[(record, condition)]
                data = np.asarray(case["trials"], dtype=float)
                time = np.asarray(case["time_ms"], dtype=float)
                channel_index = channels.index(channel)
                signal = data[:, channel_index, :]
                mean = signal.mean(axis=0)
                if len(signal) > 1:
                    sem = signal.std(axis=0, ddof=1) / np.sqrt(len(signal))
                else:
                    sem = np.zeros_like(mean)
                ci = 1.96 * sem
                label = "左三角" if condition == "left" else "右三角"
                ax.plot(time, mean, color=color, linewidth=1.35, label=label)
                ax.fill_between(time, mean - ci, mean + ci, color=color, alpha=0.15,
                                linewidth=0)
            ax.axvline(0, color="#555555", linestyle=":", linewidth=0.8)
            ax.axvline(200, color="#555555", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="#888888", linewidth=0.55)
            ax.grid(alpha=0.20)
            ax.set_xlim(-15, 800)
            if row_index == 0:
                ax.set_title(channel, fontsize=10, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(f"{RECORD_LABELS[record]}\n电位（原始数据单位）")
            if row_index == 3:
                ax.set_xlabel("提示出现后时间（ms）")
    fig.savefig(OUTPUT_DIR / "03_同批试次_真实脑电左右响应.png", dpi=300, facecolor="white")
    plt.close(fig)


def _bootstrap_diff(left: np.ndarray, right: np.ndarray, rng: np.random.Generator,
                    repeats: int = 2000) -> tuple[float, float, float]:
    estimate = float(right.mean() - left.mean())
    boot = np.empty(repeats, dtype=float)
    for index in range(repeats):
        li = rng.integers(0, len(left), len(left))
        ri = rng.integers(0, len(right), len(right))
        boot[index] = right[ri].mean() - left[li].mean()
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)


def _plot_feature_differences(cases: list[dict]) -> None:
    lookup = _case_map(cases)
    windows = tuple(FEATURE_WINDOWS_MS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.7), sharex=True, sharey=True)
    fig.suptitle("九维候选特征的右减左差异\n三类观测模式 × 三个固定时间窗",
                 fontsize=14, y=0.99)
    handles = [
        Line2D([0], [0], marker="o", color=MODE_COLORS[0], label="共同模式"),
        Line2D([0], [0], marker="o", color=MODE_COLORS[1], label="左右侧化模式"),
        Line2D([0], [0], marker="o", color=MODE_COLORS[2], label="中央－两侧模式"),
        Line2D([0], [0], color="#555555", linestyle="--", label="无条件差异  0"),
    ]
    _boxed_legend(fig, handles, y=0.925, ncol=4)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.83,
                        wspace=0.19, hspace=0.30)
    x = np.arange(len(windows), dtype=float)
    tick_labels = [f"{low:g}–{high:g}" for low, high in windows]

    for record_index, (ax, record) in enumerate(zip(axes.ravel(), RECORD_ORDER)):
        left_case, right_case = lookup[(record, "left")], lookup[(record, "right")]
        left = extract_features(left_case["trials"], left_case["time_ms"], mode_count=3)
        right = extract_features(right_case["trials"], right_case["time_ms"], mode_count=3)
        rng = np.random.default_rng(20260928 + record_index)
        for mode in range(3):
            estimates, lows, highs = [], [], []
            for window in range(3):
                index = mode * 3 + window
                estimate, low, high = _bootstrap_diff(left[:, index], right[:, index], rng)
                estimates.append(estimate)
                lows.append(estimate - low)
                highs.append(high - estimate)
            ax.errorbar(x, estimates, yerr=np.vstack([lows, highs]),
                        marker="o", markersize=4, capsize=2.5, linewidth=1.2,
                        color=MODE_COLORS[mode])
        ax.axhline(0, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_title(RECORD_LABELS[record])
        ax.set_xticks(x, tick_labels)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.set_xlabel("时间窗（ms）")
        ax.set_ylabel("右减左特征均值（原始数据单位）")
    fig.savefig(OUTPUT_DIR / "04_九维候选特征左右差异.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_predictions()
    keys = [(row["记录"], row["试次序号"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("A trial appears more than once in the selected out-of-fold predictions")
    cases, _ = load_cases_with_audit()
    summary = _summaries(rows)
    expected_n = sum(item["n"] for item in summary.values())
    if expected_n != len(rows):
        raise AssertionError("Record summary counts do not match the saved predictions")
    _plot_performance(rows, summary)
    _plot_trial_scores(rows, summary)
    _plot_erp(cases, rows)
    _plot_feature_differences(cases)
    print(f"已生成4张PNG：{OUTPUT_DIR}")
    for record in RECORD_ORDER:
        item = summary[record]
        print(f"{RECORD_LABELS[record]}: {item['correct']}/{item['n']}，"
              f"平衡准确率={item['balanced_accuracy']:.3%}")
    print(f"记录宏平均平衡准确率="
          f"{np.mean([item['balanced_accuracy'] for item in summary.values()]):.3%}")


if __name__ == "__main__":
    main()
