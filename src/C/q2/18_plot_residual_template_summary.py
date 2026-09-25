"""Plot the record-specific residual-template decoding summary for paper use."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "math-model-agent" / "code"))

from algorithms.sci_figures import (  # noqa: E402
    FigureContract,
    export_publication_figure,
    paper_figure_rc_params,
    publication_size,
)

Q2_ROOT = Path(__file__).resolve().parent
RESULT_DIR = Q2_ROOT / "output" / "revision_v3" / "17_residual_template_match"
SUMMARY_CSV = RESULT_DIR / "summary.csv"
FIGURE_BASE = RESULT_DIR / "record_specific_decoding"

RECORDS = (
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
)
CHANNEL_COLORS = {"F3": "#2878B5", "Fz": "#279E68", "F4": "#D55E00"}
ACCURACY_COLOR = "#2878B5"
BALANCED_COLOR = "#D55E00"

plt.rcParams.update(paper_figure_rc_params())
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


def load_rows() -> tuple[list[dict], dict]:
    with SUMMARY_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    by_record = {row["record"]: row for row in rows}
    detail = [by_record[name] for name in RECORDS]
    pooled = by_record["pooled_trials"]
    return detail, pooled


def draw() -> None:
    rows, pooled = load_rows()
    macro_ba = sum(float(row["balanced_accuracy"]) for row in rows) / len(rows)

    fig, (ax_time, ax_score) = plt.subplots(
        1, 2, figsize=publication_size("double", 88),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.23, wspace=0.34)

    y_positions = list(range(len(rows) - 1, -1, -1))
    labels = [name.replace("VisualCog", "").replace("_Task-", " · Task-")
              for name in RECORDS]

    # Panel A: where each record-specific feature window lies on the cue timeline.
    for y, row in zip(y_positions, rows):
        start = float(row["region_start_ms"])
        end = float(row["region_end_exclusive_ms"])
        channel = row["region_channel"]
        points = int(float(row["region_valid_points"]))
        color = CHANNEL_COLORS.get(channel, "#666666")
        ax_time.broken_barh(
            [(start, end - start)], (y - 0.17, 0.34),
            facecolors=color, edgecolors="white", linewidth=0.7,
        )
        ax_time.text(
            810, y,
            f"{start:.0f}–{end:.0f} ms · {channel} · {points}点",
            ha="left", va="center", fontsize=6.2, color="#303030",
            clip_on=False,
        )

    ax_time.axvline(0, color="#555555", lw=0.9, ls="--")
    ax_time.axvline(200, color="#777777", lw=0.9, ls=":")
    ax_time.text(5, 3.43, "提示出现", fontsize=6.2, color="#555555", va="bottom")
    ax_time.text(205, 3.43, "提示消失", fontsize=6.2, color="#666666", va="bottom")
    ax_time.set_xlim(-12, 1095)
    ax_time.set_xticks([0, 200, 400, 600, 800])
    ax_time.set_ylim(-0.52, 3.7)
    ax_time.set_yticks(y_positions, labels, fontsize=6.1)
    ax_time.set_xlabel("相对提示出现时间（ms）")
    ax_time.set_title("(a) 记录特异的候选时间窗", loc="left", fontsize=8.2, pad=7)
    ax_time.grid(axis="x", alpha=0.18, lw=0.5)
    ax_time.spines[["top", "right"]].set_visible(False)
    ax_time.tick_params(axis="both", labelsize=6.2, length=2)

    # Panel B: per-record accuracy and balanced accuracy, with chance reference.
    for y, row in zip(y_positions, rows):
        acc = float(row["accuracy"])
        ba = float(row["balanced_accuracy"])
        ax_score.scatter(
            acc, y + 0.095, marker="o", s=25, color=ACCURACY_COLOR,
            edgecolor="white", linewidth=0.45, zorder=3,
            label="准确率" if y == y_positions[0] else None,
        )
        ax_score.scatter(
            ba, y - 0.095, marker="s", s=25, facecolor="white",
            edgecolor=BALANCED_COLOR, linewidth=1.15, zorder=3,
            label="平衡准确率" if y == y_positions[0] else None,
        )
        ax_score.text(acc + 0.025, y + 0.095, f"{acc:.1%}",
                      va="center", fontsize=6.1, color=ACCURACY_COLOR)
        ax_score.text(ba + 0.025, y - 0.095, f"{ba:.1%}",
                      va="center", fontsize=6.1, color=BALANCED_COLOR)

    ax_score.axvline(0.5, color="#555555", lw=0.9, ls="--", label="50%参考线")
    ax_score.set_xlim(0, 1.0)
    ax_score.set_ylim(-0.52, 3.52)
    ax_score.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_score.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_score.set_yticks(y_positions, labels, fontsize=6.1)
    ax_score.set_xlabel("判别正确率")
    ax_score.set_title("(b) 留出时间块的判别表现", loc="left", fontsize=8.2, pad=7)
    ax_score.grid(axis="x", alpha=0.18, lw=0.5)
    ax_score.spines[["top", "right"]].set_visible(False)
    ax_score.tick_params(axis="both", labelsize=6.2, length=2)
    ax_score.legend(loc="upper left", fontsize=5.8, frameon=False, handletextpad=0.35)

    n_trials = int(pooled["n_trials"])
    correct = int(pooled["correct"])
    overall_acc = float(pooled["accuracy"])
    fig.text(
        0.56, 0.09,
        f"合并准确率：{overall_acc:.1%}（{correct}/{n_trials}）"
        f"     四记录平衡准确率宏平均：{macro_ba:.1%}",
        ha="center", va="center", fontsize=7.4, fontweight="bold",
    )
    fig.text(
        0.56, 0.035,
        "窗口按记录分别选取；时间窗选择来自全试次探索，分数为固定候选窗下的时间块交叉验证。",
        ha="center", va="center", fontsize=6.1, color="#555555",
    )

    contract = FigureContract(
        claim=(
            "记录特异的局部 ERP 模板读出在四份记录合并后正确分类 "
            f"{correct}/{n_trials} 个试次（{overall_acc:.1%}）；记录间表现不一。"
        ),
        evidence=(
            "四份记录各自候选电极及时间窗",
            "各记录留出时间块上的准确率与平衡准确率",
            "合并试次准确率与四记录平衡准确率宏平均",
        ),
        source_paths=(
            "src/C/q2/output/revision_v3/17_residual_template_match/summary.csv",
            "src/C/q2/output/14_exploratory_local_window_decoder/局部时间窗最佳训练内结果.csv",
        ),
        target_venue="CMathc question 2 paper",
        column="double",
        figure_role="model-result",
        model_name="record-specific local ERP template matcher",
        scenario=(
            "Stage1 cue EEG; each record uses its candidate channel/window; "
            "left/right ERP templates estimated from training time blocks"
        ),
        parameter_source=(
            "Candidate windows from all-trial exploratory search; "
            "ERP templates fitted within each training fold"
        ),
        randomness="No random numbers; chronological five-fold test blocks.",
        n_definition="Four MAT records; 54, 80, 83, and 80 trials; 297 trials total.",
        statistic="Pooled accuracy and per-record accuracy/balanced accuracy.",
        uncertainty=(
            "No confidence intervals shown. Candidate channel/window selection "
            "used the all-trial exploratory scan."
        ),
        review_risks=(
            "The four records use different candidate channels and time windows.",
            "The 54.2% pooled accuracy is not a cross-record generalization estimate.",
            "Residual matching is algebraically equivalent to empirical ERP template matching.",
        ),
    )
    export_publication_figure(
        fig, FIGURE_BASE, contract, dpi=450, strict=True, close=True
    )
    print(f"Saved figure files with base: {FIGURE_BASE}")


if __name__ == "__main__":
    draw()
