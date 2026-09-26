"""Re-run and compare the current 9-D baseline with the historical ERP-template readout.

The historical readout is reproduced with its original all-trial window choice,
and a training-fold window-selection version is included to show the effect of
that choice. All methods use the current Q1-cleaned MAT inputs. The residual
template run consumes the latest revision_v3 model curves; its algebraic
cancellation audit is reported explicitly.
"""
from __future__ import annotations

import csv
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Q2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Q2_ROOT.parents[2]
sys.path.insert(0, str(Q2_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "math-model-agent" / "code"))

from revision_v3.config import OUTPUT_ROOT, require_current_source_mapping_manifest  # noqa: E402
from revision_v3.evidence_chain import classify_electrode_features  # noqa: E402
from revision_v3.real_data import load_cases_with_audit  # noqa: E402


OUTPUT_DIR = Q2_ROOT / "output" / "revision_v3_method_comparison"
HISTORICAL_DIR = OUTPUT_ROOT / "17_residual_template_match"
TRAIN_ONLY_DIR = Q2_ROOT / "output" / "15_record_specific_template_decoder"
CHANNEL_NAMES = {
    "VisualCogA_Task-1": "A组记录1",
    "VisualCogA_Task-2": "A组记录2",
    "VisualCogB_Task-1": "B组记录1",
    "VisualCogB_Task-2": "B组记录2",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_current_decoders() -> tuple[list[dict], list[dict]]:
    """Re-run both within-record template procedures from the current clean MATs."""
    runpy.run_path(str(Q2_ROOT / "15_record_specific_template_decoder.py"),
                   run_name="__main__")
    runpy.run_path(str(Q2_ROOT / "17_residual_template_match.py"),
                   run_name="__main__")
    historical = _read_csv(HISTORICAL_DIR / "summary.csv")
    training_only = _read_csv(TRAIN_ONLY_DIR / "逐记录准确率.csv")
    return historical, training_only


def _baseline() -> tuple[dict, list[dict], list[dict]]:
    cases, _ = load_cases_with_audit()
    score_rows, folds, summary = classify_electrode_features(cases)
    if summary["feature_count"] != 9 or summary["heldout_record_count"] != 4:
        raise RuntimeError("The current baseline must be the fixed 9-feature, 4-record holdout.")
    return summary, folds, score_rows


def _summaries(historical: list[dict], training_only: list[dict], baseline: dict,
               score_rows: list[dict]) -> tuple[list[dict], list[dict], float]:
    legacy_total = next(row for row in historical if row["record"] == "pooled_trials")
    foldwise_total = next(row for row in training_only if row["record"] == "pooled_trials"
                          and row["method"] == "local_window")
    baseline_rows = [row for row in score_rows]
    baseline_correct = sum(bool(row["correct"]) for row in baseline_rows)
    baseline_accuracy = baseline_correct / len(baseline_rows)

    cancellation_rows = _read_csv(HISTORICAL_DIR / "trial_scores.csv")
    cancellation_error = max(float(row["algebraic_cancellation_max_abs_error"])
                             for row in cancellation_rows)

    summaries = [
        {
            "方法": "当前九维固定特征",
            "验证方式": "留一记录",
            "试次数": int(baseline["trial_count"]),
            "正确数": baseline_correct,
            "合并准确率": baseline_accuracy,
            "记录宏平均平衡准确率": float(baseline["macro_balanced_accuracy"]),
            "记录宏平均曲线下面积": float(baseline["macro_roc_auc"]),
            "窗口选择": "预先固定",
            "说明": "标准化参数只由训练记录估计",
        },
        {
            "方法": "历史局部时窗ERP模板法",
            "验证方式": "五折时间块",
            "试次数": int(legacy_total["n_trials"]),
            "正确数": int(legacy_total["correct"]),
            "合并准确率": float(legacy_total["accuracy"]),
            "记录宏平均平衡准确率": float(legacy_total["balanced_accuracy"]),
            "记录宏平均曲线下面积": float("nan"),
            "窗口选择": "全试次选窗",
            "说明": "含选窗信息泄漏；模型残差项相消，等价于实测ERP模板匹配",
        },
        {
            "方法": "局部时窗ERP模板法（折内选窗）",
            "验证方式": "五折时间块",
            "试次数": int(foldwise_total["n_trials"]),
            "正确数": int(foldwise_total["correct"]),
            "合并准确率": float(foldwise_total["accuracy"]),
            "记录宏平均平衡准确率": float(foldwise_total["macro_record_balanced_accuracy"]),
            "记录宏平均曲线下面积": float("nan"),
            "窗口选择": "仅训练折选窗",
            "说明": "电极和时间窗在每折训练数据内选择",
        },
    ]

    # The baseline folds and both template summaries are saved separately below.
    record_rows = []
    for row in historical:
        if row["record"] == "pooled_trials":
            continue
        record_rows.append({
            "方法": "历史局部时窗ERP模板法",
            "记录": CHANNEL_NAMES.get(row["record"], row["record"]),
            "试次数": int(row["n_trials"]),
            "正确数": int(row["correct"]),
            "准确率": float(row["accuracy"]),
            "平衡准确率": float(row["balanced_accuracy"]),
        })
    for row in training_only:
        if row["record"] == "pooled_trials" or row["method"] != "local_window":
            continue
        record_rows.append({
            "方法": "局部时窗ERP模板法（折内选窗）",
            "记录": CHANNEL_NAMES.get(row["record"], row["record"]),
            "试次数": int(row["n_trials"]),
            "正确数": int(row["correct"]),
            "准确率": float(row["accuracy"]),
            "平衡准确率": float(row["balanced_accuracy"]),
        })
    return summaries, record_rows, cancellation_error


def _plot(summaries: list[dict], output_path: Path, cancellation_error: float) -> None:
    values = [row["记录宏平均平衡准确率"] * 100 for row in summaries]
    labels = [
        "当前九维固定时窗特征\n留一记录",
        "历史局部时窗 ERP 模板法\n全试次选窗",
        "局部时窗 ERP 模板法\n训练折内选窗",
    ]
    colors = ["#3977A8", "#D28A35", "#4C956C"]
    fig, ax = plt.subplots(figsize=(10.4, 6.4), constrained_layout=False)
    bars = ax.barh(np.arange(3), values, color=colors, edgecolor="white", height=.58)
    ax.axvline(50, color="#444444", linestyle="--", linewidth=1.25,
               label="随机参考水平（50%）")
    ax.set_yticks(np.arange(3), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 82)
    ax.set_xlabel("记录宏平均平衡准确率（%）", fontsize=11)
    ax.set_title("当前九维基线与历史局部 ERP 模板法复跑", fontsize=15, pad=15)
    ax.grid(axis="x", alpha=.20, linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=10, length=0, pad=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B7B7B7")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    for index, (bar, row, value) in enumerate(zip(bars, summaries, values)):
        extra = ""
        if index == 0:
            extra = f"曲线下面积 {row['记录宏平均曲线下面积']:.3f}"
        else:
            extra = f"合并准确率 {row['合并准确率'] * 100:.1f}%"
        ax.text(value - 1.0, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%\n{extra}", va="center", ha="right", fontsize=9.0,
                color="white", fontweight="bold", linespacing=1.25)

    note = (
        "历史法的电极和时窗由全试次挑选，测试块参与了选窗；折内选窗版只用训练块选择。"
        "两类验证划分不同，不能据柱高直接排名。\n"
        f"历史法使用当前版本模型曲线写成残差距离；模型项相消，最大数值误差 {cancellation_error:.1e}，"
        "判别等价于实测 ERP 模板匹配。"
    )
    fig.text(.12, .035, note, ha="left", va="bottom", fontsize=8.1,
             color="#444444", linespacing=1.45)
    fig.subplots_adjust(left=.27, right=.97, top=.88, bottom=.21)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def _write_report(path: Path, summaries: list[dict], record_rows: list[dict],
                  folds: list[dict], cancellation_error: float) -> None:
    base, legacy, nested = summaries
    lines = [
        "# 当前九维基线与历史约55%方法复跑",
        "",
        "## 复跑结果",
        "",
        "![三种左右判别流程的结果对照](方法对照.png)",
        "",
        "![历史局部时窗方法的逐记录窗口与表现](../revision_v3/17_residual_template_match/record_specific_decoding.png)",
        "",
        "| 方法 | 留出方式 | 合并准确率 | 记录宏平均平衡准确率 | 曲线下面积 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        auc = "—" if not np.isfinite(row["记录宏平均曲线下面积"]) else f"{row['记录宏平均曲线下面积']:.3f}"
        lines.append(
            f"| {row['方法']} | {row['验证方式']} | {row['正确数']}/{row['试次数']}（{row['合并准确率']:.1%}） | "
            f"{row['记录宏平均平衡准确率']:.1%} | {auc} |"
        )
    lines.extend([
        "",
        "## 三种口径",
        "",
        f"- **当前基线**：九个预设特征为 F3、Fz、F4 在 `[100,250)`、`[250,500)`、`[500,800)` ms 的 ERP 均值；整份记录留出。重算结果为宏平均平衡准确率 **{base['记录宏平均平衡准确率']:.1%}**、曲线下面积 **{base['记录宏平均曲线下面积']:.3f}**。",
        f"- **历史约55%方法**：沿用历史全试次挑出的逐记录电极和时间窗，用五折时间块留出试次，并在训练折拟合左右 ERP 模板。当前清洗数据上复现为 **{legacy['正确数']}/{legacy['试次数']}（{legacy['合并准确率']:.1%}）**，记录宏平均平衡准确率 **{legacy['记录宏平均平衡准确率']:.1%}**。因为电极和时窗曾使用全体试次标签挑选，留出块信息参与了特征选择，这个分数是探索性结果。",
        f"- **折内选窗核验**：同一局部时窗模板思路，但每折只用训练块选择电极、时间窗和模板；复跑为 **{nested['正确数']}/{nested['试次数']}（{nested['合并准确率']:.1%}）**，记录宏平均平衡准确率 **{nested['记录宏平均平衡准确率']:.1%}**。它仍是同记录的时间块验证，不等价于留一记录验证。",
        "",
        "## 对当前框架的解释",
        "",
        f"历史法读取了当前 v3 的留出模型曲线并以残差形式计算距离。但对同一候选类别，`(试次−模型曲线)−(训练 ERP−模型曲线)` 中的模型曲线会代数相消；本次数值核查的最大残差为 `{cancellation_error:.3e}`。因此历史法的左右标签实际由**真实 EEG 与训练 ERP 模板的距离**决定，不是模型曲线单独完成的预测。",
        "",
        "当前较可信的对照是：固定九维特征在留一记录验证中为 46.6%；同记录、训练折内选窗的局部模板法宏平均平衡准确率为 47.9%，仍接近随机水平。历史 55.1% 不能作为独立验证优势，主要需要考虑全试次选窗带来的乐观偏差，以及它使用了不同的留出层级。",
        "",
        "## 分记录结果",
        "",
        "| 方法 | 记录 | 试次数 | 正确数 | 准确率 | 平衡准确率 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in record_rows:
        lines.append(f"| {row['方法']} | {row['记录']} | {row['试次数']} | {row['正确数']} | "
                     f"{row['准确率']:.1%} | {row['平衡准确率']:.1%} |")
    lines.extend([
        "",
        "### 当前九维基线的留出记录",
        "",
        "| 留出记录 | 试次数 | 平衡准确率 | 曲线下面积 |",
        "|---|---:|---:|---:|",
    ])
    for row in folds:
        if row.get("status") == "complete":
            lines.append(f"| {CHANNEL_NAMES.get(row['heldout_record'], row['heldout_record'])} | "
                         f"{row['n_test_trials']} | {row['balanced_accuracy']:.1%} | {row['roc_auc']:.3f} |")
    lines.extend([
        "",
        "## 复现输入与输出",
        "",
        "- 三种流程均读取第一问 `8riemann_denoise` 的 `*_clean.mat`，使用提示方向标签和提示前基线校正。",
        "- 历史残差模板法使用当前 `output/revision_v3/heldout_predictions.csv` 与历史探索窗口文件；模型输出通过残差差分参与计算，并已核验其代数抵消。",
        "- 新生成的图只有 PNG：`方法对照.png`。机器可读汇总为 `方法汇总.csv` 和 `分记录结果.csv`。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = require_current_source_mapping_manifest()
    if not (OUTPUT_ROOT / "heldout_predictions.csv").exists():
        raise FileNotFoundError("Current revision_v3 model curves are required before residual-template rerun.")

    print("Recomputing current fixed-feature leave-one-record baseline...", flush=True)
    baseline, folds, score_rows = _baseline()
    print("Re-running the historical and training-fold local ERP-template methods...", flush=True)
    historical, training_only = _run_current_decoders()
    summaries, record_rows, cancellation_error = _summaries(
        historical, training_only,
        {**baseline, "folds": folds}, score_rows)

    _write_csv(OUTPUT_DIR / "方法汇总.csv", summaries)
    _write_csv(OUTPUT_DIR / "分记录结果.csv", record_rows)
    _write_csv(OUTPUT_DIR / "基线留出记录.csv", [
        {"记录": CHANNEL_NAMES.get(row["heldout_record"], row["heldout_record"]),
         "试次数": row["n_test_trials"],
         "平衡准确率": row["balanced_accuracy"],
         "曲线下面积": row["roc_auc"],
         "训练记录": row["training_records"]}
        for row in folds if row.get("status") == "complete"
    ])
    _plot(summaries, OUTPUT_DIR / "方法对照.png", cancellation_error)
    _write_report(OUTPUT_DIR / "结果说明.md", summaries, record_rows, folds,
                  cancellation_error)

    print("\n=== Current comparison ===", flush=True)
    for row in summaries:
        print(f"{row['方法']}: BA={row['记录宏平均平衡准确率']:.4f}; "
              f"ACC={row['合并准确率']:.4f}; trials={row['正确数']}/{row['试次数']}",
              flush=True)
    print(f"Residual/model cancellation max error: {cancellation_error:.3e}", flush=True)
    print(f"Current mapping schema: {manifest['source_mapping_schema']}", flush=True)
    print(f"Saved comparison to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
