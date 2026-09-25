"""Exploratory in-sample search for interpretable local EEG features.

Searches a single-channel mean-amplitude threshold in short windows after the
Stage-1 cue onset. Feature, time window, polarity, and threshold are selected
on the same trials; all reported scores are therefore apparent training-fit
scores, not estimates of future-trial performance.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "math-model-agent" / "code"))
from algorithms.sci_figures import (  # noqa: E402
    FigureContract,
    export_publication_figure,
    paper_figure_rc_params,
    publication_size,
)

from revision_v3.config import CHANNELS
from revision_v3.real_data import load_dataset


Q2_ROOT = Path(__file__).resolve().parent
INPUT_DIR = Q2_ROOT.parent / "q1" / "output" / "8riemann_denoise"
OUTPUT_DIR = Q2_ROOT / "output" / "14_exploratory_local_window_decoder"
PAPER_FIGURE_DIR = Q2_ROOT / "output" / "revision_v3_paper_figures"
RECORDS = (
    ("Task1", "VisualCogA_Task-1"),
    ("Task1", "VisualCogA_Task-2"),
    ("Task2", "VisualCogB_Task-1"),
    ("Task2", "VisualCogB_Task-2"),
)
WINDOW_DURATIONS_MS = (25, 50, 75, 100, 150, 200)
SCAN_STEP_SAMPLES = 2
EPOCH = (0.0, 0.8)  # cue onset through 800 ms; excludes the 2.2-s target
COLORS = {"F3": "#2878B5", "Fz": "#279E68", "F4": "#D55E00"}

plt.rcParams.update(paper_figure_rc_params())
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


def best_threshold(x: np.ndarray, y: np.ndarray) -> dict:
    """Choose the best one-dimensional threshold by balanced accuracy."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    boundaries = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1, len(xs)]
    positive = (ys == 1).astype(int)
    negative = (ys == 0).astype(int)
    cum_pos = np.r_[0, np.cumsum(positive)][boundaries]
    cum_neg = np.r_[0, np.cumsum(negative)][boundaries]
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))

    # sign=+1 predicts right for values above the threshold; -1 reverses it.
    choices = []
    for j, k in enumerate(boundaries):
        choices.append((0.5 * ((n_pos - cum_pos[j]) / n_pos + cum_neg[j] / n_neg), 1, k))
        choices.append((0.5 * (cum_pos[j] / n_pos + (n_neg - cum_neg[j]) / n_neg), -1, k))
    balanced_accuracy, sign, cut = max(choices, key=lambda row: (row[0], row[1] == 1))
    if cut == 0:
        threshold = float(xs[0])
    elif cut == len(xs):
        threshold = float(xs[-1])
    else:
        threshold = float((xs[cut - 1] + xs[cut]) / 2.0)
    prediction = (sign * x > sign * threshold).astype(int)
    auc = float(roc_auc_score(y, x))
    return {
        "threshold": threshold,
        "polarity": int(sign),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(np.mean(prediction == y)),
        "auc_oriented": max(auc, 1.0 - auc),
        "prediction": prediction,
    }


def load_record(dataset: str) -> dict:
    data = load_dataset(INPUT_DIR / f"{dataset}_clean.mat")
    time = np.asarray(data["time_s"], dtype=float)
    eeg = np.asarray(data["eeg"], dtype=float)
    cue = np.asarray(data["cue_type"], dtype=int)
    baseline = (time >= -0.2) & (time < 0.0)
    epoch = (time >= EPOCH[0]) & (time < EPOCH[1])
    if baseline.sum() < 2 or epoch.sum() < 4 or not np.all(np.isin(cue, (-1, 1))):
        raise ValueError(f"{dataset}: invalid baseline, cue labels, or post-cue epoch")
    corrected = eeg - eeg[:, :, baseline].mean(axis=2, keepdims=True)
    # y=0 is left cue (-1); y=1 is right cue (+1).
    y = (cue == 1).astype(int)
    return {"dataset": dataset, "time": time, "eeg": corrected, "epoch": epoch, "y": y}


def find_best(record: dict) -> dict:
    time, eeg, y = record["time"], record["eeg"], record["y"]
    eligible = np.flatnonzero(record["epoch"])
    dt = float(np.median(np.diff(time)))
    rows = []
    for channel_index, channel in enumerate(CHANNELS):
        for duration_ms in WINDOW_DURATIONS_MS:
            n_samples = max(2, int(round(duration_ms / 1000.0 / dt)))
            last_start = eligible[-1] - n_samples + 1
            for start in range(int(eligible[0]), int(last_start) + 1, SCAN_STEP_SAMPLES):
                stop = start + n_samples
                x = eeg[:, channel_index, start:stop].mean(axis=1)
                fit = best_threshold(x, y)
                rows.append({
                    "dataset": record["dataset"],
                    "channel": channel,
                    "feature": "single-trial mean amplitude",
                    "start_ms": float(time[start] * 1000.0),
                    "end_exclusive_ms": float(time[stop] * 1000.0),
                    "last_sample_ms": float(time[stop - 1] * 1000.0),
                    "nominal_window_ms": duration_ms,
                    "window_samples": n_samples,
                    "threshold": fit["threshold"],
                    "polarity": fit["polarity"],
                    "n_left": int(np.sum(y == 0)),
                    "n_right": int(np.sum(y == 1)),
                    "accuracy_in_sample": fit["accuracy"],
                    "balanced_accuracy_in_sample": fit["balanced_accuracy"],
                    "auc_oriented_in_sample": fit["auc_oriented"],
                    "left_mean": float(x[y == 0].mean()),
                    "right_mean": float(x[y == 1].mean()),
                    "left_median": float(np.median(x[y == 0])),
                    "right_median": float(np.median(x[y == 1])),
                    "n_correct": int(np.sum(fit["prediction"] == y)),
                    "n_trials": int(len(y)),
                    "score_scope": "IN_SAMPLE_AFTER_SEARCH_NOT_HELD_OUT",
                    "feature_values": x,
                    "prediction": fit["prediction"],
                })
    # Highest balanced accuracy first, then ordinary accuracy, then the shorter window.
    rows.sort(key=lambda row: (row["balanced_accuracy_in_sample"],
                               row["accuracy_in_sample"],
                               -row["nominal_window_ms"]), reverse=True)
    return rows[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    rows = [{key: value for key, value in row.items()
             if key not in ("feature_values", "prediction")} for row in rows]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(records: dict, best_rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(
        4, 2, figsize=publication_size("double", height_mm=188.0),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )
    rng = np.random.default_rng(20260925)
    for row_index, (task, dataset) in enumerate(RECORDS):
        record = records[dataset]
        best = next(item for item in best_rows if item["dataset"] == dataset)
        time_ms = record["time"] * 1000.0
        onset = record["epoch"] & (time_ms < 800.0)
        y = record["y"]
        left = record["eeg"][y == 0]
        right = record["eeg"][y == 1]
        diff = right.mean(axis=0) - left.mean(axis=0)
        ax = axes[row_index, 0]
        for ci, channel in enumerate(CHANNELS):
            ax.plot(time_ms[onset], diff[ci, onset], color=COLORS[channel],
                    lw=1.4, label=f"{channel}: 右−左")
        ax.axhline(0, color="0.35", lw=0.8)
        ax.axvline(200, color="0.4", lw=0.9, ls=":", label="提示消失 200 ms")
        if best["channel"] in CHANNELS:
            ax.axvspan(best["start_ms"], best["end_exclusive_ms"],
                       color=COLORS[best["channel"]], alpha=0.16,
                       label=f"选中 {best['channel']} 窗口")
        ax.set_xlim(0, 800)
        ax.set_title(f"{dataset}：右−左 ERP")
        ax.set_ylabel("EEG 差值（输入单位）", fontsize=7)
        ax.grid(alpha=0.2)
        if row_index == 0:
            ax.legend(fontsize=6, ncol=2, frameon=False)

        ax = axes[row_index, 1]
        channel_index = CHANNELS.index(best["channel"])
        start_idx = int(np.argmin(np.abs(record["time"] * 1000.0 - best["start_ms"])))
        stop_idx = int(np.searchsorted(record["time"] * 1000.0,
                                        best["end_exclusive_ms"], side="left"))
        x = record["eeg"][:, channel_index, start_idx:stop_idx].mean(axis=1)
        for label, cls, color in (("左提示", 0, "#2878B5"), ("右提示", 1, "#D55E00")):
            values = x[y == cls]
            jitter = rng.normal(0, 0.035, size=len(values))
            ax.scatter(np.full(len(values), cls) + jitter, values, s=25,
                       alpha=0.72, color=color, edgecolor="white", linewidth=0.35,
                       label=f"{label} n={len(values)}")
            ax.plot([cls - 0.14, cls + 0.14], [np.median(values)] * 2,
                    color="#222222", lw=2)
        ax.axhline(best["threshold"], color="#6A3D9A", ls="--", lw=1.2,
                   label=f"阈值 {best['threshold']:.3g}")
        ax.set_xticks([0, 1], ["左提示", "右提示"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(f"{best['channel']} 窗口均值（输入单位）", fontsize=7)
        ax.set_title(
            f"{best['channel']} {best['start_ms']:.0f}–{best['last_sample_ms']:.0f} ms；"
            f"BA={best['balanced_accuracy_in_sample']:.1%}，"
            f"准确率={best['accuracy_in_sample']:.1%}", fontsize=7.2
        )
        ax.grid(axis="y", alpha=0.2)
        if row_index == 0:
            ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=6)
    axes[-1, 0].set_xlabel("相对提示出现时间（ms）", fontsize=7)
    axes[-1, 1].set_xlabel("提示方向", fontsize=7)
    fig.suptitle(
        "左右提示的局部 EEG 特征：训练内探索性拟合（非独立测试）",
        y=0.995, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.9, w_pad=1.3)
    fig.savefig(path, dpi=450, facecolor="white")
    contract = FigureContract(
        claim=("在四份记录中，按同一批 Stage1 试次搜索得到的单电极短窗平均振幅阈值，"
               "训练内准确率为 61.3%–72.2%；最佳电极与时间窗因记录而异。"),
        evidence=("四记录 0–800 ms 的右减左 ERP 差波形及选中窗口",
                  "选中电极的单试次窗口均值分布、阈值和训练内分类分数"),
        source_paths=(
            "src/C/q2/output/14_exploratory_local_window_decoder/局部时间窗最佳训练内结果.csv",
            "src/C/q1/output/8riemann_denoise/*_clean.mat",
        ),
        target_venue="CMathc question 2 paper draft",
        column="double",
        figure_role="model-result",
        model_name="single-channel local-window threshold classifier",
        scenario="Stage1 cue-locked EEG; per-record exploratory fit on all trials",
        parameter_source="per-record channel, window, polarity, and threshold selected on all trials",
        randomness="window scan deterministic; point jitter uses seed 20260925",
        n_definition="four MAT records; 54, 80, 83, and 80 trials respectively",
        statistic="training-set apparent accuracy and balanced accuracy",
        uncertainty="no held-out uncertainty interval; feature selection and scoring use the same trials",
        review_risks=(
            "Training scores are optimistic and are not independent test performance.",
            "Best electrode and window are not shared across records.",
            "EEG amplitudes retain source-data units and are not asserted to be microvolts.",
        ),
    )
    export_publication_figure(
        fig, PAPER_FIGURE_DIR / "左右提示局部特征_探索性拟合", contract,
        dpi=450, strict=True, close=True,
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = {}
    best_rows = []
    all_rows = []
    for task, dataset in RECORDS:
        record = load_record(dataset)
        records[dataset] = record
        best = find_best(record)
        best["task"] = task
        best_rows.append(best)
        print(f"{dataset}: {best['channel']} {best['start_ms']:.1f}–"
              f"{best['last_sample_ms']:.1f} ms; "
              f"BA={best['balanced_accuracy_in_sample']:.3f}; "
              f"ACC={best['accuracy_in_sample']:.3f}; "
              f"correct={best['n_correct']}/{best['n_trials']}", flush=True)
    write_csv(OUTPUT_DIR / "局部时间窗最佳训练内结果.csv", best_rows)
    make_figure(records, best_rows, OUTPUT_DIR / "局部时间窗与特征分布.png")
    lines = [
        "# 左右提示局部时间窗的探索性拟合",
        "",
        "> 这是在同一批试次上搜索电极、时间窗、阈值并计算得分的训练内结果，存在选择偏差，不能称为测试准确率或泛化性能。",
        "",
        "## 分析口径",
        "",
        "- 输入：第一问 `8riemann_denoise` 中的四份 `*_clean.mat`，通道按 F3、Fz、F4 排列。",
        "- 标签：提示标记 `cue_type`，-1 为左提示，+1 为右提示；不使用按键/行为反应标签。",
        "- 基线：每试次减去提示前 [-200, 0) ms 通道均值。",
        "- 搜索范围：提示出现后 [0, 800) ms；不包含 2.2 s 的目标出现。",
        "- 候选特征：单电极 25、50、75、100、150、200 ms 窗口均值；窗口每两个采样点移动一次。",
        "- 模型：一个特征加一个阈值；阈值方向与位置按训练内平衡准确率最大化选取。",
        "- 每份记录分别选最佳组合；四份记录的窗口/电极不强制相同。",
        "",
        "## 结果解释",
        "",
        "图左列显示四份记录在提示后 0–800 ms 的右减左 ERP 差波形，并标出该记录训练内选中的时间窗；右列显示被选特征的单试次分布与分类阈值。",
        "这些结果用于定位可能有用的局部特征。由于窗口、电极和阈值都在全体试次上挑选，分数会偏乐观；记录间候选窗口不同也表示目前尚未证明存在统一的时间-电极机制。",
        "",
        "## 重现",
        "",
        "在 `src/C/q2` 下运行 `python 14_exploratory_local_window_decoder.py`。",
    ]
    (OUTPUT_DIR / "结果说明.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
