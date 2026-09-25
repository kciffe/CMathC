# -*- coding: utf-8 -*-
"""Check whether the fixed prestimulus ERP baseline changes split-half results."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from revision.real_data import load_dataset

INPUT_DIR = ROOT.parent / "q1" / "output" / "8riemann_denoise"
OUTPUT_DIR = ROOT / "output" / "12_split_half_erp_reliability"
RECORDS = ("VisualCogA_Task-1", "VisualCogA_Task-2", "VisualCogB_Task-1", "VisualCogB_Task-2")
WINDOWS = {"onset_0_200": (0.0, 200.0), "offset_200_800": (200.0, 800.0),
           "whole_0_800": (0.0, 800.0), "whole_cue_0_2200": (0.0, 2200.0)}


def _corr(x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    rows = []
    baseline_rows = []
    for record in RECORDS:
        data = load_dataset(INPUT_DIR / f"{record}_clean.mat")
        eeg = np.asarray(data["eeg"], float)
        time_ms = np.asarray(data["time_s"], float) * 1000
        labels = np.asarray(data["cue_type"], int)
        pre = (time_ms >= -200) & (time_ms < 0)
        base_mean = eeg[:, :, pre].mean(axis=2)
        centered = eeg - base_mean[:, :, None]
        split = len(labels) // 2
        blocks = (np.arange(split), np.arange(split, len(labels)))
        baseline_lr = []
        for idx in blocks:
            l = base_mean[idx[labels[idx] == -1]].mean(axis=0)
            r = base_mean[idx[labels[idx] == 1]].mean(axis=0)
            baseline_lr.append(r - l)
        baseline_rows.append({
            "record": record,
            "pre_cue_left_right_baseline_diff_rms_first": float(np.sqrt(np.mean(baseline_lr[0]**2))),
            "pre_cue_left_right_baseline_diff_rms_second": float(np.sqrt(np.mean(baseline_lr[1]**2))),
            "pre_cue_baseline_lr_difference_cosine": float(
                np.dot(baseline_lr[0], baseline_lr[1]) /
                max(np.linalg.norm(baseline_lr[0]) * np.linalg.norm(baseline_lr[1]), 1e-12)),
            "pre_cue_common_baseline_rms_first": float(np.sqrt(np.mean(base_mean[:split].mean(axis=0)**2))),
            "pre_cue_common_baseline_rms_second": float(np.sqrt(np.mean(base_mean[split:].mean(axis=0)**2))),
            "pre_cue_common_block_change_rms": float(np.sqrt(np.mean(
                (base_mean[split:].mean(axis=0) - base_mean[:split].mean(axis=0))**2))),
        })
        for win, (lo, hi) in WINDOWS.items():
            mask = (time_ms >= lo) & (time_ms < hi)
            for baseline_name, signal in (("uncorrected", eeg), ("per_trial_-200_0_corrected", centered)):
                condition_means = {}
                for b, idx in enumerate(blocks):
                    for side, code in (("left", -1), ("right", 1)):
                        selected = idx[labels[idx] == code]
                        condition_means[(b, side)] = signal[selected][:, :, mask].mean(axis=0)
                for kind in ("common", "left_right_difference"):
                    def combine(b):
                        l, r = condition_means[(b, "left")], condition_means[(b, "right")]
                        return (l + r) / 2 if kind == "common" else (r - l)
                    first, second = combine(0), combine(1)
                    rms_second = float(np.sqrt(np.mean(second**2)))
                    rows.append({
                        "record": record, "window": win, "signal": kind, "baseline_mode": baseline_name,
                        "cross_block_corr": _corr(first, second),
                        "block1_predicts_block2_nrmse": (float(np.sqrt(np.mean((first-second)**2))/rms_second)
                                                          if rms_second > 1e-12 else float("nan")),
                        "block1_rms": float(np.sqrt(np.mean(first**2))),
                        "block2_rms": rms_second,
                    })
    out = OUTPUT_DIR / "基线敏感性对照.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    bout = OUTPUT_DIR / "刺激前条件基线差.csv"
    with bout.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_rows[0]))
        w.writeheader(); w.writerows(baseline_rows)
    print("record,window,difference_corr_uncorrected,difference_corr_corrected,delta")
    for win in WINDOWS:
        for record in RECORDS:
            match = [r for r in rows if r["record"] == record and r["window"] == win
                     and r["signal"] == "left_right_difference"]
            raw = next(r for r in match if r["baseline_mode"] == "uncorrected")
            corrected = next(r for r in match if r["baseline_mode"] == "per_trial_-200_0_corrected")
            print(f"{record},{win},{raw['cross_block_corr']:.3f},{corrected['cross_block_corr']:.3f},"
                  f"{corrected['cross_block_corr']-raw['cross_block_corr']:+.3f}")
    print("pre-cue baseline contrast:")
    for r in baseline_rows:
        print(r)

    def find(record, window, kind, baseline_mode):
        return next(r for r in rows if r["record"] == record and r["window"] == window
                    and r["signal"] == kind and r["baseline_mode"] == baseline_mode)

    report = [
        "# 基线校正敏感性检查", "",
        "## 代码口径", "",
        "留出的 Q1 `*_clean.mat` 保存的是清洗后的原始试次，Q1 清洗脚本没有在写文件时做 ERP 基线相减。本项目 ERP 分析中的基线窗口是每试次、每通道的刺激前 `[-200,0) ms`；`0–800 ms` 是观察响应的时间窗，不是被用来求基线的区间。校正公式为 `x_corr(t)=x(t)-mean(x[-200,0))`，对 0 ms 之后所有点减去同一个常数。", "",
        "## 左右差值跨块相关：校正前后", "",
        "| 记录 | 0–800 ms 未校正 | 0–800 ms 校正 | 200–800 ms 未校正 | 200–800 ms 校正 | 0–2200 ms 未校正 | 0–2200 ms 校正 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in RECORDS:
        vals = [find(record, w, "left_right_difference", mode)["cross_block_corr"]
                for w in ("whole_0_800", "offset_200_800", "whole_cue_0_2200")
                for mode in ("uncorrected", "per_trial_-200_0_corrected")]
        report.append("| " + record + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    report += ["", "## 0–800 ms 共同 ERP 跨块相关：校正前后", "",
               "| 记录 | 未校正 | 校正 | 变化 |", "|---|---:|---:|---:|"]
    for record in RECORDS:
        raw = find(record, "whole_0_800", "common", "uncorrected")["cross_block_corr"]
        corr = find(record, "whole_0_800", "common", "per_trial_-200_0_corrected")["cross_block_corr"]
        report.append(f"| {record} | {raw:.3f} | {corr:.3f} | {corr-raw:+.3f} |")
    report += ["", "## 刺激前基线本身的条件与时间块差异", "",
               "| 记录 | 左右基线差 RMS 前/后块 | 左右基线差方向一致性（余弦） | 共同基线 RMS 前/后块 | 共同基线跨块变化 RMS |",
               "|---|---:|---:|---:|---:|"]
    for r in baseline_rows:
        report.append(f"| {r['record']} | {r['pre_cue_left_right_baseline_diff_rms_first']:.3g} / "
                      f"{r['pre_cue_left_right_baseline_diff_rms_second']:.3g} | "
                      f"{r['pre_cue_baseline_lr_difference_cosine']:.3f} | "
                      f"{r['pre_cue_common_baseline_rms_first']:.3g} / "
                      f"{r['pre_cue_common_baseline_rms_second']:.3g} | "
                      f"{r['pre_cue_common_block_change_rms']:.3g} |")
    report += ["", "## 结论", "",
               "这组结果不支持‘刺激前基线相减把 0–800 ms 的左右共性波形普遍抹掉’。在 0–800 ms，左右差值的跨块相关经校正后四份记录都上升或变得较少负；共同 ERP 的变化则不一致（VisualCogB_Task-1 明显下降，其余三份变化较小或上升）。因此基线处理会改变个别记录的相关和幅值，但没有一致地削弱所有记录中的共同或左右波形。", "",
               "未校正数据的刺激前均值在左右条件间已出现明显差别，而且该差别跨块方向并不总是一致；共同基线均值也有跨块改变。这些都是提示出现之前已有的偏移/漂移，未校正 ERP 的相关可能被这些非 cue 变化成分影响。减去刺激前基线是为了把响应解释为相对提示前状态的变化；它会移除每试次的常数偏置，不会删除提示后随时间变化的波形。", "",
               "但常规均值基线并非没有代价：若刺激前窗口含有条件相关的预期活动、前一事件残留，或基线趋势继续进入响应窗，减去一个常数可能改变低频/持续成分。因而本项目不能只靠‘有/无基线’二选一作最终生理结论；可以把本对照作为敏感性分析，并把按 trial baseline 作为主 ERP 口径。文献也提醒传统基线依赖刺激前窗口代表性等假设，必要时可用把基线作为协变量的回归 ERP 方案作补充。", "",
               "本检查仅改变 ERP 展示/稳定性计算的基线处理，不重新训练模型，也不把未校正 ERP 当作更正确的真值。", ""]
    (OUTPUT_DIR / "基线敏感性说明.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
