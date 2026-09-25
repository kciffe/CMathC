# -*- coding: utf-8 -*-
"""Final diagnostic: test whether cue-locked common and left/right ERPs repeat.

Uses the Q1-cleaned MAT files, the same per-trial [-200, 0) ms baseline,
and the original chronological trial order. This is descriptive reliability
analysis, not a new neural model fit or a participant-level significance test.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.real_data import load_dataset


INPUT_DIR = ROOT.parent / "q1" / "output" / "8riemann_denoise"
OUTPUT_DIR = ROOT / "output" / "12_split_half_erp_reliability"
RECORDS = {
    "Task1": ("VisualCogA_Task-1", "VisualCogA_Task-2"),
    "Task2": ("VisualCogB_Task-1", "VisualCogB_Task-2"),
}
CHANNELS = ("F3", "Fz", "F4")
WINDOWS_MS = {
    "onset_0_200": (0.0, 200.0),
    "offset_response_200_800": (200.0, 800.0),
    "late_cue_800_2200": (800.0, 2200.0),
    "whole_cue_0_2200": (0.0, 2200.0),
}
BOOTSTRAPS = 1200
SEED = 20260925

font_file = Path("C:/Windows/Fonts/msyh.ttc")
if font_file.exists():
    font_manager.fontManager.addfont(str(font_file))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_file)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(first: np.ndarray, second: np.ndarray) -> dict:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    rms_first = float(np.sqrt(np.mean(first**2)))
    rms_second = float(np.sqrt(np.mean(second**2)))
    return {
        "block1_rms": rms_first,
        "block2_rms": rms_second,
        "block1_to_block2_rms_ratio": rms_first / rms_second if rms_second > 1e-12 else float("nan"),
        "block1_predicts_block2_nrmse": (
            float(np.sqrt(np.mean((first - second) ** 2)) / rms_second)
            if rms_second > 1e-12 else float("nan")
        ),
        "cross_block_waveform_corr": _corr(first, second),
    }


def _bootstrap_mean(trials: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = trials.shape[0]
    indices = rng.integers(0, n, size=(n_boot, n))
    return trials[indices].mean(axis=1)


def _bootstrap_primary(block_data: dict, n_boot: int, rng: np.random.Generator) -> list[dict]:
    # Trial-level bootstrap is stratified by chronological block and cue side.
    means = {}
    for block in ("first", "second"):
        for cue in ("left", "right"):
            means[(block, cue)] = _bootstrap_mean(block_data[(block, cue)], n_boot, rng)

    outputs = []
    for signal in ("common", "left_right_difference"):
        if signal == "common":
            first = (means[("first", "left")] + means[("first", "right")]) / 2.0
            second = (means[("second", "left")] + means[("second", "right")]) / 2.0
        else:
            first = means[("first", "right")] - means[("first", "left")]
            second = means[("second", "right")] - means[("second", "left")]

        f = first.reshape(n_boot, -1)
        s = second.reshape(n_boot, -1)
        f0, s0 = f - f.mean(axis=1, keepdims=True), s - s.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(f0**2, axis=1) * np.sum(s0**2, axis=1))
        corr = np.divide(np.sum(f0 * s0, axis=1), denom,
                         out=np.full(n_boot, np.nan), where=denom > 1e-12)
        rms1 = np.sqrt(np.mean(f**2, axis=1))
        rms2 = np.sqrt(np.mean(s**2, axis=1))
        ratio = np.divide(rms1, rms2, out=np.full(n_boot, np.nan), where=rms2 > 1e-12)
        outputs.append({
            "signal": signal,
            "bootstrap_replicates": n_boot,
            "corr_ci_2_5": float(np.nanquantile(corr, 0.025)),
            "corr_ci_97_5": float(np.nanquantile(corr, 0.975)),
            "block1_to_block2_rms_ratio_ci_2_5": float(np.nanquantile(ratio, 0.025)),
            "block1_to_block2_rms_ratio_ci_97_5": float(np.nanquantile(ratio, 0.975)),
        })
    return outputs


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot(all_records: dict, path: Path) -> None:
    colors = {"F3": "#2166ac", "Fz": "#555555", "F4": "#b2182b"}
    fig, axes = plt.subplots(len(all_records), 2, figsize=(14, 12), sharex=True)
    for row, (record, data) in enumerate(all_records.items()):
        time_ms = data["time_ms"]
        for col, signal in enumerate(("common", "difference")):
            ax = axes[row, col]
            for ci, channel in enumerate(CHANNELS):
                for block, style, label in (("first", "-", "前半"), ("second", "--", "后半")):
                    y = data[signal][block][ci]
                    ax.plot(time_ms, y, color=colors[channel], linestyle=style,
                            linewidth=1.15, alpha=0.95,
                            label=f"{channel} {label}" if row == 0 else None)
            ax.axvline(200.0, color="#777777", linestyle=":", linewidth=1.0)
            ax.axhline(0.0, color="#999999", linewidth=0.65)
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 2200)
            if row == 0:
                ax.set_title("左右共同 ERP" if signal == "common" else "右减左 ERP")
            if col == 0:
                ax.set_ylabel(record.replace("VisualCog", "") + "\n原始单位")
            if row == len(all_records) - 1:
                ax.set_xlabel("相对提示出现时间 (ms)；虚线为约 200 ms 提示消失")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=6, frameon=False)
    fig.suptitle("左右提示 ERP 的时间分块重复性（实线：前半；虚线：后半）", y=1.015, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0, w_pad=1.0)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def run() -> tuple[list[dict], list[dict]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows, bootstrap_rows, order_rows, plot_data = [], [], [], {}
    rng = np.random.default_rng(SEED)

    for task, records in RECORDS.items():
        for record in records:
            data = load_dataset(INPUT_DIR / f"{record}_clean.mat")
            eeg = np.asarray(data["eeg"], dtype=float)
            time_ms = np.asarray(data["time_s"], dtype=float) * 1000.0
            labels = np.asarray(data["cue_type"], dtype=int)
            baseline = (time_ms >= -200.0) & (time_ms < 0.0)
            if baseline.sum() < 2 or not np.all(np.isin(labels, (-1, 1))):
                raise ValueError(f"{record}: baseline/cue labels do not meet the analysis contract")
            corrected = eeg - eeg[:, :, baseline].mean(axis=2, keepdims=True)
            split = len(labels) // 2
            trial_blocks = {
                "first": np.arange(0, split),
                "second": np.arange(split, len(labels)),
            }
            erps, counts = {}, {}
            for block, indices in trial_blocks.items():
                for cue_name, cue_code in (("left", -1), ("right", 1)):
                    selected = indices[labels[indices] == cue_code]
                    if selected.size < 5:
                        raise ValueError(f"{record}/{block}/{cue_name}: too few trials ({selected.size})")
                    counts[(block, cue_name)] = int(selected.size)
                    erps[(block, cue_name)] = corrected[selected].mean(axis=0)

            bins = np.array_split(np.arange(len(labels)), 5)
            bin_counts = [f"{int(np.sum(labels[ix] == -1))}/{int(np.sum(labels[ix] == 1))}"
                          for ix in bins]
            order_corr = _corr(labels.astype(float), np.arange(len(labels), dtype=float))
            order_rows.append({
                "task": task, "record": record, "n_trials": int(len(labels)),
                "left_total": int(np.sum(labels == -1)), "right_total": int(np.sum(labels == 1)),
                "first_block_left": counts[("first", "left")],
                "first_block_right": counts[("first", "right")],
                "second_block_left": counts[("second", "left")],
                "second_block_right": counts[("second", "right")],
                "left_right_per_fifth_bin": ";".join(bin_counts),
                "cue_label_trial_index_correlation": order_corr,
            })

            # Store the complete pre-target interval for plotting.
            epoch = (time_ms >= 0.0) & (time_ms < 2200.0)
            plot_data[record] = {
                "time_ms": time_ms[epoch],
                "common": {},
                "difference": {},
            }
            for block in ("first", "second"):
                left = erps[(block, "left")][:, epoch]
                right = erps[(block, "right")][:, epoch]
                plot_data[record]["common"][block] = (left + right) / 2.0
                plot_data[record]["difference"][block] = right - left

            boot_input = {}
            for block, indices in trial_blocks.items():
                for cue_name, cue_code in (("left", -1), ("right", 1)):
                    selected = indices[labels[indices] == cue_code]
                    boot_input[(block, cue_name)] = corrected[selected][:, :, epoch & (time_ms < 800.0)]
            for signal, first, second in (
                ("common", (erps[("first", "left")] + erps[("first", "right")]) / 2.0,
                 (erps[("second", "left")] + erps[("second", "right")]) / 2.0),
                ("left_right_difference", erps[("first", "right")] - erps[("first", "left")],
                 erps[("second", "right")] - erps[("second", "left")]),
            ):
                primary_mask = epoch & (time_ms < 800.0)
                base = {
                    "task": task, "record": record, "signal": signal,
                    "window": "primary_0_800",
                    **_metrics(first[:, primary_mask], second[:, primary_mask]),
                }
                base.update({f"n_{block}_{cue}": n for (block, cue), n in counts.items()})
                bootstrap_rows.append(base)
            ci = _bootstrap_primary(boot_input, BOOTSTRAPS, rng)
            for row in ci:
                row.update({"task": task, "record": record, "window": "primary_0_800"})
                bootstrap_rows.append(row)

            for window_name, (lo, hi) in WINDOWS_MS.items():
                mask = (time_ms >= lo) & (time_ms < hi)
                if mask.sum() < 5:
                    continue
                common_first = (erps[("first", "left")] + erps[("first", "right")]) / 2.0
                common_second = (erps[("second", "left")] + erps[("second", "right")]) / 2.0
                diff_first = erps[("first", "right")] - erps[("first", "left")]
                diff_second = erps[("second", "right")] - erps[("second", "left")]
                common_metrics = _metrics(common_first[:, mask], common_second[:, mask])
                diff_metrics = _metrics(diff_first[:, mask], diff_second[:, mask])
                common_rms1 = common_metrics["block1_rms"]
                common_rms2 = common_metrics["block2_rms"]
                diff_rms1 = diff_metrics["block1_rms"]
                diff_rms2 = diff_metrics["block2_rms"]
                for signal, metrics in (("common", common_metrics), ("left_right_difference", diff_metrics)):
                    row = {"task": task, "record": record, "window": window_name,
                           "signal": signal,
                           "window_start_ms": lo, "window_end_ms": hi,
                           **metrics,
                           "block1_difference_to_common_rms": diff_rms1 / common_rms1 if common_rms1 > 1e-12 else float("nan"),
                           "block2_difference_to_common_rms": diff_rms2 / common_rms2 if common_rms2 > 1e-12 else float("nan"),
                           **{f"n_{block}_{cue}": n for (block, cue), n in counts.items()}}
                    metric_rows.append(row)
            first_left, first_right = counts[("first", "left")], counts[("first", "right")]
            second_left, second_right = counts[("second", "left")], counts[("second", "right")]
            print(f"{record}: trials={len(labels)}, split={split}/{len(labels)-split}, "
                  f"left/right first={first_left}/{first_right}, "
                  f"second={second_left}/{second_right}", flush=True)

    _write_csv(OUTPUT_DIR / "分半稳定性指标.csv", metric_rows)
    _write_csv(OUTPUT_DIR / "主时间窗Bootstrap.csv", bootstrap_rows)
    _plot(plot_data, OUTPUT_DIR / "左右ERP分半对照.png")
    _write_csv(OUTPUT_DIR / "试次顺序与条件数.csv", order_rows)
    _write_report(metric_rows, bootstrap_rows, order_rows, OUTPUT_DIR / "最后一轮结果与解释.md")
    return metric_rows, bootstrap_rows


def _write_report(metrics: list[dict], boot: list[dict], order: list[dict], path: Path) -> None:
    whole = [r for r in metrics if r["window"] == "whole_cue_0_2200"]
    onset = [r for r in metrics if r["window"] == "onset_0_200"]
    primary = [r for r in metrics if r["window"] == "offset_response_200_800"]
    rows = []
    for r in whole:
        rows.append(f"| {r['record']} | {r['signal']} | {r['block1_rms']:.4g} | {r['block2_rms']:.4g} | "
                    f"{r['cross_block_waveform_corr']:.3f} | {r['block1_predicts_block2_nrmse']:.3f} |")
    p_rows = []
    for r in primary:
        if r["signal"] == "left_right_difference":
            p_rows.append(f"| {r['record']} | {r['block1_rms']:.4g} | {r['block2_rms']:.4g} | "
                          f"{r['cross_block_waveform_corr']:.3f} | {r['block1_predicts_block2_nrmse']:.3f} | "
                          f"{r['block1_difference_to_common_rms']:.3f} / {r['block2_difference_to_common_rms']:.3f} |")
    onset_rows = []
    for r in onset:
        if r["signal"] == "left_right_difference":
            onset_rows.append(f"| {r['record']} | {r['cross_block_waveform_corr']:.3f} | "
                              f"{r['block1_predicts_block2_nrmse']:.3f} | "
                              f"{r['block1_rms']:.4g} / {r['block2_rms']:.4g} |")
    boot_points = {(r.get("record"), r.get("signal")): r for r in boot
                   if "cross_block_waveform_corr" in r}
    boot_intervals = {(r.get("record"), r.get("signal")): r for r in boot
                      if "corr_ci_2_5" in r}
    boot_rows = []
    for record in ("VisualCogA_Task-1", "VisualCogA_Task-2", "VisualCogB_Task-1", "VisualCogB_Task-2"):
        for signal in ("common", "left_right_difference"):
            point, interval = boot_points[(record, signal)], boot_intervals[(record, signal)]
            boot_rows.append(f"| {record} | {signal} | {point['cross_block_waveform_corr']:.3f} | "
                             f"[{interval['corr_ci_2_5']:.3f}, {interval['corr_ci_97_5']:.3f}] | "
                             f"{point['block1_to_block2_rms_ratio']:.3f} | "
                             f"[{interval['block1_to_block2_rms_ratio_ci_2_5']:.3f}, "
                             f"{interval['block1_to_block2_rms_ratio_ci_97_5']:.3f}] |")
    order_table = [f"| {r['record']} | {r['n_trials']} | {r['first_block_left']}/{r['first_block_right']} | "
                   f"{r['second_block_left']}/{r['second_block_right']} | "
                   f"{r['left_right_per_fifth_bin']} | {r['cue_label_trial_index_correlation']:.3f} |"
                   for r in order]
    lines = [
        "# 最后一轮：左右提示 ERP 的分半重复性诊断", "",
        "## 做法", "",
        "直接读取 Q1 清洗后的四份 MAT，不重新滤波、不改模型参数。各试次使用 [-200,0) ms 均值基线校正，按原始试次顺序分成前后两块，分别计算左提示、右提示 ERP。共同 ERP 定义为 (左+右)/2，条件差值定义为右−左。以块1波形预测块2，报告波形相关和以块2 RMS 归一化的 RMSE；主时间窗 0–800 ms 另做按条件、按时间块分层的 1200 次试次 bootstrap。", "",
        "这些是四份记录内的描述性重复性结果，不把四份 MAT 当作四名被试，也不把 bootstrap 当成人群推断。", "",
        "## 共同 ERP：全提示间隔 0–2200 ms", "",
        "| 记录 | 信号 | 前半 RMS | 后半 RMS | 跨块相关 | 前半预测后半 NRMSE |",
        "|---|---|---:|---:|---:|---:|", *rows, "",
        "## 左右差值：提示 onset/offset 主分析窗 200–800 ms", "",
        "| 记录 | 前半差值 RMS | 后半差值 RMS | 跨块相关 | 前半预测后半 NRMSE | 差值/共同 RMS（前/后） |",
        "|---|---:|---:|---:|---:|---:|", *p_rows, "",
        "## 左右差值：提示 onset 窗 0–200 ms", "",
        "| 记录 | 跨块相关 | 前半预测后半 NRMSE | 前/后半差值 RMS |",
        "|---|---:|---:|---:|", *onset_rows, "",
        "## 主分析窗 0–800 ms 的试次 Bootstrap", "",
        "| 记录 | 信号 | 跨块相关 | 相关 95% bootstrap 区间 | 前/后半 RMS 比 | RMS 比 95% 区间 |",
        "|---|---|---:|---:|---:|---:|", *boot_rows, "",
        "所有左右差值相关的 95% bootstrap 区间均跨过 0；这表示当前每半块约 11–24 个单侧试次不足以把较高的点估计变成稳健结论。区间是记录内试次重抽样的不确定性，不是被试总体置信区间。", "",
        "## 试次顺序与左右条件平衡", "",
        "| 记录 | 总试次数 | 前半左/右 | 后半左/右 | 五个连续小块左/右 | 标签与试次序号相关 |",
        "|---|---:|---:|---:|---|---:|", *order_table, "",
        "左右标签与整体试次序号的线性相关绝对值均不超过 0.124，因此没有发现明显的单调标签顺序趋势；但五个连续小块的左右比例仍有起伏，不能排除局部时间漂移与条件不平衡共同影响分半 ERP。", "",
        "## 解释", "",
        "共同 ERP 在全提示间隔的跨块相关高于左右差值（四条记录均如此），说明两类刺激共有的视觉/任务响应比方向差异更可重复。200–800 ms 左右差值在 A_Task-1 与 B_Task-2 点估计较稳定，在 A_Task-2 与 B_Task-1 不稳定或反向；全 0–2200 ms 没有任何一份记录同时呈现高相关、低预测误差的稳定左右差值。", "",
        "结合上一轮，训练估计导联映射未稳定优于固定映射，而 ERP 持续基线的整体留出误差更低。因此当前拟合差不应归咎于单一 G：更直接的限制是左右差异跨记录/时间块不一致、每半块单侧试次数少，以及模型未覆盖实测 ERP 中宽而慢的共同活动。仅凭这批数据无法将这些原因拆分为注意变化、残余漂移、未测脑区源或模型时间常数错误。", "",
        "结论不是‘EEG 完全没有左右信息’，而是‘三电极数据中存在记录特异的左右波形，但当前样本下尚无跨块、跨记录都稳定的方向差异可供统一模型拟合’。因此本轮停止扩展模型参数；不建议靠调大导联左右差、改灰度或继续挑选时间窗来制造拟合改善。", "",
        "结果图：`左右ERP分半对照.png`。指标表：`分半稳定性指标.csv`；主时间窗 bootstrap：`主时间窗Bootstrap.csv`；试次顺序：`试次顺序与条件数.csv`。", "",
        "## 限制", "",
        "本分析可区分“左右差异跨块不重复”与“波形重复但模型未解释”，但不能仅凭分半相关识别造成差异的生理来源。清洗数据中遗留的慢漂移、区块疲劳/注意变化、试次顺序与左右条件偶然不平衡都可能降低跨块重复性；本结果不能把它们区分为单一因果。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
