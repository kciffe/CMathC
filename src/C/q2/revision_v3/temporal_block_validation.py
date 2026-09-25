"""Time-ordered within-record validation of Stage1 cue ERPs.

Uses only the Q1 cleaned MAT epochs, already filtered/resampled by Q1. Trials
are split by their retained chronological order, never randomly permuted.
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config
    from .evaluate import extract_features, fit_classifier, predict_classifier
    from .real_data import load_dataset
except ImportError:
    import config
    from evaluate import extract_features, fit_classifier, predict_classifier
    from real_data import load_dataset


OUT = config.Q2_ROOT / "output" / "revision_v3_validation"
MODES = ("u0_common", "u1_lateral", "u2_shape")
WINDOWS = ((0.0, 180.0), (220.0, 796.875))


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def _corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mode(x):
    # x: trial x F3/Fz/F4 x time; rows of U_OBS are orthonormal sensor modes.
    return np.einsum("kc,nct->nkt", config.U_OBS, x)


def _window_features(trials, time_ms, window):
    mask = (time_ms >= window[0]) & (time_ms <= window[1])
    if not mask.any():
        raise ValueError(f"feature window {window} is outside observed time")
    return _mode(trials)[:, :2, :][:, :, mask].mean(axis=2)


def _fit_small_lda(x, y, shrinkage=0.1):
    """Training-only, balanced, fixed-shrinkage LDA for the 2D event features."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    if x.ndim != 2 or x.shape[1] != 2 or set(np.unique(y)) != {0, 1}:
        raise ValueError("expected both classes and two fixed mode features")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    means = np.stack([z[y == label].mean(axis=0) for label in (0, 1)])
    resid = z - means[y]
    cov = resid.T @ resid / len(y)
    cov = ((1 - shrinkage) * cov + shrinkage * np.trace(cov) / 2 * np.eye(2)
           + 1e-6 * np.eye(2))
    direction = np.linalg.solve(cov, means[1] - means[0])
    return mean, scale, direction, means.mean(axis=0)


def _analyze_record(task, record):
    path = config.REAL_ROOT / f"{record}_clean.mat"
    data = load_dataset(path)
    t = data["time_s"]
    cue = data["cue_type"]
    base = (t >= -0.2) & (t < 0.0)
    resp = (t >= 0.0) & (t <= 800.0 / 1000.0)
    if base.sum() < 2 or resp.sum() < 2:
        raise ValueError(f"{record}: missing baseline or Stage1 response interval")
    # The upstream clean MAT is already Q1-filtered and resampled. Match Q1's
    # per-trial prestimulus mean correction; do not apply a second filter.
    eeg = data["eeg"] - data["eeg"][:, :, base].mean(axis=2, keepdims=True)
    eeg = eeg[:, :, resp]
    time_ms = t[resp] * 1000.0
    modes = _mode(eeg)
    order = np.arange(len(cue))
    cut = len(cue) // 2
    halves = (order < cut, order >= cut)
    rows, feature_rows, waveforms, trial_features = [], [], {}, {}
    cond_data = {}
    for half_i, hmask in enumerate(halves, start=1):
        cond_data[half_i] = {}
        for cond_name, label in (("left", -1), ("right", 1)):
            idx = order[hmask & (cue == label)]
            if len(idx) < 4:
                raise ValueError(f"{record}: chronological half {half_i} has too few {cond_name} trials")
            cond_data[half_i][cond_name] = idx
            mean_modes = modes[idx].mean(axis=0)
            waveforms[(half_i, cond_name)] = mean_modes
            block_indices = order[hmask]
            rows.append({"record": record, "task": task, "half": half_i,
                         "trial_order_start_1based": int(idx.min() + 1),
                         "trial_order_end_1based": int(idx.max() + 1),
                         "half_n_left": int(np.count_nonzero(cue[block_indices] == -1)),
                         "half_n_right": int(np.count_nonzero(cue[block_indices] == 1)),
                         "condition": cond_name, "n_trials": int(len(idx)),
                         "trial_indices_1based": json.dumps((idx + 1).tolist())})
        # Per-trial fixed six-feature vectors; feature windows are frozen from v3.
        feats = {name: extract_features(eeg[idx], time_ms, mode_count=2)
                 for name, idx in cond_data[half_i].items()}
        x = np.concatenate([feats["left"], feats["right"]])
        y = np.r_[np.zeros(len(feats["left"]), dtype=int),
                  np.ones(len(feats["right"]), dtype=int)]
        feature_rows.append((half_i, x, y, feats))
        for window_name, window in (("cue_onset_0_180", WINDOWS[0]),
                                    ("cue_offset_late_220_796", WINDOWS[1])):
            local = {name: _window_features(eeg[idx], time_ms, window)
                     for name, idx in cond_data[half_i].items()}
            trial_features[(half_i, window_name)] = (
                np.concatenate([local["left"], local["right"]]),
                np.r_[np.zeros(len(local["left"]), dtype=int),
                      np.ones(len(local["right"]), dtype=int)])
    # Event-locked two-mode decoding: onset and offset/late are frozen from
    # the protocol before comparing the results, with two dimensions only.
    event_lda_rows = []
    for window_name in ("cue_onset_0_180", "cue_offset_late_220_796"):
        for train_half, test_half in ((1, 2), (2, 1)):
            xt, yt = trial_features[(train_half, window_name)]
            xv, yv = trial_features[(test_half, window_name)]
            mean, scale, direction, midpoint = _fit_small_lda(xt, yt)
            scores = (((xv - mean) / scale) - midpoint) @ direction
            pred = (scores > 0).astype(int)
            recalls = [float(np.mean(pred[yv == label] == label)) for label in (0, 1)]
            event_lda_rows.append({"record": record, "task": task,
                "window": window_name, "train_half": train_half, "test_half": test_half,
                "n_train": int(len(yt)), "n_test": int(len(yv)),
                "balanced_accuracy": float(np.mean(recalls)),
                "recall_left": recalls[0], "recall_right": recalls[1]})

    stability = []
    # Same-record, chronological half 1 vs half 2. Compare common ERPs and
    # right-minus-left waveforms without pooling records.
    for mode_i, mode_name in enumerate(MODES):
        l1, r1 = (waveforms[(1, c)][mode_i] for c in ("left", "right"))
        l2, r2 = (waveforms[(2, c)][mode_i] for c in ("left", "right"))
        d1, d2 = r1 - l1, r2 - l2
        common1, common2 = (r1 + l1) / 2, (r2 + l2) / 2
        for signal_name, a, b in (("common", common1, common2),
                                  ("right_minus_left", d1, d2),
                                  ("left_erp", l1, l2),
                                  ("right_erp", r1, r2)):
            for window_name, (lo, hi) in (("onset", WINDOWS[0]), ("offset_and_late", WINDOWS[1]),
                                          ("full_0_800", (0.0, float(time_ms[-1])))):
                m = (time_ms >= lo) & (time_ms <= hi)
                aa, bb = a[m], b[m]
                rms_a = float(np.sqrt(np.mean(aa * aa)))
                rms_b = float(np.sqrt(np.mean(bb * bb)))
                rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
                denom = max(float(np.sqrt((np.mean(aa * aa) + np.mean(bb * bb)) / 2)), 1e-12)
                stability.append({"record": record, "task": task, "signal": signal_name,
                                  "mode": mode_name, "window": window_name,
                                  "half1_rms": rms_a, "half2_rms": rms_b,
                                  "rms_ratio_half2_over_half1": rms_b / max(rms_a, 1e-12),
                                  "waveform_correlation": _corr(aa, bb),
                                  "normalized_rmse": rmse / denom,
                                  "half1_signed_mean": float(np.mean(aa)),
                                  "half2_signed_mean": float(np.mean(bb))})

    # Fixed six-feature LDA in each chronological direction, no tuning.
    x1, y1 = feature_rows[0][1:3]
    x2, y2 = feature_rows[1][1:3]
    lda_rows = []
    for train_half, test_half, xt, yt, xv, yv in (
            (1, 2, x1, y1, x2, y2), (2, 1, x2, y2, x1, y1)):
        clf = fit_classifier(xt, yt, np.full(len(yt), record, dtype=str))
        scores, pred = predict_classifier(clf, xv)
        recalls = [float(np.mean(pred[yv == lab] == lab)) for lab in (0, 1)]
        lda_rows.append({"record": record, "task": task, "train_chronological_half": train_half,
                         "test_chronological_half": test_half, "n_train": int(len(yt)),
                         "n_test": int(len(yv)), "balanced_accuracy": float(np.mean(recalls)),
                         "recall_left": recalls[0], "recall_right": recalls[1],
                         "score_mean_left": float(scores[yv == 0].mean()),
                         "score_mean_right": float(scores[yv == 1].mean())})
    return rows, stability, lda_rows, event_lda_rows, time_ms, waveforms


def _plot(all_waveforms, path):
    records = list(all_waveforms)
    fig, axes = plt.subplots(len(records), 3, figsize=(13, 3.0 * len(records)), sharex=True)
    if len(records) == 1:
        axes = axes[None, :]
    colors = {1: "#2f5d8a", 2: "#c45b46"}
    modes = ("u0_common", "u1_lateral", "u2_shape")
    for ri, (record, payload) in enumerate(all_waveforms.items()):
        t, waves = payload
        for mi, name in enumerate(modes):
            ax = axes[ri, mi]
            for half in (1, 2):
                if mi == 0:
                    y = (waves[(half, "left")][mi] + waves[(half, "right")][mi]) / 2
                else:
                    y = waves[(half, "right")][mi] - waves[(half, "left")][mi]
                ax.plot(t, y, color=colors[half], lw=1.2, label=f"chronological half {half}")
            ax.axvline(0, color="0.4", lw=.7)
            ax.axvline(200, color="0.4", lw=.7, ls="--")
            ax.axhline(0, color="0.6", lw=.6)
            ax.grid(alpha=.2)
            if ri == 0:
                ax.set_title("common ERP" if mi == 0 else f"right − left: {name}")
            if mi == 0:
                ax.set_ylabel(record + "\n(relative EEG units)")
            if ri == len(records) - 1:
                ax.set_xlabel("cue-relative time (ms)")
            if ri == 0 and mi == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Stage1 ERP reproducibility across chronological trial halves\n"
                 "same-record Q1-cleaned epochs; cue onset 0 ms, cue offset 200 ms")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _stimulus_matrix_audit():
    rows = []
    left = np.load(config.INPUT_ROOT / "stage1_cue_left.npy").astype(np.int16)
    right = np.load(config.INPUT_ROOT / "stage1_cue_right.npy").astype(np.int16)
    baseline = np.load(config.INPUT_ROOT / "stage1_baseline_circle.npy").astype(np.int16)
    white = np.full_like(baseline, 255)
    for name, cue in (("left", left), ("right", right)):
        delta = cue - baseline
        circle_delta = baseline - white
        rows.append({"condition": name, "size": f"{cue.shape[1]}x{cue.shape[0]}",
                     "baseline_circle_changed_pixels_vs_white": int(np.count_nonzero(circle_delta)),
                     "cue_minus_circle_nonzero_pixels": int(np.count_nonzero(delta)),
                     "cue_minus_circle_unique_values": json.dumps(np.unique(delta).tolist()),
                     "common_circle_pixelwise_unchanged": bool(np.all(cue[delta == 0] == baseline[delta == 0])),
                     "left_right_exact_horizontal_mirror": bool(np.array_equal(left, right[:, ::-1])),
                     "current_v3_input_is_absolute_scene": False,
                     "current_v3_input_definition": "(cue image - circle baseline) / 255"})
    return rows


def _write_report(block_rows, stability_rows, lda_rows, stimulus_rows, out):
    records = sorted({r["record"] for r in block_rows})
    lines = [
        "# Revision v3：按时间块验证与刺激输入审计", "",
        "## 本次实际执行", "",
        "- 以第一问 `8riemann_denoise/*_clean.mat` 为数据源；保留试次原始顺序。",
        "- 每份记录按保留试次顺序分成前、后两个连续时间块；在各块内分别算左右提示 ERP。",
        "- 对每个时间块计算共同响应 `(左+右)/2`、条件差异 `右−左`，并比较 u0 共同、u1 侧化、u2 形状三个固定传感器模式。",
        "- 六维 LDA 使用固定特征与固定收缩系数；前半训练、后半测试，再反向一次。所有标准化仅由训练半段估计。",
        "- Q1 clean MAT 已经过第一问滤波/降采样；本分析只重复逐试次 `[-200,0) ms` 基线均值校正，不再次滤波。",
        "", "## 同记录时间块结果", "",
        "下表的相关和 NRMSE 比较前后半段的整段波形。NRMSE 按两半 ERP 的合并 RMS 归一。它们描述时间稳定性，不是显著性检验。", "",
        "| 记录 | u0 共同相关 | u1 左右差异相关 | u2 左右差异相关 | 左右差异 u1 前/后 RMS 比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        def metric(signal, mode, field):
            found = [r[field] for r in stability_rows if r["record"] == record
                     and r["signal"] == signal and r["mode"] == mode
                     and r["window"] == "full_0_800"]
            return found[0] if found else float("nan")
        lines.append(f"| {record} | {metric('common','u0_common','waveform_correlation'):.3f} "
                     f"| {metric('right_minus_left','u1_lateral','waveform_correlation'):.3f} "
                     f"| {metric('right_minus_left','u2_shape','waveform_correlation'):.3f} "
                     f"| {metric('right_minus_left','u1_lateral','rms_ratio_half2_over_half1'):.3f} |")
    lines += ["", "六维 LDA 的时间块留出结果：", "",
              "| 记录 | 前半→后半 BA | 后半→前半 BA | 两向均值 |",
              "|---|---:|---:|---:|"]
    for record in records:
        scores = [r["balanced_accuracy"] for r in lda_rows if r["record"] == record]
        lines.append(f"| {record} | {scores[0]:.3f} | {scores[1]:.3f} | {np.mean(scores):.3f} |")
    lines += ["", "## 解释", "",
              "- 记录间的条件差异不能视为稳定的同一种头皮模式：u1 左右差异波形的前后半相关在四份记录间有正有负，且个别记录的前后半振幅比例很大。",
              "- 共同响应较稳定也不是所有记录/模式都成立；u2 的高相关只说明该传感器对比波形形状相似，不能单独证明它是三角构型特异的神经信号。",
              "- 六维 LDA 在 A_Task-1 与 B_Task-2 的时间块间较可重复，在 A_Task-2 较差，B_Task-1 接近机会水平。四份记录不是四名独立被试；这些分折仅作记录内时间泛化诊断，不能作总体显著性结论。",
              "- 因而当前结果既不支持‘左右 EEG 差异已稳定验证’，也不支持‘第一问数据整体不合格’。它提示记录间/时间块稳定性不足，需要将模型拟合限制在有重复证据的 ERP 成分上。",
              "- v3 原有留一 MAT 的真实 EEG 六维 LDA 平衡准确率为 0.453、AUC 为 0.421；它是纯 EEG 特征基线，与正向模型输出无关。",
              "", "## 刺激矩阵与 v3 输入的对应", "",
              "| 条件 | 刺激尺寸 | 圆圈非白像素 | cue−circle 非零像素 | 左右严格镜像 |",
              "|---|---:|---:|---:|---|"]
    for row in stimulus_rows:
        lines.append(f"| {row['condition']} | {row['size']} | {row['baseline_circle_changed_pixels_vs_white']} "
                     f"| {row['cue_minus_circle_nonzero_pixels']} | {row['left_right_exact_horizontal_mirror']} |")
    lines += ["", "逐像素相减适合提取三角出现/消失的亮度增量：若圆圈在两个画面中相同，它会在差分中抵消。但 v3 把该差分当作空间输入，并从零状态启动；因此它没有表示提示前圆圈已经建立的 LGN/皮层活动，也没有表示提示后圆圈继续显示。差分可作为瞬态输入，不能代替完整场景驱动的神经状态。", "",
              "## v3 代码审计对下一步的约束", "",
              "1. `model_curve_to_q1_grid` 将模型曲线只定义在 0–800 ms，完整 Q1 epoch 其余位置补零；真实 MAT 则是完整试次经双向滤波后的数据。模拟活动在 800 ms 后被硬截断，需先延长状态响应并明确分析窗外事件如何处理。",
              "2. 六源命名把视觉空间左右、形状/方向偏好和解剖源左右混用。应让构型偏好通道先进入共享动力学，再由独立、固定且明确标注的观测映射决定 F3/Fz/F4。",
              "3. `head_model._on_sphere` 把传感器、参考和源全部压到同一 90 mm 半径；点偶极又采用均匀无限导体公式。这不能解释为真实头模型导联场。下一轮比较应把原对称 G 与受约束的有效映射作为候选观测模型，不能把当前矩阵当生理真值。",
              "4. 以上观测/源定义修正完成后，才适合拟合共有 ERP 与可重复的左右差异；跨记录分类留到最后。当前 v3 的失败结果保留为冻结基线。",
              "", "## 产物", "",
              "- `chronological_block_counts.csv`：各记录连续时间块的左右试次数与试次索引。",
              "- `erp_block_stability.csv`：共同 ERP、左右条件 ERP、左右差异的分模式、分时间窗稳定性。",
              "- `within_record_temporal_lda.csv`：双向时间块六维 LDA 留出结果。",
              "- `stimulus_sequence_audit.csv`：像素差分与镜像关系审计。",
              "- `chronological_block_erps.png`：共同 ERP 与左右差异波形的前后时间块对照。",
              "", "本次完成的是同记录时间块实证关卡和代码/刺激输入审计；尚未用这些结果拟合新一版模型。"]
    tail_path = out / "observation_tail_audit.csv"
    if tail_path.exists():
        tail = list(csv.DictReader(tail_path.open(encoding="utf-8-sig")))
        rel = np.asarray([float(r["tail_vs_zero_pad_relative_rmse_0_800"]) for r in tail])
        lines[-1:-1] = ["", "## 800 ms 硬截断对分析窗的影响", "",
                        f"将 cue-only 神经状态延长至 3000 ms，再与 800 ms 后置零的同一模拟比较，八个记录×条件在 0–796.875 ms 的相对 RMSE 中位数为 **{np.median(rel):.3f}**（范围 {np.min(rel):.3f}–{np.max(rel):.3f}）。因此硬截断会改变滤波后的分析窗波形，不能忽略。该隔离比较仍未加入提示前圆圈神经状态及 2.2 s 目标事件，完整事件上下文修正尚未完成。",
                        "详见 `observation_tail_audit.csv` 与 `观测尾部审计.md`。"]
    (out / "验证结果与结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_event_window_results(event_lda_rows, out):
    records = sorted({r["record"] for r in event_lda_rows})
    lines = ["", "## 事件分窗的两维 LDA", "",
             "| 记录 | cue onset 0–180 ms BA | cue offset/late 220–796.875 ms BA |",
             "|---|---:|---:|"]
    for record in records:
        vals = {}
        for window in ("cue_onset_0_180", "cue_offset_late_220_796"):
            vals[window] = np.mean([r["balanced_accuracy"] for r in event_lda_rows
                                    if r["record"] == record and r["window"] == window])
        lines.append(f"| {record} | {vals['cue_onset_0_180']:.3f} | "
                     f"{vals['cue_offset_late_220_796']:.3f} |")
    lines += ["", "窗口由提示出现与消失时刻预先定义。每折仅使用共同、侧化两个固定模式均值；标准化和判别方向只从训练时间块估计。该结果用于记录内时间泛化诊断，不是总体显著性检验。",
              "详见 `event_window_temporal_lda.csv`。"]
    with (out / "验证结果与结论.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    block_rows, stability_rows, lda_rows, event_lda_rows, plots = [], [], [], [], {}
    for task, records in config.DATASETS.items():
        for record in records:
            blocks, stability, lda, event_lda, time_ms, waves = _analyze_record(task, record)
            block_rows.extend(blocks)
            stability_rows.extend(stability)
            lda_rows.extend(lda)
            event_lda_rows.extend(event_lda)
            plots[record] = (time_ms, waves)
            print(f"{record}: chronological trial split complete", flush=True)
    stimulus_rows = _stimulus_matrix_audit()
    _write_csv(OUT / "chronological_block_counts.csv", block_rows)
    _write_csv(OUT / "erp_block_stability.csv", stability_rows)
    _write_csv(OUT / "within_record_temporal_lda.csv", lda_rows)
    _write_csv(OUT / "event_window_temporal_lda.csv", event_lda_rows)
    _write_csv(OUT / "stimulus_sequence_audit.csv", stimulus_rows)
    _plot(plots, OUT / "chronological_block_erps.png")
    _write_report(block_rows, stability_rows, lda_rows, stimulus_rows, OUT)
    _append_event_window_results(event_lda_rows, OUT)
    summary = {
        "purpose": "Step 2 empirical reproducibility gate before further model fitting",
        "data": "Q1 cleaned MAT; already filtered/resampled by Q1; per-trial [-200,0) ms baseline correction",
        "split": "retained original trial order; first chronological half vs second chronological half",
        "n_records": len(plots), "n_block_rows": len(block_rows),
        "n_stability_rows": len(stability_rows), "n_temporal_lda_folds": len(lda_rows),
        "important_limit": "time blocks are record segments, not independent participants; preprocessing is inherited from Q1",
        "outputs": ["chronological_block_counts.csv", "erp_block_stability.csv",
                    "within_record_temporal_lda.csv", "stimulus_sequence_audit.csv",
                    "event_window_temporal_lda.csv",
                    "chronological_block_erps.png", "observation_tail_audit.csv",
                    "验证结果与结论.md"]}
    (OUT / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved chronological validation to {OUT}", flush=True)


if __name__ == "__main__":
    run()
