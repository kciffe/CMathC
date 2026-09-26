"""Run the corrected first-subquestion forward model and save auditable results."""
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

try:
    from . import config
    from .evaluate import classification_leave_one_record
    from .evidence_chain import run_evidence_chain
    from .fit import FitResult, _cached_left_frontend, fit_leave_one_record, register_frontend
    from .frontend import (choose_resolution, load_stimulus, mirror_stage1_frontend,
                           simulate_frontend, temporal_stride_audit)
    from .head_model import build_sensor_leadfield, geometry_manifest
    from .model import (ModelParams, calibrate_drive_scales,
                        observable_modes_from_real, simulate_forward)
    from .observation import model_curve_to_q1_grid
    from .real_data import load_cases_with_audit
except ImportError:
    import config
    from evaluate import classification_leave_one_record
    from evidence_chain import run_evidence_chain
    from fit import FitResult, _cached_left_frontend, fit_leave_one_record, register_frontend
    from frontend import (choose_resolution, load_stimulus, mirror_stage1_frontend,
                          simulate_frontend, temporal_stride_audit)
    from head_model import build_sensor_leadfield, geometry_manifest
    from model import (ModelParams, calibrate_drive_scales,
                       observable_modes_from_real, simulate_forward)
    from observation import model_curve_to_q1_grid
    from real_data import load_cases_with_audit


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                default=_json_default), encoding="utf-8")


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=_json_default)
                             if isinstance(value, (dict, list, tuple, np.ndarray)) else value
                             for key, value in row.items()})


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _interp_rows(values, source_time, target_time):
    return np.vstack([np.interp(target_time, source_time, row) for row in values])


def _simulate_observed(stage, condition, params, resolution, drive_scales,
                       include_offset=True, remove_position=False, amplitude=1.0,
                       base_front=None):
    stimulus = load_stimulus(stage, condition)
    if base_front is not None and stage == "Stage1":
        front = (base_front if condition == "left"
                 else mirror_stage1_frontend(base_front, stimulus))
    else:
        front = simulate_frontend(stimulus, params={"tau_a": params.tau_a},
                                  resolution=resolution, include_offset=include_offset,
                                  remove_position=remove_position)
    result = simulate_forward(front, params=params, amplitude=amplitude,
                              drive_scales=drive_scales)
    observed, observed_time = model_curve_to_q1_grid(result.eeg_scaled, result.time_ms)
    return observed, observed_time, result, front


def _fit_row(record, result):
    return {"heldout_record": record, "train_records": result.train_records,
            "tau_s_ms": result.parameters["tau_s"], "g_i": result.parameters["g_i"],
            "tau_a_ms": result.parameters["tau_a"], "shared_gain_signed": result.amplitude,
            "training_loss": result.loss, "optimizer_success": result.success,
            "fit_status": result.status, "boundary_flags": result.boundary_flags,
            "objective_evaluations": result.n_evaluations, "resolution": result.resolution}


def _compute_predictions(cases, fits, resolution, drive_scales, progress=True):
    grouped = {}
    for case in cases:
        if case["stage"] == "Stage1" and case["condition"] in ("left", "right"):
            grouped.setdefault(case["dataset"], {})[case["condition"]] = case
    predictions, metric_rows, difference_rows, long_rows = {}, [], [], []
    for record, fit in fits.items():
        if progress:
            print(f"  held-out prediction: {record}", flush=True)
        params = ModelParams.from_any(fit.parameters)
        ref_front = _cached_left_frontend(float(params.tau_a), int(resolution))
        for condition in ("left", "right"):
            case = grouped[record][condition]
            observed, out_time, _, _ = _simulate_observed(
                "Stage1", condition, params, resolution, drive_scales,
                amplitude=fit.amplitude, base_front=ref_front)
            pred = _interp_rows(observed, out_time, case["time_ms"])
            predictions[(record, condition)] = pred
            error = pred - case["real"]
            real_rms = float(np.sqrt(np.mean(case["real"] ** 2)))
            rmse = float(np.sqrt(np.mean(error ** 2)))
            row = {"record": record, "task": case["task"], "condition": condition,
                   "n_trials": case["n_trials"], "real_rms": real_rms, "rmse": rmse,
                   "nrmse": rmse / real_rms if real_rms > 0 else float("nan"),
                   "skill_vs_zero": 1.0 - (rmse / real_rms) ** 2 if real_rms > 0 else float("nan")}
            for channel_i, channel in enumerate(config.CHANNELS):
                a, b = case["real"][channel_i], pred[channel_i]
                row[f"{channel}_correlation"] = (float(np.corrcoef(a, b)[0, 1])
                    if np.std(a) > 1e-12 and np.std(b) > 1e-12 else float("nan"))
            metric_rows.append(row)
            for ti, t in enumerate(case["time_ms"]):
                for channel_i, channel in enumerate(config.CHANNELS):
                    long_rows.append({"record": record, "task": case["task"],
                                      "condition": condition, "time_ms": float(t),
                                      "channel": channel, "measured": float(case["real"][channel_i, ti]),
                                      "model_heldout": float(pred[channel_i, ti])})

        left, right = grouped[record]["left"], grouped[record]["right"]
        real_diff = right["real"] - left["real"]
        pred_diff = predictions[(record, "right")] - predictions[(record, "left")]
        diff_rmse = float(np.sqrt(np.mean((real_diff - pred_diff) ** 2)))
        diff_rms = float(np.sqrt(np.mean(real_diff ** 2)))
        mode_real = observable_modes_from_real(real_diff)
        mode_pred = observable_modes_from_real(pred_diff)
        difference_rows.append({"record": record, "task": left["task"],
            "left_right_difference_rms_measured": diff_rms,
            "left_right_difference_rms_model": float(np.sqrt(np.mean(pred_diff ** 2))),
            "left_right_difference_nrmse": diff_rmse / diff_rms if diff_rms > 0 else float("nan"),
            "left_right_difference_correlation": (float(np.corrcoef(real_diff.ravel(), pred_diff.ravel())[0, 1])
                if np.std(real_diff) > 1e-12 and np.std(pred_diff) > 1e-12 else float("nan")),
            "measured_mode_rms": np.sqrt(np.mean(mode_real ** 2, axis=1)).tolist(),
            "model_mode_rms": np.sqrt(np.mean(mode_pred ** 2, axis=1)).tolist(),
            "measured_lateral_energy_fraction": float(np.sum(mode_real[1] ** 2) / np.sum(mode_real ** 2))
                if np.sum(mode_real ** 2) > 0 else float("nan"),
            "model_lateral_energy_fraction": float(np.sum(mode_pred[1] ** 2) / np.sum(mode_pred ** 2))
                if np.sum(mode_pred ** 2) > 0 else float("nan")})
    return grouped, predictions, metric_rows, difference_rows, long_rows


def _controls(fits, resolution, drive_scales, progress=True):
    rows = []
    for record, fit in fits.items():
        if progress:
            print(f"  frozen-parameter mechanism controls: {record}", flush=True)
        params = ModelParams.from_any(fit.parameters)
        conditions = {}
        for name, kwargs in (("full", {}), ("no_spatial_position", {"remove_position": True}),
                             ("cue_onset_only", {"include_offset": False})):
            left_front = simulate_frontend(load_stimulus("Stage1", "left"),
                params={"tau_a": params.tau_a}, resolution=resolution, **kwargs)
            pair = []
            for condition in ("left", "right"):
                y, _, _, _ = _simulate_observed("Stage1", condition, params, resolution,
                    drive_scales, amplitude=fit.amplitude, base_front=left_front, **kwargs)
                pair.append(y)
            conditions[name] = pair
        full_diff = conditions["full"][1] - conditions["full"][0]
        full_rms = float(np.sqrt(np.mean(full_diff ** 2)))
        for name in ("no_spatial_position", "cue_onset_only"):
            delta = conditions[name][1] - conditions[name][0]
            rms = float(np.sqrt(np.mean(delta ** 2)))
            rows.append({"heldout_record": record, "control": name,
                         "parameters_refit": False, "shared_gain_refit": False,
                         "full_left_right_difference_rms": full_rms,
                         "control_left_right_difference_rms": rms,
                         "retained_fraction": rms / full_rms if full_rms > 1e-15 else float("nan")})
    return rows


def _classify(cases):
    score_rows, folds, summary = classification_leave_one_record(cases, mode_count=2)
    return score_rows, folds, [{"feature_set": "6 fixed common/lateral temporal means",
        "balanced_accuracy_mean": summary["balanced_accuracy"],
        "roc_auc_mean": summary["roc_auc"], "recall_left_mean": summary["recall_left"],
        "recall_right_mean": summary["recall_right"],
        "heldout_record_count": summary["n_heldout_records"],
        "interpretation": "real-only leave-one-MAT-out; four MAT records are not four independent participants"}]


def _plot_erps(grouped, predictions, path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    colors = {"measured": "#222222", "model": "#bd4b3b"}
    for col, condition in enumerate(("left", "right")):
        arrays = [group[condition]["real"] for group in grouped.values()]
        models = [predictions[(record, condition)] for record in grouped]
        times = next(iter(grouped.values()))[condition]["time_ms"]
        for row, channel in enumerate(config.CHANNELS):
            ax = axes[row, col]
            for label, values in (("Measured", arrays), ("Model, held-out MAT", models)):
                a = np.asarray(values)[:, row, :]
                mean = a.mean(axis=0)
                sd = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros_like(mean)
                color = colors["measured" if label == "Measured" else "model"]
                ax.plot(times, mean, color=color, lw=1.7, label=label)
                ax.fill_between(times, mean - sd, mean + sd, color=color, alpha=.12, linewidth=0)
            ax.axvline(0, color="0.5", lw=.8)
            ax.axvline(200, color="0.5", lw=.8, ls="--")
            ax.axhline(0, color="0.7", lw=.6)
            ax.grid(alpha=.2)
            ax.set_title(f"{condition} cue · {channel}")
            if row == 2:
                ax.set_xlabel("Cue-relative time (ms)")
            if col == 0:
                ax.set_ylabel("Q1-processed EEG (source units)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out MAT comparison (model and data share Q1 observation processing)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_differences(grouped, predictions, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    names = ("common u0", "lateral u1", "shape u2")
    time = next(iter(grouped.values()))["left"]["time_ms"]
    for mode, ax in enumerate(axes):
        real, model = [], []
        for record, group in grouped.items():
            real_diff = group["right"]["real"] - group["left"]["real"]
            model_diff = predictions[(record, "right")] - predictions[(record, "left")]
            real.append(observable_modes_from_real(real_diff)[mode])
            model.append(observable_modes_from_real(model_diff)[mode])
        for label, values, color in (("Measured", real, "#222222"),
                                     ("Model, held-out MAT", model, "#bd4b3b")):
            a = np.asarray(values)
            mean = a.mean(axis=0)
            sd = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros_like(mean)
            ax.plot(time, mean, label=label, color=color, lw=1.7)
            ax.fill_between(time, mean - sd, mean + sd, color=color, alpha=.12, linewidth=0)
        ax.axvline(0, color="0.5", lw=.8)
        ax.axvline(200, color="0.5", lw=.8, ls="--")
        ax.axhline(0, color="0.7", lw=.6)
        ax.set_ylabel(names[mode])
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Cue-relative time (ms)")
    fig.suptitle("Measured and forward-model right-minus-left modes")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_cascade(result, path):
    t = result.time_ms
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    labels = ("early / LGN-driven", "shape integration", "orientation-template")
    for pop, ax in enumerate(axes[:3]):
        ax.plot(t, result.excitatory[pop].mean(axis=0), label="E", color="#3265a8")
        ax.plot(t, result.inhibitory[pop].mean(axis=0), label="I", color="#de8755")
        ax.set_ylabel(labels[pop])
        ax.grid(alpha=.2)
    axes[3].plot(t, result.eeg_scaled[0], label="F3", color="#3265a8")
    axes[3].plot(t, result.eeg_scaled[1], label="Fz", color="#4b8d62")
    axes[3].plot(t, result.eeg_scaled[2], label="F4", color="#b85c56")
    axes[3].set_ylabel("sensor proxy")
    axes[3].set_xlabel("Cue-relative time (ms)")
    axes[3].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.axvline(0, color="0.5", lw=.8)
        ax.axvline(200, color="0.5", lw=.8, ls="--")
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Illustrative cascade under the fitted parameter median")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_lgn_spatiotemporal(frontend, path):
    """Plot the fitted representative cue's pooled LGN ON/OFF space-time activity."""
    t = np.asarray(frontend["time_ms"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, constrained_layout=True)
    for ax, key, title in zip(axes, ("lgn_on", "lgn_off"), ("LGN TCR ON", "LGN TCR OFF")):
        values = np.asarray(frontend[key], dtype=float)
        if values.ndim != 3 or values.shape[-1] != len(t):
            raise ValueError(f"{key} must have shape [8,8,time] matching time_ms")
        space_time = values.reshape(-1, len(t))
        vmax = max(float(np.quantile(space_time, .995)), 1e-9)
        im = ax.imshow(space_time, origin="lower", aspect="auto", interpolation="nearest",
                       extent=(t[0], t[-1], 0, space_time.shape[0]),
                       cmap="magma", vmin=0, vmax=vmax)
        ax.axvline(0, color="white", lw=.8, alpha=.8)
        ax.axvline(200, color="cyan", lw=.8, ls="--", alpha=.8)
        ax.set_ylabel("8×8 pooled spatial channel")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="TCR activity (relative)")
    axes[-1].set_xlabel("Cue-relative time (ms)")
    fig.suptitle("LGN spatiotemporal response to the standardized left cue\n"
                 "solid marker: cue onset; dashed marker: 200 ms cue offset")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cortical_ei(result, path):
    """Plot E/I responses for all three groups without averaging L/R channels."""
    t = result.time_ms
    titles = ("Early visual-field population", "Spatial-configuration population",
              "Triangle-template preference population")
    channel_names = (("left visual field", "right visual field"),
                     ("left visual field", "right visual field"),
                     ("left-template preference", "right-template preference"))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for pop, ax in enumerate(axes):
        for state, values, color in (("E", result.excitatory, "#3265a8"),
                                     ("I", result.inhibitory, "#d78345")):
            for ch in range(2):
                ax.plot(t, values[pop, ch], color=color,
                        linestyle="-" if ch == 0 else "--", linewidth=1.25,
                        label=f"{state} · {channel_names[pop][ch]}")
        ax.axvline(0, color="0.5", lw=.8)
        ax.axvline(200, color="0.5", lw=.8, ls="--")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Activity (0–1)")
        ax.set_title(titles[pop])
        ax.grid(alpha=.2)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel("Cue-relative time (ms)")
    fig.suptitle("Three cortical E/I population responses under the fitted parameter median")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_source_electrode_contributions(result, path):
    """Plot five signed source contributions to each electrode; verify summation."""
    lead = np.asarray(result.diagnostics["leadfield"], dtype=float)
    source = np.asarray(result.source_proxy, dtype=float)
    gain = float(result.diagnostics["amplitude"])
    contributions = gain * lead[:, :, None] * source[None, :, :]
    total = contributions.sum(axis=1)
    if not np.allclose(total, result.eeg_scaled, rtol=1e-5, atol=1e-8):
        raise AssertionError("source contributions do not sum to the saved electrode waveform")
    colors = ("#3265a8", "#64a1c8", "#4b8d62", "#8bb96d", "#9b64a8")
    plot_labels = ("Early LH ← RVF", "Early RH ← LVF", "Config LH ← RVF",
                   "Config RH ← LVF", "Template opponent midline")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for sensor, ax in enumerate(axes):
        for source_i, name in enumerate(plot_labels):
            ax.plot(result.time_ms, contributions[sensor, source_i], color=colors[source_i],
                    linewidth=1.0, alpha=.9, label=name)
        ax.plot(result.time_ms, total[sensor], color="#222222", linewidth=1.8,
                label="sum = electrode prediction")
        ax.axvline(0, color="0.5", lw=.8)
        ax.axvline(200, color="0.5", lw=.8, ls="--")
        ax.axhline(0, color="0.7", lw=.6)
        ax.set_ylabel(f"{config.CHANNELS[sensor]}\nrelative units")
        ax.grid(alpha=.2)
        if sensor == 0:
            ax.legend(frameon=False, ncol=3, fontsize=7.5)
    axes[-1].set_xlabel("Cue-relative time (ms)")
    fig.suptitle("Five source-to-electrode contributions under the fitted parameter median")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_readme(out, resolution, stride, scales, fits, metrics, differences,
                  classification, controls, max_nfev, evidence=None):
    nrmse = np.asarray([r["nrmse"] for r in metrics], dtype=float)
    diff_corr = np.asarray([r["left_right_difference_correlation"] for r in differences], dtype=float)
    fit_lines = [f"- `{record}`: tau_s={fit.parameters['tau_s']:.2f} ms, "
                 f"g_i={fit.parameters['g_i']:.3f}, tau_a={fit.parameters['tau_a']:.1f} ms, "
                 f"gain={fit.amplitude:.4g}, status=`{fit.status}`"
                 for record, fit in fits.items()]
    control_summary = {}
    for name in ("no_spatial_position", "cue_onset_only"):
        values = [r["retained_fraction"] for r in controls if r["control"] == name]
        control_summary[name] = float(np.mean(values)) if values else float("nan")
    real_lateral = np.asarray([r["measured_lateral_energy_fraction"] for r in differences], dtype=float)
    model_lateral = np.asarray([r["model_lateral_energy_fraction"] for r in differences], dtype=float)
    boundary_count = sum(bool(fit.boundary_flags.get("tau_s")) for fit in fits.values())
    nonconverged_count = sum(not bool(fit.success) for fit in fits.values())
    lines = [
        "# Revision v3：问题二第一小问计算结果", "",
        "## 实现内容", "",
        "本版本用标准化刺激矩阵作为确定性输入，经 LGN ON/OFF 与 Gabor/形状模板前端、三组 E/I 群体动力学得到五个语义明确的源代理：左右视野分别投射到对侧半球的早期视觉与构型源，以及由左右模板偏好差形成的双侧中线对手源。模板偏好组接收左右视觉视野早期活动的固定等权平均；中线对手源是低维建模假设，不是解剖定位结论。电极/参考在外表面，源坐标在头内；再以均匀无限导体点偶极近似映射到 F3/Fz/F4。该映射是规范近似，不是有限球体或个体头模型。模型曲线与真实数据统一通过第一问的 0.2–24 Hz 双向四阶滤波、128 Hz 重采样和逐通道基线校正。拟合按留一 MAT 记录进行，尺度由固定左右参考刺激预先计算，不读取留出记录确定尺度。", "",
        "## 本次设置", "",
        f"- 空间网格：{resolution} × {resolution}；前端时间特征步长：{stride:g} ms。",
        f"- 输入群体 RMS 尺度（early/shape/orientation，每个含左右通道）：`{np.asarray(scales).round(6).tolist()}`。",
        "- 导联场：规范化 10–20 位置、均匀导体点偶极近似、假定双乳突平均参考。它是可复现的示意映射，非个体头模；增益没有 μV 物理标定。",
        "- Stage1 包含 0 ms 提示出现与 200 ms 提示消失；晚期成分不直接称为 P300。",
        "", "## 留一记录拟合", "", *fit_lines,
        f"- 四折中 `tau_s` 到达 100 ms 上界：{boundary_count}/4；达到设定的 {max_nfev} 次目标评估上限而未收敛：{nonconverged_count}/4。报告这些参数时应视为边界候选值，不能当作已识别的时间常数。",
        "", f"对左右 × 四份 MAT 的整体 held-out NRMSE 均值为 **{np.nanmean(nrmse):.3f}**（按实测 ERP RMS 归一）。若该值接近 1，正向模型的绝对波形解释力弱；幅度增益不能掩盖这一点。",
        f"右减左波形相关系数的四记录均值为 **{np.nanmean(diff_corr):.3f}**。相关系数只描述波形相似，不代表幅度或机制正确。",
        f"几何映射下模型左右差异的侧化能量占比均值为 **{np.nanmean(model_lateral):.3f}**；实测为 **{np.nanmean(real_lateral):.3f}**（单记录范围 {np.nanmin(real_lateral):.3f}–{np.nanmax(real_lateral):.3f}）。这表明当前对称源/导联假设把差异限制在侧化方向，不能解释实测中出现的共同/形状模态差异。",
        "", "## 特征区分验证", "",
        f"固定 6 维特征（u0 与 u1 各自的 80–200、250–450、450–700 ms 均值）在留一 MAT 记录 shrinkage-LDA 的平衡准确率为 **{classification[0]['balanced_accuracy_mean']:.3f}**，AUC 为 **{classification[0]['roc_auc_mean']:.3f}**。四份 MAT 是记录，不应等同四个独立被试；这项跨记录结果若接近机会水平，说明当前固定特征没有稳健的记录间泛化。",
        "", "## 针对性机制对照", "",
        f"去除空间位置后保留的模型左右差异比例均值：**{control_summary['no_spatial_position']:.3f}**。仅保留 cue onset、删除 200 ms offset 后的比例：**{control_summary['cue_onset_only']:.3f}**。对照使用同一折拟合参数和同一增益，不重新拟合。比例用于描述模型生成差异的来源，不用于声称真实 EEG 的因果效应。",
        "", "## 解释边界", "",
        "1. 本模型给出从输入到传感器代理的显式正向计算链，可用于解释模型内部的差异来源；三电极与四份记录不足以识别真实脑内源分布。",
        "2. Stage1 的左右模板响应是形状方向编码的模型假设。必须同时报告实测右减左波形和留出结果；分类或拟合指标差不能单独证明生理机制。",
        "3. The cue-only model is simulated naturally through the complete 0-3000 ms epoch; no target response at 2.2 s is inserted. The measured context may contain a target-related component that leaks backward through zero-phase filtering, so absolute ERP fit remains conditional on the unverified event context.",
        "4. 真实 EEG 使用第一问已清洗数据，模型在 ERP 前执行逐试次基线均值校正；这不等于线性基线回归，也不重新裁定第一问数据清洗的有效性。",
        "5. 原始的 v1/v2 代码和结果均未覆盖或删除；25 次预算结果另存于 `output/revision_v3_max25/`。",
        "", "## 主要输出", "",
        "- `heldout_erp.png`：三通道左右条件的实测与留出模型 ERP。",
        "- `heldout_difference_modes.png`：右减左的 common/lateral/shape 模态。",
        "- `fitted_cascade.png`：拟合参数中位数下的 E/I 群体和观测代理。",
        "- `lgn_spatiotemporal.png`：代表性左 cue 下 LGN ON/OFF 空间通道随时间的响应图。",
        "- `cortical_ei_responses.png`：三组 E/I 响应，保留视野位置和模板偏好通道标签。",
        "- `source_to_electrode_contributions.png`：五源对 F3/Fz/F4 的逐时贡献；代码检查其和等于电极预测。",
        "- `lgn_spatiotemporal.png`：同一次拟合代表刺激下，8×8 pooled LGN ON/OFF 通道的时空活动。",
        "- `cortical_ei_responses.png`：三组 E/I 群体逐通道响应；模板偏好通道与视觉视野通道分开标注。",
        "- `source_to_electrode_contributions.png`：五个源代理对 F3/Fz/F4 的逐源贡献；图中校验各贡献之和等于电极预测。",
        "- `heldout_metrics.csv`、`left_right_difference.csv`、`fixed_feature_lda.csv`、`mechanism_controls.csv`。",
    ]
    if evidence is not None:
        result = evidence["classification_summary"]
        lines.extend([
            "", "## Fixed nine-dimensional electrode ERP feature validation", "",
            (f"The predeclared feature consists of F3, Fz, and F4 means in "
             f"[100,250), [250,500), and [500,800) ms. Leave-one-MAT-out "
             f"shrinkage-LDA balanced accuracy is **{result['macro_balanced_accuracy']:.3f}**, "
             f"macro AUC is **{result['macro_roc_auc']:.3f}** over "
             f"{result['heldout_record_count']} held-out records. The windows and channels "
             "were fixed before examining these fold results; scaling was estimated from training records only."),
            "Four MAT files are recordings, not verified independent participants. This result is evidence about this fixed feature under record holdout, not evidence that the forward mechanism is correct.",
            "The associated model population/source differences use the same fitted parameters, drive scales, gain, and observation operator for the left/right pair in each held-out fold. Five-source additivity is checked after filtering, resampling, and baseline correction.",
        "New outputs: `model_population_right_minus_left.png`, `model_source_electrode_right_minus_left.png`, `measured_vs_model_candidate_features.png`, `heldout_9d_electrode_lda.png`, `heldout_9d_electrode_lda_folds.csv`, `measured_candidate_features_exploratory.csv`, and `model_source_contribution_summary.csv`.",
        ])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_evidence_report(out, metrics, differences, fit_rows, evidence, event_rows):
    """Write a paper-facing summary with only PNG figures and explicit evidence tiers."""
    report = config.Q2_ROOT / "output" / "修改3.md"
    summary = evidence["classification_summary"]
    fold_rows = evidence["classification_folds"]
    mean_nrmse = float(np.nanmean([row["nrmse"] for row in metrics]))
    mean_difference_corr = float(np.nanmean(
        [row["left_right_difference_correlation"] for row in differences]))
    nonconverged = sum(row["fit_status"] != "profile_grid_converged" for row in fit_rows)
    above_chance_ba = sum(row.get("status") == "complete" and row["balanced_accuracy"] > .5
                          for row in fold_rows)
    above_chance_auc = sum(row.get("status") == "complete" and row["roc_auc"] > .5
                           for row in fold_rows)
    cue_onsets = [float(row["cue_onset_median_ms"]) for row in event_rows
                  if np.isfinite(float(row["cue_onset_median_ms"]))]
    cue_offsets = [float(row["cue_offset_median_ms"]) for row in event_rows
                   if np.isfinite(float(row["cue_offset_median_ms"]))]
    tail_audit_path = config.Q2_ROOT / "output" / "revision_v3_validation" / "observation_tail_audit.csv"
    with tail_audit_path.open("r", newline="", encoding="utf-8-sig") as stream:
        tail_errors = [float(row["tail_vs_zero_pad_relative_rmse_0_800"])
                       for row in csv.DictReader(stream)]
    if (summary["macro_balanced_accuracy"] <= .5
            or summary["macro_roc_auc"] <= .5):
        classification_conclusion = (
            "当前 9 维特征没有获得留一记录验证支持：宏平均平衡准确率或 AUC 未超过 0.5。"
            "这表示固定特征在当前跨记录测试中未显示可靠区分能力。")
    else:
        classification_conclusion = (
            "当前 9 维特征的两个宏平均指标均高于 0.5；由于只有四份记录，"
            "仍应作为初步跨记录支持，而非稳定普遍规律。")
    lines = [
        "# 问题二第一小问：修正后的正向模型与左右特征证据链", "",
        "## 本轮处理", "",
        "- 前端和皮层动力学模拟延长至 0–3000 ms，使 0–800 ms 评价段之后的模型响应自然演化；再统一执行第一问观测算子，最后截取评价窗口。没有在 800 ms 处把活动硬截断后补零。",
        "- 未把 2.2 s 目标加入 cue-only 模型。目标场景和真实记录参考电极信息仍未完全核实，分别作为上下文假设和导联近似的局限。",
        (f"- 事件核对：四份记录的 VisCue 中位起点均为 {np.median(cue_onsets):.3f} ms；"
         f"记录到的 cue 中位偏移为 {np.median(cue_offsets):.3f} ms，接近名义 200 ms。"
         "2.20 s 目标时刻来自实验流程假设，Action 通道是应答/动作标记，未被当作已确认目标触发输入。"),
        "- 每个留一 MAT 折只用其余记录拟合参数；同一折的左、右提示由完全相同的参数、输入尺度、增益和观测算子生成。",
        "- 真实 EEG 候选特征预先固定为 F3/Fz/F4 在 `[100,250)`、`[250,500)`、`[500,800)` ms 的均值，共 9 维。窗口、通道和 LDA 收缩系数没有根据留出记录调整；特征标准化只在训练记录拟合。",
        "", "## 结果概览", "",
        f"- 左右条件留出 ERP 平均 NRMSE：**{mean_nrmse:.3f}**。这是绝对波形拟合指标；数值越接近 0 越好。",
        f"- 实测与模型右减左波形相关的记录均值：**{mean_difference_corr:.3f}**。相关为负表示当前模型差波形的时间形状没有复现实测差波形。",
        f"- 四折中有 **{nonconverged}/4** 折的优化器未报告收敛；使用的是预算内最优候选，不能将其描述为已收敛或已识别的生理参数。",
        f"- 固定 9 维特征的留一 MAT 记录平衡准确率宏平均：**{summary['macro_balanced_accuracy']:.3f}**；宏平均 AUC：**{summary['macro_roc_auc']:.3f}**。四个 MAT 文件是记录而非已确认的四名独立受试者。",
        f"- 留出折中，平衡准确率高于 0.5 的记录为 {above_chance_ba}/{summary['heldout_record_count']}；AUC 高于 0.5 的记录为 {above_chance_auc}/{summary['heldout_record_count']}。**{classification_conclusion}**",
        f"- 源加和检查最大相对 RMS 残差：**{evidence['source_sum_max_relative_rms_error']:.3e}**；阈值 `1e-5`。残差来自保存电极信号时的 `float32` 舍入。",
        "", "### 留出记录结果", "",
        "| 留出记录 | 测试试次数 | 平衡准确率 | AUC | 训练记录 |", "|---|---:|---:|---:|---|",
    ]
    for row in fold_rows:
        if row.get("status") == "complete":
            lines.append(f"| {row['heldout_record']} | {row['n_test_trials']} | "
                         f"{row['balanced_accuracy']:.3f} | {row['roc_auc']:.3f} | "
                         f"{row['training_records']} |")
    lines.extend([
        "", "## 800 ms 截断与滤波尾部审计", "",
        (f"在 8 个记录×左右条件中，保留模型自然尾部与 800 ms 后强制补零两种处理所得的 0–800 ms 波形，"
         f"相对 RMSE 中位数为 **{np.median(tail_errors):.4f}**、均值为 **{np.mean(tail_errors):.4f}**，"
         f"范围 **{np.min(tail_errors):.4f}–{np.max(tail_errors):.4f}**。这量化的是滤波边界处理对分析窗的影响；"
         "延长模拟并消除这项边界差异，不代表实测 ERP 拟合因此变好，也没有解决未知目标事件上下文。"),
        "![cue-only神经群体活动自然衰减到3000ms](revision_v3_validation/persistent_activity_decay.png)",
        "![保留自然尾部与800ms后补零的滤波差异](revision_v3_validation/zero_padding_filter_effect.png)",
        "", "## 左右差异的模型分解", "",
        "模型差异按同一折的配对计算：`右提示输出 − 左提示输出`。因此这些差异没有混入左右条件各自重新拟合造成的参数变化。第一幅图显示三组皮层 E/I 通道的右减左活动；第二幅图显示五个内部源分别对 F3、Fz、F4 差波形的贡献。代码逐折、逐电极检查源贡献之和与观测电极输出的一致性。",
        "", "### 三组皮层活动：右减左", "",
        "![三组皮层E/I群体活动的右减左差异](revision_v3/model_population_right_minus_left.png)",
        "", "### 五个源到三个电极的差异贡献", "",
        "![源到电极的右减左贡献](revision_v3/model_source_electrode_right_minus_left.png)",
        "",
        "| 电极 | 模型源代理 | 跨折平均源差 RMS | 源 RMS / 总差 RMS |",
        "|---|---|---:|---:|",
    ])
    for row in evidence["source_contribution_summary"]:
        if row["source"] != "TOTAL":
            lines.append(f"| {row['sensor']} | {row['source']} | {row['mean_record_rms']:.4g} | "
                         f"{row['mean_record_rms_ratio_to_total']:.3f} |")
    lines.extend([
        "各源带符号相加得到电极差波形；RMS 比值因源间可相消，不应解释为百分比份额。",
        "", "## 真实 EEG 的固定候选特征", "",
        "下图中的蓝点和区间是每份记录内实测右减左窗口均值及试次重采样区间，红菱形是没有用该记录拟合的交叉拟合模型预测。试次重采样区间只描述单份记录内的试次不确定性，不等同于被试总体置信区间。",
        "", "![每份记录固定窗口实测特征与模型预测](revision_v3/measured_vs_model_candidate_features.png)",
        "", "## 留出检验", "",
        "分类只检验固定的 9 维 EEG 特征在未参与训练的整份 MAT 记录上的可分性。它不证明该特征来自模型指定的神经机制；模型拟合也不替代留出分类。",
        "", "![固定9维特征的留一记录分类结果](revision_v3/heldout_9d_electrode_lda.png)",
        "", "## 波形和前端检查图", "",
        "![左右条件实测与留出模型ERP](revision_v3/heldout_erp.png)",
        "", "![模型与实测的右减左观测模式](revision_v3/heldout_difference_modes.png)",
        "", "![LGN ON/OFF时空响应](revision_v3/lgn_spatiotemporal.png)",
        "", "## 结论", "",
        f"本轮完成了可审计的左右差异分解与固定时域特征留出检验。模型内部差异可从空间/模板通道追踪到群体活动、五个源代理及 F3/Fz/F4；但留出 ERP 平均 NRMSE={mean_nrmse:.3f}，右减左波形相关均值={mean_difference_corr:.3f}，尚未显示良好的实测波形复现。",
        f"当前固定 9 维特征的留一记录宏平均平衡准确率为 {summary['macro_balanced_accuracy']:.3f}、宏平均 AUC 为 {summary['macro_roc_auc']:.3f}；两者均未超过 0.5，因此本轮**没有获得稳定左右判别的留出支持**。本轮结论应限定为：**模型预测**已给出并完成源贡献核算；**实测特征**是预先限定的探索结果；**留出验证**目前不支持该固定特征具有跨记录区分能力。",
        "因此可以报告候选机制链和候选特征，但不能将模型预测或单份记录内的差异表述成已经证实的普遍生理规律。",
        "", "## 输出文件", "",
        "新增 CSV：`model_population_right_minus_left.csv`、`model_source_electrode_right_minus_left.csv`、`model_candidate_feature_predictions.csv`、`measured_candidate_features_exploratory.csv`、`heldout_9d_electrode_lda_trial_scores.csv`、`heldout_9d_electrode_lda_folds.csv`、`heldout_9d_electrode_lda_summary.csv`、`model_source_contribution_summary.csv`、`source_additivity_audit.csv`。图像均为 PNG。",
        "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run(resolution=None, max_nfev=60, reuse_fit_artifacts=False):
    out = config.OUTPUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    cases, event_rows = load_cases_with_audit()
    fit_cases = [c for c in cases if c["eligible_fit"]]
    print(f"Loaded {len(event_rows)} MAT records; {len(fit_cases)} Stage1 fitting cases.", flush=True)
    _write_csv(out / "event_audit.csv", event_rows)
    _write_csv(out / "condition_audit.csv", [
        {"record": c["dataset"], "task": c["task"], "stage": c["stage"],
         "condition": c["condition"], "n_trials": c["n_trials"], "role": c["role"],
         "n_time_samples": len(c["time_ms"]), "first_time_ms": c["time_ms"][0],
         "last_time_ms": c["time_ms"][-1]} for c in cases])

    if resolution is None:
        print("Auditing spatial resolution (64/128 grid)...", flush=True)
        resolution_report = choose_resolution()
        resolution = int(resolution_report["chosen_resolution"])
    else:
        resolution = int(resolution)
        resolution_report = {"chosen_resolution": resolution,
                             "selection_rule": "explicit command-line resolution"}
    if resolution not in (64, 128):
        raise ValueError("resolution must be 64 or 128")
    _write_json(out / "spatial_resolution_audit.json", resolution_report)
    _write_csv(out / "spatial_kernel_audit.csv", resolution_report.get("kernel_audit", []))

    print("Auditing temporal feature stride against 1-ms frontend...", flush=True)
    selected_front = None
    selected_stride = None
    stride_rows = []
    reference_left = load_stimulus("Stage1", "left")
    reference_right = load_stimulus("Stage1", "right")
    for stride in (4.0, 2.0, 1.0):
        rows, coarse, fine = temporal_stride_audit(
            stimulus=reference_left, tau_a=config.TAU_ADAPT_DEFAULT,
            resolution=resolution, stride_ms=stride)
        mirrored_coarse = mirror_stage1_frontend(coarse, reference_right)
        scales = calibrate_drive_scales({"left": fine,
                                         "right": mirror_stage1_frontend(fine, reference_right)})
        coarse_result = simulate_forward(coarse, drive_scales=scales)
        fine_result = simulate_forward(fine, drive_scales=scales)
        eeg_denominator = float(np.sqrt(np.mean(fine_result.eeg ** 2)))
        eeg_error = float(np.sqrt(np.mean((coarse_result.eeg - fine_result.eeg) ** 2)) /
                          eeg_denominator) if eeg_denominator > 1e-12 else 0.0
        critical = [r["relative_rmse_to_1ms"] for r in rows
                    if r["feature"] in ("gabor", "B", "shape", "R_L", "R_R")]
        passed = bool(max(critical, default=0.0) <= .05 and eeg_error <= .05)
        for row in rows:
            row.update({"model_relative_eeg_rmse": eeg_error, "stride_pass": passed})
            stride_rows.append(row)
        if passed:
            selected_front, selected_stride = coarse, stride
            break
        print(f"  {stride:g} ms stride fails fixed 5% audit (EEG rel RMSE {eeg_error:.4f})", flush=True)
    if selected_front is None:
        selected_stride = 1.0
        selected_front = simulate_frontend(reference_left, params={"tau_a": config.TAU_ADAPT_DEFAULT},
                                           resolution=resolution, feature_stride_ms=1.0)
    config.FRONTEND_FEATURE_STRIDE_MS = float(selected_stride)
    register_frontend(config.TAU_ADAPT_DEFAULT, resolution, selected_front)
    _write_csv(out / "frontend_temporal_stride_audit.csv", stride_rows)
    reference_pair = {"left": selected_front,
                      "right": mirror_stage1_frontend(selected_front, reference_right)}
    drive_scales = calibrate_drive_scales(reference_pair)
    scale_rows = []
    channel_names = ("left_visual_field", "right_visual_field",
                     "left_visual_field", "right_visual_field",
                     "left_triangle_template_preference", "right_triangle_template_preference")
    for group, row_labels in zip(("early", "configuration", "shape_preference"),
                                 (channel_names[:2], channel_names[2:4], channel_names[4:])):
        for index, label in enumerate(row_labels):
            scale_rows.append({"population": group, "channel": label,
                               "reference_rms_scale": float(drive_scales[len(scale_rows)//2, index]),
                               "calibration": "fixed mirrored canonical cue; 0-200 ms; equal-condition RMS"})
    _write_csv(out / "drive_scales.csv", scale_rows)
    print(f"Using resolution={resolution}, frontend stride={selected_stride:g} ms.", flush=True)

    if reuse_fit_artifacts:
        fit_details_path = out / "heldout_fit_details.json"
        if not fit_details_path.exists():
            raise FileNotFoundError(f"cannot reuse missing fits: {fit_details_path}")
        raw_fits = json.loads(fit_details_path.read_text(encoding="utf-8"))
        fits = {record: FitResult(**payload) for record, payload in raw_fits.items()}
        if set(fits) != {case["dataset"] for case in fit_cases}:
            raise ValueError("saved fit records do not match current eligible fitting records")
        if any(fit.resolution != resolution for fit in fits.values()):
            raise ValueError("saved fit resolution does not match requested resolution")
        if any(record in fit.train_records for record, fit in fits.items()):
            raise ValueError("saved fit includes its held-out record in the training set")
        print(f"Reusing {len(fits)} just-computed leave-one-MAT fits from {fit_details_path}.",
              flush=True)
    else:
        print(f"Fitting four leave-one-MAT models (max {max_nfev} objective calls/start)...", flush=True)
        fits = fit_leave_one_record(fit_cases, resolution=resolution,
                                    feature_route="opponent", observation_rank=3,
                                    progress=True, max_nfev=max_nfev,
                                    drive_scales=drive_scales)
    fit_rows = [_fit_row(record, fit) for record, fit in fits.items()]
    _write_csv(out / "heldout_fit_summary.csv", fit_rows)
    if not reuse_fit_artifacts:
        _write_json(out / "heldout_fit_details.json", {record: fit.to_dict() for record, fit in fits.items()})

    grouped, predictions, metrics, differences, long_rows = _compute_predictions(
        cases, fits, resolution, drive_scales, progress=True)
    _write_csv(out / "heldout_metrics.csv", metrics)
    _write_csv(out / "left_right_difference.csv", differences)
    _write_csv(out / "heldout_predictions.csv", long_rows)

    print("Running fixed six-feature real-EEG validation...", flush=True)
    scores, folds, classifier_summary = _classify(cases)
    _write_csv(out / "fixed_feature_lda.csv", classifier_summary)
    _write_csv(out / "fixed_feature_lda_folds.csv", folds)
    _write_csv(out / "fixed_feature_lda_trial_scores.csv", scores)

    controls_path = out / "mechanism_controls.csv"
    fit_details_path = out / "heldout_fit_details.json"
    can_reuse_controls = (reuse_fit_artifacts and controls_path.exists()
                          and fit_details_path.exists())
    if can_reuse_controls:
        with controls_path.open("r", newline="", encoding="utf-8-sig") as stream:
            controls = list(csv.DictReader(stream))
        for row in controls:
            for key in ("full_left_right_difference_rms",
                        "control_left_right_difference_rms", "retained_fraction"):
                row[key] = float(row[key])
        if {row["heldout_record"] for row in controls} != set(fits):
            raise ValueError("saved mechanism controls do not match the current fitted records")
        print("Reusing mechanism controls generated after the current fit artifacts.", flush=True)
    else:
        print("Running frozen-parameter mechanism controls...", flush=True)
        controls = _controls(fits, resolution, drive_scales)
        _write_csv(controls_path, controls)

    print("Running fixed-window 9-D evidence chain and same-parameter L/R decomposition...",
          flush=True)
    evidence = run_evidence_chain(out, cases, fits, resolution, drive_scales)

    _plot_erps(grouped, predictions, out / "heldout_erp.png")
    _plot_differences(grouped, predictions, out / "heldout_difference_modes.png")
    representative = list(fits.values())
    median_params = {name: float(np.median([f.parameters[name] for f in representative]))
                     for name in ("tau_s", "g_i", "tau_a")}
    median_gain = float(np.median([f.amplitude for f in representative]))
    median_model_params = ModelParams.from_any(median_params)
    median_front = _cached_left_frontend(median_params["tau_a"], resolution)
    cascade = simulate_forward(median_front, params=median_model_params,
                               amplitude=median_gain, drive_scales=drive_scales)
    _plot_cascade(cascade, out / "fitted_cascade.png")
    _plot_lgn_spatiotemporal(median_front, out / "lgn_spatiotemporal.png")
    _plot_cortical_ei(cascade, out / "cortical_ei_responses.png")
    _plot_source_electrode_contributions(
        cascade, out / "source_to_electrode_contributions.png")

    lead = build_sensor_leadfield()
    _write_csv(out / "leadfield.csv", [
        {"sensor": sensor, **{f"source_{i+1}": float(value) for i, value in enumerate(row)}}
        for sensor, row in zip(config.CHANNELS, lead)])
    _write_json(out / "leadfield_geometry.json", geometry_manifest())
    _write_readme(out, resolution, selected_stride, drive_scales, fits, metrics,
                  differences, classifier_summary, controls, max_nfev, evidence)
    report_path = _write_evidence_report(out, metrics, differences, fit_rows, evidence, event_rows)

    source_files = [Path(__file__), Path(__file__).with_name("config.py"),
                    Path(__file__).with_name("frontend.py"), Path(__file__).with_name("model.py"),
                    Path(__file__).with_name("fit.py"), Path(__file__).with_name("real_data.py"),
                    Path(__file__).with_name("observation.py"), Path(__file__).with_name("head_model.py"),
                    Path(__file__).with_name("evaluate.py"), Path(__file__).with_name("evidence_chain.py")]
    manifest = {"revision": "revision_v3",
                "source_mapping_schema": config.SOURCE_MAPPING_SCHEMA,
                "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__,
                "output_root": str(out), "resolution": resolution,
                "frontend_feature_stride_ms": selected_stride,
                "fit_max_objective_calls_per_start": max_nfev,
                "fit_case_count": len(fit_cases), "mat_record_count": len(event_rows),
                "primary_model": "LGN ON/OFF -> Gabor and fixed shape/opponent templates -> three E/I population pairs -> contralateral visual-field routing plus one bilateral midline shape-opponent source -> five internal source proxies -> homogeneous infinite-conductor illustrative leadfield -> Q1 observation operator",
                "shape_preference_feedforward": {
                    "input": "early visual-field E/I activity after the fixed 33 ms delay",
                    "pool_weights_left_right": list(config.SHAPE_FEEDFORWARD_FIELD_WEIGHTS),
                    "routing": "the same pooled input is sent to both template-preference channels; template-specific drives encode left/right preference",
                    "midline_opponent_source": "explicit modeling hypothesis, not anatomical localization"},
                "diagnostic_figures": ["lgn_spatiotemporal.png",
                                       "cortical_ei_responses.png",
                                       "source_to_electrode_contributions.png",
                                       *evidence["figures"]],
                "diagnostic_figure_parameters": "generated from this run's fitted-parameter median; source contributions are checked to sum to electrode prediction",
                "fitting": "leave-one-MAT-out; fixed frontend calibration; fit Stage1 left/right only; one signed shared gain",
                "observation_operator": "0.2-24 Hz 4th-order Butterworth zero-phase; 256-to-128 Hz Fourier resample; per-channel prestimulus baseline; full [-1,3)s epoch",
                "head_model": geometry_manifest(),
                "limitations": ["four files are MAT recordings, not confirmed independent participants",
                                "no individualized MRI/head model or recorded reference metadata used by illustrative leadfield",
                                "absolute source/electrode gain not calibrated to microvolts",
                                "target event after 2.2 s not generated in cue-only fit; filter comparison uses common cue-only observation operator",
                                "model ERP fit and real-only LDA answer different questions"],
                "evidence_chain": evidence,
                "evidence_report": str(report_path),
                "sha256": {path.name: _sha256(path) for path in source_files if path.exists()}}
    _write_json(out / "manifest.json", manifest)

    print("\n=== revision_v3 results ===", flush=True)
    print(f"mean held-out ERP NRMSE: {np.nanmean([r['nrmse'] for r in metrics]):.4f}", flush=True)
    print(f"mean held-out right-minus-left waveform correlation: "
          f"{np.nanmean([r['left_right_difference_correlation'] for r in differences]):.4f}", flush=True)
    print(f"6-feature LO-MAT LDA BA/AUC: {classifier_summary[0]['balanced_accuracy_mean']:.4f} / "
          f"{classifier_summary[0]['roc_auc_mean']:.4f}", flush=True)
    candidate = evidence["classification_summary"]
    print(f"9-feature fixed-window LO-MAT LDA BA/AUC: "
          f"{candidate['macro_balanced_accuracy']:.4f} / {candidate['macro_roc_auc']:.4f}", flush=True)
    for row in fit_rows:
        print(f"{row['heldout_record']}: status={row['fit_status']}, loss={row['training_loss']:.5g}, "
              f"tau_s={row['tau_s_ms']:.2f}, g_i={row['g_i']:.3f}, tau_a={row['tau_a_ms']:.1f}", flush=True)
    print(f"Saved results: {out}", flush=True)
    print(f"Saved evidence report: {report_path}", flush=True)
    return {"fits": fit_rows, "metrics": metrics, "differences": differences,
            "classifier": classifier_summary, "evidence_chain": evidence,
            "controls": controls, "output": out}


if __name__ == "__main__":
    run()
