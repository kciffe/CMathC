"""Quantify the ERP-window effect of v3's hard zero padding after 800 ms."""
import csv
import json
from pathlib import Path

import numpy as np

try:
    from . import config
    from .fit import register_frontend
    from .frontend import load_stimulus, mirror_stage1_frontend, simulate_frontend
    from .model import ModelParams, simulate_forward
    from .observation import model_curve_to_q1_grid
except ImportError:
    import config
    from fit import register_frontend
    from frontend import load_stimulus, mirror_stage1_frontend, simulate_frontend
    from model import ModelParams, simulate_forward
    from observation import model_curve_to_q1_grid


def _corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_fit_data():
    out = config.Q2_ROOT / "output" / "revision_v3"
    config.require_current_source_mapping_manifest(out / "manifest.json")
    fits = json.loads((out / "heldout_fit_details.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((out / "drive_scales.csv").open(encoding="utf-8-sig")))
    scales = np.asarray([float(r["reference_rms_scale"]) for r in rows], dtype=np.float32).reshape(3, 2)
    return fits, scales


def run():
    out = config.Q2_ROOT / "output" / "revision_v3_validation"
    out.mkdir(parents=True, exist_ok=True)
    fits, scales = _load_fit_data()
    right_stimulus = load_stimulus("Stage1", "right")
    fronts = {}
    time_ms = np.arange(0.0, 3001.0, 1.0)
    records = []
    for record, fit in fits.items():
        params = ModelParams.from_any(fit["parameters"])
        tau_a = float(params.tau_a)
        if tau_a not in fronts:
            print(f"Generating causal front end for tau_a={tau_a:g} ms through 3000 ms ...", flush=True)
            left = simulate_frontend(load_stimulus("Stage1", "left"),
                                     params={"tau_a": tau_a}, resolution=128,
                                     time_ms=time_ms, feature_stride_ms=4.0)
            right = mirror_stage1_frontend(left, right_stimulus)
            fronts[tau_a] = {"left": left, "right": right}
            register_frontend(tau_a, 128, left)
        for condition in ("left", "right"):
            result = simulate_forward(fronts[tau_a][condition], params=params,
                                      time_ms=time_ms, drive_scales=scales)
            complete, observed_time = model_curve_to_q1_grid(result.eeg_scaled, time_ms)
            truncated_signal = result.eeg_scaled.copy()
            truncated_signal[:, time_ms > 800.0] = 0.0
            truncated, truncated_time = model_curve_to_q1_grid(truncated_signal, time_ms)
            if not np.allclose(observed_time, truncated_time):
                raise RuntimeError("full-tail and truncated observation grids differ")
            win = (observed_time >= 0.0) & (observed_time <= 796.875)
            a, b = complete[:, win], truncated[:, win]
            delta = a - b
            denom = float(np.sqrt(np.mean(a * a)))
            records.append({"record": record, "condition": condition,
                            "tau_a_ms": tau_a, "tau_s_ms": params.tau_s,
                            "n_grid_samples_0_800": int(win.sum()),
                            "full_tail_rms": denom,
                            "tail_vs_zero_pad_rmse_0_800": float(np.sqrt(np.mean(delta * delta))),
                            "tail_vs_zero_pad_relative_rmse_0_800": float(np.sqrt(np.mean(delta * delta)) /
                                                                              max(denom, 1e-12)),
                            "full_vs_truncated_correlation_0_800": _corr(a, b),
                            "interpretation": "isolates model-tail truncation; does not add the target at 2.2 s"})
        print(f"  completed {record}", flush=True)
    path = out / "observation_tail_audit.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    rel = np.asarray([r["tail_vs_zero_pad_relative_rmse_0_800"] for r in records])
    text = ("# 观测时间尾部审计\n\n"
            "将 v3 cue-only 神经状态从提示出现模拟到 3000 ms，并对比：\n\n"
            "- 完整保留 800 ms 之后的状态响应，再执行第一问同款滤波、降采样与基线校正；\n"
            "- 将 800 ms 之后的神经输出强制置零，再执行相同处理。\n\n"
            f"八个记录×条件的 0–800 ms 相对波形 RMSE 中位数为 **{np.median(rel):.4f}**，"
            f"范围 **{np.min(rel):.4f}–{np.max(rel):.4f}**。这是只针对硬截断的隔离比较。\n\n"
            "限制：两种曲线都没有提示前圆圈的神经状态，也没有加入 2.2 s 目标事件；本结果不能替代完整场景序列模拟或真实完整试次预测。详见 `observation_tail_audit.csv`。\n")
    (out / "观测尾部审计.md").write_text(text, encoding="utf-8")
    print(f"Saved tail audit: median relative RMSE={np.median(rel):.4f}, range={np.min(rel):.4f}-{np.max(rel):.4f}",
          flush=True)


if __name__ == "__main__":
    run()
