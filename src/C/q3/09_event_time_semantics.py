"""Re-audit the raw cue/action timing without choosing an RT anchor.

The official appendix gives an approximate target schedule (cue ends, then
about two seconds elapse), while the raw data have no independent target-onset
channel. This script therefore reports marker-to-schedule differences, never
labels them reaction times, and leaves correctness/omission unknown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from common import detect_cues, detect_response_events, load_raw_record
from config import OUTPUT_DIR, RECORDS


AUDIT_DIR = OUTPUT_DIR / "continuation_audit"
FIGURE_DIR = AUDIT_DIR / "figures"
Q2_LEADFIELD_PATH = OUTPUT_DIR.parent.parent / "q2" / "output" / "revision_v3" / "leadfield.csv"
SCHEDULE_WAIT_AFTER_CUE_OFFSET_S = 2.0
ALTERNATE_CUE_ONSET_REFERENCE_S = 2.0
NOMINAL_CUE_ONSET_PLUS_SCHEDULE_S = 2.2


def _active_runs(signal: np.ndarray, timestamps: np.ndarray) -> list[tuple[int, int, float, float]]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    time = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    active = np.isfinite(values) & (values != 0)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    dt = float(np.median(np.diff(time)))
    return [
        (
            int(start),
            int(end),
            float(time[start]),
            float(time[end] + dt),
        )
        for start, end in zip(starts, ends)
    ]


def _build_trial_rows(record: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw = load_raw_record(record)
    time = np.asarray(raw["channels"]["TimeStamp"], dtype=np.float64).reshape(-1)
    cue_signal = np.asarray(raw["channels"]["VisCue"], dtype=np.float64).reshape(-1)
    response_raw = np.asarray(raw["response_signal"], dtype=np.float64).reshape(-1)
    sample_rate = float(raw["sample_rate_hz"])
    dt = float(np.median(np.diff(time)))

    cues = detect_cues(cue_signal, time)
    responses = detect_response_events(response_raw, time)
    cue_runs = _active_runs(cue_signal, time)
    response_runs = _active_runs(response_raw, time)
    response_run_by_start = {run[0]: run for run in response_runs}

    rows: list[dict[str, Any]] = []
    pairing_counts: list[int] = []
    for i, cue in enumerate(cues):
        trial_start = int(cue["cue_sample_index"])
        trial_end = int(cues[i + 1]["cue_sample_index"]) if i + 1 < len(cues) else len(time)
        trial_responses = [
            event for event in responses
            if trial_start <= int(event["response_sample_index"]) < trial_end
        ]
        pairing_counts.append(len(trial_responses))
        cue_run = cue_runs[i] if i < len(cue_runs) else None
        response = trial_responses[0] if trial_responses else None
        response_run = (
            response_run_by_start.get(int(response["response_sample_index"]))
            if response is not None else None
        )
        cue_offset = float(cue_run[3]) if cue_run is not None else np.nan
        target_schedule_proxy = (
            cue_offset + SCHEDULE_WAIT_AFTER_CUE_OFFSET_S
            if np.isfinite(cue_offset) else np.nan
        )
        marker_time = float(response["response_time_s"]) if response is not None else np.nan
        marker_raw = float(response["response_raw"]) if response is not None else np.nan
        marker_code = int(response["response_code"]) if response is not None else None
        choice_side = int(response["choice_side"]) if response is not None else None
        marker_duration = (
            float(response_run[3] - response_run[2]) if response_run is not None else np.nan
        )
        cue_duration = float(cue_offset - float(cue["cue_time_s"])) if np.isfinite(cue_offset) else np.nan

        rows.append(
            {
                "record": record,
                "original_trial_index": int(cue["original_trial_index"]),
                "dataset_prefix_candidate": record.split("_", maxsplit=1)[0],
                "filename_task_suffix_candidate": record.rsplit("Task-", maxsplit=1)[-1],
                "sample_rate_hz": sample_rate,
                "cue_sample_index": trial_start,
                "cue_onset_time_s": float(cue["cue_time_s"]),
                "cue_offset_estimate_s": cue_offset,
                "cue_duration_s": cue_duration,
                "cue_side": int(cue["cue_side"]),
                "channel9_response_event_count_in_cue_interval": len(trial_responses),
                "response_sample_index": int(response["response_sample_index"]) if response else None,
                "response_marker_time_s": marker_time,
                "response_raw_at_edge": marker_raw,
                "response_code_standardized": marker_code,
                "response_side": choice_side,
                "response_marker_run_duration_s": marker_duration,
                "response_minus_cue_onset_s": marker_time - float(cue["cue_time_s"]),
                "response_minus_cue_offset_s": marker_time - cue_offset,
                "target_schedule_proxy_s": target_schedule_proxy,
                "marker_minus_schedule_proxy_s": marker_time - target_schedule_proxy,
                "marker_minus_cue_plus_2p0_s_sensitivity_only": marker_time - (float(cue["cue_time_s"]) + ALTERNATE_CUE_ONSET_REFERENCE_S),
                "marker_minus_cue_plus_2p2_s_sensitivity_only": marker_time - (float(cue["cue_time_s"]) + NOMINAL_CUE_ONSET_PLUS_SCHEDULE_S),
                "target_onset_verified": False,
                "reaction_time_s": np.nan,
                "reaction_time_status": "unknown_no_independent_trial_target_onset",
                "correctness": "unknown_no_verified_trial_target_truth",
                "omission_status": "unknown_no_verified_response_deadline",
            }
        )

    timing = np.asarray([row["response_minus_cue_onset_s"] for row in rows], dtype=float)
    schedule_delta = np.asarray([row["marker_minus_schedule_proxy_s"] for row in rows], dtype=float)
    schedule_22 = np.asarray([row["marker_minus_cue_plus_2p2_s_sensitivity_only"] for row in rows], dtype=float)
    summary = {
        "record": record,
        "response_channel_label": raw["response_label"],
        "sample_rate_hz": sample_rate,
        "cue_event_count": len(cues),
        "channel9_response_edge_count": len(responses),
        "cue_intervals_with_exactly_one_channel9_edge": int(sum(v == 1 for v in pairing_counts)),
        "cue_intervals_with_no_channel9_edge": int(sum(v == 0 for v in pairing_counts)),
        "cue_intervals_with_multiple_channel9_edges": int(sum(v > 1 for v in pairing_counts)),
        "cue_duration_median_s": float(np.nanmedian([row["cue_duration_s"] for row in rows])),
        "response_minus_cue_onset_median_s": float(np.nanmedian(timing)),
        "response_minus_cue_onset_min_s": float(np.nanmin(timing)),
        "response_minus_cue_onset_max_s": float(np.nanmax(timing)),
        "response_minus_cue_onset_sd_s": float(np.nanstd(timing, ddof=1)),
        "marker_minus_approx_schedule_median_s": float(np.nanmedian(schedule_delta)),
        "marker_minus_approx_schedule_min_s": float(np.nanmin(schedule_delta)),
        "marker_minus_approx_schedule_max_s": float(np.nanmax(schedule_delta)),
        "marker_minus_cue_plus_2p2_median_s_sensitivity_only": float(np.nanmedian(schedule_22)),
        "count_marker_under_100ms_after_cue_plus_2p2_s_sensitivity_only": int(np.sum((schedule_22 >= 0) & (schedule_22 < 0.1))),
        "count_marker_under_100ms_after_approx_schedule_proxy": int(np.sum((schedule_delta >= 0) & (schedule_delta < 0.1))),
        "response_marker_run_duration_median_s": float(np.nanmedian([row["response_marker_run_duration_s"] for row in rows])),
        "response_marker_run_duration_min_s": float(np.nanmin([row["response_marker_run_duration_s"] for row in rows])),
        "response_marker_run_duration_max_s": float(np.nanmax([row["response_marker_run_duration_s"] for row in rows])),
        "actual_target_onsets_available": False,
        "verified_response_deadlines_available": False,
        "correctness_ground_truth_available": False,
        "interpretation": "The schedule proxy is cue-off plus approximately 2 s from the appendix. Marker-to-proxy differences are not reaction times.",
    }
    representative = {
        "record": record,
        "cue_signal": cue_signal,
        "response_signal": response_raw,
        "timestamps": time,
        "rows": rows,
    }
    return rows, summary, representative


def _save_timing_figure(frame: pd.DataFrame, out: Path) -> None:
    records = list(RECORDS)
    colors = ["#356b8c", "#4f8f75", "#b77c33", "#8b6e9e"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    ax = axes[0]
    rng = np.random.default_rng(37)
    for yi, (record, color) in enumerate(zip(records, colors)):
        group = frame.loc[frame["record"] == record]
        y = yi + rng.uniform(-0.16, 0.16, size=len(group))
        ax.scatter(
            group["response_minus_cue_onset_s"], y,
            s=10, alpha=0.57, color=color, edgecolors="none",
        )
    ax.axvline(2.0, color="#7a7a7a", linestyle=(0, (4, 3)), linewidth=1.1,
               label="cue onset + 2.0 s (sensitivity only)")
    ax.axvline(2.2, color="#ad4b3b", linestyle=(0, (4, 3)), linewidth=1.1,
               label="cue onset + 2.2 s (approx. schedule)")
    ax.set_yticks(range(len(records)), records)
    ax.set_ylim(-0.38, len(records) - 0.62)
    ax.set_xlim(1.99, 2.35)
    ax.set_xlabel("Channel-9 edge minus cue onset (s)")
    ax.set_ylabel("MAT record")
    ax.set_title("Observed event intervals")
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.6)

    ax = axes[1]
    for yi, (record, color) in enumerate(zip(records, colors)):
        group = frame.loc[frame["record"] == record]
        values_ms = group["marker_minus_schedule_proxy_s"].to_numpy(dtype=float) * 1000.0
        visible = values_ms <= 40.0
        ax.scatter(
            values_ms[visible],
            yi + rng.uniform(-0.16, 0.16, size=int(visible.sum())), s=10, alpha=0.57,
            color=color, edgecolors="none",
        )
        med = float(np.median(values_ms))
        ax.plot([med, med], [yi - 0.22, yi + 0.22], color="#202020", linewidth=1.4)
    ax.axvline(0, color="#ad4b3b", linestyle=(0, (4, 3)), linewidth=1.1)
    ax.set_yticks(range(len(records)), records)
    ax.set_ylim(-0.38, len(records) - 0.62)
    ax.set_xlim(-5, 40)
    ax.set_xlabel("Marker − (measured cue offset + ~2 s), ms")
    ax.set_ylabel("MAT record")
    ax.set_title("Deviation from approximate schedule")
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.6)
    outside = frame.loc[frame["marker_minus_schedule_proxy_s"] * 1000.0 > 40.0]
    if len(outside):
        detail = ", ".join(
            f"{row.record.replace('VisualCog', '')} trial {int(row.original_trial_index)}: "
            f"{row.marker_minus_schedule_proxy_s * 1000.0:.1f} ms"
            for row in outside.itertuples(index=False)
        )
        ax.text(0.98, 0.49, f"Off scale (>40 ms; point not shown): {detail}",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=7.0, color="#555555")
    fig.suptitle("Timing audit: schedule differences are not measured reaction times", fontsize=11)
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def _save_example_trace(example: dict[str, Any], out: Path) -> None:
    rows = example["rows"]
    row = rows[min(10, len(rows) - 1)]
    cue_time = float(row["cue_onset_time_s"])
    response_time = float(row["response_marker_time_s"])
    cue_offset = float(row["cue_offset_estimate_s"])
    schedule_time = cue_offset + SCHEDULE_WAIT_AFTER_CUE_OFFSET_S
    time_rel = np.asarray(example["timestamps"], dtype=float) - cue_time
    selected = (time_rel >= -0.1) & (time_rel <= 3.0)
    cue = np.asarray(example["cue_signal"], dtype=float)[selected]
    response = np.asarray(example["response_signal"], dtype=float)[selected]
    x = time_rel[selected]

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 5.4), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.8]})
    axes[0].step(x, cue, where="post", color="#356b8c", linewidth=1.1)
    axes[0].set_ylabel("VisCue code")
    axes[0].set_title(f"{example['record']} trial {row['original_trial_index']}: raw event channels")
    axes[0].grid(axis="x", color="#dddddd", linewidth=0.6)
    axes[1].step(x, response, where="post", color="#b77c33", linewidth=1.1)
    axes[1].set_ylabel("Raw channel-9 code")
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.6)
    zoom = (x >= (schedule_time - cue_time - 0.05)) & (x <= (response_time - cue_time + 0.05))
    axes[2].step(x[zoom], response[zoom], where="post", color="#b77c33", linewidth=1.4)
    axes[2].set_xlim(schedule_time - cue_time - 0.05, response_time - cue_time + 0.05)
    axes[2].set_ylabel("Zoom")
    axes[2].set_xlabel("Time from VisCue onset (s)")
    axes[2].grid(axis="x", color="#dddddd", linewidth=0.6)
    for ax in axes:
        ax.axvline(schedule_time - cue_time, color="#ad4b3b", linestyle=(0, (4, 3)), linewidth=1.2)
        ax.axvline(response_time - cue_time, color="#202020", linestyle="-", linewidth=1.0)
    axes[0].text(0.99, 0.90, "red dashed: cue offset + approx. 2 s schedule", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=7.3, color="#ad4b3b")
    axes[1].text(0.99, 0.90, "black: first channel-9 nonzero edge", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=7.3, color="#202020")
    axes[2].text(0.99, 0.90,
                 f"marker minus schedule proxy = {(response_time - schedule_time) * 1000.0:.1f} ms",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=7.3, color="#555555")
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def _build_mapping_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["cue_response_same_side"] = data["cue_side"] == data["response_side"]
    data["task_suffix_candidate"] = data["filename_task_suffix_candidate"].map(
        lambda value: f"Task-{value}"
    )
    data["prefix_candidate"] = data["dataset_prefix_candidate"]
    rows = []
    for scheme, group_column in (
        ("suffix: Task-1/Task-2", "task_suffix_candidate"),
        ("prefix: VisualCogA/VisualCogB", "prefix_candidate"),
        ("record", "record"),
    ):
        for group_name, group in data.groupby(group_column, sort=True):
            n = int(group["cue_response_same_side"].notna().sum())
            same = int(group["cue_response_same_side"].sum())
            rows.append(
                {
                    "grouping_scheme": scheme,
                    "group": group_name,
                    "n_trials": n,
                    "cue_response_same_side_count": same,
                    "cue_response_same_side_fraction": same / n if n else np.nan,
                    "interpretation": "descriptive cue-response agreement only; not correctness or task-label validation",
                }
            )
    return pd.DataFrame(rows)


def _save_task_mapping_figure(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    summary = _build_mapping_summary(frame)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    record_order = list(RECORDS)
    record_stats = summary.loc[summary["grouping_scheme"] == "record"].set_index("group")
    record_colors = ["#356b8c", "#4f8f75", "#b77c33", "#8b6e9e"]
    y = np.arange(len(record_order))
    vals = [record_stats.loc[name, "cue_response_same_side_fraction"] for name in record_order]
    axes[0].barh(y, vals, color=record_colors, height=0.62)
    axes[0].set_yticks(y, record_order)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.08)
    axes[0].set_xlabel("Fraction cue side = response side")
    axes[0].set_title("By MAT record")
    axes[0].grid(axis="x", color="#dddddd", linewidth=0.6)
    for yi, name in enumerate(record_order):
        count = int(record_stats.loc[name, "cue_response_same_side_count"])
        n = int(record_stats.loc[name, "n_trials"])
        axes[0].text(vals[yi] + 0.02, yi, f"{count}/{n}", va="center", fontsize=8)

    grouped = [
        ("Task-1 suffix", "suffix: Task-1/Task-2", "Task-1"),
        ("Task-2 suffix", "suffix: Task-1/Task-2", "Task-2"),
        ("VisualCogA prefix", "prefix: VisualCogA/VisualCogB", "VisualCogA"),
        ("VisualCogB prefix", "prefix: VisualCogA/VisualCogB", "VisualCogB"),
    ]
    labels = []
    rates = []
    counts = []
    ns = []
    for label, scheme, name in grouped:
        row = summary.loc[(summary["grouping_scheme"] == scheme) & (summary["group"] == name)].iloc[0]
        labels.append(label)
        rates.append(float(row["cue_response_same_side_fraction"]))
        counts.append(int(row["cue_response_same_side_count"]))
        ns.append(int(row["n_trials"]))
    yy = np.arange(len(labels))
    colors = ["#356b8c", "#356b8c", "#b77c33", "#b77c33"]
    axes[1].barh(yy, rates, color=colors, height=0.62)
    axes[1].set_yticks(yy, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.12)
    axes[1].set_xlabel("Fraction cue side = response side")
    axes[1].set_title("Grouping-candidate contrast")
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.6)
    for yi, (rate, count, n) in enumerate(zip(rates, counts, ns)):
        axes[1].text(rate + 0.02, yi, f"{count}/{n}", va="center", fontsize=8)
    fig.suptitle("Cue-response agreement is descriptive, not correctness", fontsize=10.5)
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    return summary


def _audit_q2_leadfield() -> dict[str, Any]:
    if not Q2_LEADFIELD_PATH.exists():
        return {"status": "missing_q2_leadfield", "path": str(Q2_LEADFIELD_PATH)}
    table = pd.read_csv(Q2_LEADFIELD_PATH)
    source_columns = [column for column in table.columns if column.startswith("source_")]
    matrix = table[source_columns].to_numpy(dtype=float)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    return {
        "status": "computed_from_saved_q2_revision_v3_leadfield",
        "path": str(Q2_LEADFIELD_PATH),
        "sensor_rows": int(matrix.shape[0]),
        "source_columns": int(matrix.shape[1]),
        "matrix_rank": rank,
        "source_nullity": int(matrix.shape[1] - rank),
        "singular_values": singular_values.tolist(),
        "condition_number_nonzero_subspace": float(singular_values[0] / singular_values[-1]),
        "interpretation": "At least two linearly independent source changes lie in the null space; the five Q2 source proxies cannot be uniquely recovered from three electrodes without additional constraints.",
    }


def _save_q2_leadfield_figure(audit: dict[str, Any], out: Path) -> None:
    if audit.get("status") != "computed_from_saved_q2_revision_v3_leadfield":
        return
    table = pd.read_csv(Q2_LEADFIELD_PATH)
    source_columns = [column for column in table.columns if column.startswith("source_")]
    matrix = table[source_columns].to_numpy(dtype=float)
    source_labels = [
        "early visual\nLH from right field",
        "early visual\nRH from left field",
        "configuration\nLH from right field",
        "configuration\nRH from left field",
        "bilateral shape\nopponent",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.5), constrained_layout=True)
    vmax = float(np.max(np.abs(matrix)))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(source_labels)), source_labels)
    ax.set_yticks(range(len(table)), table["sensor"])
    ax.set_xlabel("Q2 canonical source proxies")
    ax.set_ylabel("Scalp sensor")
    ax.set_title("Q2 forward matrix has rank 3 for 5 source proxies")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#202020")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Leadfield coefficient (saved units)")
    fig.text(0.5, -0.025, "Source nullity = 5 − rank(3×5 matrix) = 2; not uniquely invertible.",
             ha="center", fontsize=8.5, color="#444444")
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def _write_report(frame: pd.DataFrame, summaries: pd.DataFrame, report_path: Path) -> None:
    n = len(frame)
    one_to_one = int((frame["channel9_response_event_count_in_cue_interval"] == 1).sum())
    cue_dur = float(frame["cue_duration_s"].median())
    event_from_cue = float(frame["response_minus_cue_onset_s"].median())
    event_from_schedule = float(frame["marker_minus_schedule_proxy_s"].median())
    event_after_22 = int(
        ((frame["marker_minus_cue_plus_2p2_s_sensitivity_only"] >= 0)
         & (frame["marker_minus_cue_plus_2p2_s_sensitivity_only"] < 0.1)).sum()
    )
    alt_median = float(frame["marker_minus_cue_plus_2p0_s_sensitivity_only"].median())
    mapping_summary = _build_mapping_summary(frame)
    mapping_md = mapping_summary.to_markdown(index=False, floatfmt=".3f")
    q2_mapping = _audit_q2_leadfield()
    if q2_mapping.get("status") == "computed_from_saved_q2_revision_v3_leadfield":
        q2_rank_text = (
            f"Q2 保存导联矩阵维数为 {q2_mapping['sensor_rows']}×{q2_mapping['source_columns']}，"
            f"数值秩为 {q2_mapping['matrix_rank']}，因此源空间零空间维数为 "
            f"{q2_mapping['source_nullity']}。"
        )
    else:
        q2_rank_text = "未找到 Q2 保存导联矩阵，当前无法复核其源到电极矩阵秩。"
    summary_md = summaries[[
        "record", "response_channel_label", "cue_event_count", "channel9_response_edge_count",
        "cue_intervals_with_exactly_one_channel9_edge", "cue_duration_median_s",
        "response_minus_cue_onset_median_s", "marker_minus_approx_schedule_median_s",
        "response_marker_run_duration_median_s",
    ]].to_markdown(index=False, floatfmt=".4f")
    text = f"""# 问题三：事件语义与反应时间锚点审计

## 审计结论

原始数据中检出 {n} 个 VisCue 起始边沿和 {n} 个通道9 非零起始边沿；{one_to_one}/{n} 个 cue 区间恰有一个通道9边沿。cue 非零段的中位持续时间为 {cue_dur:.4f} s。跨记录配对没有显示出逐试次索引错位，但这只能核实数据内的事件配对，不能证明通道9时钟边沿就是实际鼠标动作的精确发生时刻。

通道9边沿距 cue 起始的总体中位数为 {event_from_cue:.4f} s。根据题目附录“提示消失后等待约2秒”的文字，以实测 cue 结束时刻加约2秒构造一个**计划时刻代理**，边沿相对该代理的中位差为 {event_from_schedule*1000:.1f} ms。此差值不是反应时：逐试次真实目标呈现时刻没有单独事件标记，且“约2秒”不是精确呈现日志。

以 cue+2.2 s 作敏感性参照时，{event_after_22}/{n} 个边沿位于其后100 ms以内；以 cue+2.0 s 作另一个敏感性参照时，边沿差的中位数为 {alt_median:.4f} s。两者均只展示锚点敏感性，**本审计不选取任一者作为真实 RT 起点**。

## 证据与待定解释

1. **目标呈现时刻：未被逐试次观测。** 附录支持“cue 结束后约2秒”的粗略计划；VisCue脉冲实测约0.20秒，因此计划代理约为 cue onset +2.20秒。原始通道没有 target-onset 事件，不能据此给每次试验填入精确 onset。
2. **通道9语义：书面定义为目标应答/行动，实际时序仍需核验。** 官方题面将通道9定义为点击左/右目标的应答；但原始边沿在四份记录中高度集中于上述计划时刻附近，且通道9为持续非零段而非单采样脉冲。这种吻合既不能推翻官方定义，也不能单凭文件证明其边沿是鼠标首次动作。应答边沿与目标时刻的关系仍有语义/时间戳冲突。
3. **试次配对：当前未见错位证据。** 按每个 cue 到下一 cue 的区间检查，所有记录都是一段一个通道9边沿；故逐试次顺序错位暂不支持为主要解释。跨通道同步偏移或记录程序对通道9的写入语义，仍需要原始实验日志/软件定义才能排除。

## Task 类型映射冲突

《第三.pdf》把项目1定义为位置提示、项目2定义为形状提示，但没有在文本中给四份 MAT 文件写出机器可核对的逐文件项目映射。当前 Q3 将 `Task-1/Task-2` 后缀作为候选类型；Q2 的 `revision_v3/config.py` 则把 `VisualCogA_*` 两份文件归为 Task1、`VisualCogB_*` 两份归为 Task2。两套分组给出不同的 cue-应答同侧结构：

{mapping_md}

按 `Task-1/Task-2` 后缀分组的模式与位置提示/形状提示的行为预期相容；但同侧不等于正确，也不能用它来反向证明文件映射。按 `VisualCogA/B` 前缀分组则没有区分度。由于映射来源冲突且缺少实验记录表，**项目类型仍标为候选映射**；当前 OLS 中的任务项不作项目1/项目2效应解释。后续动态模型先不依赖该任务变量，并同时保存两种候选分组供敏感性审查。

## 标签规则

- `response_side`：由通道9起始边沿的符号确定，并统一 `-1/-2 -> -2`、`+1/+2 -> +2`；保留原始边沿码。
- `reaction_time_s`：本轮保留为空，状态为 `unknown_no_independent_trial_target_onset`。
- `correctness`：本轮保留为未知；没有经过验证的逐试次目标真值与无冲突的任务类型映射。cue/response 同侧比例只是选择一致性，不等于正确率。
- `omission_status`：本轮保留为未知；没有经过验证的反应截止时刻。通道9有边沿不等于已证明不存在漏答/迟答。

## 当前可用的时间窗

- VisCue 锁定 EEG：相对已观测 cue 起始的时间窗可复现。
- `cue+2.2 s` 锁定 EEG：只作为计划时刻敏感性分支，不标成真实 target-locked ERP。
- 通道9边沿前约100 ms：可以抽取并命名为“channel-9-marker-preceding EEG”；在事件语义冲突澄清前，不将其解释为已验证的动作前生理窗口。
- cue 至 marker 前100 ms 的逐时轨迹：可以按记录的事件码建立描述性轨迹；不能把轨迹中的阶段直接命名为目标出现后的记忆匹配阶段，因为 target onset 未被独立记录。

## 动态认知模型与数据处理的边界

当前 `V/H/P` 是 ERP、theta 与 beta 特征的聚合代理加描述性 OLS；它们不是经状态转移方程估计的潜变量，也不证明视觉区、海马或前额叶来源。Q2 的机制链可为 Q3 提供正向观测形式 `y(t) = G q(t) + ε(t)`：视觉输入经 LGN/皮层群体动力学形成源代理，再由导联矩阵映射到 F3/Fz/F4。{q2_rank_text}因此 Q3 可以复用其**机制结构与前向映射思想**，但不能把五源当成从三通道 EEG 唯一反演出的真值；再增加海马/PFC源后，逆问题更欠定。

现有 Q3 特征管线使用 raw 256 Hz、ERP 0.5–30 Hz 与时频 1–80 Hz 两套滤波；Q2 对照链采用 Q1 的 0.2–24 Hz、256→128 Hz 和逐试次基线校正。新动态分析需先冻结一个共同采样率、滤波/相位处理、基线、质量排除和记录级验证合同，并保证模型预测和实测 EEG 经同一观测处理。由于零相位滤波会在事件两侧扩散波形，若用它分析亚百毫秒级阶段顺序，必须把滤波影响纳入解释限制。

## 下一步

1. 找到实验程序/原始行为日志或目标显示触发记录，核实 target onset、response marker 的定义和共同时间基准；在此之前不拟合真实 RT、正确率、漏答率或 DDM。
2. 先建立不声称解剖定位的时间域基线：按 cue 与通道9边沿索引 F3/Fz/F4 连续轨迹，做记录级留出预测/重构；预注册窗口和误差指标。
3. 再把 Q2 的 LGN→Wilson–Cowan→源→导联结构接入状态空间：用可观测的 cue 驱动视觉子模块，把未观测 target/memory match 明确记为潜输入；只有外部真值或模型可识别性检验通过后，才拟合 H/P 转移与额外源权重。
4. 每次模型或处理口径变更后，重新生成同一组审计图与留出诊断图，记录数据版本、窗口、参数、单位和未知标签数。

## 逐记录原始证据

{summary_md}

## 图件

![问题二导联矩阵的秩与源空间零空间](figures/q2_leadfield_rank_audit.png)

![逐试次事件间隔和计划时刻差值](figures/event_timing_anchor_audit.png)

![原始 VisCue 与通道9 事件波形](figures/event_channel_trace_example.png)

![两种候选任务分组下的提示-应答同侧比例](figures/task_mapping_candidate_audit.png)

图中 cue+2.0 s / cue+2.2 s 仅为敏感性参照；cue-off+约2秒是由题目附录构造的计划时刻代理。任何垂直参考线都不代表已观测的真实目标起始。
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    for record in RECORDS:
        rows, summary, details = _build_trial_rows(record)
        trial_rows.extend(rows)
        summary_rows.append(summary)
        if record == RECORDS[0]:
            example = details
    trials = pd.DataFrame(trial_rows)
    summaries = pd.DataFrame(summary_rows)
    trials.to_csv(AUDIT_DIR / "event_timing_by_trial.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(AUDIT_DIR / "event_timing_by_record.csv", index=False, encoding="utf-8-sig")

    overall = {
        "n_trials": int(len(trials)),
        "n_exactly_one_response_edge_in_cue_interval": int((trials["channel9_response_event_count_in_cue_interval"] == 1).sum()),
        "response_marker_minus_cue_onset_median_s": float(trials["response_minus_cue_onset_s"].median()),
        "response_marker_minus_schedule_proxy_median_s": float(trials["marker_minus_schedule_proxy_s"].median()),
        "response_marker_minus_cue_plus_2p2_median_s_sensitivity_only": float(trials["marker_minus_cue_plus_2p2_s_sensitivity_only"].median()),
        "n_response_markers_under_100ms_after_cue_plus_2p2_s_sensitivity_only": int(((trials["marker_minus_cue_plus_2p2_s_sensitivity_only"] >= 0) & (trials["marker_minus_cue_plus_2p2_s_sensitivity_only"] < 0.1)).sum()),
        "n_verified_reaction_times": 0,
        "n_verified_correctness_labels": 0,
        "n_verified_omission_labels": 0,
        "target_onset_status": "unknown; no independent per-trial marker in supplied allowed channels",
        "response_marker_semantics_status": "officially described as target response/action; sample timing is inconsistent with a precise reaction-time interpretation under the approximate schedule",
        "trial_pairing_status": "one channel-9 event edge in each VisCue-to-next-VisCue interval for all supplied records",
        "anchor_selection": "none; cue+2.0 and cue+2.2 are sensitivity references only",
        "task_mapping_status": "conflict: Q3 used the Task-1/Task-2 suffix as a candidate, while Q2 revision_v3/config.py groups Task1/Task2 by VisualCogA/VisualCogB prefix; task mapping remains candidate-only",
    }
    (AUDIT_DIR / "event_timing_audit_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _save_timing_figure(trials, FIGURE_DIR / "event_timing_anchor_audit")
    if example is not None:
        _save_example_trace(example, FIGURE_DIR / "event_channel_trace_example")
    mapping_summary = _save_task_mapping_figure(
        trials, FIGURE_DIR / "task_mapping_candidate_audit"
    )
    mapping_summary.to_csv(
        AUDIT_DIR / "task_mapping_candidate_audit.csv", index=False, encoding="utf-8-sig"
    )
    q2_mapping = _audit_q2_leadfield()
    (AUDIT_DIR / "q2_source_mapping_identifiability.json").write_text(
        json.dumps(q2_mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _save_q2_leadfield_figure(q2_mapping, FIGURE_DIR / "q2_leadfield_rank_audit")
    for stem in (
        "event_timing_anchor_audit",
        "event_channel_trace_example",
        "task_mapping_candidate_audit",
        "q2_leadfield_rank_audit",
    ):
        for suffix in (".svg", ".figure.json"):
            stale_path = FIGURE_DIR / f"{stem}{suffix}"
            if stale_path.exists():
                stale_path.unlink()
    _write_report(trials, summaries, AUDIT_DIR / "event_timing_semantics_report.md")
    print(f"Audited {len(trials)} trials; exactly-one-edge pairing: {overall['n_exactly_one_response_edge_in_cue_interval']}; verified RT labels: 0.")
    print(f"Wrote evidence to {AUDIT_DIR}")


if __name__ == "__main__":
    main()
