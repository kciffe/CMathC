import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).resolve().parent
REAL_ROOT = SCRIPT_DIR.parent / "q1" / "output" / "8riemann_denoise"
SIM_ROOT = SCRIPT_DIR / "output" / "04_simulated_eeg"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "05_eeg_compare"

DATASETS = {
    "Task1": ("VisualCogA_Task-1", "VisualCogA_Task-2"),
    "Task2": ("VisualCogB_Task-1", "VisualCogB_Task-2"),
}
CHANNELS = ("F3", "Fz", "F4")
EVENT_CHANNEL = 8
TARGET_ONSET_S = 2.20
MODEL_TIME_MS = np.arange(801, dtype=float)
PLOT_WINDOW_MS = (0, 800)
STAGE_WINDOWS_MS = {
    # Stage1 has cue onset at 0 ms and cue offset at 200 ms. Include both
    # events in the full comparison and inspect late activity after offset.
    "Stage1": {"compare": (0, 800), "late": (450, 700)},
    # Stage2 has a single target onset at 0 ms relative time.
    "Stage2": {"compare": (0, 500), "late": (250, 500)},
}
MAX_LAG_MS = 100

CONDITION_LABELS = {
    "left": "左提示",
    "right": "右提示",
    "dots": "双圆点",
    "inward": "相向",
    "outward": "背向",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def unwrap_matlab_string(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_dataset(dataset):
    path = REAL_ROOT / f"{dataset}_clean.mat"
    mat = loadmat(path)
    trials = np.asarray(mat["trial_data"], dtype=float)
    relative_time = np.asarray(mat["relative_time"], dtype=float)
    cue_type = np.asarray(mat["cue_type"], dtype=float).reshape(-1)
    labels = [
        unwrap_matlab_string(value)
        for value in np.asarray(mat["DataLabel"], dtype=object).reshape(-1)
    ]

    if trials.ndim != 3:
        raise ValueError(f"{dataset}: trial_data must have Trial x Channel x Time shape")
    if relative_time.shape != (trials.shape[0], trials.shape[2]):
        raise ValueError(f"{dataset}: relative_time shape does not match trial_data")
    if cue_type.size != trials.shape[0]:
        raise ValueError(f"{dataset}: cue_type length does not match trial count")
    if len(labels) <= EVENT_CHANNEL or trials.shape[1] <= EVENT_CHANNEL:
        raise ValueError(f"{dataset}: DataLabel or trial_data lacks event channel 9")
    if not np.allclose(relative_time, relative_time[:1], rtol=0.0, atol=1e-12):
        raise ValueError(f"{dataset}: trials do not share a common time axis")

    channel_indices = {}
    for channel in CHANNELS:
        try:
            channel_indices[channel] = labels.index(channel)
        except ValueError as error:
            raise ValueError(f"{dataset}: DataLabel does not contain {channel}") from error

    eeg = trials[
        :,
        [channel_indices[channel] for channel in CHANNELS],
        :,
    ]
    time_s = relative_time[0]
    event_label = labels[EVENT_CHANNEL]

    return {
        "eeg": eeg,
        "time_s": time_s,
        "cue_type": cue_type,
        "event_values": trials[:, EVENT_CHANNEL, :],
        "event_label": event_label,
        "labels": labels,
    }


def baseline_correct(eeg, time_s, window_s):
    mask = (time_s >= window_s[0]) & (time_s < window_s[1])
    if not np.any(mask):
        raise ValueError(f"baseline window {window_s} contains no real EEG samples")
    baseline = np.nanmean(eeg[:, :, mask], axis=2, keepdims=True)
    return eeg - baseline


def first_post_target_events(event_values, time_s):
    event_values = np.asarray(event_values, dtype=float)
    result = np.full(event_values.shape[0], np.nan, dtype=float)
    event_times = np.full(event_values.shape[0], np.nan, dtype=float)
    event_counts = np.zeros(event_values.shape[0], dtype=int)

    for trial_index, values in enumerate(event_values):
        starts = np.flatnonzero(
            (values != 0) & np.r_[True, values[:-1] == 0]
        )
        starts = starts[time_s[starts] >= TARGET_ONSET_S]
        event_counts[trial_index] = starts.size
        if starts.size:
            first = starts[0]
            result[trial_index] = values[first]
            event_times[trial_index] = time_s[first]

    return result, event_times, event_counts


def infer_layout(cue_type, event_value):
    product = cue_type * np.sign(event_value)
    layout = np.full(len(product), "", dtype=object)
    layout[product < 0] = "inward"
    layout[product > 0] = "outward"
    return layout


def stage_conditions(task, stage, data):
    cue_type = data["cue_type"]
    total_trials = len(cue_type)
    metadata = {
        "total_trials": total_trials,
        "unclassified_trials": 0,
        "event_label": data["event_label"],
        "mapping_rule": "",
    }

    if stage == "Stage1":
        masks = {
            "left": cue_type == -1,
            "right": cue_type == 1,
        }
        metadata["unclassified_trials"] = int(
            np.count_nonzero(~np.isin(cue_type, (-1, 1)))
        )
        metadata["mapping_rule"] = "cue_type: -1=left, +1=right"
        return masks, metadata

    if task == "Task1":
        masks = {"dots": np.ones(total_trials, dtype=bool)}
        metadata["mapping_rule"] = "all trials averaged as the double-dot condition"
        return masks, metadata

    event_label = data["event_label"]
    if not (
        event_label.startswith(("Action:", "TgtAct:"))
        and "L-" in event_label
        and "R+" in event_label
    ):
        raise ValueError(
            f"{event_label!r} does not declare the expected left-negative/right-positive marker"
        )

    event_value, event_time_s, event_counts = first_post_target_events(
        data["event_values"],
        data["time_s"],
    )
    valid = np.isfinite(event_value) & np.isin(cue_type, (-1, 1))
    layout = np.full(total_trials, "", dtype=object)
    layout[valid] = infer_layout(cue_type[valid], event_value[valid])
    masks = {
        "inward": layout == "inward",
        "outward": layout == "outward",
    }

    metadata["unclassified_trials"] = int(np.count_nonzero(~valid))
    metadata["event_label"] = event_label
    metadata["mapping_rule"] = (
        "within this MAT only: cue_type x first post-target event sign; "
        "negative=inward, positive=outward"
    )
    metadata["event_trials_without_marker"] = int(
        np.count_nonzero(~np.isfinite(event_value))
    )
    metadata["trials_with_multiple_post_target_events"] = int(
        np.count_nonzero(event_counts > 1)
    )
    finite_times = event_time_s[np.isfinite(event_time_s)]
    metadata["first_event_median_time_ms"] = (
        float((np.median(finite_times) - TARGET_ONSET_S) * 1000)
        if finite_times.size
        else ""
    )
    return masks, metadata


def load_simulation(task, stage, condition):
    signal = np.asarray(
        np.load(SIM_ROOT / task / stage / f"{condition}_eeg.npy"),
        dtype=float,
    )
    if signal.shape != (3, MODEL_TIME_MS.size):
        raise ValueError(
            f"{task}/{stage}/{condition}: expected 3 x 801 simulated EEG, got {signal.shape}"
        )
    return signal


def sample_simulation_on_real_grid(signal, relative_time_ms):
    return np.vstack(
        [
            np.interp(relative_time_ms, MODEL_TIME_MS, signal[channel])
            for channel in range(len(CHANNELS))
        ]
    )


def pearson_correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def best_lag_correlation(sim, real, time_ms, compare_window):
    compare = (
        (time_ms >= compare_window[0])
        & (time_ms <= compare_window[1])
    )
    indices = np.flatnonzero(compare)
    if indices.size < 3:
        return float("nan"), float("nan"), False

    sample_ms = float(np.median(np.diff(time_ms[indices])))
    if sample_ms <= 0:
        return float("nan"), float("nan"), False
    max_lag_samples = int(np.floor(MAX_LAG_MS / sample_ms))
    best_corr = float("-inf")
    best_lag = 0

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            left = indices[: indices.size - lag] if lag else indices
            right = indices[lag:]
        else:
            shift = -lag
            left = indices[shift:]
            right = indices[: indices.size - shift]
        if left.size < 3:
            continue
        value = pearson_correlation(sim[left], real[right])
        if np.isfinite(value) and (
            value > best_corr
            or (np.isclose(value, best_corr) and abs(lag) < abs(best_lag))
        ):
            best_corr = value
            best_lag = lag

    if not np.isfinite(best_corr):
        return float("nan"), float("nan"), False
    # Positive lag means the real ERP is later than the simulated response.
    return (
        float(best_corr),
        float(best_lag * sample_ms),
        abs(best_lag) == max_lag_samples,
    )


def positive_late_peak(signal, time_ms, late_window):
    indices = np.flatnonzero(
        (time_ms >= late_window[0]) & (time_ms <= late_window[1])
    )
    if indices.size == 0:
        return float("nan"), float("nan"), False
    local_index = int(np.argmax(signal[indices]))
    index = int(indices[local_index])
    boundary = local_index in (0, indices.size - 1)
    return float(signal[index]), float(time_ms[index]), bool(boundary)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def common_scale_late_mean(signal, time_ms, late_window):
    plot_mask = (
        (time_ms >= PLOT_WINDOW_MS[0])
        & (time_ms <= PLOT_WINDOW_MS[1])
    )
    late_mask = (
        (time_ms >= late_window[0])
        & (time_ms <= late_window[1])
    )
    scale = float(np.max(np.abs(signal[:, plot_mask]))) if np.any(plot_mask) else 0.0
    if scale == 0 or not np.any(late_mask):
        return np.full(len(CHANNELS), np.nan, dtype=float)
    return signal[:, late_mask].mean(axis=1) / scale


def channel_metric_fields(row, sim, real, time_ms, compare_window, late_window):
    compare_mask = (
        (time_ms >= compare_window[0])
        & (time_ms <= compare_window[1])
    )
    sim_common = common_scale_late_mean(sim, time_ms, late_window)
    real_common = common_scale_late_mean(real, time_ms, late_window)
    correlations = []
    lag_correlations = []

    for channel_index, channel in enumerate(CHANNELS):
        prefix = channel.lower()
        zero_lag = pearson_correlation(
            sim[channel_index, compare_mask],
            real[channel_index, compare_mask],
        )
        best_corr, best_lag_ms, best_lag_at_limit = best_lag_correlation(
            sim[channel_index],
            real[channel_index],
            time_ms,
            compare_window,
        )
        sim_peak, sim_peak_time, sim_boundary = positive_late_peak(
            sim[channel_index], time_ms, late_window
        )
        real_peak, real_peak_time, real_boundary = positive_late_peak(
            real[channel_index], time_ms, late_window
        )

        row[f"{prefix}_zero_lag_r"] = zero_lag
        row[f"{prefix}_best_lag_r"] = best_corr
        row[f"{prefix}_best_lag_real_delay_ms"] = best_lag_ms
        row[f"{prefix}_best_lag_at_search_limit"] = best_lag_at_limit
        row[f"{prefix}_sim_late_positive_peak"] = sim_peak
        row[f"{prefix}_real_late_positive_peak"] = real_peak
        row[f"{prefix}_sim_peak_time_ms"] = sim_peak_time
        row[f"{prefix}_real_peak_time_ms"] = real_peak_time
        row[f"{prefix}_sim_peak_at_window_boundary"] = sim_boundary
        row[f"{prefix}_peak_time_difference_ms"] = (
            real_peak_time - sim_peak_time
            if np.isfinite(real_peak_time) and np.isfinite(sim_peak_time)
            else float("nan")
        )
        row[f"{prefix}_real_peak_at_window_boundary"] = real_boundary
        row[f"{prefix}_sim_late_mean_common_scale"] = sim_common[channel_index]
        row[f"{prefix}_real_late_mean_common_scale"] = real_common[channel_index]
        correlations.append(zero_lag)
        lag_correlations.append(best_corr)

    sim_topography = common_scale_late_mean(sim, time_ms, late_window)
    real_topography = common_scale_late_mean(real, time_ms, late_window)
    row["mean_zero_lag_r"] = float(np.nanmean(correlations))
    row["mean_best_lag_r"] = float(np.nanmean(lag_correlations))
    row["late_topography_cosine"] = cosine_similarity(
        sim_topography,
        real_topography,
    )
    row["sim_late_f4_minus_f3_common_scale"] = float(
        sim_topography[2] - sim_topography[0]
    )
    row["real_late_f4_minus_f3_common_scale"] = float(
        real_topography[2] - real_topography[0]
    )
    return row


def comparison_row(
    record_type,
    dataset,
    stage,
    condition,
    n_trials,
    total_trials,
    unclassified_trials,
    event_label,
    mapping_rule,
    sim,
    real,
    time_ms,
    compare_window,
    late_window,
):
    row = {
        "record_type": record_type,
        "dataset": dataset,
        "stage": stage,
        "condition": condition,
        "n_trials": n_trials,
        "total_trials": total_trials,
        "unclassified_trials": unclassified_trials,
        "event_channel_label": event_label,
        "condition_mapping": mapping_rule,
        "event_timing_note": (
            "cue onset at 0 ms; cue offset at 200 ms"
            if stage == "Stage1"
            else "target onset at 0 ms relative time"
        ),
        "compare_window_ms": f"{compare_window[0]}-{compare_window[1]}",
        "late_window_ms": f"{late_window[0]}-{late_window[1]}",
        "max_lag_search_ms": MAX_LAG_MS,
        "effective_max_lag_ms": float(
            np.floor(MAX_LAG_MS / np.median(np.diff(time_ms)))
            * np.median(np.diff(time_ms))
        ),
        "real_sample_interval_ms": float(np.median(np.diff(time_ms))),
    }
    if real is not None and sim is not None:
        row = channel_metric_fields(
            row,
            sim,
            real,
            time_ms,
            compare_window,
            late_window,
        )
    return row


def csv_fields():
    fields = [
        "record_type",
        "dataset",
        "stage",
        "condition",
        "n_trials",
        "total_trials",
        "unclassified_trials",
        "event_channel_label",
        "event_trials_without_marker",
        "trials_with_multiple_post_target_events",
        "first_event_median_time_ms",
        "condition_mapping",
        "event_timing_note",
        "compare_window_ms",
        "late_window_ms",
        "max_lag_search_ms",
        "effective_max_lag_ms",
        "real_sample_interval_ms",
        "mean_zero_lag_r",
        "mean_best_lag_r",
        "late_topography_cosine",
        "sim_late_f4_minus_f3_common_scale",
        "real_late_f4_minus_f3_common_scale",
    ]
    for channel in CHANNELS:
        prefix = channel.lower()
        fields.extend(
            [
                f"{prefix}_zero_lag_r",
                f"{prefix}_best_lag_r",
                f"{prefix}_best_lag_real_delay_ms",
                f"{prefix}_best_lag_at_search_limit",
                f"{prefix}_sim_late_positive_peak",
                f"{prefix}_real_late_positive_peak",
                f"{prefix}_sim_peak_time_ms",
                f"{prefix}_real_peak_time_ms",
                f"{prefix}_sim_peak_at_window_boundary",
                f"{prefix}_peak_time_difference_ms",
                f"{prefix}_real_peak_at_window_boundary",
                f"{prefix}_sim_late_mean_common_scale",
                f"{prefix}_real_late_mean_common_scale",
            ]
        )
    return fields


def save_csv(output_dir, rows):
    path = output_dir / "模拟真实EEG对比.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def channel_zscore(signal):
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    return (signal - mean) / np.maximum(std, 1e-12)


def plot_compare(output_dir, dataset, stage, cases, late_window):
    if not cases:
        return

    fig, axes = plt.subplots(
        len(cases) * 2,
        len(CHANNELS),
        figsize=(15, max(5.5, 4.8 * len(cases))),
        squeeze=False,
        sharex=True,
    )

    for case_index, case in enumerate(cases):
        label = case["label"]
        time_ms = case["time_ms"]
        sim = case["sim"]
        real = case["real"]
        sim_z = channel_zscore(sim)
        real_z = channel_zscore(real)
        plot_mask = (
            (time_ms >= PLOT_WINDOW_MS[0])
            & (time_ms <= PLOT_WINDOW_MS[1])
        )
        sim_scale = float(np.max(np.abs(sim[:, plot_mask])))
        real_scale = float(np.max(np.abs(real[:, plot_mask])))
        sim_common = sim / sim_scale if sim_scale > 0 else np.zeros_like(sim)
        real_common = real / real_scale if real_scale > 0 else np.zeros_like(real)
        count_title = f"{label} (n={case['n_trials']})"

        for channel_index, channel in enumerate(CHANNELS):
            z_axis = axes[case_index * 2, channel_index]
            common_axis = axes[case_index * 2 + 1, channel_index]
            z_axis.plot(time_ms, sim_z[channel_index], label="模拟（逐通道 z-score）")
            z_axis.plot(time_ms, real_z[channel_index], label="真实 ERP（逐通道 z-score）")
            common_axis.plot(
                time_ms,
                sim_common[channel_index],
                label="模拟（通道共享尺度）",
            )
            common_axis.plot(
                time_ms,
                real_common[channel_index],
                label="真实 ERP（通道共享尺度）",
            )

            for axis in (z_axis, common_axis):
                axis.axvspan(
                    *late_window,
                    color="gray",
                    alpha=0.08,
                    label=f"晚期窗 {late_window[0]}–{late_window[1]} ms",
                )
                axis.axhline(0, color="black", linewidth=0.7)
                if stage == "Stage1":
                    axis.axvline(
                        200,
                        color="tab:red",
                        linestyle="--",
                        linewidth=0.9,
                        label="Cue offset (200 ms)",
                    )
                axis.set_xlim(*PLOT_WINDOW_MS)
                axis.set_title(f"{count_title} · {channel}")
                axis.set_xlabel("相对事件时间 (ms)")
                axis.grid(alpha=0.22)

            z_axis.set_ylabel("逐通道标准分")
            common_axis.set_ylabel("各数据内部共享尺度归一化")
            common_axis.set_ylim(-1.05, 1.05)

    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        f"{dataset} {stage}：模拟与真实 EEG（真实采样间隔约 "
        f"{np.median(np.diff(cases[0]['time_ms'])):.2f} ms）"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "模拟真实EEG对比.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def process_dataset_stage(task, dataset, stage):
    data = load_dataset(dataset)
    masks, metadata = stage_conditions(task, stage, data)

    if stage == "Stage1":
        onset_s = 0.0
        baseline_window_s = (-0.2, 0.0)
        conditions = ("left", "right")
    else:
        onset_s = TARGET_ONSET_S
        baseline_window_s = (2.0, TARGET_ONSET_S)
        conditions = ("dots",) if task == "Task1" else ("inward", "outward")

    compare_window = STAGE_WINDOWS_MS[stage]["compare"]
    late_window = STAGE_WINDOWS_MS[stage]["late"]

    corrected = baseline_correct(data["eeg"], data["time_s"], baseline_window_s)
    relative_time_ms = (data["time_s"] - onset_s) * 1000.0
    valid_time = (
        (relative_time_ms >= PLOT_WINDOW_MS[0])
        & (relative_time_ms <= PLOT_WINDOW_MS[1])
    )
    time_ms = relative_time_ms[valid_time]
    if time_ms.size < 2:
        raise ValueError(f"{dataset}/{stage}: no post-onset EEG samples in plot window")

    output_dir = OUTPUT_ROOT / task / dataset / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = []
    condition_data = {}
    simulation_task = task

    for condition in conditions:
        mask = masks[condition]
        n_trials = int(mask.sum())
        sim = load_simulation(simulation_task, stage, condition)
        sim_on_grid = sample_simulation_on_real_grid(sim, time_ms)

        if n_trials:
            real = np.nanmean(corrected[mask][:, :, valid_time], axis=0)
            row = comparison_row(
                "condition",
                dataset,
                stage,
                condition,
                n_trials,
                metadata["total_trials"],
                metadata["unclassified_trials"],
                metadata["event_label"],
                metadata["mapping_rule"],
                sim_on_grid,
                real,
                time_ms,
                compare_window,
                late_window,
            )
            cases.append(
                {
                    "label": CONDITION_LABELS[condition],
                    "n_trials": n_trials,
                    "sim": sim_on_grid,
                    "real": real,
                    "time_ms": time_ms,
                }
            )
            condition_data[condition] = (sim_on_grid, real)
        else:
            row = comparison_row(
                "condition",
                dataset,
                stage,
                condition,
                0,
                metadata["total_trials"],
                metadata["unclassified_trials"],
                metadata["event_label"],
                metadata["mapping_rule"],
                None,
                None,
                time_ms,
                compare_window,
                late_window,
            )
        if "event_trials_without_marker" in metadata:
            row["event_trials_without_marker"] = metadata["event_trials_without_marker"]
            row["trials_with_multiple_post_target_events"] = metadata[
                "trials_with_multiple_post_target_events"
            ]
            row["first_event_median_time_ms"] = metadata[
                "first_event_median_time_ms"
            ]
        rows.append(row)

    if stage == "Stage1" or (stage == "Stage2" and task == "Task2"):
        first, second = conditions
        if first in condition_data and second in condition_data:
            sim_a, real_a = condition_data[first]
            sim_b, real_b = condition_data[second]
            rows.append(
                comparison_row(
                    "condition_difference",
                    dataset,
                    stage,
                    f"{first}-{second}",
                    "",
                    metadata["total_trials"],
                    metadata["unclassified_trials"],
                    metadata["event_label"],
                    metadata["mapping_rule"],
                    sim_a - sim_b,
                    real_a - real_b,
                    time_ms,
                    compare_window,
                    late_window,
                )
            )

    save_csv(output_dir, rows)
    plot_compare(output_dir, dataset, stage, cases, late_window)
    return output_dir, rows


def main():
    outputs = []
    for task, datasets in DATASETS.items():
        for dataset in datasets:
            for stage in ("Stage1", "Stage2"):
                output_dir, rows = process_dataset_stage(task, dataset, stage)
                outputs.append(output_dir)
                counts = ", ".join(
                    f"{row['condition']}={row['n_trials']}"
                    for row in rows
                    if row["record_type"] == "condition"
                )
                print(f"{dataset}/{stage}: {counts} -> {output_dir}")

    print(f"完成：{len(outputs)} 个 MAT × Stage 结果写入 {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
