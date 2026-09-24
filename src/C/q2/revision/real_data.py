"""Load record-level EEG cases without inventing target-layout labels.

Action/TgtAct values are retained solely as diagnostics. The clean epochs
contain the target-adjacent +/-1 level, not the later +/-2 response pulse
seen in the original Task-2 files. Neither is a certified layout trigger.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from .config import MIN_TRIALS, Q2_ROOT, STAGES, TIME_MS
except ImportError:
    from config import MIN_TRIALS, Q2_ROOT, STAGES, TIME_MS


REAL_ROOT = Q2_ROOT.parent / "q1" / "output" / "8riemann_denoise"
DATASETS = {
    "Task1": ("VisualCogA_Task-1", "VisualCogA_Task-2"),
    "Task2": ("VisualCogB_Task-1", "VisualCogB_Task-2"),
}
CHANNELS = ("F3", "Fz", "F4")


def _matlab_string(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_dataset(path):
    """Validate a clean MAT and return channels ordered F3/Fz/F4.

    A MAT is a recording, not an independently identified participant. Units
    remain those in the source data because no physical scale is declared.
    """
    path = Path(path)
    mat = loadmat(path)
    required = {"trial_data", "relative_time", "cue_type", "DataLabel", "SampleRate", "drop"}
    missing = required.difference(mat)
    if missing:
        raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
    trials = np.asarray(mat["trial_data"], dtype=float)
    relative_time = np.asarray(mat["relative_time"], dtype=float)
    cue = np.asarray(mat["cue_type"], dtype=float).reshape(-1)
    drop = np.asarray(mat["drop"], dtype=float).reshape(-1)
    labels = [_matlab_string(value) for value in np.asarray(mat["DataLabel"], dtype=object).reshape(-1)]
    if trials.ndim != 3 or trials.shape[0] == 0 or trials.shape[2] < 2:
        raise ValueError(f"{path.name}: expected nonempty Trial x Channel x Time data")
    if relative_time.shape != (trials.shape[0], trials.shape[2]):
        raise ValueError(f"{path.name}: relative_time shape disagrees with trial_data")
    if cue.size != trials.shape[0] or drop.size != trials.shape[0]:
        raise ValueError(f"{path.name}: cue_type/drop length disagrees with trial_data")
    if not np.all(np.isfinite(cue)) or not np.all(drop == 0):
        raise ValueError(f"{path.name}: invalid cues or rejected trials remain in clean data")
    if len(labels) != trials.shape[1] or len(set(labels)) != len(labels):
        raise ValueError(f"{path.name}: DataLabel must uniquely identify every channel")
    if not all(channel in labels for channel in CHANNELS):
        raise ValueError(f"{path.name}: F3/Fz/F4 are required")
    if not np.all(np.isfinite(relative_time)) or not np.allclose(relative_time, relative_time[:1], rtol=0, atol=1e-12):
        raise ValueError(f"{path.name}: finite common time axis required")
    rate = np.asarray(mat["SampleRate"], dtype=float).reshape(-1)
    if rate.size != 1 or not np.isfinite(rate[0]) or rate[0] <= 0:
        raise ValueError(f"{path.name}: SampleRate must be one positive finite value")
    time = relative_time[0]
    if not np.allclose(np.diff(time), 1.0 / rate[0], rtol=0, atol=1e-10):
        raise ValueError(f"{path.name}: time axis must increase uniformly at SampleRate")
    eeg = trials[:, [labels.index(channel) for channel in CHANNELS], :]
    if not np.all(np.isfinite(eeg)):
        raise ValueError(f"{path.name}: nonfinite EEG must be resolved before averaging")
    event_indices = [i for i, label in enumerate(labels) if label.startswith(("Action:", "TgtAct:"))]
    event_index = event_indices[0] if len(event_indices) == 1 else None
    return {
        "eeg": eeg,
        "time_s": time,
        "cue_type": cue,
        "sample_rate": float(rate[0]),
        "labels": labels,
        "event_values": trials[:, event_index, :] if event_index is not None else None,
        "event_label": labels[event_index] if event_index is not None else "unavailable_or_ambiguous",
    }


def stage_condition_masks(task, stage, cue_type):
    """Use only certified cue labels; aggregate all unknown target layouts."""
    cue = np.asarray(cue_type).reshape(-1)
    if task not in DATASETS or stage not in STAGES:
        raise ValueError(f"Unsupported task/stage: {task}/{stage}")
    if stage == "Stage1":
        return {"left": cue == -1, "right": cue == 1}
    condition = "dots" if task == "Task1" else "target_unknown"
    return {condition: np.ones(cue.size, dtype=bool)}


def event_diagnostics(dataset, task, data):
    """Describe event values and edge latencies, with no behavioral labels."""
    values = data["event_values"]
    time = data["time_s"]
    onset = STAGES["Stage2"]["onset_s"]
    first_delays = []
    edge_counts = []
    if values is not None:
        for row in values:
            finite_nonzero = np.isfinite(row) & (row != 0)
            edges = np.flatnonzero(finite_nonzero & np.r_[True, row[:-1] == 0])
            edges = edges[time[edges] >= onset]
            edge_counts.append(edges.size)
            if edges.size:
                first_delays.append(float((time[edges[0]] - onset) * 1000))
    codes = np.unique(values[np.isfinite(values)]) if values is not None else []
    return {
        "dataset": dataset,
        "task": task,
        "total_trials": int(data["eeg"].shape[0]),
        "sample_rate_hz": data["sample_rate"],
        "epoch_start_s": float(time[0]),
        "epoch_end_s": float(time[-1]),
        "unknown_cue_trials": int(np.count_nonzero(~np.isin(data["cue_type"], (-1, 1)))),
        "event_channel_label": data["event_label"],
        "event_codes_present": ";".join(f"{value:g}" for value in codes),
        "absolute_code_2_samples": int(np.count_nonzero(np.abs(values) == 2)) if values is not None else 0,
        "first_event_median_delay_ms": float(np.median(first_delays)) if first_delays else float("nan"),
        "first_event_min_delay_ms": float(min(first_delays)) if first_delays else float("nan"),
        "first_event_max_delay_ms": float(max(first_delays)) if first_delays else float("nan"),
        "trials_with_post_target_edge": len(first_delays),
        "trials_with_multiple_post_target_edges": int(np.count_nonzero(np.asarray(edge_counts) > 1)),
        "event_role": "diagnostic_only_unknown_semantics",
        "timing_policy": "cue_aligned_epoch; fixed_target_2.20s_assumption",
    }


def load_cases_with_audit(real_root=REAL_ROOT, datasets=None):
    """Return condition cases and one event-diagnostic row per recording.

    Baselines are removed separately from each trial before averaging. All
    metrics retain the actual measured time grid; no artificial 800-ms
    endpoint or missing baseline is extrapolated into the data.
    """
    datasets = DATASETS if datasets is None else datasets
    cases, audit = [], []
    for task, records in datasets.items():
        for dataset in records:
            data = load_dataset(Path(real_root) / f"{dataset}_clean.mat")
            audit.append(event_diagnostics(dataset, task, data))
            time = data["time_s"]
            sample_interval = 1 / data["sample_rate"]
            for stage, settings in STAGES.items():
                low, high = settings["baseline"]
                baseline = (time >= low) & (time < high)
                if baseline.sum() < 2 or time[0] > low + sample_interval or time[-1] < high - sample_interval:
                    raise ValueError(f"{dataset}/{stage}: baseline window is absent or truncated")
                relative_ms = (time - settings["onset_s"]) * 1000
                window = (relative_ms >= TIME_MS[0]) & (relative_ms <= TIME_MS[-1])
                if (window.sum() < 2
                        or relative_ms[window][0] > TIME_MS[0] + sample_interval * 1000 + 1e-9
                        or relative_ms[window][-1] < TIME_MS[-1] - sample_interval * 1000 - 1e-9):
                    raise ValueError(f"{dataset}/{stage}: response interval is absent or truncated")
                corrected = data["eeg"] - data["eeg"][:, :, baseline].mean(axis=2, keepdims=True)
                masks = stage_condition_masks(task, stage, data["cue_type"])
                for condition, mask in masks.items():
                    selected = corrected[mask][:, :, window]
                    n_trials = int(mask.sum())
                    cases.append({
                        "dataset": dataset,
                        "task": task,
                        "stage": stage,
                        "condition": condition,
                        "n_trials": n_trials,
                        "time_ms": relative_ms[window].copy(),
                        "real": selected.mean(axis=0) if n_trials else np.full((3, int(window.sum())), np.nan),
                        "trials": selected,
                        "eligible_fit": bool(n_trials >= MIN_TRIALS and condition != "target_unknown"),
                        "eligibility_reason": "unknown_target_layout" if condition == "target_unknown" else ("eligible" if n_trials >= MIN_TRIALS else "below_min_trials"),
                        "baseline_window_s": tuple(settings["baseline"]),
                        "late_window_ms": tuple(settings["late"]),
                        "event_timing_assumption": "cue=0s; cue_offset=0.20s; target=2.20s",
                    })
    return cases, audit


def load_cases(real_root=REAL_ROOT, datasets=None):
    """Return cases only; see load_cases_with_audit for event diagnostics."""
    return load_cases_with_audit(real_root, datasets)[0]
