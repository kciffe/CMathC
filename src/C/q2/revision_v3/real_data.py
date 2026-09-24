"""Load record-level EEG cases without inventing target-layout labels.

Action/TgtAct values are retained solely as diagnostics. The clean epochs
contain the target-adjacent +/-1 level, not the later +/-2 response pulse
seen in the original Task-2 files. Neither is a certified layout trigger.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from .config import CHANNELS, DATASETS, MIN_TRIALS, Q2_ROOT, REAL_ROOT, STAGES
except ImportError:
    from config import CHANNELS, DATASETS, MIN_TRIALS, Q2_ROOT, REAL_ROOT, STAGES


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
    cue_indices = [i for i, label in enumerate(labels) if label.startswith("VisCue:")]
    action_indices = [i for i, label in enumerate(labels)
                      if label.startswith(("Action:", "TgtAct:"))]
    cue_index = cue_indices[0] if len(cue_indices) == 1 else None
    action_index = action_indices[0] if len(action_indices) == 1 else None
    return {
        "eeg": eeg,
        "time_s": time,
        "cue_type": cue,
        "sample_rate": float(rate[0]),
        "labels": labels,
        "cue_values": trials[:, cue_index, :] if cue_index is not None else None,
        "cue_label": labels[cue_index] if cue_index is not None else "unavailable_or_ambiguous",
        "action_values": trials[:, action_index, :] if action_index is not None else None,
        "action_label": labels[action_index] if action_index is not None else "unavailable_or_ambiguous",
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
    """Audit visual cue markers separately from action/response channels."""
    time = data["time_s"]
    cue_values = data["cue_values"]
    cue_starts, cue_last_active, cue_offsets, cue_durations = [], [], [], []
    cue_signs, cue_trial_indices = [], []
    if cue_values is not None:
        for trial_index, row in enumerate(cue_values):
            active = np.isfinite(row) & (row != 0)
            starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
            ends = np.flatnonzero(active & np.r_[~active[1:], True])
            if starts.size:
                onset_ms = float(time[starts[0]] * 1000.0)
                last_active_ms = float(time[ends[0]] * 1000.0)
                cue_starts.append(onset_ms)
                cue_last_active.append(last_active_ms)
                if ends[0] + 1 < len(time):
                    offset_ms = float(time[ends[0] + 1] * 1000.0)
                    cue_offsets.append(offset_ms)
                    cue_durations.append(offset_ms - onset_ms)
                cue_signs.append(float(np.sign(row[starts[0]])))
                cue_trial_indices.append(trial_index)
    cue_signs = np.asarray(cue_signs, dtype=float)
    cue_type = np.asarray(data["cue_type"], dtype=float)
    matched_cue_type = cue_type[cue_trial_indices] if cue_trial_indices else np.array([], dtype=float)
    cue_valid = np.isin(matched_cue_type, (-1.0, 1.0))
    cue_agreement = (float(np.mean(cue_signs[cue_valid] == matched_cue_type[cue_valid]))
                     if cue_valid.any() else float("nan"))

    action_values = data["action_values"]
    target_onset = STAGES["Stage2"]["onset_s"]
    action_delays, action_edge_counts = [], []
    if action_values is not None:
        for row in action_values:
            active = np.isfinite(row) & (row != 0)
            edges = np.flatnonzero(active & np.r_[True, ~active[:-1]])
            edges = edges[time[edges] >= target_onset]
            action_edge_counts.append(edges.size)
            if edges.size:
                action_delays.append(float((time[edges[0]] - target_onset) * 1000.0))
    action_codes = (np.unique(action_values[np.isfinite(action_values)])
                    if action_values is not None else [])
    cue_codes = np.unique(cue_values[np.isfinite(cue_values)]) if cue_values is not None else []
    return {
        "dataset": dataset,
        "task": task,
        "total_trials": int(data["eeg"].shape[0]),
        "sample_rate_hz": data["sample_rate"],
        "epoch_start_s": float(time[0]),
        "epoch_end_s": float(time[-1]),
        "unknown_cue_trials": int(np.count_nonzero(~np.isin(data["cue_type"], (-1, 1)))),
        "cue_marker_label": data["cue_label"],
        "cue_marker_codes_present": ";".join(f"{value:g}" for value in cue_codes),
        "cue_marker_trials": len(cue_starts),
        "cue_onset_median_ms": float(np.median(cue_starts)) if cue_starts else float("nan"),
        "cue_onset_min_ms": float(min(cue_starts)) if cue_starts else float("nan"),
        "cue_onset_max_ms": float(max(cue_starts)) if cue_starts else float("nan"),
        "cue_last_active_sample_median_ms": (float(np.median(cue_last_active))
                                             if cue_last_active else float("nan")),
        "cue_offset_median_ms": float(np.median(cue_offsets)) if cue_offsets else float("nan"),
        "cue_offset_observed_trials": len(cue_offsets),
        "cue_active_duration_median_ms": (float(np.median(cue_durations))
                                           if cue_durations else float("nan")),
        "cue_type_agreement": cue_agreement,
        "cue_type_agreement_trials": int(cue_signs.size),
        "action_channel_label": data["action_label"],
        "action_codes_present": ";".join(f"{value:g}" for value in action_codes),
        "action_absolute_code_2_samples": (int(np.count_nonzero(np.abs(action_values) == 2))
                                            if action_values is not None else 0),
        "action_first_post_target_edge_median_ms": (float(np.median(action_delays))
                                                    if action_delays else float("nan")),
        "action_first_post_target_edge_min_ms": min(action_delays) if action_delays else float("nan"),
        "action_first_post_target_edge_max_ms": max(action_delays) if action_delays else float("nan"),
        "action_trials_with_post_target_edge": len(action_delays),
        "action_trials_with_multiple_post_target_edges": int(np.count_nonzero(np.asarray(action_edge_counts) > 1)),
        "action_channel_role": "diagnostic_only_action_or_response_not_used_for_cue_alignment",
        "timing_policy": "cue_epoch_aligned_by_VisCue; target_onset_2.20s_from_experiment_schedule",
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
                window = (relative_ms >= 0.0) & (relative_ms <= 800.0 + 1e-9)
                if window.sum() < 2 or relative_ms[window][0] > sample_interval * 1000 + 1e-9:
                    raise ValueError(f"{dataset}/{stage}: response onset is absent or truncated")
                # The requested 800-ms endpoint need not itself be sampled.
                # Accept the final observed sample when it is within one sample
                # interval of the boundary (e.g. 796.875 ms at 128 Hz), but
                # reject genuinely truncated epochs for either stage.
                earliest_valid_end = 800.0 - sample_interval * 1000.0 - 1e-6
                if relative_ms[window][-1] < earliest_valid_end:
                    raise ValueError(f"{dataset}/{stage}: response interval is truncated before 800 ms")
                corrected = data["eeg"] - data["eeg"][:, :, baseline].mean(axis=2, keepdims=True)
                masks = stage_condition_masks(task, stage, data["cue_type"])
                for condition, mask in masks.items():
                    selected = corrected[mask][:, :, window]
                    n_trials = int(mask.sum())
                    is_stage1_label = stage == "Stage1" and condition in ("left", "right")
                    enough_trials = n_trials >= MIN_TRIALS
                    role = ("fit_and_classification" if is_stage1_label and enough_trials
                            else "description_only" if condition == "target_unknown"
                            else "frozen_parameter_control" if stage == "Stage2" and task == "Task1"
                            else "below_min_trials" if not enough_trials else "description_only")
                    cases.append({
                        "dataset": dataset,
                        "record": dataset,
                        "task": task,
                        "stage": stage,
                        "condition": condition,
                        "n_trials": n_trials,
                        "time_ms": relative_ms[window].copy(),
                        "real": selected.mean(axis=0) if n_trials else np.full((3, int(window.sum())), np.nan),
                        "trials": selected,
                        "eligible_fit": bool(is_stage1_label and enough_trials),
                        "eligible_classification": bool(is_stage1_label and enough_trials),
                        "eligible": bool(is_stage1_label and enough_trials),
                        "role": role,
                        "eligibility_reason": ("eligible_stage1_left_right" if is_stage1_label and enough_trials
                                               else "unknown_target_layout" if condition == "target_unknown"
                                               else "not_main_stage1_fit" if stage == "Stage2"
                                               else "below_min_trials"),
                        "baseline_window_s": tuple(settings["baseline"]),
                        "late_window_ms": tuple(settings["late"]),
                        "event_timing_assumption": "cue=0s; cue_offset=0.20s; target=2.20s",
                    })
    return cases, audit


def load_cases(real_root=REAL_ROOT, datasets=None):
    """Return cases only; see load_cases_with_audit for event diagnostics."""
    return load_cases_with_audit(real_root, datasets)[0]
