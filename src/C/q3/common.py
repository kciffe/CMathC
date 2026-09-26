"""Shared MAT, event-mapping, and output helpers for the Q3 scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from config import (
    ALLOWED_CHANNELS,
    OUTPUT_DIR,
    Q1_CLEAN_DIR,
    RAW_DATA_DIR,
    RECORDS,
    RESPONSE_ANALYSIS_WINDOW_S,
    RESPONSE_DEADLINE_AFTER_CUE_S,
    RESPONSE_CHANNEL_NAMES,
    TARGET_OFFSET_S,
    TARGET_OFFSET_SOURCE,
)


def matlab_text(value: Any) -> str:
    """Unwrap common MATLAB cell/string representations into plain text."""
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        return "".join(matlab_text(item) for item in value.reshape(-1)).strip()
    return str(value).strip()


def matlab_labels(data_label: Any) -> list[str]:
    return [matlab_text(item) for item in np.asarray(data_label, dtype=object).reshape(-1)]


def channel_name(label: str) -> str:
    return label.split(":", maxsplit=1)[0].strip()


def _selected_channel_indices(
    labels: list[str], include_response: bool = False
) -> dict[str, int]:
    """Select allowed EEG/event rows and, for raw MAT only, one response row."""
    selected: dict[str, int] = {}
    for index, label in enumerate(labels):
        name = channel_name(label)
        if name in ALLOWED_CHANNELS and name not in selected:
            selected[name] = index
    missing = sorted(ALLOWED_CHANNELS.difference(selected))
    if missing:
        raise ValueError(f"MAT file is missing allowed channel labels: {missing}")
    if include_response:
        response_matches = [
            (index, label)
            for index, label in enumerate(labels)
            if channel_name(label) in RESPONSE_CHANNEL_NAMES
        ]
        if len(response_matches) != 1:
            raise ValueError(
                "Raw MAT file must have exactly one Action/TgtAct response channel; "
                f"found {len(response_matches)}"
            )
        selected["Response"] = response_matches[0][0]
    return selected


def find_record_path(record: str) -> Path:
    path = RAW_DATA_DIR / f"{record}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Could not find raw MAT file for {record}: {path}")
    return path


def load_raw_record(record: str) -> dict[str, Any]:
    """Load only the allowed source rows from one continuous MAT record."""
    path = find_record_path(record)
    source = loadmat(path, squeeze_me=False, struct_as_record=False)
    data_key = "data" if "data" in source else "Data" if "Data" in source else None
    if data_key is None or "DataLabel" not in source or "SampleRate" not in source:
        raise ValueError(f"Expected data, DataLabel, and SampleRate in {path.name}")

    all_labels = matlab_labels(source["DataLabel"])
    selected = _selected_channel_indices(all_labels, include_response=True)
    matrix = np.asarray(source[data_key], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected channel x sample matrix in {path.name}, got {matrix.shape}")
    if matrix.shape[0] != len(all_labels):
        raise ValueError(
            f"DataLabel count {len(all_labels)} does not match matrix rows {matrix.shape[0]}"
        )

    channels = {
        name: matrix[index, :].copy()
        for name, index in selected.items()
        if name != "Response"
    }
    sample_rate = float(np.asarray(source["SampleRate"]).squeeze())
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in {path.name}: {sample_rate}")
    return {
        "record": record,
        "path": path,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "response_signal": matrix[selected["Response"], :].copy(),
        "response_label": all_labels[selected["Response"]],
        "labels": {name: all_labels[index] for name, index in selected.items()},
    }


def detect_cues(cue_signal: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
    cue = np.asarray(cue_signal, dtype=np.float64).reshape(-1)
    time = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if cue.size != time.size:
        raise ValueError(f"Cue and timestamp lengths differ: {cue.size} vs {time.size}")
    active = np.isfinite(cue) & (cue != 0)
    previous = np.r_[False, active[:-1]]
    starts = np.flatnonzero(active & ~previous)
    events: list[dict[str, Any]] = []
    for trial_index, sample_index in enumerate(starts):
        side = int(np.sign(cue[sample_index]))
        if side not in (-1, 1):
            continue
        events.append(
            {
                "original_trial_index": int(trial_index),
                "cue_sample_index": int(sample_index),
                "cue_time_s": float(time[sample_index]),
                "cue_side": side,
            }
        )
    return events


def standardize_response_code(raw_response: np.ndarray) -> np.ndarray:
    """Map raw {-2,-1,0,+1,+2} codes to the shared {-2,0,+2} scale."""
    raw = np.asarray(raw_response, dtype=np.float64)
    valid = np.isfinite(raw)
    unsupported = valid & ~np.isin(raw, (-2.0, -1.0, 0.0, 1.0, 2.0))
    if unsupported.any():
        values = np.unique(raw[unsupported]).tolist()
        raise ValueError(f"unsupported response code(s): {values}")
    standardized = np.zeros(raw.shape, dtype=np.int8)
    standardized[np.isin(raw, (-2.0, -1.0))] = -2
    standardized[np.isin(raw, (1.0, 2.0))] = 2
    return standardized


def classify_channel9_behavior(
    *,
    cue_side: Any,
    response_side: Any,
    cue_time_s: Any,
    response_time_s: Any,
    observation_end_time_s: Any,
    deadline_after_cue_s: float = RESPONSE_DEADLINE_AFTER_CUE_S,
) -> dict[str, Any]:
    """Apply the supplied same-direction correctness and cue+3 s timing rules.

    Channel-9 response time is the first zero-to-nonzero edge. For TgtAct,
    ``decode_response_bout`` has already required the signed ±1 stage followed
    by the same-side declared ±2 code. Missing responses are untimely only if
    the recording covers the full cue-relative deadline.
    """
    def finite_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    cue = finite_number(cue_side)
    response = finite_number(response_side)
    cue_time = finite_number(cue_time_s)
    response_time = finite_number(response_time_s)
    observation_end = finite_number(observation_end_time_s)

    if cue in (-1.0, 1.0) and response in (-1.0, 1.0):
        task_correct = int(cue == response)
        correctness_status = (
            "correct_by_same_direction_rule"
            if task_correct
            else "incorrect_by_opposite_direction_rule"
        )
    else:
        task_correct = np.nan
        correctness_status = "unknown_cue_or_response_direction"

    deadline_time = (
        cue_time + float(deadline_after_cue_s)
        if cue_time is not None
        else np.nan
    )
    latency = (
        response_time - cue_time
        if response_time is not None and cue_time is not None
        else np.nan
    )
    deadline_covered = bool(
        np.isfinite(deadline_time)
        and observation_end is not None
        and observation_end >= deadline_time
    )

    if response_time is not None and np.isfinite(deadline_time):
        timely = int(response_time <= deadline_time)
        timing_status = (
            "response_within_cue_plus_3s_deadline"
            if timely
            else "response_after_cue_plus_3s_deadline"
        )
    elif response_time is not None:
        timely = np.nan
        timing_status = "cue_time_unavailable"
    elif deadline_covered:
        timely = 0
        timing_status = "no_response_by_cue_plus_3s_deadline"
    else:
        timely = np.nan
        timing_status = "deadline_not_observed"

    return {
        "task_correct": task_correct,
        "task_correctness_status": correctness_status,
        "task_correctness_source": "channel8_channel9_same_direction_user_rule",
        "response_latency_from_cue_s": latency,
        "cue_plus_3_deadline_time_s": deadline_time,
        "cue_plus_3_deadline_covered": deadline_covered,
        "timely_response": timely,
        "timeliness_status": timing_status,
        "timeliness_source": "user_defined_channel9_onset_within_3s_after_channel8_cue",
    }


def detect_response_events(
    raw_response: np.ndarray, timestamps: np.ndarray
) -> list[dict[str, Any]]:
    raw = np.asarray(raw_response, dtype=np.float64).reshape(-1)
    time = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if raw.size != time.size:
        raise ValueError(f"Response and timestamp lengths differ: {raw.size} vs {time.size}")
    standardized = standardize_response_code(raw)
    active = standardized != 0
    # A response is one onset edge from inactive to active. A direction-code
    # change inside a sustained nonzero segment must not create a second event.
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    return [
        {
            "response_sample_index": int(index),
            "response_time_s": float(time[index]),
            "response_raw": float(raw[index]),
            "response_code": int(standardized[index]),
            "choice_side": int(standardized[index] // 2),
        }
        for index in starts
    ]


def _response_code_legend(response_label: str) -> dict[str, float]:
    return {
        side.upper(): float(value)
        for side, value in re.findall(
            r"([LR])\s*([+-]?\d+(?:\.\d+)?)", str(response_label), flags=re.IGNORECASE
        )
    }


def measure_response_window(
    raw_response: np.ndarray,
    timestamps: np.ndarray,
    cue_sample_index: int,
    trial_start_sample_index: int,
    trial_end_sample_index: int,
    sample_rate_hz: float,
    response_label: str,
    response_onset_sample_index: int | None,
) -> dict[str, Any]:
    """Measure the selected cue-centered observation window and action-bout duration.

    This window is used only to count event/code samples. Timeliness is computed
    separately by the supplied cue-relative 3 s cutoff.
    """
    signal = np.asarray(raw_response, dtype=np.float64).reshape(-1)
    time = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    cue_index = int(cue_sample_index)
    trial_start = int(trial_start_sample_index)
    trial_end = min(int(trial_end_sample_index), signal.size, time.size)
    cue_time = float(time[cue_index])
    window_start_s = cue_time + RESPONSE_ANALYSIS_WINDOW_S[0]
    window_end_s = cue_time + RESPONSE_ANALYSIS_WINDOW_S[1]
    window_start = int(np.searchsorted(time, window_start_s, side="left"))
    window_end = int(np.searchsorted(time, window_end_s, side="right"))
    window_end_clipped = min(window_end, signal.size, time.size)
    window_start_clipped = min(window_start, window_end_clipped)
    window = signal[window_start_clipped:window_end_clipped]
    window_active = np.isfinite(window) & (window != 0)

    legend = _response_code_legend(response_label)
    legend_valid = (
        set(legend) == {"L", "R"}
        and np.sign(legend["L"]) == -1
        and np.sign(legend["R"]) == 1
    )
    declared_code_count = 0
    declared_code_time = np.nan
    if legend_valid and window.size:
        code_mask = np.zeros(window.shape, dtype=bool)
        for code in legend.values():
            code_mask |= np.isclose(window, code, rtol=0.0, atol=1e-8)
        declared_code_count = int(code_mask.sum())
        if declared_code_count:
            declared_code_time = float(time[window_start_clipped + int(np.flatnonzero(code_mask)[0])])

    recording_start_s = float(time[0]) if time.size else np.inf
    recording_end_exclusive_s = (
        float(time[-1] + 1.0 / sample_rate_hz) if time.size else -np.inf
    )
    window_fully_observed = bool(
        window_start_s >= recording_start_s
        and window_end_s <= recording_end_exclusive_s + 1e-9
        and window_end <= min(signal.size, time.size)
    )
    if not window_fully_observed:
        analysis_window_observation_status = "window_not_fully_recorded"
    elif declared_code_count:
        analysis_window_observation_status = "declared_response_code_observed"
    elif window_active.any():
        analysis_window_observation_status = "nonzero_action_observed_without_declared_code"
    else:
        analysis_window_observation_status = "no_action_code_observed_in_window"

    response_duration_s = np.nan
    response_bout_end_time_s = np.nan
    if response_onset_sample_index is not None:
        bout_start = int(response_onset_sample_index)
        bout_stop_limit = min(trial_end, signal.size)
        if 0 <= bout_start < bout_stop_limit and np.isfinite(signal[bout_start]) and signal[bout_start] != 0:
            bout_stop = bout_start + 1
            while bout_stop < bout_stop_limit and np.isfinite(signal[bout_stop]) and signal[bout_stop] != 0:
                bout_stop += 1
            response_duration_s = float((bout_stop - bout_start) / sample_rate_hz)
            if bout_stop < time.size:
                response_bout_end_time_s = float(time[bout_stop])
            elif time.size:
                response_bout_end_time_s = float(time[-1] + 1.0 / sample_rate_hz)

    return {
        "response_analysis_window_start_s": window_start_s,
        "response_analysis_window_end_s": window_end_s,
        "response_analysis_window_start_sample_index": window_start_clipped,
        "response_analysis_window_end_sample_index_exclusive": window_end_clipped,
        "response_window_fully_observed": window_fully_observed,
        "response_nonzero_sample_count_in_analysis_window": int(window_active.sum()),
        "response_present_in_analysis_window": bool(window_active.any()),
        "response_declared_code_sample_count_in_analysis_window": declared_code_count,
        "response_declared_code_time_in_analysis_window_s": declared_code_time,
        "analysis_window_observation_status": analysis_window_observation_status,
        "response_duration_s": response_duration_s,
        "response_bout_end_time_s": response_bout_end_time_s,
    }


def decode_response_bout(
    raw_response: np.ndarray,
    onset_sample_index: int,
    interval_end_sample_index: int,
    response_label: str,
) -> dict[str, Any]:
    """Decode one action bout using the response codes declared in its MAT label.

    The onset remains the first zero-to-nonzero edge. Some TgtAct records change
    from a signed ±1 level to the declared signed ±2 click code within that same
    nonzero bout, so magnitude codes are not treated as separate responses.
    """
    signal = np.asarray(raw_response, dtype=np.float64).reshape(-1)
    start = int(onset_sample_index)
    stop_limit = min(int(interval_end_sample_index), signal.size)
    if start < 0 or start >= stop_limit or not np.isfinite(signal[start]) or signal[start] == 0:
        return {
            "choice_side": None,
            "decode_status": "invalid_response_bout_start",
            "declared_response_code": None,
            "bout_code_sequence": "",
        }

    stop = start + 1
    while stop < stop_limit and np.isfinite(signal[stop]) and signal[stop] != 0:
        stop += 1
    bout = signal[start:stop]
    signs = np.unique(np.sign(bout[np.isfinite(bout) & (bout != 0)]))
    code_sequence = list(dict.fromkeys(int(value) for value in bout if np.isfinite(value)))
    sequence_text = "|".join(str(value) for value in code_sequence)
    if signs.size != 1:
        return {
            "choice_side": None,
            "decode_status": "conflicting_or_missing_direction_within_response_bout",
            "declared_response_code": None,
            "bout_code_sequence": sequence_text,
        }

    legend = _response_code_legend(response_label)
    if set(legend) != {"L", "R"} or np.sign(legend["L"]) != -1 or np.sign(legend["R"]) != 1:
        return {
            "choice_side": None,
            "decode_status": "response_channel_legend_unreadable",
            "declared_response_code": None,
            "bout_code_sequence": sequence_text,
        }

    matching_sides = [
        side for side, code in legend.items()
        if np.any(np.isclose(bout, code, rtol=0.0, atol=1e-8))
    ]
    if len(matching_sides) != 1:
        return {
            "choice_side": None,
            "decode_status": "declared_response_code_missing_or_ambiguous_in_bout",
            "declared_response_code": None,
            "bout_code_sequence": sequence_text,
        }

    side = matching_sides[0]
    expected_sign = -1 if side == "L" else 1
    if int(signs[0]) != expected_sign:
        return {
            "choice_side": None,
            "decode_status": "response_bout_sign_conflicts_with_mat_legend",
            "declared_response_code": legend[side],
            "bout_code_sequence": sequence_text,
        }
    if channel_name(response_label).lower() == "tgtact":
        onset_stage_code = float(expected_sign)
        declared_code = float(legend[side])
        onset_stage_index = next(
            (i for i, value in enumerate(code_sequence) if np.isclose(value, onset_stage_code)),
            None,
        )
        declared_code_index = next(
            (i for i, value in enumerate(code_sequence) if np.isclose(value, declared_code)),
            None,
        )
        if onset_stage_index != 0 or declared_code_index is None or declared_code_index <= onset_stage_index:
            return {
                "choice_side": None,
                "decode_status": "tgtact_requires_signed_stage1_then_declared_stage2_code",
                "declared_response_code": declared_code,
                "bout_code_sequence": sequence_text,
            }
    return {
        "choice_side": expected_sign,
        "decode_status": "decoded_from_declared_channel_code",
        "declared_response_code": legend[side],
        "bout_code_sequence": sequence_text,
    }


def load_q1_clean(record: str) -> dict[str, Any]:
    """Load Q1 quality-retained trials, selecting EEG/VisCue/TimeStamp only."""
    path = Q1_CLEAN_DIR / f"{record}_clean.mat"
    if not path.exists():
        raise FileNotFoundError(f"Q1 clean MAT is missing: {path}")
    source = loadmat(path, squeeze_me=False, struct_as_record=False)
    labels = matlab_labels(source["DataLabel"])
    selected = _selected_channel_indices(labels)
    trials = np.asarray(source["trial_data"], dtype=np.float64)
    relative_time = np.asarray(source["relative_time"], dtype=np.float64)
    if trials.ndim != 3:
        raise ValueError(f"Expected trial x channel x sample data, got {trials.shape}")
    if relative_time.shape != (trials.shape[0], trials.shape[2]):
        raise ValueError(f"relative_time shape mismatch in {path.name}")
    sample_rate = float(np.asarray(source["SampleRate"]).squeeze())

    eeg = np.stack([trials[:, selected[channel], :] for channel in ("F3", "Fz", "F4")], axis=1)
    cue_data = trials[:, selected["VisCue"], :]
    timestamp_data = trials[:, selected["TimeStamp"], :]
    cue_type = np.asarray(source.get("cue_type", np.full((1, trials.shape[0]), np.nan))).reshape(-1)

    clean_events: list[dict[str, Any]] = []
    for clean_index in range(trials.shape[0]):
        cue = cue_data[clean_index]
        active = np.isfinite(cue) & (cue != 0)
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        if starts.size:
            marker_index = int(starts[0])
            marker_side = int(np.sign(cue[marker_index]))
            cue_time = float(timestamp_data[clean_index, marker_index])
        else:
            marker_index = int(np.argmin(np.abs(relative_time[clean_index])))
            marker_side = int(np.sign(cue_type[clean_index])) if clean_index < cue_type.size else 0
            cue_time = float(timestamp_data[clean_index, marker_index])
        clean_events.append(
            {
                "q1_trial_index": clean_index,
                "q1_trial_key": f"{record}:q1_{clean_index:03d}",
                "cue_time_s": cue_time,
                "cue_side": marker_side,
                "marker_found": bool(starts.size),
            }
        )

    return {
        "record": record,
        "path": path,
        "sample_rate_hz": sample_rate,
        "eeg": eeg,
        "relative_time": relative_time,
        "clean_events": clean_events,
    }


def load_quality_table(record: str) -> pd.DataFrame | None:
    path = Q1_CLEAN_DIR / f"{record}_SQI指标.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Trial", "CueType", "FinalDrop"}
    if not required.issubset(table.columns):
        return None
    return table


def build_record_event_tables(record: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create raw cue rows and timestamp-verified links to Q1 retained trials."""
    raw = load_raw_record(record)
    cue = raw["channels"]["VisCue"]
    timestamps = raw["channels"]["TimeStamp"]
    sr = raw["sample_rate_hz"]
    if not np.isfinite(timestamps).all():
        raise ValueError(f"Non-finite timestamp values in {record}")
    timestamp_delta = np.diff(timestamps)
    if timestamp_delta.size and np.any(timestamp_delta <= 0):
        raise ValueError(f"Timestamp is not strictly increasing in {record}")
    median_dt = float(np.median(timestamp_delta)) if timestamp_delta.size else np.nan
    events = detect_cues(cue, timestamps)
    response_events = detect_response_events(raw["response_signal"], timestamps)
    response_by_trial: dict[int, dict[str, Any]] = {}
    for trial_index, event in enumerate(events):
        trial_start = int(event["cue_sample_index"])
        trial_end = (
            int(events[trial_index + 1]["cue_sample_index"])
            if trial_index + 1 < len(events)
            else cue.size
        )
        trial_responses = [
            response
            for response in response_events
            if trial_start <= response["response_sample_index"] < trial_end
        ]
        response = trial_responses[0] if trial_responses else None
        decoded_response = (
            decode_response_bout(
                raw["response_signal"],
                response["response_sample_index"],
                trial_end,
                raw["response_label"],
            )
            if response is not None
            else {
                "choice_side": None,
                "decode_status": "no_response_bout",
                "declared_response_code": None,
                "bout_code_sequence": "",
            }
        )
        response_timing = measure_response_window(
            raw["response_signal"],
            timestamps,
            int(event["cue_sample_index"]),
            trial_start,
            trial_end,
            sr,
            raw["response_label"],
            int(response["response_sample_index"]) if response is not None else None,
        )
        interval_end_index = min(trial_end, timestamps.size - 1)
        response_by_trial[trial_index] = {
            "response_events": trial_responses,
            "response_event_count": len(trial_responses),
            "response": response,
            "decoded_response": decoded_response,
            "response_timing": response_timing,
            "cue_interval_observation_end_time_s": float(timestamps[interval_end_index]),
        }

    quality = load_quality_table(record)
    quality_by_trial: dict[int, dict[str, Any]] = {}
    quality_order_validated = False
    if quality is not None and len(quality) == len(events):
        trial_numbers = pd.to_numeric(quality["Trial"], errors="coerce").to_numpy()
        cue_types = pd.to_numeric(quality["CueType"], errors="coerce").to_numpy()
        final_drop = pd.to_numeric(quality["FinalDrop"], errors="coerce").to_numpy()
        expected = np.arange(len(events))
        observed_sides = np.asarray([event["cue_side"] for event in events], dtype=float)
        quality_order_validated = bool(
            np.array_equal(trial_numbers, expected)
            and np.array_equal(cue_types, observed_sides)
        )
        if quality_order_validated:
            quality_by_trial = {
                i: {"q1_final_drop": int(final_drop[i]) if np.isfinite(final_drop[i]) else None}
                for i in range(len(events))
            }

    clean = load_q1_clean(record)
    tolerance_s = max(1.5 / sr, 0.5 / clean["sample_rate_hz"])
    mapping: dict[int, dict[str, Any]] = {}
    used_raw: set[int] = set()
    mapping_rows: list[dict[str, Any]] = []
    for clean_event in clean["clean_events"]:
        differences = np.abs(
            np.asarray([event["cue_time_s"] for event in events], dtype=float)
            - clean_event["cue_time_s"]
        )
        if not differences.size:
            match_idx = None
            delta_s = np.nan
        else:
            nearest = int(np.argmin(differences))
            delta_s = float(differences[nearest])
            match_idx = nearest if delta_s <= tolerance_s and nearest not in used_raw else None
        side_matches = (
            match_idx is not None
            and int(events[match_idx]["cue_side"]) == int(clean_event["cue_side"])
        )
        if not side_matches:
            if match_idx is not None:
                delta_s = float(delta_s)
            mapping_rows.append(
                {
                    "record": record,
                    "q1_trial_index": clean_event["q1_trial_index"],
                    "q1_trial_key": clean_event["q1_trial_key"],
                    "q1_cue_time_s": clean_event["cue_time_s"],
                    "q1_cue_side": clean_event["cue_side"],
                    "original_trial_index": None,
                    "mapping_error_s": delta_s,
                    "mapping_status": "unmatched_or_cue_side_mismatch",
                }
            )
            continue
        raw_event = events[match_idx]
        used_raw.add(match_idx)
        mapping[match_idx] = {
            **clean_event,
            "mapping_error_s": delta_s,
        }
        mapping_rows.append(
            {
                "record": record,
                "q1_trial_index": clean_event["q1_trial_index"],
                "q1_trial_key": clean_event["q1_trial_key"],
                "q1_cue_time_s": clean_event["cue_time_s"],
                "q1_cue_side": clean_event["cue_side"],
                "original_trial_index": raw_event["original_trial_index"],
                "mapping_error_s": delta_s,
                "mapping_status": "matched_by_timestamp_and_cue_side",
            }
        )

    event_rows: list[dict[str, Any]] = []
    for event in events:
        original_index = int(event["original_trial_index"])
        q1_match = mapping.get(original_index)
        response_info = response_by_trial[original_index]
        response = response_info["response"]
        decoded_response = response_info["decoded_response"]
        response_count = int(response_info["response_event_count"])
        response_code = int(response["response_code"]) if response else 0
        response_time = float(response["response_time_s"]) if response else np.nan
        target_side = int(event["cue_side"])
        response_side = decoded_response["choice_side"]
        direction_consistent = int(response_side == target_side) if response_side is not None else np.nan
        behavior_labels = classify_channel9_behavior(
            cue_side=target_side,
            response_side=response_side,
            cue_time_s=event["cue_time_s"],
            response_time_s=response_time if response is not None else None,
            observation_end_time_s=response_info["cue_interval_observation_end_time_s"],
        )
        target_time_proxy = float(event["cue_time_s"] + TARGET_OFFSET_S)
        schedule_rt_proxy = response_time - target_time_proxy if response else np.nan
        if response is None:
            rt_status = "no_response_marker_observed_deadline_unknown"
        elif schedule_rt_proxy < 0:
            rt_status = "before_assumed_target_schedule_proxy"
        elif schedule_rt_proxy < 0.10:
            rt_status = "under_100ms_from_schedule_proxy_review_anchor"
        else:
            rt_status = "schedule_proxy_only_not_verified_rt"
        quality_info = quality_by_trial.get(original_index, {})
        q1_drop = quality_info.get("q1_final_drop")
        if q1_match is not None:
            eeg_quality = "retained_by_q1"
        elif q1_drop == 1:
            eeg_quality = "excluded_by_q1"
        elif q1_drop == 0:
            eeg_quality = "q1_csv_retained_but_clean_mat_unmatched"
        else:
            eeg_quality = "not_found_in_q1_clean_or_quality_csv_unavailable"
        event_rows.append(
            {
                "record": record,
                **event,
                "cue_side_text": "left" if event["cue_side"] < 0 else "right",
                # Keep the legacy field name as a schedule-only target proxy.
                "target_time_s": target_time_proxy,
                "target_time_source": TARGET_OFFSET_SOURCE,
                "task_type": "unresolved_from_allowed_channels",
                "q1_trial_index": q1_match["q1_trial_index"] if q1_match else None,
                "q1_trial_key": q1_match["q1_trial_key"] if q1_match else "",
                "q1_mapping_error_s": q1_match["mapping_error_s"] if q1_match else np.nan,
                "q1_final_drop": q1_drop,
                "eeg_quality": eeg_quality,
                "response_channel_label": raw["response_label"],
                "response_event_count": response_count,
                "response_raw": response["response_raw"] if response else np.nan,
                "response_code": response_code,
                # Channel 9's zero-to-nonzero edge is the response time t_act.
                # Keep the legacy name as an alias for existing readers.
                "t_act_s": response_time,
                "response_time_s": response_time,
                "response_side": "left" if response_side == -1 else "right" if response_side == 1 else "",
                "choice_side": response_side if response_side is not None else np.nan,
                "choice_side_text": "left" if response_side == -1 else "right" if response_side == 1 else "",
                "response_decode_status": decoded_response["decode_status"],
                "response_declared_code": decoded_response["declared_response_code"],
                "response_bout_code_sequence": decoded_response["bout_code_sequence"],
                **response_info["response_timing"],
                "cue_direction": target_side,
                "cue_interval_observation_end_time_s": response_info[
                    "cue_interval_observation_end_time_s"
                ],
                **behavior_labels,
                "target_time_schedule_proxy_s": target_time_proxy,
                "scheduled_rt_proxy_s": schedule_rt_proxy,
                # RT duration needs a verified target onset, which is not
                # available in the raw event channels.
                "reaction_time_s": np.nan,
                "rt_status": rt_status,
                "cue_response_direction_consistent": direction_consistent,
                "direction_consistency_status": (
                    "same_or_opposite_direction_used_as_correctness_by_user_rule"
                    if response_side is not None
                    else "direction_unresolved"
                ),
                "cue_interval_action_marker_count": response_count,
                "has_channel9_action_marker_in_cue_interval": int(response_count > 0),
                "response_marker_present": response is not None,
                "behavior_label_status": (
                    "channel9_response_direction_unresolved"
                    if response_count > 0 and response_side is None
                    else "multiple_channel9_action_edges_in_cue_interval"
                    if response_count > 1
                    else "cue_response_direction_comparable"
                    if response_count == 1
                    else "no_channel9_action_edge_in_cue_interval"
                ),
            }
        )

    record_info = {
        "record": record,
        "raw_path": str(raw["path"]),
        "q1_clean_path": str(clean["path"]),
        "raw_sample_rate_hz": sr,
        "q1_sample_rate_hz": clean["sample_rate_hz"],
        "raw_samples": int(cue.size),
        "raw_duration_s": float(timestamps[-1] - timestamps[0]) if timestamps.size else np.nan,
        "timestamp_median_step_s": median_dt,
        "timestamp_step_relative_error": abs(median_dt - 1.0 / sr) / (1.0 / sr)
        if np.isfinite(median_dt)
        else np.nan,
        "cue_event_count": len(events),
        "response_event_count": len(response_events),
        "response_channel_label": raw["response_label"],
        "response_trial_count": sum(info["response_event_count"] > 0 for info in response_by_trial.values()),
        "multiple_response_trial_count": sum(info["response_event_count"] > 1 for info in response_by_trial.values()),
        "no_response_marker_count": sum(info["response_event_count"] == 0 for info in response_by_trial.values()),
        "scheduled_rt_proxy_median_s": float(np.nanmedian([row["scheduled_rt_proxy_s"] for row in event_rows]))
        if any(np.isfinite(row["scheduled_rt_proxy_s"]) for row in event_rows)
        else np.nan,
        "scheduled_rt_proxy_under_100ms_count": sum(
            row["rt_status"] == "under_100ms_from_schedule_proxy_review_anchor"
            for row in event_rows
        ),
        "q1_clean_trial_count": len(clean["clean_events"]),
        "q1_timestamp_mapping_count": len(mapping),
        "q1_quality_csv_order_validated": quality_order_validated,
        "target_time_source": TARGET_OFFSET_SOURCE,
        "task_type": "unresolved_from_allowed_channels",
        "selected_eeg_channels": ",".join(("F3", "Fz", "F4")),
        "selected_behavior_channels": ",".join(("VisCue", raw["response_label"], "TimeStamp")),
    }
    return pd.DataFrame(event_rows), pd.DataFrame(mapping_rows), {**record_info, "clean": clean}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")
