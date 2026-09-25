"""Shared MAT, event-mapping, and output helpers for the Q3 scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from config import (
    ALLOWED_CHANNELS,
    OUTPUT_DIR,
    Q1_CLEAN_DIR,
    RECORDS,
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


def _selected_channel_indices(labels: list[str]) -> dict[str, int]:
    """Return indices only for explicitly allowed EEG, cue, and time signals."""
    selected: dict[str, int] = {}
    for index, label in enumerate(labels):
        name = channel_name(label)
        if name in ALLOWED_CHANNELS and name not in selected:
            selected[name] = index
    missing = sorted(ALLOWED_CHANNELS.difference(selected))
    if missing:
        raise ValueError(f"MAT file is missing allowed channel labels: {missing}")
    return selected


def find_record_path(record: str) -> Path:
    candidates = (
        Path(__file__).resolve().parents[3]
        / "data"
    ).rglob(f"{record}.mat")
    path = next(candidates, None)
    if path is None:
        raise FileNotFoundError(f"Could not find raw MAT file for {record}")
    return path


def load_raw_record(record: str) -> dict[str, Any]:
    """Load only the allowed source rows from one continuous MAT record."""
    path = find_record_path(record)
    source = loadmat(path, squeeze_me=False, struct_as_record=False)
    data_key = "data" if "data" in source else "Data" if "Data" in source else None
    if data_key is None or "DataLabel" not in source or "SampleRate" not in source:
        raise ValueError(f"Expected data, DataLabel, and SampleRate in {path.name}")

    all_labels = matlab_labels(source["DataLabel"])
    selected = _selected_channel_indices(all_labels)
    matrix = np.asarray(source[data_key], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected channel x sample matrix in {path.name}, got {matrix.shape}")
    if matrix.shape[0] != len(all_labels):
        raise ValueError(
            f"DataLabel count {len(all_labels)} does not match matrix rows {matrix.shape[0]}"
        )

    channels = {name: matrix[index, :].copy() for name, index in selected.items()}
    sample_rate = float(np.asarray(source["SampleRate"]).squeeze())
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in {path.name}: {sample_rate}")
    return {
        "record": record,
        "path": path,
        "sample_rate_hz": sample_rate,
        "channels": channels,
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
                "target_time_s": event["cue_time_s"] + TARGET_OFFSET_S,
                "target_time_source": TARGET_OFFSET_SOURCE,
                "task_type": "unresolved_from_allowed_channels",
                "q1_trial_index": q1_match["q1_trial_index"] if q1_match else None,
                "q1_trial_key": q1_match["q1_trial_key"] if q1_match else "",
                "q1_mapping_error_s": q1_match["mapping_error_s"] if q1_match else np.nan,
                "q1_final_drop": q1_drop,
                "eeg_quality": eeg_quality,
                "behavior_label_status": "unavailable_no_external_behavior_labels",
                "response_side": np.nan,
                "reaction_time_s": np.nan,
                "correct": np.nan,
                "is_omission": np.nan,
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
        "q1_clean_trial_count": len(clean["clean_events"]),
        "q1_timestamp_mapping_count": len(mapping),
        "q1_quality_csv_order_validated": quality_order_validated,
        "target_time_source": TARGET_OFFSET_SOURCE,
        "task_type": "unresolved_from_allowed_channels",
        "selected_channels": ",".join((*("F3", "Fz", "F4"), "VisCue", "TimeStamp")),
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
