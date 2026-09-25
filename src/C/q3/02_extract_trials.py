"""Create the trial table and save uniformly processed Q1 EEG epochs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from common import ensure_output_dir, load_q1_clean, load_raw_record, write_csv, write_json
from config import (
    BEHAVIOR_LABELS_PATH,
    EEG_CHANNELS,
    FILTER_BAND_HZ,
    FILTER_ORDER,
    PRE_RESPONSE_WINDOW_S,
    RECORDS,
)


def main() -> None:
    output_dir = ensure_output_dir()
    audit_path = output_dir / "event_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError("Run 01_audit_events.py first to create event_audit.csv")
    trial_table = pd.read_csv(audit_path, encoding="utf-8-sig")

    signal_blocks: list[np.ndarray] = []
    time_blocks: list[np.ndarray] = []
    record_ids: list[str] = []
    original_indices: list[int] = []
    q1_indices: list[int] = []
    cue_sides: list[int] = []
    cue_times: list[float] = []
    trial_keys: list[str] = []
    response_signal_blocks: list[np.ndarray] = []
    response_time_blocks: list[np.ndarray] = []
    response_records: list[str] = []
    response_original_indices: list[int] = []
    response_codes: list[int] = []
    response_times: list[float] = []
    response_sample_rates: list[float] = []
    record_sample_rates: dict[str, float] = {}

    for record in RECORDS:
        clean = load_q1_clean(record)
        record_rows = trial_table.loc[trial_table["record"] == record]
        retained = record_rows.loc[record_rows["q1_trial_index"].notna()]
        for row in retained.itertuples(index=False):
            clean_index = int(row.q1_trial_index)
            if clean_index < 0 or clean_index >= clean["eeg"].shape[0]:
                raise IndexError(f"Invalid Q1 trial index {clean_index} for {record}")
            clean_event = clean["clean_events"][clean_index]
            if int(clean_event["cue_side"]) != int(row.cue_side):
                raise ValueError(f"Cue direction mismatch after timestamp map in {record}")
            signal_blocks.append(clean["eeg"][clean_index])
            time_blocks.append(clean["relative_time"][clean_index])
            record_ids.append(record)
            original_indices.append(int(row.original_trial_index))
            q1_indices.append(clean_index)
            cue_sides.append(int(row.cue_side))
            cue_times.append(float(row.cue_time_s))
            trial_keys.append(str(row.q1_trial_key))

    if not signal_blocks:
        raise RuntimeError("No timestamp-matched EEG trials were available")

    signal_array = np.stack(signal_blocks).astype(np.float32, copy=False)
    time_array = np.stack(time_blocks).astype(np.float32, copy=False)
    np.savez_compressed(
        output_dir / "q3_eeg_trials.npz",
        signal=signal_array,
        relative_time=time_array,
        record=np.asarray(record_ids, dtype="U64"),
        original_trial_index=np.asarray(original_indices, dtype=np.int32),
        q1_trial_index=np.asarray(q1_indices, dtype=np.int32),
        cue_side=np.asarray(cue_sides, dtype=np.int8),
        cue_time_s=np.asarray(cue_times, dtype=np.float64),
        q1_trial_key=np.asarray(trial_keys, dtype="U96"),
        channel_order=np.asarray(("F3", "Fz", "F4"), dtype="U8"),
    )

    trial_table["eeg_trial_available"] = trial_table["q1_trial_index"].notna()

    # For response-preceding features, filter each whole continuous record in a
    # single pass and cut a fresh interval from that signal. Never append a raw
    # tail to a Q1-clean epoch.
    for record in RECORDS:
        raw = load_raw_record(record)
        record_rows = trial_table.loc[trial_table["record"] == record].sort_values(
            "original_trial_index"
        )
        eeg_continuous = np.stack(
            [raw["channels"][channel] for channel in EEG_CHANNELS], axis=0
        )
        sample_rate = float(raw["sample_rate_hz"])
        record_sample_rates[record] = sample_rate
        if not 0 < FILTER_BAND_HZ[0] < FILTER_BAND_HZ[1] < sample_rate / 2:
            raise ValueError(f"Invalid Q1 filter band for {record}: {sample_rate} Hz")
        numerator, denominator = butter(
            FILTER_ORDER,
            FILTER_BAND_HZ,
            btype="bandpass",
            fs=sample_rate,
        )
        filtered_continuous = filtfilt(
            numerator, denominator, eeg_continuous, axis=-1
        )
        timestamps = raw["channels"]["TimeStamp"]
        start_offset = int(round(PRE_RESPONSE_WINDOW_S[0] * sample_rate))
        stop_offset = int(round(PRE_RESPONSE_WINDOW_S[1] * sample_rate))
        expected_samples = stop_offset - start_offset

        for row in record_rows.itertuples(index=False):
            response_time = float(row.response_time_s) if pd.notna(row.response_time_s) else np.nan
            if not np.isfinite(response_time):
                segment = np.full((len(EEG_CHANNELS), expected_samples), np.nan, dtype=np.float32)
                rel_time = (np.arange(expected_samples) + start_offset) / sample_rate
                response_code = 0
            else:
                response_sample = int(np.argmin(np.abs(timestamps - response_time)))
                first_sample = response_sample + start_offset
                last_sample = response_sample + stop_offset
                if first_sample < 0 or last_sample > filtered_continuous.shape[1]:
                    segment = np.full((len(EEG_CHANNELS), expected_samples), np.nan, dtype=np.float32)
                    rel_time = (np.arange(expected_samples) + start_offset) / sample_rate
                else:
                    segment = filtered_continuous[:, first_sample:last_sample].astype(
                        np.float32, copy=False
                    )
                    if segment.shape[1] != expected_samples:
                        segment = np.full((len(EEG_CHANNELS), expected_samples), np.nan, dtype=np.float32)
                    rel_time = (np.arange(expected_samples) + start_offset) / sample_rate
                response_code = int(row.response_code)
            response_signal_blocks.append(segment)
            response_time_blocks.append(rel_time.astype(np.float32))
            response_records.append(record)
            response_original_indices.append(int(row.original_trial_index))
            response_codes.append(response_code)
            response_times.append(response_time)
            response_sample_rates.append(sample_rate)

    response_signal_array = np.stack(response_signal_blocks)
    response_time_array = np.stack(response_time_blocks)
    np.savez_compressed(
        output_dir / "response_locked_eeg.npz",
        signal=response_signal_array,
        relative_time=response_time_array,
        record=np.asarray(response_records, dtype="U64"),
        original_trial_index=np.asarray(response_original_indices, dtype=np.int32),
        response_code=np.asarray(response_codes, dtype=np.int8),
        response_time_s=np.asarray(response_times, dtype=np.float64),
        sample_rate_hz=np.asarray(response_sample_rates, dtype=np.float32),
        channel_order=np.asarray(EEG_CHANNELS, dtype="U8"),
        source_filter=np.asarray(
            f"continuous-record Butterworth order {FILTER_ORDER}, {FILTER_BAND_HZ[0]}-{FILTER_BAND_HZ[1]} Hz, filtfilt",
            dtype="U128",
        ),
    )

    trial_table["response_pre_window_available"] = trial_table["response_time_s"].notna()
    if "deadline_s" not in trial_table:
        trial_table["deadline_s"] = np.nan
    if BEHAVIOR_LABELS_PATH.exists():
        behavior = pd.read_csv(BEHAVIOR_LABELS_PATH, encoding="utf-8-sig")
        required_keys = {"record", "original_trial_index"}
        missing_keys = sorted(required_keys.difference(behavior.columns))
        if missing_keys:
            raise ValueError(f"Behavior labels need join keys {missing_keys}")
        if behavior.duplicated(["record", "original_trial_index"]).any():
            raise ValueError("Behavior labels contain duplicate record/trial keys")
        columns_to_join = [
            column
            for column in (
                "record",
                "original_trial_index",
                "choice_side",
                "response_side",
                "reaction_time_s",
                "correct",
                "is_omission",
                "deadline_s",
            )
            if column in behavior.columns
        ]
        rename_external = {
            column: f"external_{column}"
            for column in columns_to_join
            if column not in {"record", "original_trial_index"}
        }
        external = behavior[columns_to_join].rename(columns=rename_external).copy()
        external["original_trial_index"] = pd.to_numeric(
            external["original_trial_index"], errors="coerce"
        )
        external["record"] = external["record"].astype(str)
        external["_external_row_present"] = True
        trial_table = trial_table.merge(
            external,
            on=["record", "original_trial_index"],
            how="left",
            validate="one_to_one",
        )
        trial_table["external_behavior_row_present"] = trial_table["_external_row_present"].fillna(False)
        trial_table["behavior_label_status"] = np.where(
            trial_table["_external_row_present"].fillna(False),
            trial_table["behavior_label_status"] + ";external_row_joined",
            trial_table["behavior_label_status"],
        )
        trial_table = trial_table.drop(columns=["_external_row_present"])
    write_csv(trial_table, output_dir / "trial_table.csv")

    filter_summary = {
        "response_window_relative_to_marker_s": list(PRE_RESPONSE_WINDOW_S),
        "endpoint_interpretation": "window stops approximately 100 ms before channel 9 marker",
        "continuous_processing": f"each full record filtered in one pass, Butterworth order {FILTER_ORDER}, {FILTER_BAND_HZ[0]}-{FILTER_BAND_HZ[1]} Hz, filtfilt",
        "channels": list(EEG_CHANNELS),
        "sample_rate_hz_by_record": record_sample_rates,
    }
    write_json(filter_summary, output_dir / "response_window_processing.json")

    print(
        f"Saved {len(signal_array)} EEG trials ({signal_array.shape[1]} channels, "
        f"{signal_array.shape[2]} samples) and {len(trial_table)} raw cue rows."
    )
    print(
        f"Channel 9 responses: {int(trial_table['choice_side'].notna().sum())} marked, "
        f"{int(trial_table['is_omission'].fillna(False).sum())} without a response marker."
    )
    if BEHAVIOR_LABELS_PATH.exists():
        print("Independent behavior labels were joined by record and original trial index.")
    else:
        print("Response labels come from the standardized channel 9 marker; correctness remains unassigned.")


if __name__ == "__main__":
    main()
