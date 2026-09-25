"""Filter full continuous raw records and save independent Q3 EEG epochs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import ensure_output_dir, load_raw_record, write_json
from config import (
    EEG_CHANNELS,
    ERP_FILTER_BAND_HZ,
    FILTER_ORDER,
    RAW_CUE_EPOCH_WINDOW_S,
    RAW_SAMPLE_RATE_HZ,
    RAW_TARGET_EPOCH_WINDOW_S,
    RECORDS,
    TARGET_OFFSET_SENSITIVITY_S,
    TARGET_OFFSET_S,
    TF_FILTER_BAND_HZ,
)
from signal_processing import apply_bandpass


def _sample_index(timestamps_s: np.ndarray, event_time_s: float) -> int:
    return int(np.argmin(np.abs(timestamps_s - float(event_time_s))))


def _cut_fixed(
    eeg: np.ndarray,
    timestamps_s: np.ndarray,
    event_time_s: float,
    window_s: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    from signal_processing import extract_epoch

    epoch, relative_time = extract_epoch(eeg, timestamps_s, event_time_s, window_s)
    finite = np.isfinite(epoch)
    valid_count = int(finite.all(axis=0).sum())
    return epoch.astype(np.float32), relative_time.astype(np.float32), valid_count


def main() -> None:
    output_dir = ensure_output_dir()
    trial_path = output_dir / "trial_table.csv"
    if not trial_path.exists():
        raise FileNotFoundError("Run 02_extract_trials.py first to create trial_table.csv")
    trials = pd.read_csv(trial_path, encoding="utf-8-sig")

    event_rows: list[dict] = []
    erp_epochs: list[np.ndarray] = []
    tf_epochs: list[np.ndarray] = []
    relative_times: list[np.ndarray] = []
    epoch_sample_counts: list[int] = []
    pre_rows: list[dict] = []
    pre_erp: list[np.ndarray] = []
    pre_tf: list[np.ndarray] = []
    pre_lengths: list[int] = []
    fs_by_record: dict[str, float] = {}

    for record in RECORDS:
        raw = load_raw_record(record)
        sample_rate_hz = float(raw["sample_rate_hz"])
        if not np.isclose(sample_rate_hz, RAW_SAMPLE_RATE_HZ, rtol=0, atol=1e-9):
            raise ValueError(
                f"Expected {RAW_SAMPLE_RATE_HZ:g} Hz raw EEG, found {sample_rate_hz:g} Hz in {record}"
            )
        fs_by_record[record] = sample_rate_hz
        timestamps = np.asarray(raw["channels"]["TimeStamp"], dtype=np.float64)
        observed_step = float(np.median(np.diff(timestamps)))
        if not np.isclose(observed_step, 1.0 / sample_rate_hz, rtol=1e-4, atol=1e-8):
            raise ValueError(
                f"Timestamp step {observed_step:g}s disagrees with {sample_rate_hz:g} Hz in {record}"
            )
        eeg = np.stack([raw["channels"][channel] for channel in EEG_CHANNELS], axis=0)
        erp_continuous = apply_bandpass(
            eeg, sample_rate_hz, ERP_FILTER_BAND_HZ, order=FILTER_ORDER
        )
        tf_continuous = apply_bandpass(
            eeg, sample_rate_hz, TF_FILTER_BAND_HZ, order=FILTER_ORDER
        )

        record_trials = trials.loc[trials["record"] == record].sort_values(
            "original_trial_index"
        )
        for row in record_trials.itertuples(index=False):
            cue_time = float(row.cue_time_s)
            for stage, offset in (("cue_locked", 0.0),):
                event_time = cue_time + offset
                erp_epoch, relative_time, n_samples = _cut_fixed(
                    erp_continuous,
                    timestamps,
                    event_time,
                    RAW_CUE_EPOCH_WINDOW_S,
                )
                tf_epoch, _, tf_n_samples = _cut_fixed(
                    tf_continuous,
                    timestamps,
                    event_time,
                    RAW_CUE_EPOCH_WINDOW_S,
                )
                if n_samples != tf_n_samples:
                    raise RuntimeError("ERP and TF branches returned different cue coverage")
                event_rows.append(
                    {
                        "record": record,
                        "original_trial_index": int(row.original_trial_index),
                        "stage": stage,
                        "analysis_variant": "nominal",
                        "cue_time_s": cue_time,
                        "event_time_s": event_time,
                        "event_offset_from_cue_s": offset,
                        "target_offset_s": np.nan,
                        "cue_side": int(row.cue_side),
                        "response_code": int(row.response_code),
                        "q1_trial_index": row.q1_trial_index,
                        "q1_final_drop": row.q1_final_drop,
                        "eeg_quality": row.eeg_quality,
                    }
                )
                erp_epochs.append(erp_epoch)
                tf_epochs.append(tf_epoch)
                relative_times.append(relative_time)
                epoch_sample_counts.append(n_samples)

            for target_offset in TARGET_OFFSET_SENSITIVITY_S:
                event_time = cue_time + float(target_offset)
                erp_epoch, relative_time, n_samples = _cut_fixed(
                    erp_continuous,
                    timestamps,
                    event_time,
                    RAW_TARGET_EPOCH_WINDOW_S,
                )
                tf_epoch, _, tf_n_samples = _cut_fixed(
                    tf_continuous,
                    timestamps,
                    event_time,
                    RAW_TARGET_EPOCH_WINDOW_S,
                )
                if n_samples != tf_n_samples:
                    raise RuntimeError("ERP and TF branches returned different target coverage")
                event_rows.append(
                    {
                        "record": record,
                        "original_trial_index": int(row.original_trial_index),
                        "stage": "target_locked" if np.isclose(target_offset, TARGET_OFFSET_S) else "target_offset_sensitivity",
                        "analysis_variant": f"target_offset_{target_offset:.1f}s",
                        "cue_time_s": cue_time,
                        "event_time_s": event_time,
                        "event_offset_from_cue_s": float(target_offset),
                        "target_offset_s": float(target_offset),
                        "cue_side": int(row.cue_side),
                        "response_code": int(row.response_code),
                        "q1_trial_index": row.q1_trial_index,
                        "q1_final_drop": row.q1_final_drop,
                        "eeg_quality": row.eeg_quality,
                    }
                )
                erp_epochs.append(erp_epoch)
                tf_epochs.append(tf_epoch)
                relative_times.append(relative_time)
                epoch_sample_counts.append(n_samples)

            response_time = (
                float(row.response_time_s) if pd.notna(row.response_time_s) else np.nan
            )
            start_index = _sample_index(timestamps, cue_time)
            if np.isfinite(response_time):
                endpoint_time = response_time - 0.100
                stop_index = int(np.searchsorted(timestamps, endpoint_time, side="left"))
                first_index = start_index
                last_index = min(max(stop_index, first_index), timestamps.size)
                erp_segment = erp_continuous[:, first_index:last_index].astype(np.float32)
                tf_segment = tf_continuous[:, first_index:last_index].astype(np.float32)
                n_pre = int(erp_segment.shape[-1])
                end_status = "marker_minus_100ms"
            else:
                erp_segment = np.empty((len(EEG_CHANNELS), 0), dtype=np.float32)
                tf_segment = np.empty((len(EEG_CHANNELS), 0), dtype=np.float32)
                n_pre = 0
                end_status = "no_response_marker_endpoint_unavailable"
            pre_rows.append(
                {
                    "record": record,
                    "original_trial_index": int(row.original_trial_index),
                    "cue_time_s": cue_time,
                    "response_time_s": response_time,
                    "endpoint_time_s": response_time - 0.100 if np.isfinite(response_time) else np.nan,
                    "endpoint_status": end_status,
                    "window_duration_s": n_pre / sample_rate_hz,
                    "cue_side": int(row.cue_side),
                    "response_code": int(row.response_code),
                    "q1_trial_index": row.q1_trial_index,
                    "q1_final_drop": row.q1_final_drop,
                    "eeg_quality": row.eeg_quality,
                }
            )
            pre_erp.append(erp_segment)
            pre_tf.append(tf_segment)
            pre_lengths.append(n_pre)

    # Epochs have two requested lengths; retain a common padded shape and the
    # exact valid count so all rows remain traceable and no samples are faked.
    max_epoch_samples = max(epoch.shape[-1] for epoch in erp_epochs)
    padded_erp = np.full(
        (len(erp_epochs), len(EEG_CHANNELS), max_epoch_samples), np.nan, dtype=np.float32
    )
    padded_tf = np.full_like(padded_erp, np.nan)
    padded_time = np.full((len(erp_epochs), max_epoch_samples), np.nan, dtype=np.float32)
    for index, (erp_epoch, tf_epoch, relative_time) in enumerate(
        zip(erp_epochs, tf_epochs, relative_times)
    ):
        length = erp_epoch.shape[-1]
        padded_erp[index, :, :length] = erp_epoch
        padded_tf[index, :, :length] = tf_epoch
        padded_time[index, :length] = relative_time
    event_frame = pd.DataFrame(event_rows)
    np.savez_compressed(
        output_dir / "q3_raw_event_epochs.npz",
        erp_signal=padded_erp,
        tf_signal=padded_tf,
        relative_time_s=padded_time,
        sample_count=np.asarray(epoch_sample_counts, dtype=np.int32),
        record=event_frame["record"].to_numpy(dtype="U64"),
        original_trial_index=event_frame["original_trial_index"].to_numpy(dtype=np.int32),
        stage=event_frame["stage"].to_numpy(dtype="U32"),
        analysis_variant=event_frame["analysis_variant"].to_numpy(dtype="U32"),
        cue_time_s=event_frame["cue_time_s"].to_numpy(dtype=np.float64),
        event_time_s=event_frame["event_time_s"].to_numpy(dtype=np.float64),
        event_offset_from_cue_s=event_frame["event_offset_from_cue_s"].to_numpy(dtype=np.float32),
        target_offset_s=event_frame["target_offset_s"].to_numpy(dtype=np.float32),
        cue_side=event_frame["cue_side"].to_numpy(dtype=np.int8),
        response_code=event_frame["response_code"].to_numpy(dtype=np.int8),
        q1_trial_index=pd.to_numeric(event_frame["q1_trial_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32),
        q1_final_drop=pd.to_numeric(event_frame["q1_final_drop"], errors="coerce").fillna(-1).to_numpy(dtype=np.int8),
        eeg_quality=event_frame["eeg_quality"].fillna("unavailable").to_numpy(dtype="U48"),
        channel_order=np.asarray(EEG_CHANNELS, dtype="U8"),
        sample_rate_hz=np.full(len(event_frame), RAW_SAMPLE_RATE_HZ, dtype=np.float32),
        erp_filter_band_hz=np.asarray(ERP_FILTER_BAND_HZ, dtype=np.float32),
        tf_filter_band_hz=np.asarray(TF_FILTER_BAND_HZ, dtype=np.float32),
    )

    max_pre_samples = max(1, max(pre_lengths, default=0))
    padded_pre_erp = np.full(
        (len(pre_erp), len(EEG_CHANNELS), max_pre_samples), np.nan, dtype=np.float32
    )
    padded_pre_tf = np.full_like(padded_pre_erp, np.nan)
    for index, (erp_segment, tf_segment) in enumerate(zip(pre_erp, pre_tf)):
        length = erp_segment.shape[-1]
        if length:
            padded_pre_erp[index, :, :length] = erp_segment
            padded_pre_tf[index, :, :length] = tf_segment
    pre_frame = pd.DataFrame(pre_rows)
    np.savez_compressed(
        output_dir / "q3_pre_response_epochs.npz",
        erp_signal=padded_pre_erp,
        tf_signal=padded_pre_tf,
        sample_count=np.asarray(pre_lengths, dtype=np.int32),
        record=pre_frame["record"].to_numpy(dtype="U64"),
        original_trial_index=pre_frame["original_trial_index"].to_numpy(dtype=np.int32),
        cue_time_s=pre_frame["cue_time_s"].to_numpy(dtype=np.float64),
        response_time_s=pre_frame["response_time_s"].to_numpy(dtype=np.float64),
        endpoint_time_s=pre_frame["endpoint_time_s"].to_numpy(dtype=np.float64),
        window_duration_s=pre_frame["window_duration_s"].to_numpy(dtype=np.float32),
        endpoint_status=pre_frame["endpoint_status"].to_numpy(dtype="U48"),
        cue_side=pre_frame["cue_side"].to_numpy(dtype=np.int8),
        response_code=pre_frame["response_code"].to_numpy(dtype=np.int8),
        q1_trial_index=pd.to_numeric(pre_frame["q1_trial_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32),
        q1_final_drop=pd.to_numeric(pre_frame["q1_final_drop"], errors="coerce").fillna(-1).to_numpy(dtype=np.int8),
        eeg_quality=pre_frame["eeg_quality"].fillna("unavailable").to_numpy(dtype="U48"),
        sample_rate_hz=np.full(len(pre_frame), RAW_SAMPLE_RATE_HZ, dtype=np.float32),
    )

    processing_summary = {
        "source": "raw continuous MAT EEG channels F3/Fz/F4",
        "sampling_rate_hz_by_record": fs_by_record,
        "erp_filter": {
            "type": "zero-phase Butterworth SOS forward-backward",
            "order": FILTER_ORDER,
            "band_hz": list(ERP_FILTER_BAND_HZ),
        },
        "time_frequency_filter": {
            "type": "zero-phase Butterworth SOS forward-backward",
            "order": FILTER_ORDER,
            "band_hz": list(TF_FILTER_BAND_HZ),
        },
        "cue_window_s": list(RAW_CUE_EPOCH_WINDOW_S),
        "target_window_s": list(RAW_TARGET_EPOCH_WINDOW_S),
        "target_offset_sensitivity_s": list(TARGET_OFFSET_SENSITIVITY_S),
        "nominal_target_offset_s": TARGET_OFFSET_S,
        "target_time_source": "schedule assumption; no independent target marker detected in permitted channels",
        "response_endpoint": "response marker time minus 0.100s; response semantics remain subject to protocol verification",
        "ica": "not used: only three frontal EEG channels and no independent EOG reference",
        "q1_clean_epochs": "used only for timestamp mapping and quality-label comparison; not used as the raw EEG feature source",
        "channel_9": "not included in any EEG signal array; only its event times/codes appear as trial metadata",
        "epoch_count": int(len(event_frame)),
        "pre_response_epoch_count": int(len(pre_frame)),
    }
    write_json(processing_summary, output_dir / "raw_signal_processing.json")
    event_frame.to_csv(output_dir / "raw_epoch_index.csv", index=False, encoding="utf-8-sig")
    pre_frame.to_csv(output_dir / "pre_response_epoch_index.csv", index=False, encoding="utf-8-sig")
    print(
        f"Wrote {len(event_frame)} cue/target epochs and {len(pre_frame)} response-endpoint windows; "
        f"ERP={ERP_FILTER_BAND_HZ[0]}-{ERP_FILTER_BAND_HZ[1]} Hz, "
        f"TF={TF_FILTER_BAND_HZ[0]}-{TF_FILTER_BAND_HZ[1]} Hz."
    )


if __name__ == "__main__":
    main()
