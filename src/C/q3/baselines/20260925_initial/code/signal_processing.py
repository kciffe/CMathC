"""Raw continuous EEG processing primitives for the Q3 pipeline.

These helpers operate only on the three prespecified EEG electrodes. Event and
response channels are intentionally kept outside every feature function.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from config import HARD_CLIP_THRESHOLD_RAW


EEG_BANDS_HZ = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


def apply_bandpass(
    eeg: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    order: int = 4,
) -> np.ndarray:
    """Filter a continuous channel-by-sample EEG block with zero-phase SOS."""
    values = np.asarray(eeg, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError("EEG input must have shape (samples,) or (channels, samples)")
    if values.shape[-1] < 16:
        raise ValueError("EEG record is too short for zero-phase filtering")
    if not np.isfinite(values).all():
        raise ValueError("EEG input contains non-finite samples")
    low_hz, high_hz = map(float, band_hz)
    nyquist_hz = float(sample_rate_hz) / 2.0
    if not (0 < low_hz < high_hz < nyquist_hz):
        raise ValueError(
            f"Band {band_hz} Hz must satisfy 0 < low < high < Nyquist ({nyquist_hz:g} Hz)"
        )
    if order < 1:
        raise ValueError("Filter order must be a positive integer")
    sos = butter(
        int(order), (low_hz, high_hz), btype="bandpass", fs=float(sample_rate_hz), output="sos"
    )
    try:
        return sosfiltfilt(sos, values, axis=-1)
    except ValueError as error:
        raise ValueError(f"Could not filter EEG record: {error}") from error


def extract_epoch(
    eeg: np.ndarray,
    timestamps_s: np.ndarray,
    event_time_s: float,
    window_s: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed-size half-open epoch around the nearest event sample.

    Samples outside the record are padded with NaN. This keeps epoch length
    explicit and lets artifact/QC handling report missing edge coverage.
    """
    values = np.asarray(eeg, dtype=np.float64)
    time = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[-1] != time.size:
        raise ValueError("EEG and timestamp dimensions do not match")
    if time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    step = float(np.median(np.diff(time)))
    sample_rate_hz = 1.0 / step
    start_s, stop_s = map(float, window_s)
    if not start_s < stop_s:
        raise ValueError("Epoch window must have positive duration")
    if not time[0] <= event_time_s <= time[-1]:
        raise ValueError("Event time lies outside the record")

    event_index = int(np.argmin(np.abs(time - float(event_time_s))))
    first_offset = int(round(start_s * sample_rate_hz))
    stop_offset = int(round(stop_s * sample_rate_hz))
    relative_time = np.arange(first_offset, stop_offset, dtype=float) / sample_rate_hz
    epoch = np.full((values.shape[0], relative_time.size), np.nan, dtype=np.float64)
    source_first = max(event_index + first_offset, 0)
    source_stop = min(event_index + stop_offset, values.shape[-1])
    if source_stop > source_first:
        dest_first = source_first - (event_index + first_offset)
        dest_stop = dest_first + (source_stop - source_first)
        epoch[:, dest_first:dest_stop] = values[:, source_first:source_stop]
    return epoch, relative_time


def artifact_qc(
    eeg: np.ndarray, clip_threshold: float = HARD_CLIP_THRESHOLD_RAW
) -> dict[str, bool | float | int]:
    """Flag missing, flat, or digitally clipped EEG without modifying samples."""
    values = np.asarray(eeg, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Artifact QC expects channel-by-sample EEG")
    finite = np.isfinite(values)
    has_nonfinite = not bool(finite.all())
    if finite.any():
        finite_values = values[finite]
        max_abs = float(np.max(np.abs(finite_values)))
        peak_to_peak = float(np.max(finite_values) - np.min(finite_values))
        hard_clip = bool(np.any(np.abs(finite_values) >= clip_threshold))
        channel_ranges = []
        for channel_values in values:
            finite_channel = channel_values[np.isfinite(channel_values)]
            if finite_channel.size:
                channel_ranges.append(float(np.max(finite_channel) - np.min(finite_channel)))
        flatline = bool(channel_ranges and any(value <= 1e-8 for value in channel_ranges))
    else:
        max_abs = float("nan")
        peak_to_peak = float("nan")
        hard_clip = False
        flatline = True
    valid = not (has_nonfinite or hard_clip or flatline)
    return {
        "valid": bool(valid),
        "has_nonfinite": bool(has_nonfinite),
        "hard_clip": bool(hard_clip),
        "flatline": bool(flatline),
        "max_abs": max_abs,
        "peak_to_peak": peak_to_peak,
    }


def _integrated_bandpowers(signal: np.ndarray, sample_rate_hz: float) -> dict[str, np.ndarray]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("Bandpower input must have shape (channels, samples)")
    if values.shape[-1] < 32 or not np.isfinite(values).all():
        return {name: np.full(values.shape[0], np.nan) for name in EEG_BANDS_HZ}
    values = values - np.mean(values, axis=-1, keepdims=True)
    flat = np.std(values, axis=-1) <= np.finfo(float).eps
    frequencies, density = welch(
        values,
        fs=float(sample_rate_hz),
        nperseg=values.shape[-1],
        noverlap=0,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    result: dict[str, np.ndarray] = {}
    for name, (low_hz, high_hz) in EEG_BANDS_HZ.items():
        selected = (frequencies >= low_hz) & (frequencies <= high_hz)
        if int(selected.sum()) >= 2:
            power = np.asarray(
                np.trapezoid(density[:, selected], frequencies[selected], axis=-1),
                dtype=np.float64,
            )
            power[flat] = 0.0
            result[name] = power
        else:
            result[name] = np.full(values.shape[0], np.nan)
    return result


def summarize_window_features(
    eeg: np.ndarray, sample_rate_hz: float, prefix: str
) -> tuple[dict[str, float], dict[str, bool | float | int]]:
    """Summarize one continuous trial window without event-channel inputs."""
    values = np.asarray(eeg, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError("Window feature extraction expects three EEG channels")
    qc = artifact_qc(values)
    features: dict[str, float] = {}
    powers = _integrated_bandpowers(values, sample_rate_hz)
    for channel_index, channel in enumerate(("F3", "Fz", "F4")):
        signal = values[channel_index]
        finite_signal = signal[np.isfinite(signal)]
        features[f"{prefix}_mean_{channel}"] = (
            float(np.mean(finite_signal)) if finite_signal.size else float("nan")
        )
        features[f"{prefix}_sd_{channel}"] = (
            float(np.std(finite_signal, ddof=1)) if finite_signal.size > 1 else float("nan")
        )
        for band_name in EEG_BANDS_HZ:
            power = float(powers[band_name][channel_index])
            features[f"{prefix}_log_{band_name}_power_{channel}"] = (
                float(np.log(max(power, np.finfo(float).tiny))) if np.isfinite(power) else float("nan")
            )
    features[f"{prefix}_AI_alpha_log_F4_minus_F3"] = float(
        features[f"{prefix}_log_alpha_power_F4"]
        - features[f"{prefix}_log_alpha_power_F3"]
    )
    qc = {**qc, "n_samples": int(values.shape[-1]), "duration_s": values.shape[-1] / sample_rate_hz}
    return features, qc


def summarize_epoch_features(
    erp_eeg: np.ndarray,
    tf_eeg: np.ndarray,
    relative_time_s: np.ndarray,
    sample_rate_hz: float,
) -> tuple[dict[str, float], dict[str, bool | float | int]]:
    """Extract frontal ERP candidates and time-frequency band powers."""
    erp = np.asarray(erp_eeg, dtype=np.float64)
    tf = np.asarray(tf_eeg, dtype=np.float64)
    time = np.asarray(relative_time_s, dtype=np.float64).reshape(-1)
    if erp.shape != tf.shape or erp.shape != (3, time.size):
        raise ValueError("ERP/TF epochs must share a (3 channels, samples) shape")
    if not np.isfinite(time).all():
        raise ValueError("Relative epoch time must be finite")

    qc_erp = artifact_qc(erp)
    qc_tf = artifact_qc(tf)
    qc: dict[str, bool | float | int] = {
        "valid": bool(qc_erp["valid"] and qc_tf["valid"]),
        "erp_valid": bool(qc_erp["valid"]),
        "tf_valid": bool(qc_tf["valid"]),
        "erp_hard_clip": bool(qc_erp["hard_clip"]),
        "tf_hard_clip": bool(qc_tf["hard_clip"]),
        "erp_nonfinite": bool(qc_erp["has_nonfinite"]),
        "tf_nonfinite": bool(qc_tf["has_nonfinite"]),
        "n_samples": int(time.size),
        "power_window_s": float(np.max(time) - max(0.0, np.min(time))) if time.size else 0.0,
    }

    baseline_mask = (time >= -0.10) & (time < 0.0)
    candidate_mask = (time >= 0.25) & (time < 0.50)
    peak_mask = candidate_mask.copy()
    power_mask = time >= 0.0
    if not baseline_mask.any() or not candidate_mask.any() or not power_mask.any():
        raise ValueError("Epoch does not cover the baseline, ERP candidate, and post-event power windows")

    features: dict[str, float] = {}
    baseline_corrected_means: dict[str, float] = {}
    powers = _integrated_bandpowers(tf[:, power_mask], sample_rate_hz)
    for channel_index, channel in enumerate(("F3", "Fz", "F4")):
        baseline = float(np.mean(erp[channel_index, baseline_mask]))
        candidate = erp[channel_index, candidate_mask] - baseline
        peak_values = erp[channel_index, peak_mask] - baseline
        peak_time = time[peak_mask]
        peak_index = int(np.argmax(peak_values))
        candidate_mean = float(np.mean(candidate))
        baseline_corrected_means[channel] = candidate_mean
        features[f"erp_candidate_mean_{channel}"] = candidate_mean
        features[f"erp_mean_{channel}"] = candidate_mean
        features[f"erp_candidate_area_{channel}"] = float(
            np.trapezoid(candidate, time[candidate_mask])
        )
        features[f"erp_peak_{channel}"] = float(peak_values[peak_index])
        features[f"erp_peak_latency_s_{channel}"] = float(peak_time[peak_index])

        for band_name in EEG_BANDS_HZ:
            power = float(powers[band_name][channel_index])
            features[f"log_{band_name}_power_{channel}"] = (
                float(np.log(max(power, np.finfo(float).tiny))) if np.isfinite(power) else float("nan")
            )

    f3 = baseline_corrected_means["F3"]
    f4 = baseline_corrected_means["F4"]
    features["AI_erp_F4_minus_F3"] = float((f4 - f3) / (abs(f4) + abs(f3) + 1e-12))
    features["AI_alpha_log_F4_minus_F3"] = float(
        features["log_alpha_power_F4"] - features["log_alpha_power_F3"]
    )
    features["AI_alpha"] = features["AI_alpha_log_F4_minus_F3"]
    return features, qc
