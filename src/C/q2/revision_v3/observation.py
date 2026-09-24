"""Match Q1 filtering, resampling, and baseline handling before comparison."""
import numpy as np
from scipy.signal import butter, filtfilt, resample


def filter_resample_baseline(signal, source_fs=256.0, target_fs=128.0,
                             baseline_s=(-0.2, 0.0), highpass_hz=0.2,
                             lowpass_hz=24.0, order=4, start_s=-1.0):
    """Apply Q1 Butterworth/filtfilt, Fourier resampling and per-channel baseline.

    ``signal`` is channels x samples over the complete [-1, 3) s epoch.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 2 or x.shape[1] < 32 or not np.isfinite(x).all():
        raise ValueError("signal must be finite channels x full-epoch samples")
    ratio = source_fs / target_fs
    if source_fs <= 2 * lowpass_hz or target_fs <= 0 or not np.isclose(ratio, round(ratio)):
        raise ValueError("invalid filter or integer downsampling rates")
    b, a = butter(order, [highpass_hz, lowpass_hz], btype="bandpass", fs=source_fs)
    filtered = filtfilt(b, a, x, axis=-1)
    count = int(round(x.shape[1] / ratio))
    observed = resample(filtered, count, axis=-1)
    time_s = start_s + np.arange(count) / target_fs
    baseline = (time_s >= baseline_s[0]) & (time_s < baseline_s[1])
    if baseline.sum() < 2:
        raise ValueError("baseline interval has fewer than two output samples")
    observed -= observed[:, baseline].mean(axis=1, keepdims=True)
    return observed, time_s


def model_curve_to_q1_grid(sensor_signal, time_ms, source_fs=256.0,
                           target_fs=128.0, epoch_s=(-1.0, 3.0)):
    """Pad a modeled cue response into Q1's full epoch and process it identically."""
    values = np.asarray(sensor_signal, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] != time_ms.size:
        raise ValueError("sensor signal must be [F3/Fz/F4,time]")
    if not np.all(np.diff(time_ms) > 0):
        raise ValueError("model time must be strictly increasing")
    count = int(round((epoch_s[1] - epoch_s[0]) * source_fs))
    time_s = epoch_s[0] + np.arange(count) / source_fs
    full_epoch = np.stack([np.interp(time_s * 1000.0, time_ms, row, left=0.0, right=0.0)
                           for row in values])
    filtered, observed_time_s = filter_resample_baseline(
        full_epoch, source_fs, target_fs, start_s=epoch_s[0])
    return filtered, observed_time_s * 1000.0
