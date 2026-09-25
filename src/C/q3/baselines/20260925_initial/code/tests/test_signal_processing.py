import numpy as np
import signal_processing

from signal_processing import (
    apply_bandpass,
    artifact_qc,
    extract_epoch,
    summarize_epoch_features,
)


def test_bandpass_validates_nyquist_and_attenuates_out_of_band():
    sample_rate = 256.0
    time = np.arange(4096) / sample_rate
    signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 45 * time)
    filtered = apply_bandpass(signal[None, :], sample_rate, (1.0, 30.0))[0]

    assert np.std(filtered) < 0.8
    assert np.std(filtered) > 0.5

    try:
        apply_bandpass(signal[None, :], sample_rate, (1.0, 130.0))
    except ValueError as error:
        assert "Nyquist" in str(error)
    else:
        raise AssertionError("A passband above Nyquist must be rejected")


def test_epoch_uses_nearest_event_sample_and_exact_half_open_window():
    sample_rate = 100.0
    timestamps = np.arange(500) / sample_rate
    signal = np.arange(500, dtype=float)[None, :]
    epoch, relative_time = extract_epoch(
        signal, timestamps, event_time_s=1.005, window_s=(-0.10, 0.20)
    )

    assert epoch.shape == (1, 30)
    np.testing.assert_allclose(relative_time[[0, -1]], [-0.10, 0.19], atol=1e-12)
    assert epoch[0, 0] == 90
    assert epoch[0, -1] == 119


def test_feature_summary_returns_candidate_erp_and_all_requested_bands():
    sample_rate = 256.0
    relative_time = np.arange(-0.125, 0.875, 1 / sample_rate)
    relative_time = relative_time[: int(round(1.0 * sample_rate))]
    channels = []
    for amplitude in (1.0, 2.0, 3.0):
        channels.append(
            0.5 * np.sin(2 * np.pi * 10 * relative_time)
            + amplitude * (relative_time >= 0.25) * (relative_time < 0.50)
        )
    eeg = np.asarray(channels)

    features, qc = summarize_epoch_features(eeg, eeg, relative_time, sample_rate)

    assert qc["valid"] is True
    assert features["erp_candidate_mean_F4"] > features["erp_candidate_mean_F3"]
    assert 0.25 <= features["erp_peak_latency_s_Fz"] < 0.50
    for band in ("theta", "alpha", "beta", "gamma"):
        for channel in ("F3", "Fz", "F4"):
            assert np.isfinite(features[f"log_{band}_power_{channel}"])
    assert np.isfinite(features["AI_erp_F4_minus_F3"])


def test_artifact_qc_flags_clipping_but_does_not_mutate_input():
    eeg = np.zeros((3, 64), dtype=float)
    eeg[1, 30] = 1000.0
    before = eeg.copy()

    flags = artifact_qc(eeg)

    assert flags["hard_clip"] is True
    assert flags["valid"] is False
    np.testing.assert_array_equal(eeg, before)


def test_artifact_qc_flags_a_single_flat_channel():
    time = np.arange(64)
    eeg = np.asarray([np.zeros(64), np.sin(time), np.cos(time)])

    flags = artifact_qc(eeg)

    assert flags["flatline"] is True
    assert flags["valid"] is False


def test_all_bandpowers_share_one_welch_spectrum_per_channel(monkeypatch):
    sample_rate = 256.0
    time = np.arange(256) / sample_rate - 0.125
    eeg = np.asarray(
        [np.sin(2 * np.pi * frequency * time) for frequency in (5.0, 10.0, 20.0)]
    )
    calls = []
    original_welch = signal_processing.welch

    def counting_welch(*args, **kwargs):
        calls.append(1)
        return original_welch(*args, **kwargs)

    monkeypatch.setattr(signal_processing, "welch", counting_welch)
    signal_processing.summarize_epoch_features(eeg, eeg, time, sample_rate)

    assert len(calls) == 1
