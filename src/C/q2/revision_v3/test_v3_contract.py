"""Tests specifying the corrected input, population, and sensor interfaces."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from head_model import build_sensor_leadfield
from model import (_causal_history_sample, calibrate_drive_scales,
                   project_frontend, simulate_forward)
from observation import filter_resample_baseline
from fit import _optimize_start


def _frontend(n=201, early=0.01, shape=0.2, left=0.03, right=0.0):
    b = np.full((8, 8, n), early, dtype=np.float32)
    s = np.full((8, 8, n), shape, dtype=np.float32)
    return {"time_ms": np.arange(n, dtype=float), "B": b, "shape": s,
            "H_L": s.copy(), "H_R": s.copy(),
            "dir_left": np.full(n, left, dtype=np.float32),
            "dir_right": np.full(n, right, dtype=np.float32)}


def test_fixed_visual_calibration_puts_groups_on_a_shared_reference_scale():
    left = _frontend()
    right = _frontend(left=0.0, right=0.03)
    scales = calibrate_drive_scales({"left": left, "right": right})
    drives = np.stack([project_frontend(front, drive_scales=scales)
                       for front in (left, right)])
    active = np.arange(201) <= 200
    rms = np.sqrt(np.mean(drives[:, :, :, active] ** 2, axis=(0, 3)))
    np.testing.assert_allclose(rms, np.ones_like(rms), rtol=1e-5, atol=1e-5)


def test_head_geometry_generates_a_referenced_full_rank_three_sensor_map():
    lead = build_sensor_leadfield()
    assert lead.shape == (3, 6)
    assert np.isfinite(lead).all()
    assert np.linalg.matrix_rank(lead) == 3
    np.testing.assert_allclose(lead[0, 0], lead[2, 1], rtol=1e-8, atol=1e-10)


def test_early_population_drives_delayed_shape_populations():
    front = _frontend(n=301, early=0.05, shape=0.0, left=0.0, right=0.0)
    front["shape"].fill(0.0)
    front["H_L"].fill(0.0)
    front["H_R"].fill(0.0)
    result = simulate_forward(front, drive_scales=np.ones((3, 2)))
    assert result.excitatory[1:, :, 60:].max() > 0.0
    assert result.excitatory[1:, :, :20].max() == 0.0


def test_fractional_feedback_reads_only_the_causal_delayed_history():
    history = np.array([[0.0, 10.0, 20.0, 30.0]])
    np.testing.assert_allclose(_causal_history_sample(history, 2, 1.5), [5.0])
    np.testing.assert_allclose(_causal_history_sample(history, 1, 1.5), [0.0])


def test_sensor_voltage_is_computed_from_source_currents_and_leadfield():
    front = _frontend(n=101)
    result = simulate_forward(front, drive_scales=np.ones((3, 2)))
    np.testing.assert_allclose(result.eeg,
                               result.diagnostics["leadfield"] @ result.source_proxy,
                               rtol=1e-6, atol=1e-10)


def test_observation_operator_matches_q1_filter_resample_and_baseline():
    from scipy.signal import butter, filtfilt, resample

    fs = 256.0
    time = -1.0 + np.arange(1024) / fs
    signal = np.stack([np.exp(-((time - 0.3) / 0.12) ** 2),
                       np.sin(2 * np.pi * 3 * time),
                       np.cos(2 * np.pi * 5 * time)])
    observed, observed_time = filter_resample_baseline(signal, fs)
    b, a = butter(4, [0.2, 24.0], btype="bandpass", fs=fs)
    expected = resample(filtfilt(b, a, signal, axis=-1), 512, axis=-1)
    out_time = -1.0 + np.arange(512) / 128.0
    baseline = (out_time >= -0.2) & (out_time < 0.0)
    expected -= expected[:, baseline].mean(axis=1, keepdims=True)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(observed_time, out_time, rtol=0.0, atol=1e-12)


def test_optimizer_budget_exhaustion_does_not_reuse_a_stale_objective_value():
    result = _optimize_start(lambda x: (float(np.sum((x - 2.0) ** 2)), 1.0),
                             np.array([0.0, 0.0]), np.array([-3.0, -3.0]),
                             np.array([3.0, 3.0]), max_nfev=1)
    assert result["evaluations"] == 1
    assert result["optimizer"].success is False
    assert result["candidate"][0] == pytest.approx(8.0)
