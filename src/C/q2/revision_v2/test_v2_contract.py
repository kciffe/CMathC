"""Contracts for the v2 shape-opponent and full sensor-mode readout."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from evaluate import extract_features
from frontend import load_stimulus, mirror_stage1_frontend, simulate_frontend
from model import project_frontend, simulate_forward


def test_sensor_mode_basis_is_full_rank_and_orthonormal():
    assert hasattr(config, "U_MATRIX")
    assert config.U_MATRIX.shape == (3, 3)
    np.testing.assert_allclose(config.U_MATRIX.T @ config.U_MATRIX,
                               np.eye(3), atol=1e-12)


def test_shape_opponent_maps_swap_under_exact_reflection():
    time_ms = np.arange(0.0, 41.0)
    left_stimulus = load_stimulus("Stage1", "left")
    right_stimulus = load_stimulus("Stage1", "right")
    left = simulate_frontend(left_stimulus, time_ms=time_ms, resolution=64)
    right = mirror_stage1_frontend(left, right_stimulus)

    assert {"shape", "direction", "R_L", "R_R", "dir_left", "dir_right"} <= set(left)
    np.testing.assert_allclose(right["shape"], left["shape"][:, ::-1], atol=1e-6)
    np.testing.assert_allclose(right["direction"], -left["direction"][:, ::-1], atol=1e-6)
    np.testing.assert_allclose(right["R_L"], left["R_R"], atol=1e-6)
    np.testing.assert_allclose(right["R_R"], left["R_L"], atol=1e-6)
    assert np.all(left["dir_left"] >= 0.0)
    assert np.all(left["dir_right"] >= 0.0)
    assert left["dir_left"].max() > 5.0 * left["dir_right"].max()


def test_opponent_route_and_full_rank_readout_reconstruct_sensor_modes():
    n_time = 31
    yy, xx = np.mgrid[:8, :8]
    shape = np.ones((8, 8, n_time), dtype=np.float32)
    frontend = {
        "time_ms": np.arange(n_time, dtype=float),
        "B": np.broadcast_to((xx < 4).astype(np.float32)[..., None], (8, 8, n_time)).copy(),
        "shape": shape,
        "dir_left": np.linspace(0.0, 1.0, n_time, dtype=np.float32),
        "dir_right": np.zeros(n_time, dtype=np.float32),
    }

    drives = project_frontend(frontend, feature_route="opponent")
    assert drives.shape == (3, 2, n_time)
    result = simulate_forward(frontend, feature_route="opponent", observation_rank=3)
    assert result.observable_modes.shape == (3, n_time)
    assert result.eeg.shape == (3, n_time)
    np.testing.assert_allclose(config.U_MATRIX.T @ result.eeg,
                               result.observable_modes, rtol=1e-5, atol=1e-6)

    rank2 = simulate_forward(frontend, feature_route="opponent", observation_rank=2)
    assert np.max(np.abs(config.U2 @ rank2.eeg)) < 1e-6


def test_real_feature_extractor_can_compare_six_and_nine_mode_features():
    time_ms = np.arange(0.0, 701.0, 5.0)
    trials = np.zeros((4, 3, len(time_ms)), dtype=float)
    trials[0::2, 0] = -1.0
    trials[0::2, 2] = 1.0
    trials[1::2, 0] = 1.0
    trials[1::2, 2] = -1.0

    six = extract_features(trials, time_ms, mode_count=2)
    nine = extract_features(trials, time_ms, mode_count=3)

    assert six.shape == (4, 6)
    assert nine.shape == (4, 9)
    np.testing.assert_allclose(six, nine[:, :6])
