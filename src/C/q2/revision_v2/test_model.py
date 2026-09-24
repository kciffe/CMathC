"""Behavioral checks for the v2 low-dimensional forward model."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import project_frontend, simulate_forward


def _empty_frontend(n=9):
    z = np.zeros((8, 8, n), dtype=np.float32)
    return {"time_ms": np.arange(n, dtype=float), "B": z.copy(), "shape": z.copy(),
            "H_L": z.copy(), "H_R": z.copy(), "dir_left": np.zeros(n, dtype=np.float32),
            "dir_right": np.zeros(n, dtype=np.float32)}


def test_legacy_projection_preserves_left_right_half_field_swap():
    arrays = _empty_frontend(2)
    arrays["H_L"][:, :2, :] = 1.0
    arrays["H_R"][:, 6:, :] = 1.0

    projected = project_frontend(arrays, feature_route="legacy")

    assert projected.shape == (3, 2, 2)
    assert np.all(projected[1, 0] > projected[1, 1])
    assert np.all(projected[2, 1] > projected[2, 0])


def test_zero_drive_has_zero_population_and_sensor_response():
    result = simulate_forward(_empty_frontend())

    assert result.excitatory.shape == (3, 2, 9)
    assert result.inhibitory.shape == (3, 2, 9)
    assert result.observable_modes.shape == (3, 9)
    np.testing.assert_allclose(result.excitatory, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.inhibitory, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.eeg, 0.0, atol=1e-12)


def test_rank_two_is_an_explicit_fixed_u2_ablation():
    assert np.linalg.matrix_rank(config.U_MATRIX) == 3
    frontend = _empty_frontend(41)
    frontend["shape"][:, :, :] = 1.0
    frontend["dir_left"] = np.linspace(0, 1, 41, dtype=np.float32)
    rank2 = simulate_forward(frontend, observation_rank=2)
    rank3 = simulate_forward(frontend, observation_rank=3)
    negative_gain = simulate_forward(frontend, amplitude=-2.0)
    assert np.max(np.abs(config.U2 @ rank2.eeg)) < 1e-6
    assert np.max(np.abs(config.U2 @ rank3.eeg)) > 1e-8
    np.testing.assert_allclose(negative_gain.eeg_scaled, -2.0 * negative_gain.eeg,
                               rtol=1e-6, atol=1e-8)
