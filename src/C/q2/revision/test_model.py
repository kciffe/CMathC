"""Behavioral checks for the low-dimensional forward model."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import project_frontend, simulate_forward


def test_feature_projection_preserves_left_right_half_field_swap():
    arrays = {name: np.zeros((8, 8, 2), dtype=np.float32)
              for name in ("B", "H_L", "H_R")}
    arrays["H_L"][:, :2, :] = 1.0
    arrays["H_R"][:, 6:, :] = 1.0

    projected = project_frontend(arrays)

    assert projected.shape == (3, 2, 2)
    assert np.all(projected[1, 0] > projected[1, 1])
    assert np.all(projected[2, 1] > projected[2, 0])


def test_zero_drive_has_zero_population_and_electrode_response():
    frontend = {name: np.zeros((8, 8, 9), dtype=np.float32)
                for name in ("B", "H_L", "H_R")}
    frontend["time_ms"] = np.arange(9, dtype=float)

    result = simulate_forward(frontend)

    assert result.excitatory.shape == (3, 2, 9)
    assert result.inhibitory.shape == (3, 2, 9)
    np.testing.assert_allclose(result.excitatory, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.inhibitory, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.eeg, 0.0, atol=1e-12)


def test_rank_two_observation_constraint_is_explicit():
    assert np.linalg.matrix_rank(config.LEAD_FIELD) == 2
    x = np.arange(18, dtype=float).reshape(6, 3)
    y = config.LEAD_FIELD @ x
    np.testing.assert_allclose(y[0] - 2.0 * y[1] + y[2], 0.0, atol=1e-12)
