from pathlib import Path
import importlib
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def model():
    assert (Path(__file__).parent / "model.py").exists(), "revision model is not implemented"
    return importlib.import_module("model")


def test_stage_duration_is_explicit():
    m = model()
    t = np.array([0., 199., 200., 799., 800.])
    np.testing.assert_array_equal(m.stimulus_gate(t, "Stage1"), [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(m.stimulus_gate(t, "Stage2"), np.ones(5))


def test_configuration_detects_arrangement_with_equal_orientation_totals():
    m = model()
    a = np.zeros((4, 8, 8, 1))
    b = a.copy()
    a[1, 3, 3] = a[3, 3, 4] = 1
    b[3, 3, 3] = b[1, 3, 4] = 1
    np.testing.assert_array_equal(a.sum(axis=(1, 2)), b.sum(axis=(1, 2)))
    assert not np.allclose(m.configuration_features(a), m.configuration_features(b))


def test_zero_contrast_has_no_evoked_eeg():
    m = model()
    out = m.simulate(np.zeros((64, 64)), "Stage1")
    np.testing.assert_allclose(out["eeg"], 0, atol=1e-12)


def test_fixed_mapping_has_only_two_modes():
    m = model()
    assert np.linalg.matrix_rank(m.LEAD_FIELD) == 2
    x = np.random.default_rng(2).normal(size=(6, 40))
    y = m.LEAD_FIELD @ x
    np.testing.assert_allclose(y[0] - 2 * y[1] + y[2], 0, atol=1e-12)
