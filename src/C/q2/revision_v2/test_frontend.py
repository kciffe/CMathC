"""Numerical equivalence checks for batched spatial sampling."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frontend import (_sample_many_shifted, _sample_shifted, _shape_support,
                      load_stimulus, mirror_stage1_frontend, simulate_frontend)


def test_batched_shifted_sampling_matches_pointwise_reference_including_edges():
    rng = np.random.default_rng(9)
    frame = rng.normal(size=(13, 11)).astype(np.float32)
    rr, cc = np.mgrid[:frame.shape[0], :frame.shape[1]].astype(np.float32)
    base_coords = (rr, cc)
    offsets = np.array([[0.0, 0.0], [0.25, -1.75], [-3.2, 2.4], [8.0, -9.0]])

    actual = _sample_many_shifted(frame, offsets, base_coords)
    expected = np.stack([_sample_shifted(frame, dr, dc, base_coords)
                         for dr, dc in offsets])

    assert actual.shape == (len(offsets), *frame.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_shape_support_is_cached_by_resolution():
    first = _shape_support(64)
    second = _shape_support(64)
    other = _shape_support(128)

    assert first is second
    assert first is not other


def test_mirrored_stage1_frontend_matches_independent_right_simulation():
    time_ms = np.arange(0.0, 33.0)
    left_stimulus = load_stimulus("Stage1", "left")
    right_stimulus = load_stimulus("Stage1", "right")
    left = simulate_frontend(left_stimulus, time_ms=time_ms, resolution=64)
    expected = simulate_frontend(right_stimulus, time_ms=time_ms, resolution=64)
    actual = mirror_stage1_frontend(left, right_stimulus)

    for key in ("lgn_on", "lgn_off", "gabor", "B", "H_L", "H_R"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-4, atol=2e-7)
    np.testing.assert_allclose(actual["lgn_on_mean"], expected["lgn_on_mean"],
                               rtol=2e-5, atol=2e-7)
