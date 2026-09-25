import numpy as np

from common import detect_response_events, standardize_response_code


def test_action_and_tgtact_sign_codes_map_to_common_left_right_scale():
    raw = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])

    standardized = standardize_response_code(raw)

    np.testing.assert_array_equal(standardized, [-2, -2, 0, 2, 2])
    np.testing.assert_array_equal(raw, [-2, -1, 0, 1, 2])


def test_only_zero_to_nonzero_edges_are_response_events():
    raw = np.asarray([0, 1, 1, 1, 0, -2, -2, 0], dtype=float)
    timestamps = np.arange(raw.size, dtype=float) / 256

    events = detect_response_events(raw, timestamps)

    assert [event["response_sample_index"] for event in events] == [1, 5]
    assert [event["response_code"] for event in events] == [2, -2]
