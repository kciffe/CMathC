import numpy as np
import pytest

from common import (
    classify_channel9_behavior,
    detect_response_events,
    standardize_response_code,
)


def test_action_and_tgtact_sign_codes_map_to_common_left_right_scale():
    raw = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])

    standardized = standardize_response_code(raw)

    np.testing.assert_array_equal(standardized, [-2, -2, 0, 2, 2])
    np.testing.assert_array_equal(raw, [-2, -1, 0, 1, 2])


def test_unknown_response_code_is_rejected_instead_of_assigned_by_sign():
    with pytest.raises(ValueError, match="unsupported response code"):
        standardize_response_code(np.asarray([-3.0, 3.0]))


def test_only_zero_to_nonzero_edges_are_response_events():
    raw = np.asarray([0, 1, 1, -1, 0, -2, -2, 0], dtype=float)
    timestamps = np.arange(raw.size, dtype=float) / 256

    events = detect_response_events(raw, timestamps)

    assert [event["response_sample_index"] for event in events] == [1, 5]
    assert [event["response_code"] for event in events] == [2, -2]


def test_same_side_is_correct_and_same_sign_stage_then_declared_code_is_one_response():
    labels = classify_channel9_behavior(
        cue_side=-1,
        response_side=-1,
        cue_time_s=10.0,
        response_time_s=12.2,
        observation_end_time_s=18.0,
    )

    assert labels["task_correct"] == 1
    assert labels["task_correctness_status"] == "correct_by_same_direction_rule"
    assert labels["response_latency_from_cue_s"] == pytest.approx(2.2)
    assert labels["timely_response"] == 1
    assert labels["timeliness_status"] == "response_within_cue_plus_3s_deadline"


def test_opposite_side_is_incorrect_and_response_after_three_seconds_is_late():
    labels = classify_channel9_behavior(
        cue_side=1,
        response_side=-1,
        cue_time_s=10.0,
        response_time_s=13.1,
        observation_end_time_s=18.0,
    )

    assert labels["task_correct"] == 0
    assert labels["task_correctness_status"] == "incorrect_by_opposite_direction_rule"
    assert labels["timely_response"] == 0
    assert labels["timeliness_status"] == "response_after_cue_plus_3s_deadline"


def test_no_response_is_untimely_only_when_recording_covers_the_deadline():
    covered = classify_channel9_behavior(
        cue_side=1,
        response_side=None,
        cue_time_s=10.0,
        response_time_s=None,
        observation_end_time_s=13.5,
    )
    censored = classify_channel9_behavior(
        cue_side=1,
        response_side=None,
        cue_time_s=10.0,
        response_time_s=None,
        observation_end_time_s=12.9,
    )

    assert covered["timely_response"] == 0
    assert covered["timeliness_status"] == "no_response_by_cue_plus_3s_deadline"
    assert np.isnan(covered["task_correct"])
    assert np.isnan(censored["timely_response"])
    assert censored["timeliness_status"] == "deadline_not_observed"
