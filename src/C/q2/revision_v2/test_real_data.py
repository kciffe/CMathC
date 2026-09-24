"""Behavioral checks for conservative real-EEG condition construction."""
from pathlib import Path
import importlib
import sys

import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent))


def real_data():
    assert (Path(__file__).parent / "real_data.py").exists(), "real EEG policy is not implemented"
    return importlib.import_module("real_data")


def clean_fixture(n_trials=24):
    time = -1 + np.arange(512) / 128
    labels = ["Action:L-1/R+1", "F4", "Fz", "F3", "VisCue:L-1/R+1", "TimeStamp"]
    cue = np.resize([-1, 1], n_trials)
    trials = np.zeros((n_trials, len(labels), time.size))
    for channel, amplitude in ((1, 4), (2, 3), (3, 2)):
        trials[:, channel, :] = channel * 10 + amplitude * (time >= 0) + (amplitude + 2) * (time >= 2.2)
    trials[:, 0, time >= 2.2] = cue[:, None]
    cue_marker = (time >= 0) & (time <= 0.1953125)
    trials[:, labels.index("VisCue:L-1/R+1"), cue_marker] = cue[:, None]
    return {
        "trial_data": trials,
        "relative_time": np.tile(time, (n_trials, 1)),
        "cue_type": cue,
        "drop": np.zeros(n_trials),
        "SampleRate": 128,
        "DataLabel": np.asarray(labels, dtype=object)[:, None],
    }


def test_unknown_layout_does_not_depend_on_action_sign():
    module = real_data()
    cue = np.array([-1, 1, -1, 1])
    masks = module.stage_condition_masks("Task2", "Stage2", cue)
    assert set(masks) == {"target_unknown"}
    np.testing.assert_array_equal(masks["target_unknown"], [True, True, True, True])
    stage1 = module.stage_condition_masks("Task2", "Stage1", np.array([-1, 1, 0]))
    np.testing.assert_array_equal(stage1["left"], [True, False, False])
    np.testing.assert_array_equal(stage1["right"], [False, True, False])


def test_real_cases_reorder_channels_baseline_and_quarantine_unknown(tmp_path):
    module = real_data()
    savemat(tmp_path / "synthetic_clean.mat", clean_fixture())
    cases, audit = module.load_cases_with_audit(tmp_path, {"Task2": ("synthetic",)})
    left = next(case for case in cases if case["condition"] == "left")
    assert left["n_trials"] == 12
    assert left["eligible_fit"]
    np.testing.assert_allclose(left["real"][:, 0], [2, 3, 4])
    target = next(case for case in cases if case["condition"] == "target_unknown")
    assert target["n_trials"] == 24
    assert not target["eligible_fit"]
    assert target["trials"].shape == (24, 3, 102)
    np.testing.assert_allclose(target["real"][:, 0], [4, 5, 6])
    assert target["time_ms"][0] == pytest.approx(3.125)
    assert audit[0]["cue_marker_label"] == "VisCue:L-1/R+1"
    assert audit[0]["cue_onset_median_ms"] == pytest.approx(0.0)
    assert audit[0]["cue_last_active_sample_median_ms"] == pytest.approx(195.3125)
    assert audit[0]["cue_offset_median_ms"] == pytest.approx(203.125)
    assert audit[0]["cue_active_duration_median_ms"] == pytest.approx(203.125)
    assert audit[0]["cue_type_agreement"] == pytest.approx(1.0)
    assert audit[0]["action_channel_label"] == "Action:L-1/R+1"
    assert audit[0]["action_first_post_target_edge_median_ms"] == pytest.approx(3.125)


def test_low_trial_condition_remains_descriptive(tmp_path):
    module = real_data()
    savemat(tmp_path / "small_clean.mat", clean_fixture(8))
    cases = module.load_cases(tmp_path, {"Task1": ("small",)})
    assert len(cases) == 3
    assert all(not case["eligible_fit"] for case in cases)
    assert next(case for case in cases if case["condition"] == "dots")["n_trials"] == 8


def test_fullcurve_case_rejects_truncated_response(tmp_path):
    module = real_data()
    mat = clean_fixture()
    mat["trial_data"] = mat["trial_data"][:, :, :470]
    mat["relative_time"] = mat["relative_time"][:, :470]
    savemat(tmp_path / "truncated_clean.mat", mat)
    with pytest.raises(ValueError, match="response interval"):
        module.load_cases(tmp_path, {"Task1": ("truncated",)})


@pytest.mark.parametrize("corruption", ["sample_rate", "duplicate_channel", "dropped_trial", "nonfinite_eeg"])
def test_loader_rejects_invalid_clean_contract(tmp_path, corruption):
    module = real_data()
    mat = clean_fixture()
    if corruption == "sample_rate":
        mat["SampleRate"] = 256
    elif corruption == "duplicate_channel":
        mat["DataLabel"][1, 0] = "F3"
    elif corruption == "dropped_trial":
        mat["drop"][0] = 1
    else:
        mat["trial_data"][0, 1, 10] = np.nan
    path = tmp_path / "invalid_clean.mat"
    savemat(path, mat)
    with pytest.raises(ValueError):
        module.load_dataset(path)
