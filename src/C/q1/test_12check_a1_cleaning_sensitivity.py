import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

matplotlib.use("Agg")

SCRIPT_PATH = Path(__file__).with_name("11_2check_a1_cleaning_sensitivity.py")
SPEC = importlib.util.spec_from_file_location("a1_cleaning_sensitivity", SCRIPT_PATH)
a1_cleaning_sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a1_cleaning_sensitivity)


def test_trial_column_aligns_shuffled_sqi_rows_to_mat_trials(tmp_path):
    sample_count = 512
    trial_count = 4
    time = -1.0 + np.arange(sample_count) / 128.0
    cue_type = np.array([-1, -1, 1, 1])
    clipped_drop = np.zeros(trial_count, dtype=np.uint8)
    trial_data = np.zeros((trial_count, 10, sample_count), dtype=float)
    trial_data[:, 0, :] = np.array([10.0, 20.0, 30.0, 40.0])[:, None]

    mat_path = tmp_path / "filtered.mat"
    savemat(
        mat_path,
        {
            "trial_data": trial_data,
            "relative_time": np.tile(time, (trial_count, 1)),
            "cue_type": cue_type,
            "drop": clipped_drop,
        },
    )

    # Trial 0 and 2 are the formal survivors; CSV rows are intentionally shuffled.
    metrics = pd.DataFrame(
        {
            "Trial": [2, 0, 3, 1],
            "CueType": [1, -1, 1, -1],
            "ClippedDrop": [0, 0, 0, 0],
            "FinalDrop": [0, 0, 1, 1],
        }
    )
    csv_path = tmp_path / "sqi.csv"
    metrics.to_csv(csv_path, index=False)

    trials, aligned_time, aligned_cue, clipped_keep, current_keep = (
        a1_cleaning_sensitivity.load_comparison_inputs(mat_path, csv_path)
    )

    assert np.array_equal(trials, trial_data[:, :3, :])
    assert np.array_equal(aligned_time, time)
    assert np.array_equal(aligned_cue, cue_type)
    assert np.array_equal(clipped_keep, [True, True, True, True])
    assert np.array_equal(current_keep, [True, False, True, False])


def test_rejects_sqi_rows_with_missing_or_duplicate_trial_indices(tmp_path):
    sample_count = 512
    time = -1.0 + np.arange(sample_count) / 128.0
    mat_path = tmp_path / "filtered.mat"
    savemat(
        mat_path,
        {
            "trial_data": np.zeros((2, 10, sample_count)),
            "relative_time": np.tile(time, (2, 1)),
            "cue_type": np.array([-1, 1]),
            "drop": np.zeros(2, dtype=np.uint8),
        },
    )
    csv_path = tmp_path / "sqi.csv"
    pd.DataFrame(
        {
            "Trial": [0, 0],
            "CueType": [-1, 1],
            "ClippedDrop": [0, 0],
            "FinalDrop": [0, 0],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Trial"):
        a1_cleaning_sensitivity.load_comparison_inputs(mat_path, csv_path)
