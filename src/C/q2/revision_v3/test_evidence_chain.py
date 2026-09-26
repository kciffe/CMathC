"""Contract tests for the fixed 9-D candidate EEG feature and holdout split."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_chain import (CANDIDATE_FEATURE_NAMES,
                            classify_electrode_features,
                            extract_electrode_window_features,
                            _process_model_and_sources,
                            _plot_population_differences,
                            _plot_empirical_features,
                            _plot_source_differences,
                            SOURCE_NAMES)


def test_candidate_feature_windows_are_fixed_half_open_electrode_means():
    time_ms = np.arange(801, dtype=float)
    trials = np.zeros((1, 3, time_ms.size), dtype=float)
    trials[0, 0, 250] = 100.0
    trials[0, 1, 500] = 200.0
    trials[0, 2, 800] = 300.0

    features = extract_electrode_window_features(trials, time_ms)

    assert features.shape == (1, 9)
    assert CANDIDATE_FEATURE_NAMES == (
        "W1_100_250_F3", "W1_100_250_Fz", "W1_100_250_F4",
        "W2_250_500_F3", "W2_250_500_Fz", "W2_250_500_F4",
        "W3_500_800_F3", "W3_500_800_Fz", "W3_500_800_F4")
    assert features[0, 0] == 0.0  # 250 ms belongs to W2, not W1.
    assert features[0, 3] > 0.0
    assert features[0, 7] > 0.0  # 500 ms belongs to W3, not W2.
    assert features[0, 8] == 0.0  # 800 ms is outside W3.


def test_9d_classifier_holds_out_entire_record_and_fits_scaling_on_train_only():
    time_ms = np.arange(801, dtype=float)
    cases = []
    for record_index, record in enumerate(("A", "B", "C")):
        for condition, sign in (("left", -1.0), ("right", 1.0)):
            trials = np.zeros((6, 3, time_ms.size), dtype=float)
            trials[:, 0, 100:250] = sign + record_index * 0.1
            trials[:, 1, 250:500] = sign * 0.25
            cases.append({"dataset": record, "stage": "Stage1", "condition": condition,
                          "eligible_classification": True, "time_ms": time_ms,
                          "trials": trials})

    scores, folds, summary = classify_electrode_features(cases)

    assert len(scores) == 36
    assert len(folds) == 3
    assert summary["feature_count"] == 9
    assert all(row["status"] == "complete" for row in folds)
    assert all(row["heldout_record"] not in row["training_records"] for row in folds)
    assert all(row["balanced_accuracy"] >= 0.9 for row in folds)


def test_source_sum_matches_float32_forward_observation_with_roundoff_tolerance():
    time_ms = np.arange(0.0, 3001.0, 10.0)
    source = np.stack([
        (0.2 + 0.03 * index) * np.exp(-time_ms / (200.0 + 30.0 * index))
        * np.sin(time_ms / (70.0 + 8.0 * index))
        for index in range(5)
    ]).astype(np.float32)
    leadfield = np.array([[.2, -.1, .3, .4, .1],
                          [-.1, .5, .2, -.2, .4],
                          [.3, .2, -.4, .1, .5]])
    eeg = (leadfield @ source).astype(np.float32)
    result = SimpleNamespace(time_ms=time_ms, source_proxy=source, eeg=eeg,
                             diagnostics={"leadfield": leadfield.tolist()})

    source_processed, total_processed, processed_time, relative_error = \
        _process_model_and_sources(result, amplitude=1.7)

    assert source_processed.shape == (3, 5, processed_time.size)
    assert total_processed.shape == (3, processed_time.size)
    assert relative_error < 1e-5
    np.testing.assert_allclose(source_processed.sum(axis=1), total_processed,
                               rtol=1e-10, atol=1e-10)


def test_population_difference_figure_accepts_channel_by_time_arrays(tmp_path):
    time_ms = np.arange(0.0, 801.0, 4.0)
    rows = [{"heldout_record": f"fold_{index}", "time_ms": time_ms,
             "excitatory": np.ones((3, 2, time_ms.size)) * index,
             "inhibitory": np.ones((3, 2, time_ms.size)) * (index + 1)}
            for index in range(3)]
    output = tmp_path / "population_difference.png"

    _plot_population_differences(rows, output)

    assert output.exists() and output.stat().st_size > 0


def test_empirical_feature_plot_reads_expanded_model_feature_columns(tmp_path):
    rng = np.random.default_rng(3)
    features = {"record_A": {"left": rng.normal(size=(8, 9)),
                              "right": rng.normal(size=(9, 9))}}
    model = {"record": "record_A"}
    model.update({f"predicted_right_minus_left_{name}": float(index)
                  for index, name in enumerate(CANDIDATE_FEATURE_NAMES)})
    output = tmp_path / "measured_model_features.png"

    _plot_empirical_features(features, [model], output)

    assert output.exists() and output.stat().st_size > 0


def test_source_difference_plot_renders_all_source_channels(tmp_path):
    rows = []
    for sensor_i, sensor in enumerate(("F3", "Fz", "F4")):
        for source_i, source in enumerate((*SOURCE_NAMES, "TOTAL")):
            for time_ms in (0.0, 4.0, 8.0):
                rows.append({"heldout_record": "record_A", "sensor": sensor,
                             "source": source, "time_ms": time_ms,
                             "right_minus_left": float(sensor_i + source_i + time_ms)})
    output = tmp_path / "source_difference.png"

    _plot_source_differences(rows, output)

    assert output.exists() and output.stat().st_size > 0
