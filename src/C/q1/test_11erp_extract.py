import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat


matplotlib.use("Agg")

SCRIPT_PATH = Path(__file__).with_name("11erp_extract.py")
SPEC = importlib.util.spec_from_file_location("erp_extract", SCRIPT_PATH)
erp_extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(erp_extract)


def _save_clean_mat(path, drop, time_sample_count=512):
    sample_count = 512
    time = -1.0 + np.arange(time_sample_count) / 128.0
    trial_count = len(drop)
    savemat(
        path,
        {
            "trial_data": np.zeros((trial_count, 10, sample_count)),
            "relative_time": np.tile(time, (trial_count, 1)),
            "cue_type": np.resize(np.array([-1, 1]), trial_count),
            "drop": np.asarray(drop, dtype=np.uint8),
            "SampleRate": np.array([[128]], dtype=np.int32),
        },
    )


def test_rejects_clean_mat_that_still_contains_dropped_trials(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    result_dir = tmp_path / "result"
    input_dir.mkdir()
    result_dir.mkdir()
    _save_clean_mat(input_dir / "Synthetic_clean.mat", [0, 1, 0, 0])
    monkeypatch.setattr(erp_extract, "INPUT_DIR", input_dir)
    monkeypatch.setattr(erp_extract, "RESULT_DIR", result_dir)

    with pytest.raises(ValueError, match="drop"):
        erp_extract.process_dataset("Synthetic")

    assert list(result_dir.iterdir()) == []


def test_rejects_relative_time_length_that_does_not_match_trial_data(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    result_dir = tmp_path / "result"
    input_dir.mkdir()
    result_dir.mkdir()
    _save_clean_mat(
        input_dir / "Synthetic_clean.mat",
        [0, 0, 0, 0],
        time_sample_count=511,
    )
    monkeypatch.setattr(erp_extract, "INPUT_DIR", input_dir)
    monkeypatch.setattr(erp_extract, "RESULT_DIR", result_dir)

    with pytest.raises(ValueError, match="relative_time"):
        erp_extract.process_dataset("Synthetic")

    assert list(result_dir.iterdir()) == []


def test_candidate_peak_marks_window_edge_as_boundary():
    time = np.array([0.25, 0.375, 0.50, 0.60])
    erp = np.array([1.0, 2.0, 5.0, 100.0])

    peak = erp_extract.extract_peak(erp, time, 0.25, 0.50)

    assert peak["amplitude"] == pytest.approx(5.0)
    assert peak["latency"] == pytest.approx(0.50)
    assert peak["left_boundary"] == 0
    assert peak["right_boundary"] == 1
    assert peak["has_positive_peak"] == 1
    assert peak["status"] == "right_boundary"


def test_positive_peak_at_left_window_edge_is_not_valid():
    time = np.array([0.25, 0.375, 0.50])
    erp = np.array([5.0, 2.0, 1.0])

    peak = erp_extract.extract_peak(erp, time, 0.25, 0.50)

    assert peak["left_boundary"] == 1
    assert peak["status"] == "left_boundary"


def test_nonpositive_window_maximum_is_not_a_positive_peak():
    time = np.array([0.25, 0.375, 0.50])
    erp = np.array([-3.0, -2.0, -4.0])

    peak = erp_extract.extract_peak(erp, time, 0.25, 0.50)

    assert peak["amplitude"] == pytest.approx(-2.0)
    assert peak["has_positive_peak"] == 0
    assert peak["status"] == "no_positive_peak"


def test_main_writes_condition_averages_and_candidate_peak_metrics(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    result_dir = tmp_path / "result"
    input_dir.mkdir()
    time = -1.0 + np.arange(512) / 128.0
    cue_type = np.array([-1, -1, 1, 1])
    baseline_offsets = np.array([100.0, 200.0, 300.0, 400.0])
    cue_amplitudes = np.array([4.0, 8.0, 12.0, 16.0])
    target_amplitudes = np.array([2.0, 6.0, 10.0, 14.0])
    trial_data = np.zeros((4, 10, 512), dtype=float)
    trial_data[:, 0, :] = baseline_offsets[:, None]
    cue_index = int(np.flatnonzero(np.isclose(time, 0.296875))[0])
    target_index = int(np.flatnonzero(np.isclose(time, 2.5))[0])
    trial_data[:, 0, cue_index] += cue_amplitudes
    trial_data[:, 0, target_index] += target_amplitudes
    savemat(
        input_dir / "Synthetic_clean.mat",
        {
            "trial_data": trial_data,
            "relative_time": np.tile(time, (4, 1)),
            "cue_type": cue_type,
            "drop": np.zeros(4, dtype=np.uint8),
            "SampleRate": np.array([[128]], dtype=np.int32),
        },
    )
    monkeypatch.setattr(erp_extract, "INPUT_DIR", input_dir)
    monkeypatch.setattr(erp_extract, "RESULT_DIR", result_dir)
    monkeypatch.setattr(erp_extract, "DATASETS", ("Synthetic",))

    erp_extract.main()

    metrics = pd.read_csv(result_dir / "ERP候选峰指标.csv", encoding="utf-8-sig")
    fz_left = metrics[(metrics["Condition"] == "左条件") & (metrics["Channel"] == "Fz")].iloc[0]
    fz_right = metrics[(metrics["Condition"] == "右条件") & (metrics["Channel"] == "Fz")].iloc[0]
    assert len(metrics) == 6
    assert {
        "CueHasPositivePeak",
        "CuePeakStatus",
        "TargetHasPositivePeak",
        "TargetPeakStatus",
    }.issubset(metrics.columns)
    assert fz_left["N_trials"] == 2
    assert fz_left["CuePeakAmplitude"] == pytest.approx(6.0)
    assert fz_left["CuePeakLatencyFromCue"] == pytest.approx(0.296875)
    assert fz_left["CueHasPositivePeak"] == 1
    assert fz_left["CuePeakStatus"] == "valid"
    assert fz_left["TargetPeakAmplitude"] == pytest.approx(4.0)
    assert fz_left["TargetPeakLatencyFromTarget"] == pytest.approx(0.3)
    assert fz_left["TargetPeakStatus"] == "valid"
    assert fz_right["CuePeakAmplitude"] == pytest.approx(14.0)
    assert (result_dir / "Synthetic_ERP.png").is_file()
