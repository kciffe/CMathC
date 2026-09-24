import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy.io import savemat


BASE_DIR = Path(__file__).parent


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, BASE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_y_axis_labels(monkeypatch):
    labels = []
    original = Axes.set_ylabel

    def record_label(axis, label, *args, **kwargs):
        labels.append(label)
        return original(axis, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_ylabel", record_label)
    return labels


def test_task7_trial_plots_mark_eeg_values_as_raw_data_units(tmp_path, monkeypatch):
    plotter = load_module("7plot_trial.py", "plot_trial_units_test")
    labels = capture_y_axis_labels(monkeypatch)
    time = np.tile(np.linspace(-0.5, 3.0, 8), (10, 1))

    plotter.plot_trials(
        trial_data=np.zeros((10, 10, 8)),
        relative_time=time,
        cue_type=np.array([-1] * 5 + [1] * 5),
        action_onsets=np.full(10, np.nan),
        save_path=tmp_path / "task7.png",
        title="task7",
    )

    assert labels == ["EEG（原始数据单位）"] * 10


def test_task8_sqi_plots_label_the_dimensionless_index(tmp_path, monkeypatch):
    plotter = load_module("8riemann_denoise.py", "riemann_units_test")
    plotter.RESULT_DIR = tmp_path
    labels = capture_y_axis_labels(monkeypatch)

    plotter.save_diagnostic_plots(
        "Dataset",
        np.array([0.1, 0.2, 0.4]),
        0.2,
        np.array([0.1, 0.2, 0.4]),
        np.array([0, 1, 2]),
        np.array([True, False, False]),
    )

    assert labels == ["SQI（无量纲）", "SQI（无量纲）"]


def test_task9_boundary_trial_plots_mark_raw_data_units(tmp_path, monkeypatch):
    plotter = load_module("9check_riemann_drop.py", "boundary_units_test")
    plotter.RESULT_DIR = tmp_path
    labels = capture_y_axis_labels(monkeypatch)
    selected = pd.DataFrame([{"Trial": 0, "SQI": 0.3}])
    groups = {name: selected for name, _ in plotter.GROUPS}
    time = np.tile(np.linspace(-0.2, 1.0, 9), (1, 1))

    plotter.plot_dataset("Dataset", np.zeros((1, 3, 9)), time, groups, 0.3)

    assert len(labels) == 3
    assert all(label.endswith("EEG 振幅（原始数据单位）") for label in labels)


def test_task11_erp_plots_mark_raw_data_units(tmp_path, monkeypatch):
    plotter = load_module("11erp_extract.py", "erp_units_test")
    plotter.INPUT_DIR = tmp_path
    plotter.RESULT_DIR = tmp_path
    labels = capture_y_axis_labels(monkeypatch)
    time = np.arange(-26, 385, dtype=float) / 128
    savemat(
        tmp_path / "Dataset_clean.mat",
        {
            "trial_data": np.zeros((2, 10, len(time))),
            "relative_time": np.tile(time, (2, 1)),
            "cue_type": np.array([[-1, 1]]),
            "drop": np.zeros((1, 2), dtype=np.uint8),
        },
    )

    plotter.process_dataset("Dataset")

    assert labels == ["基线校正后 ERP 幅值（原始数据单位）"] * 2


def test_task12_cleaning_sensitivity_plot_marks_raw_data_units(tmp_path, monkeypatch):
    plotter = load_module(
        "12check_a1_cleaning_sensitivity.py", "a1_cleaning_sensitivity_units_test"
    )
    plotter.INPUT_MAT = tmp_path / "filtered.mat"
    plotter.INPUT_CSV = tmp_path / "sqi.csv"
    plotter.RESULT_DIR = tmp_path
    labels = capture_y_axis_labels(monkeypatch)

    time = np.linspace(-1.0, 3.0, 65)
    savemat(
        plotter.INPUT_MAT,
        {
            "trial_data": np.zeros((4, 10, len(time))),
            "relative_time": np.tile(time, (4, 1)),
            "cue_type": np.array([[-1, -1, 1, 1]]),
            "drop": np.zeros((1, 4), dtype=np.uint8),
        },
    )
    pd.DataFrame(
        {
            "Trial": [0, 1, 2, 3],
            "CueType": [-1, -1, 1, 1],
            "ClippedDrop": [0, 0, 0, 0],
            "FinalDrop": [0, 1, 0, 1],
        }
    ).to_csv(plotter.INPUT_CSV, index=False)

    plotter.main()

    assert labels == ["基线校正后 ERP 幅值（原始数据单位）"] * 6
