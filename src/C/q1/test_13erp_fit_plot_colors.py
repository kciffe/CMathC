import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba


SCRIPT_PATH = Path(__file__).with_name("13erp_fit.py")
SPEC = importlib.util.spec_from_file_location("erp_fit_plot_colors", SCRIPT_PATH)
erp_fit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(erp_fit)


def test_target_fit_stays_green_when_cue_fit_is_skipped(tmp_path, monkeypatch):
    time = -1.0 + np.arange(512) / 128.0
    trials = np.zeros((2, 3, len(time)), dtype=float)
    target_shape = 7.0 * np.exp(-((time - 2.55) ** 2) / (2 * 0.05**2))
    trials[0, 1, :] = target_shape

    monkeypatch.setattr(
        erp_fit,
        "loadmat",
        lambda _: {
            "trial_data": trials,
            "relative_time": np.tile(time, (2, 1)),
            "cue_type": np.array([[-1], [1]]),
        },
    )
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    monkeypatch.setattr(erp_fit, "RESULT_DIR", result_dir)
    monkeypatch.setattr(erp_fit, "CLEAN_DIR", tmp_path)

    peak_rows = []
    for condition in ("左条件", "右条件"):
        for channel in ("F3", "Fz", "F4"):
            peak_rows.append({
                "Dataset": "Synthetic",
                "Condition": condition,
                "Channel": channel,
                "CuePeakStatus": "no_positive_peak",
                "TargetPeakStatus": (
                    "valid" if condition == "左条件" and channel == "F3"
                    else "no_positive_peak"
                ),
            })

    def successful_fit(window_time, window_signal, start, end):
        return {
            "A": 7.0,
            "Mu": 2.55,
            "Sigma": 0.05,
            "R2": 0.9,
            "RMSE": 0.1,
            "FitValid": 1,
            "Fitted": target_shape[(time >= start) & (time <= end)],
        }

    monkeypatch.setattr(erp_fit, "fit_gaussian", successful_fit)

    created = {}
    original_subplots = erp_fit.plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        created["axes"] = axes
        return fig, axes

    monkeypatch.setattr(erp_fit.plt, "subplots", capture_subplots)

    erp_fit.process_dataset("Synthetic", pd.DataFrame(peak_rows))

    target_lines = [
        line for line in created["axes"][0, 0].lines
        if line.get_label() == "P300候选窗·目标后"
    ]
    assert len(target_lines) == 1
    assert to_rgba(target_lines[0].get_color()) == to_rgba("tab:green")
    plt.close("all")


def test_exploratory_mode_fits_every_window_and_keeps_upstream_status(
    tmp_path,
    monkeypatch,
):
    time = -1.0 + np.arange(512) / 128.0
    trials = np.zeros((2, 3, len(time)), dtype=float)
    monkeypatch.setattr(
        erp_fit,
        "loadmat",
        lambda _: {
            "trial_data": trials,
            "relative_time": np.tile(time, (2, 1)),
            "cue_type": np.array([[-1], [1]]),
        },
    )

    result_dir = tmp_path / "exploratory"
    result_dir.mkdir()
    monkeypatch.setattr(erp_fit, "CLEAN_DIR", tmp_path)

    peak_rows = []
    for condition, _ in erp_fit.CONDITIONS:
        for channel, _ in erp_fit.CHANNELS:
            peak_rows.append({
                "Dataset": "Synthetic",
                "Condition": condition,
                "Channel": channel,
                "CuePeakStatus": "no_positive_peak",
                "TargetPeakStatus": "right_boundary",
            })

    calls = []

    def successful_fit(window_time, window_signal, start, end):
        calls.append((start, end))
        return {
            "A": 1.0,
            "Mu": (start + end) / 2,
            "Sigma": 0.05,
            "R2": 0.1,
            "RMSE": 1.0,
            "FitValid": 1,
            "Fitted": np.full_like(window_signal, 1.0),
        }

    monkeypatch.setattr(erp_fit, "fit_gaussian", successful_fit)

    captured = {}
    original_subplots = erp_fit.plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(erp_fit.plt, "subplots", capture_subplots)

    rows = erp_fit.process_dataset(
        "Synthetic",
        pd.DataFrame(peak_rows),
        exploratory_all_windows=True,
        result_dir=result_dir,
    )

    assert len(calls) == 12
    assert sum(row["Attempted"] for row in rows) == 12
    assert {row["FitType"] for row in rows} == {"ExploratoryAllWindows"}
    assert {row["UpstreamPeakStatus"] for row in rows} == {
        "no_positive_peak",
        "right_boundary",
    }
    assert sum(
        line.get_label().startswith("探索性全窗拟合")
        for ax in captured["axes"].flat
        for line in ax.lines
    ) == 12
    assert (result_dir / "Synthetic_ERP高斯拟合_探索性全窗.png").is_file()
    plt.close("all")
