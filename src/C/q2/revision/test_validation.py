"""Leakage, boundary and fixed-decoder behavioral checks."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def validation():
    assert (Path(__file__).parent / "validation.py").exists(), "validation is not implemented"
    return importlib.import_module("validation")


def fixture():
    t = np.arange(801.)
    center = np.exp(-((t - 550) / 65) ** 2)
    other = np.exp(-((t - 630) / 65) ** 2)
    predictions = {("Stage1", "left", tau): {"time_ms": t, "eeg": np.tile(center if tau == 30 else other, (3, 1))}
                   for tau in (30, 40, 50, 60)}
    cases = [{"dataset": record, "task": "Task1", "stage": "Stage1", "condition": "left",
              "time_ms": t, "real": np.tile(gain * center, (3, 1)), "n_trials": 20,
              "eligible_fit": True} for record, gain in (("train_a", 2), ("train_b", 2), ("held", 100))]
    return cases, predictions


def test_selection_and_gain_ignore_heldout_waveform_and_ineligible_case():
    m = validation()
    cases, predictions = fixture()
    first = m.select_tau(cases, predictions, "held")
    assert first["tau_ms"] == 30
    assert first["gain"] == pytest.approx(2.)
    assert first["training_records"] == ["train_a", "train_b"]
    cases[-1]["real"] *= -10000
    cases.append({**cases[0], "dataset": "quarantined", "eligible_fit": False, "real": -10000 * cases[0]["real"]})
    second = m.select_tau(cases, predictions, "held")
    assert second["tau_ms"] == first["tau_ms"]
    assert second["gain"] == pytest.approx(first["gain"])
    assert second["training_records"] == first["training_records"]


def test_boundary_peak_rejects_candidate_and_marks_fixed_fallback():
    m = validation()
    cases, predictions = fixture()
    for p in predictions.values():
        p["eeg"] = np.tile(np.arange(801.), (3, 1))
    selected = m.select_tau(cases, predictions, "held")
    assert selected["tau_ms"] == 40
    assert selected["fallback"]
    assert all(row["boundary_rejected"] for row in selected["scan"])
    metrics = m.case_metrics(cases[0], predictions[("Stage1", "left", 40)], 1)
    assert metrics["sim_edge_Fz"]
    assert metrics["latency_error_Fz_ms"] == pytest.approx(150.)


def test_gain_balances_records_stages_and_cannot_reverse_polarity():
    m = validation()
    cases, predictions = fixture()
    train = [cases[0], {**cases[1], "real": 6 * predictions[("Stage1", "left", 30)]["eeg"]}]
    train.extend([{**cases[0], "condition": "right"} for _ in range(4)])
    predictor = lambda case: predictions[("Stage1", "left", 30)]
    assert m.fit_gain(train, predictor) == pytest.approx(4.)
    for case in train:
        case["real"] = -case["real"]
    assert m.fit_gain(train, predictor) == 0.


def test_fixed_window_decoder_generalizes_separable_trials_without_label_leakage(tmp_path):
    m = validation()
    t = np.arange(0, 801, 5.)
    cases = []
    for dataset in ("a", "b", "c"):
        for condition, sign in (("left", -1), ("right", 1)):
            trial = np.stack([-sign * np.ones(t.size), np.zeros(t.size), sign * np.ones(t.size)])
            cases.append({"dataset": dataset, "task": "Task1", "stage": "Stage1", "condition": condition,
                          "time_ms": t, "real": trial, "trials": np.tile(trial, (12, 1, 1)),
                          "n_trials": 12, "eligible_fit": True})
    result = m.decode_stage1(cases, tmp_path)
    assert result["mean_balanced_accuracy"] == 1.
    assert len(result["folds"]) == 3
    assert all(row["held_out"] not in row["training_records"] for row in result["folds"])
    missing = m.decode_stage1([c for c in cases if c["condition"] == "left"], tmp_path / "missing")
    assert missing["mean_balanced_accuracy"] is None
    assert missing["status"] == "insufficient_data"
