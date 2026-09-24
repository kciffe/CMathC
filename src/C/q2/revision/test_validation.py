"""Behavioral checks for fixed-window ERP features and leave-record LDA."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
import config
from evaluate import (classification_leave_one_record, extract_features,
                      fit_classifier, predict_classifier, erp_metrics)
import run_revision


def _trial_for_sign(sign, time_ms):
    # A pure u1 pattern: F3=-sign, Fz=0, F4=sign.
    return np.stack([-sign * np.ones_like(time_ms),
                     np.zeros_like(time_ms),
                     sign * np.ones_like(time_ms)])


def test_feature_extraction_returns_three_windows_times_two_modes():
    time_ms = np.arange(0.0, 701.0, 5.0)
    trials = np.stack([_trial_for_sign(-1, time_ms), _trial_for_sign(1, time_ms)])

    features = extract_features(trials, time_ms)

    assert features.shape == (2, 6)
    assert np.all(features[0, 1::2] < 0)
    assert np.all(features[1, 1::2] > 0)


def test_erp_metrics_separate_observable_fit_from_unexplained_u2():
    time_ms = np.arange(8, dtype=float)
    modes = np.vstack([np.linspace(0.0, 1.0, len(time_ms)),
                       np.linspace(1.0, -1.0, len(time_ms))])
    prediction = config.U_OBS.T @ modes
    u2_signal = np.array([0.0, 2.0, -1.0, 3.0, 2.0, -2.0, 1.0, 0.0])
    real = prediction + config.U2[:, None] * u2_signal[None, :]

    metrics = erp_metrics(real, prediction, time_ms, "Stage1")

    assert metrics["observable_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["u2_unexplained_rms"] == pytest.approx(np.sqrt(np.mean(u2_signal ** 2)))
    assert metrics["u2_energy_share"] > 0.0
    assert metrics["full_sensor_rmse"] > 0.0


def test_shrinkage_lda_uses_training_records_and_separates_clear_features():
    rng = np.random.default_rng(12)
    labels = np.tile([0, 1], 40)
    record_ids = np.repeat(["r1", "r2", "r3", "r4"], 20)
    features = rng.normal(0.0, 0.08, (80, 6))
    features[:, 1] += np.where(labels == 1, 1.5, -1.5)
    classifier = fit_classifier(features, labels, record_ids)

    scores, predicted = predict_classifier(classifier, features)

    assert classifier["training_records"] == ["r1", "r2", "r3", "r4"]
    assert np.isfinite(scores).all()
    assert np.mean(predicted == labels) == 1.0


def test_leave_one_record_classifier_never_trains_on_heldout_record():
    time_ms = np.arange(0.0, 701.0, 5.0)
    cases = []
    for record in ("a", "b", "c"):
        for condition, sign in (("left", -1), ("right", 1)):
            trial = _trial_for_sign(sign, time_ms)
            cases.append({"dataset": record, "stage": "Stage1",
                          "condition": condition, "time_ms": time_ms,
                          "trials": np.tile(trial, (8, 1, 1)),
                          "eligible_classification": True})

    scores, folds, summary = classification_leave_one_record(cases)

    assert len(scores) == 48
    assert len(folds) == 3
    assert all(row["status"] == "complete" for row in folds)
    assert all(row["balanced_accuracy"] == 1.0 for row in folds)
    assert summary["balanced_accuracy"] == 1.0


def test_full_workflow_exposes_stage2_forward_runner():
    assert callable(run_revision.run_forward_condition)


def test_repeated_condition_runner_reuses_deterministic_frontend(monkeypatch):
    evaluate._cached_condition_frontend.cache_clear()
    frontend_calls = []
    monkeypatch.setattr(evaluate, "load_stimulus", lambda stage, condition:
                        (stage, condition))

    def fake_frontend(stimulus, params, resolution, include_offset, remove_position):
        frontend_calls.append((stimulus, params["tau_a"], resolution,
                               include_offset, remove_position))
        return {"sentinel": stimulus}

    model_calls = []
    monkeypatch.setattr(evaluate, "simulate_frontend", fake_frontend)
    monkeypatch.setattr(evaluate, "simulate_forward", lambda front, params, amplitude:
                        model_calls.append((front, amplitude)) or
                        SimpleNamespace(eeg_scaled=np.full((3, 2), amplitude)))

    first = evaluate.run_forward_condition("Stage1", "left", {"tau_a": 80.0},
                                           resolution=128, amplitude=1.0)
    second = evaluate.run_forward_condition("Stage1", "left", {"tau_a": 80.0},
                                            resolution=128, amplitude=2.0)

    assert len(frontend_calls) == 1
    assert len(model_calls) == 2
    np.testing.assert_array_equal(first.eeg_scaled, 1.0)
    np.testing.assert_array_equal(second.eeg_scaled, 2.0)
    evaluate._cached_condition_frontend.cache_clear()
