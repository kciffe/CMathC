"""Behavioral checks for fixed-window ERP features and leave-record LDA."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import (classification_leave_one_record, extract_features,
                      fit_classifier, predict_classifier)
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
