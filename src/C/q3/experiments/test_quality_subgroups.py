import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from summarize_quality_subgroups import summarize


def test_subgroup_summary_accepts_predictions_from_multiple_training_scopes(tmp_path):
    features = []
    for record in ("r1", "r2"):
        for trial, side, q1_pass in (
            (0, -1, True),
            (1, 1, True),
            (2, -1, False),
            (3, 1, False),
        ):
            features.append(
                {
                    "record": record,
                    "original_trial_index": trial,
                    "stage": "cue_locked",
                    "cue_side": side,
                    "qc_valid": True,
                    "q1_quality_pass": q1_pass,
                }
            )
    feature_path = tmp_path / "features.csv"
    pd.DataFrame(features).to_csv(feature_path, index=False)

    predictions = []
    for sample, q1_only in (("raw_qc_pass", False), ("q1_quality_matched", True)):
        for record in ("r1", "r2"):
            for trial, side, q1_pass in (
                (0, -1, True),
                (1, 1, True),
                (2, -1, False),
                (3, 1, False),
            ):
                if q1_only and not q1_pass:
                    continue
                predictions.append(
                    {
                        "sample": sample,
                        "record": record,
                        "original_trial_index": trial,
                        "stage": "cue_locked",
                        "true_cue_side": side,
                        "predicted_cue_side": side,
                    }
                )
    result_path = tmp_path / "results"
    result_path.mkdir()
    pd.DataFrame(predictions).to_csv(result_path / "baseline_oof_predictions.csv", index=False)
    pd.DataFrame(predictions).to_csv(result_path / "nested_selected_oof_predictions.csv", index=False)

    summary = summarize(result_path, feature_path)

    assert set(summary["training_scope"]) == {"raw_qc_pass", "q1_quality_matched"}
    assert set(summary["evaluated_q1_quality_pass"]) == {True, False}
    assert (summary["mean_record_balanced_accuracy"] == 1.0).all()
