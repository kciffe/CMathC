"""Post-hoc quality-subgroup summary of already generated outer-fold predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score


Q3_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = Q3_DIR / "baselines" / "20260925_initial" / "output" / "trial_features.csv"
DEFAULT_RESULTS = Q3_DIR / "output" / "experiments" / "20260925_model_search_v1"


def summarize(results_dir: Path, features_path: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path, encoding="utf-8-sig")
    metadata = features.loc[
        features["stage"].isin(("cue_locked", "target_locked"))
        & features["qc_valid"].fillna(False).astype(bool),
        ["record", "original_trial_index", "stage", "cue_side", "q1_quality_pass"],
    ].copy()
    metadata["q1_quality_pass"] = metadata["q1_quality_pass"].fillna(False).astype(bool)
    rows = []
    for method, filename in (
        ("fixed_baseline", "baseline_oof_predictions.csv"),
        ("nested_selected", "nested_selected_oof_predictions.csv"),
    ):
        predictions = pd.read_csv(results_dir / filename, encoding="utf-8-sig")
        predictions = predictions.merge(
            metadata,
            on=["record", "original_trial_index", "stage"],
            how="left",
            validate="many_to_one",
        )
        if predictions["q1_quality_pass"].isna().any():
            raise ValueError(f"Could not map every held-out prediction in {filename} to Q1 quality")
        if not (predictions["true_cue_side"] == predictions["cue_side"]).all():
            raise AssertionError("Prediction labels differ from the feature table labels")

        for sample, sample_frame in predictions.groupby("sample"):
            for stage, stage_frame in sample_frame.groupby("stage"):
                for q1_pass, subgroup in stage_frame.groupby("q1_quality_pass"):
                    per_record = {}
                    sizes = {}
                    for record, record_frame in subgroup.groupby("record"):
                        per_record[str(record)] = float(
                            balanced_accuracy_score(
                                record_frame["true_cue_side"], record_frame["predicted_cue_side"]
                            )
                        )
                        sizes[str(record)] = int(len(record_frame))
                    rows.append(
                        {
                            "model": method,
                            "training_scope": sample,
                            "stage": stage,
                            "evaluated_q1_quality_pass": bool(q1_pass),
                            "n_trials": int(len(subgroup)),
                            "n_records": int(len(per_record)),
                            "mean_record_balanced_accuracy": float(pd.Series(per_record).mean()),
                            "pooled_balanced_accuracy": float(
                                balanced_accuracy_score(
                                    subgroup["true_cue_side"], subgroup["predicted_cue_side"]
                                )
                            ),
                            "record_balanced_accuracy": json.dumps(per_record, sort_keys=True),
                            "record_n": json.dumps(sizes, sort_keys=True),
                        }
                    )
    result = pd.DataFrame(rows).sort_values(
        ["training_scope", "stage", "evaluated_q1_quality_pass", "model"]
    )
    return result.reset_index(drop=True)


def write_findings(results_dir: Path, subgroup_metrics: pd.DataFrame) -> None:
    search_summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    by_scope_stage = {
        (item["sample"], item["stage"]): item
        for item in search_summary["selections_and_deltas"]
    }
    subgroup_summary = {}
    for (scope, stage, q1_pass), group in subgroup_metrics.groupby(
        ["training_scope", "stage", "evaluated_q1_quality_pass"]
    ):
        values = group.set_index("model")["mean_record_balanced_accuracy"]
        if "fixed_baseline" not in values or "nested_selected" not in values:
            continue
        subgroup_summary[f"{scope}|{stage}|q1_pass={bool(q1_pass)}"] = {
            "n_trials": int(group.iloc[0]["n_trials"]),
            "baseline_mean_record_ba": float(values["fixed_baseline"]),
            "nested_selected_mean_record_ba": float(values["nested_selected"]),
            "delta_percentage_points": float(
                100.0 * (values["nested_selected"] - values["fixed_baseline"])
            ),
            "per_record_ba": {
                row["model"]: json.loads(row["record_balanced_accuracy"])
                for row in group.to_dict(orient="records")
            },
        }
    (results_dir / "quality_subgroup_summary.json").write_text(
        json.dumps(subgroup_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Exploratory model search findings",
        "",
        "The frozen initial baseline is unchanged. This comparison evaluates the nested model-selection procedure against the fixed 13-feature Logistic baseline on the same outer leave-one-record-out folds.",
        "",
        "## Main results",
        "",
        "| Training/evaluation sample | Stage | Baseline BA | Nested-selected BA | Change | Per-fold changes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for scope, scope_name in (
        ("raw_qc_pass", "Raw-QC full sample"),
        ("q1_quality_matched", "Q1 quality-matched sensitivity"),
    ):
        for stage, stage_name in (("cue_locked", "Cue"), ("target_locked", "Target")):
            item = by_scope_stage[(scope, stage)]
            fold_text = ", ".join(
                f"{record}: {delta:+.1f} pp"
                for record, delta in item["delta_by_fold_percentage_points"].items()
            )
            lines.append(
                f"| {scope_name} | {stage_name} | {item['baseline_mean_balanced_accuracy']:.3f} | "
                f"{item['nested_selected_mean_balanced_accuracy']:.3f} | "
                f"{item['delta_percentage_points']:+.2f} pp | {fold_text} |"
            )
    lines.extend(
        [
            "",
            "The raw-QC cue improvement is positive in all four held-out records, but the resulting BA is still about 0.521. It does not transfer to the analysis retrained only on the 297 Q1-matched trials. The target stage decreases under nested selection.",
            "",
            "## Quality-subgroup check",
            "",
            "These rows evaluate predictions from models trained and selected on each complete raw-QC training fold, then split the held-out predictions by Q1 quality status. This is a diagnostic subgroup analysis; it does not change model selection.",
            "",
            "| Stage | Held-out Q1 status | Trials | Baseline BA | Nested-selected BA | Change |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    raw_subgroups = subgroup_metrics.loc[subgroup_metrics["training_scope"] == "raw_qc_pass"]
    for stage, stage_name in (("cue_locked", "Cue"), ("target_locked", "Target")):
        for q1_pass, status_name in ((True, "Q1 retained"), (False, "Q1 excluded, raw-QC passed")):
            key = f"raw_qc_pass|{stage}|q1_pass={q1_pass}"
            item = subgroup_summary.get(key)
            if not item:
                continue
            lines.append(
                f"| {stage_name} | {status_name} | {item['n_trials']} | "
                f"{item['baseline_mean_record_ba']:.3f} | {item['nested_selected_mean_record_ba']:.3f} | "
                f"{item['delta_percentage_points']:+.2f} pp |"
            )
    lines.extend(
        [
            "",
            "For raw-QC-trained cue models, the improvement is concentrated in the Q1-retained held-out subset; performance falls on the Q1-excluded/raw-QC-passed subset. Yet when both training and evaluation are restricted to Q1-matched trials, nested selection yields essentially no cue gain. This makes the observed improvement sensitive to the training-sample definition.",
            "",
            "## Model choices and limits",
            "",
            "- The candidate set contains 49 fixed configurations across Logistic, shrinkage LDA, linear SVM, and RBF SVM, with predeclared feature sets and regularization/kernel values.",
            "- Selected configuration names differ across held-out records, so this does not identify one stable replacement classifier.",
            "- Four recording files are only four outer validation groups. These results are exploratory, with no confidence interval or significance claim.",
            "- Do not choose a model by the highest outer-fold candidate score; use the nested-selected row as the estimate of the inner selection procedure.",
            "- The nominal target time remains cue+2.2 s by schedule assumption; target-stage results remain timing-sensitive.",
            "",
            "Detailed files: `summary.json`, `nested_selected_fold_metrics.csv`, `outer_candidate_summary.csv`, `quality_subgroup_metrics.csv`, and `quality_subgroup_summary.json`.",
        ]
    )
    (results_dir / "experiment_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    result = summarize(args.results_dir, args.features)
    result.to_csv(args.results_dir / "quality_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    write_findings(args.results_dir, result)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
