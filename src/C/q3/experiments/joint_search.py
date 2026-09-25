"""Nested union search combining round-one classifiers and temporal/CSP features.

This is a retrospective exploratory development procedure: it combines the two
predeclared candidate grids and evaluates their joint selection using the same
record-wise inner/outer folds. Do not interpret it as independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Q3_DIR = HERE.parent
sys.path.insert(0, str(HERE))
import model_search as previous  # noqa: E402
import temporal_search as temporal  # noqa: E402

DEFAULT_OUTPUT = Q3_DIR / "output" / "experiments" / "20260925_joint_search_v2"
ROUND1_DIR = Q3_DIR / "output" / "experiments" / "20260925_model_search_v1"
TEMPORAL_DIR = Q3_DIR / "output" / "experiments" / "20260925_temporal_search_v1"


@dataclass(frozen=True)
class JointCandidate:
    source: str
    name: str
    order: int
    candidate: Any

    def as_row(self) -> dict[str, Any]:
        inner = self.candidate.as_row()
        return {
            "source": self.source,
            "name": self.name,
            "candidate_order": self.order,
            "candidate_config": json.dumps(inner, ensure_ascii=False, default=str),
        }


def select_best_by_inner(scored: list[tuple[float, int, Any]]) -> tuple[float, int, Any]:
    """Select by inner mean BA; ties go to the earlier predeclared candidate."""
    if not scored:
        raise ValueError("At least one scored candidate is required")
    return max(scored, key=lambda item: (item[0], -item[1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_union() -> list[JointCandidate]:
    result: list[JointCandidate] = []
    for candidate in previous.candidate_grid():
        result.append(JointCandidate("round1_classifier", f"round1::{candidate.name}", len(result), candidate))
    for candidate in temporal.candidate_grid():
        if candidate.name == "baseline_logistic_l2_1":
            continue  # Exact same fixed baseline is already present in round one.
        result.append(JointCandidate("temporal_or_csp", f"temporal::{candidate.name}", len(result), candidate))
    return result


def _score_candidate(
    item: JointCandidate,
    model_frame: pd.DataFrame,
    temporal_frame: pd.DataFrame,
    fixed: dict[str, np.ndarray],
    csp: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    if item.source == "round1_classifier":
        return previous.score_fold(
            model_frame.iloc[train_idx], model_frame.iloc[eval_idx], item.candidate
        )
    return temporal.score_indices(
        temporal_frame, fixed, csp, train_idx, eval_idx, item.candidate
    )


def _inner_score(
    item: JointCandidate,
    model_frame: pd.DataFrame,
    temporal_frame: pd.DataFrame,
    fixed: dict[str, np.ndarray],
    csp: np.ndarray,
    outer_train: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    if item.source == "round1_classifier":
        return previous._candidate_inner_score(model_frame.iloc[outer_train].reset_index(drop=True), item.candidate)
    return temporal._inner_score(temporal_frame, fixed, csp, outer_train, item.candidate)


def _align_frames(model_frame: pd.DataFrame, temporal_frame: pd.DataFrame) -> None:
    keys = ["record", "original_trial_index", "cue_side"]
    left = model_frame[keys].reset_index(drop=True)
    right = temporal_frame[keys].reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("Round-one and temporal feature rows do not align exactly")


def run_experiment(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {output_dir}")
    if not ROUND1_DIR.exists() or not TEMPORAL_DIR.exists():
        raise FileNotFoundError("Run round-one and temporal search experiments before joint search")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_union()
    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    baseline_predictions: list[pd.DataFrame] = []
    selected_predictions: list[pd.DataFrame] = []
    stage_summaries: list[dict[str, Any]] = []
    features_path = previous.DEFAULT_INPUT
    epochs_path = temporal.DEFAULT_EPOCHS
    features = pd.read_csv(features_path, encoding="utf-8-sig")

    for stage in ("cue_locked", "target_locked"):
        temporal_full, fixed_full, csp_full = temporal.build_stage_data(features_path, epochs_path, stage)
        for sample in ("raw_qc_pass", "q1_quality_matched"):
            model_full = previous.prepare_scope(features, stage, sample)
            temporal_indices = temporal.prepare_scope(temporal_full, sample)
            temporal_scope = temporal_full.iloc[temporal_indices].reset_index(drop=True)
            fixed_scope = {key: value[temporal_indices] for key, value in fixed_full.items()}
            csp_scope = csp_full[temporal_indices]
            model_scope = model_full.reset_index(drop=True)
            _align_frames(model_scope, temporal_scope)
            folds = previous.grouped_leave_one_group_out(model_scope["record"].to_numpy())
            baseline = candidates[0]

            for outer_train, outer_test, held_out in folds:
                training_records = sorted(model_scope.iloc[outer_train]["record"].unique().tolist())
                if held_out in training_records:
                    raise AssertionError("Outer held-out record leaked into training")
                baseline_metrics, baseline_pred = _score_candidate(
                    baseline, model_scope, temporal_scope, fixed_scope, csp_scope,
                    outer_train, outer_test,
                )
                baseline_predictions.append(baseline_pred.assign(
                    sample=sample, stage=stage, candidate=baseline.name, held_out_record=held_out
                ))
                scores: list[tuple[float, int, JointCandidate]] = []
                for candidate in candidates:
                    mean_inner, details = _inner_score(
                        candidate, model_scope, temporal_scope, fixed_scope, csp_scope, outer_train
                    )
                    scores.append((mean_inner, candidate.order, candidate))
                    inner_rows.extend({
                        "sample": sample, "stage": stage, "outer_held_out_record": held_out,
                        "candidate": candidate.name, **candidate.as_row(),
                        "inner_mean_balanced_accuracy": mean_inner, **detail,
                    } for detail in details)
                    fold_metrics, _ = _score_candidate(
                        candidate, model_scope, temporal_scope, fixed_scope, csp_scope,
                        outer_train, outer_test,
                    )
                    outer_rows.append({
                        "sample": sample, "stage": stage, "candidate": candidate.name,
                        **candidate.as_row(), "held_out_record": held_out,
                        "n_train": int(len(outer_train)), "n_test": int(len(outer_test)),
                        "training_records": ";".join(training_records), **fold_metrics,
                    })
                best_inner, _, chosen = select_best_by_inner(scores)
                chosen_metrics, chosen_pred = _score_candidate(
                    chosen, model_scope, temporal_scope, fixed_scope, csp_scope,
                    outer_train, outer_test,
                )
                selected_rows.append({
                    "sample": sample, "stage": stage, "held_out_record": held_out,
                    "n_train": int(len(outer_train)), "n_test": int(len(outer_test)),
                    "training_records": ";".join(training_records),
                    "selected_by_inner_balanced_accuracy": best_inner,
                    "selected_candidate": chosen.name, **chosen.as_row(), **chosen_metrics,
                    "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                    "balanced_accuracy_delta": chosen_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
                })
                selected_predictions.append(chosen_pred.assign(
                    sample=sample, stage=stage, candidate=chosen.name, held_out_record=held_out
                ))

            selected = pd.DataFrame([r for r in selected_rows if r["sample"] == sample and r["stage"] == stage])
            baseline_outer = pd.DataFrame([r for r in outer_rows if r["sample"] == sample and r["stage"] == stage and r["candidate"] == baseline.name])
            stage_summaries.append({
                "sample": sample, "stage": stage, "n_trials": int(len(model_scope)),
                "valid_record_folds": int(selected["held_out_record"].nunique()),
                "baseline_mean_balanced_accuracy": float(baseline_outer["balanced_accuracy"].mean()),
                "joint_nested_selected_mean_balanced_accuracy": float(selected["balanced_accuracy"].mean()),
                "delta_percentage_points": float(100 * selected["balanced_accuracy_delta"].mean()),
                "selected_fold_balanced_accuracy": selected.set_index("held_out_record")["balanced_accuracy"].to_dict(),
                "selected_candidates_by_fold": selected.set_index("held_out_record")["selected_candidate"].to_dict(),
                "delta_by_fold_percentage_points": selected.set_index("held_out_record")["balanced_accuracy_delta"].mul(100).to_dict(),
            })

    outer_df = pd.DataFrame(outer_rows)
    inner_df = pd.DataFrame(inner_rows)
    selected_df = pd.DataFrame(selected_rows)
    summary_df = (
        outer_df.groupby(["sample", "stage", "candidate", "source"], dropna=False)
        .agg(mean_accuracy=("accuracy", "mean"), mean_balanced_accuracy=("balanced_accuracy", "mean"),
             sd_balanced_accuracy=("balanced_accuracy", "std"), mean_auc=("auc", "mean"),
             mean_macro_f1=("macro_f1", "mean"), valid_record_folds=("held_out_record", "nunique"))
        .reset_index()
    )
    outer_df.to_csv(output_dir / "outer_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "outer_candidate_summary.csv", index=False, encoding="utf-8-sig")
    inner_df.to_csv(output_dir / "inner_cv_scores.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(output_dir / "nested_selected_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(baseline_predictions, ignore_index=True).to_csv(output_dir / "baseline_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(selected_predictions, ignore_index=True).to_csv(output_dir / "nested_selected_oof_predictions.csv", index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "experiment": "joint nested search over round-one classifiers and temporal/CSP features",
        "development_only": True,
        "candidate_count": len(candidates),
        "primary_metric": "balanced_accuracy",
        "outer_validation": "leave one complete recording out",
        "inner_selection": "leave one of the other three records out; train-only scaling/CSP fitting",
        "results": stage_summaries,
        "limitations": [
            "This retrospective union combines two exploratory candidate grids after both component rounds had been inspected.",
            "The same four outer records were viewed in earlier rounds; even nested scores are development estimates, not independent confirmation.",
            "Only four recording-level folds are available, and fold-to-fold variability is large.",
            "Results measure cue-side prediction from EEG and do not establish clinical diagnostic performance.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_findings(output_dir, summary)
    artifact_paths = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "experiment": summary["experiment"], "created_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "inputs": {"trial_features_sha256": _sha256(features_path), "raw_event_epochs_sha256": _sha256(epochs_path)},
        "component_artifacts": {
            "round1_summary_sha256": _sha256(ROUND1_DIR / "summary.json"),
            "round1_fold_metrics_sha256": _sha256(ROUND1_DIR / "nested_selected_fold_metrics.csv"),
            "temporal_summary_sha256": _sha256(TEMPORAL_DIR / "summary.json"),
            "temporal_fold_metrics_sha256": _sha256(TEMPORAL_DIR / "nested_selected_fold_metrics.csv"),
        },
        "sources": {"joint_search.py": _sha256(Path(__file__)), "model_search.py": _sha256(HERE / "model_search.py"), "temporal_search.py": _sha256(HERE / "temporal_search.py")},
        "artifacts": [{"path": p.name, "sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifact_paths],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_findings(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Joint model and temporal feature search", "",
        "This is a development-only nested search. The original baseline snapshot is unchanged.", "",
        "| Sample | Stage | Baseline BA | Joint nested BA | Delta |", "|---|---|---:|---:|---:|",
    ]
    for item in summary["results"]:
        lines.append(
            f"| {item['sample']} | {item['stage']} | {item['baseline_mean_balanced_accuracy']:.4f} | "
            f"{item['joint_nested_selected_mean_balanced_accuracy']:.4f} | {item['delta_percentage_points']:+.2f} pp |"
        )
    lines.extend([
        "", "Per-record selections are in `nested_selected_fold_metrics.csv`. Candidate outer maxima are exploratory and must not be presented as unbiased results.", "",
        "## Limitations", "", *[f"- {limitation}" for limitation in summary["limitations"]],
    ])
    (output_dir / "experiment_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_experiment(args.output_dir)
    for item in summary["results"]:
        print(
            f"{item['sample']} {item['stage']}: baseline BA="
            f"{item['baseline_mean_balanced_accuracy']:.3f}, joint BA="
            f"{item['joint_nested_selected_mean_balanced_accuracy']:.3f}, "
            f"delta={item['delta_percentage_points']:+.2f} pp"
        )
    print(f"Wrote joint search results to {args.output_dir}")


if __name__ == "__main__":
    main()
