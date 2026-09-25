"""Exploratory nested record-wise search over time-resolved EEG features.

This experiment is isolated from the primary Q3 pipeline and reads only the
frozen initial snapshot. All scaling and CSP fitting happen inside each
training partition. The four record folds have already been inspected in an
earlier exploratory round, so results are development estimates, not an
independent final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


HERE = Path(__file__).resolve().parent
Q3_DIR = HERE.parent
BASELINE_DIR = Q3_DIR / "baselines" / "20260925_initial"
DEFAULT_FEATURES = BASELINE_DIR / "output" / "trial_features.csv"
DEFAULT_EPOCHS = BASELINE_DIR / "output" / "q3_raw_event_epochs.npz"
DEFAULT_OUTPUT = Q3_DIR / "output" / "experiments" / "20260925_temporal_search_v1"
ROUND1_DIR = Q3_DIR / "output" / "experiments" / "20260925_model_search_v1"
BASELINE_FEATURES = (
    "erp_mean_F3", "erp_mean_Fz", "erp_mean_F4",
    "log_theta_power_F3", "log_theta_power_Fz", "log_theta_power_F4",
    "log_alpha_power_F3", "log_alpha_power_Fz", "log_alpha_power_F4",
    "log_beta_power_F3", "log_beta_power_Fz", "log_beta_power_F4", "AI_alpha",
)
RANDOM_SEED = 20260925
STAGE_CONFIG = {
    "cue_locked": {"variant": "nominal", "duration_s": 0.5},
    "target_locked": {"variant": "target_offset_2.2s", "duration_s": 0.8},
}


@dataclass(frozen=True)
class Candidate:
    name: str
    representation: str
    l2: float
    edge_components: int = 0
    shrinkage: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def baseline_correct_signal(signal: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Subtract each channel's mean over the pre-event interval [-0.1, 0)."""
    signal = np.asarray(signal, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if signal.ndim != 2 or time_s.ndim != 1 or signal.shape[1] != len(time_s):
        raise ValueError("signal must be channels x time and match time_s")
    # EEG grids may encode the event-time sample as a tiny negative float.
    baseline = (time_s >= -0.1 - 1e-9) & (time_s < -1e-9)
    if not baseline.any():
        raise ValueError("epoch has no samples in the pre-event baseline interval")
    return signal - np.mean(signal[:, baseline], axis=1, keepdims=True)


def erp_bin_features(
    corrected_signal: np.ndarray,
    time_s: np.ndarray,
    duration_s: float,
    bin_width_s: float,
) -> tuple[np.ndarray, list[str]]:
    """Return temporal mean amplitudes in channel-major, time-bin order."""
    signal = np.asarray(corrected_signal, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if signal.ndim != 2 or signal.shape[1] != len(time_s):
        raise ValueError("signal must be channels x time and match time_s")
    if duration_s <= 0 or bin_width_s <= 0:
        raise ValueError("duration_s and bin_width_s must be positive")
    channels = ("F3", "Fz", "F4")[: signal.shape[0]]
    out: list[float] = []
    names: list[str] = []
    n_bins = int(round(duration_s / bin_width_s))
    bin_ids = np.floor((time_s + 1e-9) / bin_width_s).astype(int)
    for channel_index, channel in enumerate(channels):
        for bin_index in range(n_bins):
            left = bin_index * bin_width_s
            right = min((bin_index + 1) * bin_width_s, duration_s)
            mask = (bin_ids == bin_index) & (time_s >= -1e-9) & (time_s < duration_s - 1e-9)
            if not mask.any():
                raise ValueError(f"No samples fall in ERP bin [{left}, {right})")
            out.append(float(np.mean(signal[channel_index, mask])))
            names.append(f"{channel}_erp_{left:.2f}_{right:.2f}s")
    return np.asarray(out, dtype=float), names


def fit_csp_filters(
    signals: np.ndarray,
    labels: np.ndarray,
    edge_components: int = 1,
    shrinkage: float = 0.01,
) -> np.ndarray:
    """Fit binary CSP spatial filters using only the provided training trials."""
    signals = np.asarray(signals, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if signals.ndim != 3 or len(labels) != len(signals):
        raise ValueError("signals must be trials x channels x time and match labels")
    classes = np.unique(labels)
    if len(classes) != 2 or edge_components < 1 or edge_components > signals.shape[1]:
        raise ValueError("CSP requires two classes and a valid number of edge components")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1")

    class_covariances = []
    for label in classes:
        covariances = []
        for trial in signals[labels == label]:
            trial = trial - np.mean(trial, axis=1, keepdims=True)
            covariance = trial @ trial.T
            trace = float(np.trace(covariance))
            covariance = covariance / trace if trace > np.finfo(float).eps else np.eye(signals.shape[1]) / signals.shape[1]
            covariances.append(covariance)
        mean_cov = np.mean(covariances, axis=0)
        mean_cov = (1 - shrinkage) * mean_cov + shrinkage * np.eye(signals.shape[1]) / signals.shape[1]
        class_covariances.append(mean_cov)

    positive, negative = class_covariances
    total = positive + negative
    eigenvalues, eigenvectors = eigh(positive, total, check_finite=True)
    order = np.argsort(eigenvalues)
    selected_candidates = np.concatenate([order[:edge_components], order[-edge_components:][::-1]])
    selected = np.asarray(list(dict.fromkeys(selected_candidates.tolist())), dtype=int)
    return eigenvectors[:, selected].T


def transform_csp(signals: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Transform trials to log projected variance fractions."""
    signals = np.asarray(signals, dtype=float)
    filters = np.asarray(filters, dtype=float)
    if signals.ndim != 3 or filters.ndim != 2 or signals.shape[1] != filters.shape[1]:
        raise ValueError("signals and CSP filters have incompatible shapes")
    projected = np.einsum("fc,nct->nft", filters, signals)
    variances = np.var(projected, axis=2)
    fractions = variances / np.maximum(variances.sum(axis=1, keepdims=True), np.finfo(float).eps)
    return np.log(np.maximum(fractions, np.finfo(float).eps))


def grouped_leave_one_group_out(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    group_values = np.asarray(groups).astype(str)
    folds = []
    for held_out in sorted(np.unique(group_values)):
        valid = np.flatnonzero(group_values == held_out)
        train = np.flatnonzero(group_values != held_out)
        if len(train) and len(valid):
            folds.append((train, valid, held_out))
    return folds


def candidate_grid() -> list[Candidate]:
    """Fixed compact feature/CSP grid; order supplies a deterministic tie break."""
    candidates = [Candidate("baseline_logistic_l2_1", "baseline_13", 1.0)]
    for representation in ("baseline_13", "erp_100ms", "erp_50ms", "spatial_100ms", "wave64"):
        for l2 in (0.1, 10.0, 100.0):
            if representation == "baseline_13" and l2 == 1.0:
                continue
            candidates.append(Candidate(f"logistic_{representation}_l2_{l2:g}", representation, l2))
        if representation != "baseline_13":
            candidates.append(Candidate(f"logistic_{representation}_l2_1", representation, 1.0))
    for components in (1, 2):
        for shrinkage in (0.01, 0.1):
            for l2 in (0.1, 1.0, 10.0):
                name = f"csp_e{components}_shrink{shrinkage:g}_l2_{l2:g}"
                candidates.append(Candidate(name, "csp", l2, components, shrinkage))
    return candidates


def scale_train_test(
    train: np.ndarray, evaluate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale[~np.isfinite(scale) | (scale <= np.finfo(float).eps)] = 1.0
    return (train - center) / scale, (evaluate - center) / scale


def fit_logistic_scores(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, l2: float
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    y_binary = (np.asarray(y_train) > 0).astype(float)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ parameters
        loss = np.logaddexp(0.0, logits).sum() - np.dot(y_binary, logits)
        loss += 0.5 * l2 * np.dot(parameters[1:], parameters[1:])
        residual = expit(logits) - y_binary
        gradient = design.T @ residual
        gradient[1:] += l2 * parameters[1:]
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return expit(np.column_stack([np.ones(len(x_eval)), x_eval]) @ result.x)


def _epoch_index(npz: Any) -> dict[tuple[str, int, str, str], int]:
    keys = zip(npz["record"], npz["original_trial_index"], npz["stage"], npz["analysis_variant"])
    mapping: dict[tuple[str, int, str, str], int] = {}
    for index, (record, trial, stage, variant) in enumerate(keys):
        key = (str(record), int(trial), str(stage), str(variant))
        if key in mapping:
            raise ValueError(f"Duplicate epoch key: {key}")
        mapping[key] = index
    return mapping


def _waveform_features(signal: np.ndarray, time_s: np.ndarray, duration: float) -> np.ndarray:
    grid = np.arange(0.0, duration - 1e-9, 1 / 64.0)
    selected = (time_s >= -1e-10) & (time_s < duration - 1e-10)
    times = time_s[selected]
    values = signal[:, selected]
    if len(times) < 2 or grid[-1] > times[-1] + 1e-8:
        raise ValueError("Epoch is too short for the requested waveform window")
    return np.concatenate([np.interp(grid, times, channel) for channel in values])


def build_stage_data(
    feature_path: Path, epoch_path: Path, stage: str
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Align feature rows and raw epochs; return metadata, fixed features, CSP signals."""
    config = STAGE_CONFIG[stage]
    frame = pd.read_csv(feature_path, encoding="utf-8-sig")
    frame = frame.loc[
        (frame["stage"] == stage)
        & (frame["analysis_variant"] == config["variant"])
        & frame["qc_valid"].fillna(False).astype(bool)
    ].copy()
    frame["cue_side"] = pd.to_numeric(frame["cue_side"], errors="coerce")
    frame["record"] = frame["record"].astype(str)
    frame = frame.dropna(subset=["cue_side", *BASELINE_FEATURES]).copy()
    frame["cue_side"] = frame["cue_side"].astype(int)
    frame = frame.sort_values(["record", "original_trial_index"]).reset_index(drop=True)

    with np.load(epoch_path, allow_pickle=False) as npz:
        mapping = _epoch_index(npz)
        erp_all = npz["erp_signal"]
        times_all = npz["relative_time_s"]
        counts = npz["sample_count"]
        aligned = []
        feature_100 = []
        feature_050 = []
        feature_spatial = []
        feature_wave = []
        csp_signals = []
        duration = config["duration_s"]
        for row in frame.itertuples(index=False):
            key = (str(row.record), int(row.original_trial_index), stage, config["variant"])
            if key not in mapping:
                raise KeyError(f"No epoch matches feature row {key}")
            epoch_index = mapping[key]
            count = int(counts[epoch_index])
            times = np.asarray(times_all[epoch_index, :count], dtype=float)
            signal = baseline_correct_signal(erp_all[epoch_index, :, :count], times)
            aligned.append(epoch_index)
            values, _ = erp_bin_features(signal, times, duration, 0.1)
            feature_100.append(values)
            values, _ = erp_bin_features(signal, times, duration, 0.05)
            feature_050.append(values)
            spatial_signal = np.vstack([signal, signal[0] - signal[1], signal[2] - signal[1], signal[0] - signal[2]])
            spatial_values, _ = erp_bin_features(spatial_signal, times, duration, 0.1)
            feature_spatial.append(spatial_values)
            feature_wave.append(_waveform_features(signal, times, duration))
            post_mask = (times >= -1e-10) & (times < duration - 1e-10)
            csp_signals.append(signal[:, post_mask])

    expected_lengths = {len(item) for item in csp_signals}
    if len(expected_lengths) != 1:
        raise ValueError(f"CSP epoch lengths vary unexpectedly: {expected_lengths}")
    fixed = {
        "baseline_13": frame.loc[:, BASELINE_FEATURES].to_numpy(dtype=float),
        "erp_100ms": np.asarray(feature_100, dtype=float),
        "erp_50ms": np.asarray(feature_050, dtype=float),
        "spatial_100ms": np.asarray(feature_spatial, dtype=float),
        "wave64": np.asarray(feature_wave, dtype=float),
    }
    return frame, fixed, np.asarray(csp_signals, dtype=float)


def _features_for_split(
    representation: str,
    candidate: Candidate,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    fixed: dict[str, np.ndarray],
    csp_signals: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if representation == "csp":
        filters = fit_csp_filters(
            csp_signals[train_idx], labels[train_idx], candidate.edge_components, candidate.shrinkage
        )
        return (
            transform_csp(csp_signals[train_idx], filters),
            transform_csp(csp_signals[eval_idx], filters),
        )
    return fixed[representation][train_idx], fixed[representation][eval_idx]


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    predicted = np.where(scores >= 0.5, 1, -1)
    return (
        {
            "accuracy": float(accuracy_score(y_true, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
            "auc": float(roc_auc_score((y_true > 0).astype(int), scores)),
            "macro_f1": float(f1_score(y_true, predicted, labels=[-1, 1], average="macro", zero_division=0)),
        },
        predicted,
    )


def score_indices(
    frame: pd.DataFrame,
    fixed: dict[str, np.ndarray],
    csp_signals: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    candidate: Candidate,
) -> tuple[dict[str, float], pd.DataFrame]:
    labels = frame["cue_side"].to_numpy(dtype=int)
    x_train_raw, x_eval_raw = _features_for_split(
        candidate.representation, candidate, train_idx, eval_idx, fixed, csp_signals, labels
    )
    x_train, x_eval = scale_train_test(x_train_raw, x_eval_raw)
    scores = fit_logistic_scores(x_train, labels[train_idx], x_eval, candidate.l2)
    metrics, predicted = _metrics(labels[eval_idx], scores)
    predictions = pd.DataFrame({
        "record": frame.iloc[eval_idx]["record"].astype(str).to_numpy(),
        "original_trial_index": frame.iloc[eval_idx]["original_trial_index"].to_numpy(dtype=int),
        "true_cue_side": labels[eval_idx],
        "predicted_cue_side": predicted,
        "score_right": scores,
    })
    return metrics, predictions


def _inner_score(
    frame: pd.DataFrame,
    fixed: dict[str, np.ndarray],
    csp_signals: np.ndarray,
    outer_train_idx: np.ndarray,
    candidate: Candidate,
) -> tuple[float, list[dict[str, Any]]]:
    records = frame.iloc[outer_train_idx]["record"].to_numpy()
    details = []
    for train_local, valid_local, held_out in grouped_leave_one_group_out(records):
        train_idx, valid_idx = outer_train_idx[train_local], outer_train_idx[valid_local]
        metrics, _ = score_indices(frame, fixed, csp_signals, train_idx, valid_idx, candidate)
        details.append({
            "inner_held_out_record": held_out,
            "n_train": int(len(train_idx)), "n_validation": int(len(valid_idx)), **metrics,
        })
    return float(np.mean([item["balanced_accuracy"] for item in details])), details


def prepare_scope(frame: pd.DataFrame, sample: str) -> np.ndarray:
    if sample == "raw_qc_pass":
        mask = np.ones(len(frame), dtype=bool)
    elif sample == "q1_quality_matched":
        mask = frame["q1_quality_pass"].fillna(False).astype(bool).to_numpy()
    else:
        raise ValueError(f"Unknown sample scope: {sample}")
    selected = frame.loc[mask, "cue_side"].to_numpy(dtype=int)
    if len(np.unique(selected)) < 2:
        raise ValueError(f"Scope {sample} contains fewer than two response classes")
    return np.flatnonzero(mask)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(
    feature_path: Path = DEFAULT_FEATURES,
    epoch_path: Path = DEFAULT_EPOCHS,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_grid()
    all_outer: list[dict[str, Any]] = []
    all_inner: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    baseline_predictions: list[pd.DataFrame] = []
    selected_predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for stage in ("cue_locked", "target_locked"):
        frame, fixed, csp_signals = build_stage_data(feature_path, epoch_path, stage)
        for sample in ("raw_qc_pass", "q1_quality_matched"):
            scope_idx = prepare_scope(frame, sample)
            scope_frame = frame.iloc[scope_idx].reset_index(drop=True)
            scope_fixed = {name: values[scope_idx] for name, values in fixed.items()}
            scope_csp = csp_signals[scope_idx]
            labels = scope_frame["cue_side"].to_numpy(dtype=int)
            folds = grouped_leave_one_group_out(scope_frame["record"].to_numpy())
            baseline = candidates[0]
            for outer_train, outer_valid, held_out in folds:
                training_records = sorted(scope_frame.iloc[outer_train]["record"].unique().tolist())
                if held_out in training_records:
                    raise AssertionError("Outer held-out record leaked into training data")
                baseline_metrics, baseline_pred = score_indices(
                    scope_frame, scope_fixed, scope_csp, outer_train, outer_valid, baseline
                )
                baseline_predictions.append(baseline_pred.assign(
                    sample=sample, stage=stage, candidate=baseline.name, held_out_record=held_out
                ))
                candidate_scores: list[tuple[float, int, Candidate]] = []
                for order, candidate in enumerate(candidates):
                    inner_mean, inner_details = _inner_score(
                        scope_frame, scope_fixed, scope_csp, outer_train, candidate
                    )
                    all_inner.extend({
                        "sample": sample, "stage": stage, "outer_held_out_record": held_out,
                        "candidate_order": order, **candidate.as_row(),
                        "inner_mean_balanced_accuracy": inner_mean, **detail,
                    } for detail in inner_details)
                    candidate_scores.append((inner_mean, order, candidate))
                    outer_metrics, _ = score_indices(
                        scope_frame, scope_fixed, scope_csp, outer_train, outer_valid, candidate
                    )
                    all_outer.append({
                        "sample": sample, "stage": stage, **candidate.as_row(),
                        "held_out_record": held_out, "n_train": int(len(outer_train)),
                        "n_test": int(len(outer_valid)), "training_records": ";".join(training_records),
                        **outer_metrics,
                    })
                best_inner, _, chosen = max(candidate_scores, key=lambda item: (item[0], -item[1]))
                chosen_metrics, chosen_pred = score_indices(
                    scope_frame, scope_fixed, scope_csp, outer_train, outer_valid, chosen
                )
                selected_rows.append({
                    "sample": sample, "stage": stage, "held_out_record": held_out,
                    "n_train": int(len(outer_train)), "n_test": int(len(outer_valid)),
                    "training_records": ";".join(training_records),
                    "selected_by_inner_balanced_accuracy": best_inner,
                    **chosen.as_row(), **chosen_metrics,
                    "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                    "balanced_accuracy_delta": chosen_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
                })
                selected_predictions.append(chosen_pred.assign(
                    sample=sample, stage=stage, candidate=chosen.name, held_out_record=held_out
                ))

            selected_subset = pd.DataFrame([row for row in selected_rows if row["sample"] == sample and row["stage"] == stage])
            base_subset = pd.DataFrame([row for row in all_outer if row["sample"] == sample and row["stage"] == stage and row["name"] == baseline.name])
            selected_mean = float(selected_subset["balanced_accuracy"].mean())
            base_mean = float(base_subset["balanced_accuracy"].mean())
            summaries.append({
                "sample": sample, "stage": stage, "n_trials": int(len(scope_idx)),
                "valid_record_folds": int(selected_subset["held_out_record"].nunique()),
                "baseline_mean_balanced_accuracy": base_mean,
                "nested_selected_mean_balanced_accuracy": selected_mean,
                "delta_percentage_points": 100 * (selected_mean - base_mean),
                "baseline_fold_balanced_accuracy": base_subset.set_index("held_out_record")["balanced_accuracy"].to_dict(),
                "selected_fold_balanced_accuracy": selected_subset.set_index("held_out_record")["balanced_accuracy"].to_dict(),
                "selected_candidates_by_fold": selected_subset.set_index("held_out_record")["name"].to_dict(),
                "delta_by_fold_percentage_points": (selected_subset.set_index("held_out_record")["balanced_accuracy"] - base_subset.set_index("held_out_record")["balanced_accuracy"]).mul(100).to_dict(),
            })

    outer_df = pd.DataFrame(all_outer)
    inner_df = pd.DataFrame(all_inner)
    selected_df = pd.DataFrame(selected_rows)
    candidate_summary = (
        outer_df.groupby(["sample", "stage", "name", "representation", "l2", "edge_components", "shrinkage"], dropna=False)
        .agg(mean_accuracy=("accuracy", "mean"), mean_balanced_accuracy=("balanced_accuracy", "mean"),
             sd_balanced_accuracy=("balanced_accuracy", "std"), mean_auc=("auc", "mean"),
             mean_macro_f1=("macro_f1", "mean"), valid_record_folds=("held_out_record", "nunique"))
        .reset_index()
    )
    outer_df.to_csv(output_dir / "outer_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(output_dir / "outer_candidate_summary.csv", index=False, encoding="utf-8-sig")
    inner_df.to_csv(output_dir / "inner_cv_scores.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(output_dir / "nested_selected_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(baseline_predictions, ignore_index=True).to_csv(output_dir / "baseline_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(selected_predictions, ignore_index=True).to_csv(output_dir / "nested_selected_oof_predictions.csv", index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "experiment": "time-resolved ERP and train-fold CSP nested leave-one-record-out search",
        "development_only": True,
        "inputs": {"trial_features": str(feature_path.resolve()), "raw_event_epochs": str(epoch_path.resolve())},
        "primary_metric": "balanced_accuracy",
        "outer_validation": "leave one complete recording out",
        "inner_selection": "leave one of the other three records out; scaling and CSP fit on training rows only",
        "candidate_count": len(candidates),
        "candidate_families": sorted({candidate.representation for candidate in candidates}),
        "results": summaries,
        "limitations": [
            "The same four recording files were inspected in the previous exploratory round; these scores are development-only and require independent replication.",
            "Only four outer recording folds are available, so record-level scores have high uncertainty.",
            "Cue side is the prediction label at both event anchors; the score measures side information, not clinical diagnosis.",
            "Target-locked epochs use the scheduled cue+2.2 s event anchor.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_findings(output_dir, summary)
    artifact_paths = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "experiment": summary["experiment"],
        "created_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "inputs": {"trial_features_sha256": _sha256(feature_path), "raw_event_epochs_sha256": _sha256(epoch_path)},
        "source_sha256": _sha256(Path(__file__)),
        "artifacts": [{"path": item.name, "sha256": _sha256(item), "bytes": item.stat().st_size} for item in artifact_paths],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_findings(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Temporal ERP / CSP search findings",
        "",
        "This is an exploratory development report. The frozen initial baseline remains under `baselines/20260925_initial/`.",
        "",
        "| Sample | Stage | Baseline BA | Nested selected BA | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for item in summary["results"]:
        lines.append(
            f"| {item['sample']} | {item['stage']} | {item['baseline_mean_balanced_accuracy']:.4f} | "
            f"{item['nested_selected_mean_balanced_accuracy']:.4f} | {item['delta_percentage_points']:+.2f} pp |"
        )
    lines.extend([
        "", "## Per-record selection", "",
        "The selected candidate for each held-out record is listed in `nested_selected_fold_metrics.csv`; use those rows and the OOF files when discussing consistency.",
        "Do not select a winner from the outer candidate maximum and present it as an unbiased estimate.",
        "", "## Limits", "",
        *[f"- {item}" for item in summary["limitations"]],
    ])
    (output_dir / "experiment_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--epochs", type=Path, default=DEFAULT_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_experiment(args.features, args.epochs, args.output_dir)
    for item in summary["results"]:
        print(
            f"{item['sample']} {item['stage']}: baseline BA="
            f"{item['baseline_mean_balanced_accuracy']:.3f}, selected BA="
            f"{item['nested_selected_mean_balanced_accuracy']:.3f}, "
            f"delta={item['delta_percentage_points']:+.2f} pp"
        )
    print(f"Wrote temporal search results to {args.output_dir}")


if __name__ == "__main__":
    main()
