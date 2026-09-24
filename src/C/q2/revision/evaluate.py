"""Held-out ERP, feature-classification, and preregistered mechanism checks."""
from functools import lru_cache

import numpy as np

try:
    from . import config
    from .fit import _interpolate_prediction
    from .frontend import load_stimulus, simulate_frontend
    from .model import (ModelParams, observable_modes_from_real,
                        unexplained_mode_from_real, simulate_forward)
except ImportError:
    import config
    from fit import _interpolate_prediction
    from frontend import load_stimulus, simulate_frontend
    from model import (ModelParams, observable_modes_from_real,
                       unexplained_mode_from_real, simulate_forward)


def extract_features(trials, time_ms):
    """Return the preregistered six real-EEG features (three windows x u0/u1)."""
    trials = np.asarray(trials, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)
    if trials.ndim != 3 or trials.shape[1] != 3 or trials.shape[2] != time_ms.size:
        raise ValueError("trials must be [trial, F3/Fz/F4, time]")
    modes = np.einsum("kc,nct->nkt", np.stack([config.U0, config.U1]), trials)
    features = []
    for low, high in config.FEATURE_WINDOWS_MS:
        mask = (time_ms >= low) & ((time_ms < high) if high != 700.0 else time_ms <= high)
        if not mask.any():
            raise ValueError(f"feature time window {low}-{high} ms has no measured samples")
        features.extend([modes[:, 0, mask].mean(axis=1), modes[:, 1, mask].mean(axis=1)])
    return np.stack(features, axis=1)


def fit_classifier(train_features, train_labels, train_record_ids):
    """Fit fixed-shrinkage LDA with equal record, class, and within-class weights."""
    x = np.asarray(train_features, dtype=float)
    y = np.asarray(train_labels, dtype=int)
    records = np.asarray(train_record_ids, dtype=str)
    if x.ndim != 2 or x.shape[1] != 6 or y.shape != (x.shape[0],) or records.shape != y.shape:
        raise ValueError("LDA requires N x 6 features and matching labels/record ids")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both left(0) and right(1) training classes are required")
    recs = np.unique(records)
    weights = np.zeros(len(y), dtype=float)
    for rec in recs:
        for label in (0, 1):
            mask = (records == rec) & (y == label)
            if not mask.any():
                raise ValueError(f"training record {rec} lacks class {label}")
            weights[mask] = 1.0 / (len(recs) * 2.0 * mask.sum())
    mean = np.sum(weights[:, None] * x, axis=0)
    variance = np.sum(weights[:, None] * (x - mean) ** 2, axis=0)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    class_means = []
    for label in (0, 1):
        mask = y == label
        local_w = weights[mask]
        local_w = local_w / local_w.sum()
        class_means.append(np.sum(local_w[:, None] * z[mask], axis=0))
    residual = np.empty_like(z)
    for label in (0, 1):
        residual[y == label] = z[y == label] - class_means[label]
    covariance = np.einsum("n,ni,nj->ij", weights, residual, residual)
    trace = float(np.trace(covariance))
    covariance = ((1.0 - config.LDA_SHRINKAGE) * covariance
                  + config.LDA_SHRINKAGE * trace / 6.0 * np.eye(6)
                  + 1e-6 * np.eye(6))
    direction = np.linalg.solve(covariance, class_means[1] - class_means[0])
    midpoint = (class_means[0] + class_means[1]) / 2.0
    return {"mean": mean, "scale": scale, "class_means": np.stack(class_means),
            "covariance": covariance, "direction": direction, "midpoint": midpoint,
            "training_records": recs.tolist(), "shrinkage": config.LDA_SHRINKAGE}


def predict_classifier(classifier, features):
    x = (np.asarray(features, dtype=float) - classifier["mean"]) / classifier["scale"]
    scores = (x - classifier["midpoint"]) @ classifier["direction"]
    return scores, (scores > 0).astype(int)


def _auc(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    positive, negative = s[y == 1], s[y == 0]
    if not positive.size or not negative.size:
        return float("nan")
    compare = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(compare > 0) + 0.5 * np.count_nonzero(compare == 0)) / compare.size)


def classification_leave_one_record(cases):
    """Train/evaluate one real-only 6-D LDA per held-out Stage1 MAT."""
    stage1 = [c for c in cases if c["stage"] == "Stage1" and c["eligible_classification"]]
    records = sorted({c["dataset"] for c in stage1})
    score_rows, fold_rows = [], []
    for heldout in records:
        train_x, train_y, train_records = [], [], []
        test_cases = [c for c in stage1 if c["dataset"] == heldout]
        if {c["condition"] for c in test_cases} != {"left", "right"}:
            fold_rows.append({"heldout_record": heldout, "status": "incomplete_left_right"})
            continue
        for case in stage1:
            if case["dataset"] == heldout:
                continue
            feat = extract_features(case["trials"], case["time_ms"])
            label = 0 if case["condition"] == "left" else 1
            train_x.append(feat)
            train_y.extend([label] * feat.shape[0])
            train_records.extend([case["dataset"]] * feat.shape[0])
        classifier = fit_classifier(np.concatenate(train_x), np.asarray(train_y), np.asarray(train_records))
        ys, scores, preds = [], [], []
        for case in sorted(test_cases, key=lambda c: c["condition"]):
            label = 0 if case["condition"] == "left" else 1
            feat = extract_features(case["trials"], case["time_ms"])
            local_scores, local_preds = predict_classifier(classifier, feat)
            ys.extend([label] * len(local_scores))
            scores.extend(local_scores.tolist())
            preds.extend(local_preds.tolist())
            for trial_i, (score, prediction) in enumerate(zip(local_scores, local_preds)):
                score_rows.append({"heldout_record": heldout, "condition": case["condition"],
                                   "trial_index": trial_i, "true_label": "right" if label else "left",
                                   "score_right_minus_left": float(score),
                                   "predicted_label": "right" if prediction else "left",
                                   "correct": bool(prediction == label)})
        y, score, pred = np.asarray(ys), np.asarray(scores), np.asarray(preds)
        recalls = [float(np.mean(pred[y == k] == k)) for k in (0, 1)]
        cm = [[int(np.count_nonzero((y == i) & (pred == j))) for j in (0, 1)] for i in (0, 1)]
        fold_rows.append({"heldout_record": heldout, "status": "complete", "n_trials": int(len(y)),
                          "balanced_accuracy": float(np.mean(recalls)), "recall_left": recalls[0],
                          "recall_right": recalls[1], "roc_auc": _auc(y, score),
                          "confusion_matrix": str(cm), "n_ties": int(np.count_nonzero(score == 0))})
    complete = [r for r in fold_rows if r.get("status") == "complete"]
    summary = {metric: float(np.mean([r[metric] for r in complete])) if complete else float("nan")
               for metric in ("balanced_accuracy", "roc_auc", "recall_left", "recall_right")}
    return score_rows, fold_rows, summary


def evaluate_fold(frozen_fit, frozen_classifier, heldout_cases, baseline=None):
    """Evaluate a held-out fold without access to an optimizer or refit state.

    ``baseline`` may map cue condition to a dictionary containing ``eeg`` and
    ``time_ms``; it is comparison-only and is never used to change the fit.
    """
    fit_values = frozen_fit.to_dict() if hasattr(frozen_fit, "to_dict") else dict(frozen_fit)
    params = ModelParams.from_any(fit_values.get("parameters", fit_values))
    amplitude = float(fit_values.get("amplitude", 1.0))
    resolution = int(fit_values.get("resolution", 64))
    erp_rows, trial_rows = [], []
    for case in heldout_cases:
        if case["stage"] != "Stage1" or case["condition"] not in ("left", "right"):
            continue
        result = run_forward_condition("Stage1", case["condition"], params, resolution,
                                       amplitude=amplitude)
        predicted = _interpolate_prediction(result.eeg_scaled, result.time_ms, case["time_ms"])
        metrics = erp_metrics(case["real"], predicted, case["time_ms"], "Stage1")
        erp_rows.append({"record": case["dataset"], "condition": case["condition"],
                         "n_trials": case["n_trials"], "rmse": metrics["rmse"],
                         "nrmse_by_real_rms": metrics["nrmse_by_real_rms"],
                         "channel_corr_mean": metrics["channel_corr_mean"],
                         "late_peaks": metrics["late_peaks"]})
        if frozen_classifier is not None and case.get("eligible_classification", False):
            features = extract_features(case["trials"], case["time_ms"])
            scores, labels = predict_classifier(frozen_classifier, features)
            truth = 1 if case["condition"] == "right" else 0
            trial_rows.extend({"record": case["dataset"], "trial_index": i,
                               "true_label": truth, "score": float(score),
                               "predicted_label": int(label)}
                              for i, (score, label) in enumerate(zip(scores, labels)))
        if baseline is not None and case["condition"] in baseline:
            old = baseline[case["condition"]]
            old_prediction = _interpolate_prediction(old["eeg"], old["time_ms"], case["time_ms"])
            erp_rows[-1]["baseline_metrics"] = erp_metrics(case["real"], old_prediction,
                                                             case["time_ms"], "Stage1")
    return {"erp_rows": erp_rows, "trial_rows": trial_rows}


def peak_summary(signal, time_ms, window):
    """Positive internal peak and explicit edge/no-peak status per channel."""
    y = np.asarray(signal, dtype=float)
    t = np.asarray(time_ms, dtype=float)
    lo, hi = window
    inds = np.flatnonzero((t >= lo) & (t <= hi))
    rows = []
    for ci, channel in enumerate(config.CHANNELS):
        if not inds.size:
            rows.append({"channel": channel, "peak_amplitude": float("nan"),
                         "peak_time_ms": float("nan"), "peak_status": "window_absent"})
            continue
        values = y[ci, inds]
        k = int(np.argmax(values))
        edge = k == 0 or k == len(inds) - 1
        status = ("no_positive_peak" if values[k] <= 0 else
                  "truncated_at_window_edge" if edge else "internal_positive_peak")
        rows.append({"channel": channel, "peak_amplitude": float(values[k]),
                     "peak_time_ms": float(t[inds[k]]), "peak_status": status})
    return rows


def erp_metrics(real, predicted, time_ms, stage):
    """Report rank-two observable fit and full sensor error separately."""
    y, p = np.asarray(real, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.shape != (3, len(time_ms)):
        raise ValueError("ERP arrays must share [3,time] shape")
    full_residual = y - p
    full_scale = float(np.sqrt(np.mean(y * y)))
    full_rmse = float(np.sqrt(np.mean(full_residual * full_residual)))
    corrs = []
    for channel in range(3):
        a, b = y[channel], p[channel]
        corrs.append(float(np.corrcoef(a, b)[0, 1])
                      if np.std(a) >= 1e-12 and np.std(b) >= 1e-12 else float("nan"))
    real_modes = observable_modes_from_real(y)
    predicted_modes = observable_modes_from_real(p)
    observable_residual = real_modes - predicted_modes
    observable_scale = float(np.sqrt(np.mean(real_modes * real_modes)))
    observable_rmse = float(np.sqrt(np.mean(observable_residual * observable_residual)))
    mode_corrs = []
    for mode in range(2):
        a, b = real_modes[mode], predicted_modes[mode]
        mode_corrs.append(float(np.corrcoef(a, b)[0, 1])
                           if np.std(a) >= 1e-12 and np.std(b) >= 1e-12 else float("nan"))
    real_u2 = unexplained_mode_from_real(y)
    predicted_u2 = unexplained_mode_from_real(p)
    u2_residual = real_u2 - predicted_u2
    all_mode_energy = (np.sum(np.mean(real_modes * real_modes, axis=-1))
                       + np.mean(real_u2 * real_u2))
    u2_energy_share = (float(np.mean(real_u2 * real_u2) / all_mode_energy)
                       if all_mode_energy > 0 else float("nan"))
    late = config.STAGES[stage]["late"]
    finite = np.isfinite(corrs)
    finite_modes = np.isfinite(mode_corrs)
    return {"rmse": full_rmse,
            "nrmse_by_real_rms": full_rmse / full_scale if full_scale else float("nan"),
            "channel_corr_mean": float(np.mean(np.asarray(corrs)[finite])) if finite.any() else float("nan"),
            "channel_correlations": corrs,
            "full_sensor_rmse": full_rmse,
            "full_sensor_nrmse_by_real_rms": full_rmse / full_scale if full_scale else float("nan"),
            "observable_rmse": observable_rmse,
            "observable_nrmse_by_real_rms": (observable_rmse / observable_scale
                                              if observable_scale else float("nan")),
            "observable_mode_correlations": mode_corrs,
            "observable_corr_mean": (float(np.mean(np.asarray(mode_corrs)[finite_modes]))
                                     if finite_modes.any() else float("nan")),
            "u2_unexplained_rms": float(np.sqrt(np.mean(u2_residual * u2_residual))),
            "real_u2_rms": float(np.sqrt(np.mean(real_u2 * real_u2))),
            "predicted_u2_rms": float(np.sqrt(np.mean(predicted_u2 * predicted_u2))),
            "u2_energy_share": u2_energy_share,
            "late_peaks": peak_summary(y, time_ms, late)}


def estimate_shared_amplitude(cases, predictions):
    """One nonnegative scalar from training records/cases, each case equally weighted."""
    if not cases:
        return 0.0
    records = sorted({c["dataset"] for c in cases})
    weights = {id(c): 1.0 / len(records) / 2.0 for c in cases}
    numerator, denominator = 0.0, 0.0
    for case in cases:
        pred_item = predictions[(case["dataset"], case["condition"])]
        pred = observable_modes_from_real(
            _interpolate_prediction(pred_item["eeg"], pred_item["time_ms"], case["time_ms"]))
        obs, weight = observable_modes_from_real(case["real"]), weights[id(case)]
        numerator += weight * float(np.mean(pred * obs))
        denominator += weight * float(np.mean(pred * pred))
    return max(0.0, numerator / denominator) if denominator > 1e-18 else 0.0


@lru_cache(maxsize=32)
def _cached_condition_frontend(stage, condition, tau_a, resolution, include_offset, remove_position):
    stimulus = load_stimulus(stage, condition)
    return simulate_frontend(stimulus, params={"tau_a": float(tau_a)},
                             resolution=int(resolution), include_offset=bool(include_offset),
                             remove_position=bool(remove_position))


def run_forward_condition(stage, condition, parameters, resolution=64,
                          include_offset=True, remove_position=False, amplitude=1.0):
    params = ModelParams.from_any(parameters)
    front = _cached_condition_frontend(stage, condition, float(params.tau_a), int(resolution),
                                       bool(include_offset), bool(remove_position))
    return simulate_forward(front, params=params, amplitude=amplitude)


def mechanism_control_rows(fit_by_record, resolution=64, progress=False):
    """Apply no-position and no-offset controls with frozen fold parameters."""
    out = []
    for record, fit in fit_by_record.items():
        if progress:
            print(f"  mechanism controls for {record}...", flush=True)
        params = ModelParams.from_any(fit.parameters)
        regular = {condition: run_forward_condition("Stage1", condition, params, resolution,
                                                     amplitude=fit.amplitude).eeg_scaled
                   for condition in ("left", "right")}
        regular_difference = regular["right"] - regular["left"]
        regular_rms = float(np.sqrt(np.mean(regular_difference ** 2)))
        for control, kwargs in (("remove_position", {"remove_position": True}),
                                ("remove_cue_offset", {"include_offset": False})):
            if progress:
                print(f"    {control}", flush=True)
            predictions = {}
            for condition in ("left", "right"):
                result = run_forward_condition("Stage1", condition, params, resolution,
                                               amplitude=fit.amplitude, **kwargs)
                predictions[condition] = result.eeg_scaled
            time_ms = result.time_ms
            late = (time_ms >= 250) & (time_ms <= 700)
            difference = predictions["right"] - predictions["left"]
            control_rms = float(np.sqrt(np.mean(difference ** 2)))
            out.append({"heldout_record": record, "control": control,
                        "parameter_refit": False, "amplitude_refit": False,
                        "regular_difference_rms_all": regular_rms,
                        "left_right_difference_rms_all": control_rms,
                        "difference_retained_fraction": control_rms / regular_rms if regular_rms > 1e-12 else float("nan"),
                        "left_right_difference_rms_250_700": float(np.sqrt(np.mean(difference[:, late] ** 2))),
                        "left_Fz_late_peak_ms": peak_summary(predictions["left"], time_ms, (250.0, 700.0))[1]["peak_time_ms"],
                        "right_Fz_late_peak_ms": peak_summary(predictions["right"], time_ms, (250.0, 700.0))[1]["peak_time_ms"]})
    return out
