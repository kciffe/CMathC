"""Independent real-EEG separability and held-out forward-model evaluation."""
from functools import lru_cache

import numpy as np

try:
    from . import config
    from .fit import _interpolate_prediction
    from .frontend import load_stimulus, simulate_frontend
    from .model import (ModelParams, observable_modes_from_real,
                        simulate_forward)
except ImportError:
    import config
    from fit import _interpolate_prediction
    from frontend import load_stimulus, simulate_frontend
    from model import ModelParams, observable_modes_from_real, simulate_forward


def extract_features(trials, time_ms, mode_count=3):
    """Compute per-trial sensor-mode means in three fixed windows.

    Feature order is mode-major, so the six-feature u0/u1 vector is exactly
    the prefix of the nine-feature u0/u1/u2 vector.
    """
    trials = np.asarray(trials, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)
    if (trials.ndim != 3 or trials.shape[1] != 3 or trials.shape[2] != time_ms.size
            or mode_count not in (2, 3)):
        raise ValueError("trials must be [trial,F3/Fz/F4,time], mode_count 2 or 3")
    modes = observable_modes_from_real(trials)[:, :mode_count, :]
    features = []
    for mode in range(mode_count):
        for low, high in config.FEATURE_WINDOWS_MS:
            mask = (time_ms >= low) & (time_ms <= high)
            if not mask.any():
                raise ValueError(f"feature window {low}-{high} ms has no measured samples")
            features.append(modes[:, mode, :][:, mask].mean(axis=1))
    return np.stack(features, axis=1)


def fit_classifier(train_features, train_labels, train_record_ids):
    """Fit a fixed-shrinkage LDA with equal record/class contribution."""
    x = np.asarray(train_features, dtype=float)
    y = np.asarray(train_labels, dtype=int)
    records = np.asarray(train_record_ids, dtype=str)
    if x.ndim != 2 or x.shape[1] not in (6, 9) or y.shape != (x.shape[0],) or records.shape != y.shape:
        raise ValueError("LDA requires N x 6 or N x 9 features and matching labels/records")
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
        local_w = weights[mask] / weights[mask].sum()
        class_means.append(np.sum(local_w[:, None] * z[mask], axis=0))
    residual = np.empty_like(z)
    for label in (0, 1):
        residual[y == label] = z[y == label] - class_means[label]
    covariance = np.einsum("n,ni,nj->ij", weights, residual, residual)
    dimension = x.shape[1]
    trace = float(np.trace(covariance))
    covariance = ((1.0 - config.LDA_SHRINKAGE) * covariance
                  + config.LDA_SHRINKAGE * trace / dimension * np.eye(dimension)
                  + 1e-6 * np.eye(dimension))
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
    y, s = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    positive, negative = s[y == 1], s[y == 0]
    if not positive.size or not negative.size:
        return float("nan")
    compare = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(compare > 0) + .5 * np.count_nonzero(compare == 0)) / compare.size)


def classification_leave_one_record(cases, mode_count=3):
    """Real-only LO-record LDA; no forward-model output enters the classifier."""
    stage1 = [c for c in cases if c["stage"] == "Stage1" and c["eligible_classification"]]
    records = sorted({c["dataset"] for c in stage1})
    score_rows, fold_rows = [], []
    for heldout in records:
        train_x, train_y, train_records = [], [], []
        test_cases = [c for c in stage1 if c["dataset"] == heldout]
        if {c["condition"] for c in test_cases} != {"left", "right"}:
            fold_rows.append({"heldout_record": heldout, "mode_count": mode_count,
                              "status": "incomplete_left_right"})
            continue
        for case in stage1:
            if case["dataset"] == heldout:
                continue
            feat = extract_features(case["trials"], case["time_ms"], mode_count)
            label = int(case["condition"] == "right")
            train_x.append(feat)
            train_y.extend([label] * len(feat))
            train_records.extend([case["dataset"]] * len(feat))
        classifier = fit_classifier(np.concatenate(train_x), np.asarray(train_y),
                                    np.asarray(train_records))
        ys, scores, preds = [], [], []
        for case in sorted(test_cases, key=lambda c: c["condition"]):
            label = int(case["condition"] == "right")
            feat = extract_features(case["trials"], case["time_ms"], mode_count)
            local_scores, local_preds = predict_classifier(classifier, feat)
            ys.extend([label] * len(local_scores))
            scores.extend(local_scores.tolist())
            preds.extend(local_preds.tolist())
            for trial_i, (score, prediction) in enumerate(zip(local_scores, local_preds)):
                score_rows.append({"heldout_record": heldout, "mode_count": mode_count,
                                   "condition": case["condition"], "trial_index": trial_i,
                                   "true_label": "right" if label else "left",
                                   "score_right_minus_left": float(score),
                                   "predicted_label": "right" if prediction else "left",
                                   "correct": bool(prediction == label)})
        y, score, pred = np.asarray(ys), np.asarray(scores), np.asarray(preds)
        recalls = [float(np.mean(pred[y == k] == k)) for k in (0, 1)]
        cm = [[int(np.count_nonzero((y == i) & (pred == j))) for j in (0, 1)] for i in (0, 1)]
        fold_rows.append({"heldout_record": heldout, "mode_count": mode_count,
                          "status": "complete", "n_trials": int(len(y)),
                          "balanced_accuracy": float(np.mean(recalls)), "recall_left": recalls[0],
                          "recall_right": recalls[1], "roc_auc": _auc(y, score),
                          "confusion_matrix": str(cm), "n_ties": int(np.count_nonzero(score == 0))})
    complete = [r for r in fold_rows if r.get("status") == "complete"]
    summary = {metric: float(np.mean([r[metric] for r in complete])) if complete else float("nan")
               for metric in ("balanced_accuracy", "roc_auc", "recall_left", "recall_right")}
    summary.update({"mode_count": mode_count, "n_heldout_records": len(complete)})
    return score_rows, fold_rows, summary


def split_half_stability(cases, repeats=config.SPLIT_HALF_REPEATS,
                         seed=config.SPLIT_HALF_SEED):
    """Assess reproducibility of right-minus-left mode waveforms within each MAT."""
    rng = np.random.default_rng(seed)
    summaries = []
    for record in sorted({c["dataset"] for c in cases if c["stage"] == "Stage1"}):
        by_condition = {c["condition"]: c for c in cases
                        if c["dataset"] == record and c["stage"] == "Stage1"
                        and c["condition"] in ("left", "right")}
        if set(by_condition) != {"left", "right"}:
            continue
        left, right = by_condition["left"], by_condition["right"]
        if not np.array_equal(left["time_ms"], right["time_ms"]):
            raise ValueError(f"{record}: left/right time axes differ")
        left_modes = observable_modes_from_real(left["trials"])
        right_modes = observable_modes_from_real(right["trials"])
        n_l, n_r, time = len(left_modes), len(right_modes), left["time_ms"]
        if min(n_l, n_r) < 4:
            continue
        corr_samples = [[] for _ in range(3)]
        rms_samples = [[] for _ in range(3)]
        for _ in range(repeats):
            il, ir = rng.permutation(n_l), rng.permutation(n_r)
            half_l, half_r = n_l // 2, n_r // 2
            d1 = (right_modes[ir[:half_r]].mean(axis=0)
                  - left_modes[il[:half_l]].mean(axis=0))
            d2 = (right_modes[ir[half_r:2 * half_r]].mean(axis=0)
                  - left_modes[il[half_l:2 * half_l]].mean(axis=0))
            for mode in range(3):
                a, b = d1[mode], d2[mode]
                if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                    corr_samples[mode].append(float(np.corrcoef(a, b)[0, 1]))
                rms_samples[mode].append(float(np.sqrt(np.mean(a * a))))
        for mode, label in enumerate(("u0_common", "u1_lateral", "u2_shape")):
            corr = np.asarray(corr_samples[mode], dtype=float)
            rms = np.asarray(rms_samples[mode], dtype=float)
            summaries.append({"record": record, "mode": label, "mode_index": mode,
                              "n_trials_left": n_l, "n_trials_right": n_r,
                              "split_repeats": repeats, "valid_correlation_repeats": int(corr.size),
                              "split_half_correlation_median": float(np.median(corr)) if corr.size else float("nan"),
                              "split_half_correlation_q25": float(np.quantile(corr, .25)) if corr.size else float("nan"),
                              "split_half_correlation_q75": float(np.quantile(corr, .75)) if corr.size else float("nan"),
                              "difference_rms_median": float(np.median(rms)),
                              "difference_rms_q25": float(np.quantile(rms, .25)),
                              "difference_rms_q75": float(np.quantile(rms, .75)),
                              "time_start_ms": float(time[0]), "time_end_ms": float(time[-1])})
    return summaries


def _separation_stat(features, labels, dimensions):
    x = np.asarray(features[:, :dimensions], dtype=float)
    labels = np.asarray(labels, dtype=int)
    scale = np.std(x, axis=0, ddof=1)
    scale[scale < 1e-12] = 1.0
    difference = (x[labels == 1].mean(axis=0) - x[labels == 0].mean(axis=0)) / scale
    return float(difference @ difference)


def within_record_permutation(cases, repeats=config.PERMUTATION_REPEATS,
                              seed=config.PERMUTATION_SEED):
    """Within-MAT label permutation of fixed 6- and 9-feature mean separation."""
    rng = np.random.default_rng(seed)
    rows = []
    for record in sorted({c["dataset"] for c in cases if c["stage"] == "Stage1"}):
        by_condition = {c["condition"]: c for c in cases
                        if c["dataset"] == record and c["stage"] == "Stage1"
                        and c["condition"] in ("left", "right")}
        if set(by_condition) != {"left", "right"}:
            continue
        left, right = by_condition["left"], by_condition["right"]
        features = np.concatenate([extract_features(left["trials"], left["time_ms"], 3),
                                   extract_features(right["trials"], right["time_ms"], 3)])
        labels = np.r_[np.zeros(len(left["trials"]), dtype=int),
                       np.ones(len(right["trials"]), dtype=int)]
        observed6 = _separation_stat(features, labels, 6)
        observed9 = _separation_stat(features, labels, 9)
        null6 = np.empty(repeats)
        null9 = np.empty(repeats)
        for i in range(repeats):
            permuted = rng.permutation(labels)
            null6[i] = _separation_stat(features, permuted, 6)
            null9[i] = _separation_stat(features, permuted, 9)
        p6 = (1 + np.count_nonzero(null6 >= observed6)) / (repeats + 1)
        p9 = (1 + np.count_nonzero(null9 >= observed9)) / (repeats + 1)
        rows.append({"record": record, "n_left": len(left["trials"]), "n_right": len(right["trials"]),
                     "permutation_repeats": repeats, "separation_stat_6d": observed6,
                     "permutation_p_6d_exploratory": float(p6),
                     "separation_stat_9d": observed9, "permutation_p_9d_primary": float(p9),
                     "null_q95_9d": float(np.quantile(null9, .95)),
                     "interpretation": "within-record trial-label association; not LO-record generalization"})
    # Correct the four primary 9-D recordwise tests as one family. The 6-D
    # values remain explicitly exploratory and are not used for the primary
    # conclusion.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["permutation_p_9d_primary"])
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * rows[index]["permutation_p_9d_primary"])
        running = max(running, adjusted)
        rows[index]["permutation_p_9d_holm"] = running
    return rows


def peak_summary(signal, time_ms, window):
    y, t = np.asarray(signal, dtype=float), np.asarray(time_ms, dtype=float)
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
    y, p = np.asarray(real, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.shape != (3, len(time_ms)):
        raise ValueError("ERP arrays must share [F3/Fz/F4,time] shape")
    modes_y, modes_p = observable_modes_from_real(y), observable_modes_from_real(p)
    full_rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    scale = float(np.sqrt(np.mean(y * y)))
    channel_corrs, mode_corrs = [], []
    for channel in range(3):
        a, b = y[channel], p[channel]
        channel_corrs.append(float(np.corrcoef(a, b)[0, 1])
                             if np.std(a) >= 1e-12 and np.std(b) >= 1e-12 else float("nan"))
    for mode in range(3):
        a, b = modes_y[mode], modes_p[mode]
        mode_corrs.append(float(np.corrcoef(a, b)[0, 1])
                           if np.std(a) >= 1e-12 and np.std(b) >= 1e-12 else float("nan"))
    finite = np.isfinite(channel_corrs)
    mode_finite = np.isfinite(mode_corrs)
    return {"full_sensor_rmse": full_rmse,
            "full_sensor_nrmse_by_real_rms": full_rmse / scale if scale else float("nan"),
            "channel_correlations": channel_corrs,
            "channel_corr_mean": float(np.mean(np.asarray(channel_corrs)[finite])) if finite.any() else float("nan"),
            "mode_correlations": mode_corrs,
            "mode_corr_mean": float(np.mean(np.asarray(mode_corrs)[mode_finite])) if mode_finite.any() else float("nan"),
            "real_modes_rms": np.sqrt(np.mean(modes_y ** 2, axis=1)).tolist(),
            "predicted_modes_rms": np.sqrt(np.mean(modes_p ** 2, axis=1)).tolist(),
            "late_peaks": peak_summary(y, time_ms, config.STAGES[stage]["late"])}


def estimate_shared_amplitude(cases, predictions):
    """Estimate one signed shared gain from training cases in the 3-mode space."""
    if not cases:
        return 0.0
    records = sorted({c["dataset"] for c in cases})
    weights = {id(c): 1.0 / len(records) / 2.0 for c in cases}
    numerator = denominator = 0.0
    for case in cases:
        pred_item = predictions[(case["dataset"], case["condition"])]
        pred = observable_modes_from_real(_interpolate_prediction(
            pred_item["eeg"], pred_item["time_ms"], case["time_ms"]))
        obs = observable_modes_from_real(case["real"])
        weight = weights[id(case)]
        numerator += weight * float(np.mean(pred * obs))
        denominator += weight * float(np.mean(pred * pred))
    return numerator / denominator if denominator > 1e-18 else 0.0


@lru_cache(maxsize=32)
def _cached_condition_frontend(stage, condition, tau_a, resolution, include_offset, remove_position):
    stimulus = load_stimulus(stage, condition)
    return simulate_frontend(stimulus, params={"tau_a": float(tau_a)}, resolution=int(resolution),
                             include_offset=bool(include_offset), remove_position=bool(remove_position))


def run_forward_condition(stage, condition, parameters, resolution=64,
                          include_offset=True, remove_position=False, amplitude=1.0,
                          feature_route="opponent", observation_rank=3):
    params = ModelParams.from_any(parameters)
    front = _cached_condition_frontend(stage, condition, float(params.tau_a), int(resolution),
                                       bool(include_offset), bool(remove_position))
    return simulate_forward(front, params=params, amplitude=amplitude,
                            feature_route=feature_route, observation_rank=observation_rank)


def mechanism_control_rows(fit_by_record, resolution=64, progress=False,
                           feature_route="opponent", observation_rank=3):
    """Apply the no-position and no-cue-offset controls with frozen fit values."""
    rows = []
    for record, fit in fit_by_record.items():
        if progress:
            print(f"  mechanism controls: {record}", flush=True)
        params = ModelParams.from_any(fit.parameters)
        regular = {condition: run_forward_condition("Stage1", condition, params, resolution,
                     amplitude=fit.amplitude, feature_route=feature_route,
                     observation_rank=observation_rank).eeg_scaled
                   for condition in ("left", "right")}
        regular_diff = regular["right"] - regular["left"]
        regular_rms = float(np.sqrt(np.mean(regular_diff ** 2)))
        for control, kwargs in (("remove_position", {"remove_position": True}),
                                ("remove_cue_offset", {"include_offset": False})):
            predictions = {}
            for condition in ("left", "right"):
                predictions[condition] = run_forward_condition(
                    "Stage1", condition, params, resolution, amplitude=fit.amplitude,
                    feature_route=feature_route, observation_rank=observation_rank,
                    **kwargs).eeg_scaled
            difference = predictions["right"] - predictions["left"]
            rms = float(np.sqrt(np.mean(difference ** 2)))
            rows.append({"heldout_record": record, "control": control,
                         "parameter_refit": False, "amplitude_refit": False,
                         "regular_lr_difference_rms": regular_rms,
                         "control_lr_difference_rms": rms,
                         "difference_retained_fraction": rms / regular_rms if regular_rms > 1e-12 else float("nan")})
    return rows
