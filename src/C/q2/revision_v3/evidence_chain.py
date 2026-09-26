"""Fixed-window, record-held-out evidence chain for left/right cue analysis.

The model analysis and real-EEG classifier are deliberately separate:
cross-fitted forward predictions decompose the model's right-minus-left
response, while a fixed 9-D ERP feature is evaluated on a held-out MAT record.
No feature or time-window selection is performed from held-out data.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config
    from .evaluate import fit_classifier, predict_classifier, _auc
    from .fit import _cached_left_frontend
    from .frontend import load_stimulus, mirror_stage1_frontend
    from .model import ModelParams, simulate_forward
    from .observation import filter_resample_baseline
except ImportError:
    import config
    from evaluate import fit_classifier, predict_classifier, _auc
    from fit import _cached_left_frontend
    from frontend import load_stimulus, mirror_stage1_frontend
    from model import ModelParams, simulate_forward
    from observation import filter_resample_baseline


ELECTRODES = ("F3", "Fz", "F4")
WINDOWS_MS = tuple(config.CANDIDATE_ERP_WINDOWS_MS)
CANDIDATE_FEATURE_NAMES = tuple(
    f"{window_label}_{electrode}"
    for window_label in config.FEATURE_WINDOW_LABELS
    for electrode in ELECTRODES)
SOURCE_NAMES = (
    "early_left_hemisphere_from_right_visual_field",
    "early_right_hemisphere_from_left_visual_field",
    "configuration_left_hemisphere_from_right_visual_field",
    "configuration_right_hemisphere_from_left_visual_field",
    "bilateral_shape_preference_opponent_midline",
)
POPULATION_NAMES = ("early_visual", "configuration", "shape_preference")
POPULATION_CHANNELS = (
    ("left_visual_field", "right_visual_field"),
    ("left_visual_field", "right_visual_field"),
    ("left_triangle_template_preference", "right_triangle_template_preference"),
)


def extract_electrode_window_features(trials, time_ms):
    """Return window-major means for F3/Fz/F4 using half-open intervals."""
    values = np.asarray(trials, dtype=float)
    time = np.asarray(time_ms, dtype=float)
    if values.ndim != 3 or values.shape[1] != 3 or values.shape[2] != time.size:
        raise ValueError("trials must have shape [trial, F3/Fz/F4, time]")
    if not np.isfinite(values).all() or not np.isfinite(time).all():
        raise ValueError("trials and time_ms must be finite")
    columns = []
    for low, high in WINDOWS_MS:
        mask = (time >= low) & (time < high)
        if not mask.any():
            raise ValueError(f"fixed feature window [{low}, {high}) ms has no samples")
        columns.extend(values[:, channel, :][:, mask].mean(axis=1)
                       for channel in range(3))
    return np.column_stack(columns)


def _case_features(case):
    return extract_electrode_window_features(case["trials"], case["time_ms"])


def classify_electrode_features(cases):
    """Leave-one-MAT-out shrinkage LDA using only the predeclared nine means."""
    stage1 = [case for case in cases
              if case["stage"] == "Stage1" and case["eligible_classification"]]
    records = sorted({case["dataset"] for case in stage1})
    if len(records) < 3:
        raise ValueError("at least three eligible MAT records are needed for record holdout")
    score_rows, fold_rows = [], []
    for heldout in records:
        heldout_cases = [case for case in stage1 if case["dataset"] == heldout]
        if {case["condition"] for case in heldout_cases} != {"left", "right"}:
            fold_rows.append({"heldout_record": heldout, "status": "incomplete_left_right"})
            continue
        train_features, train_labels, train_record_ids = [], [], []
        for case in stage1:
            if case["dataset"] == heldout:
                continue
            features = _case_features(case)
            label = int(case["condition"] == "right")
            train_features.append(features)
            train_labels.extend([label] * len(features))
            train_record_ids.extend([case["dataset"]] * len(features))
        classifier = fit_classifier(np.concatenate(train_features),
                                    np.asarray(train_labels),
                                    np.asarray(train_record_ids))
        y_true, scores, predictions = [], [], []
        for case in sorted(heldout_cases, key=lambda row: row["condition"]):
            label = int(case["condition"] == "right")
            features = _case_features(case)
            score, prediction = predict_classifier(classifier, features)
            y_true.extend([label] * len(score))
            scores.extend(score.tolist())
            predictions.extend(prediction.tolist())
            for index, (value, predicted) in enumerate(zip(score, prediction)):
                score_rows.append({
                    "heldout_record": heldout,
                    "condition": case["condition"],
                    "trial_index": index,
                    "true_label": "right" if label else "left",
                    "score_right_minus_left": float(value),
                    "predicted_label": "right" if predicted else "left",
                    "correct": bool(predicted == label),
                    "feature_count": len(CANDIDATE_FEATURE_NAMES),
                    "feature_windows_ms": "[100,250);[250,500);[500,800)",
                    "training_records": ";".join(classifier["training_records"]),
                })
        y = np.asarray(y_true, dtype=int)
        score = np.asarray(scores, dtype=float)
        pred = np.asarray(predictions, dtype=int)
        recalls = [float(np.mean(pred[y == label] == label)) for label in (0, 1)]
        cm = [[int(np.count_nonzero((y == actual) & (pred == guessed)))
               for guessed in (0, 1)] for actual in (0, 1)]
        fold_rows.append({
            "heldout_record": heldout,
            "training_records": ";".join(classifier["training_records"]),
            "status": "complete",
            "n_test_trials": int(len(y)),
            "n_left": int(np.count_nonzero(y == 0)),
            "n_right": int(np.count_nonzero(y == 1)),
            "feature_count": len(CANDIDATE_FEATURE_NAMES),
            "classifier": "fixed-shrinkage LDA; train-only scaling",
            "shrinkage": float(classifier["shrinkage"]),
            "balanced_accuracy": float(np.mean(recalls)),
            "recall_left": recalls[0],
            "recall_right": recalls[1],
            "roc_auc": _auc(y, score),
            "confusion_matrix_left_right": str(cm),
        })
    complete = [row for row in fold_rows if row.get("status") == "complete"]
    if not complete:
        raise ValueError("no complete leave-one-record-out folds")
    summary = {
        "status": "complete",
        "validation": "leave-one-MAT-record-out; all preprocessing and features fixed in advance",
        "feature_count": len(CANDIDATE_FEATURE_NAMES),
        "feature_names": ";".join(CANDIDATE_FEATURE_NAMES),
        "window_definition_ms": "[100,250);[250,500);[500,800)",
        "classifier": "fixed-shrinkage LDA; standardization fit on training records only",
        "shrinkage": float(config.LDA_SHRINKAGE),
        "heldout_record_count": len(complete),
        "trial_count": len(score_rows),
        "macro_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in complete])),
        "macro_roc_auc": float(np.mean([row["roc_auc"] for row in complete])),
        "record_level_balanced_accuracy_sd": float(np.std(
            [row["balanced_accuracy"] for row in complete], ddof=1)) if len(complete) > 1 else 0.0,
        "caveat": "four MAT files are recordings, not verified independent participants",
    }
    return score_rows, fold_rows, summary


def _full_epoch_process(signal, time_ms):
    """Apply the exact Q1 filter/resample/baseline operator to arbitrary channels."""
    source_fs, target_fs = 256.0, 128.0
    epoch_start_s, epoch_end_s = -1.0, 3.0
    sample_count = int(round((epoch_end_s - epoch_start_s) * source_fs))
    epoch_time_s = epoch_start_s + np.arange(sample_count) / source_fs
    full_epoch = np.vstack([
        np.interp(epoch_time_s * 1000.0, time_ms, row, left=0.0, right=0.0)
        for row in np.asarray(signal, dtype=float)
    ])
    return filter_resample_baseline(full_epoch, source_fs=source_fs,
                                    target_fs=target_fs,
                                    start_s=epoch_start_s)


def _process_model_and_sources(result, amplitude):
    lead = np.asarray(result.diagnostics["leadfield"], dtype=float)
    source = np.asarray(result.source_proxy, dtype=float)
    contributions = float(amplitude) * lead[:, :, None] * source[None, :, :]
    total = float(amplitude) * np.asarray(result.eeg, dtype=float)
    combined = np.vstack((contributions.reshape(15, -1), total))
    processed, time_s = _full_epoch_process(combined, result.time_ms)
    source_processed = processed[:15].reshape(3, 5, -1)
    saved_total_processed = processed[15:]
    total_processed = source_processed.sum(axis=1)
    residual = total_processed - saved_total_processed
    relative_rms_error = float(np.sqrt(np.mean(residual ** 2)) /
                               max(np.sqrt(np.mean(saved_total_processed ** 2)), 1e-12))
    # The model stores leadfield @ source as float32, whereas source-wise
    # contributions are expanded in float64. Compare after the same linear
    # observation operator and allow only the propagated single-precision
    # roundoff; a missing source or mismatched leadfield exceeds this bound.
    if relative_rms_error > 1e-5:
        raise AssertionError(
            "filtered source contributions do not reproduce the saved EEG "
            f"within float32 roundoff (relative RMS error={relative_rms_error:.3g})")
    return source_processed, total_processed, time_s * 1000.0, relative_rms_error


def _bootstrap_feature_difference(left, right, repeats=1000, seed=20260927):
    rng = np.random.default_rng(seed)
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    observed = right.mean(axis=0) - left.mean(axis=0)
    boot = np.empty((repeats, observed.size), dtype=float)
    for index in range(repeats):
        li = rng.integers(0, len(left), len(left))
        ri = rng.integers(0, len(right), len(right))
        boot[index] = right[ri].mean(axis=0) - left[li].mean(axis=0)
    return observed, np.quantile(boot, .025, axis=0), np.quantile(boot, .975, axis=0)


def _write_csv(path, rows):
    import csv
    import json
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (list, tuple, dict, np.ndarray)) else value
                             for key, value in row.items()})


def _plot_population_differences(population_results, path):
    time = population_results[0]["time_ms"]
    keep = (time >= 0) & (time <= 800)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    colors = ("#3569a8", "#cf7844")
    for pop, ax in enumerate(axes):
        for state, array, linestyle in (("E", "excitatory", "-"),
                                        ("I", "inhibitory", "--")):
            per_fold = np.asarray([row[array][pop][:, keep] for row in population_results])
            mean = per_fold.mean(axis=0)
            for channel in range(2):
                ax.plot(time[keep], mean[channel], color=colors[channel],
                        linestyle=linestyle, lw=1.5,
                        label=f"{state}: {POPULATION_CHANNELS[pop][channel]}")
        ax.axhline(0, color="0.55", lw=.7)
        ax.axvline(0, color="0.4", lw=.8)
        ax.axvline(200, color="0.4", lw=.8, ls=":")
        ax.set_title(POPULATION_NAMES[pop].replace("_", " "))
        ax.set_ylabel("Right cue − left cue\n(activity units)")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel("Cue-relative time (ms); dotted line = cue offset at 200 ms")
    fig.suptitle("Cross-fitted model: population activity differences (same parameters per pair)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_source_differences(source_rows, path):
    time = np.asarray(sorted({row["time_ms"] for row in source_rows}), dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    palette = plt.get_cmap("tab10").colors
    for sensor, ax in enumerate(axes):
        for source_i, source_name in enumerate(SOURCE_NAMES):
            values_by_record = []
            for record in sorted({row["heldout_record"] for row in source_rows}):
                selected = [row for row in source_rows
                            if row["heldout_record"] == record
                            and row["sensor"] == ELECTRODES[sensor]
                            and row["source"] == source_name]
                lookup = {row["time_ms"]: row["right_minus_left"] for row in selected}
                values_by_record.append([lookup.get(float(t), np.nan) for t in time])
            arr = np.asarray(values_by_record, dtype=float)
            ax.plot(time, np.nanmean(arr, axis=0), color=palette[source_i], lw=1.1,
                    label=source_name.replace("_", " "))
        totals = []
        for record in sorted({row["heldout_record"] for row in source_rows}):
            selected = [row for row in source_rows if row["heldout_record"] == record
                        and row["sensor"] == ELECTRODES[sensor] and row["source"] == "TOTAL"]
            lookup = {row["time_ms"]: row["right_minus_left"] for row in selected}
            totals.append([lookup.get(float(t), np.nan) for t in time])
        ax.plot(time, np.nanmean(np.asarray(totals), axis=0), color="#222222",
                lw=1.8, label="sum / total")
        ax.axhline(0, color="0.6", lw=.7)
        ax.axvline(0, color="0.4", lw=.8)
        ax.axvline(200, color="0.4", lw=.8, ls=":")
        ax.set_ylabel(f"{ELECTRODES[sensor]}\nrelative units")
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015),
               ncol=3, frameon=False, fontsize=7.5)
    axes[-1].set_xlabel("Cue-relative time (ms); right minus left")
    fig.suptitle("Cross-fitted source contributions to electrode difference waveforms")
    fig.subplots_adjust(left=.10, right=.98, top=.92, bottom=.18, hspace=.20)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_empirical_features(case_features, model_feature_rows, path):
    records = sorted(case_features)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(CANDIDATE_FEATURE_NAMES))
    model_by_record = {row["record"]: row for row in model_feature_rows}
    for ax, record in zip(axes.flat, records):
        left, right = case_features[record]["left"], case_features[record]["right"]
        estimate, low, high = _bootstrap_feature_difference(left, right,
                                                            seed=20260927 + records.index(record))
        yerr = np.vstack((estimate - low, high - estimate))
        ax.errorbar(x, estimate, yerr=yerr, fmt="o", color="#386cb0", capsize=2,
                    label="Measured ERP: right − left (trial bootstrap 95% CI)")
        model = np.asarray([
            model_by_record[record][f"predicted_right_minus_left_{name}"]
            for name in CANDIDATE_FEATURE_NAMES
        ], dtype=float)
        ax.scatter(x, model, marker="D", s=28, color="#c54b45",
                   label="Cross-fitted model prediction", zorder=3)
        ax.axhline(0, color="0.5", lw=.7)
        ax.set_title(record)
        ax.set_xticks(x, [name.replace("_", "\n") for name in CANDIDATE_FEATURE_NAMES],
                      rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Window mean (processed EEG units)")
        ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Fixed-window measured feature differences and out-of-record model predictions\n"
                 "CIs resample trials within each recording; they are not participant-level intervals")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heldout_classifier(folds, path):
    complete = [row for row in folds if row.get("status") == "complete"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    labels = [row["heldout_record"].replace("VisualCog", "") for row in complete]
    values = [row["balanced_accuracy"] for row in complete]
    bars = ax.bar(labels, values, color="#4c88a8", width=.68)
    ax.axhline(.5, color="#a13d3d", ls="--", lw=1.1, label="chance reference")
    ax.axhline(np.mean(values), color="#333333", ls=":", lw=1.1,
               label=f"record-macro mean = {np.mean(values):.3f}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .015, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(1.0, max(values, default=.5) + .12))
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Nine fixed electrode × time-window means; leave-one-MAT-record-out")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_evidence_chain(out, cases, fits, resolution, drive_scales):
    """Generate model-difference, empirical-feature, and held-out validation outputs."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for case in cases:
        if case["stage"] == "Stage1" and case["condition"] in ("left", "right"):
            grouped.setdefault(case["dataset"], {})[case["condition"]] = case

    population_results, source_rows, model_feature_rows = [], [], []
    source_sum_audit = []
    empirical_features = {}
    prediction_rows = []
    for fold_index, (record, fit) in enumerate(sorted(fits.items())):
        if record not in grouped or set(grouped[record]) != {"left", "right"}:
            raise ValueError(f"missing left/right Stage1 cases for {record}")
        if record in fit.train_records:
            raise AssertionError(f"cross-fit model for {record} used the held-out record")
        params = ModelParams.from_any(fit.parameters)
        left_front = _cached_left_frontend(float(params.tau_a), int(resolution))
        right_front = mirror_stage1_frontend(left_front, load_stimulus("Stage1", "right"))
        left_result = simulate_forward(left_front, params=params, amplitude=fit.amplitude,
                                       drive_scales=drive_scales)
        right_result = simulate_forward(right_front, params=params, amplitude=fit.amplitude,
                                        drive_scales=drive_scales)
        if not np.array_equal(left_result.time_ms, right_result.time_ms):
            raise AssertionError("left/right model pair must use the same time grid")
        for state_name, left_state, right_state in (
                ("excitatory", left_result.excitatory, right_result.excitatory),
                ("inhibitory", left_result.inhibitory, right_result.inhibitory)):
            difference = right_state - left_state
            for pop, pop_name in enumerate(POPULATION_NAMES):
                for channel, channel_name in enumerate(POPULATION_CHANNELS[pop]):
                    for time_i, time in enumerate(left_result.time_ms):
                        if 0 <= time <= 800:
                            prediction_rows.append({
                                "heldout_record": record,
                                "train_records": ";".join(fit.train_records),
                                "population": pop_name,
                                "state": "E" if state_name == "excitatory" else "I",
                                "channel_semantics": channel_name,
                                "time_ms": float(time),
                                "right_minus_left_activity": float(difference[pop, channel, time_i]),
                                "model_parameters_shared_within_pair": True,
                            })
        population_results.append({
            "heldout_record": record,
            "time_ms": left_result.time_ms,
            "excitatory": right_result.excitatory - left_result.excitatory,
            "inhibitory": right_result.inhibitory - left_result.inhibitory,
        })

        left_sources, left_total, processed_time, left_error = _process_model_and_sources(
            left_result, fit.amplitude)
        right_sources, right_total, right_time, right_error = _process_model_and_sources(
            right_result, fit.amplitude)
        if not np.array_equal(processed_time, right_time):
            raise AssertionError("source and electrode observation grids differ")
        source_delta = right_sources - left_sources
        total_delta = right_total - left_total
        if not np.allclose(source_delta.sum(axis=1), total_delta, rtol=1e-7, atol=1e-8):
            raise AssertionError("right-minus-left source contributions do not sum to total EEG difference")
        source_sum_audit.append({"heldout_record": record,
                                 "left_source_sum_relative_rms_error": left_error,
                                 "right_source_sum_relative_rms_error": right_error,
                                 "max_relative_rms_error": max(left_error, right_error),
                                 "tolerance": 1e-5,
                                 "status": "pass_with_float32_roundoff"})
        time_keep = (processed_time >= 0) & (processed_time <= 800)
        for sensor, electrode in enumerate(ELECTRODES):
            for source_i, source_name in enumerate(SOURCE_NAMES):
                for time_i in np.flatnonzero(time_keep):
                    source_rows.append({
                        "heldout_record": record,
                        "train_records": ";".join(fit.train_records),
                        "sensor": electrode,
                        "source": source_name,
                        "time_ms": float(processed_time[time_i]),
                        "right_minus_left": float(source_delta[sensor, source_i, time_i]),
                    })
            for time_i in np.flatnonzero(time_keep):
                source_rows.append({
                    "heldout_record": record,
                    "train_records": ";".join(fit.train_records),
                    "sensor": electrode,
                    "source": "TOTAL",
                    "time_ms": float(processed_time[time_i]),
                    "right_minus_left": float(total_delta[sensor, time_i]),
                })

        model_condition_features = {}
        for condition, processed_total in (("left", left_total), ("right", right_total)):
            case = grouped[record][condition]
            prediction = np.vstack([np.interp(case["time_ms"], processed_time, row)
                                    for row in processed_total])
            model_condition_features[condition] = extract_electrode_window_features(
                prediction[None, :, :], case["time_ms"])[0]
            empirical_features.setdefault(record, {})[condition] = _case_features(case)
        model_delta = (model_condition_features["right"] - model_condition_features["left"])
        model_feature_rows.append({
            "record": record,
            "training_records": ";".join(fit.train_records),
            **{f"predicted_right_minus_left_{name}": float(value)
               for name, value in zip(CANDIDATE_FEATURE_NAMES, model_delta)},
        })

    score_rows, fold_rows, classifier_summary = classify_electrode_features(cases)
    empirical_rows = []
    for record in sorted(empirical_features):
        left = empirical_features[record]["left"]
        right = empirical_features[record]["right"]
        estimate, low, high = _bootstrap_feature_difference(left, right,
                                                            seed=20260927 + sorted(empirical_features).index(record))
        model_map = {row["record"]: row for row in model_feature_rows}[record]
        for index, name in enumerate(CANDIDATE_FEATURE_NAMES):
            empirical_rows.append({
                "record": record,
                "feature": name,
                "n_left_trials": int(len(left)),
                "n_right_trials": int(len(right)),
                "measured_right_minus_left": float(estimate[index]),
                "trial_bootstrap_95ci_low": float(low[index]),
                "trial_bootstrap_95ci_high": float(high[index]),
                "crossfit_model_right_minus_left": float(model_map[f"predicted_right_minus_left_{name}"]),
                "evidence_role": "within-record exploratory; window/electrode fixed before inspection",
            })

    source_summary_rows = []
    heldout_records = sorted({row["heldout_record"] for row in source_rows})
    for sensor in ELECTRODES:
        for source_name in (*SOURCE_NAMES, "TOTAL"):
            per_record_rms, relative_rms = [], []
            for record in heldout_records:
                selected = [row for row in source_rows
                            if row["heldout_record"] == record
                            and row["sensor"] == sensor and row["source"] == source_name]
                values = np.asarray([row["right_minus_left"] for row in selected], dtype=float)
                source_rms = float(np.sqrt(np.mean(values * values))) if values.size else float("nan")
                per_record_rms.append(source_rms)
                if source_name != "TOTAL":
                    total_rows = [row for row in source_rows
                                  if row["heldout_record"] == record
                                  and row["sensor"] == sensor and row["source"] == "TOTAL"]
                    total = np.asarray([row["right_minus_left"] for row in total_rows], dtype=float)
                    total_rms = float(np.sqrt(np.mean(total * total))) if total.size else float("nan")
                    relative_rms.append(source_rms / total_rms if total_rms > 1e-12 else float("nan"))
            source_summary_rows.append({
                "sensor": sensor,
                "source": source_name,
                "mean_record_rms": float(np.nanmean(per_record_rms)),
                "mean_record_rms_ratio_to_total": (float(np.nanmean(relative_rms))
                                                    if relative_rms else 1.0),
                "n_heldout_records": len(heldout_records),
                "interpretation": "RMS ratio is descriptive; signed sources can cancel, so ratios are not additive shares",
            })

    _write_csv(out / "model_population_right_minus_left.csv", prediction_rows)
    _write_csv(out / "model_source_electrode_right_minus_left.csv", source_rows)
    _write_csv(out / "model_candidate_feature_predictions.csv", model_feature_rows)
    _write_csv(out / "measured_candidate_features_exploratory.csv", empirical_rows)
    _write_csv(out / "source_additivity_audit.csv", source_sum_audit)
    _write_csv(out / "model_source_contribution_summary.csv", source_summary_rows)
    _write_csv(out / "heldout_9d_electrode_lda_trial_scores.csv", score_rows)
    _write_csv(out / "heldout_9d_electrode_lda_folds.csv", fold_rows)
    _write_csv(out / "heldout_9d_electrode_lda_summary.csv", [classifier_summary])

    _plot_population_differences(population_results,
                                 out / "model_population_right_minus_left.png")
    _plot_source_differences(source_rows,
                             out / "model_source_electrode_right_minus_left.png")
    _plot_empirical_features(empirical_features, model_feature_rows,
                             out / "measured_vs_model_candidate_features.png")
    _plot_heldout_classifier(fold_rows, out / "heldout_9d_electrode_lda.png")
    return {"classification_summary": classifier_summary,
            "classification_folds": fold_rows,
            "population_rows": len(prediction_rows),
            "source_rows": len(source_rows),
            "feature_rows": len(empirical_rows),
            "source_sum_identity": "verified after Q1 observation processing; residual compared with float32 EEG storage roundoff",
            "source_sum_max_relative_rms_error": max(row["max_relative_rms_error"]
                                                       for row in source_sum_audit),
            "source_contribution_summary": source_summary_rows,
            "parameter_identity": "left/right paired simulations share parameters, gain, input scaling and observation operator",
            "figures": ["model_population_right_minus_left.png",
                        "model_source_electrode_right_minus_left.png",
                        "measured_vs_model_candidate_features.png",
                        "heldout_9d_electrode_lda.png"]}
