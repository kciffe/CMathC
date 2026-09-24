"""Run the v2 Q2 triangle-cue model and independent real-EEG checks.

Primary analysis: signed contrast -> LGN ON/OFF -> Gabor/configuration
responses -> three fixed Wilson-Cowan population pairs -> three orthogonal
sensor-space modes -> F3/Fz/F4. The modes are functional coordinates, not
anatomical sources or a subject-specific lead field.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

try:
    from . import config
    from .evaluate import (classification_leave_one_record, erp_metrics,
                           run_forward_condition, split_half_stability,
                           within_record_permutation)
    from .fit import (_cached_left_frontend, _interpolate_prediction,
                      fit_leave_one_record, register_frontend)
    from .frontend import (choose_resolution, load_stimulus,
                           mirror_stage1_frontend, scale_kernel_audit,
                           temporal_stride_audit)
    from .model import ModelParams, observable_modes_from_real, simulate_forward
    from .real_data import load_cases_with_audit
except ImportError:
    import config
    from evaluate import (classification_leave_one_record, erp_metrics,
                          run_forward_condition, split_half_stability,
                          within_record_permutation)
    from fit import (_cached_left_frontend, _interpolate_prediction,
                     fit_leave_one_record, register_frontend)
    from frontend import (choose_resolution, load_stimulus, mirror_stage1_frontend,
                          scale_kernel_audit, temporal_stride_audit)
    from model import ModelParams, observable_modes_from_real, simulate_forward
    from real_data import load_cases_with_audit


V1_NPZ = config.Q2_ROOT / "output" / "revision_v1" / "model_outputs.npz"
MODEL_IDS = ("v1_fitted", "v2_opponent_rank3", "v2_opponent_rank2",
             "v2_legacy_rank3", "v2_legacy_rank2", "zero_predictor")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(value), ensure_ascii=False)
                             if isinstance(value, (dict, list, tuple, np.ndarray)) else value
                             for key, value in row.items()})


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_v1_predictions(cases):
    """Read frozen v1 held-out curves only; never refit or alter v1 files."""
    if not V1_NPZ.exists():
        return {}
    result = {}
    with np.load(V1_NPZ, allow_pickle=False) as data:
        for case in cases:
            key = f"fitted__{case['dataset']}__Stage1__{case['condition']}"
            if key not in data:
                continue
            time_key = f"time__real__{case['dataset']}__Stage1__{case['condition']}"
            if time_key not in data:
                continue
            prediction = np.asarray(data[key], dtype=float)
            time = np.asarray(data[time_key], dtype=float)
            if prediction.shape != (3, len(time)):
                continue
            result[(case["dataset"], case["condition"])] = {"eeg": prediction, "time_ms": time}
    return result


def _condition_audit(cases, event_rows):
    event_by_record = {row["dataset"]: row for row in event_rows}
    rows = []
    for case in cases:
        row = {key: case[key] for key in ("dataset", "task", "stage", "condition", "n_trials",
                                           "eligible_fit", "eligible_classification", "role",
                                           "eligibility_reason", "baseline_window_s", "late_window_ms")}
        row.update({"n_time_samples": len(case["time_ms"]),
                    "observed_first_ms": float(case["time_ms"][0]),
                    "observed_last_ms": float(case["time_ms"][-1])})
        row.update({f"event_{key}": value for key, value in event_by_record[case["dataset"]].items()
                    if key not in ("dataset", "task")})
        rows.append(row)
    return rows


def _fit_row(record, fit):
    return {"heldout_record": record, "train_records": fit.train_records,
            "feature_route": fit.feature_route, "observation_rank": fit.observation_rank,
            "resolution": fit.resolution, "tau_s_ms": fit.parameters["tau_s"],
            "g_i": fit.parameters["g_i"], "tau_a_ms": fit.parameters["tau_a"],
            "shared_gain_signed": fit.amplitude, "training_loss": fit.loss,
            "optimizer_success": fit.success, "fit_status": fit.status,
            "boundary_flags": fit.boundary_flags, "objective_evaluations": fit.n_evaluations}


def _model_predictions(cases, fits, resolution, v1_predictions, progress=True):
    predictions = {}
    metrics = []
    cases_by_record = {}
    for case in cases:
        if case["stage"] == "Stage1" and case["condition"] in ("left", "right"):
            cases_by_record.setdefault(case["dataset"], {})[case["condition"]] = case
    for record, fit in fits.items():
        if progress:
            print(f"  generating held-out predictions: {record}", flush=True)
        params = ModelParams.from_any(fit.parameters)
        ablation_specs = (("v2_opponent_rank3", "opponent", 3),
                          ("v2_opponent_rank2", "opponent", 2),
                          ("v2_legacy_rank3", "legacy", 3),
                          ("v2_legacy_rank2", "legacy", 2))
        for condition, case in cases_by_record[record].items():
            key = (record, condition)
            if key in v1_predictions:
                old = v1_predictions[key]
                predictions[(record, condition, "v1_fitted")] = _interpolate_prediction(
                    old["eeg"], old["time_ms"], case["time_ms"])
            for model_id, route, rank in ablation_specs:
                result = run_forward_condition("Stage1", condition, params,
                    resolution=resolution, amplitude=fit.amplitude,
                    feature_route=route, observation_rank=rank)
                predictions[(record, condition, model_id)] = _interpolate_prediction(
                    result.eeg_scaled, result.time_ms, case["time_ms"])
            predictions[(record, condition, "zero_predictor")] = np.zeros_like(case["real"])
            for model_id in MODEL_IDS:
                pred = predictions.get((record, condition, model_id))
                if pred is None:
                    continue
                metric = erp_metrics(case["real"], pred, case["time_ms"], "Stage1")
                metrics.append({"record": record, "task": case["task"], "condition": condition,
                                "n_trials": case["n_trials"], "model": model_id,
                                "fit_status": fit.status if model_id != "v1_fitted" else "frozen_v1_heldout",
                                "full_sensor_nrmse": metric["full_sensor_nrmse_by_real_rms"],
                                "normalized_mse_skill_vs_zero": 1.0 - metric["full_sensor_nrmse_by_real_rms"] ** 2,
                                "channel_corr_mean": metric["channel_corr_mean"],
                                "channel_correlations": metric["channel_correlations"],
                                "mode_corr_mean": metric["mode_corr_mean"],
                                "mode_correlations": metric["mode_correlations"],
                                "real_modes_rms": metric["real_modes_rms"],
                                "predicted_modes_rms": metric["predicted_modes_rms"],
                                "late_peaks": metric["late_peaks"],
                                "ablation_parameters_refit": model_id == "v2_opponent_rank3",
                                "amplitude_shared_across_conditions": True})
    return predictions, metrics, cases_by_record


def _difference_metrics(cases_by_record, predictions):
    rows = []
    for record, conditions in cases_by_record.items():
        if set(conditions) != {"left", "right"}:
            continue
        left, right = conditions["left"], conditions["right"]
        time = left["time_ms"]
        if not np.array_equal(time, right["time_ms"]):
            raise ValueError(f"{record}: left/right time grids differ")
        real = right["real"] - left["real"]
        real_modes = observable_modes_from_real(real)
        real_rms = float(np.sqrt(np.mean(real * real)))
        for model_id in MODEL_IDS:
            lpred = predictions.get((record, "left", model_id))
            rpred = predictions.get((record, "right", model_id))
            if lpred is None or rpred is None:
                continue
            pred = rpred - lpred
            pred_modes = observable_modes_from_real(pred)
            corr = (float(np.corrcoef(real.ravel(), pred.ravel())[0, 1])
                    if np.std(real) > 1e-12 and np.std(pred) > 1e-12 else float("nan"))
            rows.append({"record": record, "task": left["task"], "model": model_id,
                         "real_difference_rms": real_rms,
                         "predicted_difference_rms": float(np.sqrt(np.mean(pred * pred))),
                         "difference_rms_ratio": float(np.sqrt(np.mean(pred * pred)) / real_rms)
                             if real_rms > 1e-12 else float("nan"),
                         "difference_nrmse": float(np.sqrt(np.mean((real - pred) ** 2)) / real_rms)
                             if real_rms > 1e-12 else float("nan"),
                         "sensor_difference_correlation": corr,
                         "real_mode_difference_rms": np.sqrt(np.mean(real_modes ** 2, axis=1)).tolist(),
                         "predicted_mode_difference_rms": np.sqrt(np.mean(pred_modes ** 2, axis=1)).tolist(),
                         "u1_difference_correlation": (float(np.corrcoef(real_modes[1], pred_modes[1])[0, 1])
                             if np.std(real_modes[1]) > 1e-12 and np.std(pred_modes[1]) > 1e-12 else float("nan"))})
    return rows


def _plot_template(front, out_path):
    right_stim = load_stimulus("Stage1", "right")
    right = mirror_stage1_frontend(front, right_stim)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, label, result in ((axes[0], "Left cue", front), (axes[1], "Right cue", right)):
        ax.plot(result["time_ms"], result["R_L"], label="left-template response")
        ax.plot(result["time_ms"], result["R_R"], label="right-template response")
        ax.axvline(0, color="0.4", lw=.8)
        ax.axvline(200, color="0.4", lw=.8, ls="--")
        ax.set(title=label, xlabel="Cue-relative time (ms)", ylabel="Template response (relative)")
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Fixed left/right shape-template opponent responses")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _aligned_curves(cases_by_record, predictions, condition, model_id, grid):
    values = []
    for record, conds in cases_by_record.items():
        case = conds.get(condition)
        if case is None:
            continue
        if model_id == "real":
            curve = case["real"]
        else:
            curve = predictions.get((record, condition, model_id))
            if curve is None:
                continue
        values.append(np.vstack([np.interp(grid, case["time_ms"], row) for row in curve]))
    return np.asarray(values)


def _plot_erp(cases_by_record, predictions, out_path):
    reference = next(iter(next(iter(cases_by_record.values())).values()))
    grid = reference["time_ms"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    colors = {"real": "black", "v1_fitted": "#5B6F9C", "v2_opponent_rank3": "#C24E42"}
    names = {"real": "Measured ERP", "v1_fitted": "v1 held-out", "v2_opponent_rank3": "v2 opponent, rank 3"}
    for col, condition in enumerate(("left", "right")):
        for row, channel in enumerate(config.CHANNELS):
            ax = axes[row, col]
            for method in colors:
                arr = _aligned_curves(cases_by_record, predictions, condition, method, grid)
                if not arr.size:
                    continue
                mean = arr[:, row].mean(axis=0)
                spread = arr[:, row].std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)
                ax.plot(grid, mean, color=colors[method], label=names[method], lw=1.7)
                if len(arr) > 1:
                    ax.fill_between(grid, mean - spread, mean + spread,
                                    color=colors[method], alpha=.12, linewidth=0)
            ax.axvline(0, color="0.5", lw=.8)
            ax.axvline(200, color="0.5", lw=.8, ls="--")
            ax.axhline(0, color="0.7", lw=.7)
            ax.set_title(f"{condition.capitalize()} cue — {channel}")
            ax.grid(alpha=.2)
            if row == 2:
                ax.set_xlabel("Cue-relative time (ms)")
            if col == 0:
                ax.set_ylabel("EEG (input units)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out recording ERPs and frozen forward-model predictions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_difference(cases_by_record, predictions, out_path):
    reference = next(iter(next(iter(cases_by_record.values())).values()))
    grid = reference["time_ms"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    methods = (("real", "Measured"), ("v1_fitted", "v1 held-out"),
               ("v2_opponent_rank3", "v2 opponent, rank 3"))
    palette = {"real": "black", "v1_fitted": "#5B6F9C", "v2_opponent_rank3": "#C24E42"}
    labels = ("u0 common mode", "u1 lateral opponent mode", "u2 common shape mode")
    for mode, ax in enumerate(axes):
        for method, label in methods:
            traces = []
            for record, conditions in cases_by_record.items():
                if not {"left", "right"} <= set(conditions):
                    continue
                if method == "real":
                    diff = conditions["right"]["real"] - conditions["left"]["real"]
                else:
                    lpred = predictions.get((record, "left", method))
                    rpred = predictions.get((record, "right", method))
                    if lpred is None or rpred is None:
                        continue
                    diff = rpred - lpred
                modes = observable_modes_from_real(diff)
                traces.append(np.interp(grid, conditions["left"]["time_ms"], modes[mode]))
            if traces:
                arr = np.asarray(traces)
                mean = arr.mean(axis=0)
                spread = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)
                ax.plot(grid, mean, color=palette[method], label=label)
                if len(arr) > 1:
                    ax.fill_between(grid, mean - spread, mean + spread,
                                    color=palette[method], alpha=.12, linewidth=0)
        ax.axvline(0, color="0.5", lw=.8)
        ax.axvline(200, color="0.5", lw=.8, ls="--")
        ax.axhline(0, color="0.7", lw=.7)
        ax.set_ylabel(labels[mode])
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Cue-relative time (ms)")
    fig.suptitle("Measured and predicted right-minus-left difference modes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_separability(loo_rows, split_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    records = sorted({row["heldout_record"] for row in loo_rows})
    x = np.arange(len(records))
    width = .35
    for offset, mode_count, color in ((-width / 2, 2, "#6686A8"),
                                      (width / 2, 3, "#D27C52")):
        rows = {row["heldout_record"]: row for row in loo_rows if row["mode_count"] == mode_count}
        axes[0].bar(x + offset, [rows[r].get("balanced_accuracy", np.nan) for r in records],
                    width, label=f"{mode_count * 3} features", color=color)
    axes[0].axhline(.5, color="0.35", ls="--", lw=1, label="chance")
    axes[0].set_xticks(x, [r.replace("VisualCog", "") for r in records], rotation=25, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Leave-one-record-out LDA")
    axes[0].legend(frameon=False)
    modes = ("u0_common", "u1_lateral", "u2_shape")
    for j, mode in enumerate(modes):
        rows = [r for r in split_rows if r["mode"] == mode]
        ys = [r["split_half_correlation_median"] for r in rows]
        lo = [r["split_half_correlation_median"] - r["split_half_correlation_q25"] for r in rows]
        hi = [r["split_half_correlation_q75"] - r["split_half_correlation_median"] for r in rows]
        axes[1].errorbar(np.full(len(ys), j) + np.linspace(-.08, .08, max(1, len(ys))), ys,
                         yerr=np.vstack([lo, hi]), fmt="o", capsize=3, label=mode)
    axes[1].axhline(0, color="0.35", lw=.8)
    axes[1].set_xticks(range(3), ("u0 common", "u1 lateral", "u2 shape"))
    axes[1].set_ylabel("Split-half correlation")
    axes[1].set_title("Within-record difference-wave stability")
    axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _run_real_data_evaluation(cases, out):
    loo_rows, score_rows, summary_rows = [], [], []
    for mode_count in (2, 3):
        scores, folds, summary = classification_leave_one_record(cases, mode_count=mode_count)
        score_rows.extend(scores)
        loo_rows.extend(folds)
        summary_rows.append(summary)
    stability = split_half_stability(cases)
    permutations = within_record_permutation(cases)
    _write_csv(out / "real_eeg_LO_record_LDA.csv", loo_rows)
    _write_csv(out / "real_eeg_trial_scores.csv", score_rows)
    _write_csv(out / "real_eeg_LDA_summary.csv", summary_rows)
    _write_csv(out / "split_half_stability.csv", stability)
    _write_csv(out / "within_record_permutation.csv", permutations)
    _plot_separability(loo_rows, stability, out / "real_eeg_separability.png")
    return {"loo_rows": loo_rows, "score_rows": score_rows, "summary_rows": summary_rows,
            "stability": stability, "permutations": permutations}


def run_revision(mode="full", resolution=None, max_nfev=35, skip_resolution_audit=False,
                 progress=True):
    """Run real-data separability checks, forward fitting, controls, and plots."""
    out = config.OUTPUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    cases, event_audit = load_cases_with_audit()
    condition_audit = _condition_audit(cases, event_audit)
    _write_csv(out / "condition_audit.csv", condition_audit)
    _write_csv(out / "event_audit.csv", event_audit)

    data_results = _run_real_data_evaluation(cases, out)
    if mode == "data-only":
        print(f"Data-only evaluation written to: {out}", flush=True)
        return data_results

    kernel_rows = scale_kernel_audit()
    if resolution is None:
        if skip_resolution_audit:
            resolution = 64
            resolution_audit = {"chosen_resolution": 64,
                                "selection_rule": "fixed to v1 grid; feature-scale audit skipped by request"}
        else:
            if progress:
                print("Auditing 64/128 spatial feature scales...", flush=True)
            resolution_audit = choose_resolution()
            resolution = resolution_audit["chosen_resolution"]
    else:
        resolution_audit = {"chosen_resolution": int(resolution),
                            "selection_rule": "resolution fixed by CLI"}
    _write_csv(out / "spatial_kernel_audit.csv", kernel_rows)
    _write_json(out / "spatial_resolution_audit.json", resolution_audit)
    if not resolution_audit.get("resolution_pass", True):
        raise RuntimeError("no spatial resolution passed the fixed kernel audit")

    # Select the coarsest deterministic feature stride that stays within a
    # fixed numerical tolerance against a 1-ms feature and model reference.
    stride_audit_rows = []
    selected_frontend = None
    selected_stride = None
    for stride in (config.FRONTEND_FEATURE_STRIDE_MS, 2.0, 1.0):
        if progress:
            print(f"Auditing {stride:g}-ms frontend feature stride...", flush=True)
        feature_rows, coarse, fine = temporal_stride_audit(
            tau_a=config.PARAM_DEFAULTS["tau_a"], resolution=resolution, stride_ms=stride)
        p = ModelParams()
        coarse_model = simulate_forward(coarse, p, feature_route="opponent", observation_rank=3)
        fine_model = simulate_forward(fine, p, feature_route="opponent", observation_rank=3)
        model_rms = float(np.sqrt(np.mean(fine_model.eeg ** 2)))
        model_error = float(np.sqrt(np.mean((coarse_model.eeg - fine_model.eeg) ** 2))
                            / model_rms) if model_rms > 1e-12 else 0.0
        for row in feature_rows:
            row.update({"model_relative_eeg_rmse": model_error,
                        "model_reference_stride_ms": 1.0})
            stride_audit_rows.append(row)
        critical = [row["relative_rmse_to_1ms"] for row in feature_rows
                    if row["feature"] in ("gabor", "B", "shape", "R_L", "R_R")]
        passed = bool(max(critical, default=0.0) <= .05 and model_error <= .05)
        for row in feature_rows:
            row["stride_pass"] = passed
        if passed:
            selected_frontend, selected_stride = coarse, float(stride)
            break
    if selected_frontend is None:
        raise RuntimeError("1-ms frontend reference failed its own numerical check")
    config.FRONTEND_FEATURE_STRIDE_MS = selected_stride
    register_frontend(config.PARAM_DEFAULTS["tau_a"], resolution, selected_frontend)
    _write_csv(out / "frontend_temporal_stride_audit.csv", stride_audit_rows)
    if progress:
        print(f"Selected frontend feature stride: {selected_stride:g} ms", flush=True)

    if progress:
        print(f"Fitting v2 opponent/rank-{config.OBSERVATION_RANK} held-out models "
              f"on {resolution} grid (max_nfev={max_nfev})...", flush=True)
    fits = fit_leave_one_record(cases, resolution=resolution,
                                feature_route="opponent", observation_rank=3,
                                progress=progress, max_nfev=max_nfev)
    fit_rows = [_fit_row(record, fit) for record, fit in fits.items()]
    _write_csv(out / "heldout_fit_summary.csv", fit_rows)
    _write_json(out / "heldout_fit_details.json",
                {record: fit.to_dict() for record, fit in fits.items()})

    v1 = _load_v1_predictions(cases)
    if progress:
        print(f"Loaded {len(v1)} frozen v1 held-out ERP curves for comparison.", flush=True)
    predictions, metric_rows, cases_by_record = _model_predictions(
        cases, fits, resolution, v1, progress=progress)
    difference_rows = _difference_metrics(cases_by_record, predictions)
    _write_csv(out / "heldout_forward_metrics.csv", metric_rows)
    _write_csv(out / "heldout_left_right_difference.csv", difference_rows)

    # Use the profile point already computed in the training cache; it is only
    # a visualization of the deterministic front-end, not an EEG fit.
    template_front = _cached_left_frontend(float(config.PARAM_DEFAULTS["tau_a"]), int(resolution))
    _plot_template(template_front, out / "shape_opponent_template_responses.png")
    _plot_erp(cases_by_record, predictions, out / "heldout_ERP_comparison.png")
    _plot_difference(cases_by_record, predictions, out / "heldout_left_right_difference.png")

    source_files = [Path(__file__), Path(__file__).with_name("config.py"),
                    Path(__file__).with_name("frontend.py"), Path(__file__).with_name("model.py"),
                    Path(__file__).with_name("fit.py"), Path(__file__).with_name("evaluate.py"),
                    Path(__file__).with_name("real_data.py")]
    manifest = {
        "revision": "revision_v2",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "resolution": int(resolution),
                    "feature_stride_ms": config.FRONTEND_FEATURE_STRIDE_MS,
                    "fit_max_nfev_per_start": int(max_nfev)},
        "data": {"real_root": str(config.REAL_ROOT), "record_count": len(event_audit),
                 "case_count": len(cases), "baseline": "per-trial mean subtraction from the fixed prestimulus interval",
                 "epoch_sample_grid": "measured 128-Hz samples; predictions linearly sampled to each recording's grid"},
        "primary_model": {"route": "signed contrast -> center-surround LGN ON/OFF -> Gabor -> common shape + rectified orientation opponents -> three Wilson-Cowan population pairs",
                          "observation": "fixed orthonormal u0/u1/u2 sensor coordinates mapped to F3/Fz/F4",
                          "free_observation_weights": False,
                          "fit_parameters": ["tau_s", "g_i", "tau_a", "one shared signed gain"],
                          "fit_scope": "Stage1 left/right cue ERPs; leave-one-MAT-out; training records only",
                          "output_units": "relative model units; not calibrated microvolts"},
        "real_data_validation": {"classifier": "real-only shrinkage LDA on 6 and 9 fixed trial features, leave-one-MAT-out",
                                 "within_record_permutation": config.PERMUTATION_REPEATS,
                                 "split_half_repeats": config.SPLIT_HALF_REPEATS,
                                 "interpretation": "separability is not evidence by itself that the neural forward mechanism is correct"},
        "ablation_policy": "legacy route and rank-2 output use primary fold parameters and signed amplitude without refitting",
        "v1_comparison_available": len(v1),
        "limitations": ["u modes are fixed sensor-space coordinates, not anatomical sources",
                        "no individualized head model or calibrated conductivity/volume conduction",
                        "only Stage1 left/right cues are fitted; unknown Stage2 layouts are not relabeled",
                        "P300 is not directly fitted or claimed", "shared gain has no direct microvolt interpretation"],
        "source_sha256": {str(path.name): _sha256(path) for path in source_files if path.exists()},
    }
    _write_json(out / "manifest.json", manifest)
    _print_summary(fit_rows, metric_rows, difference_rows, data_results["summary_rows"],
                   data_results["permutations"], out)
    return {"fits": fit_rows, "metrics": metric_rows, "differences": difference_rows,
            "data": data_results, "output": out}


def _print_summary(fits, metrics, differences, lda_summary, permutations, out):
    print("\n=== revision_v2 evaluation ===", flush=True)
    for row in lda_summary:
        print(f"LO-record LDA {row['mode_count'] * 3} features: BA="
              f"{row['balanced_accuracy']:.3f}, AUC={row['roc_auc']:.3f} over "
              f"{row['n_heldout_records']} records", flush=True)
    for row in permutations:
        print(f"{row['record']}: within-record 9D permutation "
              f"p={row['permutation_p_9d_primary']:.4f}, "
              f"Holm p={row['permutation_p_9d_holm']:.4f}", flush=True)
    for model in MODEL_IDS:
        selected = [r for r in metrics if r["model"] == model]
        if selected:
            print(f"{model}: mean held-out ERP NRMSE="
                  f"{np.mean([r['full_sensor_nrmse'] for r in selected]):.3f}", flush=True)
    selected_diff = [r for r in differences if r["model"] == "v2_opponent_rank3"]
    if selected_diff:
        ratios = [r["difference_rms_ratio"] for r in selected_diff]
        print(f"v2 right-left difference RMS ratio median={np.median(ratios):.3f}", flush=True)
    print(f"Fit folds={len(fits)}; artifacts: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "data-only"), default="full")
    parser.add_argument("--resolution", type=int, choices=(64, 128), default=None)
    parser.add_argument("--max-nfev", type=int, default=35)
    parser.add_argument("--skip-resolution-audit", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    run_revision(mode=args.mode, resolution=args.resolution, max_nfev=args.max_nfev,
                 skip_resolution_audit=args.skip_resolution_audit,
                 progress=not args.quiet_progress)


if __name__ == "__main__":
    main()
