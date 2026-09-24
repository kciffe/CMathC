"""Single entry point for the independent Q2 revision_v1 forward model."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from . import config
    from .evaluate import (classification_leave_one_record, erp_metrics,
                           estimate_shared_amplitude, mechanism_control_rows)
    from .fit import _interpolate_prediction, fit_model
    from .frontend import choose_resolution, load_stimulus, simulate_frontend, template_manifest
    from .model import ModelParams, simulate_forward
    from .real_data import load_cases_with_audit
except ImportError:
    import config
    from evaluate import (classification_leave_one_record, erp_metrics,
                          estimate_shared_amplitude, mechanism_control_rows)
    from fit import _interpolate_prediction, fit_model
    from frontend import choose_resolution, load_stimulus, simulate_frontend, template_manifest
    from model import ModelParams, simulate_forward
    from real_data import load_cases_with_audit

LEGACY_BASELINE = config.Q2_ROOT / "output" / "04_simulated_eeg"
LEGACY_SCRIPTS = ("01_gabor_frontend.py", "02_lgn.py", "03_cortex.py",
                  "03_2_cortex_diagnosis.py", "04_simulated_eeg.py",
                  "04_2_mapping_diagnosis.py", "05_simulated_real_eeg_compare.py",
                  "05_2_fullcurve_calibration.py")


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


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
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2,
                               default=_json_default, allow_nan=False), encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_json_safe(v), ensure_ascii=False, default=_json_default,
                                           allow_nan=False)
                             if isinstance(v, (dict, list, tuple, np.ndarray)) else v
                             for k, v in row.items()})


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _legacy_hashes():
    paths = [config.Q2_ROOT / name for name in LEGACY_SCRIPTS]
    if LEGACY_BASELINE.exists():
        paths.extend(p for p in LEGACY_BASELINE.rglob("*") if p.is_file())
    return {str(p.relative_to(config.Q2_ROOT)): _hash(p) for p in paths if p.exists()}


def _source_hashes():
    paths = sorted(config.INPUT_ROOT.glob("*.npy")) + sorted(Path(__file__).parent.glob("*.py"))
    paths += sorted(config.REAL_ROOT.glob("*_clean.mat"))
    result = {}
    for path in paths:
        try:
            label = str(path.relative_to(config.Q2_ROOT))
        except ValueError:
            label = str(path.relative_to(config.Q2_ROOT.parent))
        result[label] = _hash(path)
    return result


def _condition_audit(cases, event_rows):
    event_map = {r["dataset"]: r for r in event_rows}
    rows = []
    for case in cases:
        row = {k: case[k] for k in ("dataset", "task", "stage", "condition", "n_trials",
                                    "eligible", "eligible_fit", "eligible_classification",
                                    "role", "eligibility_reason", "baseline_window_s", "late_window_ms")}
        row.update({"n_time_samples": len(case["time_ms"]),
                    "observed_first_ms": float(case["time_ms"][0]),
                    "observed_last_ms": float(case["time_ms"][-1])})
        row.update({f"event_{k}": v for k, v in event_map[case["dataset"]].items()
                    if k not in ("dataset", "task")})
        rows.append(row)
    return rows


def _load_old(task, condition):
    path = LEGACY_BASELINE / task / "Stage1" / f"{condition}_eeg.npy"
    if not path.exists():
        return None
    eeg = np.asarray(np.load(path), dtype=float)
    if eeg.shape == (801, 3):
        eeg = eeg.T
    if eeg.ndim != 2 or eeg.shape[0] != 3:
        return None
    return {"eeg": eeg, "time_ms": np.linspace(0.0, 800.0, eeg.shape[1]), "path": str(path)}


def _canonical_models(resolution):
    models = {}
    params = ModelParams()
    for stage, condition in config.STIMULI:
        front = simulate_frontend(load_stimulus(stage, condition),
                                  params={"tau_a": params.tau_a}, resolution=resolution)
        models[(stage, condition)] = {"frontend": front, "model": simulate_forward(front, params)}
    return models


def _dt_convergence_check(resolution, canonical):
    """Compare default Stage1 predictions at 1 ms and 0.5 ms integration steps."""
    rows = []
    time_half = np.arange(0.0, 800.0001, 0.5)
    for condition in ("left", "right"):
        stimulus = load_stimulus("Stage1", condition)
        front = simulate_frontend(stimulus, params={"tau_a": config.PARAM_DEFAULTS["tau_a"]},
                                  time_ms=time_half, resolution=resolution)
        fine = simulate_forward(front, params=ModelParams())
        coarse = canonical[("Stage1", condition)]["model"]
        reference = coarse.eeg
        downsampled = fine.eeg[:, ::2]
        difference = downsampled - reference
        rows.append({"stage": "Stage1", "condition": condition, "dt_coarse_ms": 1.0,
                     "dt_fine_ms": 0.5, "relative_eeg_rms_difference": float(
                         np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-12)),
                     "maximum_absolute_difference": float(np.max(np.abs(difference)))})
    return rows


def _models_for_parameters(conditions, parameters, resolution):
    params = ModelParams.from_any(parameters)
    models = {}
    for stage, condition in conditions:
        front = simulate_frontend(load_stimulus(stage, condition),
                                  params={"tau_a": params.tau_a}, resolution=resolution)
        models[(stage, condition)] = {"frontend": front,
                                      "model": simulate_forward(front, params=params)}
    return models


def _metric_row(case, method, prediction, source_time, amp=1.0, params=None, loss=None):
    curve = _interpolate_prediction(amp * prediction, source_time, case["time_ms"])
    metrics = erp_metrics(case["real"], curve, case["time_ms"], case["stage"])
    u2 = config.U2 @ np.asarray(case["real"])
    late_mask = ((case["time_ms"] >= config.STAGES[case["stage"]]["late"][0])
                 & (case["time_ms"] <= config.STAGES[case["stage"]]["late"][1]))
    row = {"heldout_record": case["dataset"], "task": case["task"], "stage": case["stage"],
           "condition": case["condition"], "method": method, "n_trials": case["n_trials"],
           "amplitude": float(amp), "rmse": metrics["rmse"],
           "nrmse_by_real_rms": metrics["nrmse_by_real_rms"],
           "channel_corr_mean": metrics["channel_corr_mean"],
           "channel_correlations": metrics["channel_correlations"],
           "real_u2_rms": float(np.sqrt(np.mean(u2 * u2))),
           "real_u2_late_rms": float(np.sqrt(np.mean(u2[late_mask] ** 2))) if late_mask.any() else float("nan")}
    if params:
        row.update({f"param_{k}": v for k, v in params.items()})
    if loss is not None:
        row["training_loss"] = float(loss)
    for peak in metrics["late_peaks"]:
        ch = peak["channel"]
        row[f"{ch}_late_peak_amplitude"] = peak["peak_amplitude"]
        row[f"{ch}_late_peak_time_ms"] = peak["peak_time_ms"]
        row[f"{ch}_late_peak_status"] = peak["peak_status"]
    return row, curve


def _plot_mechanism(canonical, out):
    left, right = canonical[("Stage1", "left")], canonical[("Stage1", "right")]
    fl, fr, ml = left["frontend"], right["frontend"], left["model"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax[0, 0].plot(fl["time_ms"], fl["lgn_on_mean"], label="ON", color="#168AAD")
    ax[0, 0].plot(fl["time_ms"], fl["lgn_off_mean"], label="OFF", color="#D17A22")
    ax[0, 0].axvline(200, color="0.5", linestyle=":")
    ax[0, 0].set(title="LGN onset and offset response", xlabel="Time from cue onset (ms)", ylabel="Mean rate")
    ax[0, 0].legend(frameon=False)
    for oi, angle in enumerate(config.GABOR_ANGLES_DEG):
        ax[0, 1].plot(fl["time_ms"], fl["gabor"][oi].mean(axis=(0, 1)), label=f"{angle:g}° left")
        ax[0, 1].plot(fr["time_ms"], fr["gabor"][oi].mean(axis=(0, 1)), linestyle="--",
                      label=f"{angle:g}° right")
    ax[0, 1].set(title="V1 direction channels", xlabel="Time (ms)", ylabel="Gabor energy")
    ax[0, 1].legend(frameon=False, fontsize=7, ncol=2)
    for name, color in (("B", "0.35"), ("H_L", "#168AAD"), ("H_R", "#D17A22")):
        ax[1, 0].plot(fl["time_ms"], fl[name].mean(axis=(0, 1)), color=color, label=f"Left / {name}")
    ax[1, 0].plot(fr["time_ms"], fr["H_L"].mean(axis=(0, 1)), "--", color="#168AAD", label="Right / H_L")
    ax[1, 0].plot(fr["time_ms"], fr["H_R"].mean(axis=(0, 1)), "--", color="#D17A22", label="Right / H_R")
    ax[1, 0].set(title="Edge and shape configuration", xlabel="Time (ms)", ylabel="Feature response")
    ax[1, 0].legend(frameon=False, fontsize=7, ncol=2)
    for k, name, color in ((0, "B", "0.35"), (1, "H_L", "#168AAD"), (2, "H_R", "#D17A22")):
        ax[1, 1].plot(ml.time_ms, ml.excitatory[k].mean(axis=0) - ml.inhibitory[k].mean(axis=0),
                      color=color, label=f"{name}: E-I")
    ax[1, 1].set(title="Six low-dimensional WC populations", xlabel="Time (ms)", ylabel="E-I (relative)")
    ax[1, 1].legend(frameon=False)
    for a in ax.flat: a.grid(alpha=.18)
    fig.suptitle("Forward chain: signed contrast → LGN → Gabor/configuration → E/I → EEG proxy")
    fig.savefig(out / "机制链.png", dpi=180)
    plt.close(fig)


def _plot_erp(plot_cases, out):
    records = sorted({key[0] for key in plot_cases})
    fig, axes = plt.subplots(max(1, len(records)), 3, figsize=(12, max(6, 2.7 * len(records))),
                             squeeze=False, constrained_layout=True)
    if not records:
        axes[0, 0].text(.5, .5, "No held-out ERP prediction was available", ha="center", va="center")
        for ax in axes.flat[1:]: ax.set_axis_off()
        fig.savefig(out / "留出ERP比较.png", dpi=180)
        plt.close(fig)
        return
    colors = {"left": "#168AAD", "right": "#D17A22"}
    for ri, record in enumerate(records):
        for ci, channel in enumerate(config.CHANNELS):
            ax = axes[ri, ci]
            for condition in ("left", "right"):
                item = plot_cases.get((record, condition))
                if item is None: continue
                t, curves = item["time_ms"], item["curves"]
                ax.plot(t, curves["real"][ci], color=colors[condition],
                        label=f"{condition} measured" if ri == 0 else None)
                ax.plot(t, curves["fitted"][ci], "--", color=colors[condition],
                        label=f"{condition} fitted" if ri == 0 else None)
                if "default" in curves:
                    ax.plot(t, curves["default"][ci], "-.", color=colors[condition], alpha=.8,
                            label=f"{condition} default" if ri == 0 else None)
                if "old" in curves:
                    ax.plot(t, curves["old"][ci], ":", color=colors[condition], alpha=.75,
                            label=f"{condition} old" if ri == 0 else None)
            ax.axvline(200, color="0.55", linestyle=":")
            ax.set(title=f"{record} — {channel}", xlabel="Time from cue onset (ms)", ylabel="Relative signal")
            ax.grid(alpha=.15)
            if ri == 0 and ci == 0: ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Leave-one-record-out Stage1 ERP (no test-record refit)")
    fig.savefig(out / "留出ERP比较.png", dpi=180)
    plt.close(fig)


def _plot_classification(class_rows, controls, out):
    complete = [r for r in class_rows if r.get("status") == "complete"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if complete:
        x = np.arange(len(complete))
        ax[0].bar(x - .18, [r["balanced_accuracy"] for r in complete], .36, label="BA", color="#168AAD")
        ax[0].bar(x + .18, [r["roc_auc"] for r in complete], .36, label="AUC", color="#D17A22")
        ax[0].set_xticks(x, [r["heldout_record"] for r in complete], rotation=25, ha="right")
        ax[0].set_ylim(0, 1)
        ax[0].axhline(.5, color="0.5", linestyle=":")
        ax[0].legend(frameon=False)
    ax[0].set(title="Real-trial 6-D shrinkage LDA", ylabel="Held-out score")
    position = [r for r in controls if r["control"] == "remove_position"]
    if position:
        x = np.arange(len(position))
        ax[1].bar(x, [r["difference_retained_fraction"] for r in position], color="#6A994E")
        ax[1].set_xticks(x, [r["heldout_record"] for r in position], rotation=25, ha="right")
        ax[1].axhline(1.0, color="0.5", linestyle=":")
    ax[1].set(title="Frozen-parameter position ablation", ylabel="Left-right difference retained")
    for a in ax: a.grid(axis="y", alpha=.18)
    fig.savefig(out / "分类与机制对照.png", dpi=180)
    plt.close(fig)


def _save_outputs(path, canonical, cases, per_case):
    arrays = {}
    for (stage, condition), item in canonical.items():
        prefix = f"default__{stage}__{condition}"
        arrays[f"time__{prefix}"] = item["model"].time_ms
        arrays[f"eeg__{prefix}"] = item["model"].eeg
        arrays[f"source__{prefix}"] = item["model"].source_proxy
        arrays[f"observable_modes__{prefix}"] = item["model"].observable_modes
        arrays[f"mean_E_minus_I__{prefix}"] = (item["model"].excitatory.mean(axis=1)
                                                 - item["model"].inhibitory.mean(axis=1))
        arrays[f"gabor__{prefix}"] = item["frontend"]["gabor"]
        arrays[f"lgn_on__{prefix}"] = item["frontend"]["lgn_on"]
        arrays[f"lgn_off__{prefix}"] = item["frontend"]["lgn_off"]
        arrays[f"B__{prefix}"] = item["frontend"]["B"]
        arrays[f"HL__{prefix}"] = item["frontend"]["H_L"]
        arrays[f"HR__{prefix}"] = item["frontend"]["H_R"]
    for case in cases:
        key = f"{case['dataset']}__{case['stage']}__{case['condition']}"
        arrays[f"time__real__{key}"] = case["time_ms"]
        arrays[f"real__{key}"] = case["real"]
        for method, curve in per_case.get((case["dataset"], case["condition"]), {}).items():
            if method == "real":
                continue
            arrays[f"{method}__{key}"] = curve
    np.savez_compressed(path, **arrays)


def _base_manifest(mode, scale, event_audit, legacy_hashes):
    return {
        "project": "Q2 revision_v1 low-dimensional forward model", "run_mode": mode,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "resolution_audit": scale, "time_ms": config.TIME_MS, "stages": config.STAGES,
        "datasets": config.DATASETS, "parameters_default": config.PARAM_DEFAULTS,
        "parameter_bounds": config.PARAM_BOUNDS, "fixed_wc": config.WC_FIXED,
        "pixel_parameters_256": config.PIXEL_PARAMS,
        "lgn_tau_ms": config.LGN_TAU_MS,
        "gabor_angles_deg": config.GABOR_ANGLES_DEG,
        "feature_windows_ms": config.FEATURE_WINDOWS_MS,
        "lda_shrinkage": config.LDA_SHRINKAGE,
        "G_rank": int(np.linalg.matrix_rank(config.LEAD_FIELD)),
        "G": config.LEAD_FIELD, "triangle_templates": template_manifest(),
        "real_event_audit": event_audit,
        "input_and_revision_hashes": _source_hashes(), "legacy_hashes_before": legacy_hashes,
        "interpretation_limits": [
            "model observations use relative units and are not calibrated microvolts",
            "the four MAT files are records, not four independently verified participants",
            "the LDA evaluates measured-feature separability and does not validate the neural mechanism by itself",
            "Task2 Stage2 target layout remains unknown and is descriptive only",
            "inward/outward are standardized model counterfactuals, not paired real conditions",
            "the frozen rank-2 observation matrix is illustrative, not an anatomical lead field",
            "model amplitudes are scaled in training folds and do not establish an absolute physiological unit",
        ],
    }


def run(mode="validate"):
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    legacy_before = _legacy_hashes()
    cases, event_audit = load_cases_with_audit()
    _write_csv(config.OUTPUT_ROOT / "condition_audit.csv", _condition_audit(cases, event_audit))
    scale = choose_resolution()
    manifest = _base_manifest(mode, scale, event_audit, legacy_before)
    if mode == "audit":
        manifest["run_status"] = "audit_complete"
        _write_json(config.OUTPUT_ROOT / "manifest.json", manifest)
        return
    if not scale["resolution_pass"]:
        manifest["run_status"] = "stopped_scale_audit_failed"
        _write_json(config.OUTPUT_ROOT / "manifest.json", manifest)
        raise RuntimeError("both preset spatial scales failed; no EEG fitting was run")

    resolution = scale["chosen_resolution"]
    print(f"Frozen spatial resolution: {resolution}x{resolution}", flush=True)
    canonical = _canonical_models(resolution)
    dt_check = _dt_convergence_check(resolution, canonical)
    if mode == "simulate":
        _save_outputs(config.OUTPUT_ROOT / "model_outputs.npz", canonical, [], {})
        _plot_mechanism(canonical, config.OUTPUT_ROOT)
        manifest["run_status"] = "partial_mode_outputs"
        manifest["canonical_conditions"] = [f"{s}/{c}" for s, c in canonical]
        manifest["dt_convergence_check"] = dt_check
        _write_json(config.OUTPUT_ROOT / "manifest.json", manifest)
        return

    main_cases = [c for c in cases if c["stage"] == "Stage1" and c["eligible_fit"]]
    records = sorted({c["dataset"] for c in main_cases})
    fit_by_record, summary, curve_store, plot_cases = {}, [], {}, {}
    for heldout in records:
        train = [c for c in main_cases if c["dataset"] != heldout]
        try:
            fit = fit_model(train, resolution=resolution)
        except Exception as exc:
            summary.append({"heldout_record": heldout, "stage": "Stage1",
                            "method": "fit_failed", "fit_status": type(exc).__name__,
                            "failure_reason": str(exc), "train_records": sorted({c["dataset"] for c in train})})
            continue
        fit_by_record[heldout] = fit
        fitted_models = _models_for_parameters(
            (("Stage1", "left"), ("Stage1", "right")), fit.parameters, resolution)
        default_map = {(c["dataset"], c["condition"]): {
                           "eeg": canonical[("Stage1", c["condition"])]["model"].eeg,
                           "time_ms": canonical[("Stage1", c["condition"])]["model"].time_ms}
                       for c in train}
        default_amp = estimate_shared_amplitude(train, default_map)
        old_map = {}
        for c in train:
            old = _load_old(c["task"], c["condition"])
            if old is None:
                old_map = {}
                break
            old_map[(c["dataset"], c["condition"])] = old
        old_amp = estimate_shared_amplitude(train, old_map) if old_map else float("nan")
        for case in [c for c in cases if c["dataset"] == heldout]:
            if case["stage"] == "Stage1":
                condition = case["condition"]
                pred = canonical[("Stage1", condition)]["model"]
                fit_pred = fitted_models[("Stage1", condition)]["model"]
                row, fitted = _metric_row(case, "revision_fitted", fit_pred.eeg, fit_pred.time_ms,
                                          fit.amplitude, fit.parameters, fit.loss)
                row.update({"fit_status": fit.status, "fit_success": fit.success,
                            "boundary_flags": fit.boundary_flags, "train_records": fit.train_records})
                summary.append(row)
                row, default = _metric_row(case, "revision_default", pred.eeg, pred.time_ms,
                                           default_amp, config.PARAM_DEFAULTS)
                summary.append(row)
                curves = {"real": case["real"], "fitted": fitted, "default": default}
                old = _load_old(case["task"], condition)
                if old is not None and old_map:
                    old_curve = _interpolate_prediction(old_amp * old["eeg"], old["time_ms"], case["time_ms"])
                    met = erp_metrics(case["real"], old_curve, case["time_ms"], "Stage1")
                    summary.append({"heldout_record": heldout, "task": case["task"], "stage": "Stage1",
                                    "condition": condition, "method": "old_04_baseline",
                                    "n_trials": case["n_trials"], "amplitude": old_amp,
                                    "rmse": met["rmse"], "nrmse_by_real_rms": met["nrmse_by_real_rms"],
                                    "channel_corr_mean": met["channel_corr_mean"],
                                    "channel_correlations": met["channel_correlations"]})
                    curves["old"] = old_curve
                curve_store[(heldout, condition)] = curves
                plot_cases[(heldout, condition)] = {"time_ms": case["time_ms"], "curves": curves}
            elif case["stage"] == "Stage2" and case["task"] == "Task1":
                dots = run_forward_condition("Stage2", "dots", fit.parameters, resolution,
                                             amplitude=fit.amplitude)
                dots_curve = _interpolate_prediction(dots.eeg_scaled, dots.time_ms, case["time_ms"])
                met = erp_metrics(case["real"], dots_curve, case["time_ms"], "Stage2")
                summary.append({"heldout_record": heldout, "task": case["task"], "stage": "Stage2",
                                "condition": "dots", "method": "frozen_parameter_dots_control",
                                "n_trials": case["n_trials"], "amplitude": fit.amplitude,
                                "param_tau_s": fit.parameters["tau_s"], "param_g_i": fit.parameters["g_i"],
                                "param_tau_a": fit.parameters["tau_a"], "rmse": met["rmse"],
                                "nrmse_by_real_rms": met["nrmse_by_real_rms"],
                                "channel_corr_mean": met["channel_corr_mean"]})
                curve_store[(heldout, "dots")] = {"frozen_dots": dots_curve}
            elif case["stage"] == "Stage2" and case["condition"] == "target_unknown":
                late = (case["time_ms"] >= 250.0) & (case["time_ms"] <= 500.0)
                row = {"heldout_record": heldout, "task": case["task"], "stage": "Stage2",
                       "condition": "target_unknown", "method": "descriptive_erp_only",
                       "role": case["role"], "n_trials": case["n_trials"],
                       "observed_last_ms": float(case["time_ms"][-1])}
                for ci, channel in enumerate(config.CHANNELS):
                    row[f"{channel}_late_mean"] = float(case["real"][ci, late].mean()) if late.any() else float("nan")
                    row[f"{channel}_late_rms"] = float(np.sqrt(np.mean(case["real"][ci, late] ** 2))) if late.any() else float("nan")
                summary.append(row)
        pair = {c["condition"]: c for c in cases
                if c["dataset"] == heldout and c["stage"] == "Stage1" and c["condition"] in ("left", "right")}
        if set(pair) == {"left", "right"}:
            left_case, right_case = pair["left"], pair["right"]
            time_ms = left_case["time_ms"]
            real_diff = right_case["real"] - left_case["real"]
            right_model = fitted_models[("Stage1", "right")]["model"]
            left_model = fitted_models[("Stage1", "left")]["model"]
            fit_diff = fit.amplitude * (
                _interpolate_prediction(right_model.eeg, right_model.time_ms, time_ms)
                - _interpolate_prediction(left_model.eeg, left_model.time_ms, time_ms))
            summary.append({"heldout_record": heldout, "task": left_case["task"],
                            "stage": "Stage1", "condition": "right_minus_left",
                            "method": "fitted_lateral_effect", "n_trials": min(left_case["n_trials"], right_case["n_trials"]),
                            "real_difference_rms": float(np.sqrt(np.mean(real_diff ** 2))),
                            "model_difference_rms": float(np.sqrt(np.mean(fit_diff ** 2))),
                            "difference_rmse": float(np.sqrt(np.mean((real_diff - fit_diff) ** 2))),
                            "real_F4_minus_F3_rms": float(np.sqrt(np.mean((real_diff[2] - real_diff[0]) ** 2))),
                            "model_F4_minus_F3_rms": float(np.sqrt(np.mean((fit_diff[2] - fit_diff[0]) ** 2)))})

    described = {r.get("heldout_record") for r in summary if r.get("method") == "descriptive_erp_only"}
    for case in cases:
        if case["stage"] != "Stage2" or case["condition"] != "target_unknown" or case["dataset"] in described:
            continue
        late = (case["time_ms"] >= 250.0) & (case["time_ms"] <= 500.0)
        row = {"heldout_record": case["dataset"], "task": case["task"], "stage": "Stage2",
               "condition": "target_unknown", "method": "descriptive_erp_only",
               "role": case["role"], "n_trials": case["n_trials"],
               "observed_last_ms": float(case["time_ms"][-1])}
        for ci, channel in enumerate(config.CHANNELS):
            row[f"{channel}_late_mean"] = float(case["real"][ci, late].mean()) if late.any() else float("nan")
            row[f"{channel}_late_rms"] = float(np.sqrt(np.mean(case["real"][ci, late] ** 2))) if late.any() else float("nan")
        summary.append(row)

    scores, class_rows, class_mean = classification_leave_one_record(cases)
    controls = mechanism_control_rows(fit_by_record, resolution)
    for row in class_rows:
        summary.append({"heldout_record": row.get("heldout_record"),
                        "method": "real_only_shrinkage_LDA", **row})
    summary.append({"heldout_record": "FOUR_RECORD_MEAN", "method": "real_only_shrinkage_LDA",
                    **class_mean})
    _write_csv(config.OUTPUT_ROOT / "fold_summary.csv", summary)
    _write_csv(config.OUTPUT_ROOT / "trial_scores.csv", scores)
    _write_csv(config.OUTPUT_ROOT / "mechanism_controls.csv", controls)
    _save_outputs(config.OUTPUT_ROOT / "model_outputs.npz", canonical, cases, curve_store)
    _plot_mechanism(canonical, config.OUTPUT_ROOT)
    _plot_erp(plot_cases, config.OUTPUT_ROOT)
    _plot_classification(class_rows, controls, config.OUTPUT_ROOT)
    manifest.update({
        "run_status": "validate_complete" if len(fit_by_record) == len(records) else "validate_with_fit_failures",
        "n_fit_failures": len(records) - len(fit_by_record),
        "fit_by_heldout_record": {r: fit.to_dict() for r, fit in fit_by_record.items()},
        "classification_folds": class_rows, "classification_four_record_mean": class_mean,
        "fit_case_count": len(main_cases),
        "dt_convergence_check": dt_check,
        "control_case_count": sum(c["stage"] == "Stage2" and c["task"] == "Task1" for c in cases),
        "unknown_layout_case_count": sum(c["condition"] == "target_unknown" for c in cases),
        "legacy_baseline_status": "available" if all(_load_old(c["task"], c["condition"])
            is not None for c in main_cases) else "partially_unavailable",
    })
    _write_json(config.OUTPUT_ROOT / "manifest.json", manifest)
    if _legacy_hashes() != legacy_before:
        raise RuntimeError("legacy scripts or baseline output changed during revision run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "simulate", "validate"), default="validate")
    run(parser.parse_args().mode)


if __name__ == "__main__":
    main()
