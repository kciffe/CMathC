"""Cue-aligned Q3 dynamic-state model and record-held-out EEG validation.

Candidate target times and stimulus identities are sensitivity scenarios. The
script never treats channel 9 as a verified reaction time, target-onset marker,
correctness label, EEG feature, or model input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[3]
Q3_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, Q3_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dynamic_cognitive_model as macro
from common import detect_cues, load_raw_record
from config import EEG_CHANNELS, RECORDS


OUTPUT_DIR = Q3_DIR / "output" / "dynamic_heldout_validation"
FIGURE_DIR = OUTPUT_DIR / "figures"
EVENT_AUDIT_PATH = Q3_DIR / "output" / "continuation_audit" / "event_timing_by_trial.csv"
TIME_START_S = -0.2
TIME_STOP_S = 3.0
MODEL_RATE_HZ = macro.Q2_RATE_HZ
OUTPUT_RATE_HZ = 128.0
OUTPUT_TIME_S = TIME_START_S + np.arange(
    int(np.floor((TIME_STOP_S - TIME_START_S) * OUTPUT_RATE_HZ)) + 1,
    dtype=float,
) / OUTPUT_RATE_HZ
DEFAULT_PREPROCESSING = ("none", "causal", "zero_phase")
DEFAULT_ONSETS: tuple[float | None, ...] = (None, 2.0, 2.2, 2.4)
DEFAULT_TARGET_TYPES = ("dots", "inward", "outward")
DEFAULT_MATCH_EVIDENCE = (-1.0, 1.0)
RIDGE_ALPHA = 1.0
RAW_CLIP_LIMIT = 999.5

WINDOWS = {
    "cue_stage": (0.0, 0.8),
    "late_stage": (0.8, 2.8),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DengXian", "Arial"],
    "axes.unicode_minus": False,
})

PREPROCESSING_LABELS = {
    "none": "不滤波",
    "causal": "因果滤波",
    "zero_phase": "零相位滤波",
}
TARGET_TYPE_LABELS = {
    "dots": "点阵刺激",
    "inward": "向内运动刺激",
    "outward": "向外运动刺激",
}
MODEL_LABELS = {
    "Training_mean_template": "训练集均值模板",
    "Q2_visual_only": "问题二视觉模型",
    "Q2_plus_memory": "问题二视觉＋记忆模型",
    "Q2_plus_memory_control": "问题二视觉＋记忆＋控制模型",
    "Measured held-out EEG": "留出实测脑电",
}


def _preprocessing_label(mode: str) -> str:
    return PREPROCESSING_LABELS.get(mode, mode)


def _target_type_label(target_type: str) -> str:
    return TARGET_TYPE_LABELS.get(target_type, target_type)


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _record_label(record: str) -> str:
    try:
        index = list(RECORDS).index(record) + 1
    except ValueError:
        return "留出记录"
    return f"记录{index}"


def _format_onset_cn(value: float | None) -> str:
    return "无目标输入" if value is None else f"{value:.1f} 秒"


def _add_framed_figure_legend(fig, *args, **kwargs):
    style = {
        "frameon": True,
        "fancybox": True,
        "framealpha": 1.0,
        "facecolor": "white",
        "edgecolor": "#c8c8c8",
        "borderpad": 0.7,
        "labelspacing": 0.6,
        "handletextpad": 0.8,
        "columnspacing": 1.8,
        "handlelength": 2.0,
    }
    style.update(kwargs)
    legend = fig.legend(*args, **style)
    legend.get_frame().set_linewidth(0.8)
    return legend


def extract_fixed_epoch(
    signal: np.ndarray,
    timestamps_s: np.ndarray,
    cue_time_s: float,
    next_cue_time_s: float | None,
    start_s: float = TIME_START_S,
    stop_s: float = TIME_STOP_S,
    target_fs_hz: float = MODEL_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a continuous three-channel record onto one cue-relative grid."""

    values = np.asarray(signal, dtype=float)
    times = np.asarray(timestamps_s, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] != times.size:
        raise ValueError("signal must be [F3/Fz/F4,time] and match timestamps")
    if times.size < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not np.isfinite(values).all() or not np.isfinite(cue_time_s):
        raise ValueError("signal and cue time must be finite")
    if not np.isfinite(target_fs_hz) or target_fs_hz <= 0 or not start_s < stop_s:
        raise ValueError("epoch bounds and target sample rate must be valid")
    relative_time = np.arange(
        start_s, stop_s + 0.5 / target_fs_hz, 1.0 / target_fs_hz
    )
    absolute_time = cue_time_s + relative_time
    tolerance = 0.51 / target_fs_hz
    if absolute_time[0] < times[0] - tolerance or absolute_time[-1] > times[-1] + tolerance:
        raise ValueError("cue-locked epoch extends beyond available continuous data")
    if next_cue_time_s is not None and absolute_time[-1] >= next_cue_time_s:
        raise ValueError("cue-locked epoch overlaps the next VisCue event")
    epoch = np.vstack(
        [np.interp(absolute_time, times, channel) for channel in values]
    )
    return epoch, relative_time


def _validate_model_arrays(
    observed: np.ndarray,
    q2_sensor: np.ndarray,
    memory: np.ndarray,
    control: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(observed, dtype=float)
    q2 = np.asarray(q2_sensor, dtype=float)
    h = np.asarray(memory, dtype=float)
    p = np.asarray(control, dtype=float)
    if y.ndim != 3 or y.shape[1] != 3:
        raise ValueError("observed EEG must have shape [group,3,time]")
    if q2.shape != y.shape or h.shape != (y.shape[0], y.shape[2]) or p.shape != h.shape:
        raise ValueError("Q2, memory, control, and observed EEG arrays must align")
    if not all(np.isfinite(array).all() for array in (y, q2, h, p)):
        raise ValueError("all observed signals and model states must be finite")
    return y, q2, h, p


def _features_for_channel(
    q2_channel: np.ndarray,
    memory: np.ndarray,
    control: np.ndarray,
    include_memory: bool,
    include_control: bool,
    q2_control_interaction: np.ndarray | None = None,
) -> np.ndarray:
    pieces = [q2_channel.reshape(-1, 1)]
    if include_memory:
        pieces.append(memory.reshape(-1, 1))
    if include_control:
        interaction = q2_channel * control if q2_control_interaction is None else q2_control_interaction
        pieces.extend((control.reshape(-1, 1), interaction.reshape(-1, 1)))
    return np.concatenate(pieces, axis=1)


def fit_sensor_mapping(
    observed: np.ndarray,
    q2_sensor: np.ndarray,
    memory: np.ndarray,
    control: np.ndarray,
    *,
    include_memory: bool = True,
    include_control: bool = True,
    ridge_alpha: float = RIDGE_ALPHA,
    q2_control_interaction: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a training-only sensor observation map with per-channel ridge gains.

    The learned memory/control sensor coefficients are predictive loadings, not
    cortical source locations. Standardization moments are estimated on the
    training records only and carried unchanged to the held-out record.
    """

    y, q2, h, p = _validate_model_arrays(observed, q2_sensor, memory, control)
    interaction = None if q2_control_interaction is None else np.asarray(q2_control_interaction, dtype=float)
    if interaction is not None and (interaction.shape != q2.shape or not np.isfinite(interaction).all()):
        raise ValueError("Q2-control interaction must be finite and align with EEG sensors")
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("ridge_alpha must be finite and nonnegative")
    fits: list[dict[str, np.ndarray | float]] = []
    for channel_index in range(3):
        x = _features_for_channel(
            q2[:, channel_index, :], h, p, include_memory, include_control,
            None if interaction is None else interaction[:, channel_index, :],
        )
        target = y[:, channel_index, :].reshape(-1)
        means = x.mean(axis=0)
        scales = x.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        standardized = (x - means) / scales
        target_mean = float(target.mean())
        centered_target = target - target_mean
        gram = standardized.T @ standardized
        coefficients = np.linalg.solve(
            gram + float(ridge_alpha) * np.eye(gram.shape[0]),
            standardized.T @ centered_target,
        )
        fits.append({
            "feature_mean": means,
            "feature_scale": scales,
            "target_mean": target_mean,
            "coefficients": coefficients,
        })
    return {
        "include_memory": bool(include_memory),
        "include_control": bool(include_control),
        "ridge_alpha": float(ridge_alpha),
        "channel_fits": fits,
    }


def predict_sensor_mapping(
    fit: dict[str, Any],
    q2_sensor: np.ndarray,
    memory: np.ndarray,
    control: np.ndarray,
    q2_control_interaction: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the training-only observation map to aligned test data."""

    q2 = np.asarray(q2_sensor, dtype=float)
    h = np.asarray(memory, dtype=float)
    p = np.asarray(control, dtype=float)
    if q2.ndim != 3 or q2.shape[1] != 3 or h.shape != (q2.shape[0], q2.shape[2]) or p.shape != h.shape:
        raise ValueError("prediction Q2, memory, and control arrays must align")
    interaction = None if q2_control_interaction is None else np.asarray(q2_control_interaction, dtype=float)
    if interaction is not None and (interaction.shape != q2.shape or not np.isfinite(interaction).all()):
        raise ValueError("prediction Q2-control interaction must align with EEG sensors")
    prediction = np.empty_like(q2)
    for channel_index, channel_fit in enumerate(fit["channel_fits"]):
        x = _features_for_channel(
            q2[:, channel_index, :], h, p,
            fit["include_memory"], fit["include_control"],
            None if interaction is None else interaction[:, channel_index, :],
        )
        z = (x - channel_fit["feature_mean"]) / channel_fit["feature_scale"]
        prediction[:, channel_index, :] = (
            channel_fit["target_mean"] + z @ channel_fit["coefficients"]
        ).reshape(q2.shape[0], q2.shape[2])
    return prediction


def predict_training_side_template(
    observed_train: np.ndarray,
    train_cue_sides: np.ndarray,
    test_cue_sides: np.ndarray,
) -> np.ndarray:
    """Predict each held-out group with the training mean waveform for its cue side."""

    values = np.asarray(observed_train, dtype=float)
    train_sides = np.asarray(train_cue_sides).reshape(-1)
    test_sides = np.asarray(test_cue_sides).reshape(-1)
    if values.ndim != 3 or values.shape[1] != 3 or values.shape[0] != train_sides.size:
        raise ValueError("training template arrays must align as [group,3,time]")
    if not np.isfinite(values).all() or not np.isfinite(train_sides).all() or not np.isfinite(test_sides).all():
        raise ValueError("training template inputs must be finite")
    templates: dict[float, np.ndarray] = {}
    for side in np.unique(train_sides):
        templates[float(side)] = values[train_sides == side].mean(axis=0)
    missing = [float(side) for side in np.unique(test_sides) if float(side) not in templates]
    if missing:
        raise ValueError(f"no training cue-side template for test group(s): {missing}")
    return np.stack([templates[float(side)] for side in test_sides], axis=0)


def sensor_contributions(
    fit: dict[str, Any],
    q2_sensor: np.ndarray,
    memory: np.ndarray,
    control: np.ndarray,
    q2_control_interaction: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return additive, fitted sensor-space components for mechanism figures."""

    q2 = np.asarray(q2_sensor, dtype=float)
    h = np.asarray(memory, dtype=float)
    p = np.asarray(control, dtype=float)
    interaction = q2 * p[:, None, :] if q2_control_interaction is None else np.asarray(q2_control_interaction, dtype=float)
    if interaction.shape != q2.shape:
        raise ValueError("contribution Q2-control interaction must align with EEG sensors")
    result = {name: np.zeros_like(q2) for name in ("baseline", "q2_visual", "memory", "control_direct", "control_feedback")}
    for channel_index, channel_fit in enumerate(fit["channel_fits"]):
        means = channel_fit["feature_mean"]
        scales = channel_fit["feature_scale"]
        beta = channel_fit["coefficients"]
        result["baseline"][0, channel_index, :] = channel_fit["target_mean"] - float(np.sum(means / scales * beta))
        feature_index = 0
        result["q2_visual"][0, channel_index, :] = ((q2[0, channel_index] - means[feature_index]) / scales[feature_index]) * beta[feature_index]
        feature_index += 1
        if fit["include_memory"]:
            result["memory"][0, channel_index, :] = ((h[0] - means[feature_index]) / scales[feature_index]) * beta[feature_index]
            feature_index += 1
        if fit["include_control"]:
            result["control_direct"][0, channel_index, :] = ((p[0] - means[feature_index]) / scales[feature_index]) * beta[feature_index]
            feature_index += 1
            result["control_feedback"][0, channel_index, :] = ((interaction[0, channel_index] - means[feature_index]) / scales[feature_index]) * beta[feature_index]
    return result


def _preprocess_aligned(
    signal: np.ndarray,
    source_time_s: np.ndarray,
    source_fs_hz: float,
    config: macro.PreprocessingConfig,
) -> np.ndarray:
    processed, processed_time = macro.apply_preprocessing(
        signal, source_time_s, source_fs_hz, config
    )
    aligned = np.vstack(
        [np.interp(OUTPUT_TIME_S, processed_time, channel) for channel in processed]
    )
    return aligned


def _preprocess_state(
    state: np.ndarray,
    source_time_s: np.ndarray,
    config: macro.PreprocessingConfig,
) -> np.ndarray:
    repeated = np.repeat(np.asarray(state, dtype=float)[None, :], 3, axis=0)
    return _preprocess_aligned(repeated, source_time_s, MODEL_RATE_HZ, config)[0]


def filter_continuous_eeg(
    signal: np.ndarray,
    source_fs_hz: float,
    config: macro.PreprocessingConfig,
) -> np.ndarray:
    """Filter a continuous record before event epoching to avoid epoch-edge resets."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 32:
        raise ValueError("continuous EEG must have shape [3,time] with at least 32 samples")
    if not np.isfinite(values).all() or not np.isfinite(source_fs_hz) or source_fs_hz <= 0:
        raise ValueError("continuous EEG and sample rate must be finite and valid")
    if config.mode == "none":
        return values.copy()
    if config.mode not in {"causal", "zero_phase"}:
        raise ValueError("filter mode must be none, causal, or zero_phase")
    if not 0.0 < config.low_hz < config.high_hz < source_fs_hz / 2.0:
        raise ValueError("band edges must lie strictly below the continuous-record Nyquist")
    sos = butter(
        int(config.order), (config.low_hz, config.high_hz),
        btype="bandpass", fs=float(source_fs_hz), output="sos",
    )
    if config.mode == "causal":
        return sosfilt(sos, values, axis=-1)
    return sosfiltfilt(sos, values, axis=-1)


def _load_observed_epochs(
    preprocessing_modes: Iterable[str],
) -> tuple[dict[str, dict[tuple[str, int], np.ndarray]], pd.DataFrame]:
    """Extract cue-only keyed EEG means while retaining a per-trial QC ledger."""

    mode_list = tuple(preprocessing_modes)
    config_by_mode = {
        mode: macro.PreprocessingConfig(
            mode=mode, low_hz=0.5, high_hz=30.0, order=4,
            target_fs_hz=OUTPUT_RATE_HZ,
            baseline_window_s=(TIME_START_S, 0.0),
        )
        for mode in mode_list
    }
    grouped: dict[str, dict[tuple[str, int], list[np.ndarray]]] = {
        mode: {} for mode in mode_list
    }
    audit_rows: list[dict[str, Any]] = []
    for record in RECORDS:
        raw = load_raw_record(record)
        timestamps = np.asarray(raw["channels"]["TimeStamp"], dtype=float).reshape(-1)
        eeg = np.stack([np.asarray(raw["channels"][channel], dtype=float) for channel in EEG_CHANNELS])
        source_fs_hz = float(raw["sample_rate_hz"])
        filtered_by_mode = {
            mode: filter_continuous_eeg(eeg, source_fs_hz, cfg)
            for mode, cfg in config_by_mode.items()
        }
        resample_config = macro.PreprocessingConfig(
            mode="none", low_hz=0.5, high_hz=30.0, order=4,
            target_fs_hz=OUTPUT_RATE_HZ,
            baseline_window_s=(TIME_START_S, 0.0),
        )
        events = detect_cues(raw["channels"]["VisCue"], timestamps)
        for event_index, event in enumerate(events):
            cue_time = float(event["cue_time_s"])
            next_cue = float(events[event_index + 1]["cue_time_s"]) if event_index + 1 < len(events) else None
            side = int(event["cue_side"])
            row: dict[str, Any] = {
                "record": record,
                "original_trial_index": int(event["original_trial_index"]),
                "cue_time_s": cue_time,
                "cue_side": side,
                "next_cue_time_s": next_cue,
                "included": False,
                "exclusion_reason": "",
            }
            try:
                raw_epoch, epoch_time = extract_fixed_epoch(
                    eeg, timestamps, cue_time, next_cue,
                    start_s=TIME_START_S, stop_s=TIME_STOP_S,
                    target_fs_hz=MODEL_RATE_HZ,
                )
            except ValueError as error:
                row["exclusion_reason"] = str(error)
                audit_rows.append(row)
                continue
            if not np.isfinite(raw_epoch).all():
                row["exclusion_reason"] = "nonfinite_raw_eeg"
                audit_rows.append(row)
                continue
            if np.any(np.abs(raw_epoch) >= RAW_CLIP_LIMIT):
                row["exclusion_reason"] = "raw_amplitude_at_or_above_999p5"
                audit_rows.append(row)
                continue
            if np.any(np.ptp(raw_epoch, axis=1) < 1e-8):
                row["exclusion_reason"] = "flat_eeg_channel"
                audit_rows.append(row)
                continue
            try:
                for mode, cfg in config_by_mode.items():
                    filtered_epoch, filtered_time = extract_fixed_epoch(
                        filtered_by_mode[mode], timestamps, cue_time, next_cue,
                        start_s=TIME_START_S, stop_s=TIME_STOP_S,
                        target_fs_hz=MODEL_RATE_HZ,
                    )
                    processed = _preprocess_aligned(
                        filtered_epoch, filtered_time, MODEL_RATE_HZ, resample_config
                    )
                    grouped[mode].setdefault((record, side), []).append(processed)
            except ValueError as error:
                row["exclusion_reason"] = f"preprocessing_error:{error}"
                audit_rows.append(row)
                continue
            row["included"] = True
            audit_rows.append(row)

    means: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    for mode, groups in grouped.items():
        means[mode] = {
            key: np.mean(np.stack(epochs, axis=0), axis=0)
            for key, epochs in groups.items()
            if epochs
        }
    audit = pd.DataFrame(audit_rows)
    return means, audit


def _scenario_grid(
    target_types: tuple[str, ...],
    target_onsets_s: tuple[float | None, ...],
    target_duration_s: float | None,
    match_evidence_values: tuple[float, ...],
    cue_sides: tuple[str, ...] = ("left", "right"),
) -> tuple[macro.DynamicScenario, ...]:
    scenarios: list[macro.DynamicScenario] = []
    for cue_side in cue_sides:
        for onset in target_onsets_s:
            # A no-target branch is kept once; target image type and match are
            # undefined when there is no candidate target event.
            types = target_types if onset is not None else (target_types[0],)
            evidence_values = match_evidence_values if onset is not None else (1.0,)
            for target_type in types:
                for evidence in evidence_values:
                    scenarios.append(macro.DynamicScenario(
                        cue_side=cue_side,
                        target_stimulus=target_type,
                        target_onset_s=onset,
                        target_duration_s=target_duration_s,
                        match_evidence=float(evidence),
                    ))
    return tuple(scenarios)


def _simulate_q2_base(
    scenario: macro.DynamicScenario,
    resolution: int,
    simulation_stop_s: float,
) -> dict[str, Any]:
    """Run the Q2 forward chain once and keep its arrays in memory for this run."""

    dynamic_scenario = macro.DynamicScenario(
        scenario.cue_side, scenario.target_stimulus, scenario.target_onset_s,
        scenario.target_duration_s, 1.0,
    )
    simulated = macro.simulate_q2_scenario(
        dynamic_scenario, resolution=resolution,
        simulation_stop_s=simulation_stop_s,
    )
    compact = {
        key: np.asarray(simulated[key])
        for key in ("time_s", "cue_gate", "target_gate", "visual_drive", "q2_sensor", "q2_source_proxy")
    }
    compact["projection_max_abs_error"] = float(simulated["projection_max_abs_error"])
    return compact


def _remove_stale_q2_caches(output_dir: Path) -> int:
    """Remove only this script's obsolete Q2 NPZ caches; leave other files alone."""

    cache_dir = output_dir / "q2_cache"
    if not cache_dir.is_dir():
        return 0
    removed = 0
    for path in cache_dir.glob("q2_forward_*.npz"):
        path.unlink()
        removed += 1
    try:
        cache_dir.rmdir()
    except OSError:
        pass
    return removed


def _make_processed_basis(
    q2_base: dict[str, Any],
    scenario: macro.DynamicScenario,
    preprocessing: macro.PreprocessingConfig,
) -> dict[str, np.ndarray | float]:
    states = macro.integrate_macro_states(
        q2_base["visual_drive"], q2_base["cue_gate"], q2_base["target_gate"],
        scenario.match_evidence, dt_s=1.0 / MODEL_RATE_HZ,
    )
    raw_q2 = np.asarray(q2_base["q2_sensor"], dtype=float)
    raw_control = np.asarray(states["control"], dtype=float)
    processed_q2 = _preprocess_aligned(
        raw_q2, q2_base["time_s"], MODEL_RATE_HZ, preprocessing
    )
    processed_memory = _preprocess_state(
        states["memory"], q2_base["time_s"], preprocessing
    )
    processed_control = _preprocess_state(
        raw_control, q2_base["time_s"], preprocessing
    )
    raw_feedback = raw_q2 * raw_control[None, :]
    processed_feedback = _preprocess_aligned(
        raw_feedback, q2_base["time_s"], MODEL_RATE_HZ, preprocessing
    )
    return {
        "q2_sensor": processed_q2,
        "memory": processed_memory,
        "control": processed_control,
        "q2_control_interaction": processed_feedback,
        "visual_state": np.interp(
            OUTPUT_TIME_S, q2_base["time_s"], states["visual"]
        ),
        "memory_state": np.interp(
            OUTPUT_TIME_S, q2_base["time_s"], states["memory"]
        ),
        "control_state": np.interp(
            OUTPUT_TIME_S, q2_base["time_s"], states["control"]
        ),
        "projection_max_abs_error": float(q2_base["projection_max_abs_error"]),
    }


def _score_prediction(
    observed: np.ndarray,
    predicted: np.ndarray,
    relative_time_s: np.ndarray,
    window: tuple[float, float],
) -> tuple[float, float, float]:
    mask = (relative_time_s >= window[0]) & (relative_time_s < window[1])
    actual = np.asarray(observed, dtype=float)[..., mask].reshape(-1)
    estimate = np.asarray(predicted, dtype=float)[..., mask].reshape(-1)
    residual = actual - estimate
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    scale = float(np.std(actual, ddof=0))
    nrmse = rmse / scale if scale > 1e-12 else float("nan")
    if np.std(actual) <= 1e-12 or np.std(estimate) <= 1e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(actual, estimate)[0, 1])
    return rmse, nrmse, correlation


def _candidate_window(onset_s: float | None) -> tuple[float, float] | None:
    if onset_s is None:
        return None
    stop_s = min(float(onset_s) + 0.6, TIME_STOP_S)
    return (float(onset_s), stop_s) if stop_s > onset_s else None


def _candidate_key(scenario: macro.DynamicScenario) -> tuple[str, float | None, float]:
    return (scenario.target_stimulus, scenario.target_onset_s, float(scenario.match_evidence))


def _basis_arrays(
    group_keys: list[tuple[str, int]],
    mode: str,
    target_type: str,
    onset_s: float | None,
    evidence: float,
    observed: dict[str, dict[tuple[str, int], np.ndarray]],
    basis_lookup: dict[tuple[str, str, str, float | None, float], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.stack([observed[mode][key] for key in group_keys], axis=0)
    q2: list[np.ndarray] = []
    memory: list[np.ndarray] = []
    control: list[np.ndarray] = []
    interaction: list[np.ndarray] = []
    for _, cue_side_code in group_keys:
        cue_side = "left" if cue_side_code < 0 else "right"
        basis = basis_lookup[(mode, cue_side, target_type, onset_s, evidence)]
        q2.append(basis["q2_sensor"])
        memory.append(basis["memory"])
        control.append(basis["control"])
        interaction.append(basis["q2_control_interaction"])
    return (
        y,
        np.stack(q2),
        np.stack(memory),
        np.stack(control),
        np.stack(interaction),
        np.asarray([1.0 if key[1] < 0 else -1.0 for key in group_keys]),
    )


def _run_record_heldout(
    observed: dict[str, dict[tuple[str, int], np.ndarray]],
    scenarios: tuple[macro.DynamicScenario, ...],
    basis_lookup: dict[tuple[str, str, str, float | None, float], dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any] | None]:
    metric_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    exemplar: dict[str, Any] | None = None
    exemplar_score: tuple[float, float, float] | None = None
    preferred_mode = "causal" if "causal" in observed else next(iter(observed))
    models = (
        ("Training_mean_template", False, False),
        ("Q2_visual_only", False, False),
        ("Q2_plus_memory", True, False),
        ("Q2_plus_memory_control", True, True),
    )
    group_keys_all = sorted(
        (key for key in observed[next(iter(observed))]),
        key=lambda item: (RECORDS.index(item[0]) if item[0] in RECORDS else 999, item[1]),
    )
    scenario_candidates = {
        _candidate_key(scenario)
        for scenario in scenarios
        if scenario.cue_side == "left"
    }
    for mode in observed:
        for target_type, onset_s, evidence in sorted(
            scenario_candidates,
            key=lambda item: (item[1] is not None, -1.0 if item[1] is None else item[1], item[0], item[2]),
        ):
            for heldout_record in RECORDS:
                train_keys = [key for key in group_keys_all if key[0] != heldout_record]
                test_keys = [key for key in group_keys_all if key[0] == heldout_record]
                if not train_keys or not test_keys:
                    continue
                y_train, q2_train, h_train, p_train, qp_train, train_sides = _basis_arrays(
                    train_keys, mode, target_type, onset_s, evidence,
                    observed, basis_lookup,
                )
                y_test, q2_test, h_test, p_test, qp_test, test_sides = _basis_arrays(
                    test_keys, mode, target_type, onset_s, evidence,
                    observed, basis_lookup,
                )
                for model_name, include_memory, include_control in models:
                    fit = None
                    if model_name == "Training_mean_template":
                        y_pred = predict_training_side_template(y_train, train_sides, test_sides)
                    else:
                        fit = fit_sensor_mapping(
                            y_train, q2_train, h_train, p_train,
                            include_memory=include_memory,
                            include_control=include_control,
                            ridge_alpha=RIDGE_ALPHA,
                            q2_control_interaction=qp_train,
                        )
                        y_pred = predict_sensor_mapping(
                            fit, q2_test, h_test, p_test,
                            q2_control_interaction=qp_test,
                        )
                    if not np.isfinite(y_pred).all():
                        raise FloatingPointError(
                            f"nonfinite prediction for {mode}/{heldout_record}/{model_name}"
                        )
                    prediction_records.append({
                        "preprocessing": mode,
                        "target_stimulus_candidate": target_type,
                        "target_onset_candidate_s": onset_s,
                        "match_evidence_scenario": evidence,
                        "heldout_record": heldout_record,
                        "model": model_name,
                        "observed": y_test.copy(),
                        "predicted": y_pred.copy(),
                        "group_keys": list(test_keys),
                    })
                    windows = dict(WINDOWS)
                    candidate_window = _candidate_window(onset_s)
                    if candidate_window is not None:
                        windows["candidate_target_600ms"] = candidate_window
                    for group_index, group_key in enumerate(test_keys):
                        for window_name, window_bounds in windows.items():
                            rmse, nrmse, corr = _score_prediction(
                                y_test[group_index], y_pred[group_index],
                                OUTPUT_TIME_S, window_bounds,
                            )
                            metric_rows.append({
                                "preprocessing": mode,
                                "target_stimulus_candidate": target_type,
                                "target_onset_candidate_s": onset_s,
                                "target_duration_s_assumption": (
                                    next(s.target_duration_s for s in scenarios if s.target_stimulus == target_type and s.target_onset_s == onset_s)
                                ),
                                "match_evidence_scenario": evidence if onset_s is not None else np.nan,
                                "heldout_record": heldout_record,
                                "cue_side": "left" if group_key[1] < 0 else "right",
                                "model": model_name,
                                "evaluation_window": window_name,
                                "window_start_s": window_bounds[0],
                                "window_stop_s": window_bounds[1],
                                "rmse_raw_units": rmse,
                                "nrmse_by_heldout_sd": nrmse,
                                "correlation": corr,
                                "n_train_records": len({key[0] for key in train_keys}),
                                "n_train_record_cue_means": len(train_keys),
                                "n_test_trials_in_cue_group": np.nan,
                                "fit_parameters_train_only": True,
                                "response_channel_used": False,
                            })
                    if (
                        model_name == "Q2_plus_memory_control"
                        and mode == preferred_mode
                        and heldout_record == RECORDS[-1]
                    ):
                        candidate_score = (
                            0.0 if target_type == "dots" else 1.0,
                            10.0 if onset_s is None else abs(float(onset_s) - 2.2),
                            0.0 if evidence < 0 else 1.0,
                        )
                        if exemplar_score is not None and candidate_score >= exemplar_score:
                            continue
                        exemplar_score = candidate_score
                        exemplar = {
                        "fit": fit,
                            "observed": y_test[:1].copy(),
                            "predicted": y_pred[:1].copy(),
                            "q2": q2_test[:1].copy(),
                            "memory": h_test[:1].copy(),
                            "control": p_test[:1].copy(),
                            "interaction": qp_test[:1].copy(),
                            "basis": basis_lookup[(mode, "left", target_type, onset_s, evidence)],
                            "record": heldout_record,
                            "cue_side": "left",
                            "scenario": (target_type, onset_s, evidence),
                        }
    return pd.DataFrame(metric_rows), prediction_records, exemplar


def _plot_event_windows(audit_path: Path, output_path: Path) -> dict[str, Any]:
    if not audit_path.exists():
        raise FileNotFoundError(f"event-semantics audit is required: {audit_path}")
    events = pd.read_csv(audit_path, encoding="utf-8-sig")
    if "response_marker_time_s" not in events or "cue_onset_time_s" not in events:
        raise ValueError("event audit lacks cue/response marker times")
    marker_relative = (
        events["response_marker_time_s"].astype(float)
        - events["cue_onset_time_s"].astype(float)
    ).dropna().to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6))
    axes[0].hist(marker_relative, bins=np.arange(1.7, 2.61, 0.025), color="#547D9A", alpha=0.86)
    for candidate, color in ((2.0, "#4C8C6B"), (2.2, "#D1843D"), (2.4, "#9B5C71")):
        axes[0].axvline(candidate, color=color, linestyle="--", linewidth=1.15,
                        label=f"候选目标时刻 {candidate:.1f} 秒")
    median_marker = float(np.median(marker_relative))
    axes[0].axvline(median_marker, color="#383838", linewidth=1.2,
                    label=f"第9通道标记中位数 {median_marker:.3f} 秒")
    axes[0].set_xlim(1.7, 2.6)
    axes[0].set_xlabel("相对视觉提示的时间（秒）")
    axes[0].set_ylabel("试次数")
    axes[0].set_title("实测事件标记与候选目标时刻")
    axes[0].grid(axis="y", color="#e5e5e5", linewidth=0.55)

    segments = (
        ("基线", -0.2, 0.0, "#B7C7D0"),
        ("提示阶段", 0.0, 0.8, "#79A4BF"),
        ("晚期阶段", 0.8, 2.8, "#A7C79B"),
        ("完整分析时段", -0.2, 3.0, None),
    )
    ax = axes[1]
    y_positions = [3.2, 2.3, 1.4, 0.5]
    for (label, start, stop, color), y in zip(segments, y_positions):
        if color is not None:
            ax.barh(y, stop - start, left=start, height=0.40, color=color, alpha=0.9)
        else:
            ax.plot([start, stop], [y, y], color="#333333", linewidth=1.25)
            ax.plot([start, start], [y - 0.14, y + 0.14], color="#333333", linewidth=1.0)
            ax.plot([stop, stop], [y - 0.14, y + 0.14], color="#333333", linewidth=1.0)
        if color is None:
            ax.text(
                start + 0.04, y + 0.20, label, ha="left", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.0},
            )
        else:
            ax.text(start - 0.05, y, label, ha="right", va="center", fontsize=8)
    for candidate, color in ((2.0, "#4C8C6B"), (2.2, "#D1843D"), (2.4, "#9B5C71")):
        ax.axvline(candidate, color=color, linestyle="--", linewidth=1.0)
        ax.axvspan(candidate, min(candidate + 0.6, 3.0), color=color, alpha=0.08)
    ax.axvline(median_marker, color="#383838", linewidth=1.0, alpha=0.8)
    ax.set_xlim(-0.45, 3.1)
    ax.set_ylim(0.0, 3.8)
    ax.set_yticks([])
    ax.set_xlabel("相对视觉提示的时间（秒）")
    ax.set_title("分析时间窗；目标周边窗口仅代表候选情景")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.55)
    fig.suptitle(
        f"审计 {len(events)} 个试次；提示至标记的中位间隔为 {median_marker:.3f} 秒。\n"
        "第9通道标记尚不能确认为目标出现时刻或真实反应时。",
        fontsize=10, y=0.99,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    _add_framed_figure_legend(
        fig, handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
        ncol=4, fontsize=7.5,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.12, hspace=0.42)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_audited_trials": int(len(events)),
        "n_marker_times": int(marker_relative.size),
        "marker_relative_median_s": median_marker,
        "marker_relative_q05_s": float(np.quantile(marker_relative, 0.05)),
        "marker_relative_q95_s": float(np.quantile(marker_relative, 0.95)),
    }


def _plot_observed_waveforms(
    observed_groups: dict[tuple[str, int], np.ndarray],
    preprocessing: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.8), sharex=True)
    group_values = np.stack(list(observed_groups.values()), axis=0)
    means = group_values.mean(axis=0)
    spread = group_values.std(axis=0, ddof=1) if group_values.shape[0] > 1 else np.zeros_like(means)
    colors = ("#39769C", "#4F906C", "#B66B49")
    for channel_index, (axis, channel, color) in enumerate(zip(axes, macro.CHANNELS, colors)):
        for values in group_values[:, channel_index, :]:
            axis.plot(OUTPUT_TIME_S, values, color="#9A9A9A", linewidth=0.45, alpha=0.32)
        axis.plot(OUTPUT_TIME_S, means[channel_index], color=color, linewidth=1.45,
                  label=f"{group_values.shape[0]} 个记录×提示方向组的均值")
        axis.fill_between(
            OUTPUT_TIME_S,
            means[channel_index] - spread[channel_index],
            means[channel_index] + spread[channel_index],
            color=color, alpha=0.15, linewidth=0,
            label="组间均值 ±1 个标准差",
        )
        axis.axhline(0.0, color="#777777", linewidth=0.55)
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.axvspan(0.0, 0.8, color="#79A4BF", alpha=0.06)
        axis.axvspan(0.8, 2.8, color="#A7C79B", alpha=0.05)
        for onset in (2.0, 2.2, 2.4):
            axis.axvline(onset, color="#A36849", linestyle=(0, (3, 3)), linewidth=0.65, alpha=0.75)
        axis.set_ylabel(f"{channel}\n脑电幅值（原始单位）")
        axis.grid(color="#e5e5e5", linewidth=0.5)
    axes[-1].set_xlabel("相对视觉提示的时间（秒）")
    axes[-1].set_xlim(TIME_START_S, TIME_STOP_S)
    fig.suptitle(
        "视觉提示对齐的三通道实测脑电波形\n"
        f"预处理：{_preprocessing_label(preprocessing)}；"
        f"带宽：{'未滤波' if preprocessing == 'none' else '0.5–30 赫兹'}；采样率 128 赫兹；"
        "基线窗：提示前 [-0.2, 0) 秒",
        fontsize=9.5, y=0.99,
    )
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color=colors[0], linewidth=1.5, label=f"{group_values.shape[0]} 个记录×提示方向组的均值"),
        Patch(facecolor=colors[0], alpha=0.15, edgecolor="none", label="组间均值 ±1 个标准差"),
    ]
    _add_framed_figure_legend(
        fig, handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=2, fontsize=7.5,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.10, hspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_states_and_contributions(exemplar: dict[str, Any], output_path: Path) -> None:
    basis = exemplar["basis"]
    fit = exemplar["fit"]
    contributions = sensor_contributions(
        fit, exemplar["q2"], exemplar["memory"], exemplar["control"],
        q2_control_interaction=exemplar["interaction"],
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), sharex="col")
    state_names = (("visual_state", "视觉状态 V", "#3D7193"),
                   ("memory_state", "记忆相关状态 H", "#C0843D"),
                   ("control_state", "控制状态 P", "#6B9270"))
    for axis, (key, title, color) in zip(axes[0], state_names):
        axis.plot(OUTPUT_TIME_S, basis[key], color=color, linewidth=1.35)
        axis.set_title(title)
        axis.set_ylabel("相对状态量")
        axis.grid(color="#e5e5e5", linewidth=0.5)
        axis.axvline(0.0, color="#333333", linewidth=0.7)
        if exemplar["scenario"][1] is not None:
            axis.axvline(exemplar["scenario"][1], color="#A36849", linestyle="--", linewidth=0.85)
    channel_colors = ("#39769C", "#4F906C", "#B66B49")
    for channel_index, (axis, channel, color) in enumerate(zip(axes[1], macro.CHANNELS, channel_colors)):
        axis.plot(OUTPUT_TIME_S, contributions["q2_visual"][0, channel_index],
                  color="#39769C", linewidth=1.0, label="问题二视觉输入")
        axis.plot(OUTPUT_TIME_S, contributions["memory"][0, channel_index],
                  color="#C0843D", linewidth=1.0, label="记忆状态成分")
        control_total = contributions["control_direct"][0, channel_index] + contributions["control_feedback"][0, channel_index]
        axis.plot(OUTPUT_TIME_S, control_total, color="#6B9270", linewidth=1.0, label="控制及交互成分")
        axis.plot(OUTPUT_TIME_S, exemplar["observed"][0, channel_index],
                  color="#333333", linewidth=0.8, alpha=0.7, label="留出实测脑电")
        axis.set_title(f"电极 {channel}：训练折估计的观测成分")
        axis.set_xlabel("相对视觉提示的时间（秒）")
        axis.set_ylabel("处理后脑电幅值")
        axis.grid(color="#e5e5e5", linewidth=0.5)
        axis.axvline(0.0, color="#333333", linewidth=0.7)
    target_type, onset_s, evidence = exemplar["scenario"]
    target_text = "无目标输入" if onset_s is None else f"候选目标时刻 {onset_s:.1f} 秒"
    evidence_text = "不设匹配分支" if onset_s is None else ("不匹配情景" if evidence < 0 else "匹配情景")
    fig.suptitle(
        "候选动态状态及其电极空间观测贡献\n"
        f"留出{_record_label(exemplar['record'])}／{('左侧' if exemplar['cue_side'] == 'left' else '右侧')}提示；"
        f"{_target_type_label(target_type)}，{target_text}，{evidence_text}；状态不代表解剖定位",
        fontsize=10, y=0.99,
    )
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#39769C", linewidth=1.1, label="问题二视觉输入"),
        Line2D([0], [0], color="#C0843D", linewidth=1.1, label="记忆状态成分"),
        Line2D([0], [0], color="#6B9270", linewidth=1.1, label="控制及交互成分"),
        Line2D([0], [0], color="#333333", linewidth=0.9, label="留出实测脑电"),
    ]
    _add_framed_figure_legend(
        fig, handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
        ncol=4, fontsize=7.5,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.13, wspace=0.52, hspace=0.30)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heldout_predictions(
    prediction_records: list[dict[str, Any]],
    nominal_onset_s: float | None,
    target_type: str,
    preprocessing: str,
    output_path: Path,
) -> None:
    from matplotlib.lines import Line2D

    selected = [
        row for row in prediction_records
        if row["preprocessing"] == preprocessing
        and row["target_stimulus_candidate"] == target_type
        and row["target_onset_candidate_s"] == nominal_onset_s
        and row["model"] in {"Q2_visual_only", "Q2_plus_memory"}
    ]
    if not selected:
        raise ValueError("no held-out predictions available for the requested illustration scenario")
    records = list(dict.fromkeys(row["heldout_record"] for row in selected))
    record_colors = {record: plt.get_cmap("tab10")(index) for index, record in enumerate(records)}
    model_styles = {
        "Measured held-out EEG": "-",
        "Q2_visual_only": "--",
        "Q2_plus_memory": ":",
    }
    curves: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        record_rows = [row for row in selected if row["heldout_record"] == record]
        evidence_values = list(dict.fromkeys(row["match_evidence_scenario"] for row in record_rows))
        evidence_values = sorted(evidence_values)
        observed_by_evidence: list[np.ndarray] = []
        predictions_by_model: dict[str, list[np.ndarray]] = {
            "Q2_visual_only": [],
            "Q2_plus_memory": [],
        }
        for evidence in evidence_values:
            for model_name in ("Q2_visual_only", "Q2_plus_memory"):
                rows = [
                    row for row in record_rows
                    if row["model"] == model_name
                    and row["match_evidence_scenario"] == evidence
                ]
                if not rows:
                    continue
                row = rows[0]
                if model_name == "Q2_visual_only":
                    observed_by_evidence.append(row["observed"].mean(axis=0))
                predictions_by_model[model_name].append(row["predicted"].mean(axis=0))
        if not observed_by_evidence:
            continue
        curves[record] = {
            "Measured held-out EEG": np.mean(observed_by_evidence, axis=0),
            "Q2_visual_only": np.mean(predictions_by_model["Q2_visual_only"], axis=0),
            "Q2_plus_memory": np.mean(predictions_by_model["Q2_plus_memory"], axis=0),
        }

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.8), sharex=True)
    for channel_index, (axis, channel) in enumerate(zip(axes, macro.CHANNELS)):
        for record in records:
            if record not in curves:
                continue
            color = record_colors[record]
            for model_name, linestyle in model_styles.items():
                axis.plot(
                    OUTPUT_TIME_S,
                    curves[record][model_name][channel_index],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.45 if model_name == "Measured held-out EEG" else 1.0,
                    alpha=0.95,
                )
        axis.axvline(0.0, color="#333333", linewidth=0.7)
        axis.axvspan(0.8, 2.8, color="#A7C79B", alpha=0.05)
        if nominal_onset_s is not None:
            axis.axvline(nominal_onset_s, color="#A36849", linestyle="--", linewidth=0.85)
        axis.set_ylabel(f"{channel}\n脑电幅值")
        axis.grid(color="#e5e5e5", linewidth=0.5)
    axes[-1].set_xlabel("相对视觉提示的时间（秒）")
    axes[-1].set_xlim(TIME_START_S, TIME_STOP_S)
    record_handles = [
        Line2D([0], [0], color=record_colors[record], linewidth=1.6, label=_record_label(record))
        for record in records if record in curves
    ]
    model_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.3, linestyle=linestyle, label=_model_label(model_name))
        for model_name, linestyle in model_styles.items()
    ]
    _add_framed_figure_legend(
        fig, handles=record_handles + model_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=4, fontsize=7.5,
    )
    fig.suptitle(
        "整份记录留出：实测脑电与模型预测对照\n"
        f"{_preprocessing_label(preprocessing)}；{_target_type_label(target_type)}候选目标时刻为{_format_onset_cn(nominal_onset_s)}； "
        "颜色表示不同留出记录，曲线对提示方向与匹配分支取均值",
        fontsize=10, y=0.99,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.10, hspace=0.12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _paired_memory_gains(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "preprocessing", "target_stimulus_candidate", "target_onset_candidate_s",
        "match_evidence_scenario", "heldout_record", "cue_side", "evaluation_window",
    ]
    base = metrics.loc[metrics["model"] == "Q2_visual_only", keys + ["nrmse_by_heldout_sd"]]
    memory = metrics.loc[metrics["model"] == "Q2_plus_memory", keys + ["nrmse_by_heldout_sd"]]
    paired = base.merge(memory, on=keys, suffixes=("_q2", "_memory"), validate="one_to_one")
    paired["memory_improvement_nrmse"] = paired["nrmse_by_heldout_sd_q2"] - paired["nrmse_by_heldout_sd_memory"]
    paired["memory_improvement_percent"] = 100.0 * paired["memory_improvement_nrmse"] / paired["nrmse_by_heldout_sd_q2"].replace(0.0, np.nan)
    return paired


def _plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    modes = list(dict.fromkeys(metrics["preprocessing"].tolist()))
    fig, axes = plt.subplots(1, len(modes), figsize=(4.3 * len(modes), 4.8), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    model_order = ("Training_mean_template", "Q2_visual_only", "Q2_plus_memory", "Q2_plus_memory_control")
    window_order = ("cue_stage", "late_stage")
    palette = {"cue_stage": "#91B4C4", "late_stage": "#D18A4A"}
    for axis, mode in zip(axes, modes):
        subset = metrics.loc[
            (metrics["preprocessing"] == mode)
            & metrics["target_onset_candidate_s"].notna()
            & (metrics["evaluation_window"].isin(window_order))
        ]
        positions = np.arange(len(model_order))
        width = 0.34
        for wi, window in enumerate(window_order):
            values = [
                subset.loc[(subset["model"] == model) & (subset["evaluation_window"] == window), "nrmse_by_heldout_sd"].mean()
                for model in model_order
            ]
            axis.bar(positions + (wi - 0.5) * width, values, width=width,
                     color=palette[window], label={"cue_stage": "早期提示阶段", "late_stage": "晚期认知阶段"}[window])
        axis.set_title(_preprocessing_label(mode))
        axis.set_xticks(positions, ["训练集\n均值模板", "问题二视觉\n模型", "问题二视觉+\n记忆", "问题二视觉+记忆+\n控制"], fontsize=8)
        axis.grid(axis="y", color="#e5e5e5", linewidth=0.5)
        axis.set_ylabel("平均留出归一化均方根误差")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("跨记录留出误差比较\n（汇总候选目标情景）", y=0.99)
    _add_framed_figure_legend(
        fig, handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=2, fontsize=7.5,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.79, bottom=0.22, wspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_target_sensitivity(metrics: pd.DataFrame, output_path: Path) -> None:
    paired = _paired_memory_gains(metrics)
    late = paired.loc[
        (paired["evaluation_window"] == "late_stage")
        & paired["target_onset_candidate_s"].notna()
    ]
    modes = list(dict.fromkeys(late["preprocessing"].tolist()))
    target_types = list(dict.fromkeys(late["target_stimulus_candidate"].tolist()))
    record_level = (
        late.groupby(
            ["preprocessing", "target_stimulus_candidate", "target_onset_candidate_s", "heldout_record"],
            as_index=False,
        )["memory_improvement_nrmse"].mean()
    )
    fig, axes = plt.subplots(1, len(modes), figsize=(4.4 * len(modes), 5.2), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    colors = ("#39769C", "#C0843D", "#6B9270")
    for axis, mode in zip(axes, modes):
        subset = record_level.loc[record_level["preprocessing"] == mode]
        for target_type, color in zip(target_types, colors):
            means = subset.loc[subset["target_stimulus_candidate"] == target_type].groupby("target_onset_candidate_s")["memory_improvement_nrmse"].mean()
            stds = subset.loc[subset["target_stimulus_candidate"] == target_type].groupby("target_onset_candidate_s")["memory_improvement_nrmse"].std()
            xs = means.index.to_numpy(dtype=float)
            ys = means.to_numpy(dtype=float)
            err = np.nan_to_num(stds.to_numpy(dtype=float), nan=0.0)
            axis.errorbar(xs, ys, yerr=err, marker="o", capsize=2.5,
                          linewidth=1.15, color=color, label=_target_type_label(target_type))
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=0.8)
        axis.set_title(_preprocessing_label(mode))
        axis.set_xlabel("候选目标时刻（相对提示，秒）")
        axis.grid(color="#e5e5e5", linewidth=0.5)
    axes[0].set_ylabel("记忆状态误差改善量\n（问题二视觉模型误差 − 问题二视觉＋记忆模型误差）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("不同候选目标时刻下记忆状态的晚期预测增益\n正值表示加入记忆后预测误差下降", y=0.99)
    _add_framed_figure_legend(
        fig, handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=3, fontsize=7.5,
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=0.74, bottom=0.18, wspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_float_list(value: str, *, allow_none: bool = False) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for part in value.split(","):
        token = part.strip().lower()
        if allow_none and token in {"none", "no-target", "no_target"}:
            parsed.append(None)
        elif token:
            parsed.append(float(token))
    if not parsed:
        raise ValueError("at least one comma-separated value is required")
    return tuple(parsed)


def _format_onset(value: float | None) -> str:
    return "no target" if value is None else f"{value:.1f} s"


def _summary_results(metrics: pd.DataFrame) -> dict[str, Any]:
    paired = _paired_memory_gains(metrics)
    late = paired.loc[paired["evaluation_window"] == "late_stage"]
    candidate_late = late.loc[late["target_onset_candidate_s"].notna()]
    record_onset_gain = (
        candidate_late.groupby(
            ["preprocessing", "target_onset_candidate_s", "heldout_record"],
            as_index=False,
        )["memory_improvement_nrmse"].mean()
    )
    mode_onset = (
        record_onset_gain.groupby(["preprocessing", "target_onset_candidate_s"])
        ["memory_improvement_nrmse"].mean()
        .reset_index()
    )
    positive = mode_onset.loc[mode_onset["memory_improvement_nrmse"] > 0]
    candidate_metrics = metrics.loc[
            metrics["target_onset_candidate_s"].notna()
            & (metrics["evaluation_window"] == "late_stage")
        ]
    record_model_scores = (
        candidate_metrics.groupby(
            ["preprocessing", "model", "heldout_record"], as_index=False
        )["nrmse_by_heldout_sd"].mean()
    )
    overall = (
        record_model_scores.groupby(["preprocessing", "model"])
        ["nrmse_by_heldout_sd"].agg(["mean", "std", "count"])
        .reset_index()
    )
    q2_mem = overall.loc[overall["model"] == "Q2_visual_only"].set_index("preprocessing")
    mem = overall.loc[overall["model"] == "Q2_plus_memory"].set_index("preprocessing")
    memory_delta_by_mode: dict[str, dict[str, float]] = {}
    for mode in sorted(set(q2_mem.index).intersection(mem.index)):
        q2_error = float(q2_mem.loc[mode, "mean"])
        memory_error = float(mem.loc[mode, "mean"])
        memory_delta_by_mode[mode] = {
            "q2_only_late_nrmse": q2_error,
            "q2_plus_memory_late_nrmse": memory_error,
            "memory_improvement_nrmse": q2_error - memory_error,
            "memory_improvement_percent": 100.0 * (q2_error - memory_error) / q2_error if q2_error > 0 else float("nan"),
        }
    return {
        "paired_memory_gain_late_stage": late,
        "candidate_memory_gain_late_stage": candidate_late,
        "memory_gain_by_preprocessing_and_onset": mode_onset,
        "positive_preprocessing_onset_cells": int(len(positive)),
        "total_preprocessing_onset_cells": int(len(mode_onset)),
        "late_model_metrics": overall,
        "memory_delta_by_preprocessing": memory_delta_by_mode,
    }


def _write_detailed_report(
    output_path: Path,
    event_summary: dict[str, Any],
    trial_audit: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: dict[str, Any],
    scenarios: tuple[macro.DynamicScenario, ...],
    figure_dir: Path,
    resolution: int,
    target_duration_s: float | None,
    q2_projection_max_abs_error: float,
) -> None:
    n_trials = int(len(trial_audit))
    n_included = int(trial_audit["included"].sum())
    n_records = int(trial_audit.loc[trial_audit["included"], "record"].nunique())
    onsets = sorted({scenario.target_onset_s for scenario in scenarios}, key=lambda x: (x is not None, -1.0 if x is None else x))
    target_types = sorted({scenario.target_stimulus for scenario in scenarios})
    preprocess_modes = sorted(set(metrics["preprocessing"]))
    duration_text = "持续到仿真结束" if target_duration_s is None else f"{target_duration_s:g} 秒（候选假设）"
    model_summary = summary["late_model_metrics"].copy()
    model_summary["preprocessing"] = model_summary["preprocessing"].map(_preprocessing_label)
    model_summary["model"] = model_summary["model"].map(_model_label)
    model_summary = model_summary.rename(columns={
        "preprocessing": "预处理方式",
        "model": "模型",
        "mean": "平均晚期误差",
        "std": "记录间标准差",
        "count": "留出记录数",
    })
    model_table = model_summary.to_markdown(index=False, floatfmt=".4f")
    onset_summary = summary["memory_gain_by_preprocessing_and_onset"].copy()
    onset_summary["preprocessing"] = onset_summary["preprocessing"].map(_preprocessing_label)
    onset_summary = onset_summary.rename(columns={
        "preprocessing": "预处理方式",
        "target_onset_candidate_s": "候选目标时刻（秒）",
        "memory_improvement_nrmse": "误差改善量",
    })
    onset_table = onset_summary.to_markdown(index=False, floatfmt=".4f")
    memory_lines = []
    for mode, result in summary["memory_delta_by_preprocessing"].items():
        memory_lines.append(
            f"| {_preprocessing_label(mode)} | {result['q2_only_late_nrmse']:.4f} | "
            f"{result['q2_plus_memory_late_nrmse']:.4f} | "
            f"{result['memory_improvement_nrmse']:+.4f} | "
            f"{result['memory_improvement_percent']:+.2f}% |"
        )
    memory_table = "\n".join(memory_lines) if memory_lines else "| 暂无 | | | | |"
    parameter_notes = {
        "tau_visual_s": ("s", "视觉驱动平滑时标；快速过程的结构性初值，未用实测 EEG 校准。"),
        "tau_memory_s": ("s", "记忆保持时标；表示 cue 到候选 target 的持续状态假设，未拟合。"),
        "tau_control_s": ("s", "控制态响应/衰减时标；结构性初值，未拟合。"),
        "cue_storage_gain": ("相对量", "提示阶段写入 H 的非负增益，初值 0.8。"),
        "target_memory_gain": ("相对量", "候选目标匹配证据更新 H 的增益，初值 0.8。"),
        "memory_to_control_gain": ("相对量", "H 驱动 P 的耦合增益，初值 0.25。"),
        "conflict_gain": ("相对量", "不匹配情景驱动 P 的增益，初值 0.8。"),
        "topdown_control_gain": ("相对量", "P 对 Q2 视觉传感器响应的调制增益，初值 0.10。"),
    }
    macro_parameter_table = pd.DataFrame([
        {
            "参数": name,
            "初始值": value,
            "单位/尺度": parameter_notes[name][0],
            "设计目的与依据": parameter_notes[name][1],
        }
        for name, value in asdict(macro.MacroParameters()).items()
    ]).to_markdown(index=False)
    beneficial_modes = [
        mode for mode, result in summary["memory_delta_by_preprocessing"].items()
        if result["memory_improvement_nrmse"] > 0
    ]
    if (
        summary["positive_preprocessing_onset_cells"] == summary["total_preprocessing_onset_cells"]
        and len(beneficial_modes) == len(summary["memory_delta_by_preprocessing"])
    ):
        memory_assessment = (
            "**本次判断：** 记忆态在所测预处理和候选时刻下均降低了留出晚期误差，"
            "结果支持继续检验该记忆机制；仍需在独立记录/受试者及核实事件后复验。"
        )
    else:
        mode_text = ", ".join(_preprocessing_label(mode) for mode in beneficial_modes) if beneficial_modes else "无"
        memory_assessment = (
            f"**本次判断：现有结果不支持记忆态稳定改善晚期 EEG 预测。** "
            f"只有 {summary['positive_preprocessing_onset_cells']}/"
            f"{summary['total_preprocessing_onset_cells']} 个预处理×目标时刻聚合格的误差改善为正；"
            f"整体改善只见于 {mode_text} 预处理，其他预处理下不改善或变差。"
            "状态方程已构建并完成留出检验，但当前样本与固定机制参数不足以支持稳定记忆态增益。"
        )
    document = r'''# 问题三实现流程与模型说明

## 1. 本轮结论与证据范围

本轮针对题目要求的“借助问题二所建立的脑电信号形成机制，建立认知宏观模型并用给定脑电信号验证”，构建了一个**可检验的动态认知宏观模型**：问题二的视觉处理与脑电正向形成链提供视觉驱动和三通道观测接口；问题三增加独立的记忆相关状态和控制状态，并以整条记录留出的脑电预测误差检验新增状态是否有增量信息。题目没有规定必须达到某个预测百分点；留出误差与候选时刻敏感性是本实现采用的验证证据。

事件审计文件给出 {event_summary['n_audited_trials']} 个视觉提示试次；第9通道响应标记相对视觉提示的中位时间为 {event_summary['marker_relative_median_s']:.4f} 秒，5%–95% 分位区间为 {event_summary['marker_relative_q05_s']:.4f}–{event_summary['marker_relative_q95_s']:.4f} 秒。第9通道只用于事件时间审计；目标呈现时间、正式反应时、正确性和漏答判定均未获独立核实，所以本轮不做行为标签拟合，也不把标记减去候选目标时刻称为真实反应时。

原始脑电中审计 {n_trials} 个视觉提示事件，按非有限值、原始幅度阈值与平直通道规则保留 {n_included} 个完整试次（覆盖 {n_records} 个记录文件）。主验证为**逐份记录留出**：每折留出一份完整记录，观测方程的电极系数和标准化参数只在另外三份训练记录中估计。该验证检验跨记录外推；记录文件未能确认独立受试者身份，因此不等同于严格的跨受试者验证。

主指标是留出脑电的归一化均方根误差（NRMSE，即 RMSE 除以留出实测值标准差），数值越低越好。晚期认知窗固定为视觉提示后 0.8–2.8 秒；它不由响应标记或单一目标时刻反推。候选刺激类型为 {', '.join(_target_type_label(x) for x in target_types)}；候选目标相对视觉提示时刻为 {', '.join(_format_onset_cn(x) for x in onsets)}；候选目标持续时间为 {duration_text}。这些仅用于敏感性分析，不是已确认的实验真值。

| 预处理 | Q2视觉基线晚期 NRMSE | 加记忆状态晚期 NRMSE | 误差改善（正数为改善） | 相对改善 |
|---|---:|---:|---:|---:|
{memory_table}
@@MEMORY_ASSESSMENT@@

在全部预处理×非空目标时刻的聚合格中，记忆状态晚期误差改善为正的格数是 {summary['positive_preprocessing_onset_cells']}/{summary['total_preprocessing_onset_cells']}。每种预处理、每个目标时刻的改善见下表；表内已对候选刺激类型、匹配/不匹配情景、留出记录和左右提示方向取平均，完整逐条件结果保存在 `output/dynamic_heldout_validation/dynamic_cv_metrics.csv`。

{onset_table}

**判断方法**：只有在留出晚期 EEG 的平均误差下降、且改善跨多个候选时刻与预处理口径保持同向时，数据才支持记忆状态对预测有可重复的增量信息。本次如未满足此条件，应如实写为“已建立并完成记录留出验证，但当前数据/机制参数尚未显示稳定的记忆态预测增益”，而不把模型结构本身当作机制证据。控制态增益另由 `Q2+memory+control` 与前两模型对照；它不改变记忆态的主要判断。

## 2. 输入、输出与模块边界

本轮动态模型的主运行顺序如下。先准备事件语义审计表，再运行主验证入口；Q2场景模拟、预处理、宏观状态积分、观测系数拟合、留出计分和出图都由主入口按顺序调用。

```powershell
python src/C/q3/09_event_time_semantics.py
python src/C/q3/dynamic_validation.py
```

特征分类、功率分析和静态路径分析脚本保留为辅助分析，不是下表动态状态的输入，也不参与留出评分。`dynamic_cognitive_model.py` 是被主入口调用的机制模块，不需要单独运行才能得到本报告结果。

| 模块/执行位置 | 输入 | 如何执行、目的与意义 | 主要输出 |
|---|---|---|---|
| 事件语义审计：`09_event_time_semantics.py` | VisCue、TimeStamp、第9通道边沿及 Q2 导联信息 | 按事件边沿汇总每试次时间，比较标记与候选目标时刻；检查事件映射和三电极对五源的可辨识性。用来确定可证事实与未知量，不把候选时刻强行定为真值 | `continuation_audit/event_timing_by_trial.csv` 及审计摘要 |
| 试次切片与质量控制：`dynamic_validation.py::_load_observed_epochs` | 原始连续 F3/Fz/F4、提示事件、时间戳 | 从连续记录提取提示起点，插值到共同时间网格；检查边界、重叠、非有限值、平直通道和幅值阈值。用途是确保各预处理分支使用同一批有效试次 | `observed_trial_audit.csv`、记录×提示方向均值 |
| 统一预处理：`dynamic_validation.py::_preprocess_*` | 连续实测脑电、Q2电极轨迹、状态轨迹与交互项 | 先对整条连续实测记录滤波，再切片；比较不滤波、单向因果滤波和离线零相位滤波；统一重采样到 128 赫兹并作提示前基线修正。用途是隔离预处理选择对结论的影响 | 三种预处理下可直接比较的 EEG 与模型基函数 |
| 问题二前向接口：`dynamic_cognitive_model.py::simulate_q2_scenario` | 提示方向、候选刺激形状/时刻、问题二参数和导联矩阵 G | 调用问题二视觉特征、LGN 与 Wilson–Cowan 皮层前向计算，形成五个源代理并经 G 正向投影至 F3/Fz/F4；不反演五源。用途是把问题二脑电形成机制接入问题三的视觉驱动与观测端 | 候选情景的视觉驱动、Q2电极轨迹、五源正向代理 |
| V/H/P 状态积分：`dynamic_cognitive_model.py::integrate_macro_states` | Q2早期视觉驱动、提示/目标门控、假设匹配证据 | 按各自时间常数递推 V、H、P；分别表达视觉驱动、记忆相关保持/匹配、控制/冲突过程。用途是建立随时间演化的不同认知过程，而非按三个电极给状态贴标签 | V/H/P 三条状态轨迹 |
| 电极观测与状态消融：`dynamic_validation.py::_run_record_heldout` | 训练记录的实测波形、Q2正向轨迹、V/H/P及交互项 | 每折只用训练记录估计标准化量和岭回归系数，逐步比较视觉、视觉＋记忆、视觉＋记忆＋控制，并以训练记录均值模板作简单基线。用途是检验状态是否带来跨记录预测增益 | 各电极的留出预测、拟合贡献和逐折误差 |
| 汇总、敏感性与出图：`dynamic_validation.py::run_validation` | 全部留出折分数、候选情景及事件审计表 | 按预处理与候选目标时刻汇总晚期误差，生成六张中文图和结果表；目标类型/时刻仅按预先列出的情景并列比较，不按留出误差挑选真值 | `dynamic_cv_metrics.csv`、`validation_summary.json`、`figures/*.png` |

## 3. 事件审计与有效窗口

### 3.1 Cue 对齐

每份连续记录从 VisCue 的零到非零边沿提取 cue 起点。令第 i 个 cue 的采样时间为 $t_i^c$，三通道实测 EEG 为 $\mathbf y_i(t)=[F3,Fz,F4]^T$。将数据插值到 $\tau\in[-0.2,3.0]$ s 的共同网格：

$$\mathbf y_i^c(\tau)=\operatorname{interp}\{\mathbf y(t_i^c+\tau)\},\qquad \Delta\tau=1/250\;\mathrm{s}.$$

若窗口超出记录边界、与下一 cue 重叠、包含非有限样本、绝对原始振幅达到 999.5 或任一电极在全窗内平直，则该试次在所有预处理分支中统一排除，并记录原因。窗口长度和抽样率对实测与模拟信号一致。

### 3.2 反应标记语义

审计能确认的只是每个 VisCue 区间中通道9有一次起始边沿。相对于 cue 的标记中位时间约为 {event_summary['marker_relative_median_s']:.4f} s；它与候选 cue+2.x 时刻过于接近，现有通道又没有独立的 target-onset 标记。故标记边沿保留为事件审计证据，不作为逐试次目标真值、真实反应时、正确错误、漏答或 EEG 预测特征。此处理避免用不确定行为标签监督模型，也避免事件泄漏。

### 3.3 阶段窗

基线窗 $[-0.2,0)$ s；早期 cue 阶段 $[0,0.8)$ s；晚期阶段 $[0.8,2.8)$ s。候选 target-onset 另评估 $[t_T,t_T+0.6)$ s（在 epoch 末端截断）。所有候选时间并列报告；不按验证误差挑选“最佳真实时刻”。

![事件标记与分析窗](output/dynamic_heldout_validation/figures/analysis_event_windows.png)

**图的目的与判读：** 上图把第9通道起始标记与 2.0、2.2、2.4 秒三个候选目标时刻放在同一时间轴上；下图标出基线、提示阶段、晚期阶段、完整切片及候选目标周边窗口。标记集中在约 2.215 秒，靠近候选时刻，说明现有记录不能单独判定该边沿是目标时刻还是应答时刻。该图用于限定解释范围，不用于把某一个候选时刻认定为真实实验事件。

![F3/Fz/F4认知阶段实测波形](output/dynamic_heldout_validation/figures/observed_three_channel_stages.png)

**图的目的与判读：** 该图展示所有保留记录×提示方向组在三个电极上的提示对齐波形、组均值及组间标准差，用于确认进入模型的实测信号在早期和晚期时间窗的尺度与差异。阴影是八个记录×提示方向组均值之间的标准差，不是受试者总体置信区间；原始采集单位未能独立校准到微伏。

## 4. Q2 前向机制接口

Q2 向 Q3 提供一个清楚的输入输出接口。cue 与候选 target 按候选时序组成连续刺激场景，经过 Q2 的空间视觉特征与 LGN 前端、Wilson–Cowan 兴奋/抑制群体状态以及五个源代理的正向导联投影：

$$\mathbf q_{Q2}(t)\in\mathbb R^5,\qquad \mathbf y_{Q2}(t)=G\mathbf q_{Q2}(t)\in\mathbb R^3.$$

$G$ 只用于五源到 F3/Fz/F4 的前向映射；$3\times5$ 逆问题不唯一，Q3 不做源反演。Q2 既有保存曲线只覆盖 cue-only 的早期片段，不能外推成完整认知试次。本实现按当前 Q2 方程对每种候选 target 重跑完整前向场景。由于 task 类型与文件名映射冲突，代码不将某个刺激类型绑定到某类真实试次。

Q2 早期视觉兴奋群体的空间均值作为宏观视觉过程的驱动：

$$u_{Q2,k}=\operatorname{mean}_{x,y}E_{\mathrm{early}}(x,y,k).$$

它是 Q2 网络中的视觉驱动量；后续状态不是将 F3/Fz/F4 分别重命名，而是独立的 V、H、P 过程。

每次候选情景模拟后还检查正向投影的一致性：矩阵乘法重算的三电极轨迹与 Q2 前向函数输出逐点对照，最大绝对差为 {q2_projection_max_abs_error:.3e}。这一项只证明程序采用了同一正向观测接口，不证明五个源代理可由三电极唯一识别。

## 5. 动态认知状态模型

设 $C_k\in[0,1]$ 和 $T_k\in[0,1]$ 分别为 cue 与候选 target 的时间门控，$r\in[-1,1]$ 为假设性的 match-evidence 情景量（正值表示匹配、负值表示不匹配；不是正确性标签）。使用指数稳定离散化 $\rho_j=\exp(-\Delta t/\tau_j)$：

$$V_{k+1}=\rho_VV_k+(1-\rho_V)u_{Q2,k},$$

$$H_{k+1}=\rho_HH_k+(1-\rho_H)\left(g_sV_kC_k+g_mV_kT_kr\right),$$

$$P_{k+1}=\rho_PP_k+(1-\rho_P)\left(g_{HP}|H_k|T_k+g_\delta T_k\frac{1-r}{2}\right).$$

其中 $V$ 是有时间常数的视觉驱动状态，$H$ 是 cue 写入并在候选 target 时受匹配证据调制的记忆相关状态，$P$ 是在候选 target 时由记忆强度及冲突情景驱动的控制状态。三者有不同状态方程和时间常数 $\tau_V,\tau_H,\tau_P$，因而不是按电极数量定义的三个标签。所有状态参数为机制构造阶段设定的有限正值，目前未通过 EEG 反演或行为结果估计；相关默认值记录在 `dynamic_cognitive_model.py` 的 `MacroParameters`。

![宏观状态与留出电极贡献](output/dynamic_heldout_validation/figures/dynamic_states_sensor_contributions.png)

**图的目的与判读：** 上排显示同一假设情景下 V、H、P 三个不同时间状态；下排按 Q2视觉、记忆直接项、控制直接项与控制×Q2反馈项分解训练折观测方程，并叠加留出实测波形。该图说明模型如何把不同状态映射到每个电极，可检查状态时程和贡献量级。下排系数只来自该折训练记录；这不是脑区定位，也不能证明海马或前额叶皮层来源。

## 6. 观测方程与预处理

每电极的观测模型将问题二正向 EEG 与宏观状态显式连接：

$$\hat y_{c,k}=a_c+b_{Q,c}\tilde y_{Q2,c,k}+\lambda_{H,c}\tilde H_k+\lambda_{P,c}\tilde P_k+\kappa_c\widetilde{y_{Q2,c,k}P_k}+\epsilon_{c,k}.$$

波浪号表示同一预处理算子后的量。$\lambda_H$ 与 $\lambda_P$ 是每电极的功能性传感器加载；$\kappa$ 表示控制态调制 Q2 视觉响应的交互。$a_c,b_{Q,c},\lambda_{H,c},\lambda_{P,c},\kappa_c$ 只用训练记录估计，并且按模型逐步加入：

1. `Q2_visual_only`：$a_c+b_{Q,c}\tilde y_{Q2,c}$，视觉机制基线；
2. `Q2_plus_memory`：在视觉基线上加 $\lambda_{H,c}\tilde H$，检验记忆相关状态的增量预测信息；
3. `Q2_plus_memory_control`：再加 $\lambda_{P,c}\tilde P+\kappa_c\widetilde{y_{Q2,c}P}$，检验控制状态及其对视觉响应的调制。

所有通道独立求解小规模 ridge 回归：

$$\hat\beta_c=\arg\min_{\beta}\|z_c-X_c\beta\|_2^2+\alpha\|\beta\|_2^2,\qquad \alpha={RIDGE_ALPHA:g}.$$

另加一个不含 Q2 状态的诊断基线：对每个留出 cue 侧，以训练记录同一 cue 侧的平均波形作为整条预测模板，$\hat{\mathbf y}_{mean,s}(t)=|R_{train,s}|^{-1}\sum_{r\in R_{train,s}}\bar{\mathbf y}_{r,s}(t)$。它用于检查复杂动态模型是否至少超过跨记录的简单模板；主要机制比较仍是 `Q2_visual_only` 与 `Q2_plus_memory`。

为避免在每个短 epoch 起点重置滤波器，实测 EEG 的 0.5–30 Hz 滤波先作用于整条连续记录，再依据 VisCue 切段；Q2 候选场景在完整模拟时间序列上应用相同带宽、滤波方向、重采样和 cue 前基线步骤。三种预处理均采用相同的 epoch 和模型比较窗口。零相位仍只作为离线敏感性分支。

回归特征在训练折内标准化，测试折沿用训练均值、尺度和系数，不重新校准。虽然观测加载在训练中可估计，但三个电极不足以建立唯一的神经源分解；它们用于验证状态是否有跨记录预测价值，不能作解剖解释。

共同预处理设置为四阶 Butterworth 0.5–30 Hz、128 Hz 输出采样率和 cue 前 [-0.2,0) s 基线修正。`causal` 是单向滤波；`zero_phase` 是离线零相位滤波，不能用于无未来数据的在线推断；`none` 用于检查滤波依赖性。每种算子同时用于实测 epoch、Q2 传感器轨迹及宏观状态/交互项，保证模型比较没有预处理口径差异。

## 7. 记录留出验证与模型比较

每一折留出一份完整记录；由其余记录的“记录×cue 侧均值”拟合观测系数。训练行覆盖相同完整 cue-locked 时间网格；验证只在留出记录单独打分。测试标签只有 EEG 本身，VisCue 侧仅用于选择对应 Q2 视觉输入；通道9没有进入 EEG 特征、潜在状态、回归、参数选择或指标计算。

对每个留出分组、模型、阶段窗报告：

$$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{n=1}^N(y_n-\hat y_n)^2},\qquad
\mathrm{NRMSE}=\frac{\mathrm{RMSE}}{\operatorname{SD}(y)},\qquad
\Delta_H=\mathrm{NRMSE}_{Q2}-\mathrm{NRMSE}_{Q2+H}.$$

$\Delta_H>0$ 表示加记忆状态降低该留出 EEG 窗的预测误差。对每个预处理×目标时刻组合，进一步跨候选刺激、匹配分支、四份留出记录及左右 cue 聚合；各组原始值并未被隐藏，见逐折 CSV。早期 cue 窗用于检查是否只有 cue 拟合改善；晚期窗是记忆增量的主评价窗；目标周边窗检查候选目标附近的动态响应。

![各模型的留出误差](output/dynamic_heldout_validation/figures/heldout_model_comparison.png)

**图的目的与判读：** 该图对比训练集均值模板、Q2视觉模型、加入记忆状态、再加入控制状态四种模型，并分开展示提示阶段和晚期阶段。它回答状态扩充是否降低完整留出记录上的误差；柱高是汇总均值，需结合下方候选时刻敏感性图和逐折 CSV 判断稳定性。

![不同目标时刻下的记忆态误差改善](output/dynamic_heldout_validation/figures/memory_target_time_sensitivity.png)

**图的目的与判读：** 该图检查记忆增益是否依赖某一个假设的目标出现时刻或某种预处理。横轴是 2.0、2.2、2.4 秒候选时刻，颜色区分候选刺激形状，点和误差线概括四份留出记录；正负号直接表示误差下降或上升。

![留出预测与实测波形对照](output/dynamic_heldout_validation/figures/heldout_predictions_vs_measured.png)

**图的目的与判读：** 每种颜色代表一份完整留出记录，实线为实测波形，不同线型区分 Q2视觉基线与加入记忆状态后的预测。该图只展示点阵刺激、约 2.2 秒候选目标和因果滤波下的代表性情景，并对左右提示与匹配分支取均值；它用于观察波形形态，不替代全部候选情景的误差统计。

## 8. 结果表与如何判读

{model_table}

上表先在每份留出记录内对刺激情景、匹配分支和左右 cue 侧取平均，再报告四份记录的均值、标准差和记录数（$n=4$）；情景和 cue 组是重复的敏感性条件，不作为独立样本数计算。

NRMSE 是每条留出组内按实测标准差归一化后计算，再跨留出组汇总；它不等于分类准确率，也不表示疾病诊断效能。应重点看：

- `Q2_plus_memory` 是否在晚期窗优于 `Q2_visual_only`；
- 改善是否在候选 target 2.0/2.2/2.4 s 下同向，而非只在单一假定时刻出现；
- 改善是否在 none/causal/zero-phase 三种预处理中大体同向；
- `Q2_plus_memory_control` 的额外收益是否跨留出记录存在，而不是只靠某个记录/电极。

如果这些条件不成立，结论就是现有样本不能支持新增状态提升晚期 EEG 预测；动态状态仍是已明确方程和可复核结果的机制假设，分类、ERP/功率和静态路径分析仅作辅助证据。行为关联本轮不报告，因为目标真值、正确性、截止时间与真实反应时未验证。

## 9. 中间产物、命令与复现

关键文件：

- `output/dynamic_heldout_validation/observed_trial_audit.csv`：每试次纳入状态和 QC 原因；
- `output/dynamic_heldout_validation/dynamic_cv_metrics.csv`：完整预处理×情景×模型×留出记录×cue 侧×阶段分数；
- `output/dynamic_heldout_validation/validation_summary.json`：汇总指标及事件语义限制；
- `output/dynamic_heldout_validation/figures/`：六张中文 PNG 中间图；
- Q2 前向数组只在当前运行期间保留于内存，不写出 `.npz` 缓存；下一次运行将重新计算候选场景。

默认复现：

```powershell
python src/C/q3/dynamic_validation.py
```

候选情景与预处理可单独配置；例如只比较两个目标时刻和因果/零相位处理：

```powershell
python src/C/q3/dynamic_validation.py --target-onsets 2.0,2.4 --target-types dots,inward --preprocessing causal,zero_phase --target-duration 0.2
```

## 10. 适用范围与当前不能声称的结论

1. 目标 onset、刺激与文件 task 的映射、trial correctness、omission 和实际 RT 仍未知；情景敏感性不能替代实验记录确认。
2. 四份文件作为四个留出域；在受试者身份未知时不能说结果经过独立受试者外部验证。
3. 宏观状态方程和时间常数当前是理论构造；拟合的是观测加载，不是对潜在过程的唯一识别。
4. Q2 的五源通过问题二导联矩阵做正向投影；三电极观测不能唯一反演五源，也不能证明某潜在状态来自海马或 PFC。
5. 现有任务数据不足以把已标准化的通道9边沿视作真实 RT 标签；正确性、漏答和行为—脑电关系保持未知。
6. 旧版分类、频带功率、ERP 与静态路径分析可作为辅助描述，不代替本报告的动态方程和留出预测检验。
'''
    replacements = {
        "{event_summary['n_audited_trials']}": str(event_summary["n_audited_trials"]),
        "{event_summary['marker_relative_median_s']:.4f}": f"{event_summary['marker_relative_median_s']:.4f}",
        "{event_summary['marker_relative_q05_s']:.4f}": f"{event_summary['marker_relative_q05_s']:.4f}",
        "{event_summary['marker_relative_q95_s']:.4f}": f"{event_summary['marker_relative_q95_s']:.4f}",
        "{n_trials}": str(n_trials),
        "{n_included}": str(n_included),
        "{n_records}": str(n_records),
        "{', '.join(target_types)}": ", ".join(target_types),
        "{', '.join(_format_onset(x) for x in onsets)}": ", ".join(_format_onset(x) for x in onsets),
        "{', '.join(_target_type_label(x) for x in target_types)}": ", ".join(_target_type_label(x) for x in target_types),
        "{', '.join(_format_onset_cn(x) for x in onsets)}": ", ".join(_format_onset_cn(x) for x in onsets),
        "{duration_text}": duration_text,
        "{q2_projection_max_abs_error:.3e}": f"{q2_projection_max_abs_error:.3e}",
        "{memory_table}": memory_table,
        "@@MEMORY_ASSESSMENT@@": memory_assessment,
        "{summary['positive_preprocessing_onset_cells']}": str(summary["positive_preprocessing_onset_cells"]),
        "{summary['total_preprocessing_onset_cells']}": str(summary["total_preprocessing_onset_cells"]),
        "{onset_table}": onset_table,
        "{model_table}": model_table,
        "{RIDGE_ALPHA:g}": f"{RIDGE_ALPHA:g}",
    }
    for token, value in replacements.items():
        document = document.replace(token, value)
    parameter_section = (
        "\n\n### 动态状态方程的参数初值\n\n"
        + macro_parameter_table
        + "\n\n上述时间常数与相对增益是可解释的构造初值，未由外层留出数据调参。"
        "模型结果不理想时不能利用外层留出分数反选这些参数；后续若要估计，"
        "应只在每个外层训练集内部执行内层验证。\n"
    )
    section_position = document.find("\n## 6.")
    if section_position >= 0:
        document = document[:section_position] + parameter_section + document[section_position:]
    reasons = [
        "**机制参数仍是构造值。** $\\tau_H$、写入/匹配增益及控制耦合尚未在训练记录的内层验证中估计；若真实保持时长或更新时序不同，当前 H 的时间轨迹会错位。",
        "**事件与刺激类型不确定。** 目标出现时刻没有独立标记，Task-1/Task-2 与 Q2 项目条件的映射未确认；把全部候选情景汇总会稀释某一真实但未知的条件效应。",
        "**记录间差异明显。** 四份记录是目前可用的留出单位，但独立受试者身份未知；传感器 EEG 的尺度和慢变形态可能存在记录间偏移。当前 NRMSE 大于 1，且预测波形未能跟随全部实测缓慢起伏，说明绝对拟合仍弱。",
        f"**可用试次数有限。** 原始幅度 QC 排除了 {n_trials - n_included}/{n_trials} 个 epoch；其余约 {n_included} 个有效 trial 不能补足目标真值、正确性、漏答和正式反应时标签。",
        "**预处理影响了误差水平。** 因果与零相位过滤下的整体 NRMSE 不同，而记忆态改善也未跨三种预处理一致；零相位仅是离线敏感性结果，不能当作实时 BCI 性能。",
        "**观测方程仍较简化。** 只使用 F3/Fz/F4 的线性加载和单一控制×Q2 交互，未覆盖其它脑区、电极或被试特异的噪声/参考方式；三电极无法据此确认海马或 PFC 来源。",
    ]
    reason_text = "\n".join(f"{index}. {reason}" for index, reason in enumerate(reasons, start=1))
    document += (
        "\n\n## 11. 本次效果有限的可能原因（推测）\n\n"
        "以下解释是根据本轮设置与留出结果提出的待检验假设，不是由当前数据单独证明的因果结论：\n\n"
        f"{reason_text}\n\n"
        "### 后续优先改进顺序\n\n"
        "1. 先取得 target-onset、刺激真值/类型与通道9事件写入说明，厘清每个文件对应的任务条件；更新事件图和候选网格。\n"
        "2. 冻结连续记录滤波、重采样、伪迹 QC 与阶段窗；对被排除的 45 个试次检查是否为真实硬件饱和或可修复的数据格式问题，不为增加样本而放宽阈值。\n"
        "3. 事件语义确认后，在每个外层留出折的训练记录内用内层验证估计 $\\tau_H$、增益和观测加载；外层留出记录仍只用于最终一次评估，禁止按外层分数挑参数。\n"
        "4. 增加经验证的 target-locked 波形/状态对照，并在标签可用时单独分析 RT、正确性或实际选择；通道9继续仅作事件/行为变量，不进入 EEG 特征。\n"
        "5. 若跨记录与事件敏感性后仍没有稳定的晚期预测增益，应保留动态模型作为可复核的机制假设，并明确写出现有数据不足以支持记忆机制。\n"
    )
    output_path.write_text(document, encoding="utf-8")


def run_validation(
    output_dir: Path = OUTPUT_DIR,
    *,
    target_types: tuple[str, ...] = DEFAULT_TARGET_TYPES,
    target_onsets_s: tuple[float | None, ...] = DEFAULT_ONSETS,
    target_duration_s: float | None = 0.2,
    match_evidence_values: tuple[float, ...] = DEFAULT_MATCH_EVIDENCE,
    preprocessing_modes: tuple[str, ...] = DEFAULT_PREPROCESSING,
    resolution: int = 128,
    simulation_stop_s: float = TIME_STOP_S,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    removed_caches = _remove_stale_q2_caches(output_dir)
    if not preprocessing_modes or not set(preprocessing_modes).issubset(set(macro.FILTER_MODES)):
        raise ValueError("preprocessing_modes must be a nonempty subset of none/causal/zero_phase")
    if simulation_stop_s < TIME_STOP_S:
        raise ValueError("simulation_stop_s must cover the full EEG validation window")
    if target_types[0] not in macro.TARGET_TYPES:
        raise ValueError("target types must be a nonempty subset of Q2-supported candidate stimuli")
    if not set(target_types).issubset(set(macro.TARGET_TYPES)):
        raise ValueError("target types must be a subset of dots/inward/outward")
    scenarios = _scenario_grid(
        tuple(target_types), tuple(target_onsets_s), target_duration_s,
        tuple(float(x) for x in match_evidence_values),
    )
    observed, trial_audit = _load_observed_epochs(preprocessing_modes)
    if trial_audit.empty or not trial_audit["included"].any():
        raise RuntimeError("no usable cue-locked EEG epochs remain after QC")
    trial_audit.to_csv(output_dir / "observed_trial_audit.csv", index=False, encoding="utf-8-sig")
    available_records = sorted(trial_audit.loc[trial_audit["included"], "record"].unique())
    if len(available_records) < 2:
        raise RuntimeError("record-held-out validation requires at least two usable records")
    expected_groups = {
        (record, side)
        for record in available_records
        for side in (-1, 1)
    }
    for mode in observed:
        for key in expected_groups:
            if key not in observed[mode]:
                print(f"WARNING: no retained {mode} EEG trials for {key}", flush=True)

    event_summary = _plot_event_windows(EVENT_AUDIT_PATH, figure_dir / "analysis_event_windows.png")
    observed_mode = "causal" if "causal" in preprocessing_modes else preprocessing_modes[0]
    _plot_observed_waveforms(
        observed[observed_mode], observed_mode,
        figure_dir / "observed_three_channel_stages.png",
    )

    basis_lookup: dict[tuple[str, str, str, float | None, float], dict[str, Any]] = {}
    base_cache: dict[tuple[str, str, float | None], dict[str, Any]] = {}
    unique_inputs = sorted(
        {(scenario.cue_side, scenario.target_stimulus, scenario.target_onset_s) for scenario in scenarios},
        key=lambda item: (item[0], item[2] is not None, -1.0 if item[2] is None else item[2], item[1]),
    )
    for index, (cue_side, target_type, onset_s) in enumerate(unique_inputs, start=1):
        scenario = next(
            scenario for scenario in scenarios
            if scenario.cue_side == cue_side
            and scenario.target_stimulus == target_type
            and scenario.target_onset_s == onset_s
        )
        print(
            f"Q2 forward scenario {index}/{len(unique_inputs)}: "
            f"{cue_side}, {target_type}, target={_format_onset(onset_s)}",
            flush=True,
        )
        base = _simulate_q2_base(scenario, resolution, simulation_stop_s)
        base_cache[(cue_side, target_type, onset_s)] = base
        for mode in preprocessing_modes:
            cfg = macro.PreprocessingConfig(
                mode=mode, low_hz=0.5, high_hz=30.0, order=4,
                target_fs_hz=OUTPUT_RATE_HZ,
                baseline_window_s=(TIME_START_S, 0.0),
            )
            for evidence in sorted({
                item.match_evidence for item in scenarios
                if item.cue_side == cue_side
                and item.target_stimulus == target_type
                and item.target_onset_s == onset_s
            }):
                candidate = macro.DynamicScenario(
                    cue_side, target_type, onset_s, target_duration_s, float(evidence)
                )
                basis_lookup[(mode, cue_side, target_type, onset_s, float(evidence))] = _make_processed_basis(base, candidate, cfg)

    metrics, predictions, exemplar = _run_record_heldout(observed, scenarios, basis_lookup)
    if metrics.empty:
        raise RuntimeError("record-held-out validation produced no score rows")
    trial_counts = (
        trial_audit.loc[trial_audit["included"]]
        .groupby(["record", "cue_side"])
        .size()
        .to_dict()
    )
    metrics["n_test_trials_in_cue_group"] = [
        trial_counts.get((row.heldout_record, -1 if row.cue_side == "left" else 1), 0)
        for row in metrics.itertuples(index=False)
    ]
    metrics.to_csv(output_dir / "dynamic_cv_metrics.csv", index=False, encoding="utf-8-sig")
    if exemplar is None:
        raise RuntimeError("no 2.2 s dots/mismatch causal fold available for state-contribution plot")
    _plot_states_and_contributions(
        exemplar, figure_dir / "dynamic_states_sensor_contributions.png"
    )

    candidate_onsets = [value for value in target_onsets_s if value is not None]
    nominal_onset = min(candidate_onsets, key=lambda value: abs(value - 2.2)) if candidate_onsets else None
    if nominal_onset is None:
        nominal_type = target_types[0]
    else:
        nominal_type = "dots" if "dots" in target_types else target_types[0]
    _plot_heldout_predictions(
        predictions, nominal_onset, nominal_type, observed_mode,
        figure_dir / "heldout_predictions_vs_measured.png",
    )
    _plot_model_comparison(metrics, figure_dir / "heldout_model_comparison.png")
    _plot_target_sensitivity(metrics, figure_dir / "memory_target_time_sensitivity.png")

    summary = _summary_results(metrics)
    summary_json = {
        "n_trials_audited": int(len(trial_audit)),
        "n_trials_included": int(trial_audit["included"].sum()),
        "n_records_included": int(trial_audit.loc[trial_audit["included"], "record"].nunique()),
        "preprocessing_modes": list(preprocessing_modes),
        "target_types_candidate_only": list(target_types),
        "target_onsets_s_candidate_only": list(target_onsets_s),
        "target_duration_s_assumption": target_duration_s,
        "match_evidence_scenarios_not_labels": list(match_evidence_values),
        "resolution": resolution,
        "simulation_stop_s": simulation_stop_s,
        "q2_projection_max_abs_error": float(metrics["heldout_record"].notna().sum() and max(base["projection_max_abs_error"] for base in base_cache.values())),
        "late_stage_nrmse_by_model_and_preprocessing": summary["late_model_metrics"].to_dict(orient="records"),
        "memory_improvement_by_preprocessing": summary["memory_delta_by_preprocessing"],
        "positive_preprocessing_onset_cells": summary["positive_preprocessing_onset_cells"],
        "total_preprocessing_onset_cells": summary["total_preprocessing_onset_cells"],
        "event_marker_summary": event_summary,
        "target_onset_status": "candidate sensitivity only; no independent target marker",
        "behavioral_labels_used": False,
        "channel9_used_in_eeg_or_model": False,
        "split": "leave one complete record file out; participant independence unverified",
        "ridge_alpha": RIDGE_ALPHA,
        "macro_parameters": asdict(macro.MacroParameters()),
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path = Q3_DIR / "问题三实现流程与模型说明.md"
    _write_detailed_report(
        report_path, event_summary, trial_audit, metrics, summary,
        scenarios, figure_dir, resolution, target_duration_s,
        max(base["projection_max_abs_error"] for base in base_cache.values()),
    )
    # Keep figures PNG-only; Q2 simulation arrays are held in memory and are
    # never written as persistent NPZ cache files.
    for stale in figure_dir.iterdir():
        if stale.is_file() and stale.suffix.lower() != ".png":
            stale.unlink()
    if removed_caches:
        print(f"Removed stale Q2 NPZ caches: {removed_caches}", flush=True)
    print(f"Included EEG epochs: {int(trial_audit['included'].sum())}/{len(trial_audit)}", flush=True)
    print(f"Late-stage model metrics saved: {output_dir / 'dynamic_cv_metrics.csv'}", flush=True)
    print(f"Detailed method/results note: {report_path}", flush=True)
    print(f"PNG figures only: {figure_dir}", flush=True)
    for mode, result in summary["memory_delta_by_preprocessing"].items():
        print(
            f"{mode}: late NRMSE Q2={result['q2_only_late_nrmse']:.4f}, "
            f"Q2+memory={result['q2_plus_memory_late_nrmse']:.4f}, "
            f"improvement={result['memory_improvement_percent']:+.2f}%",
            flush=True,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target-types", default=",".join(DEFAULT_TARGET_TYPES))
    parser.add_argument("--target-onsets", default="none,2.0,2.2,2.4")
    parser.add_argument("--target-duration", default="0.2", help="seconds, or 'until-end'")
    parser.add_argument("--match-evidence", default="-1,1")
    parser.add_argument("--preprocessing", default=",".join(DEFAULT_PREPROCESSING))
    parser.add_argument("--resolution", type=int, choices=(64, 128), default=128)
    parser.add_argument("--simulation-stop", type=float, default=TIME_STOP_S)
    args = parser.parse_args()
    duration = None if args.target_duration.lower() in {"none", "until-end"} else float(args.target_duration)
    target_types = tuple(item.strip() for item in args.target_types.split(",") if item.strip())
    preprocessing = tuple(item.strip() for item in args.preprocessing.split(",") if item.strip())
    run_validation(
        args.output_dir,
        target_types=target_types,
        target_onsets_s=_parse_float_list(args.target_onsets, allow_none=True),
        target_duration_s=duration,
        match_evidence_values=tuple(float(x) for x in _parse_float_list(args.match_evidence)),
        preprocessing_modes=preprocessing,
        resolution=args.resolution,
        simulation_stop_s=args.simulation_stop,
    )


if __name__ == "__main__":
    main()
