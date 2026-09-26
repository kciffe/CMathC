"""Minimal, scenario-driven Q3 dynamic model linked to the Q2 forward chain.

This is a mechanism and numerical-check build, not a fit to measured EEG.
Target timing, target type, match evidence, and the observation preprocessing
are explicit scenario inputs because their trial-level meanings are not yet
verified in the supplied MAT records.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.C.q2.revision_v3 import config as q2_config
from src.C.q2.revision_v3.frontend import load_stimulus, simulate_frontend
from src.C.q2.revision_v3.model import ModelParams, simulate_forward


OUTPUT_DIR = REPO_ROOT / "src" / "C" / "q3" / "output" / "dynamic_cognitive_model"
FIGURE_DIR = OUTPUT_DIR / "figures"
CHANNELS = ("F3", "Fz", "F4")
Q2_RATE_HZ = 250.0  # Q2's selected 4 ms mechanistic time grid.
Q2_STEP_MS = 1000.0 / Q2_RATE_HZ
SIMULATION_START_S = -0.2
SIMULATION_STOP_S = 3.0
CUE_DURATION_S = 0.203125  # Median measured VisCue pulse duration; cue onset is the anchor.
TARGET_ONSETS_S = (2.0, 2.2, 2.4)  # Schedule sensitivity only, not selected target-onset truth.
TARGET_TYPES = ("dots", "inward", "outward")  # Candidate stimuli; record/task mapping is unresolved.
MATCH_EVIDENCE_VALUES = (1.0, -1.0)  # Synthetic match/mismatch branches, never behavioral labels.
FILTER_MODES = ("none", "causal", "zero_phase")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DengXian", "Arial"],
    "axes.unicode_minus": False,
})


@dataclass(frozen=True)
class MacroParameters:
    """Nominal functional dynamics used only for the unfit mechanism check."""

    tau_visual_s: float = 0.060
    tau_memory_s: float = 1.0
    tau_control_s: float = 0.250
    cue_storage_gain: float = 0.8
    target_memory_gain: float = 0.8
    memory_to_control_gain: float = 0.25
    conflict_gain: float = 0.8
    topdown_control_gain: float = 0.10


@dataclass(frozen=True)
class PreprocessingConfig:
    """A selectable common observation operator for model and measured EEG."""

    mode: str = "causal"
    low_hz: float = 0.5
    high_hz: float = 30.0
    order: int = 4
    target_fs_hz: float = 128.0
    baseline_window_s: tuple[float, float] = (-0.2, 0.0)


@dataclass(frozen=True)
class DynamicScenario:
    cue_side: str
    target_stimulus: str
    target_onset_s: float | None
    target_duration_s: float | None
    match_evidence: float


def build_sensitivity_scenarios(
    cue_side: str = "left",
    target_types: tuple[str, ...] = TARGET_TYPES,
    target_onsets_s: tuple[float | None, ...] = TARGET_ONSETS_S,
    target_duration_s: float | None = 0.2,
    match_evidence_values: tuple[float, ...] = MATCH_EVIDENCE_VALUES,
) -> tuple[DynamicScenario, ...]:
    """Build an explicit Cartesian grid of candidate, not observed, scenarios."""

    target_types = tuple(target_types)
    target_onsets_s = tuple(target_onsets_s)
    match_evidence_values = tuple(float(x) for x in match_evidence_values)
    if cue_side not in {"left", "right"}:
        raise ValueError("cue_side must be left or right")
    if not target_types or not set(target_types).issubset({"dots", "inward", "outward"}):
        raise ValueError("target_types must be a nonempty subset of dots/inward/outward")
    if not target_onsets_s:
        raise ValueError("target_onsets_s must contain at least one candidate")
    if any(value is not None and (not np.isfinite(value) or value < 0.0)
           for value in target_onsets_s):
        raise ValueError("target onset candidates must be nonnegative finite values or None")
    if not match_evidence_values or any(
        not np.isfinite(value) or not -1.0 <= value <= 1.0
        for value in match_evidence_values
    ):
        raise ValueError("match evidence candidates must lie in [-1, 1]")
    if target_duration_s is not None and (
        not np.isfinite(target_duration_s) or target_duration_s <= 0.0
    ):
        raise ValueError("target_duration_s must be positive, finite, or None")
    return tuple(
        DynamicScenario(cue_side, target_type, onset, target_duration_s, evidence)
        for target_type in target_types
        for onset in target_onsets_s
        for evidence in match_evidence_values
    )


def _validate_gate(name: str, gate: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(gate, dtype=float).reshape(-1)
    if value.size != size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector matching the time axis")
    if np.any((value < 0.0) | (value > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return value


def make_event_gates(
    time_s: np.ndarray,
    cue_onset_s: float = 0.0,
    cue_duration_s: float = CUE_DURATION_S,
    target_onset_s: float | None = 2.2,
    target_duration_s: float | None = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create configurable cue/target gates; target None means no target input."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    if time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not np.isfinite(cue_onset_s) or not np.isfinite(cue_duration_s) or cue_duration_s <= 0:
        raise ValueError("cue onset and positive duration must be finite")
    cue = ((time >= cue_onset_s) & (time < cue_onset_s + cue_duration_s)).astype(float)
    target = np.zeros_like(cue)
    if target_onset_s is None:
        return cue, target
    if not np.isfinite(target_onset_s):
        raise ValueError("target_onset_s must be finite or None")
    if target_duration_s is None:
        target = (time >= target_onset_s).astype(float)
    else:
        if not np.isfinite(target_duration_s) or target_duration_s <= 0:
            raise ValueError("target_duration_s must be positive, finite, or None")
        target = (
            (time >= target_onset_s)
            & (time < target_onset_s + target_duration_s)
        ).astype(float)
    return cue, target


def integrate_macro_states(
    visual_drive: np.ndarray,
    cue_gate: np.ndarray,
    target_gate: np.ndarray,
    match_evidence: float,
    dt_s: float,
    params: MacroParameters | None = None,
) -> dict[str, np.ndarray]:
    """Integrate separate visual, memory, and control functional states.

    Equations use a stable exponential Euler update:

      V[k+1] = rho_v V[k] + (1-rho_v) U_Q2[k]
      H[k+1] = rho_h H[k] + (1-rho_h) (g_store V[k] cue[k]
                 + g_match V[k] target[k] match_evidence)
      P[k+1] = rho_p P[k] + (1-rho_p) (g_hp |H[k]| target[k]
                 + g_conflict target[k] (1-match_evidence)/2)

    ``match_evidence`` is a hypothetical scenario value in [-1, 1], not a
    correctness label. V is driven by Q2's early cortical population state.
    """

    p = MacroParameters() if params is None else params
    drive = np.asarray(visual_drive, dtype=float).reshape(-1)
    if drive.size < 2 or not np.isfinite(drive).all() or np.any(drive < 0.0):
        raise ValueError("visual_drive must be a finite nonnegative time series")
    cue = _validate_gate("cue_gate", cue_gate, drive.size)
    target = _validate_gate("target_gate", target_gate, drive.size)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    if not np.isfinite(match_evidence) or not -1.0 <= match_evidence <= 1.0:
        raise ValueError("match_evidence must be a scenario value in [-1, 1]")
    timescales = (p.tau_visual_s, p.tau_memory_s, p.tau_control_s)
    gains = (
        p.cue_storage_gain, p.target_memory_gain, p.memory_to_control_gain,
        p.conflict_gain, p.topdown_control_gain,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in timescales):
        raise ValueError("all macro-state time constants must be positive and finite")
    if not all(np.isfinite(value) and value >= 0.0 for value in gains):
        raise ValueError("all macro-state gains must be finite and nonnegative")

    rho_v, rho_h, rho_p = (float(np.exp(-dt_s / tau)) for tau in timescales)
    visual = np.zeros_like(drive)
    memory = np.zeros_like(drive)
    control = np.zeros_like(drive)
    visual[0] = drive[0]
    for k in range(1, drive.size):
        visual[k] = rho_v * visual[k - 1] + (1.0 - rho_v) * drive[k]
        memory_input = (
            p.cue_storage_gain * visual[k] * cue[k]
            + p.target_memory_gain * visual[k] * target[k] * match_evidence
        )
        memory[k] = rho_h * memory[k - 1] + (1.0 - rho_h) * memory_input
        conflict_input = target[k] * (1.0 - match_evidence) / 2.0
        control_input = (
            p.memory_to_control_gain * abs(memory[k - 1]) * target[k]
            + p.conflict_gain * conflict_input
        )
        control[k] = rho_p * control[k - 1] + (1.0 - rho_p) * control_input
    return {"visual": visual, "memory": memory, "control": control}


def observe_macro_model(
    q2_sensor: np.ndarray,
    memory: np.ndarray,
    control: np.ndarray,
    memory_loading: np.ndarray,
    control_loading: np.ndarray,
    topdown_control_gain: float,
) -> np.ndarray:
    """Map Q2 visual EEG plus functional memory/control states to 3 sensors.

    y_hat[k] = (1 + beta_pc P[k]) G q_Q2[k]
                + l_H H[k] + l_P P[k]

    The loading vectors are functional sensor-space scenarios, not anatomical
    source leadfields. The Q2 leadfield is only used in its forward direction.
    """

    q2 = np.asarray(q2_sensor, dtype=float)
    h = np.asarray(memory, dtype=float).reshape(-1)
    p = np.asarray(control, dtype=float).reshape(-1)
    l_h = np.asarray(memory_loading, dtype=float).reshape(-1)
    l_p = np.asarray(control_loading, dtype=float).reshape(-1)
    if q2.ndim != 2 or q2.shape[0] != 3 or q2.shape[1] != h.size or h.size != p.size:
        raise ValueError("q2_sensor, memory, and control must align as [3,time]")
    if l_h.size != 3 or l_p.size != 3:
        raise ValueError("memory and control loadings must each contain F3/Fz/F4 weights")
    if not all(np.isfinite(x).all() for x in (q2, h, p, l_h, l_p)):
        raise ValueError("observation inputs must be finite")
    if not np.isfinite(topdown_control_gain) or topdown_control_gain < 0.0:
        raise ValueError("topdown_control_gain must be finite and nonnegative")
    if np.any(1.0 + topdown_control_gain * p < 0.0):
        raise ValueError("top-down gain would invert the Q2 visual sensor signal")
    return (
        q2 * (1.0 + topdown_control_gain * p)[None, :]
        + l_h[:, None] * h[None, :]
        + l_p[:, None] * p[None, :]
    )


def apply_preprocessing(
    signal: np.ndarray,
    time_s: np.ndarray,
    source_fs_hz: float,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply selectable filtering, resampling, and baseline to sensor traces."""

    cfg = PreprocessingConfig() if config is None else config
    values = np.asarray(signal, dtype=float)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] != time.size:
        raise ValueError("signal must be [F3/Fz/F4,time] and match time_s")
    if time.size < 32 or not np.isfinite(values).all() or not np.isfinite(time).all():
        raise ValueError("signal and time must be finite and contain at least 32 samples")
    if np.any(np.diff(time) <= 0.0) or not np.isfinite(source_fs_hz) or source_fs_hz <= 0:
        raise ValueError("time must increase and source sampling rate must be positive")
    if not np.allclose(np.diff(time), 1.0 / source_fs_hz, rtol=1e-4, atol=1e-8):
        raise ValueError("time axis step disagrees with source_fs_hz")
    if cfg.mode not in {"none", "causal", "zero_phase"}:
        raise ValueError("preprocessing mode must be none, causal, or zero_phase")
    if cfg.order < 1 or cfg.target_fs_hz <= 0.0:
        raise ValueError("filter order and target sample rate must be positive")

    processed = values.copy()
    if cfg.mode != "none":
        if not 0.0 < cfg.low_hz < cfg.high_hz < source_fs_hz / 2.0:
            raise ValueError("band edges must lie strictly below source Nyquist")
        sos = butter(
            int(cfg.order), (cfg.low_hz, cfg.high_hz), btype="bandpass",
            fs=float(source_fs_hz), output="sos",
        )
        if cfg.mode == "causal":
            processed = sosfilt(sos, processed, axis=-1)
        else:
            processed = sosfiltfilt(sos, processed, axis=-1)

    ratio = Fraction(float(cfg.target_fs_hz) / float(source_fs_hz)).limit_denominator(10000)
    processed = resample_poly(processed, ratio.numerator, ratio.denominator, axis=-1)
    output_time = time[0] + np.arange(processed.shape[-1], dtype=float) / cfg.target_fs_hz
    keep = output_time <= time[-1] + 0.5 / cfg.target_fs_hz
    processed = processed[:, keep]
    output_time = output_time[keep]
    baseline_start, baseline_stop = cfg.baseline_window_s
    if not baseline_start < baseline_stop:
        raise ValueError("baseline_window_s must have positive duration")
    baseline = (output_time >= baseline_start) & (output_time < baseline_stop)
    if int(baseline.sum()) < 2:
        raise ValueError("baseline window must contain at least two output samples")
    processed -= processed[:, baseline].mean(axis=1, keepdims=True)
    return processed, output_time


def _q2_drive_scales() -> np.ndarray:
    path = q2_config.OUTPUT_ROOT / "drive_scales.csv"
    if not path.exists():
        raise FileNotFoundError(f"Q2 fixed drive-scale calibration is required: {path}")
    order = (
        ("early", "left_visual_field"), ("early", "right_visual_field"),
        ("configuration", "left_visual_field"), ("configuration", "right_visual_field"),
        ("shape_preference", "left_triangle_template_preference"),
        ("shape_preference", "right_triangle_template_preference"),
    )
    rows = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            rows[(row["population"], row["channel"])] = float(row["reference_rms_scale"])
    missing = set(order).difference(rows)
    if missing:
        raise ValueError(f"Q2 drive-scale table is missing entries: {sorted(missing)}")
    return np.asarray([rows[key] for key in order], dtype=float).reshape(3, 2)


def simulate_q2_scenario(
    scenario: DynamicScenario,
    macro_params: MacroParameters | None = None,
    q2_params: ModelParams | None = None,
    resolution: int = 128,
    dt_ms: float = Q2_STEP_MS,
    simulation_stop_s: float = SIMULATION_STOP_S,
) -> dict[str, Any]:
    """Run Q2 visual mechanisms over one configurable cue/target sequence."""

    if scenario.cue_side not in {"left", "right"}:
        raise ValueError("cue_side must be left or right")
    if scenario.target_stimulus not in {"dots", "inward", "outward"}:
        raise ValueError("target_stimulus must be dots, inward, or outward")
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError("dt_ms must be positive and finite")
    if not np.isfinite(simulation_stop_s) or simulation_stop_s <= 0.2:
        raise ValueError("simulation_stop_s must be positive and extend past cue onset")

    time_ms = np.arange(
        SIMULATION_START_S * 1000.0,
        simulation_stop_s * 1000.0 + 0.5 * dt_ms,
        dt_ms,
        dtype=float,
    )
    time_s = time_ms / 1000.0
    cue_gate, target_gate = make_event_gates(
        time_s,
        cue_onset_s=0.0,
        cue_duration_s=CUE_DURATION_S,
        target_onset_s=scenario.target_onset_s,
        target_duration_s=scenario.target_duration_s,
    )
    cue = load_stimulus("Stage1", scenario.cue_side)
    target = load_stimulus("Stage2", scenario.target_stimulus)
    scenes = np.stack((np.zeros_like(cue.signed_contrast), cue.signed_contrast,
                       target.signed_contrast))
    scene_index = np.zeros(time_ms.size, dtype=np.int64)
    scene_index[cue_gate > 0.0] = 1
    scene_index[target_gate > 0.0] = 2
    if resolution not in (64, 128):
        raise ValueError("resolution must match Q2's supported 64 or 128 grid")
    lgn_state = {
        "adaptation": np.zeros((resolution, resolution), dtype=np.float32),
        "tcr": np.zeros((2, resolution, resolution), dtype=np.float32),
        "interneuron": np.zeros((2, resolution, resolution), dtype=np.float32),
        "trn": np.zeros((2, resolution, resolution), dtype=np.float32),
    }
    q2_model_params = ModelParams() if q2_params is None else q2_params
    frontend = simulate_frontend(
        cue,
        params={"tau_a": q2_model_params.tau_a},
        time_ms=time_ms,
        resolution=resolution,
        feature_stride_ms=4.0,
        scene_contrasts=scenes,
        scene_index=scene_index,
        initial_lgn_state=lgn_state,
    )
    forward = simulate_forward(
        frontend,
        params=q2_model_params,
        time_ms=time_ms,
        drive_scales=_q2_drive_scales(),
    )
    q2_sensor = np.asarray(forward.eeg, dtype=float)
    source_proxy = np.asarray(forward.source_proxy, dtype=float)
    leadfield = np.asarray(forward.diagnostics["leadfield"], dtype=float)
    projection_error = float(np.max(np.abs(q2_sensor - leadfield @ source_proxy)))
    visual_drive = np.mean(np.asarray(forward.excitatory[0], dtype=float), axis=0)
    states = integrate_macro_states(
        visual_drive=visual_drive,
        cue_gate=cue_gate,
        target_gate=target_gate,
        match_evidence=scenario.match_evidence,
        dt_s=dt_ms / 1000.0,
        params=macro_params,
    )
    u_obs = np.asarray(q2_config.U_OBS, dtype=float)
    memory_loading = u_obs[2]
    control_loading = u_obs[1]
    p = MacroParameters() if macro_params is None else macro_params
    model_sensor = observe_macro_model(
        q2_sensor,
        states["memory"],
        states["control"],
        memory_loading,
        control_loading,
        topdown_control_gain=p.topdown_control_gain,
    )
    return {
        "time_s": time_s,
        "cue_gate": cue_gate,
        "target_gate": target_gate,
        "visual_drive": visual_drive,
        "states": states,
        "q2_sensor": q2_sensor,
        "q2_source_proxy": source_proxy,
        "leadfield": leadfield,
        "model_sensor": model_sensor,
        "memory_loading": memory_loading,
        "control_loading": control_loading,
        "projection_max_abs_error": projection_error,
        "scenario": scenario,
        "q2_params": q2_model_params,
        "macro_params": MacroParameters() if macro_params is None else macro_params,
    }


def _run_sensitivity_scenarios() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    q2_cache: dict[tuple[str, str, float], dict[str, Any]] = {}
    for target_type in TARGET_TYPES:
        for onset_s in TARGET_ONSETS_S:
            cache_key = ("left", target_type, onset_s)
            base = q2_cache.get(cache_key)
            if base is None:
                base = simulate_q2_scenario(
                    DynamicScenario("left", target_type, onset_s, 0.2, 1.0)
                )
                q2_cache[cache_key] = base
            for evidence in MATCH_EVIDENCE_VALUES:
                scenario = DynamicScenario("left", target_type, onset_s, 0.2, evidence)
                macro_params = base["macro_params"]
                states = integrate_macro_states(
                    base["visual_drive"], base["cue_gate"], base["target_gate"],
                    evidence, Q2_STEP_MS / 1000.0, macro_params,
                )
                sensor = observe_macro_model(
                    base["q2_sensor"], states["memory"], states["control"],
                    base["memory_loading"], base["control_loading"],
                    macro_params.topdown_control_gain,
                )
                trajectory = dict(base)
                trajectory.update({"states": states, "model_sensor": sensor,
                                   "scenario": scenario})
                trajectories.append(trajectory)
                projection_error = base["projection_max_abs_error"]
                rows.append({
                    "cue_side": scenario.cue_side,
                    "target_stimulus_candidate": target_type,
                    "target_onset_candidate_s": onset_s,
                    "target_duration_s_assumption": 0.2,
                    "match_evidence_scenario": evidence,
                    "q2_source_count_forward_only": int(base["q2_source_proxy"].shape[0]),
                    "sensor_count": int(base["q2_sensor"].shape[0]),
                    "q2_projection_max_abs_error": projection_error,
                    "visual_peak": float(states["visual"].max()),
                    "memory_peak_abs": float(np.max(np.abs(states["memory"]))),
                    "control_peak": float(states["control"].max()),
                    "sensor_peak_abs_relative_units": float(np.max(np.abs(sensor))),
                    "all_values_finite": bool(
                        np.isfinite(base["q2_source_proxy"]).all()
                        and np.isfinite(states["visual"]).all()
                        and np.isfinite(states["memory"]).all()
                        and np.isfinite(states["control"]).all()
                        and np.isfinite(sensor).all()
                    ),
                    "fit_to_measured_eeg": False,
                    "behavioral_label_used": False,
                })
    return trajectories, pd.DataFrame(rows)


def _plot_scenarios(trajectories: list[dict[str, Any]], output_path: Path) -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), sharex="col")
    colors = {2.0: "#3D7193", 2.2: "#D18737", 2.4: "#4D8B70"}
    scenario_handles = []
    for onset in TARGET_ONSETS_S:
        for evidence, style in ((1.0, "-"), (-1.0, "--")):
            matches = [
                row for row in trajectories
                if row["scenario"].target_stimulus == "dots"
                and row["scenario"].target_onset_s == onset
                and row["scenario"].match_evidence == evidence
            ]
            if not matches:
                continue
            row = matches[0]
            time = row["time_s"]
            states = row["states"]
            label = f"目标时刻 {onset:.1f} 秒，{'匹配' if evidence > 0 else '不匹配'}"
            scenario_handles.append(
                Line2D([0], [0], color=colors[onset], linestyle=style,
                       linewidth=1.25, label=label)
            )
            for axis, key, title in zip(
                axes[0], ("visual", "memory", "control"),
                ("问题二视觉驱动状态 V", "记忆相关状态 H", "控制状态 P"),
            ):
                axis.plot(time, states[key], color=colors[onset], linestyle=style,
                          linewidth=1.15)
                axis.set_title(title)
                axis.set_ylabel("相对状态量")
                axis.grid(color="#e4e4e4", linewidth=0.55)
                axis.set_xlim(SIMULATION_START_S, SIMULATION_STOP_S)

    representative = next(
        row for row in trajectories
        if row["scenario"].target_stimulus == "dots"
        and row["scenario"].target_onset_s == 2.2
        and row["scenario"].match_evidence == -1.0
    )
    configs = (
        PreprocessingConfig(mode="none"),
        PreprocessingConfig(mode="causal"),
        PreprocessingConfig(mode="zero_phase"),
    )
    styles = {"none": ":", "causal": "-", "zero_phase": "--"}
    preprocessing_labels = {
        "none": "不滤波",
        "causal": "因果滤波",
        "zero_phase": "零相位滤波",
    }
    processed_curves = {}
    for cfg in configs:
        processed, time = apply_preprocessing(
            representative["model_sensor"], representative["time_s"], Q2_RATE_HZ, cfg
        )
        processed_curves[cfg.mode] = (processed, time)
    for channel_index, (axis, channel) in enumerate(zip(axes[1], CHANNELS)):
        for cfg in configs:
            values, time = processed_curves[cfg.mode]
            axis.plot(time, values[channel_index], linestyle=styles[cfg.mode],
                      linewidth=1.05)
        axis.axvline(2.2, color="#9b4438", linestyle=(0, (3, 3)), linewidth=0.9)
        axis.set_title(f"电极 {channel}：观测处理敏感性")
        axis.set_ylabel("脑电幅值（模型单位）")
        axis.set_xlim(SIMULATION_START_S, SIMULATION_STOP_S)
        axis.grid(color="#e4e4e4", linewidth=0.55)
    for axis in axes[1]:
        axis.set_xlabel("相对视觉提示的时间（秒）")

    fig.suptitle(
        "候选认知状态轨迹与预处理敏感性\n"
        "问题二前向模型生成的候选情景；未拟合或验证实测脑电",
        fontsize=11, y=0.985,
    )
    preprocessing_handles = [
        Line2D([0], [0], color="#555555", linestyle=styles[mode], linewidth=1.25,
               label=preprocessing_labels[mode])
        for mode in ("none", "causal", "zero_phase")
    ]
    preprocessing_handles.append(
        Line2D([0], [0], color="#9b4438", linestyle=(0, (3, 3)), linewidth=1.0,
               label="候选目标时刻 2.2 秒")
    )

    def add_framed_legend(handles, y: float, ncol: int) -> None:
        legend = fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
            ncol=ncol, fontsize=7.2, frameon=True, fancybox=True,
            framealpha=1.0, facecolor="white", edgecolor="#c8c8c8",
            borderpad=0.55, labelspacing=0.45, handletextpad=0.7,
            columnspacing=1.4, handlelength=1.8,
        )
        legend.get_frame().set_linewidth(0.8)

    add_framed_legend(scenario_handles + preprocessing_handles, 0.91, 3)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.09,
                        wspace=0.25, hspace=0.34)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report(rows: pd.DataFrame, report_path: Path, figure_path: Path) -> None:
    onsets = ", ".join(f"{value:.1f}" for value in TARGET_ONSETS_S)
    types = ", ".join(TARGET_TYPES)
    finite_count = int(rows["all_values_finite"].sum())
    total = int(len(rows))
    projection_max = float(rows["q2_projection_max_abs_error"].max())
    mismatch = rows.loc[rows["match_evidence_scenario"] == -1.0]
    matched = rows.loc[rows["match_evidence_scenario"] == 1.0]
    mismatch_peak = float(mismatch["control_peak"].mean())
    match_peak = float(matched["control_peak"].mean())
    report = """# Q3 最小动态认知模型：机制接口与数值检查

## 目的与范围

本版本实现用户确认的路线1：用 Q2 视觉生成链作为视觉过程驱动，通过宏观状态转移表示记忆保持/匹配与控制响应，再用观测方程映射至 F3/Fz/F4。当前只检验模型方程、Q2 前向接口与情景配置，不拟合实测 EEG、不估计反应时/正确率/漏答率，也不作脑区定位结论。

## Q2 机制接口

候选场景依次构造 cue 与 target 的连续图像输入。Q2 的 LGN ON/OFF、Gabor/构型前端、Wilson–Cowan E/I 群体、源代理及导联矩阵正向生成 `q_Q2(t)` 和 `y_Q2(t)=Gq_Q2(t)`。Q3 只调用前向链；不对 `G` 求逆。五个 Q2 源代理通过 `G` 产生三电极预测，源通道数多于电极数是允许的前向计算。

Q2当前保存的留出曲线仍是 cue-only 且覆盖约800 ms；本脚本重新生成**候选 cue—target 场景**，不会把旧曲线外推为完整试次。

## 宏观过程状态

取 Q2 早期视觉皮层群体的平均兴奋状态作为输入 `u_Q2(t)`，并以 `V(t)` 作有限时间常数平滑。cue 窗口把视觉状态写入记忆相关状态；候选 target 窗口按场景的 `match_evidence` 更新记忆状态。控制状态由目标时段的记忆强度和候选冲突项驱动：

$$
V_{k+1}=\\rho_VV_k+(1-\\rho_V)u_{Q2,k},\\quad
H_{k+1}=\\rho_HH_k+(1-\\rho_H)(g_sV_kC_k+g_mV_kT_k r),
$$
$$
P_{k+1}=\\rho_PP_k+(1-\\rho_P)\\left(g_{HP}|H_k|T_k+g_\\delta T_k\\frac{{1-r}}{{2}}\\right),
\\quad \\rho_j=\\exp(-\\Delta t/\\tau_j).
$$

这里 `C_k/T_k` 是由配置的 cue/target 时窗产生的门控，`r∈[-1,1]` 是纯**情景参数**（正值匹配、负值不匹配），不是观测到的正确性。`V` 来自 Q2 网络内部的视觉动态；`H/P` 是功能性宏观状态，不等同于海马/PFC源。

## 传感器观测方程

$$
\\hat{{\\mathbf y}}_k=(1+g_{{PV}}P_k)G\\mathbf q_{{Q2,k}}
 +\\boldsymbol{{\\ell}}_H H_k+\\boldsymbol{{\\ell}}_P P_k+\\boldsymbol{{\\epsilon}}_k,
$$

其中 `Gq_Q2` 是问题二导联矩阵的正向传感器输出；`ell_H`、`ell_P` 暂取固定正交传感器模式作为**可替换的加载情景**，用于验证接口，不代表海马或前额叶的已知头皮拓扑。此阶段不拟合其幅度、时间常数或传感器加载，避免用三通道无约束拟合多个潜过程。

## 可配置情景与预处理

- cue侧：left/right；当前敏感性演示用 left cue。
- 候选target类型：@@types@@。
- 候选target相对cue时刻（秒）：@@onsets@@；它们是时序敏感性分支，不是已确认目标呈现时刻。
- 本次数值演示假设 target 持续200 ms；代码接口可改持续时间或设为持续至模拟终点。MAT未提供可验证的逐试次target offset。
- `match_evidence`分别运行匹配/不匹配情景；不与真实正确/错误标签连接。
- 共同观测处理提供 `none`、单向因果滤波、双向零相位滤波三个配置；均可配置频带、阶数、目标采样率、基线窗。因果滤波保留时间因果性但有频率相关相位延迟；零相位滤波便于离线形态对照，但不能独立证明精细的阶段先后。

## 数值检查结果

- 情景数：@@total@@（3种target × 3个候选时刻 × 匹配/不匹配）。有限值通过：@@finite_count@@/@@total@@。
- `max |y_Q2−Gq_Q2|`：@@projection_max@@（验证问题二前向投影接口）。
- 匹配情景平均控制峰值：@@match_peak@@；不匹配情景平均控制峰值：@@mismatch_peak@@。这只验证设定方程的冲突分支方向，不是行为效应证据。
- 数值检查通过的状态与传感器范围逐情景保存在 `dynamic_model_checks.csv`。

## 图片

![候选动态状态、传感器投影与预处理敏感性](figures/dynamic_model_sensitivity.png)

图中曲线完全由问题二模型和透明的情景参数生成，不是实测EEG拟合或脑区来源证据。

## 后续实测拟合前必须补齐

1. 目标显示时间戳、反应标记的写入语义、行为反应截止时间及文件到任务的映射。
2. 选定统一预处理算子后，对实测EEG与模型输出应用完全相同的滤波、重采样和基线步骤。
3. 冻结上述方程和参数范围，再做整份记录留出拟合；同时比较 Q2-only、Q2+记忆状态、Q2+记忆/控制状态，并报告参数可辨识性与时间轨迹误差。

重现命令：`python src/C/q3/dynamic_cognitive_model.py`。
"""
    replacements = {
        "@@types@@": types,
        "@@onsets@@": onsets,
        "@@total@@": str(total),
        "@@finite_count@@": str(finite_count),
        "@@projection_max@@": f"{projection_max:.3e}",
        "@@match_peak@@": f"{match_peak:.6g}",
        "@@mismatch_peak@@": f"{mismatch_peak:.6g}",
    }
    for token, value in replacements.items():
        report = report.replace(token, value)
    report_path.write_text(report, encoding="utf-8")


def run_sensitivity(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trajectories, rows = _run_sensitivity_scenarios()
    checks_path = output_dir / "dynamic_model_checks.csv"
    rows.to_csv(checks_path, index=False, encoding="utf-8-sig")
    figure_path = figure_dir / "dynamic_model_sensitivity.png"
    _plot_scenarios(trajectories, figure_path)
    report_path = output_dir / "dynamic_model_report.md"
    _write_report(rows, report_path, figure_path)
    print(f"Scenarios finite: {int(rows['all_values_finite'].sum())}/{len(rows)}")
    print(f"Max Q2 projection identity error: {rows['q2_projection_max_abs_error'].max():.3e}")
    print(f"Wrote PNG-only figure: {figure_path}")
    print(f"Wrote checks and report to {output_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_sensitivity(args.output_dir)


if __name__ == "__main__":
    main()
