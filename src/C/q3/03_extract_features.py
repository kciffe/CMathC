"""Extract event-locked frontal EEG features from the Q1 clean epochs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import welch

from common import ensure_output_dir, write_csv
from config import (
    BANDS_HZ,
    EEG_CHANNELS,
    ERP_BASELINE_WINDOW_S,
    ERP_CANDIDATE_WINDOW_S,
    ERP_PEAK_WINDOW_S,
    FILTER_BAND_HZ,
    POWER_WINDOW_S,
    Q1_SAMPLE_RATE_HZ,
    TARGET_OFFSET_S,
)


def _window_mask(time: np.ndarray, event_offset_s: float, window: tuple[float, float]) -> np.ndarray:
    start, end = window
    return (time >= event_offset_s + start) & (time < event_offset_s + end)


def _bandpower(signal: np.ndarray, sample_rate_hz: float, band: tuple[float, float]) -> float:
    values = np.asarray(signal, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 32 or np.std(values) == 0:
        return np.nan
    nperseg = min(values.size, 256)
    frequencies, power = welch(
        values,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    if selected.sum() < 2:
        return np.nan
    return float(np.trapezoid(power[selected], frequencies[selected]))


def _stage_features(
    eeg: np.ndarray,
    time: np.ndarray,
    event_offset_s: float,
    sample_rate_hz: float,
) -> tuple[dict[str, float | int], dict[str, int]]:
    baseline_mask = _window_mask(time, event_offset_s, ERP_BASELINE_WINDOW_S)
    candidate_mask = _window_mask(time, event_offset_s, ERP_CANDIDATE_WINDOW_S)
    peak_mask = _window_mask(time, event_offset_s, ERP_PEAK_WINDOW_S)
    power_mask = _window_mask(time, event_offset_s, POWER_WINDOW_S)
    window_counts = {
        "baseline": int(baseline_mask.sum()),
        "erp_candidate": int(candidate_mask.sum()),
        "erp_peak": int(peak_mask.sum()),
        "power": int(power_mask.sum()),
    }
    result: dict[str, float | int] = {}

    for channel_index, channel in enumerate(EEG_CHANNELS):
        values = eeg[channel_index]
        if not (baseline_mask.any() and candidate_mask.any() and peak_mask.any() and power_mask.any()):
            for feature in (
                "erp_mean",
                "erp_area",
                "erp_peak",
                "erp_peak_latency_s",
                "log_theta_power",
                "log_alpha_power",
                "log_beta_power",
            ):
                result[f"{feature}_{channel}"] = np.nan
            continue

        baseline = float(np.mean(values[baseline_mask]))
        candidate = values[candidate_mask] - baseline
        candidate_time = time[candidate_mask] - event_offset_s
        peak_values = values[peak_mask] - baseline
        peak_time = time[peak_mask] - event_offset_s
        peak_index = int(np.argmax(peak_values))
        result[f"erp_mean_{channel}"] = float(np.mean(candidate))
        result[f"erp_area_{channel}"] = float(np.trapezoid(candidate, candidate_time))
        result[f"erp_peak_{channel}"] = float(peak_values[peak_index])
        result[f"erp_peak_latency_s_{channel}"] = float(peak_time[peak_index])

        segment = values[power_mask]
        for band_name, bounds in BANDS_HZ.items():
            power = _bandpower(segment, sample_rate_hz, bounds)
            result[f"log_{band_name}_power_{channel}"] = (
                float(np.log(max(power, np.finfo(float).tiny))) if np.isfinite(power) else np.nan
            )

    if all(np.isfinite(result.get(f"log_alpha_power_{channel}", np.nan)) for channel in ("F3", "F4")):
        result["AI_alpha"] = float(result["log_alpha_power_F4"] - result["log_alpha_power_F3"])
    else:
        result["AI_alpha"] = np.nan
    return result, window_counts


def _pre_response_features(
    eeg: np.ndarray, sample_rate_hz: float
) -> dict[str, float]:
    result: dict[str, float] = {}
    if eeg.ndim != 2 or eeg.shape[0] != len(EEG_CHANNELS) or not np.isfinite(eeg).all():
        return result
    for channel_index, channel in enumerate(EEG_CHANNELS):
        values = eeg[channel_index]
        result[f"pre_response_mean_{channel}"] = float(np.mean(values))
        result[f"pre_response_sd_{channel}"] = float(np.std(values, ddof=1))
        for band_name, bounds in BANDS_HZ.items():
            power = _bandpower(values, sample_rate_hz, bounds)
            result[f"log_{band_name}_power_{channel}"] = (
                float(np.log(max(power, np.finfo(float).tiny))) if np.isfinite(power) else np.nan
            )
    if all(np.isfinite(result.get(f"log_alpha_power_{channel}", np.nan)) for channel in ("F3", "F4")):
        result["AI_alpha"] = float(result["log_alpha_power_F4"] - result["log_alpha_power_F3"])
    else:
        result["AI_alpha"] = np.nan
    return result


def feature_definitions() -> pd.DataFrame:
    rows = [
        {
            "feature_family": "frontal ERP/P300 candidate",
            "feature": "erp_mean_{F3,Fz,F4}",
            "definition": "mean EEG in 0.25-0.50 s after event minus pre-event baseline mean",
            "window_s": str(ERP_CANDIDATE_WINDOW_S),
            "interpretation_limit": "frontal P300 candidate; not a standard midline/parietal P300 measurement",
        },
        {
            "feature_family": "frontal ERP/P300 candidate",
            "feature": "erp_peak_{F3,Fz,F4}, erp_peak_latency_s_{F3,Fz,F4}",
            "definition": "largest baseline-corrected positive sample and its latency",
            "window_s": str(ERP_PEAK_WINDOW_S),
            "interpretation_limit": "single-trial peak is noise-sensitive; candidate feature only",
        },
        {
            "feature_family": "frontal ERP/P300 candidate",
            "feature": "erp_area_{F3,Fz,F4}",
            "definition": "trapezoidal integral of baseline-corrected EEG",
            "window_s": str(ERP_CANDIDATE_WINDOW_S),
            "interpretation_limit": "signal units times seconds; source units are inherited from Q1 MAT",
        },
        {
            "feature_family": "log band power",
            "feature": "log_theta/alpha/beta_power_{F3,Fz,F4}",
            "definition": "natural log of Welch-integrated band power",
            "window_s": str(POWER_WINDOW_S),
            "interpretation_limit": f"Q1 processing band is {FILTER_BAND_HZ[0]}-{FILTER_BAND_HZ[1]} Hz; gamma/PAC excluded",
        },
        {
            "feature_family": "frontal alpha asymmetry",
            "feature": "AI_alpha",
            "definition": "log(P_alpha,F4) - log(P_alpha,F3)",
            "window_s": str(POWER_WINDOW_S),
            "interpretation_limit": "three frontal electrodes only; not a whole-scalp lateralization estimate",
        },
        {
            "feature_family": "pre-response EEG",
            "feature": "pre_response_mean/sd_{F3,Fz,F4}, log_theta/alpha/beta_power_{F3,Fz,F4}, AI_alpha",
            "definition": "features from a full-continuous-record filtered interval ending about 100 ms before the channel 9 response marker",
            "window_s": "[-1.10, -0.10) relative to response onset",
            "interpretation_limit": "response-locked exploratory features; excluded from behavior prediction because their window endpoint depends on response time",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = ensure_output_dir()
    trials_path = output_dir / "q3_eeg_trials.npz"
    if not trials_path.exists():
        raise FileNotFoundError("Run 02_extract_trials.py first to create q3_eeg_trials.npz")
    archive = np.load(trials_path, allow_pickle=False)
    signals = archive["signal"]
    times = archive["relative_time"]
    records = archive["record"].astype(str)
    raw_indices = archive["original_trial_index"]
    q1_indices = archive["q1_trial_index"]
    cue_sides = archive["cue_side"]
    cue_times = archive["cue_time_s"]
    q1_keys = archive["q1_trial_key"].astype(str)

    feature_rows: list[dict] = []
    for index in range(signals.shape[0]):
        for stage, event_offset in (("cue_locked", 0.0), ("target_locked", TARGET_OFFSET_S)):
            features, counts = _stage_features(
                signals[index], times[index], event_offset, Q1_SAMPLE_RATE_HZ
            )
            feature_rows.append(
                {
                    "record": records[index],
                    "original_trial_index": int(raw_indices[index]),
                    "q1_trial_index": int(q1_indices[index]),
                    "q1_trial_key": q1_keys[index],
                    "cue_time_s": float(cue_times[index]),
                    "cue_side": int(cue_sides[index]),
                    "cue_side_text": "left" if cue_sides[index] < 0 else "right",
                    "task_type": "unresolved_from_allowed_channels",
                    "stage": stage,
                    "event_offset_from_cue_s": float(event_offset),
                    "event_time_s": float(cue_times[index] + event_offset),
                    "erp_baseline_samples": counts["baseline"],
                    "erp_candidate_samples": counts["erp_candidate"],
                    "erp_peak_samples": counts["erp_peak"],
                    "power_samples": counts["power"],
                    "q1_sample_rate_hz": Q1_SAMPLE_RATE_HZ,
                    **features,
                }
            )

    response_path = output_dir / "response_locked_eeg.npz"
    if not response_path.exists():
        raise FileNotFoundError("Run 02_extract_trials.py first to create response_locked_eeg.npz")
    response_archive = np.load(response_path, allow_pickle=False)
    response_features = response_archive["signal"]
    response_times = response_archive["relative_time"]
    response_records = response_archive["record"].astype(str)
    response_indices = response_archive["original_trial_index"]
    response_codes = response_archive["response_code"]
    response_times_abs = response_archive["response_time_s"]
    response_rates = response_archive["sample_rate_hz"]
    trial_table = pd.read_csv(output_dir / "trial_table.csv", encoding="utf-8-sig")
    trial_lookup = {
        (str(row.record), int(row.original_trial_index)): row
        for row in trial_table.itertuples(index=False)
    }
    for index, signal in enumerate(response_features):
        if int(response_codes[index]) == 0 or not np.isfinite(signal).all():
            continue
        record = response_records[index]
        original_index = int(response_indices[index])
        trial = trial_lookup[(record, original_index)]
        rel_time = response_times[index]
        features = _pre_response_features(signal, float(response_rates[index]))
        response_start = float(rel_time[0])
        response_end = float(rel_time[-1])
        feature_rows.append(
            {
                "record": record,
                "original_trial_index": original_index,
                "q1_trial_index": int(trial.q1_trial_index)
                if pd.notna(trial.q1_trial_index)
                else np.nan,
                "q1_trial_key": str(trial.q1_trial_key),
                "cue_time_s": float(trial.cue_time_s),
                "cue_side": int(trial.cue_side),
                "cue_side_text": str(trial.cue_side_text),
                "task_type": str(trial.task_type),
                "stage": "pre_response_100ms_end",
                "event_offset_from_cue_s": float(response_times_abs[index] - trial.cue_time_s),
                "event_time_s": float(response_times_abs[index]),
                "response_time_s": float(response_times_abs[index]),
                "response_code": int(response_codes[index]),
                "response_side": str(trial.response_side),
                "choice_side": int(trial.choice_side),
                "reaction_time_s": float(trial.reaction_time_s)
                if pd.notna(trial.reaction_time_s)
                else np.nan,
                "rt_status": str(trial.rt_status),
                "is_omission": bool(trial.is_omission),
                "correct": np.nan,
                "erp_baseline_samples": 0,
                "erp_candidate_samples": 0,
                "erp_peak_samples": 0,
                "power_samples": int(signal.shape[1]),
                "power_window_start_relative_response_s": response_start,
                "power_window_end_relative_response_s": response_end,
                "q1_sample_rate_hz": float(response_rates[index]),
                **features,
            }
        )

    feature_table = pd.DataFrame(feature_rows)
    write_csv(feature_table, output_dir / "trial_features.csv")
    write_csv(feature_definitions(), output_dir / "feature_definitions.csv")

    # A compact distribution plot exposes sample size and cue-side overlap.
    target = feature_table.loc[feature_table["stage"] == "target_locked"]
    plot_columns = ["erp_mean_Fz", "log_theta_power_Fz", "log_alpha_power_Fz", "AI_alpha"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, column in zip(axes.flat, plot_columns):
        values = [
            target.loc[target["cue_side"] == side, column].dropna().to_numpy()
            for side in (-1, 1)
        ]
        ax.boxplot(values, tick_labels=["Left cue", "Right cue"], showfliers=False)
        ax.set_title(column)
        ax.set_ylabel("Feature value")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Target-locked frontal EEG features by VisCue direction")
    fig.savefig(output_dir / "features_by_cue_side.png", dpi=180)
    plt.close(fig)

    print(
        f"Wrote {len(feature_table)} trial-stage rows: Q1 clean cue/target windows "
        "plus continuous-record response-preceding windows."
    )


if __name__ == "__main__":
    main()
