"""Extract per-trial features from the raw-data ERP and time-frequency branches."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from common import ensure_output_dir, write_csv, write_json
from config import EEG_CHANNELS, HARD_CLIP_THRESHOLD_RAW
from signal_processing import summarize_epoch_features, summarize_window_features


def _nan_features(feature_names: list[str]) -> dict[str, float]:
    return {name: float("nan") for name in feature_names}


def feature_definitions() -> pd.DataFrame:
    rows = [
        {
            "feature_family": "frontal ERP candidate",
            "feature": "erp_mean_{F3,Fz,F4}, erp_peak_{F3,Fz,F4}, erp_peak_latency_s_{F3,Fz,F4}",
            "definition": "baseline-corrected mean and positive peak in 250-500 ms after the event",
            "window_s": "baseline [-0.10,0.00); candidate [0.25,0.50)",
            "interpretation_limit": "frontal ERP/P300 candidate; three frontal electrodes do not establish the standard parietal P300 topography",
        },
        {
            "feature_family": "frontal ERP asymmetry",
            "feature": "AI_erp_F4_minus_F3",
            "definition": "(mean_F4 - mean_F3) / (abs(mean_F4) + abs(mean_F3) + epsilon), based on baseline-corrected 250-500 ms means",
            "window_s": "[0.25,0.50)",
            "interpretation_limit": "three frontal electrodes only; exploratory lateralization index",
        },
        {
            "feature_family": "time-frequency log band power",
            "feature": "log_{theta,alpha,beta,gamma}_power_{F3,Fz,F4}",
            "definition": "natural log of Welch-integrated power in theta 4-8, alpha 8-13, beta 13-30, gamma 30-80 Hz",
            "window_s": "post-event portion of cue [0,0.50) or target [0,0.80) epoch",
            "interpretation_limit": "one short trial window; theta/alpha power has low frequency resolution and gamma is exploratory",
        },
        {
            "feature_family": "frontal alpha asymmetry",
            "feature": "AI_alpha_log_F4_minus_F3",
            "definition": "log(P_alpha,F4) - log(P_alpha,F3)",
            "window_s": "post-event portion of cue or target epoch",
            "interpretation_limit": "three frontal electrodes only; not a whole-scalp lateralization estimate",
        },
        {
            "feature_family": "response-endpoint cumulative window",
            "feature": "pre_response_cumulative_* and pre_response_terminal_500ms_*",
            "definition": "mean, standard deviation, and log band power over [cue, marker-100 ms] and its final 500 ms",
            "window_s": "variable length; endpoint is the channel-9 event time minus 100 ms",
            "interpretation_limit": "response/event semantics need protocol verification; excluded from choice prediction because the endpoint is action-conditioned",
        },
    ]
    return pd.DataFrame(rows)


def _read_archive(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}; run 02b_preprocess_raw.py after 02_extract_trials.py"
        )
    return np.load(path, allow_pickle=False)


def main() -> None:
    output_dir = ensure_output_dir()
    event_archive = _read_archive(output_dir / "q3_raw_event_epochs.npz")
    n_rows = int(event_archive["sample_count"].size)
    event_rows: list[dict] = []
    all_feature_names: set[str] = set()

    for index in range(n_rows):
        length = int(event_archive["sample_count"][index])
        erp = event_archive["erp_signal"][index, :, :length].astype(np.float64)
        tf = event_archive["tf_signal"][index, :, :length].astype(np.float64)
        relative_time = event_archive["relative_time_s"][index, :length].astype(np.float64)
        feature_values: dict[str, float] = {}
        qc = {
            "qc_valid": False,
            "erp_qc_valid": False,
            "tf_qc_valid": False,
            "erp_hard_clip": False,
            "tf_hard_clip": False,
            "erp_nonfinite": False,
            "tf_nonfinite": False,
        }
        if length:
            try:
                feature_values, epoch_qc = summarize_epoch_features(
                    erp, tf, relative_time, float(event_archive["sample_rate_hz"][index])
                )
                qc = {
                    "qc_valid": bool(epoch_qc["valid"]),
                    "erp_qc_valid": bool(epoch_qc["erp_valid"]),
                    "tf_qc_valid": bool(epoch_qc["tf_valid"]),
                    "erp_hard_clip": bool(epoch_qc["erp_hard_clip"]),
                    "tf_hard_clip": bool(epoch_qc["tf_hard_clip"]),
                    "erp_nonfinite": bool(epoch_qc["erp_nonfinite"]),
                    "tf_nonfinite": bool(epoch_qc["tf_nonfinite"]),
                }
                if not qc["erp_qc_valid"]:
                    for name in list(feature_values):
                        if name.startswith(("erp_", "AI_erp")):
                            feature_values[name] = float("nan")
                if not qc["tf_qc_valid"]:
                    for name in list(feature_values):
                        if name.startswith(("log_", "AI_alpha")):
                            feature_values[name] = float("nan")
            except ValueError:
                feature_values = {}
        all_feature_names.update(feature_values)

        record = str(event_archive["record"][index])
        task_text = record.rsplit("Task-", maxsplit=1)[-1]
        filename_task_code = int(task_text) if task_text in {"1", "2"} else np.nan
        q1_index = int(event_archive["q1_trial_index"][index])
        q1_drop = int(event_archive["q1_final_drop"][index])
        q1_quality_pass = bool(q1_index >= 0 and q1_drop == 0)
        event_rows.append(
            {
                "record": record,
                "original_trial_index": int(event_archive["original_trial_index"][index]),
                "q1_trial_index": q1_index if q1_index >= 0 else np.nan,
                "q1_quality_pass": q1_quality_pass,
                "q1_final_drop": q1_drop if q1_drop >= 0 else np.nan,
                "eeg_quality": str(event_archive["eeg_quality"][index]),
                "cue_time_s": float(event_archive["cue_time_s"][index]),
                "event_time_s": float(event_archive["event_time_s"][index]),
                "event_offset_from_cue_s": float(event_archive["event_offset_from_cue_s"][index]),
                "target_offset_s": float(event_archive["target_offset_s"][index]),
                "cue_side": int(event_archive["cue_side"][index]),
                "response_code": int(event_archive["response_code"][index]),
                "choice_side": int(np.sign(event_archive["response_code"][index])),
                "stage": str(event_archive["stage"][index]),
                "analysis_variant": str(event_archive["analysis_variant"][index]),
                "filename_task_code_candidate": filename_task_code,
                "task_mapping_status": "filename suffix only; project/task semantics not independently verified",
                "sample_rate_hz": float(event_archive["sample_rate_hz"][index]),
                "window_n_samples": length,
                "window_duration_s": length / float(event_archive["sample_rate_hz"][index]),
                "erp_filter_band_hz": "0.5-30",
                "tf_filter_band_hz": "1-80",
                "hard_clip_threshold_raw_value": HARD_CLIP_THRESHOLD_RAW,
                **qc,
                **feature_values,
            }
        )

    # Normalize the feature schema across rows, including rows with failed QC.
    core_feature_names = {
        f"{base}_{channel}"
        for channel in EEG_CHANNELS
        for base in (
            "erp_candidate_mean",
            "erp_mean",
            "erp_candidate_area",
            "erp_peak",
            "erp_peak_latency_s",
        )
    }
    core_feature_names.update(
        f"log_{band}_power_{channel}"
        for channel in EEG_CHANNELS
        for band in ("theta", "alpha", "beta", "gamma")
    )
    core_feature_names.update({"AI_erp_F4_minus_F3", "AI_alpha_log_F4_minus_F3", "AI_alpha"})
    for row in event_rows:
        for name in core_feature_names:
            row.setdefault(name, float("nan"))
    event_features = pd.DataFrame(event_rows)

    pre_archive = _read_archive(output_dir / "q3_pre_response_epochs.npz")
    pre_rows: list[dict] = []
    for index in range(pre_archive["sample_count"].size):
        length = int(pre_archive["sample_count"][index])
        record = str(pre_archive["record"][index])
        task_text = record.rsplit("Task-", maxsplit=1)[-1]
        filename_task_code = int(task_text) if task_text in {"1", "2"} else np.nan
        row: dict = {
            "record": record,
            "original_trial_index": int(pre_archive["original_trial_index"][index]),
            "q1_trial_index": int(pre_archive["q1_trial_index"][index])
            if int(pre_archive["q1_trial_index"][index]) >= 0
            else np.nan,
            "q1_quality_pass": bool(
                int(pre_archive["q1_trial_index"][index]) >= 0
                and int(pre_archive["q1_final_drop"][index]) == 0
            ),
            "q1_final_drop": int(pre_archive["q1_final_drop"][index])
            if int(pre_archive["q1_final_drop"][index]) >= 0
            else np.nan,
            "eeg_quality": str(pre_archive["eeg_quality"][index]),
            "cue_time_s": float(pre_archive["cue_time_s"][index]),
            "event_time_s": float(pre_archive["endpoint_time_s"][index]),
            "event_offset_from_cue_s": float(
                pre_archive["endpoint_time_s"][index] - pre_archive["cue_time_s"][index]
            )
            if np.isfinite(pre_archive["endpoint_time_s"][index])
            else np.nan,
            "target_offset_s": np.nan,
            "cue_side": int(pre_archive["cue_side"][index]),
            "response_code": int(pre_archive["response_code"][index]),
            "choice_side": int(np.sign(pre_archive["response_code"][index])),
            "stage": "pre_response_endpoint",
            "analysis_variant": "cue_to_marker_minus_100ms",
            "filename_task_code_candidate": filename_task_code,
            "task_mapping_status": "filename suffix only; project/task semantics not independently verified",
            "sample_rate_hz": float(pre_archive["sample_rate_hz"][index]),
            "window_n_samples": length,
            "window_duration_s": float(pre_archive["window_duration_s"][index]),
            "response_time_s": float(pre_archive["response_time_s"][index]),
            "endpoint_status": str(pre_archive["endpoint_status"][index]),
        }
        if length >= 32:
            erp_signal = pre_archive["erp_signal"][index, :, :length]
            tf_signal = pre_archive["tf_signal"][index, :, :length]
            cumulative_features, erp_qc = summarize_window_features(
                erp_signal, row["sample_rate_hz"], "pre_response_cumulative"
            )
            cumulative_tf_features, tf_qc = summarize_window_features(
                tf_signal, row["sample_rate_hz"], "pre_response_cumulative_tf"
            )
            terminal_length = min(length, int(round(0.5 * row["sample_rate_hz"])))
            terminal_features, terminal_erp_qc = summarize_window_features(
                erp_signal[:, -terminal_length:], row["sample_rate_hz"], "pre_response_terminal_500ms"
            )
            terminal_tf_features, terminal_tf_qc = summarize_window_features(
                tf_signal[:, -terminal_length:], row["sample_rate_hz"], "pre_response_terminal_500ms_tf"
            )
            if not erp_qc["valid"]:
                cumulative_features = {
                    key: (float("nan") if key.startswith("pre_response_cumulative_") else value)
                    for key, value in cumulative_features.items()
                }
            if not tf_qc["valid"]:
                cumulative_tf_features = {
                    key: float("nan") for key in cumulative_tf_features
                }
            if not terminal_erp_qc["valid"]:
                terminal_features = {key: float("nan") for key in terminal_features}
            if not terminal_tf_qc["valid"]:
                terminal_tf_features = {key: float("nan") for key in terminal_tf_features}
            row.update(cumulative_features)
            row.update(cumulative_tf_features)
            row.update(terminal_features)
            row.update(terminal_tf_features)
            row.update(
                {
                    "qc_valid": bool(erp_qc["valid"] and tf_qc["valid"]),
                    "erp_qc_valid": bool(erp_qc["valid"]),
                    "tf_qc_valid": bool(tf_qc["valid"]),
                    "erp_hard_clip": bool(erp_qc["hard_clip"]),
                    "tf_hard_clip": bool(tf_qc["hard_clip"]),
                    "terminal_erp_qc_valid": bool(terminal_erp_qc["valid"]),
                    "terminal_tf_qc_valid": bool(terminal_tf_qc["valid"]),
                }
            )
        else:
            row.update(
                {
                    "qc_valid": False,
                    "erp_qc_valid": False,
                    "tf_qc_valid": False,
                    "erp_hard_clip": False,
                    "tf_hard_clip": False,
                    "terminal_erp_qc_valid": False,
                    "terminal_tf_qc_valid": False,
                }
            )
        pre_rows.append(row)

    pre_features = pd.DataFrame(pre_rows)
    feature_table = pd.concat([event_features, pre_features], ignore_index=True, sort=False)
    write_csv(feature_table, output_dir / "trial_features.csv")
    write_csv(feature_definitions(), output_dir / "feature_definitions.csv")

    qc_summary = {
        "event_stage_rows": int(len(event_features)),
        "cue_rows": int((event_features["stage"] == "cue_locked").sum()),
        "nominal_target_rows": int((event_features["stage"] == "target_locked").sum()),
        "target_offset_sensitivity_rows": int((event_features["stage"] == "target_offset_sensitivity").sum()),
        "event_qc_pass_rows": int(event_features["qc_valid"].fillna(False).sum()),
        "event_erp_qc_pass_rows": int(event_features["erp_qc_valid"].fillna(False).sum()),
        "event_tf_qc_pass_rows": int(event_features["tf_qc_valid"].fillna(False).sum()),
        "event_hard_clip_erp_rows": int(event_features["erp_hard_clip"].fillna(False).sum()),
        "event_hard_clip_tf_rows": int(event_features["tf_hard_clip"].fillna(False).sum()),
        "q1_quality_pass_nominal_cue_target": int(
            event_features.loc[event_features["stage"].isin(("cue_locked", "target_locked")), "q1_quality_pass"].sum()
        ),
        "pre_response_rows": int(len(pre_features)),
        "pre_response_qc_pass_rows": int(pre_features["qc_valid"].fillna(False).sum()),
        "hard_clip_threshold_raw_value": HARD_CLIP_THRESHOLD_RAW,
        "note": "Hard clipping at the observed raw-value cap/non-finite/flatline are audited; source units are not independently verified, and no ICA is claimed for three EEG-only channels.",
    }
    write_json(qc_summary, output_dir / "raw_feature_qc_summary.json")
    nominal_quality = event_features.loc[
        event_features["stage"].isin(("cue_locked", "target_locked"))
    ].copy()
    nominal_quality["q1_quality_pass"] = nominal_quality["q1_quality_pass"].fillna(False).astype(bool)
    nominal_quality["qc_valid"] = nominal_quality["qc_valid"].fillna(False).astype(bool)
    agreement_rows = []
    for stage, group in nominal_quality.groupby("stage"):
        agreement_rows.append(
            {
                "stage": stage,
                "n_trials": int(len(group)),
                "q1_pass_raw_qc_pass": int((group["q1_quality_pass"] & group["qc_valid"]).sum()),
                "q1_pass_raw_qc_fail": int((group["q1_quality_pass"] & ~group["qc_valid"]).sum()),
                "q1_fail_raw_qc_pass": int((~group["q1_quality_pass"] & group["qc_valid"]).sum()),
                "q1_fail_raw_qc_fail": int((~group["q1_quality_pass"] & ~group["qc_valid"]).sum()),
            }
        )
    write_csv(pd.DataFrame(agreement_rows), output_dir / "quality_agreement.csv")
    write_json(
        {
            "status": "not_computed",
            "features": ["PLV", "theta-gamma PAC"],
            "reason": "Short 0.5-0.8 s event windows and three frontal electrodes do not support reliable connectivity estimates without validated surrogate tests; no values were fabricated.",
        },
        output_dir / "exploratory_connectivity_status.json",
    )

    nominal_target = event_features.loc[
        (event_features["stage"] == "target_locked") & event_features["qc_valid"].fillna(False)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, column in zip(
        axes.flat,
        ("erp_mean_Fz", "log_theta_power_Fz", "log_alpha_power_Fz", "AI_erp_F4_minus_F3"),
    ):
        if column not in nominal_target:
            ax.text(0.5, 0.5, "Feature unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        values = [
            nominal_target.loc[nominal_target["cue_side"] == side, column].dropna().to_numpy()
            for side in (-1, 1)
        ]
        if all(len(group) for group in values):
            ax.boxplot(values, tick_labels=["Left cue", "Right cue"], showfliers=False)
        else:
            ax.text(0.5, 0.5, "Insufficient QC-passed trials", ha="center", va="center")
        ax.set_title(column)
        ax.set_ylabel("Feature value")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Raw continuous EEG target-locked features by VisCue direction")
    fig.savefig(output_dir / "features_by_cue_side.png", dpi=180)
    plt.close(fig)
    print(
        f"Wrote {len(feature_table)} feature rows from raw EEG; "
        f"event QC passed {qc_summary['event_qc_pass_rows']}/{qc_summary['event_stage_rows']}."
    )


if __name__ == "__main__":
    main()
