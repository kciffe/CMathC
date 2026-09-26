"""Build a reproducible continuation audit from the existing Q3 artifacts.

This is an analysis/report adapter. It does not modify raw MAT files or refit
the existing classifiers/state proxies.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})


ROOT = Path.cwd()
Q3 = ROOT / "src" / "C" / "q3"
OUT = Q3 / "output" / "continuation_audit"
MAT_DIR = ROOT / "data" / "第二十三届中国研究生数学建模竞赛+-+中文题目" / "中文题目" / "C题"
RECORDS = ["VisualCogA_Task-1", "VisualCogA_Task-2", "VisualCogB_Task-1", "VisualCogB_Task-2"]
sys.path.insert(0, str(Q3))
from common import detect_response_events, load_raw_record, standardize_response_code

RESPONSE_CODE_MAP = {-2.0: -2, -1.0: -2, 0.0: 0, 1.0: 2, 2.0: 2}
STATE_NAMES = {
    "V_visual_like_raw": "V 视觉前馈代理",
    "H_memory_like_raw": "H 记忆匹配代理",
    "P_control_like_raw": "P 认知控制代理",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value):
    """Convert NumPy scalar values to standard-library JSON scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def lag1(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x)
    x = x[good]
    if len(x) < 3 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return {"n_adjacent_pairs": max(0, len(x) - 1), "correlation": None}
    return {
        "n_adjacent_pairs": int(len(x) - 1),
        "correlation": float(np.corrcoef(x[:-1], x[1:])[0, 1]),
    }


def record_effects(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    variants = [
        ("cue_locked", "nominal cue window"),
        ("target_locked", "target window; cue+2.2 s schedule assumption"),
    ]
    for stage, stage_note in variants:
        subset = states.loc[states["stage"].eq(stage)].copy()
        if stage == "target_locked":
            offset = pd.to_numeric(subset["target_offset_s"], errors="coerce")
            subset = subset.loc[np.isclose(offset, 2.2, atol=1e-5)]
        subset = subset.loc[subset["qc_valid"].fillna(False).astype(bool)]
        for record in RECORDS:
            part = subset.loc[subset["record"].eq(record)]
            left = part.loc[part["cue_side"].eq(-1)]
            right = part.loc[part["cue_side"].eq(1)]
            for column, label in STATE_NAMES.items():
                a = pd.to_numeric(left[column], errors="coerce").dropna().to_numpy(float)
                b = pd.to_numeric(right[column], errors="coerce").dropna().to_numpy(float)
                pooled = np.nan
                if len(a) > 1 and len(b) > 1:
                    pooled = float(np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)))
                d = float((np.mean(b) - np.mean(a)) / pooled) if np.isfinite(pooled) and pooled > 0 else np.nan
                rows.append({
                    "stage": stage,
                    "stage_note": stage_note,
                    "record": record,
                    "state_proxy": label,
                    "n_left": int(len(a)),
                    "n_right": int(len(b)),
                    "mean_left": float(np.mean(a)) if len(a) else np.nan,
                    "mean_right": float(np.mean(b)) if len(b) else np.nan,
                    "mean_difference_right_minus_left": float(np.mean(b) - np.mean(a)) if len(a) and len(b) else np.nan,
                    "pooled_sd": pooled,
                    "cohens_d_right_minus_left": d,
                    "inference": "descriptive per-record effect; no participant-level p-value",
                })
    return pd.DataFrame(rows)


def record_lag_diagnostics(trials: pd.DataFrame) -> list[dict]:
    result = []
    for record, part in trials.groupby("record", sort=False):
        part = part.sort_values("original_trial_index")
        cue_times = pd.to_numeric(part["cue_time_s"], errors="coerce")
        gaps = cue_times.diff().dropna()
        result.append({
            "record": record,
            "n_trials": int(len(part)),
            "cue_side_lag1": lag1(part["cue_side"]),
            "choice_side_lag1": lag1(part["choice_side"]) if "choice_side" in part else None,
            "median_cue_interval_s": float(gaps.median()) if len(gaps) else None,
            "median_interval_from_cue_to_channel9_event_s": float((pd.to_numeric(part["response_time_s"], errors="coerce") - cue_times).median()) if "response_time_s" in part else None,
        })
    return result


def response_mapping_audit(trials: pd.DataFrame) -> pd.DataFrame:
    """Count each raw channel-9 code at sample, edge, and trial levels."""
    sample_counts = {code: 0 for code in RESPONSE_CODE_MAP}
    edge_counts = {code: 0 for code in RESPONSE_CODE_MAP}
    trial_counts = {code: 0 for code in RESPONSE_CODE_MAP}
    for record in RECORDS:
        raw = load_raw_record(record)
        response = np.asarray(raw["response_signal"], dtype=float)
        standardize_response_code(response)  # Reject undocumented numeric codes.
        for code in RESPONSE_CODE_MAP:
            sample_counts[code] += int(np.count_nonzero(response == code))
        events = detect_response_events(response, raw["channels"]["TimeStamp"])
        for event in events:
            edge_counts[float(event["response_raw"])] += 1

    raw_trial_codes = pd.to_numeric(trials["response_raw"], errors="coerce")
    for code in RESPONSE_CODE_MAP:
        trial_counts[code] = int(raw_trial_codes.eq(code).sum())

    rows = []
    for raw_code, standardized in RESPONSE_CODE_MAP.items():
        rows.append({
            "response_raw": int(raw_code),
            "response_code": standardized,
            "meaning": "left response" if standardized < 0 else "right response" if standardized > 0 else "inactive/no response marker",
            "raw_channel_sample_count": sample_counts[raw_code],
            "detected_response_edge_count": edge_counts[raw_code],
            "trial_table_event_count": trial_counts[raw_code],
            "mapping_rule": f"{int(raw_code)} -> {standardized}",
        })
    return pd.DataFrame(rows)


def render_effect_plot(effects: pd.DataFrame, path: Path) -> None:
    colors = {"VisualCogA_Task-1": "#174A6E", "VisualCogA_Task-2": "#3F83B5", "VisualCogB_Task-1": "#E58B45", "VisualCogB_Task-2": "#4F9676"}
    offsets = dict(zip(RECORDS, [-.18, -.06, .06, .18]))
    stages = [("cue_locked", "Cue-locked (primary)"), ("target_locked", "Target-locked (anchor assumed)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), sharey=True)
    for ax, (stage, title) in zip(axes, stages):
        part = effects.loc[effects["stage"].eq(stage)]
        for y, state in enumerate(STATE_NAMES.values()):
            sub = part.loc[part["state_proxy"].eq(state)]
            for _, row in sub.iterrows():
                d = row["cohens_d_right_minus_left"]
                if np.isfinite(d):
                    ax.scatter(d, y + offsets[row["record"]], s=56, color=colors[row["record"]], zorder=3)
        ax.axvline(0, color="#59666D", lw=1)
        ax.set_title(title, color="#174A6E", weight="bold")
        ax.set_xlabel("Cohen d (right cue − left cue)")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(range(len(STATE_NAMES)), list(STATE_NAMES.values()))
    axes[0].invert_yaxis()
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[record], markersize=7, label=record.replace("VisualCog", "").replace("_Task-", "-")) for record in RECORDS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .90), title="Recording")
    fig.suptitle("V/H/P proxy cue-side contrasts by recording", color="#174A6E", weight="bold")
    fig.text(.5, .015, "Four recording-level estimates are shown separately; target panel uses cue+2.2 s and is only a sensitivity result.", ha="center", fontsize=9, color="#59666D")
    fig.tight_layout(rect=(0, .06, 1, .85))
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    output = Q3 / "output"
    records = pd.read_csv(output / "record_audit.csv", encoding="utf-8-sig")
    trials = pd.read_csv(output / "trial_table.csv", encoding="utf-8-sig")
    features = pd.read_csv(output / "trial_features.csv", encoding="utf-8-sig", low_memory=False)
    states = pd.read_csv(output / "state_scores.csv", encoding="utf-8-sig", low_memory=False)
    selected = load_json(output / "experiments" / "20260925_model_search_v1" / "summary.json")
    behavior = load_json(output / "behavior_model_status.json")
    validation = load_json(output / "validation_summary.json")
    official_problem = next(p for p in MAT_DIR.glob("*.docx") if p.stat().st_size > 100_000)
    solution_pdf = next(p for p in MAT_DIR.glob("*.pdf") if p.stem.startswith("第三"))

    cue_features = features.loc[features["stage"].eq("cue_locked") & features["qc_valid"].fillna(False).astype(bool)]
    cue_result = next(item for item in selected["selections_and_deltas"] if item["sample"] == "raw_qc_pass" and item["stage"] == "cue_locked")
    target_result = next(item for item in selected["selections_and_deltas"] if item["sample"] == "raw_qc_pass" and item["stage"] == "target_locked")
    matched_cue = next(item for item in selected["selections_and_deltas"] if item["sample"] == "q1_quality_matched" and item["stage"] == "cue_locked")
    valid_rt = pd.to_numeric(trials["reaction_time_s"], errors="coerce").dropna()
    response_minus_cue = pd.to_numeric(trials["response_time_s"], errors="coerce") - pd.to_numeric(trials["cue_time_s"], errors="coerce")
    alt_rt_20 = response_minus_cue - 2.0
    alt_rt_22 = response_minus_cue - 2.2
    timing_offsets = pd.DataFrame([
        {"assumed_target_offset_s": 2.0, "median_marker_minus_assumed_target_s": float(alt_rt_20.median()), "n_under_100ms": int((alt_rt_20 < .1).sum()), "status": "sensitivity only; not selected by plausibility"},
        {"assumed_target_offset_s": 2.2, "median_marker_minus_assumed_target_s": float(alt_rt_22.median()), "n_under_100ms": int((alt_rt_22 < .1).sum()), "status": "protocol schedule assumption; no per-trial target marker"},
    ])
    timing_offsets.to_csv(OUT / "response_timing_anchor_sensitivity.csv", index=False, encoding="utf-8-sig")
    response_mapping = response_mapping_audit(trials)
    response_mapping.to_csv(OUT / "response_code_mapping.csv", index=False, encoding="utf-8-sig")

    effects = record_effects(states)
    effects.to_csv(OUT / "state_proxy_effects_by_record.csv", index=False, encoding="utf-8-sig")
    effect_plot = OUT / "state_proxy_effects_by_record.png"
    render_effect_plot(effects, effect_plot)

    lag_diagnostics = record_lag_diagnostics(trials)
    raw_mat_files = sorted(MAT_DIR.glob("VisualCog*.mat"))
    mat_by_name = {p.stem: p for p in raw_mat_files}
    file_manifest = []
    for _, row in records.iterrows():
        record = row["record"]
        path = mat_by_name.get(record)
        file_manifest.append({
            "record": record,
            "raw_mat_path": str(path.resolve()) if path else row.get("raw_path"),
            "raw_mat_sha256": sha256(path) if path else None,
            "raw_mat_bytes": int(path.stat().st_size) if path else None,
            "sample_rate_hz": float(row["raw_sample_rate_hz"]),
            "raw_samples": int(row["raw_samples"]),
            "cue_events": int(row["cue_event_count"]),
            "channel9_response_edges": int(row["response_event_count"]),
            "q1_time_mapped_trials": int(row["q1_timestamp_mapping_count"]),
            "target_time_source": row["target_time_source"],
        })

    state_summary = []
    for (stage, state), part in effects.groupby(["stage", "state_proxy"], sort=False):
        vals = pd.to_numeric(part["cohens_d_right_minus_left"], errors="coerce").dropna()
        state_summary.append({
            "stage": stage,
            "state_proxy": state,
            "n_record_estimates": int(len(vals)),
            "equal_record_mean_cohens_d": float(vals.mean()) if len(vals) else None,
            "records_positive": int((vals > 0).sum()),
            "records_negative": int((vals < 0).sum()),
            "interpretation": "descriptive only; four record groups, no participant-level inference",
        })

    per_record_summary = records[[
        "record", "raw_sample_rate_hz", "cue_event_count", "response_event_count",
        "scheduled_rt_proxy_median_s", "scheduled_rt_proxy_under_100ms_count",
        "response_trial_count", "no_response_marker_count", "target_time_source", "task_type",
    ]].copy().rename(columns={"no_response_marker_count": "cue_intervals_without_action_marker_count"})
    outcome_by_record = trials.groupby("record", sort=True).agg(
        direction_consistent_trials=(
            "cue_response_direction_consistent",
            lambda values: int(pd.to_numeric(values, errors="coerce").eq(1).sum()),
        ),
        direction_inconsistent_trials=(
            "cue_response_direction_consistent",
            lambda values: int(pd.to_numeric(values, errors="coerce").eq(0).sum()),
        ),
        cue_intervals_with_action_marker_count=(
            "has_channel9_action_marker_in_cue_interval",
            lambda values: int(pd.to_numeric(values, errors="coerce").eq(1).sum()),
        ),
        cue_intervals_without_action_marker_count=(
            "has_channel9_action_marker_in_cue_interval",
            lambda values: int(pd.to_numeric(values, errors="coerce").eq(0).sum()),
        ),
        trials_with_declared_code_in_analysis_window_count=(
            "response_present_in_analysis_window",
            lambda values: int(pd.to_numeric(values, errors="coerce").eq(1).sum()),
        ),
    ).reset_index()
    per_record_summary = per_record_summary.merge(
        outcome_by_record, on="record", how="left", validate="one_to_one"
    )
    per_record_summary.to_csv(OUT / "record_event_semantics_summary.csv", index=False, encoding="utf-8-sig")

    source_files = [
        output / "record_audit.csv", output / "trial_table.csv", output / "trial_features.csv",
        output / "state_scores.csv", output / "validation_summary.json", output / "behavior_model_status.json",
        output / "experiments" / "20260925_model_search_v1" / "summary.json",
        output / "feature_definitions.csv", Q3 / "README.md", Q3 / "TODO.md", Q3 / "config.py",
        Q3 / "output" / "continuation_audit" / "data_dictionary.csv",
        Q3 / "output" / "continuation_audit" / "build_continuation_audit.py",
        Q3 / "common.py",
        official_problem, solution_pdf,
    ] + raw_mat_files
    source_manifest = [{"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size} for p in source_files if p.is_file()]

    profile = {
        "analysis_unit": "trial nested within continuous recording",
        "raw_record_count": int(len(records)),
        "raw_viscue_trial_count": int(len(trials)),
        "cue_raw_qc_valid_trial_count": int(len(cue_features)),
        "cue_label_counts_raw_qc": cue_features["cue_side"].value_counts().sort_index().to_dict(),
        "record_level": file_manifest,
        "time_diagnostics": {
            "per_record_trial_order_and_interval_diagnostics": lag_diagnostics,
            "forecasting_panel": "not applicable: the objective is cross-record cue decoding/state description, not future-value forecasting",
            "validation_holdout": "leave one complete recording out; participant-level interpretation is unavailable until file-to-participant mapping is verified",
        },
        "feature_dictionary": "src/C/q3/output/feature_definitions.csv",
        "channel9_response_code_mapping": response_mapping.to_dict(orient="records"),
        "state_proxy_effect_summary": state_summary,
    }
    quality = {
        "status": "PASS_WITH_UNRESOLVED_EVENT_METADATA",
        "cue_trials": int(len(trials)),
        "raw_qc_valid_cue_trials": int(len(cue_features)),
        "q1_timestamp_matched_trials": int(records["q1_timestamp_mapping_count"].sum()),
        "channel9_action_marker_count_total": int(records["response_event_count"].sum()),
        "channel9_raw_code_mapping": response_mapping.to_dict(orient="records"),
        "direction_consistency_labels_present": int(pd.to_numeric(trials["cue_response_direction_consistent"], errors="coerce").notna().sum()),
        "direction_consistent_trials": int(pd.to_numeric(trials["cue_response_direction_consistent"], errors="coerce").eq(1).sum()),
        "direction_inconsistent_trials": int(pd.to_numeric(trials["cue_response_direction_consistent"], errors="coerce").eq(0).sum()),
        "cue_interval_action_marker_count_total": int(pd.to_numeric(trials["cue_interval_action_marker_count"], errors="coerce").fillna(0).sum()),
        "cue_intervals_with_action_marker_count": int(pd.to_numeric(trials["has_channel9_action_marker_in_cue_interval"], errors="coerce").eq(1).sum()),
        "cue_intervals_without_action_marker_count": int(pd.to_numeric(trials["has_channel9_action_marker_in_cue_interval"], errors="coerce").eq(0).sum()),
        "trials_with_declared_code_in_analysis_window_count": int(pd.to_numeric(trials["response_present_in_analysis_window"], errors="coerce").eq(1).sum()),
        "declared_code_samples_in_analysis_window_total": int(pd.to_numeric(trials["response_declared_code_sample_count_in_analysis_window"], errors="coerce").fillna(0).sum()),
        "analysis_window_s_relative_to_channel8_cue": [-1.0, 5.0],
        "analysis_window_rule": "count DataLabel-declared channel-9 code samples in cue-relative [-1,+5] s; timeliness is classified separately using cue+3.0 s",
        "task_correctness_status": "derived_by_user_supplied_same_direction_rule",
        "task_correct_count": int(pd.to_numeric(trials["task_correct"], errors="coerce").eq(1).sum()),
        "task_incorrect_count": int(pd.to_numeric(trials["task_correct"], errors="coerce").eq(0).sum()),
        "actual_lateness_status": "classified_by_user_supplied_cue_plus_3s_rule",
        "timely_response_count": int(pd.to_numeric(trials["timely_response"], errors="coerce").eq(1).sum()),
        "not_timely_response_count": int(pd.to_numeric(trials["timely_response"], errors="coerce").eq(0).sum()),
        "formal_deadline_data_available": "deadline_s" in trials and pd.to_numeric(trials["deadline_s"], errors="coerce").notna().any(),
        "user_timeliness_cutoff_s_after_cue": 3.0,
        "target_time_per_trial_marker_present": False,
        "target_time_source": "cue duration about 0.2 s plus about 2 s after cue disappearance per official task appendix; exact onset remains unmarked",
        "assumed_2p2_rt_median_s": float(valid_rt.median()) if len(valid_rt) else None,
        "assumed_2p2_rt_under_100ms": int((valid_rt < .1).sum()),
        "assumed_2p0_rt_median_s": float(alt_rt_20.median()),
        "response_duration_rule": "duration of the first contiguous channel-9 nonzero action bout, sample count / sample rate; not target-onset-to-action-onset reaction time",
        "missingness_note": "Correctness and timeliness are available from the supplied same-direction and cue+3.0 s rules. Target-relative reaction time remains unknown without a per-trial target onset; the cue-relative code-count window remains separate.",
        "data_units_note": "Raw EEG unit is not independently documented in the MAT metadata.",
    }
    cleaning = [
        {"action": "preserve raw MAT inputs", "reason": "auditability; source bytes are hashed in processed_data_manifest"},
        {"action": "read raw F3/Fz/F4 EEG for Q3 features", "reason": "Q1 clean data are only a timestamp/quality mapping and matched-sample reference"},
        {"action": "filter continuous records before epoching", "reason": "ERP 0.5-30 Hz and time-frequency 1-80 Hz branches use documented zero-phase SOS filters"},
        {"action": "flag nonfinite, hard-clipped and flatline epochs", "reason": "quality flag retains exclusion reason; no silent interpolation"},
        {"action": "map channel-9 raw -2/-1 to -2, 0 to 0, and +1/+2 to +2 while retaining raw values", "reason": "official task appendix defines negative as left and positive as right; Action/TgtAct use different code magnitudes"},
        {"action": "keep cue+2.2 s as an explicitly assumed target anchor", "reason": "the task describes an approximate schedule but the permitted channels have no per-trial target-onset marker"},
        {"action": "decode the DataLabel-declared channel-9 code, record cue-response direction consistency and action-marker counts, and count declared codes in the custom analysis window; do not assign task-correctness or lateness labels", "reason": "Task-2 cue-to-target mapping, per-trial target onset, and the experiment's formal deadline are not verified"},
    ]
    eda = {
        "cue_classifier": {
            "raw_qc_baseline_ba": cue_result["baseline_mean_balanced_accuracy"],
            "raw_qc_nested_ba": cue_result["nested_selected_mean_balanced_accuracy"],
            "raw_qc_delta_pp": cue_result["delta_percentage_points"],
            "n_trials": cue_result["n_trials"],
            "outer_groups": cue_result["valid_record_folds"],
            "q1_matched_delta_pp": matched_cue["delta_percentage_points"],
            "target_stage_delta_pp": target_result["delta_percentage_points"],
            "warning": "52.06% is cue-side balanced accuracy in one exploratory nested search, not correctness, diagnosis, or the full Q3 model score",
        },
        "v_h_p_cue_side_effects_by_record": effects.loc[effects["stage"].eq("cue_locked")].to_dict(orient="records"),
        "v_h_p_equal_record_descriptives": state_summary,
        "response_timing_anchor_sensitivity": timing_offsets.to_dict(orient="records"),
        "behavior_choice_model": behavior["model_comparison"]["mean_balanced_accuracy"],
        "ddm_status": behavior["ddm_status"],
        "ddm_gate": behavior["rt_quality"],
        "timing_notes": lag_diagnostics,
        "figures": [str(effect_plot.resolve())],
    }
    leakage = {
        "status": "PASS_FOR_CUE_EEG_CLASSIFIER_FEATURE_MATRIX",
        "target": "VisCue cue_side (-1/+1)",
        "features": selected["feature_sets"]["full_baseline_13"],
        "forbidden_predictor_fields": ["response_raw", "response_code", "choice_side", "response_time_s", "reaction_time_s", "cue_response_direction_consistent", "task_correct", "verified_omission", "VisCue label"],
        "channel9_in_cue_classifier_feature_set": False,
        "cue_label_as_feature": False,
        "folding": selected["outer_validation"],
        "scaling_and_selection": "standardization and candidate selection performed using inner training records only",
        "scope_note": "Response columns remain in the audit trial table as metadata; the explicitly listed EEG feature matrix is separate.",
        "researcher_selection_note": "Several exploratory searches reused these four recording groups; this score is a development estimate and needs untouched-record replication.",
    }
    processed = {
        "analysis_outputs": [
            {"path": str((output / "trial_table.csv").resolve()), "sha256": sha256(output / "trial_table.csv"), "rows": int(len(trials))},
            {"path": str((output / "trial_features.csv").resolve()), "sha256": sha256(output / "trial_features.csv"), "rows": int(len(features))},
            {"path": str((output / "state_scores.csv").resolve()), "sha256": sha256(output / "state_scores.csv"), "rows": int(len(states))},
        ],
        "raw_inputs": file_manifest,
        "source_manifest": source_manifest,
        "derived_files_created": [str((OUT / "state_proxy_effects_by_record.csv").resolve()), str(effect_plot.resolve()), str((OUT / "response_timing_anchor_sensitivity.csv").resolve()), str((OUT / "response_code_mapping.csv").resolve())],
        "raw_data_modified": False,
    }

    (OUT / "data_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    (OUT / "data_quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    (OUT / "cleaning_actions.json").write_text(json.dumps(cleaning, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    (OUT / "eda_findings.json").write_text(json.dumps(eda, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    (OUT / "leakage_report.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    (OUT / "processed_data_manifest.json").write_text(json.dumps(processed, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")

    cue_effects = effects.loc[effects["stage"].eq("cue_locked")]
    effect_table = markdown_table(cue_effects, ["record", "state_proxy", "n_left", "n_right", "cohens_d_right_minus_left"], ["记录", "代理状态", "左提示n", "右提示n", "Cohen d（右−左）"])
    analysis = f'''# 问题三后续分析与数据语义审计

## 1. 本轮完成范围

本轮对照官方题目附录、最新《第三.pdf》和 Q3 当前输出，补充了事件语义审计、按记录的 V/H/P 代理效应量、目标时间锚敏感性及数据/标签泄漏清单。原始 MAT 文件未修改；没有在四份记录上继续追逐更高分类百分点。

## 2. 事件语义与 DDM 门槛

官方题目附录将通道9定义为目标点击应答，负值为左、正值为右；当前 Q3 保留原始值，并将 raw -2/-1 标准化为 -2、raw +1/+2 标准化为 +2。因此，**应答方向语义有题目依据**。附录同时说明提示约呈现 0.2 s，目标在提示消失后约 2 s 出现。结合该排程可形成 cue+2.2 s 的近似锚点，但 MAT 通道中没有逐试次目标 onset marker，故目标 onset 和 RT 仍不能视为精确测量。

编码统一规则明确为 **raw -2/-1 → response_code -2（左），raw 0 → 0（无活动），raw +1/+2 → response_code +2（右）**。映射只作用于派生应答码，`response_raw` 和 MAT 原始通道均保留。下表分别统计原始通道采样点、非零起始边沿和试次表事件，避免把持续多个采样点的 marker 重复计成多次应答。

{markdown_table(response_mapping, ["response_raw", "response_code", "meaning", "raw_channel_sample_count", "detected_response_edge_count", "trial_table_event_count"], ["原始码", "标准码", "含义", "原始采样点数", "检测边沿数", "试次表事件数"])}

按 cue+2.2 s 计算，通道9事件相对目标锚点的 RT 中位数为 **{float(valid_rt.median()):.3f} s**，其中 **{int((valid_rt < .1).sum())}/{len(valid_rt)}** 小于 0.1 s。若把目标暂按 cue+2.0 s 计算，中位数变为 **{float(alt_rt_20.median()):.3f} s**，但这只是时间锚敏感性，不是从行为合理性反推目标时刻。项目截止时间、项目2目标真值和漏答标签仍缺失，故本轮维持 **DDM 未拟合**，也不生成正确率/漏答率结论。

## 3. V/H/P 代理状态的跨记录描述

现有状态代理定义为：V=三通道额前 ERP 均值，H=三通道 log-theta 功率均值，P=三通道 log-beta 功率均值。它们是可审计的观测特征组合，不是由自由载荷矩阵估计的潜变量，也不支持 LGN、海马或 PFC 的源定位解释。下表给出每个记录内左右提示差异的 Cohen d；四个记录单独展示，不把试次数当成独立被试来计算显著性。

{effect_table}

![V/H/P代理状态左右提示效应图](state_proxy_effects_by_record.png)

*图1　四个记录中 V/H/P 代理状态的左右提示效应量。目标锁定面板使用 cue+2.2 s 排程假设，仅作敏感性描述。*

当前既有留记录消融显示，cue 阶段完整代理加任务候选的 BA 为 0.489，V-only 为 0.494（变化 −0.005）；target 阶段对应为 0.530 与 0.497（变化 +0.033）。特征重建误差有所下降，但它说明代理特征之间存在可预测关系，不能证明三个潜在脑区状态被识别。任务后缀的正式含义也尚未由逐记录元数据确认。

## 4. 已有解码与行为结果的正确位置

raw-QC cue 左右提示方向的嵌套分类 BA 为 **{cue_result['nested_selected_mean_balanced_accuracy']:.4f}**，固定 Logistic 基线为 **{cue_result['baseline_mean_balanced_accuracy']:.4f}**，差值 **+{cue_result['delta_percentage_points']:.2f} 个百分点**。在 Q1 质量匹配样本上，差值为 **{matched_cue['delta_percentage_points']:.2f} 个百分点**；target 锁定差值为 **{target_result['delta_percentage_points']:.2f} 个百分点**。因此 52.06% 仍是对样本口径敏感的探索性 cue 解码分支。

行为选择比较中，提示+文件名任务候选的 BA 为 **{behavior['model_comparison']['mean_balanced_accuracy']['cue_plus_task_code']:.3f}**，加入 EEG 代理后为 **{behavior['model_comparison']['mean_balanced_accuracy']['cue_task_plus_EEG']:.3f}**，当前没有观察到 EEG 增益。它预测的是通道9左右选择，既不表示正确率，也不能验证完整 DDM。

## 5. 后续执行顺序

1. 获取逐记录的被试/会话/项目映射、目标 onset 时间戳、截止时间和项目2目标真值；记录来源并与通道9事件逐条核对。
2. 若有可信元数据，重算 response-locked 窗口与 RT；否则保留本报告的代理状态与 cue 解码，并明确目标/应答窗口不可定量验证。
3. 仅当正确/错误/漏答、RT 与截止时间都可信时，先做 DDM 参数恢复模拟，再拟合并按独立被试或记录留出验证。缺任何一项时就把 DDM 报告为数据条件不足。

## 6. 可复现产物

- `state_proxy_effects_by_record.csv`：按记录、事件阶段和状态代理列出左右提示均值、Cohen d 与样本量。
- `response_timing_anchor_sensitivity.csv`：cue+2.0 s 与 cue+2.2 s 两种锚点下的应答时间差异；不用于挑选最佳锚点。
- `response_code_mapping.csv`：逐原始编码的标准码、含义，以及原始通道采样点数/响应边沿数/试次事件数。
- `data_profile.json`、`data_quality_report.json`、`cleaning_actions.json`、`eda_findings.json`、`leakage_report.json` 和 `processed_data_manifest.json`：本轮审计和计算来源。
'''
    (OUT / "data_analysis.md").write_text(analysis, encoding="utf-8")

    print(json.dumps({
        "output_dir": str(OUT.resolve()),
        "cue_trials": int(len(trials)),
        "cue_raw_qc": int(len(cue_features)),
        "rt_median_cue_plus_2_2": float(valid_rt.median()),
        "rt_median_cue_plus_2_0": float(alt_rt_20.median()),
        "state_effect_rows": int(len(effects)),
        "main_ba": cue_result["nested_selected_mean_balanced_accuracy"],
        "files": sorted(p.name for p in OUT.iterdir() if p.is_file()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
