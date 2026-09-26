"""Build the Q3 trial table from the event audit and Q1 quality mapping only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import ensure_output_dir, write_csv, write_json
from config import BEHAVIOR_LABELS_PATH


def attach_optional_behavior_labels(
    trial_table: pd.DataFrame, labels_path
) -> pd.DataFrame:
    if not labels_path.exists():
        return trial_table
    behavior = pd.read_csv(labels_path, encoding="utf-8-sig")
    required_keys = {"record", "original_trial_index"}
    missing_keys = sorted(required_keys.difference(behavior.columns))
    if missing_keys:
        raise ValueError(f"Behavior labels need join keys {missing_keys}")
    if behavior.duplicated(["record", "original_trial_index"]).any():
        raise ValueError("Behavior labels contain duplicate record/trial keys")
    columns_to_join = [
        column
        for column in (
            "record",
            "original_trial_index",
            "choice_side",
            "response_side",
            "reaction_time_s",
            "cue_response_direction_consistent",
            "cue_interval_action_marker_count",
            "deadline_s",
        )
        if column in behavior.columns
    ]
    external = behavior[columns_to_join].rename(
        columns={
            column: f"external_{column}"
            for column in columns_to_join
            if column not in {"record", "original_trial_index"}
        }
    ).copy()
    external["original_trial_index"] = pd.to_numeric(
        external["original_trial_index"], errors="coerce"
    )
    external["record"] = external["record"].astype(str)
    external["_external_row_present"] = True
    trial_table = trial_table.merge(
        external,
        on=["record", "original_trial_index"],
        how="left",
        validate="one_to_one",
    )
    trial_table["external_behavior_row_present"] = trial_table[
        "_external_row_present"
    ].fillna(False)
    trial_table["behavior_label_status"] = np.where(
        trial_table["_external_row_present"].fillna(False),
        trial_table["behavior_label_status"] + ";external_row_joined",
        trial_table["behavior_label_status"],
    )
    return trial_table.drop(columns=["_external_row_present"])


def main() -> None:
    output_dir = ensure_output_dir()
    audit_path = output_dir / "event_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError("Run 01_audit_events.py first to create event_audit.csv")
    trial_table = pd.read_csv(audit_path, encoding="utf-8-sig")

    # Q1 clean epochs contribute only the timestamp/quality mapping already
    # audited by script 01. No Q1 EEG array is read or copied here.
    trial_table["eeg_trial_available"] = trial_table["q1_trial_index"].notna()
    trial_table["q1_quality_pass"] = (
        trial_table["q1_trial_index"].notna()
        & pd.to_numeric(trial_table["q1_final_drop"], errors="coerce").eq(0)
    )
    if "t_act_s" not in trial_table and "response_time_s" in trial_table:
        trial_table["t_act_s"] = trial_table["response_time_s"]
    trial_table["response_pre_window_available"] = trial_table["t_act_s"].notna()
    if "response_marker_present" not in trial_table:
        trial_table["response_marker_present"] = trial_table["t_act_s"].notna()
    if "deadline_s" not in trial_table:
        trial_table["deadline_s"] = np.nan
    trial_table = attach_optional_behavior_labels(trial_table, BEHAVIOR_LABELS_PATH)
    write_csv(trial_table, output_dir / "trial_table.csv")

    quality_columns = [
        column
        for column in (
            "record",
            "original_trial_index",
            "q1_trial_index",
            "q1_trial_key",
            "q1_mapping_error_s",
            "q1_final_drop",
            "eeg_quality",
            "q1_quality_pass",
        )
        if column in trial_table.columns
    ]
    write_csv(trial_table[quality_columns], output_dir / "q1_quality_mapping_reference.csv")
    direction_consistency = pd.to_numeric(
        trial_table["cue_response_direction_consistent"], errors="coerce"
    )
    task_correct = pd.to_numeric(trial_table["task_correct"], errors="coerce")
    timely_response = pd.to_numeric(trial_table["timely_response"], errors="coerce")
    action_marker_count = pd.to_numeric(
        trial_table["cue_interval_action_marker_count"], errors="coerce"
    ).fillna(0)
    declared_code_count = pd.to_numeric(
        trial_table["response_declared_code_sample_count_in_analysis_window"],
        errors="coerce",
    )
    summary = {
        "raw_cue_trial_count": int(len(trial_table)),
        "q1_timestamp_matched_count": int(trial_table["q1_trial_index"].notna().sum()),
        "q1_quality_pass_count": int(trial_table["q1_quality_pass"].sum()),
        "channel9_marker_count": int(
            pd.to_numeric(trial_table["response_event_count"], errors="coerce").fillna(0).gt(0).sum()
        ),
        "channel9_decoded_choice_count": int(trial_table["choice_side"].notna().sum()),
        "channel9_marker_with_unresolved_side_count": int(
            (
                pd.to_numeric(trial_table["response_event_count"], errors="coerce").fillna(0).gt(0)
                & trial_table["choice_side"].isna()
            ).sum()
        ),
        "channel9_no_marker_observed_count": int((~trial_table["response_marker_present"].fillna(False)).sum()),
        "cue_response_direction_consistent_count": int(direction_consistency.eq(1).sum()),
        "cue_response_direction_inconsistent_count": int(direction_consistency.eq(0).sum()),
        "direction_consistency_labeled_count": int(direction_consistency.notna().sum()),
        "task_correct_count": int(task_correct.eq(1).sum()),
        "task_incorrect_count": int(task_correct.eq(0).sum()),
        "task_correctness_unresolved_count": int(task_correct.isna().sum()),
        "timely_response_count": int(timely_response.eq(1).sum()),
        "late_or_missing_by_deadline_count": int(timely_response.eq(0).sum()),
        "timeliness_unresolved_count": int(timely_response.isna().sum()),
        "cue_intervals_with_action_marker_count": int(action_marker_count.gt(0).sum()),
        "cue_intervals_without_action_marker_count": int(action_marker_count.eq(0).sum()),
        "trials_with_declared_code_in_analysis_window": int(declared_code_count.gt(0).sum()),
        "response_duration_labeled_count": int(pd.to_numeric(trial_table["response_duration_s"], errors="coerce").notna().sum()),
        "response_duration_median_s": float(pd.to_numeric(trial_table["response_duration_s"], errors="coerce").median()),
        "q1_usage": "event timestamp mapping and quality labels only; no Q1 EEG samples are read or used as Q3 feature input",
        "channel9_usage": "t_act is the first zero-to-nonzero edge; response side is decoded using the channel-specific L/R code in DataLabel within that bout; never included in EEG predictor/features",
        "correctness_rule": "same direction sign on channel-8 cue and decoded channel-9 response is correct; opposite signs are incorrect, per the user-specified rule",
        "timeliness_rule": "channel-9 onset by cue+3.0 s is timely; later or no response by a fully observed deadline is not timely",
        "cue_interval_action_marker_rule": "count channel-9 action edges between this VisCue onset and the next VisCue onset; do not equate marker absence with an experiment-level omission",
        "analysis_window_s_relative_to_channel8_cue": [-1.0, 5.0],
        "analysis_window_rule": "cue-relative [-1,+5] s counts declared channel-9 codes only; it is separate from the cue+3.0 s timeliness rule",
        "task_correctness_status": "derived_from_user_supplied_same_direction_rule; unresolved only when cue/response side cannot be decoded",
        "timeliness_status": "derived_from_user_supplied_cue_plus_3s_deadline; unresolved only when the recording does not cover the deadline",
        "target_relative_reaction_time_status": "unknown_without_per_trial_target_onset; cue-to-response latency is separately observed",
        "response_duration_rule": "duration of the first contiguous channel-9 nonzero bout, measured as sample count / sample rate; this is not target-to-response reaction time",
        "behavior_label_caveat": "correctness and timeliness use user-specified channel-8/channel-9 and cue+3.0s rules; target-relative reaction time still requires a verified target onset",
    }
    write_json(summary, output_dir / "trial_table_build_summary.json")
    print(
        f"Wrote {len(trial_table)} trial rows; Q1 timestamp/quality matches="
        f"{summary['q1_quality_pass_count']}; channel-9 markers="
        f"{summary['channel9_marker_count']}."
    )


if __name__ == "__main__":
    main()
