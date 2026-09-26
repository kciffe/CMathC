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
            "correct",
            "is_omission",
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
    omission_labels = pd.to_numeric(
        trial_table["is_omission"],
        errors="coerce",
    )
    correctness_labels = pd.to_numeric(trial_table["correct"], errors="coerce")
    external_correctness = pd.to_numeric(
        trial_table["external_correct"]
        if "external_correct" in trial_table
        else pd.Series(np.nan, index=trial_table.index),
        errors="coerce",
    )
    external_omissions = pd.to_numeric(
        trial_table["external_is_omission"]
        if "external_is_omission" in trial_table
        else pd.Series(np.nan, index=trial_table.index),
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
        "correct_count": int(correctness_labels.eq(1).sum()),
        "incorrect_count": int(correctness_labels.eq(0).sum()),
        "correctness_labeled_count": int(correctness_labels.notna().sum()),
        "omission_count": int(omission_labels.eq(1).sum()),
        "omission_labeled_count": int(omission_labels.notna().sum()),
        "timely_count": int(pd.to_numeric(trial_table["is_timely"], errors="coerce").eq(1).sum()),
        "late_count": int(pd.to_numeric(trial_table["is_late"], errors="coerce").eq(1).sum()),
        "timeliness_unresolved_count": int(pd.to_numeric(trial_table["is_timely"], errors="coerce").isna().sum()),
        "response_duration_labeled_count": int(pd.to_numeric(trial_table["response_duration_s"], errors="coerce").notna().sum()),
        "response_duration_median_s": float(pd.to_numeric(trial_table["response_duration_s"], errors="coerce").median()),
        "external_correctness_disagreement_count": int(
            (external_correctness.notna() & correctness_labels.notna() & external_correctness.ne(correctness_labels)).sum()
        ),
        "external_omission_disagreement_count": int(
            (external_omissions.notna() & omission_labels.notna() & external_omissions.ne(omission_labels)).sum()
        ),
        "q1_usage": "event timestamp mapping and quality labels only; no Q1 EEG samples are read or used as Q3 feature input",
        "channel9_usage": "t_act is the first zero-to-nonzero edge; response side is decoded using the channel-specific L/R code in DataLabel within that bout; never included in EEG predictor/features",
        "correctness_rule": "correct=1 when the channel-9 side decoded from its DataLabel-declared code matches channel-8 VisCue target side; correct=0 otherwise; no-response or undecodable trials have correctness=null",
        "omission_rule": "omission=1 when no channel-9 action bout occurs between this VisCue onset and the next VisCue onset",
        "timeliness_window_s_relative_to_channel8_cue": [-1.0, 5.0],
        "timeliness_rule": "timely when the response code declared in the channel-9 DataLabel occurs in the cue-centered window; Task-1 Action uses +/-1 and Task-2 TgtAct uses +/-2; otherwise an action in the cue interval is late",
        "response_duration_rule": "duration of the first contiguous channel-9 nonzero bout, measured as sample count / sample rate; this is not target-to-response reaction time",
        "behavior_label_caveat": "correctness, omission, timeliness, and channel-9 bout duration are operational labels from the supplied channel semantics; target-onset reaction time remains unavailable",
    }
    write_json(summary, output_dir / "trial_table_build_summary.json")
    print(
        f"Wrote {len(trial_table)} trial rows; Q1 timestamp/quality matches="
        f"{summary['q1_quality_pass_count']}; channel-9 markers="
        f"{summary['channel9_marker_count']}."
    )


if __name__ == "__main__":
    main()
