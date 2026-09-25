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
    trial_table["response_pre_window_available"] = trial_table["response_time_s"].notna()
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
    summary = {
        "raw_cue_trial_count": int(len(trial_table)),
        "q1_timestamp_matched_count": int(trial_table["q1_trial_index"].notna().sum()),
        "q1_quality_pass_count": int(trial_table["q1_quality_pass"].sum()),
        "channel9_marker_count": int(trial_table["choice_side"].notna().sum()),
        "channel9_no_marker_count": int(trial_table["is_omission"].fillna(False).sum()),
        "q1_usage": "event timestamp mapping and quality labels only; no Q1 EEG samples are read or used as Q3 feature input",
        "channel9_usage": "event direction/time metadata; never included in EEG feature values",
        "behavior_label_caveat": "channel-9 marker absence is not declared a behavioral omission without a verified response deadline",
    }
    write_json(summary, output_dir / "trial_table_build_summary.json")
    print(
        f"Wrote {len(trial_table)} trial rows; Q1 timestamp/quality matches="
        f"{summary['q1_quality_pass_count']}; channel-9 markers="
        f"{summary['channel9_marker_count']}."
    )


if __name__ == "__main__":
    main()
