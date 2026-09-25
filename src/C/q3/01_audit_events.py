"""Audit VisCue events and timestamp-match Q1 quality-retained trials."""

from __future__ import annotations

import pandas as pd

from common import build_record_event_tables, ensure_output_dir, write_csv, write_json
from config import RECORDS


def main() -> None:
    output_dir = ensure_output_dir()
    event_tables: list[pd.DataFrame] = []
    mapping_tables: list[pd.DataFrame] = []
    record_summaries: list[dict] = []

    for record in RECORDS:
        events, mappings, summary = build_record_event_tables(record)
        event_tables.append(events)
        mapping_tables.append(mappings)
        record_summaries.append({key: value for key, value in summary.items() if key != "clean"})

    event_table = pd.concat(event_tables, ignore_index=True)
    mapping_table = pd.concat(mapping_tables, ignore_index=True)
    write_csv(event_table, output_dir / "event_audit.csv")
    write_csv(mapping_table, output_dir / "trial_mapping_audit.csv")
    write_csv(pd.DataFrame(record_summaries), output_dir / "record_audit.csv")

    summary = {
        "records": record_summaries,
        "total_raw_cue_events": int(len(event_table)),
        "total_q1_timestamp_matches": int(event_table["q1_trial_index"].notna().sum()),
        "target_time": "cue_time + 2.2 s (schedule assumption; not verified from an allowed event channel)",
        "response_labels": "channel 9 Action/TgtAct sign standardized to -2/0/+2; raw values are preserved",
        "response_channel_scope": "behavior response side/time only; excluded from EEG feature matrices",
        "correctness": "not inferred without verified target-side truth",
    }
    write_json(summary, output_dir / "event_audit_summary.json")
    print(f"Wrote {len(event_table)} cue events and {len(mapping_table)} clean-trial mappings to {output_dir}")
    for row in record_summaries:
        print(
            f"{row['record']}: cues={row['cue_event_count']}, "
            f"Q1 clean={row['q1_clean_trial_count']}, "
            f"timestamp matches={row['q1_timestamp_mapping_count']}, "
            f"responses={row['response_trial_count']}, "
            f"omission markers={row['omission_marker_count']}, "
            f"median RT={row['rt_median_s']:.3f}s, "
            f"RT<100ms={row['rt_under_100ms_count']}"
        )


if __name__ == "__main__":
    main()
