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
        "response_labels": "channel 9 zero-to-nonzero edge is t_act; response side is decoded from the L/R code declared in DataLabel within the same nonzero bout; raw codes are preserved",
        "response_channel_scope": "t_act and response side metadata / endpoint definition only; excluded from EEG predictors and features",
        "correctness": "channel 8 VisCue gives target side; compare with the side decoded from channel 9's DataLabel-declared code within the first action bout",
        "correct_trials": int(pd.to_numeric(event_table["correct"], errors="coerce").eq(1).sum()),
        "incorrect_trials": int(pd.to_numeric(event_table["correct"], errors="coerce").eq(0).sum()),
        "response_bouts_with_unresolved_side": int(
            (
                pd.to_numeric(event_table["response_event_count"], errors="coerce").fillna(0).gt(0)
                & event_table["choice_side"].isna()
            ).sum()
        ),
        "omission_trials": int(pd.to_numeric(event_table["is_omission"], errors="coerce").eq(1).sum()),
        "omission_definition": "no channel-9 action bout in the VisCue onset-to-next-VisCue interval",
        "timeliness_window_s_relative_to_channel8_cue": [-1.0, 5.0],
        "timeliness_rule": "timely when a channel-9 code declared in DataLabel (Action +/-1 or TgtAct +/-2) appears in the cue-centered window; an action in the cue-to-next-cue interval without that code in the window is late",
        "timely_trials": int(pd.to_numeric(event_table["is_timely"], errors="coerce").eq(1).sum()),
        "late_trials": int(pd.to_numeric(event_table["is_late"], errors="coerce").eq(1).sum()),
        "timeliness_unresolved_trials": int(pd.to_numeric(event_table["is_timely"], errors="coerce").isna().sum()),
        "response_bout_duration_status": "duration of the first contiguous channel-9 nonzero bout, measured as sample count / sampling rate; not target-to-response reaction time",
        "response_bout_duration_median_s": float(pd.to_numeric(event_table["response_duration_s"], errors="coerce").median()),
        "reaction_time": "duration remains unknown without a verified per-trial target onset; cue+2.2s is a schedule proxy only",
    }
    write_json(summary, output_dir / "event_audit_summary.json")
    print(f"Wrote {len(event_table)} cue events and {len(mapping_table)} clean-trial mappings to {output_dir}")
    for row in record_summaries:
        print(
            f"{row['record']}: cues={row['cue_event_count']}, "
            f"Q1 clean={row['q1_clean_trial_count']}, "
            f"timestamp matches={row['q1_timestamp_mapping_count']}, "
            f"responses={row['response_trial_count']}, "
            f"no response marker={row['no_response_marker_count']}, "
            f"timely={int(pd.to_numeric(event_table.loc[event_table['record'].eq(row['record']), 'is_timely'], errors='coerce').eq(1).sum())}, "
            f"late={int(pd.to_numeric(event_table.loc[event_table['record'].eq(row['record']), 'is_late'], errors='coerce').eq(1).sum())}, "
            f"median schedule-RT proxy={row['scheduled_rt_proxy_median_s']:.3f}s, "
            f"proxy<100ms={row['scheduled_rt_proxy_under_100ms_count']}"
        )


if __name__ == "__main__":
    main()
