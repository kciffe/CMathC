# Expanded feature and model search

This is an exploratory, development-only nested leave-one-record-out search. The frozen baseline remains unchanged.

| Sample | Stage | Baseline BA | Nested selected BA | Delta |
|---|---|---:|---:|---:|
| raw_qc_pass | cue_locked | 0.4601 | 0.4623 | +0.22 pp |
| raw_qc_pass | target_locked | 0.5225 | 0.5020 | -2.05 pp |
| q1_quality_matched | cue_locked | 0.4868 | 0.5034 | +1.67 pp |
| q1_quality_matched | target_locked | 0.5104 | 0.5093 | -0.12 pp |

## Interpretation

The first-round outer records have already been viewed in previous searches. Treat every score as development evidence; do not report the maximum outer-candidate score as an unbiased estimate.
Feature selection and scaling are re-fitted inside each inner and outer training split. Response-conditioned pre-response windows and behavioral labels are excluded.

## Limitations

- The same four records were inspected in previous exploration rounds; this remains development-only and needs independent record-level replication.
- Candidate expansion itself creates selection optimism even though every candidate is evaluated in nested folds.
- Gamma power is exploratory; the feature set remains limited to frontal F3/Fz/F4 channels.
- Results measure cue-side prediction and do not establish correctness, omission rate, or clinical diagnosis.
