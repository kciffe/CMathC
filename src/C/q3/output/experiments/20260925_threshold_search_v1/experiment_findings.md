# Nested threshold calibration findings

Thresholds were selected from inner held-record scores; outer records were scored after selection.

| Sample | Stage | Baseline BA | Nested threshold BA | Delta |
|---|---|---:|---:|---:|
| raw_qc_pass | cue_locked | 0.4601 | 0.5022 | +4.21 pp |
| q1_quality_matched | cue_locked | 0.4868 | 0.4701 | -1.67 pp |
| raw_qc_pass | target_locked | 0.5225 | 0.5060 | -1.65 pp |
| q1_quality_matched | target_locked | 0.5104 | 0.5141 | +0.36 pp |

See fold-level selections and thresholds in `nested_selected_fold_metrics.csv`.

## Limits

- This threshold search follows several exploratory rounds on the same four outer records; it is a development result and is prone to selection optimism.
- Only probability-producing Logistic/LDA models use the threshold grid; SVM margins retain their fixed native zero threshold.
- Only four recording-level folds are available, so results are uncertain and need an independent recording-level replication.
