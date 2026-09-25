# Temporal ERP / CSP search findings

This is an exploratory development report. The frozen initial baseline remains under `baselines/20260925_initial/`.

| Sample | Stage | Baseline BA | Nested selected BA | Delta |
|---|---|---:|---:|---:|
| raw_qc_pass | cue_locked | 0.4601 | 0.4911 | +3.10 pp |
| q1_quality_matched | cue_locked | 0.4868 | 0.4063 | -8.05 pp |
| raw_qc_pass | target_locked | 0.5225 | 0.5050 | -1.75 pp |
| q1_quality_matched | target_locked | 0.5104 | 0.4570 | -5.34 pp |

## Per-record selection

The selected candidate for each held-out record is listed in `nested_selected_fold_metrics.csv`; use those rows and the OOF files when discussing consistency.
Do not select a winner from the outer candidate maximum and present it as an unbiased estimate.

## Limits

- The same four recording files were inspected in the previous exploratory round; these scores are development-only and require independent replication.
- Only four outer recording folds are available, so record-level scores have high uncertainty.
- Cue side is the prediction label at both event anchors; the score measures side information, not clinical diagnosis.
- Target-locked epochs use the scheduled cue+2.2 s event anchor.
