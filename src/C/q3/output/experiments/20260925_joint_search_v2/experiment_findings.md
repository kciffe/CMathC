# Joint model and temporal feature search

This is a development-only nested search. The original baseline snapshot is unchanged.

| Sample | Stage | Baseline BA | Joint nested BA | Delta |
|---|---|---:|---:|---:|
| raw_qc_pass | cue_locked | 0.4601 | 0.5025 | +4.24 pp |
| q1_quality_matched | cue_locked | 0.4868 | 0.4520 | -3.47 pp |
| raw_qc_pass | target_locked | 0.5225 | 0.4978 | -2.47 pp |
| q1_quality_matched | target_locked | 0.5104 | 0.4816 | -2.89 pp |

Per-record selections are in `nested_selected_fold_metrics.csv`. Candidate outer maxima are exploratory and must not be presented as unbiased results.

## Limitations

- This retrospective union combines two exploratory candidate grids after both component rounds had been inspected.
- The same four outer records were viewed in earlier rounds; even nested scores are development estimates, not independent confirmation.
- Only four recording-level folds are available, and fold-to-fold variability is large.
- Results measure cue-side prediction from EEG and do not establish clinical diagnostic performance.
