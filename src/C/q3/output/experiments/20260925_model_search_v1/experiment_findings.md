# Exploratory model search findings

The frozen initial baseline is unchanged. This comparison evaluates the nested model-selection procedure against the fixed 13-feature Logistic baseline on the same outer leave-one-record-out folds.

## Main results

| Training/evaluation sample | Stage | Baseline BA | Nested-selected BA | Change | Per-fold changes |
|---|---|---:|---:|---:|---|
| Raw-QC full sample | Cue | 0.460 | 0.521 | +6.05 pp | VisualCogA_Task-1: +5.3 pp, VisualCogA_Task-2: +11.1 pp, VisualCogB_Task-1: +4.3 pp, VisualCogB_Task-2: +3.6 pp |
| Raw-QC full sample | Target | 0.522 | 0.501 | -2.15 pp | VisualCogA_Task-1: -1.7 pp, VisualCogA_Task-2: -1.6 pp, VisualCogB_Task-1: -6.4 pp, VisualCogB_Task-2: +1.0 pp |
| Q1 quality-matched sensitivity | Cue | 0.487 | 0.487 | -0.02 pp | VisualCogA_Task-1: -3.4 pp, VisualCogA_Task-2: +1.0 pp, VisualCogB_Task-1: -1.2 pp, VisualCogB_Task-2: +3.6 pp |
| Q1 quality-matched sensitivity | Target | 0.510 | 0.497 | -1.34 pp | VisualCogA_Task-1: -3.4 pp, VisualCogA_Task-2: +0.6 pp, VisualCogB_Task-1: -2.2 pp, VisualCogB_Task-2: -0.4 pp |

The raw-QC cue improvement is positive in all four held-out records, but the resulting BA is still about 0.521. It does not transfer to the analysis retrained only on the 297 Q1-matched trials. The target stage decreases under nested selection.

## Quality-subgroup check

These rows evaluate predictions from models trained and selected on each complete raw-QC training fold, then split the held-out predictions by Q1 quality status. This is a diagnostic subgroup analysis; it does not change model selection.

| Stage | Held-out Q1 status | Trials | Baseline BA | Nested-selected BA | Change |
|---|---|---:|---:|---:|---:|
| Cue | Q1 retained | 297 | 0.442 | 0.542 | +9.96 pp |
| Cue | Q1 excluded, raw-QC passed | 90 | 0.520 | 0.473 | -4.65 pp |
| Target | Q1 retained | 297 | 0.546 | 0.505 | -4.08 pp |
| Target | Q1 excluded, raw-QC passed | 99 | 0.427 | 0.496 | +6.89 pp |

For raw-QC-trained cue models, the improvement is concentrated in the Q1-retained held-out subset; performance falls on the Q1-excluded/raw-QC-passed subset. Yet when both training and evaluation are restricted to Q1-matched trials, nested selection yields essentially no cue gain. This makes the observed improvement sensitive to the training-sample definition.

## Model choices and limits

- The candidate set contains 49 fixed configurations across Logistic, shrinkage LDA, linear SVM, and RBF SVM, with predeclared feature sets and regularization/kernel values.
- Selected configuration names differ across held-out records, so this does not identify one stable replacement classifier.
- Four recording files are only four outer validation groups. These results are exploratory, with no confidence interval or significance claim.
- Do not choose a model by the highest outer-fold candidate score; use the nested-selected row as the estimate of the inner selection procedure.
- The nominal target time remains cue+2.2 s by schedule assumption; target-stage results remain timing-sensitive.

Detailed files: `summary.json`, `nested_selected_fold_metrics.csv`, `outer_candidate_summary.csv`, `quality_subgroup_metrics.csv`, and `quality_subgroup_summary.json`.
