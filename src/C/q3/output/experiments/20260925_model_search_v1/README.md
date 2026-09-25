# Exploratory Q3 model search (2026-09-25)

This search is separate from the primary Q3 pipeline. The frozen reference is `src/C/q3/baselines/20260925_initial/`.

## Protocol

- Primary metric: balanced accuracy; all results also include accuracy, AUC, macro-F1, and fold-level values.
- Outer split: leave one entire recording file out. Inner selection: leave one of the three outer-training records out.
- Scaling is fit only on each training partition. Candidate choice is made only from inner-fold balanced accuracy; the outer record is scored once after selection.
- The fixed baseline is the same 13-feature regularized Logistic model used in the original Q3 validation. Candidate families are Logistic with alternative regularization/feature sets, robust scaling, shrinkage LDA, linear SVM, and RBF SVM.
- `AI_alpha` is a linear contrast of the three alpha log-power columns. The candidate set includes a version that drops this redundant column.
- No threshold tuning or target-offset selection is performed.

## Interpretation

The nested-selected model is the main estimate of the model-selection procedure. The fixed-candidate table is exploratory; do not select a model based on its outer-fold maximum and claim that maximum as an unbiased score. Four outer records are too few to establish stable generalization.

See `summary.json`, `nested_selected_fold_metrics.csv`, `outer_candidate_summary.csv`, and the out-of-fold prediction files.
