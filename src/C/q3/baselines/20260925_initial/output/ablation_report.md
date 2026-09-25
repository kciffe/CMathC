# V/H/P ablation results

Evaluation uses leave-one-record-out folds, with all scaling estimated from the three training records. The Q1-retained subset is reported as a matched-sample sensitivity check.

## Cue-side prediction

Balanced accuracy and AUC are in `ablation_cv_metrics.csv`. Models use channel-8 cue direction as the target; cue direction is not included in the EEG proxy predictors. The V-only versus V-plus-gamma models provide the gamma inclusion sensitivity check; PAC remains unavailable.

## Feature reconstruction

`ablation_reconstruction_metrics.csv` reports held-out standardized RMSE by target feature. Each target is removed from its proxy anchor before the model is fitted. AIC/BIC are auxiliary likelihood summaries, not significance tests.

## Interpretation limits

V/H/P are observed feature composites used because three electrodes do not identify a free sparse measurement matrix. Task dependence is exploratory and uses the Task-1/Task-2 filename suffix, whose formal project meaning remains unverified. Four recording files do not establish participant-level generalization.
