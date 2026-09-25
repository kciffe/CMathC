# Q3 EEG cognitive model

This pipeline follows the latest Q3 solution with a separate raw-EEG analysis path. It uses F3/Fz/F4 for EEG features, VisCue for cue-side labels, and channel 9/TimeStamp for response-event metadata. Channel 9 is never included in EEG feature arrays or V/H/P state predictors. Q3 does not import the Q2 simulation chain.

## Run order

From the repository root:

```powershell
python src/C/q3/01_audit_events.py
python src/C/q3/02_extract_trials.py
python src/C/q3/02b_preprocess_raw.py
python src/C/q3/03_extract_features.py
python src/C/q3/04_cognitive_state.py
python src/C/q3/05_behavior_model.py
python src/C/q3/06_validate.py
python src/C/q3/07_ablation.py
```

The scripts write tables, summaries, and figures to `src/C/q3/output/`. Run `python -m pytest src/C/q3/tests -q` for the Q3 unit tests.

## Signal and event handling

- The raw data contain 400 VisCue trials across four files at 256 Hz. Q1-clean output is used for event timestamp mapping and a same-trial quality comparison; it is not the source of the main EEG features.
- `02b_preprocess_raw.py` reads only raw F3/Fz/F4 EEG channels and filters each complete record before epoching. The ERP branch is zero-phase 0.5-30 Hz; the time-frequency branch is zero-phase 1-80 Hz. ICA is not claimed because there are only three EEG channels and no separate EOG reference.
- Cue epochs cover [-0.10, 0.50) s relative to VisCue. Target epochs cover [-0.10, 0.80) s relative to cue+2.2 s. The target time has no independent event marker, so target-offset sensitivity is checked from 2.0 to 2.4 s.
- Channel 9 preserves its original event values and maps negative codes to -2, zero to 0, positive to +2. Only zero-to-nonzero edges count as events. It supplies side/time metadata and the response-endpoint window; it is excluded from all EEG features.
- The pre-response signal window is [cue, channel-9 event time - 0.10 s], with cumulative and terminal-500-ms summaries. The marker semantics and the cue+2.2 s target-time anchor must be verified before interpreting these as behavioral RT windows.
- The raw QC flags windows with non-finite samples, a flat channel, or any absolute value at/above 999.5 raw data units. The source unit is not independently documented. Each stage keeps its QC flag and exclusion reason; no epoch is silently deleted.
- PLV and theta-gamma PAC remain uncomputed because the event windows are short and no validated surrogate test is available. Gamma power is exploratory.

## Model and validation

- `V/H/P` are transparent observed feature proxies: frontal ERP candidate amplitude, frontal log theta power, and frontal log beta power. A free latent loading matrix is not fitted because three electrodes do not establish an identifiable source model. Path coefficients are descriptive OLS estimates, not source activity.
- `Task-1`/`Task-2` is stored only as a filename-code candidate. Its formal mapping to project types and participant identity are not verified.
- EEG cue-side validation uses leave-one-record-out folds and training-fold scaling. Results include the full raw-QC sample and the 297-trial Q1-quality-matched sample. The target-offset table is a robustness check, not a procedure for selecting the most favorable time.
- V/H/P ablations compare cue prediction and leave-one-feature-out reconstruction. AIC/BIC are auxiliary fit summaries, not significance tests.
- Correctness and omission status are not inferred from channel 9 alone. All 400 files contain a nonzero channel-9 marker, but this does not prove that the protocol had no behavioral omissions.
- DDM is skipped while the assumed target anchor yields a median interval near 15 ms and no verified response deadline is available. The choice-only Logistic result is exploratory and does not represent accuracy or diagnosis.

## Main output files

- `output/raw_signal_processing.json`, `output/raw_feature_qc_summary.json`: filtering and QC settings/counts.
- `output/trial_features.csv`, `output/feature_definitions.csv`: one row per trial and event/response window.
- `output/state_scores.csv`, `output/cognitive_path_coefficients.csv`, `output/state_condition_effects.csv`: anchored states and descriptive task/cue summaries.
- `output/validation_summary.json`, `output/validation_interpretation.md`: grouped prediction, matched-sample comparison, timing sensitivity, and interpretation limits.
- `output/quality_agreement.csv`, `output/behavior_cue_choice_alignment.csv`, `output/behavior_model_comparison.csv`: Q1/raw-QC agreement and response-choice model diagnostics.
- `output/ablation_summary.json`, `output/ablation_report.md`, `output/ablation_effects.png`: V/H/P ablation results.
- `output/legacy_q1_baseline_reference.json`: archived pre-update baseline used for direct comparison.
- `output/legacy_q1_pipeline/`: preserved outputs from the superseded Q1-derived feature pipeline; not used by current Q3 scripts.

## Exploratory classifier search

The initial metrics and code are frozen under `baselines/20260925_initial/`. A separate 49-configuration nested leave-one-record-out search is in `experiments/model_search.py`; results are under `output/experiments/20260925_model_search_v1/`. On raw-QC cue trials, inner-fold model selection raised BA by 6.05 percentage points (0.460 to 0.521) across the four outer records. Target-window BA fell by 2.15 points, and the cue lift was absent when training and evaluation were restricted to the 297 Q1-matched trials. These exploratory results do not replace the primary baseline.

To run another isolated experiment, choose a new output directory so prior results remain intact:

```powershell
python src/C/q3/experiments/model_search.py --output-dir src/C/q3/output/experiments/my_model_search
python src/C/q3/experiments/summarize_quality_subgroups.py --results-dir src/C/q3/output/experiments/my_model_search
```

## Further optimization rounds

The later time-resolved ERP/CSP, joint-candidate, and training-fold threshold searches are documented in [optimization_report.md](optimization_report.md). The best nested result remains the first-round raw-QC cue search (+6.05 pp BA versus the frozen baseline); this is development-only and does not replace the primary pipeline.

Reproduce the isolated searches from the repository root:

```powershell
python src/C/q3/experiments/temporal_search.py --output-dir src/C/q3/output/experiments/temporal_reproduction
python src/C/q3/experiments/joint_search.py --output-dir src/C/q3/output/experiments/joint_reproduction
python src/C/q3/experiments/threshold_search.py --output-dir src/C/q3/output/experiments/threshold_reproduction
python src/C/q3/experiments/expanded_search.py --output-dir src/C/q3/output/experiments/expanded_reproduction
```
