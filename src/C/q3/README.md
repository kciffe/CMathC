# Q3 EEG cognitive model

This pipeline follows the latest Q3 solution with a separate raw-EEG analysis path. It uses F3/Fz/F4 for EEG features, VisCue for cue-side labels, and channel 9/TimeStamp for response-event metadata and response-window endpoints. Per the reference model, the unique channel-9 zero-to-nonzero edge is the absolute response time `t_act`; it is not included in EEG features, dynamic-state predictors, or sensor-model inputs. The original feature/classification scripts remain independent of Q2; the new dynamic cognitive model explicitly calls Q2's visual forward simulation and leadfield projection for candidate cue/target scenes, without inverting the five source proxies.

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
python src/C/q3/09_event_time_semantics.py
python src/C/q3/dynamic_validation.py
```

After `02_extract_trials.py`, optionally run `python src/C/q3/08_behavior_event_audit.py` to cross-check channel-derived outcomes and audit target onset, deadlines, and record mapping. Channel 8/9 correctness and cue-interval omission labels do not require external files. Run `09_event_time_semantics.py` to regenerate the raw event-time, file-grouping, and Q2 leadfield identifiability audit in `output/continuation_audit/`; its report is [`event_timing_semantics_report.md`](output/continuation_audit/event_timing_semantics_report.md).

Run `python src/C/q3/dynamic_validation.py` after the event audit to generate candidate-scene Q2 forward trajectories, V/H/P macro states, and leave-one-record-out EEG comparisons. The target onset, stimulus type, target duration, match-evidence branch, and preprocessing modes are command-line configurable; defaults include a no-target scenario plus 2.0/2.2/2.4 s target candidates and `none`/`causal`/`zero_phase` processing. Candidate conditions are sensitivity scenarios, not verified trial truth. The script writes the detailed formulas, step-by-step methods, fold metrics, and six Chinese-labeled PNG intermediate figures to `问题三实现流程与模型说明.md` and `output/dynamic_heldout_validation/`. Q2 forward trajectories are kept in memory for the current run; no persistent `.npz` cache is retained.

The scripts write tables, summaries, and figures to `src/C/q3/output/`. Run `python -m pytest src/C/q3/tests src/C/q3/experiments -q` for the Q3 unit and experiment tests.

## Signal and event handling

- The raw data contain 400 VisCue trials across four files at 256 Hz. Q1-clean output is used for event timestamp mapping and a same-trial quality comparison; it is not the source of the main EEG features.
- `02b_preprocess_raw.py` reads only raw F3/Fz/F4 EEG channels and filters each complete record before epoching. The ERP branch is zero-phase 0.5-30 Hz; the time-frequency branch is zero-phase 1-80 Hz. ICA is not claimed because there are only three EEG channels and no separate EOG reference.
- Cue epochs cover [-0.10, 0.50) s relative to VisCue. Target epochs cover [-0.10, 0.80) s relative to cue+2.2 s. The target time has no independent event marker, so target-offset sensitivity is checked from 2.0 to 2.4 s.
- Channel 9 response magnitude depends on its MAT `DataLabel`: `Action:L-1/R+1` uses ±1, while `TgtAct:L-2/R+2` uses ±2. In the Task-2 records, one nonzero bout starts with signed ±1 and later contains the declared same-sign ±2 code; these levels are one action bout, not separate responses. Q3 keeps `t_act` at the first zero-to-nonzero edge, decodes choice from the code declared in `DataLabel` within that bout, and leaves direction unresolved if the declared code is absent or signs conflict. Raw codes are preserved.
- The pre-response signal window is [cue, `t_act` - 0.10 s], with cumulative and terminal-500-ms summaries. The dynamic validation reports these windows at record×cue-side level using the group's median `t_act`. Channel 8 gives the target side (-1 left, +1 right), and channel 9's declared action code gives the chosen side; `correct` is their match. `is_omission=1` means no channel-9 action bout between this cue onset and the next cue onset. Late-response status is not modeled separately. Target onset is not independently marked, so `reaction_time_s` remains unavailable; `t_act_s` is the observed absolute onset edge.
- The raw QC flags windows with non-finite samples, a flat channel, or any absolute value at/above 999.5 raw data units. The source unit is not independently documented. Each stage keeps its QC flag and exclusion reason; no epoch is silently deleted.
- PLV and theta-gamma PAC remain uncomputed because the event windows are short and no validated surrogate test is available. Gamma power is exploratory.

## Model and validation

- `V/H/P` are transparent observed feature proxies: frontal ERP candidate amplitude, frontal log theta power, and frontal log beta power. A free latent loading matrix is not fitted because three electrodes do not establish an identifiable source model. Path coefficients are descriptive OLS estimates, not source activity.
- `Task-1`/`Task-2` is stored only as a filename-code candidate. Its formal mapping to project types and participant identity are not verified.
- EEG cue-side validation uses leave-one-record-out folds and training-fold scaling. Results include the full raw-QC sample and the 297-trial Q1-quality-matched sample. The target-offset table is a robustness check, not a procedure for selecting the most favorable time.
- V/H/P ablations compare cue prediction and leave-one-feature-out reconstruction. AIC/BIC are auxiliary fit summaries, not significance tests.
- Correctness labels are derived per trial by comparing channel-9 action side with channel-8 target side. A no-response interval has `correct=null` and `is_omission=1`; it is not counted as an incorrect choice. The supplied 400 trials each contain one action edge, so there are no observed omission intervals in this dataset.
- The correctness model predicts that channel-derived outcome from cue side and/or cue-locked V/H/P features, with leave-one-record-out validation. Channel 8 and channel 9 outcomes are labels only, never EEG predictors. No omission classifier is fitted because the data have no positive omission examples.
- DDM is skipped because per-trial target onset is unavailable. The cue+2.2 s schedule proxy happens to place the marker about 15 ms later on median, but this is not treated as RT duration. The choice-direction Logistic and correctness Logistic answer different questions; neither is a diagnosis model.

## Main output files

- `output/raw_signal_processing.json`, `output/raw_feature_qc_summary.json`: filtering and QC settings/counts.
- `output/trial_features.csv`, `output/feature_definitions.csv`: one row per trial and event/response window.
- `output/state_scores.csv`, `output/cognitive_path_coefficients.csv`, `output/state_condition_effects.csv`: anchored states and descriptive task/cue summaries.
- `output/validation_summary.json`, `output/validation_interpretation.md`: grouped prediction, matched-sample comparison, timing sensitivity, and interpretation limits.
- `output/dynamic_heldout_validation/dynamic_cv_metrics.csv`, `output/dynamic_heldout_validation/t_act_endpoint_cv_metrics.csv`: fixed-window and channel-9 `t_act−100 ms` leave-one-record-out EEG scores.
- `output/quality_agreement.csv`, `output/behavior_cue_choice_alignment.csv`, `output/behavior_model_comparison.csv`: Q1/raw-QC agreement and response-choice model diagnostics.
- `output/behavior_correctness_predictions.csv`, `output/behavior_correctness_model_comparison.csv`, `output/behavior_outcomes_by_record.csv`: held-out correctness predictions and channel-derived outcomes.
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
