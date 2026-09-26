# Q3 EEG cognitive model

This pipeline follows the latest Q3 solution with a separate raw-EEG analysis path. It uses F3/Fz/F4 for EEG features, VisCue for cue-side labels, and channel 9/TimeStamp for response-event metadata and response-window endpoints. Per the reference model, the unique channel-9 zero-to-nonzero edge is the absolute response time `t_act`; it is not included in EEG features, dynamic-state predictors, or sensor-model inputs. The original feature/classification scripts remain independent of Q2; the new dynamic cognitive model explicitly calls Q2's visual forward simulation and leadfield projection for candidate cue/target scenes, without inverting the five source proxies.

## Run order

From the repository root:

```powershell
python src/C/q3/01_audit_events.py
python src/C/q3/02_extract_trials.py
python src/C/q3/08_behavior_event_audit.py
python src/C/q3/02b_preprocess_raw.py
python src/C/q3/03_extract_features.py
python src/C/q3/04_cognitive_state.py
python src/C/q3/05_behavior_model.py
python src/C/q3/06_validate.py
python src/C/q3/07_ablation.py
python src/C/q3/09_event_time_semantics.py
python src/C/q3/dynamic_validation.py
```

After `02_extract_trials.py`, run `python src/C/q3/08_behavior_event_audit.py` to write channel-derived direction-consistency and action-marker counts, and audit optional task-truth, target-onset, deadline, and record-mapping metadata. The channel-8/channel-9 direction comparison is not task correctness because the Task-2 cue-to-target mapping has not been verified. The cue-relative [-1,+5] s interval is an analysis window for counting declared channel-9 codes; it does not establish timely or late responses. Run `09_event_time_semantics.py` to regenerate the raw event-time, candidate file grouping, Q2 leadfield identifiability, and action-marker window audit in `output/continuation_audit/`; its report is [`event_timing_semantics_report.md`](output/continuation_audit/event_timing_semantics_report.md).

Run `python src/C/q3/dynamic_validation.py` after the event audit to generate candidate-scene Q2 forward trajectories, V/H/P macro states, and leave-one-record-out EEG comparisons. The target onset, stimulus type, target duration, match-evidence branch, and preprocessing modes are command-line configurable; defaults include a no-target scenario plus 2.0/2.2/2.4 s target candidates and `none`/`causal`/`zero_phase` processing. Candidate conditions are sensitivity scenarios, not verified trial truth. The script writes its run-specific method/results report to `output/dynamic_heldout_validation/dynamic_validation_report.md` and writes six Chinese-labeled PNG intermediate figures and fold metrics to `output/dynamic_heldout_validation/`. The architecture and formula guide `问题三实现流程与模型说明.md` is maintained separately so validation runs do not overwrite it. Q2 forward trajectories are kept in memory for the current run; no persistent `.npz` cache is retained.

The scripts write tables, summaries, and figures to `src/C/q3/output/`. Run `python -m pytest src/C/q3/tests src/C/q3/experiments -q` for the Q3 unit and experiment tests.

## Signal and event handling

- The raw data contain 400 VisCue trials across four files at 256 Hz. Q1-clean output is used for event timestamp mapping and a same-trial quality comparison; it is not the source of the main EEG features.
- `02b_preprocess_raw.py` reads only raw F3/Fz/F4 EEG channels and filters each complete record before epoching. The ERP branch is zero-phase 0.5-30 Hz; the time-frequency branch is zero-phase 1-80 Hz. ICA is not claimed because there are only three EEG channels and no separate EOG reference.
- Cue epochs cover [-0.10, 0.50) s relative to VisCue. Target epochs cover [-0.10, 0.80) s relative to cue+2.2 s. The target time has no independent event marker, so target-offset sensitivity is checked from 2.0 to 2.4 s.
- Channel 9 response magnitude depends on its MAT `DataLabel`: `Action:L-1/R+1` uses ±1, while `TgtAct:L-2/R+2` uses ±2. In the Task-2 records, one nonzero bout starts with signed ±1 and later contains the declared same-sign ±2 code; these levels are one action bout, not separate responses. Q3 keeps `t_act` at the first zero-to-nonzero edge, decodes response direction from the code declared in `DataLabel` within that bout, and leaves direction unresolved if the declared code is absent or signs conflict. Raw codes are preserved.
- The pre-response signal window is [cue, `t_act` - 0.10 s], with cumulative and terminal-500-ms summaries. The dynamic validation reports these windows at record×cue-side level using the group's median `t_act`. Channel 8 gives the cue direction (-1/+1), and channel 9's declared action code gives response direction. Their same-side/different-side relation is a direction-consistency observation, not task correctness while the Task-2 cue-to-target mapping remains unverified. A channel-9 edge count in the cue-to-next-cue interval is an event count; it is not an omission rate without an experiment-defined response interval.
- The cue-relative [-1,+5] s interval is a selected analysis window. A `DataLabel`-declared channel-9 code in that interval is counted as an observed code; it is not evidence that the response was timely. Actual lateness cannot be determined without per-trial target onset and the formal response deadline. `response_duration_s` is the duration of the first contiguous channel-9 nonzero bout, measured as sample count / sample rate. This duration is not target-to-response RT; target onset is not independently marked, so `reaction_time_s` remains unavailable. `t_act_s` is the observed absolute onset edge.
- The raw QC flags windows with non-finite samples, a flat channel, or any absolute value at/above 999.5 raw data units. The source unit is not independently documented. Each stage keeps its QC flag and exclusion reason; no epoch is silently deleted.
- PLV and theta-gamma PAC remain uncomputed because the event windows are short and no validated surrogate test is available. Gamma power is exploratory.

## Model and validation

- `V/H/P` are transparent observed feature proxies: frontal ERP candidate amplitude, frontal log theta power, and frontal log beta power. A free latent loading matrix is not fitted because three electrodes do not establish an identifiable source model. Path coefficients are descriptive OLS estimates, not source activity.
- `Task-1`/`Task-2` is stored only as a filename-code candidate. Its formal mapping to project types and participant identity are not verified.
- EEG cue-side validation uses leave-one-record-out folds and training-fold scaling. Results include the full raw-QC sample and the 297-trial Q1-quality-matched sample. The target-offset table is a robustness check, not a procedure for selecting the most favorable time.
- V/H/P ablations compare cue prediction and leave-one-feature-out reconstruction. AIC/BIC are auxiliary fit summaries, not significance tests.
- The dataset contains 293 trials with matching cue/response directions and 107 with differing directions. These are direction-consistency counts, not correct/incorrect counts or task accuracy: the Task-2 cue-to-target mapping has not been verified. Each of the 400 cue-to-next-cue intervals contains a channel-9 edge; report this as an event count, not as a verified zero omission rate.
- The second behavioral classifier predicts direction consistency from cue side and/or cue-locked V/H/P features, with leave-one-record-out validation. Channel 8 and channel 9 outcomes are labels only, never EEG predictors. This is a consistency-prediction result, not a task-correctness model or diagnosis.
- All 400 trials contain at least one declared response code in the selected [-1,+5] s window. This is a count of trials with a marker in that window, not a count of “timely” trials. Actual lateness cannot be determined without per-trial target onset and the formal response deadline. DDM is skipped because per-trial target onset is unavailable. The cue+2.2 s schedule proxy happens to place the marker about 15 ms later on median, but this is not treated as RT duration. The choice-direction Logistic and consistency Logistic answer different questions; neither is a diagnosis model.

## Main output files

- `output/raw_signal_processing.json`, `output/raw_feature_qc_summary.json`: filtering and QC settings/counts.
- `output/trial_features.csv`, `output/feature_definitions.csv`: one row per trial and event/response window.
- `output/state_scores.csv`, `output/cognitive_path_coefficients.csv`, `output/state_condition_effects.csv`: anchored states and descriptive task/cue summaries.
- `output/validation_summary.json`, `output/validation_interpretation.md`: grouped prediction, matched-sample comparison, timing sensitivity, and interpretation limits.
- `output/dynamic_heldout_validation/dynamic_cv_metrics.csv`, `output/dynamic_heldout_validation/t_act_endpoint_cv_metrics.csv`: fixed-window and channel-9 `t_act−100 ms` leave-one-record-out EEG scores.
- `output/quality_agreement.csv`, `output/behavior_cue_choice_alignment.csv`, `output/behavior_model_comparison.csv`: Q1/raw-QC agreement and response-choice model diagnostics.
- `output/behavior_consistency_predictions.csv`, `output/behavior_consistency_model_comparison.csv`, `output/behavior_outcomes_by_record.csv`: held-out direction-consistency predictions and channel-derived direction/event counts. Older `behavior_correctness_*` files are legacy artifacts and should not be read as task correctness results.
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
