# Q3 raw EEG validation results and possible explanations

## Observed result

- Raw VisCue trials: 400; cue epochs passing raw artifact QC: 387; Q1-clean matched sensitivity sample: 297 (left 152, right 145).
- Q1/raw-QC agreement: cue_locked: Q1/raw-QC both pass 297, Q1 fail/raw-QC pass 90, both fail 13; target_locked: Q1/raw-QC both pass 297, Q1 fail/raw-QC pass 99, both fail 4.
- Missing primary ERP amplitude values after QC: 0.
- Leave-one-record-out cue-side prediction: cue-locked balanced accuracy 0.460, accuracy 0.467, AUC 0.483, macro-F1 0.449; target-locked balanced accuracy 0.522, accuracy 0.528, AUC 0.526, macro-F1 0.509.
- On the Q1-matched 297-trial sample, the new raw-filter balanced accuracy was 0.487 for cue and 0.510 for target; changes from the old Q1-clean baseline were +0.027 and +0.003.
- Target-offset sensitivity (2.0-2.4 s) balanced accuracy ranged from 0.483 to 0.541; this is a timing robustness range, not an offset-selection procedure.
- Across-record feature-effect direction agrees in all four records for 2/13 cue-locked features and 3/13 target-locked features.
- Channel 9 markers: 400; median t_act relative to cue is 2.215 s. The cue+2.2 s schedule proxy is under 100 ms before t_act for 399 trials; this proxy is not RT duration.
- Channel 9 choice Logistic, leave-one-record-out balanced accuracy: 0.747.
- Choice baseline comparison: cue/task-code BA 0.755; cue/task-code+EEG BA 0.747; EEG increment -0.007.
- Channel-8/9 correctness by filename task-code candidate: code 1: 196/198 same-side (99.0%); code 2: 87/189 same-side (46.0%). Correctness uses the supplied channel semantics; filename task grouping is descriptive only.
- Trial outcomes: correct 293, incorrect 107, omission 0; correctness-model LO-record-out BA: cue_side_only BA 0.500; EEG_state_only BA 0.497; cue_side_plus_EEG_state BA 0.500.

The main results use raw continuous EEG filtered in separate 0.5-30 Hz ERP and 1-80 Hz time-frequency branches. The Q1-clean epochs are used only for mapping and a matched-sample comparison. Leave-one-record-out is the main validation; it is not leave-one-participant-out because file-to-participant identity is unverified.

## Plausible explanations if performance is weak (hypotheses)

1. **Recording shift.** Four held-out records provide only four independent validation groups; electrode offsets, session state, or participant differences can overwhelm cue-locked effects.
2. **Short windows and single-trial noise.** Cue (0.5 s) and target (0.8 s) windows contain few cycles at theta/alpha frequencies, while single-trial frontal ERP peaks are noisy.
3. **Frontal-only coverage.** F3/Fz/F4 cannot measure posterior visual topography or support reliable localization of LGN, hippocampal, or PFC generators.
4. **Target timing assumption.** The target event is not independently marked; the offset sensitivity table should be read as a robustness check, not as a search for the best event time.
5. **Artifact exclusions.** The raw QC removes event windows with hard clipping, non-finite samples, or flatline; remaining unflagged movement or muscle artifacts may still reduce signal quality.
6. **Unverified file/task mapping.** Task-1/Task-2 is retained as a filename code candidate only; project semantics and participant identity are not assumed.
7. **No added choice information from EEG in this split.** The cue/task-code baseline and cue/task-code-plus-EEG balanced accuracies should be compared directly. If they are similar, this may mean the three-channel proxies add little across recordings, or that the provisional channel-9 label is strongly tied to cue/task coding; it does not prove a cognitive mechanism.

These are hypotheses, not established causes. PLV and theta-gamma PAC are marked unavailable because the short windows do not support reliable estimates without surrogate validation. ICA is not claimed: the available signal has three EEG electrodes and no separate EOG reference.

## Interpretation boundary

Channel 9 supplies response direction and t_act and is never an EEG feature or V/H/P input. Correctness compares channel-9 action side with the channel-8 target side. An omission is no channel-9 action edge between a cue onset and the next cue onset; late-response timing is not modeled separately. Target onset is not independently marked, so reaction-time duration and DDM remain unavailable. V/H/P are anchored functional proxies, not localized brain sources.
