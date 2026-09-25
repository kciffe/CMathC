# Q3 EEG analysis

This pipeline uses F3/Fz/F4 as EEG features, VisCue and TimeStamp for cue alignment, and the ninth Action/TgtAct channel for response direction and timing. Channel 9 never enters an EEG feature array or predictor. Q3 does not import the Q2 simulation chain.

## Run order

From the repository root:

```powershell
python src/C/q3/01_audit_events.py
python src/C/q3/02_extract_trials.py
python src/C/q3/03_extract_features.py
python src/C/q3/04_cognitive_state.py
python src/C/q3/05_behavior_model.py
python src/C/q3/06_validate.py
```

The scripts write intermediate tables, summaries, and figures to `src/C/q3/output/`.

## Data and interpretation

- Raw VisCue events are detected from the allowed cue signal. Q1 retained trials are matched to them by the event timestamp and cue direction, not by the post-screening row number.
- Channel 9 preserves each response onset's `response_raw` value and adds `response_code`: any negative raw code becomes -2, zero remains 0, and any positive raw code becomes +2. The zero-to-nonzero edge marks an event; a sustained marker is not counted repeatedly.
- `choice_side` is stored as -1/+1 for modeling. A missing marker is recorded as a no-response trial. Correctness stays blank until the task's target-side truth is verified.
- EEG features are extracted from Q1's quality-retained `clean.mat` epochs, which were filtered at 0.2–24 Hz and downsampled to 128 Hz. Only the original `F3`, `Fz`, and `F4` signals are selected. Gamma power and theta-gamma coupling are outside the retained bandwidth.
- The target stage is locked to cue time + 2.2 seconds, based on the 0.2 s cue and approximately 2 s wait in the task description. No separate target-display marker is present, so this remains a schedule assumption.
- The pre-response window ends about 100 ms before the channel 9 onset. It is extracted from a single uniformly filtered pass over each full continuous EEG record; it is never stitched to a Q1 epoch. These response-locked features are exploratory and excluded from choice prediction because their endpoint depends on response time.
- `V/H/P` are predefined feature composites: frontal ERP candidate amplitude, frontal log theta power, and frontal log beta power. These are functional proxies, not identified brain-region sources or a dynamic state-transition model.
- Task type and participant identity are not inferred from recording filenames. Validation holds out each recording file in turn; it cannot establish leave-participant-out generalization.
- The main cue-side validation is leave-one-record-out. An additional repeated within-record split is written as a diagnostic only; it can be optimistic because train and test trials come from the same recording.
- ERP outputs are frontal ERP/P300 candidates. Three frontal electrodes cannot establish the standard parietal P300 topography.
- DDM is gated on RT quality and deadline information. The current RTs are mostly close to the scheduled target anchor, so they are flagged and choice-only logistic regression is used as the fallback.

## Optional independent behavior labels

Optional independently verified correctness or deadline information can be placed in `src/C/q3/input/behavior_labels.csv`, keyed by `record` and `original_trial_index`:

```text
record,original_trial_index,correct,deadline_s
VisualCogA_Task-1,0,,1.5
```

The current choice label comes from channel 9, so an external choice file is not needed. External columns are preserved with an `external_` prefix and do not overwrite channel 9 values.
