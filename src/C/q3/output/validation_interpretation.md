# Q3 validation result and possible explanations

## Observed result

- VisCue events: 400; Q1 quality-retained, timestamp-matched EEG trials: 297.
- Retained cue labels: left 152, right 145; missing values among the 13 validation features: 0.
- Leave-one-record-out cue-side prediction: cue-locked balanced accuracy 0.460, accuracy 0.449; target-locked balanced accuracy 0.508, accuracy 0.506.
- Across-record feature-effect direction agrees in all four records for 0/13 cue-locked features and 2/13 target-locked features.
- Channel 9 response markers: 400 choices, 0 trials without a marker; median RT from the scheduled target anchor 0.015 s, with 399 RTs under 100 ms.
- Channel 9 choice Logistic, leave-one-record-out balanced accuracy: 0.506.

The current EEG cue-side decoding effect is weak: both primary balanced accuracies are near 0.5. The channel-9 choice model is reported separately and does not establish correctness or clinical diagnosis.

## Plausible causes (hypotheses)

1. **Recording-to-recording variation.** The cue-side feature differences do not keep a common direction across the four records. The repeated within-record diagnostic is higher for a few record/stage pairs and near or below chance for others, which is consistent with session-specific effects that do not transfer reliably.
2. **Limited usable sample size.** Q1 retained 297 of 400 raw cue trials (54–83 trials per recording). With 13 EEG features and only four held-out recordings, the cross-record estimate has substantial sampling uncertainty.
3. **Restricted scalp coverage and bandwidth.** The analysis uses only F3/Fz/F4 and Q1's 0.2–24 Hz clean signal. It cannot capture posterior scalp patterns often used for P300 analysis or activity above 24 Hz.
4. **Target-stage timing is assumed.** Target-locked features use cue time + 2.2 s from the experimental schedule; this offset was not verified from an allowed event channel. Timing variation would blur target-locked responses.
5. **Conditions remain unresolved.** Task type and participant grouping could not be verified from the allowed signals, so potentially different conditions are pooled and each recording file is only a proxy for a held-out group.
6. **The RT anchor may not match the recorded action clock.** 399 of 400 RTs are under 100 ms when target time is set to cue + 2.2 s. This could reflect a schedule-to-marker mismatch or channel timing semantics; until verified, it does not support a DDM fit.

These are plausible explanations, not established causes. Cue-to-Q1 timestamp matches were one-to-one, sampling intervals matched the stated 256 Hz rate, all planned cue/target windows had full coverage, and those validation features had no missing values; this makes obvious cue mapping or feature-window truncation errors less likely.

## Interpretation boundary

Channel 9 supplies response direction and timing only; it is never an EEG feature. Correctness remains unassigned because target-side truth is not verified for all tasks. The target offset remains a protocol assumption. RTs near 15 ms from that anchor are flagged, so the DDM is skipped and the choice-only logistic model is used instead.
