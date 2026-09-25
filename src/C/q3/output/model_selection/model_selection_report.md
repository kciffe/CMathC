# Q3 model selection and design review

## Selected route

Use the proposed three-layer model: observed EEG features, an anchored sparse measurement model for trial-level `V/H/P`, and a structural path model `V -> H -> P` with direct `V -> P` input. Q2 contributes the source-field and pathway rationale; Q3 fits against the measured EEG. Channel 9 contributes response outcomes/timing only.

The primary latent route is conditional on rank, anchor and simulation-recovery checks. A predefined standardized V/H/P proxy regression is the required baseline and fallback. Three-channel scalp EEG supports functional macro-state descriptions, not hippocampal or LGN source localization.

## Data and current evidence

Four files contain 400 cue trials and 400 response markers; 297 trials map to Q1 quality-retained EEG. Raw sample rate is 256 Hz. Existing features use Q1 clean 0.2-24 Hz data, so they cannot support the proposed 30-80 Hz gamma branch. Current leave-one-file-out balanced accuracy is 0.460 for cue-locked EEG, 0.508 for target-locked EEG, and 0.506 for EEG-state choice prediction.

Target time is not separately marked. With the provisional `cue+2.2 s` anchor, median RT is 0.015 s and 399/400 are below 0.1 s. There are zero missing response markers. This evidence blocks reliable DDM/omission claims until event timing, deadline and label semantics are verified.

## Implementation decisions

- Process raw Fz/F3/F4 in separate continuous ERP (0.5-30 Hz) and time-frequency (1-80 Hz) branches; keep Q1 clean output for matching/quality metadata only.
- Do not make three-channel ICA the default artifact-removal step. Use auditable artifact flags and sensitivity checks.
- Use the PDF's loading mask only after defining every feature row and fixing factor sign/scale; start with diagonal/shrinkage residual covariance.
- Add task-by-path interactions only if task identity is verified and effective group/trial counts support them.
- Use leave-one-file-out validation, with all transforms fitted on training files. Report per-file metrics and baselines. Do not use a random 7:3 trial split.
- Treat gamma, PLV and theta-gamma PAC as exploratory until window-length and surrogate checks support them. Treat AIC/BIC as model-comparison measures, not significance tests.
- Keep DDM optional until target onset, deadline, correctness/choice mapping, effective sample support and parameter recovery are established.

## Falsification and fallback

Reject a V/H/P latent interpretation if factor scales/signs are unstable, design matrices are rank-deficient, simulated parameters are not recovered, or the full model fails to improve held-out reconstruction over the proxy baseline. In that case report observed EEG features and predefined proxy associations without naming latent coefficients as brain-region activity.