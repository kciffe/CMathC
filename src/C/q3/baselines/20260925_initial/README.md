# Q3 initial baseline snapshot (2026-09-25)

This directory freezes the Q3 pipeline before model-improvement experiments. `code/` contains the Python source and unit tests used for the current baseline. `output/` is a full copy of the generated Q3 outputs at snapshot time. The baseline remains unchanged by subsequent experiments.

Primary benchmark (leave-one-record-out, raw-QC sample):
- Cue-side prediction: balanced accuracy 0.4601, accuracy 0.4673, AUC 0.4834 (387 QC-passed epochs).
- Nominal target-side prediction: balanced accuracy 0.5225, accuracy 0.5281, AUC 0.5257 (396 QC-passed epochs).
- Behavior choice: cue + filename-task-code candidate BA 0.7547; cue/task + EEG BA 0.7473. This is exploratory because the task-code mapping and response semantics are unverified.

All preprocessing and evaluation details are preserved in `README.md`, `TODO.md`, and `output/validation_summary.json`. SHA-256 hashes for snapshot contents are recorded in `manifest.json`.
