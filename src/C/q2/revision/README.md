# Q2 revision_v1 implementation

This directory contains an independent implementation of the approved
[Q2 project plan](../documents/问题二_项目实施方案.md). It does not replace or
write into the original 01–05 scripts or their outputs.

## Modules

- config.py: fixed paths, timings, parameters, stimulus matrix names, and the
  rank-2 observation matrix.
- real_data.py: per-MAT clean EEG loading, F3/Fz/F4 ordering, baseline
  correction, Stage1 cue grouping, and conservative Stage2 labels.
- frontend.py: signed contrast, center-surround LGN ON/OFF dynamics, four
  Gabor channels, mirrored triangle configuration codes, and 64/128 scale audits.
- model.py: six E/I Wilson–Cowan populations, relative synaptic source proxy,
  and fixed rank-2 observation mapping.
- fit.py: bounded Stage1-only leave-one-MAT fitting of tau_s, g_i, and tau_a,
  with one nonnegative amplitude per training fold.
- evaluate.py: leave-one-MAT real-trial shrinkage LDA, ERP measures, and the
  frozen-parameter no-position/no-offset controls.
- run_revision.py: one entry point for audit, forward simulation, and
  leave-one-MAT evaluation.

## Run modes

From the repository root, run one of:

    ./.venv/Scripts/python.exe src/C/q2/revision/run_revision.py --mode audit
    ./.venv/Scripts/python.exe src/C/q2/revision/run_revision.py --mode simulate
    ./.venv/Scripts/python.exe src/C/q2/revision/run_revision.py --mode validate

Validate is the full four-fold workflow. The implementation has not yet been
run end to end; the draft test_*.py files are historical probes and are not
evidence that this implementation passes. Run the audit and simulation first,
inspect failures and runtime, then execute full validation.

Outputs are written only to src/C/q2/output/revision_v1. The full run is
designed to produce six data artifacts and three composite figures. Task2
Stage2 remains target_unknown; inward/outward are model counterfactuals only.
EEG observation units are relative, not calibrated microvolts.
