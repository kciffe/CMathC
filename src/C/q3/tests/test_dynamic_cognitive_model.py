from __future__ import annotations

import importlib
import importlib.util

import numpy as np


def _model_module():
    spec = importlib.util.find_spec("dynamic_cognitive_model")
    assert spec is not None, "the configurable dynamic cognitive model has not been implemented"
    return importlib.import_module("dynamic_cognitive_model")


def test_cue_trace_is_retained_and_control_waits_for_target_event():
    model = _model_module()
    dt_s = 0.01
    time_s = np.arange(300) * dt_s
    visual = np.ones_like(time_s)
    cue_gate = ((time_s >= 0.0) & (time_s < 0.2)).astype(float)
    target_gate = ((time_s >= 2.0) & (time_s < 2.2)).astype(float)

    states = model.integrate_macro_states(
        visual_drive=visual,
        cue_gate=cue_gate,
        target_gate=target_gate,
        match_evidence=1.0,
        dt_s=dt_s,
    )

    assert states["memory"][40] > 0.0
    assert np.all(states["control"][time_s < 2.0] == 0.0)
    assert states["control"][205] > 0.0


def test_mismatch_scenario_drives_more_control_than_match_scenario():
    model = _model_module()
    dt_s = 0.01
    time_s = np.arange(300) * dt_s
    visual = np.ones_like(time_s)
    cue_gate = ((time_s >= 0.0) & (time_s < 0.2)).astype(float)
    target_gate = ((time_s >= 2.0) & (time_s < 2.2)).astype(float)

    match = model.integrate_macro_states(
        visual, cue_gate, target_gate, match_evidence=1.0, dt_s=dt_s
    )
    mismatch = model.integrate_macro_states(
        visual, cue_gate, target_gate, match_evidence=-1.0, dt_s=dt_s
    )

    assert mismatch["control"].max() > match["control"].max()


def test_sensor_observation_uses_q2_forward_signal_and_macro_loadings():
    model = _model_module()
    q2_sensor = np.arange(15, dtype=float).reshape(3, 5) / 10.0
    memory = np.array([0.0, 0.2, 0.4, 0.2, 0.0])
    control = np.array([0.0, 0.0, 0.1, 0.2, 0.1])
    memory_loading = np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)
    control_loading = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    feedback_gain = 0.15

    observed = model.observe_macro_model(
        q2_sensor,
        memory,
        control,
        memory_loading,
        control_loading,
        topdown_control_gain=feedback_gain,
    )
    expected = (
        q2_sensor * (1.0 + feedback_gain * control)[None, :]
        + memory_loading[:, None] * memory[None, :]
        + control_loading[:, None] * control[None, :]
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_target_time_and_preprocessing_are_configurable_and_finite():
    model = _model_module()
    time_s = np.arange(-0.2, 2.801, 0.004)
    cue_gate, target_gate = model.make_event_gates(
        time_s, cue_onset_s=0.0, cue_duration_s=0.2,
        target_onset_s=2.4, target_duration_s=0.2,
    )
    assert np.isclose(time_s[np.flatnonzero(target_gate)[0]], 2.4, atol=1e-12)
    assert np.isclose(time_s[np.flatnonzero(cue_gate)[0]], 0.0, atol=1e-12)

    rng = np.random.default_rng(20260925)
    signal = rng.normal(size=(3, time_s.size))
    for mode in ("none", "causal", "zero_phase"):
        processed, processed_time = model.apply_preprocessing(
            signal,
            time_s,
            source_fs_hz=250.0,
            config=model.PreprocessingConfig(mode=mode, target_fs_hz=128.0),
        )
        assert processed.shape[0] == 3
        assert processed.shape[1] == processed_time.size
        assert np.isfinite(processed).all()
        baseline = (processed_time >= -0.2) & (processed_time < 0.0)
        np.testing.assert_allclose(processed[:, baseline].mean(axis=1), 0.0, atol=1e-12)


def test_candidate_scenario_grid_keeps_target_and_match_assumptions_explicit():
    model = _model_module()
    scenarios = model.build_sensitivity_scenarios(
        cue_side="right",
        target_types=("dots", "inward"),
        target_onsets_s=(2.0, 2.4),
        target_duration_s=0.3,
        match_evidence_values=(1.0, -1.0),
    )

    assert len(scenarios) == 8
    assert {scenario.cue_side for scenario in scenarios} == {"right"}
    assert {scenario.target_stimulus for scenario in scenarios} == {"dots", "inward"}
    assert {scenario.target_onset_s for scenario in scenarios} == {2.0, 2.4}
    assert {scenario.target_duration_s for scenario in scenarios} == {0.3}
    assert {scenario.match_evidence for scenario in scenarios} == {-1.0, 1.0}
