"""Regression checks for the bounded forward-model optimizer."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import fit


def test_tau_a_profile_has_fixed_candidates_covering_bounds():
    np.testing.assert_array_equal(config.FIT_TAU_A_GRID, np.arange(40.0, 161.0, 20.0))
    assert config.FIT_TAU_A_GRID[0] == config.PARAM_BOUNDS["tau_a"][0]
    assert config.FIT_TAU_A_GRID[-1] == config.PARAM_BOUNDS["tau_a"][1]


def test_two_parameter_optimizer_uses_resolvable_steps():
    options = fit._optimizer_options(60)

    np.testing.assert_array_equal(options["eps"], [0.5, 0.01])
    assert options["maxfun"] == 60
    assert options["maxiter"] == 60


def test_fixed_tau_a_frontend_is_cached(monkeypatch):
    fit._cached_left_frontend.cache_clear()
    calls = []
    sentinel = {"time_ms": np.array([0.0, 1.0])}
    monkeypatch.setattr(fit, "load_stimulus", lambda *_: object())

    def fake_frontend(stimulus, params, resolution):
        calls.append((params["tau_a"], resolution))
        return sentinel

    monkeypatch.setattr(fit, "simulate_frontend", fake_frontend)
    first = fit._cached_left_frontend(80.0, 128)
    second = fit._cached_left_frontend(80.0, 128)

    assert first is second is sentinel
    assert calls == [(80.0, 128)]
    fit._cached_left_frontend.cache_clear()


def test_multistart_runs_only_for_the_winning_tau_a_profile(monkeypatch):
    monkeypatch.setattr(config, "FIT_TAU_A_GRID", np.array([40.0, 60.0]))
    monkeypatch.setattr(config, "FIT_STARTS", (
        np.array([40.0, 1.0, 80.0]), np.array([28.0, 0.7, 55.0]),
        np.array([75.0, 1.3, 135.0])))
    time_ms = np.array([0.0, 1.0])
    front = {"time_ms": time_ms}
    monkeypatch.setattr(fit, "load_stimulus", lambda *_: object())
    monkeypatch.setattr(fit, "_cached_left_frontend", lambda *_: front)
    monkeypatch.setattr(fit, "mirror_stage1_frontend", lambda *_: front)
    monkeypatch.setattr(fit, "simulate_forward", lambda *_args, **_kwargs:
                        SimpleNamespace(eeg=np.ones((3, 2)), time_ms=time_ms))
    optimizer_starts = []

    def fake_minimize(objective, x0, **_kwargs):
        optimizer_starts.append(np.asarray(x0).copy())
        objective(np.asarray(x0))
        return SimpleNamespace(success=True, status=0, nfev=1, message="test")

    monkeypatch.setattr(fit, "minimize", fake_minimize)
    cases = []
    for record in ("A", "B"):
        for condition in ("left", "right"):
            cases.append({"dataset": record, "stage": "Stage1", "condition": condition,
                          "eligible_fit": True, "time_ms": time_ms,
                          "real": np.zeros((3, 2))})

    result = fit.fit_model(cases, resolution=64)

    assert len(optimizer_starts) == 4  # two profile starts + two winner confirmations
    assert [row["tau_a_profile_ms"] for row in result.starts] == [40.0, 60.0, 40.0, 40.0]


def test_fit_objective_ignores_unobservable_u2_residual(monkeypatch):
    time_ms = np.arange(4, dtype=float)
    observable = np.array([[1.0, 2.0, 3.0, 2.0],
                           [0.5, 1.0, 0.0, -0.5]])
    prediction = config.U_OBS.T @ observable
    frontend = {"time_ms": time_ms}
    monkeypatch.setattr(fit, "simulate_forward", lambda *_args, **_kwargs:
                        SimpleNamespace(eeg=prediction, time_ms=time_ms))
    u2 = config.U2[:, None] * np.array([[0.0, 3.0, -2.0, 1.0]])
    cases = [
        {"condition": condition, "dataset": "record", "time_ms": time_ms,
         "real": 1.5 * prediction + extra}
        for condition, extra in (("left", np.zeros_like(prediction)), ("right", u2))
    ]
    evaluate_candidate = fit._make_candidate_evaluator(
        80.0, {"left": frontend, "right": frontend}, cases, {"record": 0.5}
    )

    loss, amplitude = evaluate_candidate(np.array([40.0, 1.0]))

    assert amplitude == pytest.approx(1.5)
    assert loss == pytest.approx(0.0, abs=1e-12)
