"""Bounded leave-one-record forward-model fitting for the Stage1 ERPs."""
from dataclasses import asdict, dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

try:
    from . import config
    from .frontend import load_stimulus, mirror_stage1_frontend, simulate_frontend
    from .model import ModelParams, simulate_forward
except ImportError:
    import config
    from frontend import load_stimulus, mirror_stage1_frontend, simulate_frontend
    from model import ModelParams, simulate_forward


@dataclass
class FitResult:
    train_records: list
    parameters: dict
    amplitude: float
    loss: float
    success: bool
    status: str
    boundary_flags: dict
    starts: list
    n_evaluations: int
    resolution: int

    def to_dict(self):
        return asdict(self)


def _interpolate_prediction(prediction, source_time, target_time):
    return np.vstack([np.interp(target_time, source_time, row) for row in prediction])


def _optimizer_options(max_nfev):
    """Return optimizer settings with parameter-resolvable finite differences."""
    return {
        "maxfun": int(max_nfev),
        "maxiter": int(max_nfev),
        "ftol": 1e-8,
        "maxls": 12,
        "eps": config.FIT_FINITE_DIFF_STEPS.copy(),
    }


@lru_cache(maxsize=32)
def _cached_left_frontend(tau_a, resolution):
    """Cache the expensive left Stage1 frontend for a fixed profile point."""
    stimulus = load_stimulus("Stage1", "left")
    return simulate_frontend(stimulus, params={"tau_a": float(tau_a)},
                             resolution=int(resolution))


def _make_candidate_evaluator(tau_a, fronts, cases, weights):
    """Build the weighted ERP loss for one fixed visual-adaptation profile."""
    def evaluate_candidate(x):
        params = ModelParams(float(x[0]), float(x[1]), float(tau_a))
        model_by_condition = {
            condition: simulate_forward(front, params=params, time_ms=front["time_ms"])
            for condition, front in fronts.items()
        }
        predicted, actual, case_weights = [], [], []
        for case in cases:
            result = model_by_condition[case["condition"]]
            pred = _interpolate_prediction(result.eeg, result.time_ms, case["time_ms"])
            predicted.append(pred)
            actual.append(np.asarray(case["real"], dtype=float))
            case_weights.append(weights[case["dataset"]])
        denom = sum(w * np.mean(pred * pred) for pred, w in zip(predicted, case_weights))
        if not np.isfinite(denom) or denom <= 1e-18:
            return float("inf"), 0.0
        numer = sum(w * np.mean(pred * obs)
                    for pred, obs, w in zip(predicted, actual, case_weights))
        amplitude = max(0.0, float(numer / denom))
        loss = sum(w * np.mean((obs - amplitude * pred) ** 2)
                   for pred, obs, w in zip(predicted, actual, case_weights))
        return (float(loss), amplitude) if np.isfinite(loss) else (float("inf"), amplitude)
    return evaluate_candidate


def _optimize_start(evaluate_candidate, raw_start, lower, upper, max_nfev):
    """Run one bounded start while retaining its best finite evaluation."""
    start = np.clip(np.asarray(raw_start, dtype=float), lower, upper)
    local = {"evaluations": 0, "last_value": None, "best": None}

    def objective(x):
        if local["evaluations"] >= max_nfev:
            return local["last_value"] if local["last_value"] is not None else 1e30
        local["evaluations"] += 1
        try:
            loss, amplitude = evaluate_candidate(x)
        except (FloatingPointError, ValueError, OverflowError):
            loss, amplitude = float("inf"), 0.0
        if not np.isfinite(loss) or loss >= 1e29:
            loss = 1e30
        local["last_value"] = loss
        if loss < 1e29:
            candidate = (loss, amplitude, np.asarray(x, dtype=float).copy())
            if local["best"] is None or loss < local["best"][0]:
                local["best"] = candidate
        return loss

    opt = minimize(objective, start, method="L-BFGS-B", bounds=list(zip(lower, upper)),
                   options=_optimizer_options(max_nfev))
    candidate = local["best"] or (float("inf"), 0.0, start)
    return {"candidate": candidate, "start": start, "optimizer": opt,
            "evaluations": int(local["evaluations"])}


def fit_model(train_cases, model_spec=None, seed=config.SEED, resolution=64,
              max_nfev=config.MAX_NFEV_PER_START, progress=False):
    """Fit tau_s/g_i per fixed tau_a profile and one shared nonnegative scale.

    Each expensive visual frontend is generated once per tau_a profile. One
    deterministic start is run at every profile; alternate starts confirm only
    the best profile. A record and its cue conditions receive equal weight.
    """
    cases = [c for c in train_cases if c["stage"] == "Stage1" and c["eligible_fit"]]
    records = sorted({c["dataset"] for c in cases})
    if len(records) < 2:
        raise ValueError("at least two training MAT records are required")
    by_record = {r: {c["condition"] for c in cases if c["dataset"] == r} for r in records}
    if any(labels != {"left", "right"} for labels in by_record.values()):
        raise ValueError("each fitting record must have eligible left and right Stage1 cases")

    right_stimulus = load_stimulus("Stage1", "right")
    record_count = len(records)
    weights = {c["dataset"]: 1.0 / record_count / 2.0 for c in cases}
    lower = np.array([config.PARAM_BOUNDS[k][0] for k in ("tau_s", "g_i")])
    upper = np.array([config.PARAM_BOUNDS[k][1] for k in ("tau_s", "g_i")])
    starts_2d = [np.asarray(start[:2], dtype=float) for start in config.FIT_STARTS]
    starts, best = [], None
    best_evaluator = None
    eval_counter = 0

    profile_count = len(config.FIT_TAU_A_GRID)
    for profile_i, tau_a in enumerate(config.FIT_TAU_A_GRID, start=1):
        tau_a = float(tau_a)
        profile_start_time = perf_counter()
        if progress:
            print(f"    tau_a profile {profile_i}/{profile_count}: {tau_a:g} ms started",
                  flush=True)
        left_front = _cached_left_frontend(tau_a, int(resolution))
        fronts = {"left": left_front,
                  "right": mirror_stage1_frontend(left_front, right_stimulus)}

        evaluate_candidate = _make_candidate_evaluator(tau_a, fronts, cases, weights)
        optimized = _optimize_start(evaluate_candidate, starts_2d[0], lower, upper, max_nfev)
        eval_counter += optimized["evaluations"]
        loss, amplitude, xbest = optimized["candidate"]
        opt = optimized["optimizer"]
        params = {"tau_s": float(xbest[0]), "g_i": float(xbest[1]), "tau_a": tau_a}
        starts.append({"tau_a_profile_ms": tau_a, "start_index": 0,
                       "fit_scope": "profile_primary", "initial": optimized["start"].tolist(),
                       "optimizer_success": bool(opt.success), "optimizer_status": int(opt.status),
                       "optimizer_message": str(opt.message), "nfev_reported": int(opt.nfev),
                       "n_objective_evaluations": optimized["evaluations"],
                       "best_loss": float(loss), "amplitude": float(amplitude),
                       "parameters": params})
        if np.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, amplitude, params, bool(opt.success), int(opt.status))
            best_evaluator = evaluate_candidate
        if progress:
            profile_best = min((row["best_loss"] for row in starts
                                if row["tau_a_profile_ms"] == tau_a), default=float("inf"))
            print(f"    tau_a profile {profile_i}/{profile_count} finished in "
                  f"{perf_counter() - profile_start_time:.1f}s; best loss={profile_best:.6g}",
                  flush=True)

    if best is None:
        raise RuntimeError("all tau_a profile fits failed; no finite model candidate")
    # Confirm the selected profile from the other fixed starts. This keeps a
    # small local-minimum check without multiplying every expensive profile fit.
    selected_tau_a = best[2]["tau_a"]
    for start_i, raw_start in enumerate(starts_2d[1:], start=1):
        if progress:
            print(f"    confirming selected tau_a={selected_tau_a:g} ms from start {start_i + 1}/"
                  f"{len(starts_2d)}", flush=True)
        optimized = _optimize_start(best_evaluator, raw_start, lower, upper, max_nfev)
        eval_counter += optimized["evaluations"]
        loss, amplitude, xbest = optimized["candidate"]
        opt = optimized["optimizer"]
        params = {"tau_s": float(xbest[0]), "g_i": float(xbest[1]), "tau_a": selected_tau_a}
        starts.append({"tau_a_profile_ms": selected_tau_a, "start_index": start_i,
                       "fit_scope": "winner_confirmation", "initial": optimized["start"].tolist(),
                       "optimizer_success": bool(opt.success), "optimizer_status": int(opt.status),
                       "optimizer_message": str(opt.message), "nfev_reported": int(opt.nfev),
                       "n_objective_evaluations": optimized["evaluations"],
                       "best_loss": float(loss), "amplitude": float(amplitude),
                       "parameters": params})
        if np.isfinite(loss) and loss < best[0]:
            best = (loss, amplitude, params, bool(opt.success), int(opt.status))
    loss, amplitude, values, success, status = best
    state = "profile_grid_converged" if success else "profile_grid_best_candidate_optimizer_not_converged"
    flags = {name: bool(min(abs(values[name] - config.PARAM_BOUNDS[name][0]),
                            abs(config.PARAM_BOUNDS[name][1] - values[name]))
                        <= 0.02 * (config.PARAM_BOUNDS[name][1] - config.PARAM_BOUNDS[name][0]))
             for name in ("tau_s", "g_i", "tau_a")}
    return FitResult(records, values, float(amplitude), float(loss), bool(success), state,
                     flags, starts, int(eval_counter), int(resolution))


def fit_leave_one_record(cases, resolution=64):
    """Fit one immutable parameter set per held-out recording."""
    eligible = [c for c in cases if c["stage"] == "Stage1" and c["eligible_fit"]]
    records = sorted({c["dataset"] for c in eligible})
    results = {}
    for heldout in records:
        train = [c for c in eligible if c["dataset"] != heldout]
        results[heldout] = fit_model(train, resolution=resolution)
    return results
