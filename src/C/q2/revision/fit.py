"""Bounded leave-one-record forward-model fitting for the Stage1 ERPs."""
from dataclasses import asdict, dataclass
from functools import lru_cache

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


def fit_model(train_cases, model_spec=None, seed=config.SEED, resolution=64,
              max_nfev=config.MAX_NFEV_PER_START):
    """Fit tauS, gI, tauA and one shared nonnegative amplitude using Stage1 only.

    A record receives equal weight and its two cue conditions receive equal
    weight. The held-out record must not be included in ``train_cases``.
    """
    cases = [c for c in train_cases if c["stage"] == "Stage1" and c["eligible_fit"]]
    records = sorted({c["dataset"] for c in cases})
    if len(records) < 2:
        raise ValueError("at least two training MAT records are required")
    by_record = {r: {c["condition"] for c in cases if c["dataset"] == r} for r in records}
    if any(labels != {"left", "right"} for labels in by_record.values()):
        raise ValueError("each fitting record must have eligible left and right Stage1 cases")

    # The visual front end is shared by records. Cache only a few recent tauA
    # values because it is the only fitted parameter that affects that stage.
    right_stimulus = load_stimulus("Stage1", "right")

    @lru_cache(maxsize=6)
    def cached_left_frontend(tau_a):
        stimulus = load_stimulus("Stage1", "left")
        return simulate_frontend(stimulus, params={"tau_a": tau_a}, resolution=resolution)

    def cached_frontend(condition, tau_a):
        left = cached_left_frontend(tau_a)
        if condition == "left":
            return left
        return mirror_stage1_frontend(left, right_stimulus)

    record_count = len(records)
    weights = {c["dataset"]: 1.0 / record_count / 2.0 for c in cases}
    eval_counter = {"n": 0}

    def evaluate_candidate(x, return_predictions=False):
        tau_s, g_i, tau_a = map(float, x)
        params = ModelParams(tau_s, g_i, tau_a)
        model_by_condition = {}
        for condition in ("left", "right"):
            front = cached_frontend(condition, round(tau_a, 7))
            model_by_condition[condition] = simulate_forward(front, params=params,
                                                              time_ms=front["time_ms"])
        predicted = []
        actual = []
        case_weights = []
        for case in cases:
            result = model_by_condition[case["condition"]]
            pred = _interpolate_prediction(result.eeg, result.time_ms, case["time_ms"])
            predicted.append(pred)
            actual.append(np.asarray(case["real"], dtype=float))
            case_weights.append(weights[case["dataset"]])
        denom = sum(w * np.mean(pred * pred) for pred, w in zip(predicted, case_weights))
        if not np.isfinite(denom) or denom <= 1e-18:
            return float("inf"), 0.0, predicted
        numer = sum(w * np.mean(pred * obs) for pred, obs, w in zip(predicted, actual, case_weights))
        amplitude = max(0.0, float(numer / denom))
        loss = sum(w * np.mean((obs - amplitude * pred) ** 2)
                   for pred, obs, w in zip(predicted, actual, case_weights))
        if not np.isfinite(loss):
            return float("inf"), amplitude, predicted
        return float(loss), amplitude, predicted

    lower = np.array([config.PARAM_BOUNDS[k][0] for k in ("tau_s", "g_i", "tau_a")])
    upper = np.array([config.PARAM_BOUNDS[k][1] for k in ("tau_s", "g_i", "tau_a")])
    starts = []
    best = None
    for start_i, start in enumerate(config.FIT_STARTS):
        start = np.clip(np.asarray(start, dtype=float), lower, upper)
        local = {"evaluations": 0, "last_value": None, "best": None}

        def objective(x):
            if local["evaluations"] >= max_nfev:
                return local["last_value"] if local["last_value"] is not None else 1e30
            local["evaluations"] += 1
            eval_counter["n"] += 1
            try:
                loss, amplitude, _ = evaluate_candidate(x)
            except (FloatingPointError, ValueError, OverflowError):
                loss, amplitude = float("inf"), 0.0
            valid = np.isfinite(loss) and loss < 1e29
            if not valid:
                loss = 1e30
            local["last_value"] = loss
            if valid:
                candidate = (loss, amplitude, np.asarray(x, dtype=float).copy())
                if local["best"] is None or loss < local["best"][0]:
                    local["best"] = candidate
            return loss

        opt = minimize(objective, start, method="L-BFGS-B", bounds=list(zip(lower, upper)),
                       options={"maxfun": int(max_nfev), "maxiter": int(max_nfev),
                                "ftol": 1e-8, "maxls": 12})
        # The optimizer may terminate at a point it did not evaluate after the
        # bounded objective budget. Score the best cached point from this start.
        candidate = local["best"]
        if candidate is None:
            candidate = (float("inf"), 0.0, start)
        loss, amplitude, xbest = candidate
        starts.append({"start_index": start_i, "initial": start.tolist(),
                       "optimizer_success": bool(opt.success), "optimizer_status": int(opt.status),
                       "optimizer_message": str(opt.message), "nfev_reported": int(opt.nfev),
                       "n_objective_evaluations": int(local["evaluations"]),
                       "best_loss": float(loss), "amplitude": float(amplitude),
                       "parameters": dict(zip(("tau_s", "g_i", "tau_a"), map(float, xbest)))})
        if np.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, amplitude, xbest.copy(), bool(opt.success), int(opt.status))

    default = np.array([config.PARAM_DEFAULTS[k] for k in ("tau_s", "g_i", "tau_a")], dtype=float)
    if best is None:
        try:
            loss, amplitude, _ = evaluate_candidate(default)
            if not np.isfinite(loss):
                raise FloatingPointError("default model has no finite candidate loss")
            xbest, success, status, state = default, False, -1, "fit_failed_default_used"
        except Exception as exc:
            raise RuntimeError(f"all fitted candidates and default failed: {exc}") from exc
    else:
        loss, amplitude, xbest, success, status = best
        state = "converged" if success else "best_finite_candidate_optimizer_not_converged"
    names = ("tau_s", "g_i", "tau_a")
    values = dict(zip(names, map(float, xbest)))
    flags = {name: bool(min(abs(values[name] - config.PARAM_BOUNDS[name][0]),
                            abs(config.PARAM_BOUNDS[name][1] - values[name]))
                        <= 0.02 * (config.PARAM_BOUNDS[name][1] - config.PARAM_BOUNDS[name][0]))
             for name in names}
    return FitResult(records, values, float(amplitude), float(loss), bool(success), state,
                     flags, starts, int(eval_counter["n"]), int(resolution))


def fit_leave_one_record(cases, resolution=64):
    """Fit one immutable parameter set per held-out recording."""
    eligible = [c for c in cases if c["stage"] == "Stage1" and c["eligible_fit"]]
    records = sorted({c["dataset"] for c in eligible})
    results = {}
    for heldout in records:
        train = [c for c in eligible if c["dataset"] != heldout]
        results[heldout] = fit_model(train, resolution=resolution)
    return results
