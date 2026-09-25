"""First-round context and sensor-mapping validation for revision v3.

The functions here are deliberately small and deterministic so the scene
timeline, abstract source labels, and regularized observation fit can be tested
independently of the expensive visual simulation.
"""

import numpy as np

try:
    from .head_model import SOURCE_LABELS
except ImportError:
    from head_model import SOURCE_LABELS


def build_scene_timeline(time_ms, cue_onset_ms=0.0, cue_offset_ms=200.0,
                         target_onset_ms=None):
    """Return scene ids: 0=circle background, 1=cue, 2=task target.

    The target event is optional because only the cue event is needed for the
    0--800 ms readout. When included, it represents a declared nuisance scene
    used to reproduce the complete epoch passed through Q1's zero-phase filter.
    """
    time = np.asarray(time_ms, dtype=float)
    if (time.ndim != 1 or time.size < 2 or not np.isfinite(time).all()
            or not np.all(np.diff(time) > 0)):
        raise ValueError("time_ms must be a finite, strictly increasing vector")
    if not np.isfinite(cue_onset_ms) or not np.isfinite(cue_offset_ms) or cue_offset_ms <= cue_onset_ms:
        raise ValueError("cue onset and offset must define a positive duration")
    scene = np.zeros(time.size, dtype=np.int8)
    scene[(time >= cue_onset_ms) & (time < cue_offset_ms)] = 1
    if target_onset_ms is not None:
        if not np.isfinite(target_onset_ms) or target_onset_ms <= cue_offset_ms:
            raise ValueError("target onset must follow the cue offset")
        scene[time >= target_onset_ms] = 2
    return scene


def source_channel_labels():
    """Names for the five sources after explicit field/preference routing."""
    return SOURCE_LABELS


def fit_effective_mapping(source_curves, observed_curves, prior_map, alpha=1.0):
    """Fit a ridge-regularized linear map toward ``prior_map`` using training ERPs.

    source_curves has shape [condition, source, time], observed_curves has
    [condition, sensor, time]. Source standardization is estimated from these
    training curves only and undone before returning the [sensor,source] map.
    ``alpha`` is dimensionless relative to mean source covariance trace.
    """
    source = np.asarray(source_curves, dtype=float)
    observed = np.asarray(observed_curves, dtype=float)
    prior = np.asarray(prior_map, dtype=float)
    if (source.ndim != 3 or observed.ndim != 3 or source.shape[0] != observed.shape[0]
            or source.shape[2] != observed.shape[2] or prior.shape != (observed.shape[1], source.shape[1])):
        raise ValueError("source, observation, and prior mapping shapes are inconsistent")
    if not (np.isfinite(source).all() and np.isfinite(observed).all()
            and np.isfinite(prior).all() and np.isfinite(alpha) and alpha >= 0):
        raise ValueError("mapping inputs and alpha must be finite; alpha must be nonnegative")
    x = source.transpose(1, 0, 2).reshape(source.shape[1], -1)
    y = observed.transpose(1, 0, 2).reshape(observed.shape[1], -1)
    scales = np.sqrt(np.mean(x * x, axis=1))
    scales[scales < 1e-12] = 1.0
    xs = x / scales[:, None]
    n = xs.shape[1]
    covariance = (xs @ xs.T) / n
    cross_covariance = (y @ xs.T) / n
    penalty = float(alpha) * float(np.trace(covariance)) / source.shape[1]
    prior_standardized = prior * scales[None, :]
    system = covariance + penalty * np.eye(source.shape[1])
    standardized_map = (cross_covariance + penalty * prior_standardized) @ np.linalg.pinv(system)
    return standardized_map / scales[None, :]


def fit_shared_gain(prediction, observed):
    """Estimate one signed scalar using training data only."""
    pred = np.asarray(prediction, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if pred.shape != obs.shape or not np.isfinite(pred).all() or not np.isfinite(obs).all():
        raise ValueError("prediction and observation must be matching finite arrays")
    denom = float(np.sum(pred * pred))
    return float(np.sum(pred * obs) / denom) if denom > 1e-20 else 0.0


def model_sources_to_q1_grid(source_proxy, time_ms, source_fs=256.0, target_fs=128.0):
    """Filter a full-epoch source matrix with Q1's operator on its 256-Hz grid."""
    try:
        from .observation import filter_resample_baseline
    except ImportError:
        from observation import filter_resample_baseline
    source = np.asarray(source_proxy, dtype=float)
    time = np.asarray(time_ms, dtype=float)
    if source.ndim != 2 or source.shape[1] != time.size or not np.all(np.diff(time) > 0):
        raise ValueError("source_proxy must be [source,time] on a strictly increasing axis")
    input_time = -1.0 + np.arange(int(round(4.0 * source_fs))) / source_fs
    sampled = np.vstack([np.interp(input_time * 1000.0, time, row) for row in source])
    observed, observed_time = filter_resample_baseline(sampled, source_fs, target_fs,
                                                        start_s=-1.0)
    return observed, observed_time * 1000.0
