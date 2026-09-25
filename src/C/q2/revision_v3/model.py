"""Low-dimensional forward model with explicit field/preference routing.

Population-channel semantics are kept separate from cortical-source labels:
visual-field halves route contralaterally to early/configuration proxies, while
triangle-template preference becomes one bilateral midline opponent proxy.
The resulting F3/Fz/F4 curves use a canonical point-dipole approximation and
relative source currents; they are not individualized source localization or
calibrated microvolt predictions.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

try:
    from . import config
    from .frontend import activation
    from .head_model import SOURCE_LABELS, build_sensor_leadfield
except ImportError:
    import config
    from frontend import activation
    from head_model import SOURCE_LABELS, build_sensor_leadfield


@dataclass(frozen=True)
class ModelParams:
    tau_s: float = config.PARAM_DEFAULTS["tau_s"]
    g_i: float = config.PARAM_DEFAULTS["g_i"]
    tau_a: float = config.PARAM_DEFAULTS["tau_a"]

    @classmethod
    def from_any(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        return cls(float(value.get("tau_s", config.PARAM_DEFAULTS["tau_s"])),
                   float(value.get("g_i", config.PARAM_DEFAULTS["g_i"])),
                   float(value.get("tau_a", config.PARAM_DEFAULTS["tau_a"])))


@dataclass
class ModelResult:
    time_ms: np.ndarray
    excitatory: np.ndarray  # population, channel, time
    inhibitory: np.ndarray
    source_proxy: np.ndarray  # five semantically mapped filtered E-I currents, relative units
    observable_modes: np.ndarray  # u0/u1/u2 coordinates, time
    eeg: np.ndarray  # F3/Fz/F4 relative sensor units, time
    eeg_scaled: np.ndarray
    u2_unexplained: np.ndarray
    diagnostics: dict


def _project_half_fields(field):
    """Project a square map onto two normalized visual-field halves."""
    x = np.asarray(field, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != x.shape[1] or x.shape[0] % 2:
        raise ValueError("spatial fields must be even-sized [y,x,time] arrays")
    return np.stack([x[:, :x.shape[1] // 2].mean(axis=(0, 1)) / np.sqrt(2.0),
                     x[:, x.shape[1] // 2:].mean(axis=(0, 1)) / np.sqrt(2.0)])


def calibrate_drive_scales(frontends, feature_route="opponent", window_ms=(0.0, 200.0)):
    """Calibrate every input channel on fixed, mirrored reference stimuli.

    The scale is computed once from visual inputs only, with equal weight for
    the canonical left and right cue. It must be passed unchanged into fits.
    """
    if not isinstance(frontends, dict) or not frontends:
        raise ValueError("frontends must map condition names to reference feature maps")
    drive_sets = []
    for frontend in frontends.values():
        time = np.asarray(frontend["time_ms"], dtype=float)
        mask = (time >= window_ms[0]) & (time <= window_ms[1])
        if not mask.any():
            raise ValueError("calibration interval does not overlap feature time")
        drive_sets.append(project_frontend(frontend, feature_route=feature_route)[:, :, mask])
    stacked = np.stack(drive_sets, axis=0)
    scales = np.sqrt(np.mean(stacked ** 2, axis=(0, 3)))
    active = scales > 1e-8
    scales[~active] = 1.0
    return scales.astype(np.float32)


def project_frontend(frontend, feature_route="opponent", drive_scales=None):
    """Project frontend maps to three populations with two channels each.

    The opponent route uses B (early contrast), a common shape map, and two
    rectified orientation-template channels. The legacy route preserves the
    former B/H_L/H_R design for a frozen-parameter ablation.
    """
    if feature_route not in ("opponent", "legacy"):
        raise ValueError("feature_route must be 'opponent' or 'legacy'")
    names = ("B", "H_L", "H_R") if feature_route == "legacy" else ("B", "shape")
    drives = [_project_half_fields(frontend[name]) for name in names]
    if feature_route == "opponent":
        left = np.asarray(frontend["dir_left"], dtype=np.float32)
        right = np.asarray(frontend["dir_right"], dtype=np.float32)
        if left.ndim != 1 or right.shape != left.shape:
            raise ValueError("opponent template drives must be matching [time] vectors")
        if np.any(left < -1e-7) or np.any(right < -1e-7):
            raise ValueError("opponent template drives must be nonnegative")
        drives.append(np.stack([left, right]))
    result = np.stack(drives).astype(np.float32)
    if result.shape[1] != 2 or result.shape[-1] != len(frontend["time_ms"]):
        raise ValueError("projected drive shape is inconsistent with the time axis")
    if drive_scales is not None:
        scales = np.asarray(drive_scales, dtype=np.float32)
        if scales.shape != result.shape[:2] or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("drive_scales must be a positive finite [population,channel] matrix")
        result = result / scales[:, :, None]
    return result


def _delay(signal, time_ms, delay_ms):
    signal = np.asarray(signal)
    target = time_ms - delay_ms
    flat = signal.reshape(-1, signal.shape[-1])
    out = np.stack([np.interp(target, time_ms, row, left=0.0, right=0.0) for row in flat])
    return out.reshape(signal.shape).astype(signal.dtype, copy=False)


def _causal_history_sample(signal, step, delay_samples):
    """Read a fractional delayed state from the already simulated history."""
    whole = int(np.floor(delay_samples))
    fraction = float(delay_samples - whole)
    index = int(step) - whole
    if index < 0 or (index == 0 and fraction > 1e-12):
        return np.zeros(signal.shape[:-1], dtype=signal.dtype)
    current = signal[..., index]
    if fraction <= 1e-12:
        return current
    previous = signal[..., index - 1] if index > 0 else np.zeros_like(current)
    return (1.0 - fraction) * current + fraction * previous


def _route_early_feedforward(early_activity, step, delay_samples):
    """Return field-aligned and common bilateral routes from early activity.

    The configuration group receives field-aligned channels. Both channels
    of the shape-template-preference group receive the same fixed weighted
    average of left/right visual-field activity; their template drives remain
    separate and are what encode preference.
    """
    field = _causal_history_sample(early_activity, step, delay_samples)
    weights = np.asarray(config.SHAPE_FEEDFORWARD_FIELD_WEIGHTS, dtype=float)
    if (field.ndim != 1 or weights.shape != field.shape
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("shape feedforward weights must be finite, nonnegative, and sum to one")
    pooled = float(np.dot(weights, field))
    preference = np.full(2, pooled, dtype=field.dtype)
    return field, preference


def _wc_populations(drives, time_ms, params):
    dt = float(np.median(np.diff(time_ms)))
    drive = np.empty_like(drives, dtype=np.float32)
    drive[0] = _delay(drives[0], time_ms, config.WC_FIXED["delay_early"])
    drive[1:] = _delay(drives[1:], time_ms, config.WC_FIXED["delay_shape"])
    e = np.zeros_like(drive, dtype=np.float32)
    inh = np.zeros_like(drive, dtype=np.float32)
    w = config.WC_FIXED
    tau_e = np.array([w["tau_e_early"], params.tau_s, params.tau_s], dtype=float)[:, None]
    tau_i = tau_e * np.array([w["tau_i_early"] / w["tau_e_early"],
                              w["tau_i_ratio"], w["tau_i_ratio"]])[:, None]
    delayed_steps = w["delay_shape"] / dt
    for ti in range(len(time_ms) - 1):
        e_input = (w["w_ee"] * e[:, :, ti] - params.g_i * w["w_ei"] * inh[:, :, ti]
                   + w["g_p"] * drive[:, :, ti])
        i_input = (w["w_ie"] * e[:, :, ti] - w["w_ii"] * inh[:, :, ti]
                   + w["g_q"] * drive[:, :, ti])
        feedforward_e_field, feedforward_e_preference = _route_early_feedforward(
            e[0], ti, delayed_steps)
        feedforward_i_field, feedforward_i_preference = _route_early_feedforward(
            inh[0], ti, delayed_steps)
        e_input[1] += w["w_feedforward_e"] * feedforward_e_field
        e_input[2] += w["w_feedforward_e"] * feedforward_e_preference
        i_input[1] += w["w_feedforward_i"] * feedforward_i_field
        i_input[2] += w["w_feedforward_i"] * feedforward_i_preference
        fe = activation(e_input, 5.0, 0.35)
        fi = activation(i_input, 5.0, 0.35)
        e[:, :, ti + 1] = e[:, :, ti] + dt * (-e[:, :, ti] + (1 - e[:, :, ti]) * fe) / tau_e
        inh[:, :, ti + 1] = (inh[:, :, ti] + dt * (-inh[:, :, ti]
                          + (1 - inh[:, :, ti]) * fi) / tau_i)
    if (not np.isfinite(e).all() or not np.isfinite(inh).all()
            or min(float(e.min()), float(inh.min())) < -1e-6
            or max(float(e.max()), float(inh.max())) > 1.000001):
        raise FloatingPointError("Wilson-Cowan E/I state left the [0,1] range")
    return e, inh, drive


def _synaptic_kernel(tau_ms, dt_ms, n_samples):
    t = np.arange(n_samples, dtype=float) * dt_ms
    h = t * np.exp(-t / tau_ms) / (tau_ms * tau_ms)
    area = h.sum() * dt_ms
    return (h / area if area > 0 else h).astype(np.float32)


def map_population_to_source_channels(population_currents):
    """Map population channels to sources without conflating their meanings.

    Input order is ``[early, configuration, shape_preference] x [L,R] x time``.
    The first two pairs are visual-field halves and use contralateral cortical
    sources. The final pair is left/right triangle-template preference and is
    reduced to one signed bilateral midline opponent current.
    """
    values = np.asarray(population_currents, dtype=np.float32)
    if values.ndim != 3 or values.shape[:2] != (3, 2):
        raise ValueError("population_currents must have shape [3 populations, 2 channels, time]")
    if not np.isfinite(values).all():
        raise ValueError("population_currents must be finite")
    early, configuration, preference = values
    return np.stack((
        early[1],             # left cortical hemisphere <- right visual field
        early[0],             # right cortical hemisphere <- left visual field
        configuration[1],     # left cortical hemisphere <- right visual field
        configuration[0],     # right cortical hemisphere <- left visual field
        preference[0] - preference[1],  # bilateral shape-template opponent
    )).astype(np.float32, copy=False)


def _source_proxy(e, inh, time_ms):
    """Filter E/I activity, then map visual fields and shape preference to sources."""
    dt = float(np.median(np.diff(time_ms)))
    he = _synaptic_kernel(10.0, dt, len(time_ms))
    hi = _synaptic_kernel(20.0, dt, len(time_ms))
    pe = fftconvolve(e, he[None, None, :], mode="full", axes=(-1,))[..., :len(time_ms)] * dt
    pi = fftconvolve(inh, hi[None, None, :], mode="full", axes=(-1,))[..., :len(time_ms)] * dt
    return map_population_to_source_channels(pe - pi)


def simulate_forward(frontend, params=None, time_ms=None, amplitude=1.0,
                     feature_route="opponent", drive_scales=None,
                     leadfield=None):
    """Run the cascade and project five explicit source currents to F3/Fz/F4.

    The sensor map is calculated from fixed source/electrode geometry and a
    homogeneous-conductor approximation. Only one shared signed scale may be
    fitted; no sensor-mode or per-electrode weights are optimized.
    """
    params = ModelParams.from_any(params)
    time = np.asarray(frontend["time_ms"] if time_ms is None else time_ms, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError("time_ms must be a strictly increasing vector")
    if (params.tau_s <= 0 or params.g_i <= 0 or params.tau_a <= 0
            or not np.isfinite(amplitude)):
        raise ValueError("invalid parameters or amplitude")
    drives = project_frontend(frontend, feature_route=feature_route,
                              drive_scales=drive_scales)
    e, inh, delayed_drive = _wc_populations(drives, time, params)
    source = _source_proxy(e, inh, time)
    lead = build_sensor_leadfield() if leadfield is None else np.asarray(leadfield, dtype=float)
    if lead.shape != (3, len(SOURCE_LABELS)) or not np.isfinite(lead).all():
        raise ValueError(f"leadfield must be a finite [F3/Fz/F4, {len(SOURCE_LABELS)} sources] matrix")
    eeg = (lead @ source).astype(np.float32)
    modes = observable_modes_from_real(eeg)
    return ModelResult(
        time_ms=time.copy(), excitatory=e, inhibitory=inh, source_proxy=source,
        observable_modes=modes, eeg=eeg, eeg_scaled=(float(amplitude) * eeg).astype(np.float32),
        u2_unexplained=np.zeros(len(time), dtype=np.float32),
        diagnostics={"feature_route": feature_route, "amplitude": float(amplitude),
                     "drive_max": float(np.max(delayed_drive)),
                     "drive_scales": (np.asarray(drive_scales).tolist()
                                      if drive_scales is not None else None),
                     "leadfield": lead.tolist(),
                     "E_range": [float(e.min()), float(e.max())],
                     "I_range": [float(inh.min()), float(inh.max())],
                     "population_channel_labels": {
                         "early": ("left_visual_field", "right_visual_field"),
                         "configuration": ("left_visual_field", "right_visual_field"),
                         "shape_preference": ("left_triangle_template_preference",
                                              "right_triangle_template_preference")},
                     "source_channel_labels": SOURCE_LABELS,
                     "visual_field_to_hemisphere": {
                         "left_visual_field": "right_hemisphere",
                         "right_visual_field": "left_hemisphere"},
                     "shape_preference_source_rule": (
                         "left preference minus right preference -> bilateral midline opponent proxy; this is a modeling hypothesis, not an anatomical localization"),
                     "shape_preference_feedforward_rule": (
                         "both template-preference channels receive the same fixed weighted pool of early left/right visual-field activity; template-specific drives encode preference"),
                     "shape_preference_feedforward_weights":
                         list(config.SHAPE_FEEDFORWARD_FIELD_WEIGHTS),
                     "mode_labels": ("u0_common", "u1_lateral", "u2_shape"),
                     "observation": "geometric leadfield applied to five semantically mapped source-current proxies"})


def observable_modes_from_real(real):
    """Project F3/Fz/F4 measurements onto fixed orthogonal sensor coordinates."""
    y = np.asarray(real, dtype=float)
    if y.shape[-2] != 3:
        raise ValueError("real data channel axis must be F3/Fz/F4")
    return np.einsum("kc,...ct->...kt", config.U_OBS, y)


def unexplained_mode_from_real(real):
    """Return u2, previously omitted by the rank-two v1 observation map."""
    return observable_modes_from_real(real)[..., 2, :]
