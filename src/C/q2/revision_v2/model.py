"""Low-dimensional forward model with explicit shape-opponent readout.

The three output coordinates are fixed orthogonal sensor-space modes, not
anatomical sources or a subject-specific head model. The model therefore
supports relative waveform comparisons, not absolute source localization.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

try:
    from . import config
    from .frontend import activation
except ImportError:
    import config
    from frontend import activation


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
    source_proxy: np.ndarray  # six filtered E-I channel signals, relative units
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


def project_frontend(frontend, feature_route="opponent"):
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
    return result


def _delay(signal, time_ms, delay_ms):
    signal = np.asarray(signal)
    target = time_ms - delay_ms
    flat = signal.reshape(-1, signal.shape[-1])
    out = np.stack([np.interp(target, time_ms, row, left=0.0, right=0.0) for row in flat])
    return out.reshape(signal.shape).astype(signal.dtype, copy=False)


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
    for ti in range(len(time_ms) - 1):
        e_input = (w["w_ee"] * e[:, :, ti] - params.g_i * w["w_ei"] * inh[:, :, ti]
                   + w["g_p"] * drive[:, :, ti])
        i_input = (w["w_ie"] * e[:, :, ti] - w["w_ii"] * inh[:, :, ti]
                   + w["g_q"] * drive[:, :, ti])
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


def _source_proxy(e, inh, time_ms):
    """Return filtered E-I activity for the three two-channel populations."""
    dt = float(np.median(np.diff(time_ms)))
    he = _synaptic_kernel(10.0, dt, len(time_ms))
    hi = _synaptic_kernel(20.0, dt, len(time_ms))
    pe = fftconvolve(e, he[None, None, :], mode="full", axes=(-1,))[..., :len(time_ms)] * dt
    pi = fftconvolve(inh, hi[None, None, :], mode="full", axes=(-1,))[..., :len(time_ms)] * dt
    return (pe - pi).reshape(6, len(time_ms)).astype(np.float32)


def _readout_modes(source, feature_route):
    early_left, early_right = source[0], source[1]
    group1_left, group1_right = source[2], source[3]
    group2_left, group2_right = source[4], source[5]
    common_early = (early_left + early_right) / np.sqrt(2.0)
    if feature_route == "opponent":
        opponent = (group2_right - group2_left) / np.sqrt(2.0)
        common_shape = (group1_left + group1_right) / np.sqrt(2.0)
    else:
        opponent = ((group2_left + group2_right) - (group1_left + group1_right)) / 2.0
        common_shape = ((group1_left + group1_right) + (group2_left + group2_right)) / 2.0
    return np.stack([common_early, opponent, common_shape]).astype(np.float32)


def simulate_forward(frontend, params=None, time_ms=None, amplitude=1.0,
                     feature_route="opponent", observation_rank=3):
    """Run fixed WC populations and map three functional modes to sensors.

    ``observation_rank=2`` is a fixed ablation that zeros u2; the primary
    model uses all three orthogonal sensor-space modes. ``amplitude`` is one
    shared signed scalar only, allowing the arbitrary global source polarity
    of the relative-unit model. No channel-specific lead weights are fitted.
    """
    params = ModelParams.from_any(params)
    time = np.asarray(frontend["time_ms"] if time_ms is None else time_ms, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError("time_ms must be a strictly increasing vector")
    if (params.tau_s <= 0 or params.g_i <= 0 or params.tau_a <= 0
            or not np.isfinite(amplitude) or observation_rank not in (2, 3)):
        raise ValueError("invalid parameters, amplitude, or observation rank")
    drives = project_frontend(frontend, feature_route=feature_route)
    e, inh, delayed_drive = _wc_populations(drives, time, params)
    source = _source_proxy(e, inh, time)
    modes = _readout_modes(source, feature_route)
    if observation_rank == 2:
        modes[2] = 0.0
    eeg = (config.U_MATRIX @ modes).astype(np.float32)
    if not np.allclose(config.U_OBS @ eeg, modes, rtol=1e-5, atol=2e-6):
        raise FloatingPointError("fixed sensor-mode transform failed reconstruction")
    return ModelResult(
        time_ms=time.copy(), excitatory=e, inhibitory=inh, source_proxy=source,
        observable_modes=modes, eeg=eeg, eeg_scaled=(float(amplitude) * eeg).astype(np.float32),
        u2_unexplained=np.zeros(len(time), dtype=np.float32),
        diagnostics={"feature_route": feature_route, "observation_rank": observation_rank,
                     "amplitude": float(amplitude), "drive_max": float(np.max(delayed_drive)),
                     "E_range": [float(e.min()), float(e.max())],
                     "I_range": [float(inh.min()), float(inh.max())],
                     "channel_labels": ("early_left", "early_right", "shape_left", "shape_right",
                                        "opponent_left", "opponent_right"),
                     "mode_labels": ("u0_common_early", "u1_opponent", "u2_common_shape")})


def observable_modes_from_real(real):
    """Project F3/Fz/F4 measurements onto fixed orthogonal sensor coordinates."""
    y = np.asarray(real, dtype=float)
    if y.shape[-2] != 3:
        raise ValueError("real data channel axis must be F3/Fz/F4")
    return np.einsum("kc,...ct->...kt", config.U_OBS, y)


def unexplained_mode_from_real(real):
    """Return u2, previously omitted by the rank-two v1 observation map."""
    return observable_modes_from_real(real)[..., 2, :]
