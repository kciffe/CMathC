"""Low-dimensional Wilson-Cowan populations and fixed observation map."""
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
    excitatory: np.ndarray  # population, hemisphere, time
    inhibitory: np.ndarray
    source_proxy: np.ndarray  # six fixed source modes, time
    observable_modes: np.ndarray  # rank-2 G coordinates, time
    eeg: np.ndarray  # F3, Fz, F4, time; relative units
    eeg_scaled: np.ndarray
    u2_unexplained: np.ndarray
    diagnostics: dict


def _project_half_fields(field):
    """Orthonormal left/right half-field projection under area-mean inner product."""
    n = field.shape[0]
    if field.shape[0] != field.shape[1] or n % 2:
        raise ValueError("spatial fields must be even-sized squares")
    # Coefficients are <field, b> where the area-average inner product uses
    # b=sqrt(2) on one half and zero elsewhere. The half-field mean is thus
    # divided by sqrt(2), not multiplied by it.
    left = field[:, :n // 2].mean(axis=(0, 1)) / np.sqrt(2.0)
    right = field[:, n // 2:].mean(axis=(0, 1)) / np.sqrt(2.0)
    return np.stack([left, right])


def project_frontend(frontend):
    """Project B, H_L, H_R to six fixed half-field drives."""
    drives = []
    for name in ("B", "H_L", "H_R"):
        x = np.asarray(frontend[name], dtype=np.float32)
        if x.ndim != 3 or x.shape[0] != x.shape[1]:
            raise ValueError(f"frontend {name} must be [y,x,time]")
        drives.append(_project_half_fields(x))
    return np.stack(drives).astype(np.float32)  # feature class, hemisphere, time


def _delay(signal, time_ms, delay_ms):
    signal = np.asarray(signal)
    target = time_ms - delay_ms
    flat = signal.reshape(-1, signal.shape[-1])
    out = np.stack([np.interp(target, time_ms, row, left=0.0, right=0.0)
                    for row in flat])
    return out.reshape(signal.shape).astype(signal.dtype, copy=False)


def _wc_populations(drives, time_ms, params):
    dt = float(np.median(np.diff(time_ms)))
    p = np.empty_like(drives, dtype=np.float32)
    p[0] = _delay(drives[0], time_ms, config.WC_FIXED["delay_early"])
    p[1:] = _delay(drives[1:], time_ms, config.WC_FIXED["delay_shape"])
    e = np.zeros_like(p, dtype=np.float32)
    inh = np.zeros_like(p, dtype=np.float32)
    w = config.WC_FIXED
    for ti in range(len(time_ms) - 1):
        e_input = w["w_ee"] * e[:, :, ti] - params.g_i * w["w_ei"] * inh[:, :, ti] + w["g_p"] * p[:, :, ti]
        i_input = w["w_ie"] * e[:, :, ti] - w["w_ii"] * inh[:, :, ti] + w["g_q"] * p[:, :, ti]
        fe = activation(e_input, 5.0, 0.35)
        fi = activation(i_input, 5.0, 0.35)
        tau_e = np.array([w["tau_e_early"], params.tau_s, params.tau_s], dtype=float)[:, None]
        tau_i = tau_e * np.array([w["tau_i_early"] / w["tau_e_early"], w["tau_i_ratio"], w["tau_i_ratio"]])[:, None]
        e[:, :, ti + 1] = e[:, :, ti] + dt * (-e[:, :, ti] + (1.0 - e[:, :, ti]) * fe) / tau_e
        inh[:, :, ti + 1] = inh[:, :, ti] + dt * (-inh[:, :, ti] + (1.0 - inh[:, :, ti]) * fi) / tau_i
    tol = 1e-6
    if (not np.isfinite(e).all() or not np.isfinite(inh).all()
            or min(float(e.min()), float(inh.min())) < -tol
            or max(float(e.max()), float(inh.max())) > 1.0 + tol):
        raise FloatingPointError("Wilson-Cowan E/I state left the [0,1] range")
    return e, inh, p


def _synaptic_kernel(tau_ms, dt_ms, n_samples):
    t = np.arange(n_samples, dtype=float) * dt_ms
    h = np.where(t >= 0, t * np.exp(-t / tau_ms) / (tau_ms * tau_ms), 0.0)
    # Discrete area normalization keeps the proxy scale stable as dt changes.
    area = h.sum() * dt_ms
    return (h / area if area > 0 else h).astype(np.float32)


def _source_proxy(e, inh, time_ms):
    dt = float(np.median(np.diff(time_ms)))
    h_e = _synaptic_kernel(10.0, dt, len(time_ms))
    h_i = _synaptic_kernel(20.0, dt, len(time_ms))
    pe = fftconvolve(e, h_e[None, None, :], mode="full", axes=(-1,))[..., :len(time_ms)] * dt
    pi = fftconvolve(inh, h_i[None, None, :], mode="full", axes=(-1,))[..., :len(time_ms)] * dt
    population_source = pe - pi
    p_b = population_source[0]
    p_l = population_source[1]
    p_r = population_source[2]
    sigma = (p_r + p_l) / np.sqrt(2.0)
    delta = (p_r - p_l) / np.sqrt(2.0)
    return np.concatenate([p_b, sigma, delta], axis=0).astype(np.float32)


def _frozen_rank2_map(g):
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    rank = int(np.linalg.matrix_rank(g))
    if rank != 2:
        raise ValueError(f"frozen lead matrix expected rank 2, received {rank}")
    u2, s2, vt2 = u[:, :rank].copy(), s[:rank].copy(), vt[:rank].copy()
    # Fixed signs make the saved observable coordinates deterministic.
    for row in range(rank):
        pivot = np.argmax(np.abs(u2[:, row]))
        if u2[pivot, row] < 0:
            u2[:, row] *= -1
            vt2[row] *= -1
    return u2, s2, vt2


def simulate_forward(frontend, params=None, fixed_observation=None, time_ms=None, amplitude=1.0):
    """Run the six E/I populations, source proxy, and frozen rank-2 EEG map."""
    params = ModelParams.from_any(params)
    if time_ms is None:
        time_ms = np.asarray(frontend["time_ms"], dtype=float)
    else:
        time_ms = np.asarray(time_ms, dtype=float)
    if time_ms.ndim != 1 or len(time_ms) < 2 or not np.all(np.diff(time_ms) > 0):
        raise ValueError("time_ms must be strictly increasing")
    if params.tau_s <= 0 or params.g_i <= 0 or params.tau_a <= 0 or amplitude < 0:
        raise ValueError("positive time constants/g_i and nonnegative amplitude required")
    drives = project_frontend(frontend)
    e, inh, delayed_drive = _wc_populations(drives, time_ms, params)
    source = _source_proxy(e, inh, time_ms)
    g = config.LEAD_FIELD if fixed_observation is None else np.asarray(fixed_observation, dtype=float)
    if g.shape != (3, 6) or np.linalg.matrix_rank(g) != 2:
        raise ValueError("observation matrix must be fixed 3x6 rank 2")
    u2, s2, vt2 = _frozen_rank2_map(g)
    modes = (s2[:, None] * (vt2 @ source)).astype(np.float32)
    prediction = (u2 @ modes).astype(np.float32)
    direct = (g @ source).astype(np.float32)
    if not np.allclose(prediction, direct, rtol=1e-5, atol=1e-6):
        raise FloatingPointError("rank-2 SVD readout disagrees with the fixed lead map")
    eeg_scaled = (amplitude * prediction).astype(np.float32)
    return ModelResult(
        time_ms=time_ms.copy(), excitatory=e, inhibitory=inh, source_proxy=source,
        observable_modes=modes, eeg=prediction, eeg_scaled=eeg_scaled,
        u2_unexplained=np.zeros_like(prediction[0]),
        diagnostics={"G_rank": 2, "G": g.copy(), "amplitude": float(amplitude),
                     "drive_max": float(np.max(delayed_drive)),
                     "E_range": [float(e.min()), float(e.max())],
                     "I_range": [float(inh.min()), float(inh.max())],
                     "source_labels": ["B_left", "B_right", "Sigma_left", "Sigma_right", "Delta_left", "Delta_right"]})


def observable_modes_from_real(real):
    """Project F3/Fz/F4 onto the two modes observable by the rank-2 lead map."""
    y = np.asarray(real, dtype=float)
    if y.shape[-2] != 3:
        raise ValueError("real data channel axis must be F3/Fz/F4")
    return np.einsum("kc,...ct->...kt", config.U_OBS, y)


def unexplained_mode_from_real(real):
    """Project the sensor component orthogonal to the rank-2 lead map (u2)."""
    y = np.asarray(real, dtype=float)
    if y.shape[-2] != 3:
        raise ValueError("real data channel axis must be F3/Fz/F4")
    return np.einsum("c,...ct->...t", config.U2, y)
