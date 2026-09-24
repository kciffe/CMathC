"""Phenomenological forward model. No anatomical source inversion or EEG units."""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from config import Q2_ROOT

# Exactly the original illustrative weights, written in float64.
# Source labels here: V1 space L/R, IT unary space L/R, IT conjunction space L/R.
LEAD_FIELD = np.array([[.32, .24, .48, .36, .72, .58],
                       [.28, .28, .42, .42, .65, .65],
                       [.24, .32, .36, .48, .58, .72]])
ANGLES = (0, 45, 90, 135)
OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))
STIMULI = {("Stage1", "left"): "stage1_cue_left",
           ("Stage1", "right"): "stage1_cue_right",
           ("Stage2", "dots"): "stage2_task1_target_dots",
           ("Stage2", "inward"): "stage2_task2_target_inward",
           ("Stage2", "outward"): "stage2_task2_target_outward"}


def load_contrast(stage, condition):
    name = STIMULI[(stage, condition)]
    base = "stage1_baseline_circle" if stage == "Stage1" else "stage2_baseline_blank"
    image = np.load(Q2_ROOT / "input" / f"{name}.npy").astype(float)
    baseline = np.load(Q2_ROOT / "input" / f"{base}.npy").astype(float)
    # Area averaging, not nearest-neighbour aliasing; retains mirror equivariance.
    return ((image - baseline) / 255).reshape(64, 4, 64, 4).mean(axis=(1, 3))


def stimulus_gate(time_ms, stage, include_offset=True):
    if stage not in ("Stage1", "Stage2"):
        raise ValueError(stage)
    return ((time_ms >= 0) & ((time_ms < 200) if stage == "Stage1" and include_offset else True)).astype(float)


def activation(x):
    zero = 1 / (1 + np.exp(.6))
    return np.clip((1 / (1 + np.exp(-np.clip(4 * (x - .15), -60, 60))) - zero) / (1 - zero), 0, 1)


def lgn_relay(contrast, time_ms, stage, dt_ms, include_offset=True):
    # Center-surround precedes all orientation filtering. Pixel scale is uncalibrated.
    center = gaussian_filter(contrast, .75, mode="reflect")
    surround = gaussian_filter(contrast, 2., mode="reflect")
    field = center - surround
    gate = stimulus_gate(time_ms, stage, include_offset)
    relay = np.zeros((*contrast.shape, len(time_ms)), dtype=np.float32)
    adaptation = np.zeros_like(contrast)
    tcr = np.zeros((2, *contrast.shape))
    interneuron = np.zeros_like(tcr)
    trn = np.zeros_like(tcr)
    means = np.zeros((2, len(time_ms)))
    drive_means = np.zeros_like(means)
    for k in range(len(time_ms) - 1):
        signed = field * gate[k]
        transient = signed - adaptation
        drive = np.stack([.2 * np.maximum(signed, 0) + .8 * np.maximum(transient, 0),
                          .2 * np.maximum(-signed, 0) + .8 * np.maximum(-transient, 0)])
        drive_means[:, k] = drive.mean(axis=(1, 2))
        rt = activation(2.2 * drive - .9 * interneuron - .8 * trn)
        it = activation(1.2 * drive + .7 * tcr)
        nt = activation(.9 * tcr)
        tcr += dt_ms * (rt - tcr) / 18
        interneuron += dt_ms * (it - interneuron) / 12
        trn += dt_ms * (nt - trn) / 25
        adaptation += dt_ms * (signed - adaptation) / 80
        relay[..., k + 1] = tcr[0] - tcr[1]
        means[:, k + 1] = tcr.mean(axis=(1, 2))
    drive_means[:, -1] = drive_means[:, -2]
    return relay, means, drive_means


def gabor_kernel(angle, phase):
    y, x = np.mgrid[-4:5, -4:5]
    theta = np.deg2rad(angle)
    u, v = x * np.cos(theta) + y * np.sin(theta), -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-(u * u + .25 * v * v) / 2) * np.cos(2 * np.pi * u / 2.5 + phase)
    kernel -= kernel.mean()
    return kernel / np.abs(kernel).sum()


def v1_features(relay):
    features = []
    for angle in ANGLES:
        even = fftconvolve(relay, gabor_kernel(angle, 0)[..., None], mode="same", axes=(0, 1))
        odd = fftconvolve(relay, gabor_kernel(angle, np.pi / 2)[..., None], mode="same", axes=(0, 1))
        energy = np.hypot(even, odd)
        features.append(energy.reshape(8, 8, 8, 8, -1).mean(axis=(1, 3)))
    return np.stack(features)


def configuration_features(v1):
    """Ordered orientation pairs at distinct spatial locations; no wrap-around.

    Shape 4 x 8 x 8 x T -> 64 x 8 x 8 x T.
    For each displacement, all 4x4 ordered orientation pairs are retained.
    sqrt(product) has the same units as the single-feature activity.
    """
    results = []
    for dy, dx in OFFSETS:
        shifted = np.zeros_like(v1)
        ys = slice(0, 8 - dy)
        xs = slice(0, 8 - dx) if dx >= 0 else slice(-dx, 8)
        yt = slice(dy, 8)
        xt = slice(dx, 8) if dx >= 0 else slice(0, 8 + dx)
        shifted[:, ys, xs] = v1[:, yt, xt]
        for first in range(4):
            for second in range(4):
                results.append(np.sqrt(np.maximum(v1[first] * shifted[second], 0)))
    return np.stack(results)


def delayed(x, delay_ms, dt_ms):
    n = int(round(delay_ms / dt_ms))
    out = np.zeros_like(x)
    if n == 0:
        return x.copy()
    if n < x.shape[-1]:
        out[..., n:] = x[..., :-n]
    return out


def population(drive, tau_e, dt_ms):
    """Wilson-Cowan rates and low-pass net synaptic-drive proxy (relative units)."""
    e = np.zeros_like(drive, dtype=np.float32)
    inh = np.zeros(drive.shape[:-1])
    proxy = np.zeros_like(e)
    for k in range(drive.shape[-1] - 1):
        net = 1.2 * e[..., k] - inh + 5 * drive[..., k]
        et = activation(net)
        it = activation(e[..., k] - .8 * inh + 2 * drive[..., k])
        e[..., k + 1] = e[..., k] + dt_ms * (-e[..., k] + (1 - e[..., k]) * et) / tau_e
        inh += dt_ms * (-inh + (1 - inh) * it) / (.55 * tau_e)
        proxy[..., k + 1] = proxy[..., k] + dt_ms * (net - proxy[..., k]) / 10
    return e, proxy


def spatial_sources(source):
    return np.stack([source[:, :, :4].mean(axis=(0, 1, 2)),
                     source[:, :, 4:].mean(axis=(0, 1, 2))])


def cortex_from_v1(v1_drive, time_ms, tau_it_ms=40, dt_ms=1):
    v1, v1_current = population(delayed(v1_drive, 8, dt_ms), 18, dt_ms)
    it, it_current = population(delayed(v1, 25, dt_ms), tau_it_ms, dt_ms)
    conjunction = configuration_features(v1)
    cfg, cfg_current = population(delayed(conjunction, 25, dt_ms), tau_it_ms, dt_ms)
    sources = np.concatenate([spatial_sources(v1_current), spatial_sources(it_current), spatial_sources(cfg_current)])
    # Full features preserved here; spatial pooling is only an observation model.
    active = (time_ms >= 50) & (time_ms <= 200)
    return {"eeg": LEAD_FIELD @ sources, "sources": sources,
            "v1_joint_feature": v1[..., active].mean(axis=-1),
            "it_joint_feature": it[..., active].mean(axis=-1),
            "it_configuration_feature": cfg[..., active].mean(axis=-1),
            "v1_drive": v1_drive,
            "population_means": np.stack([v1.mean(axis=(0, 1, 2)), it.mean(axis=(0, 1, 2)), cfg.mean(axis=(0, 1, 2))])}


def simulate(contrast, stage, tau_it_ms=40, dt_ms=1, include_offset=True):
    contrast = np.asarray(contrast, dtype=float)
    if contrast.shape != (64, 64) or not np.isfinite(contrast).all():
        raise ValueError("contrast must be a finite 64x64 matrix")
    if dt_ms <= 0 or dt_ms > 1 or tau_it_ms <= 0:
        raise ValueError("use 0 < dt_ms <= 1 and positive tau_it_ms")
    time_ms = np.arange(round(800 / dt_ms) + 1) * dt_ms
    relay, means, drives = lgn_relay(contrast, time_ms, stage, dt_ms, include_offset)
    out = cortex_from_v1(v1_features(relay), time_ms, tau_it_ms, dt_ms)
    out.update(time_ms=time_ms, lgn_mean=means, lgn_drive_mean=drives,
               stimulus_gate=stimulus_gate(time_ms, stage, include_offset))
    return out
