"""Deterministic stimulus, LGN, Gabor, and triangle-configuration front end."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull

try:
    from . import config
except ImportError:
    import config


@dataclass(frozen=True)
class Stimulus:
    stage: str
    condition: str
    image: np.ndarray
    baseline: np.ndarray
    signed_contrast: np.ndarray
    image_name: str
    baseline_name: str


@dataclass
class ShapeTemplate:
    name: str
    vertices_rc: np.ndarray
    center_rc: np.ndarray
    edge_points_rc: np.ndarray
    edge_angles: np.ndarray
    corner_points_rc: np.ndarray
    corner_angles: np.ndarray
    mask_iou: float


def load_stimulus(stage, condition):
    """Load one standardized matrix and its matching baseline."""
    try:
        image_name, baseline_name = config.STIMULI[(stage, condition)]
    except KeyError as exc:
        raise ValueError(f"unsupported stimulus {stage}/{condition}") from exc
    image = np.load(config.INPUT_ROOT / f"{image_name}.npy").astype(np.float32)
    baseline = np.load(config.INPUT_ROOT / f"{baseline_name}.npy").astype(np.float32)
    if image.shape != (256, 256) or baseline.shape != image.shape:
        raise ValueError(f"{image_name}: expected matching 256x256 image and baseline")
    if not np.isfinite(image).all() or not np.isfinite(baseline).all():
        raise ValueError(f"{image_name}: nonfinite stimulus values")
    return Stimulus(stage, condition, image, baseline, (image - baseline) / 255.0,
                    image_name, baseline_name)


def resize_area(image, size):
    """Area-average a square image to a divisor grid (64 or 128)."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or image.shape[0] % size:
        raise ValueError("area resize requires a square image and an integer scale ratio")
    ratio = image.shape[0] // size
    if ratio == 1:
        return image.copy()
    return image.reshape(size, ratio, size, ratio).mean(axis=(1, 3), dtype=np.float32)


def _normal_pdf(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def activation(x, beta, threshold):
    """Truncated logistic with F(0)=0 and F(+infinity)=1."""
    x = np.asarray(x)
    z = np.clip(beta * (x - threshold), -60.0, 60.0)
    at_zero = 1.0 / (1.0 + np.exp(beta * threshold))
    value = (1.0 / (1.0 + np.exp(-z)) - at_zero) / (1.0 - at_zero)
    value = np.where(x == 0.0, 0.0, value)
    return np.clip(value, 0.0, 1.0)


def _triangle_mask(stimulus):
    changed = np.abs(stimulus.signed_contrast) > 0
    # The cue and its baseline share an identical central circle; the signed
    # difference therefore identifies only the cue triangle.
    return changed


def _triangle_vertices(mask):
    boundary = mask & ~binary_erosion(mask)
    points = np.argwhere(boundary).astype(float)  # row, column in 256-grid
    if len(points) < 3:
        raise ValueError("triangle mask has fewer than three boundary points")
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    best = None
    best_area = -1.0
    for i, j, k in combinations(range(len(hull_points)), 3):
        p, q, r = hull_points[[i, j, k]]
        v1, v2 = q - p, r - p
        area = abs(v1[0] * v2[1] - v1[1] * v2[0]) * 0.5
        if area > best_area:
            best_area, best = area, np.array([p, q, r])
    # Stable clockwise ordering around centroid.
    center = best.mean(axis=0)
    angle = np.arctan2(best[:, 0] - center[0], best[:, 1] - center[1])
    return best[np.argsort(angle)]


def _raster_triangle(vertices, shape):
    rr, cc = np.mgrid[:shape[0], :shape[1]]
    p = np.stack([rr.ravel(), cc.ravel()], axis=1)
    a, b, c = vertices
    v0, v1, v2 = c - a, b - a, p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
    return ((u >= 0) & (v >= 0) & (u + v <= 1)).reshape(shape)


def _edge_normal_angle(p0, p1):
    # Gabor angle describes the carrier normal; triangle edges are unoriented.
    dy, dx = p1 - p0
    return float((np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0)


@lru_cache(maxsize=1)
def build_triangle_templates():
    left = load_stimulus("Stage1", "left")
    right = load_stimulus("Stage1", "right")
    mask_left, mask_right = _triangle_mask(left), _triangle_mask(right)
    if not np.array_equal(mask_right, mask_left[:, ::-1]):
        raise ValueError("left/right cue masks are not exact horizontal mirrors")
    templates = []
    for name, mask in (("left", mask_left), ("right", mask_right)):
        vertices = _triangle_vertices(mask)
        triangle = _raster_triangle(vertices, mask.shape)
        union = np.count_nonzero(triangle | mask)
        iou = np.count_nonzero(triangle & mask) / union if union else 0.0
        if iou < 0.90:
            raise ValueError(f"{name} cue triangle hull IoU {iou:.3f} is below 0.90")
        edge_points, edge_angles, corner_points, corner_angles = [], [], [], []
        for idx in range(3):
            p0, p1 = vertices[idx], vertices[(idx + 1) % 3]
            angle = _edge_normal_angle(p0, p1)
            edge_points.extend([p0 + frac * (p1 - p0) for frac in (0.25, 0.50, 0.75)])
            edge_angles.extend([angle] * 3)
            # Two near-corner supports, one along each adjacent edge.
            prev = vertices[(idx - 1) % 3]
            next_ = vertices[(idx + 1) % 3]
            corner_points.extend([p0 + 0.25 * (prev - p0), p0 + 0.25 * (next_ - p0)])
            corner_angles.extend([_edge_normal_angle(prev, p0), _edge_normal_angle(p0, next_)])
        center_rc = np.argwhere(mask).mean(axis=0)
        templates.append(ShapeTemplate(name, vertices, center_rc, np.asarray(edge_points),
                                        np.asarray(edge_angles), np.asarray(corner_points),
                                        np.asarray(corner_angles), float(iou)))
    return {t.name: t for t in templates}


def map_original_coordinates(points_rc, size):
    ratio = 256.0 / float(size)
    return (np.asarray(points_rc, dtype=float) + 0.5) / ratio - 0.5


def _gabor_pair(angle_deg, size):
    ratio = 256.0 / float(size)
    sigma = config.PIXEL_PARAMS["gabor_sigma"] / ratio
    wavelength = config.PIXEL_PARAMS["gabor_wavelength"] / ratio
    gamma = 0.5
    radius = max(2, int(np.ceil(4.0 * sigma)))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    theta = np.deg2rad(angle_deg)
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-0.5 * ((u / sigma) ** 2 + (v / (gamma * sigma)) ** 2))
    phase = 2.0 * np.pi * u / wavelength
    even = envelope * np.cos(phase)
    odd = envelope * np.sin(phase)
    even -= even.mean()
    odd -= odd.mean()
    norm = np.sqrt(np.sum(even * even + odd * odd))
    if norm <= np.finfo(float).eps:
        raise ValueError("degenerate Gabor kernel")
    return (even / norm).astype(np.float32), (odd / norm).astype(np.float32)


def _orientation_weights(angle_deg):
    angles = np.asarray(config.GABOR_ANGLES_DEG)
    a = angle_deg % 180.0
    upper = int(np.searchsorted(angles, a, side="right")) % len(angles)
    lower = (upper - 1) % len(angles)
    lo, hi = angles[lower], angles[upper]
    if upper == 0:
        hi += 180.0
    target = a if a >= lo else a + 180.0
    alpha = (target - lo) / (hi - lo)
    return lower, upper, 1.0 - alpha, alpha


def _sample_shifted(frame, dr, dc, base_coords):
    coords = np.array([base_coords[0] + dr, base_coords[1] + dc])
    return map_coordinates(frame, coords, order=1, mode="constant", cval=0.0, prefilter=False)


def _sample_many_shifted(frame, offsets, base_coords):
    """Sample a regular image grid at fixed subpixel offsets with zero padding."""
    frame = np.asarray(frame, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    if frame.ndim != 2 or offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("frame must be 2D and offsets must have shape [sample, row/column]")
    rows, cols = base_coords
    if rows.shape != frame.shape or cols.shape != frame.shape:
        raise ValueError("base coordinate grids must match the sampled frame")
    height, width = frame.shape
    result = np.zeros((len(offsets), height, width), dtype=np.float32)
    for index, (dr, dc) in enumerate(offsets):
        row0, col0 = int(np.floor(dr)), int(np.floor(dc))
        fr, fc = np.float32(dr - row0), np.float32(dc - col0)
        # scipy mode="constant" returns cval for any coordinate outside the
        # image. Restrict the output support first, then add the four bilinear
        # neighbors through contiguous slices instead of allocating index grids.
        r_valid0 = max(0, int(np.ceil(-dr)))
        r_valid1 = min(height - 1, int(np.floor(height - 1 - dr)))
        c_valid0 = max(0, int(np.ceil(-dc)))
        c_valid1 = min(width - 1, int(np.floor(width - 1 - dc)))
        if r_valid1 < r_valid0 or c_valid1 < c_valid0:
            continue
        for r_step, r_weight in ((0, 1.0 - fr), (1, fr)):
            source_dr = row0 + r_step
            r0 = max(r_valid0, -source_dr)
            r1 = min(r_valid1, height - 1 - source_dr)
            if r1 < r0:
                continue
            for c_step, c_weight in ((0, 1.0 - fc), (1, fc)):
                weight = np.float32(r_weight * c_weight)
                if weight == 0.0:
                    continue
                source_dc = col0 + c_step
                c0 = max(c_valid0, -source_dc)
                c1 = min(c_valid1, width - 1 - source_dc)
                if c1 < c0:
                    continue
                result[index, r0:r1 + 1, c0:c1 + 1] += (
                    weight * frame[r0 + source_dr:r1 + source_dr + 1,
                                   c0 + source_dc:c1 + source_dc + 1])
    return result


@lru_cache(maxsize=2)
def _shape_support(size):
    """Cache fixed triangle support offsets, orientation weights, and pixel grid."""
    templates = build_triangle_templates()
    offsets, angles, spans = [], [], {}
    for side, template in templates.items():
        edge = map_original_coordinates(template.edge_points_rc, size)
        corner = map_original_coordinates(template.corner_points_rc, size)
        center = map_original_coordinates(template.center_rc, size)
        start = len(offsets)
        offsets.extend(np.concatenate([edge, corner], axis=0) - center)
        angles.extend(np.concatenate([template.edge_angles, template.corner_angles]))
        spans[side] = {"edge": slice(start, start + len(edge)),
                       "corner": slice(start + len(edge), start + len(edge) + len(corner))}

    weight_rows = [_orientation_weights(float(angle)) for angle in angles]
    pairs = np.asarray([(row[0], row[1]) for row in weight_rows], dtype=np.int8)
    weights = np.asarray([(row[2], row[3]) for row in weight_rows], dtype=np.float32)
    rr, cc = np.mgrid[:size, :size].astype(np.float32)
    return np.asarray(offsets, dtype=np.float32), pairs, weights, spans, (rr, cc)


def _pool8(frame):
    n = frame.shape[0]
    ratio = n // config.POOL_SIZE
    return frame.reshape(config.POOL_SIZE, ratio, config.POOL_SIZE, ratio).mean(axis=(1, 3))


def _shape_maps(v_orient, size, eps=1e-8):
    """Build B/H maps from [orientation, y, x] Gabor energy at one time."""
    ratio = 256.0 / size
    rho = config.PIXEL_PARAMS["configuration_blur"] / ratio
    smoothed = np.stack([gaussian_filter(v, rho, mode="constant", cval=0.0) for v in v_orient])
    offsets, pairs, weights, spans, base_coords = _shape_support(size)
    bmap = np.sqrt(np.sum(v_orient * v_orient, axis=0)) / 2.0

    # Evaluate fixed supports by orientation pair. The sampler uses four
    # slice-based bilinear contributions per offset, avoiding coordinate-grid
    # allocations in every time step.
    sampled_response = np.empty((len(offsets), *bmap.shape), dtype=np.float32)
    for lo, hi in np.unique(pairs, axis=0):
        indices = np.flatnonzero((pairs[:, 0] == lo) & (pairs[:, 1] == hi))
        sample_lo = _sample_many_shifted(smoothed[lo], offsets[indices], base_coords)
        sample_hi = _sample_many_shifted(smoothed[hi], offsets[indices], base_coords)
        wl = weights[indices, 0, None, None]
        wh = weights[indices, 1, None, None]
        sampled_response[indices] = wl * sample_lo + wh * sample_hi
    log_response = np.log(np.maximum(sampled_response, 0.0) + eps)

    outputs = {}
    for side, span in spans.items():
        edge = log_response[span["edge"]].reshape(3, 3, *bmap.shape)
        edge_segment = np.mean(edge, axis=1)
        corner = log_response[span["corner"]].reshape(3, 2, *bmap.shape).mean(axis=1)
        # Equal log-domain weight to three edges and three corners.
        hmap = np.maximum(np.exp((edge_segment.sum(axis=0) + corner.sum(axis=0)) / 6.0) - eps, 0.0)
        outputs[side] = hmap.astype(np.float32)
    return bmap.astype(np.float32), outputs["left"], outputs["right"]


def _simulate_lgn(contrast, stage, time_ms, tau_a, dt_ms=1.0, include_offset=True):
    size = contrast.shape[0]
    ratio = 256.0 / size
    center = gaussian_filter(contrast, config.PIXEL_PARAMS["dog_center_sigma"] / ratio,
                             mode="constant", cval=0.0)
    surround = gaussian_filter(contrast, config.PIXEL_PARAMS["dog_surround_sigma"] / ratio,
                               mode="constant", cval=0.0)
    dog = center - surround
    relay = np.zeros((size, size, len(time_ms)), dtype=np.float32)
    adaptation = np.zeros((size, size), dtype=np.float32)
    tcr = np.zeros((2, size, size), dtype=np.float32)
    interneuron = np.zeros_like(tcr)
    trn = np.zeros_like(tcr)
    on_mean = np.zeros(len(time_ms), dtype=np.float32)
    off_mean = np.zeros_like(on_mean)
    on_pooled = np.zeros((8, 8, len(time_ms)), dtype=np.float32)
    off_pooled = np.zeros_like(on_pooled)
    for ti, time in enumerate(time_ms):
        gate = ((time >= 0) and (time < 200 if stage == "Stage1" and include_offset else True))
        signed = dog * float(gate)
        transient = signed - adaptation
        drive = np.stack([0.8 * np.maximum(transient, 0) + 0.2 * np.maximum(signed, 0),
                          0.8 * np.maximum(-transient, 0) + 0.2 * np.maximum(-signed, 0)])
        if ti:
            r_target = activation(2.2 * drive - 0.9 * interneuron - 0.8 * trn, 4.0, 0.5)
            n_target = activation(1.2 * drive + 0.7 * tcr, 4.0, 0.5)
            t_target = activation(0.9 * tcr, 4.0, 0.5)
            tcr += dt_ms * (r_target - tcr) / config.LGN_TAU_MS["tcr"]
            interneuron += dt_ms * (n_target - interneuron) / config.LGN_TAU_MS["in"]
            trn += dt_ms * (t_target - trn) / config.LGN_TAU_MS["trn"]
            adaptation += dt_ms * (signed - adaptation) / tau_a
        relay[:, :, ti] = tcr[0] - tcr[1]
        on_mean[ti] = tcr[0].mean()
        off_mean[ti] = tcr[1].mean()
        on_pooled[:, :, ti] = _pool8(tcr[0])
        off_pooled[:, :, ti] = _pool8(tcr[1])
    if (min(float(tcr.min()), float(interneuron.min()), float(trn.min())) < -1e-6
            or max(float(tcr.max()), float(interneuron.max()), float(trn.max())) > 1.000001):
        raise FloatingPointError("LGN TCR state left the [0,1] range")
    return relay, on_mean, off_mean, on_pooled, off_pooled


def simulate_frontend(stimulus, events=None, params=None, time_ms=None,
                      resolution=64, include_offset=True, remove_position=False,
                      capture_spatial_audit=False):
    """Return pooled early and configuration features for a standard stimulus."""
    if resolution not in (64, 128):
        raise ValueError("the forward model supports 64 or 128 spatial grids")
    params = {} if params is None else params
    tau_a = float(params.get("tau_a", config.TAU_ADAPT_DEFAULT))
    if tau_a <= 0:
        raise ValueError("tau_a must be positive")
    time_ms = config.TIME_MS if time_ms is None else np.asarray(time_ms, dtype=float)
    if time_ms.ndim != 1 or len(time_ms) < 2 or not np.all(np.diff(time_ms) > 0):
        raise ValueError("time_ms must be a strictly increasing one-dimensional axis")
    contrast = resize_area(stimulus.signed_contrast, resolution)
    relay, on_mean, off_mean, on_pooled, off_pooled = _simulate_lgn(
        contrast, stimulus.stage, time_ms, tau_a, dt_ms=float(np.median(np.diff(time_ms))),
        include_offset=include_offset)
    templates = build_triangle_templates()
    t_count = len(time_ms)
    gabor_small = np.zeros((4, 8, 8, t_count), dtype=np.float32)
    b_small = np.zeros((8, 8, t_count), dtype=np.float32)
    h_left = np.zeros_like(b_small)
    h_right = np.zeros_like(b_small)
    audit_times = (50.0, 100.0, 250.0, 400.0, 600.0)
    audit_indices = {int(np.argmin(np.abs(time_ms - t))): float(time_ms[np.argmin(np.abs(time_ms - t))])
                     for t in audit_times} if capture_spatial_audit else {}
    audit_maps = {}
    batch_size = 16 if resolution == 128 else 32
    kernels = [_gabor_pair(a, resolution) for a in config.GABOR_ANGLES_DEG]
    for start in range(0, t_count, batch_size):
        stop = min(t_count, start + batch_size)
        block = relay[:, :, start:stop]
        v_block = np.empty((4, resolution, resolution, stop - start), dtype=np.float32)
        for oi, (even, odd) in enumerate(kernels):
            e = fftconvolve(block, even[:, :, None], mode="same", axes=(0, 1))
            o = fftconvolve(block, odd[:, :, None], mode="same", axes=(0, 1))
            v_block[oi] = np.hypot(e, o).astype(np.float32, copy=False)
        for local_t in range(stop - start):
            v = v_block[:, :, :, local_t]
            if remove_position:
                v = np.broadcast_to(v.mean(axis=(1, 2), keepdims=True), v.shape)
            bmap, hl, hr = _shape_maps(v, resolution)
            gi = start + local_t
            if gi in audit_indices:
                audit_maps[audit_indices[gi]] = {"V1": v.copy(), "B": bmap.copy(),
                                                 "H_L": hl.copy(), "H_R": hr.copy()}
            for oi in range(4):
                gabor_small[oi, :, :, gi] = _pool8(v[oi])
            b_small[:, :, gi] = _pool8(bmap)
            h_left[:, :, gi] = _pool8(hl)
            h_right[:, :, gi] = _pool8(hr)
        del v_block
    # Config maps have equal maxima as a geometry scale convention, not as a
    # per-condition normalization. Shared normalization preserves contrasts.
    return {
        "time_ms": time_ms.copy(), "resolution": int(resolution),
        "lgn_on_mean": on_mean, "lgn_off_mean": off_mean,
        "lgn_total_mean": on_mean + off_mean, "lgn_on": on_pooled, "lgn_off": off_pooled,
        "gabor": gabor_small, "B": b_small, "H_L": h_left, "H_R": h_right,
        "stimulus": stimulus, "include_offset": bool(include_offset),
        "remove_position": bool(remove_position), "template_iou_left": templates["left"].mask_iou,
        "template_iou_right": templates["right"].mask_iou, "audit_maps": audit_maps,
    }


def mirror_stage1_frontend(left_frontend, right_stimulus):
    """Derive the right-cue maps by exact reflection of the left-cue maps.

    This symmetry shortcut is valid only for the standardized Stage1 pair,
    whose signed-contrast matrices are exact horizontal mirrors.
    """
    source = left_frontend.get("stimulus")
    if (source is None or source.stage != "Stage1" or source.condition != "left"
            or right_stimulus.stage != "Stage1" or right_stimulus.condition != "right"):
        raise ValueError("mirror_stage1_frontend requires the standardized Stage1 left/right pair")
    if not np.array_equal(right_stimulus.signed_contrast, source.signed_contrast[:, ::-1]):
        raise ValueError("Stage1 right contrast is not the exact mirror of left contrast")

    result = dict(left_frontend)
    orientation_mirror = np.array([0, 3, 2, 1])  # 0,45,90,135 degrees under x reflection
    result["stimulus"] = right_stimulus
    result["gabor"] = left_frontend["gabor"][orientation_mirror, :, ::-1, :].copy()
    for key in ("lgn_on", "lgn_off", "B"):
        result[key] = left_frontend[key][:, ::-1, :].copy()
    result["H_L"] = left_frontend["H_R"][:, ::-1, :].copy()
    result["H_R"] = left_frontend["H_L"][:, ::-1, :].copy()
    result["lgn_on_mean"] = left_frontend["lgn_on_mean"].copy()
    result["lgn_off_mean"] = left_frontend["lgn_off_mean"].copy()
    result["lgn_total_mean"] = left_frontend["lgn_total_mean"].copy()
    result["template_iou_left"] = left_frontend["template_iou_right"]
    result["template_iou_right"] = left_frontend["template_iou_left"]
    result["audit_maps"] = {
        time: {"V1": maps["V1"][orientation_mirror, :, ::-1].copy(),
               "B": maps["B"][:, ::-1].copy(),
               "H_L": maps["H_R"][:, ::-1].copy(),
               "H_R": maps["H_L"][:, ::-1].copy()}
        for time, maps in left_frontend["audit_maps"].items()
    }
    return result


def _kernel_energy_moment(kernel, spacing):
    kernel = np.asarray(kernel)
    energy = np.abs(kernel.astype(np.complex128)) ** 2
    energy /= energy.sum()
    rr, cc = np.mgrid[:kernel.shape[0], :kernel.shape[1]].astype(float)
    xy = np.column_stack([cc.ravel() * spacing, rr.ravel() * spacing])
    weights = energy.ravel()
    mu = weights @ xy
    centered = xy - mu
    return (centered * weights[:, None]).T @ centered


def _audit_kernels(size):
    ratio = 256.0 / size
    sigc = config.PIXEL_PARAMS["dog_center_sigma"] / ratio
    sigs = config.PIXEL_PARAMS["dog_surround_sigma"] / ratio
    radius = int(np.ceil(4 * sigs))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    gc, gs = _normal_pdf(x, sigc) * _normal_pdf(y, sigc), _normal_pdf(x, sigs) * _normal_pdf(y, sigs)
    dog = gc - gs
    even, odd = _gabor_pair(0.0, size)
    return {"dog": dog, "gabor": even.astype(np.complex64) + 1j * odd}


def scale_kernel_audit():
    """Compare discrete-kernel energy moments and sampled Gabor wavelength."""
    rows = []
    reference = _audit_kernels(256)
    ref_moments = {name: _kernel_energy_moment(k, 1.0) for name, k in reference.items()}
    for size in (64, 128):
        kernels = _audit_kernels(size)
        errors = {}
        for name, kernel in kernels.items():
            moment = _kernel_energy_moment(kernel, 256.0 / size)
            errors[name] = float(np.linalg.norm(moment - ref_moments[name], "fro") /
                                 max(np.linalg.norm(ref_moments[name], "fro"), 1e-12))
        # Peak radial frequency of the sampled even/odd quadrature energy.
        ge = kernels["gabor"]
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(ge, s=(512, 512))))
        cy, cx = np.array(spectrum.shape) // 2
        fy = np.fft.fftshift(np.fft.fftfreq(spectrum.shape[0], d=256.0 / size))
        fx = np.fft.fftshift(np.fft.fftfreq(spectrum.shape[1], d=256.0 / size))
        yy, xx = np.meshgrid(fy, fx, indexing="ij")
        radial = np.sqrt(xx * xx + yy * yy)
        keep = radial > 0
        peak = float(radial[keep][np.argmax(spectrum[keep])])
        wavelength = float(1.0 / peak) if peak else float("inf")
        rows.append({"resolution": size, "dog_moment_relative_error": errors["dog"],
                     "gabor_moment_relative_error": errors["gabor"],
                     "gabor_peak_wavelength_px": wavelength,
                     "gabor_wavelength_relative_error": abs(wavelength - 10.0) / 10.0,
                     "kernel_pass": errors["dog"] <= .10 and errors["gabor"] <= .10
                                   and abs(wavelength - 10.0) / 10.0 <= .10})
    return rows


def scale_feature_audit(tau_a=80.0):
    """Compare 128-grid V1/config maps against independent 64-grid maps."""
    summaries = []
    left_stimulus = load_stimulus("Stage1", "left")
    low_left = simulate_frontend(left_stimulus, params={"tau_a": tau_a}, resolution=64,
                                 capture_spatial_audit=True)
    high_left = simulate_frontend(left_stimulus, params={"tau_a": tau_a}, resolution=128,
                                  capture_spatial_audit=True)
    fronts = {
        "left": (low_left, high_left),
        "right": (mirror_stage1_frontend(low_left, load_stimulus("Stage1", "right")),
                  mirror_stage1_frontend(high_left, load_stimulus("Stage1", "right"))),
    }
    for condition in ("left", "right"):
        low, high = fronts[condition]
        similarities, zero_count, zero_mismatch = [], 0, 0
        names = ("V1", "B", "H_L", "H_R")
        for time in low["audit_maps"]:
            for name in names:
                a, b = low["audit_maps"][time][name], high["audit_maps"][time][name]
                if name == "V1":
                    pairs = [(a[i], resize_area(b[i], 64)) for i in range(4)]
                else:
                    pairs = [(a, resize_area(b, 64))]
                for x, y in pairs:
                    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
                    if nx <= 1e-12 or ny <= 1e-12:
                        zero_count += 1
                        if (nx <= 1e-12) != (ny <= 1e-12):
                            zero_mismatch += 1
                        continue
                    similarities.append(float(np.sum(x * y) / (nx * ny)))
        min_cos = min(similarities) if similarities else float("nan")
        summaries.append({"condition": condition, "n_nonzero_pairs": len(similarities),
                          "n_zero_pairs": zero_count, "n_zero_mismatch": zero_mismatch,
                          "minimum_cosine": min_cos,
                          "mean_cosine": float(np.mean(similarities)) if similarities else float("nan"),
                          "features_pass": bool(similarities and min_cos >= 0.95 and zero_mismatch == 0)})
    return summaries


def choose_resolution():
    """Freeze 64 when all preset checks pass; otherwise use 128 before EEG fitting."""
    kernel_rows = scale_kernel_audit()
    feature_rows = scale_feature_audit()
    by_size = {row["resolution"]: row for row in kernel_rows}
    pass64 = bool(by_size[64]["kernel_pass"] and all(x["features_pass"] for x in feature_rows))
    # The cross-resolution feature discrepancy is precisely what selects 128;
    # only the 128-vs-256 kernel audit must still pass at the chosen scale.
    pass128 = bool(by_size[128]["kernel_pass"])
    chosen = 64 if pass64 else 128
    return {"chosen_resolution": chosen, "resolution_pass": pass64 or pass128,
            "kernel_audit": kernel_rows, "feature_audit": feature_rows,
            "selection_rule": "64 if all 64-grid tests pass; otherwise freeze 128 before EEG fitting"}


def template_manifest():
    """Serializable fixed triangle geometry and support definitions."""
    result = {}
    for name, template in build_triangle_templates().items():
        result[name] = {"vertices_rc_256": template.vertices_rc,
                        "center_rc_256": template.center_rc,
                        "edge_points_rc_256": template.edge_points_rc,
                        "edge_normal_deg": template.edge_angles,
                        "corner_points_rc_256": template.corner_points_rc,
                        "corner_normal_deg": template.corner_angles,
                        "mask_hull_iou": template.mask_iou}
    return result
