# 问题二核心算法：视觉前端提供 B、shape、dir_left、dir_right 特征。
# 以下保留皮层群体动力学、功能源映射和头皮电极正向投影。

W = {"w_ee": 1.4, "w_ei": 1.1, "w_ie": 1.0, "w_ii": 0.8,
     "g_p": 2.2, "g_q": 1.2, "tau_e": 18.0, "tau_i": 10.0,
     "tau_i_ratio": 0.55, "delay_early": 8.0, "delay_shape": 33.0,
     "feed_e": 0.9, "feed_i": 0.45}
SIGMA, RADIUS = 0.33, 0.09
ELECTRODES = np.array([[-35, 55, 62], [0, 66, 61], [35, 55, 62]], float)
REFERENCES = np.array([[-70, -50, 5], [70, -50, 5]], float)
SOURCES = np.array([[-10, -58, 45], [10, -58, 45], [-36, -25, 50],
                    [36, -25, 50], [0, -34, 56]], float)


def activation(x, beta, threshold):
    z = np.clip(beta * (np.asarray(x) - threshold), -60, 60)
    f0 = 1 / (1 + np.exp(beta * threshold))
    return np.clip((1 / (1 + np.exp(-z)) - f0) / (1 - f0), 0, 1)


def project_features(frontend, scales=None):
    # 空间图按左右视野平均；两个模板偏好通道保持分离。
    def halves(x):
        mid = x.shape[1] // 2
        return np.stack([x[:, :mid].mean((0, 1)), x[:, mid:].mean((0, 1))]) / np.sqrt(2)
    drives = np.stack([halves(frontend["B"]), halves(frontend["shape"]),
                       [frontend["dir_left"], frontend["dir_right"]]]).astype(np.float32)
    return drives if scales is None else drives / np.asarray(scales)[:, :, None]


def delay_signal(x, time_ms, delay_ms):
    target = time_ms - delay_ms
    rows = np.asarray(x).reshape(-1, len(time_ms))
    return np.stack([np.interp(target, time_ms, row, left=0, right=0)
                     for row in rows]).reshape(x.shape)


def history_at(x, step, delay_samples):
    # 只从已计算状态取样，并对分数延迟作线性插值。
    whole = int(np.floor(delay_samples))
    frac, index = delay_samples - whole, step - whole
    if index < 0 or (index == 0 and frac > 1e-12):
        return np.zeros(x.shape[:-1], dtype=x.dtype)
    current = x[..., index]
    previous = x[..., index - 1] if index else np.zeros_like(current)
    return (1 - frac) * current + frac * previous


def wilson_cowan(drives, time_ms, tau_s=40.0, g_i=1.0):
    # 三群体分别表示早期对比、空间构型和三角模板偏好。
    dt = np.median(np.diff(time_ms))
    drive = np.empty_like(drives, dtype=np.float32)
    drive[0] = delay_signal(drives[0], time_ms, W["delay_early"])
    drive[1:] = delay_signal(drives[1:], time_ms, W["delay_shape"])
    e, inh = np.zeros_like(drive), np.zeros_like(drive)
    tau_e = np.array([W["tau_e"], tau_s, tau_s])[:, None]
    tau_i = tau_e * np.array([W["tau_i"] / W["tau_e"], W["tau_i_ratio"], W["tau_i_ratio"]])[:, None]
    for t in range(len(time_ms) - 1):
        e_in = W["w_ee"] * e[:, :, t] - g_i * W["w_ei"] * inh[:, :, t] + W["g_p"] * drive[:, :, t]
        i_in = W["w_ie"] * e[:, :, t] - W["w_ii"] * inh[:, :, t] + W["g_q"] * drive[:, :, t]
        ef, inf = history_at(e[0], t, W["delay_shape"] / dt), history_at(inh[0], t, W["delay_shape"] / dt)
        e_in[1] += W["feed_e"] * ef
        e_in[2] += W["feed_e"] * ef.mean()
        i_in[1] += W["feed_i"] * inf
        i_in[2] += W["feed_i"] * inf.mean()
        fe, fi = activation(e_in, 5, 0.35), activation(i_in, 5, 0.35)
        e[:, :, t + 1] = e[:, :, t] + dt * (-e[:, :, t] + (1 - e[:, :, t]) * fe) / tau_e
        inh[:, :, t + 1] = inh[:, :, t] + dt * (-inh[:, :, t] + (1 - inh[:, :, t]) * fi) / tau_i
    return e, inh


def source_currents(e, inh, time_ms):
    # 兴奋和抑制活动经突触核滤波，再映射为四个半球源和一个中线对手源。
    dt = np.median(np.diff(time_ms))
    def kernel(tau):
        t = np.arange(len(time_ms)) * dt
        h = t * np.exp(-t / tau) / tau**2
        return h / (h.sum() * dt)
    pe = fftconvolve(e, kernel(10)[None, None, :], mode="full", axes=-1)[..., :len(time_ms)] * dt
    pi = fftconvolve(inh, kernel(20)[None, None, :], mode="full", axes=-1)[..., :len(time_ms)] * dt
    early, shape, preference = pe - pi
    return np.stack([early[1], early[0], shape[1], shape[0],
                     preference[0] - preference[1]])


def leadfield():
    # 点偶极均匀导体近似，电极位于头表并使用双侧乳突参考。
    surface = lambda p: RADIUS * p / np.linalg.norm(p, axis=1, keepdims=True)
    electrodes, references, sources = surface(ELECTRODES) , surface(REFERENCES), SOURCES / 1000
    normals = sources / np.linalg.norm(sources, axis=1, keepdims=True)
    def potential(points):
        delta = points[:, None] - sources[None]
        distance = np.linalg.norm(delta, axis=-1)
        return np.einsum("sc,esc->es", normals, delta) / (4 * np.pi * SIGMA * distance**3)
    return potential(electrodes) - potential(references).mean(axis=0, keepdims=True)


def simulate_q2(frontend, time_ms, tau_s=40.0, g_i=1.0, scales=None):
    e, inh = wilson_cowan(project_features(frontend, scales), time_ms, tau_s, g_i)
    sources = source_currents(e, inh, time_ms)
    return leadfield() @ sources
