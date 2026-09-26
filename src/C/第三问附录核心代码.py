

# Q3 动态认知状态模型的名义参数
PARAMS = {
    "tau_visual": 0.060,
    "tau_memory": 1.0,
    "tau_control": 0.250,
    "cue_storage_gain": 0.8,
    "target_memory_gain": 0.8,
    "memory_to_control_gain": 0.25,
    "conflict_gain": 0.8,
    "topdown_control_gain": 0.10,
}

def make_event_gates(time_s, cue_onset_s=0.0, cue_duration_s=0.203125,
                     target_onset_s=2.2, target_duration_s=0.2):
    """根据事件时刻生成提示与目标的阶跃门控信号。"""
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    cue = ((time_s >= cue_onset_s) &
           (time_s < cue_onset_s + cue_duration_s)).astype(float)
    target = np.zeros_like(cue)
    if target_onset_s is not None:
        if target_duration_s is None:
            target = (time_s >= target_onset_s).astype(float)
        else:
            target = ((time_s >= target_onset_s) &
                      (time_s < target_onset_s + target_duration_s)).astype(float)
    return cue, target

def integrate_macro_states(visual_drive, cue_gate, target_gate, match_evidence,
                           dt_s, params=PARAMS):
    """用指数欧拉递推计算视觉、记忆和控制状态。"""
    drive = np.asarray(visual_drive, dtype=float).reshape(-1)
    cue = np.asarray(cue_gate, dtype=float).reshape(-1)
    target = np.asarray(target_gate, dtype=float).reshape(-1)
    rho_v, rho_h, rho_p = [
        np.exp(-dt_s / params[key])
        for key in ("tau_visual", "tau_memory", "tau_control")
    ]
    visual = np.zeros_like(drive)
    memory = np.zeros_like(drive)
    control = np.zeros_like(drive)
    visual[0] = drive[0]
    for k in range(1, drive.size):
        visual[k] = rho_v * visual[k - 1] + (1 - rho_v) * drive[k]
        memory_input = (
            params["cue_storage_gain"] * visual[k] * cue[k]
            + params["target_memory_gain"] * visual[k] * target[k]
            * match_evidence
        )
        memory[k] = rho_h * memory[k - 1] + (1 - rho_h) * memory_input
        conflict = target[k] * (1 - match_evidence) / 2
        control_input = (
            params["memory_to_control_gain"] * abs(memory[k - 1]) * target[k]
            + params["conflict_gain"] * conflict
        )
        control[k] = rho_p * control[k - 1] + (1 - rho_p) * control_input
    return {"visual": visual, "memory": memory, "control": control}

def observe_macro_model(q2_sensor, memory, control,
                        memory_loading, control_loading, params=PARAMS):
    """将 Q2 视觉响应及记忆、控制状态映射到 F3、Fz、F4。"""
    q2 = np.asarray(q2_sensor, dtype=float)
    h = np.asarray(memory, dtype=float).reshape(-1)
    p = np.asarray(control, dtype=float).reshape(-1)
    l_h = np.asarray(memory_loading, dtype=float).reshape(-1)
    l_p = np.asarray(control_loading, dtype=float).reshape(-1)
    return (q2 * (1 + params["topdown_control_gain"] * p)[None, :]
            + l_h[:, None] * h[None, :]
            + l_p[:, None] * p[None, :])

def _features_for_channel(q2_channel, memory, control, interaction):
    """按通道构造视觉、记忆、控制及视觉反馈特征。"""
    return np.column_stack((
        q2_channel.reshape(-1),
        memory.reshape(-1),
        control.reshape(-1),
        interaction.reshape(-1),
    ))

def fit_sensor_mapping(observed, q2_sensor, memory, control,
                       q2_control_interaction, ridge_alpha=1.0):
    """仅用训练记录拟合各 EEG 通道的岭回归观测映射。"""
    y = np.asarray(observed, dtype=float)
    q2 = np.asarray(q2_sensor, dtype=float)
    h = np.asarray(memory, dtype=float)
    p = np.asarray(control, dtype=float)
    interaction = np.asarray(q2_control_interaction, dtype=float)
    fits = []
    for channel in range(3):
        x = _features_for_channel(
            q2[:, channel], h, p, interaction[:, channel]
        )
        target = y[:, channel].reshape(-1)
        mean_x = x.mean(axis=0)
        scale_x = x.std(axis=0)
        scale_x[scale_x < 1e-12] = 1.0
        z = (x - mean_x) / scale_x
        mean_y = target.mean()
        beta = np.linalg.solve(
            z.T @ z + ridge_alpha * np.eye(z.shape[1]),
            z.T @ (target - mean_y),
        )
        fits.append((mean_x, scale_x, mean_y, beta))
    return fits

def predict_sensor_mapping(fits, q2_sensor, memory, control,
                           q2_control_interaction):
    """将训练阶段估计的映射应用于留出记录。"""
    q2 = np.asarray(q2_sensor, dtype=float)
    h = np.asarray(memory, dtype=float)
    p = np.asarray(control, dtype=float)
    interaction = np.asarray(q2_control_interaction, dtype=float)
    prediction = np.empty_like(q2)
    for channel, (mean_x, scale_x, mean_y, beta) in enumerate(fits):
        x = _features_for_channel(
            q2[:, channel], h, p, interaction[:, channel]
        )
        prediction[:, channel] = (
            mean_y + ((x - mean_x) / scale_x) @ beta
        ).reshape(q2.shape[0], q2.shape[2])
    return prediction

def score_prediction(observed, predicted, time_s, window):
    """计算指定时间窗内的 RMSE、标准化 RMSE 与相关系数。"""
    mask = (time_s >= window[0]) & (time_s < window[1])
    actual = np.asarray(observed)[..., mask].reshape(-1)
    estimate = np.asarray(predicted)[..., mask].reshape(-1)
    rmse = np.sqrt(np.mean((actual - estimate) ** 2))
    scale = np.std(actual)
    nrmse = rmse / scale if scale > 1e-12 else np.nan
    corr = (np.corrcoef(actual, estimate)[0, 1]
            if np.std(actual) > 1e-12 and np.std(estimate) > 1e-12
            else np.nan)
    return {"RMSE": rmse, "NRMSE": nrmse, "相关系数": corr}

def main():
    """依次执行事件审计、Q2场景模拟、Q3留记录验证与结果汇总。"""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path: sys.path.insert(0, root)
    event_audit = importlib.import_module("src.C.q3.09_event_time_semantics")
    validation = importlib.import_module("src.C.q3.dynamic_validation")
    event_audit.main()
    return validation.run_validation()

if __name__ == "__main__":
    main()
