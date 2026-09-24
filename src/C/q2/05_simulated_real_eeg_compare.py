import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).resolve().parent
REAL_ROOT = SCRIPT_DIR.parent / "q1" / "output" / "8riemann_denoise"
SIM_ROOT = SCRIPT_DIR / "output" / "04_simulated_eeg"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "05_eeg_compare"

DATASETS = {
    "Task1": ["VisualCogA_Task-1", "VisualCogA_Task-2"],
    "Task2": ["VisualCogB_Task-1", "VisualCogB_Task-2"],
}

CHANNELS = ["F3", "Fz", "F4"]
CHANNEL_INDEX = [1, 0, 2]

TARGET_ONSET = 2.20
MODEL_TIME_MS = np.arange(801, dtype=np.float32)
COMPARE_WINDOW = (0, 500)
LATE_WINDOW = (250, 500)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def baseline_correct(eeg, time, window):
    mask = (time >= window[0]) & (time < window[1])
    baseline = np.nanmean(eeg[:, :, mask], axis=2, keepdims=True)
    return eeg - baseline


def infer_layout(cue_type, action_value):
    product = cue_type * action_value
    result = np.full(len(product), "", dtype=object)
    result[product < 0] = "inward"
    result[product > 0] = "outward"
    return result


def channel_zscore(signal):
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    return (signal - mean) / (std + 1e-12)


def correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cosine_similarity(a, b):
    return float(
        np.dot(a, b)
        / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )


def late_peak(signal, time_ms):
    mask = (time_ms >= LATE_WINDOW[0]) & (time_ms <= LATE_WINDOW[1])
    values = signal[mask]
    times = time_ms[mask]
    index = int(np.argmax(values))
    return float(values[index]), float(times[index])


def load_real_task(task):
    eeg_list = []
    cue_list = []
    action_list = []

    for dataset in DATASETS[task]:
        mat = loadmat(REAL_ROOT / f"{dataset}_clean.mat")
        trials = np.asarray(mat["trial_data"], dtype=float)
        relative_time = np.asarray(mat["relative_time"], dtype=float)
        cue_type = np.asarray(mat["cue_type"]).reshape(-1)

        time = relative_time[0]
        eeg_list.append(trials[:, CHANNEL_INDEX, :])
        cue_list.append(cue_type)

        action = trials[:, 8, :]
        post_target = action[:, time >= TARGET_ONSET]
        index = np.argmax(np.abs(post_target), axis=1)
        action_value = post_target[np.arange(len(post_target)), index]
        action_value[np.max(np.abs(post_target), axis=1) == 0] = 0
        action_list.append(action_value)

    return (
        np.concatenate(eeg_list, axis=0),
        np.concatenate(cue_list),
        np.concatenate(action_list),
        time,
    )


def interpolate_erp(erp, real_time_s, onset_s):
    relative_ms = (real_time_s - onset_s) * 1000.0
    output = np.zeros((3, len(MODEL_TIME_MS)), dtype=np.float32)

    for channel in range(3):
        output[channel] = np.interp(
            MODEL_TIME_MS,
            relative_ms,
            erp[channel],
        )

    return output


def real_stage_conditions(task, stage):
    eeg, cue_type, action_value, time = load_real_task(task)

    if stage == "Stage1":
        eeg = baseline_correct(eeg, time, (-0.2, 0.0))
        conditions = {
            "left": cue_type == -1,
            "right": cue_type == 1,
        }
        onset = 0.0

    elif task == "Task1":
        eeg = baseline_correct(eeg, time, (2.0, TARGET_ONSET))
        conditions = {
            "dots": np.ones(len(eeg), dtype=bool),
        }
        onset = TARGET_ONSET

    else:
        eeg = baseline_correct(eeg, time, (2.0, TARGET_ONSET))
        layout = infer_layout(cue_type, action_value)
        conditions = {
            "inward": layout == "inward",
            "outward": layout == "outward",
        }
        onset = TARGET_ONSET

    result = {}

    for name, mask in conditions.items():
        erp = np.nanmean(eeg[mask], axis=0)
        result[name] = {
            "erp": interpolate_erp(erp, time, onset),
            "n_trials": int(mask.sum()),
        }

    return result


def compare_condition(sim, real, n_trials):
    compare_mask = (
        (MODEL_TIME_MS >= COMPARE_WINDOW[0])
        & (MODEL_TIME_MS <= COMPARE_WINDOW[1])
    )
    late_mask = (
        (MODEL_TIME_MS >= LATE_WINDOW[0])
        & (MODEL_TIME_MS <= LATE_WINDOW[1])
    )

    row = {
        "比较类型": "单条件",
        "条件": "",
        "真实Trial数": n_trials,
    }

    correlations = []

    for i, channel in enumerate(CHANNELS):
        corr = correlation(
            sim[i, compare_mask],
            real[i, compare_mask],
        )
        correlations.append(corr)

        sim_peak, sim_time = late_peak(sim[i], MODEL_TIME_MS)
        real_peak, real_time = late_peak(real[i], MODEL_TIME_MS)

        row[f"{channel}波形相关"] = corr
        row[f"{channel}模拟晚期峰值"] = sim_peak
        row[f"{channel}真实晚期峰值"] = real_peak
        row[f"{channel}模拟峰时间(ms)"] = sim_time
        row[f"{channel}真实峰时间(ms)"] = real_time
        row[f"{channel}峰时间差(ms)"] = abs(sim_time - real_time)

    sim_spatial = sim[:, late_mask].mean(axis=1)
    real_spatial = real[:, late_mask].mean(axis=1)

    row["平均波形相关"] = float(np.mean(correlations))
    row["晚期空间余弦相似度"] = cosine_similarity(sim_spatial, real_spatial)
    row["模拟晚期F4-F3均值"] = float((sim[2] - sim[0])[late_mask].mean())
    row["真实晚期F4-F3均值"] = float((real[2] - real[0])[late_mask].mean())

    return row


def condition_difference_row(label, sim_a, sim_b, real_a, real_b):
    compare_mask = (
        (MODEL_TIME_MS >= COMPARE_WINDOW[0])
        & (MODEL_TIME_MS <= COMPARE_WINDOW[1])
    )

    sim_diff = sim_a - sim_b
    real_diff = real_a - real_b

    row = {
        "比较类型": "条件差异",
        "条件": label,
        "真实Trial数": "",
    }

    correlations = []

    for i, channel in enumerate(CHANNELS):
        corr = correlation(
            sim_diff[i, compare_mask],
            real_diff[i, compare_mask],
        )
        correlations.append(corr)
        row[f"{channel}波形相关"] = corr

    row["平均波形相关"] = float(np.mean(correlations))

    return row


def save_csv(output_dir, rows):
    fields = [
        "比较类型",
        "条件",
        "真实Trial数",
        "F3波形相关",
        "Fz波形相关",
        "F4波形相关",
        "平均波形相关",
        "F3模拟晚期峰值",
        "F3真实晚期峰值",
        "F3模拟峰时间(ms)",
        "F3真实峰时间(ms)",
        "F3峰时间差(ms)",
        "Fz模拟晚期峰值",
        "Fz真实晚期峰值",
        "Fz模拟峰时间(ms)",
        "Fz真实峰时间(ms)",
        "Fz峰时间差(ms)",
        "F4模拟晚期峰值",
        "F4真实晚期峰值",
        "F4模拟峰时间(ms)",
        "F4真实峰时间(ms)",
        "F4峰时间差(ms)",
        "晚期空间余弦相似度",
        "模拟晚期F4-F3均值",
        "真实晚期F4-F3均值",
    ]

    with (output_dir / "模拟真实EEG对比.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_compare(output_dir, title, cases):
    fig, axes = plt.subplots(
        len(cases),
        3,
        figsize=(14, 3.6 * len(cases)),
        squeeze=False,
        sharex=True,
    )

    for row, (label, sim, real) in enumerate(cases):
        sim_z = channel_zscore(sim)
        real_z = channel_zscore(real)

        for col, channel in enumerate(CHANNELS):
            axis = axes[row, col]
            axis.plot(MODEL_TIME_MS, sim_z[col], label="模拟")
            axis.plot(MODEL_TIME_MS, real_z[col], label="真实ERP")
            axis.axvspan(250, 500, alpha=0.08)
            axis.axhline(0, linewidth=0.8)
            axis.set_title(f"{label} - {channel}")
            axis.set_xlabel("相对事件时间 (ms)")
            axis.set_ylabel("标准化波形")
            axis.grid(alpha=0.25)

            if row == 0 and col == 0:
                axis.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(
        output_dir / "模拟真实EEG对比.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def process_stage(task, stage, stimuli, title, difference_pair=None):
    output_dir = OUTPUT_ROOT / task / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    real_conditions = real_stage_conditions(task, stage)

    rows = []
    cases = []
    data = {}

    for name, label in stimuli:
        sim = np.load(
            SIM_ROOT / task / stage / f"{name}_eeg.npy"
        ).astype(float)

        real = real_conditions[name]["erp"]
        n_trials = real_conditions[name]["n_trials"]

        row = compare_condition(sim, real, n_trials)
        row["条件"] = label
        rows.append(row)

        cases.append((label, sim, real))
        data[name] = (sim, real)

    if difference_pair is not None:
        name_a, name_b, label = difference_pair
        sim_a, real_a = data[name_a]
        sim_b, real_b = data[name_b]

        rows.append(
            condition_difference_row(
                label,
                sim_a,
                sim_b,
                real_a,
                real_b,
            )
        )

    save_csv(output_dir, rows)
    plot_compare(output_dir, title, cases)


def main():
    process_stage(
        "Task1",
        "Stage1",
        [("left", "左提示"), ("right", "右提示")],
        "Task1 Stage1 模拟与真实 EEG 对比",
        ("left", "right", "左提示-右提示"),
    )

    process_stage(
        "Task1",
        "Stage2",
        [("dots", "双圆点目标")],
        "Task1 Stage2 模拟与真实 EEG 对比",
    )

    process_stage(
        "Task2",
        "Stage1",
        [("left", "左提示"), ("right", "右提示")],
        "Task2 Stage1 模拟与真实 EEG 对比",
        ("left", "right", "左提示-右提示"),
    )

    process_stage(
        "Task2",
        "Stage2",
        [("inward", "相向"), ("outward", "背向")],
        "Task2 Stage2 模拟与真实 EEG 对比",
        ("inward", "outward", "相向-背向"),
    )

    print("模拟与真实 EEG 对比完成：", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
