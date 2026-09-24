import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
GABOR_ROOT = SCRIPT_DIR / "output" / "01_gabor_frontend"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "02_lgn"

ANGLES = (0, 45, 90, 135)
GRID = 4
DT = 1.0
T_MAX = 800.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def spatial_mean(image):
    height, width = image.shape
    return np.array([
        image[row * height // GRID:(row + 1) * height // GRID,
              col * width // GRID:(col + 1) * width // GRID].mean()
        for row in range(GRID)
        for col in range(GRID)
    ], dtype=np.float32)


def gabor_features(responses):
    return np.concatenate([spatial_mean(response) for response in responses])


def delta_features(signed_delta):
    # 暗色目标出现时 delta 为负；LGN 驱动使用其正的对比幅值。
    return spatial_mean(np.clip(-signed_delta, 0, None))


def normalize_pair(first, second):
    scale = max(float(first.max()), float(second.max()), 1e-12)
    return first / scale, second / scale


def normalize_single(features):
    return features / max(float(features.max()), 1e-12)


def activation(value):
    sigmoid = 1.0 / (1.0 + np.exp(-4.0 * (value - 0.5)))
    zero = 1.0 / (1.0 + np.exp(2.0))
    response = np.clip((sigmoid - zero) / (1.0 - zero), 0.0, 1.0)
    return np.where(value <= 0.0, 0.0, response)


def alpha_pulse(time_ms, event_ms, delay_ms, tau_ms):
    elapsed = time_ms - event_ms - delay_ms
    pulse = np.zeros_like(time_ms, dtype=np.float32)
    active = elapsed >= 0
    scaled_time = elapsed[active] / tau_ms
    pulse[active] = scaled_time * np.exp(1.0 - scaled_time)
    return pulse


def temporal_inputs(time_ms, cue_duration_ms=None):
    # 暗刺激出现产生 OFF onset；提示消失后亮度恢复产生延迟的 ON onset。
    off = alpha_pulse(time_ms, 0.0, 0.0, 20.0)
    on = np.zeros_like(time_ms, dtype=np.float32)
    if cue_duration_ms is not None:
        on = alpha_pulse(time_ms, cue_duration_ms, 10.0, 25.0)
    return on, off


def run_population(features, drive_wave):
    channels = len(features)
    steps = len(drive_wave)
    tcr = np.zeros((channels, steps), dtype=np.float32)
    interneuron = np.zeros_like(tcr)
    trn = np.zeros_like(tcr)

    for step in range(steps - 1):
        drive = features * drive_wave[step]
        tcr_target = activation(
            2.2 * drive - 0.9 * interneuron[:, step] - 0.8 * trn[:, step]
        )
        in_target = activation(1.2 * drive + 0.7 * tcr[:, step])
        trn_target = activation(0.9 * tcr[:, step])

        tcr[:, step + 1] = tcr[:, step] + DT * (-tcr[:, step] + tcr_target) / 18.0
        interneuron[:, step + 1] = (
            interneuron[:, step] + DT * (-interneuron[:, step] + in_target) / 12.0
        )
        trn[:, step + 1] = trn[:, step] + DT * (-trn[:, step] + trn_target) / 25.0

    return tcr, interneuron, trn


def simulate_lgn(features, cue_duration_ms=None):
    time_ms = np.arange(0, T_MAX + DT, DT, dtype=np.float32)
    on_wave, off_wave = temporal_inputs(time_ms, cue_duration_ms)
    tcr_on, in_on, trn_on = run_population(features, on_wave)
    tcr_off, in_off, trn_off = run_population(features, off_wave)

    return {
        "time": time_ms,
        "tcr_on": tcr_on,
        "tcr_off": tcr_off,
        "tcr_total": tcr_on + tcr_off,
        "in_on": in_on,
        "in_off": in_off,
        "trn_on": trn_on,
        "trn_off": trn_off,
    }


def save_result(output_dir, name, result, features):
    np.save(output_dir / f"{name}_feature.npy", features)
    np.save(output_dir / f"{name}_lgn_on.npy", result["tcr_on"])
    np.save(output_dir / f"{name}_lgn_off.npy", result["tcr_off"])
    # TCR 总响应作为后续 cortex 的输入。
    np.save(output_dir / f"{name}_lgn_total.npy", result["tcr_total"])


def summary_row(label, result):
    on_mean = result["tcr_on"].mean(axis=0)
    off_mean = result["tcr_off"].mean(axis=0)
    total_mean = result["tcr_total"].mean(axis=0)
    time_ms = result["time"]
    return {
        "条件": label,
        "OFF峰值": float(off_mean.max()),
        "OFF峰值时间(ms)": float(time_ms[np.argmax(off_mean)]),
        "ON峰值": float(on_mean.max()),
        "ON峰值时间(ms)": float(time_ms[np.argmax(on_mean)]),
        "总响应峰值": float(total_mean.max()),
        "总响应峰值时间(ms)": float(time_ms[np.argmax(total_mean)]),
    }


def save_summary(output_dir, rows):
    path = output_dir / "LGN响应汇总.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_response(output_dir, title, cases):
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4), squeeze=False)
    for axis, (label, result) in zip(axes[0], cases):
        time_ms = result["time"]
        axis.plot(time_ms, result["tcr_off"].mean(axis=0), label="OFF")
        axis.plot(time_ms, result["tcr_on"].mean(axis=0), label="ON")
        axis.plot(time_ms, result["tcr_total"].mean(axis=0), label="TCR总响应")
        axis.set_title(label)
        axis.set_xlabel("时间 (ms)")
        axis.set_ylabel("群体响应")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "LGN平均响应.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def peak_map(result):
    peak = result["tcr_total"].max(axis=1)
    if len(peak) == len(ANGLES) * GRID * GRID:
        return peak.reshape(len(ANGLES), GRID, GRID).mean(axis=0)
    return peak.reshape(GRID, GRID)


def plot_peak_map(output_dir, cases):
    maps = [peak_map(result) for _, result in cases]
    if len(cases) == 1:
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(maps[0])
        axis.set_title(cases[0][0])
        for row in range(GRID):
            for col in range(GRID):
                axis.text(col, row, f"{maps[0][row, col]:.3f}",
                          ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis)
        axes = [axis]
    else:
        difference = maps[0] - maps[1]
        vmax = max(float(maps[0].max()), float(maps[1].max()))
        dmax = max(float(np.abs(difference).max()), 1e-12)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        axes[0].imshow(maps[0], vmin=0, vmax=vmax)
        axes[0].set_title(cases[0][0])
        axes[1].imshow(maps[1], vmin=0, vmax=vmax)
        axes[1].set_title(cases[1][0])
        image = axes[2].imshow(difference, cmap="seismic", vmin=-dmax, vmax=dmax)
        axes[2].set_title(f"{cases[0][0]} − {cases[1][0]}")
        fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xticks(range(GRID), range(1, GRID + 1))
        axis.set_yticks(range(GRID), range(1, GRID + 1))
    fig.tight_layout()
    fig.savefig(output_dir / "LGN空间峰值热图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_features(path, kind):
    source = np.load(path)
    if kind == "gabor":
        return gabor_features(source)
    return delta_features(source)


def process_pair(task, stage, stimuli, feature_kind, cue_duration_ms, title):
    input_dir = GABOR_ROOT / task / stage
    output_dir = OUTPUT_ROOT / task / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [name for name, _, _ in stimuli]
    labels = [label for _, label, _ in stimuli]
    features = [make_features(input_dir / filename, feature_kind)
                for _, _, filename in stimuli]
    features[0], features[1] = normalize_pair(features[0], features[1])
    cases = []

    for name, label, feature in zip(names, labels, features):
        result = simulate_lgn(feature, cue_duration_ms)
        if not cases:
            np.save(output_dir / "time_ms.npy", result["time"])
        save_result(output_dir, name, result, feature)
        cases.append((label, result))

    save_summary(output_dir, [summary_row(label, result) for label, result in cases])
    plot_response(output_dir, title, cases)
    plot_peak_map(output_dir, cases)


def process_single(task, stage, name, label, filename, feature_kind, title):
    input_dir = GABOR_ROOT / task / stage
    output_dir = OUTPUT_ROOT / task / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    features = normalize_single(make_features(input_dir / filename, feature_kind))
    result = simulate_lgn(features)
    np.save(output_dir / "time_ms.npy", result["time"])
    save_result(output_dir, name, result, features)
    save_summary(output_dir, [summary_row(label, result)])
    plot_response(output_dir, title, [(label, result)])
    plot_peak_map(output_dir, [(label, result)])


def main():
    left_right = [
        ("left", "左提示", "stage1_gabor_left.npy"),
        ("right", "右提示", "stage1_gabor_right.npy"),
    ]
    process_pair("Task1", "Stage1", left_right, "gabor", 200.0,
                 "Task1 Stage1 LGN 响应")
    process_single("Task1", "Stage2", "dots", "双圆点",
                   "stage2_task1_delta_signed.npy", "delta",
                   "Task1 Stage2 双圆点 LGN 响应")
    process_pair("Task2", "Stage1", left_right, "gabor", 200.0,
                 "Task2 Stage1 LGN 响应")
    process_pair("Task2", "Stage2", [
        ("inward", "相向", "stage2_task2_gabor_inward.npy"),
        ("outward", "背向", "stage2_task2_gabor_outward.npy"),
    ], "gabor", None, "Task2 Stage2 LGN 响应")
    print("LGN 处理完成：", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
