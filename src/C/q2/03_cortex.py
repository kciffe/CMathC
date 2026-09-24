import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LGN_ROOT = SCRIPT_DIR / "output" / "02_lgn"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "03_cortex"

DT = 1.0
GRID = 4

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def activation(value, threshold=0.35, slope=5.0):
    sigmoid = 1.0 / (1.0 + np.exp(-slope * (value - threshold)))
    zero = 1.0 / (1.0 + np.exp(slope * threshold))
    response = (sigmoid - zero) / (1.0 - zero)
    return np.clip(response, 0.0, 1.0)


def shift_signal(signal, delay_ms):
    delay = int(round(delay_ms / DT))
    shifted = np.zeros_like(signal)
    if delay == 0:
        return signal.copy()
    shifted[:, delay:] = signal[:, :-delay]
    return shifted


def wilson_cowan(input_signal, tau_e, tau_i,
                  w_ee, w_ei, w_ie, w_ii,
                  gain_e, gain_i):
    channels, steps = input_signal.shape
    e = np.zeros((channels, steps), dtype=np.float32)
    i = np.zeros_like(e)

    for step in range(steps - 1):
        e_drive = (
            w_ee * e[:, step]
            - w_ei * i[:, step]
            + gain_e * input_signal[:, step]
        )

        i_drive = (
            w_ie * e[:, step]
            - w_ii * i[:, step]
            + gain_i * input_signal[:, step]
        )

        e_target = activation(e_drive)
        i_target = activation(i_drive)

        e[:, step + 1] = e[:, step] + DT * (
            -e[:, step]
            + (1.0 - e[:, step]) * e_target
        ) / tau_e

        i[:, step + 1] = i[:, step] + DT * (
            -i[:, step]
            + (1.0 - i[:, step]) * i_target
        ) / tau_i

    return e, i


def pool_v1_to_it(v1_e):
    # 64通道 = 4个方向 × 16个空间区域。
    # IT把同一空间位置的不同方向组合起来，得到16个空间形状通道。
    if v1_e.shape[0] == 64:
        return v1_e.reshape(4, 16, v1_e.shape[1]).mean(axis=0)
    return v1_e.copy()


def pool_it_to_pfc(it_e):
    # PFC保留左右两个空间通道，后面可用于F3/F4差异建模。
    grid = it_e.reshape(GRID, GRID, it_e.shape[1])

    left = grid[:, :2].mean(axis=(0, 1))
    right = grid[:, 2:].mean(axis=(0, 1))
    global_mean = grid.mean(axis=(0, 1))

    return np.stack([
        0.7 * left + 0.3 * global_mean,
        0.7 * right + 0.3 * global_mean,
    ])


def simulate_cortex(lgn_total):
    # V1：快速视觉皮层，继续保留LGN的空间/方向通道。
    v1_input = shift_signal(lgn_total, 8.0)
    v1_e, v1_i = wilson_cowan(
        v1_input,
        tau_e=18.0,
        tau_i=10.0,
        w_ee=1.4,
        w_ei=1.1,
        w_ie=1.0,
        w_ii=0.8,
        gain_e=2.2,
        gain_i=1.2,
    )

    # IT：组合V1不同方向信息，形成较慢的形状表征。
    it_input = pool_v1_to_it(v1_e)
    it_input = shift_signal(it_input, 25.0)
    it_e, it_i = wilson_cowan(
        it_input,
        tau_e=40.0,
        tau_i=22.0,
        w_ee=1.5,
        w_ei=1.0,
        w_ie=1.1,
        w_ii=0.8,
        gain_e=1.8,
        gain_i=0.9,
    )

    # PFC：更慢的认知整合，并保留左/右两个通道。
    pfc_input = pool_it_to_pfc(it_e)
    pfc_input = shift_signal(pfc_input, 50.0)
    pfc_e, pfc_i = wilson_cowan(
        pfc_input,
        tau_e=70.0,
        tau_i=40.0,
        w_ee=1.6,
        w_ei=1.1,
        w_ie=1.0,
        w_ii=0.9,
        gain_e=1.6,
        gain_i=0.7,
    )

    return {
        "v1_e": v1_e,
        "v1_i": v1_i,
        "it_e": it_e,
        "it_i": it_i,
        "pfc_e": pfc_e,
        "pfc_i": pfc_i,
        "v1_source": v1_e - v1_i,
        "it_source": it_e - it_i,
        "pfc_source": pfc_e - pfc_i,
    }


def save_result(output_dir, name, result):
    for key, value in result.items():
        if key.endswith("_source"):
            np.save(output_dir / f"{name}_{key}.npy", value)


def peak_info(signal, time_ms):
    mean_signal = signal.mean(axis=0)
    index = int(np.argmax(mean_signal))
    return float(mean_signal[index]), float(time_ms[index])


def summary_row(label, result, time_ms):
    v1_peak, v1_time = peak_info(result["v1_e"], time_ms)
    it_peak, it_time = peak_info(result["it_e"], time_ms)
    pfc_peak, pfc_time = peak_info(result["pfc_e"], time_ms)

    left = result["pfc_e"][0]
    right = result["pfc_e"][1]
    lateral = right - left
    lateral_index = int(np.argmax(np.abs(lateral)))

    return {
        "条件": label,
        "V1峰值": v1_peak,
        "V1峰值时间(ms)": v1_time,
        "IT峰值": it_peak,
        "IT峰值时间(ms)": it_time,
        "PFC峰值": pfc_peak,
        "PFC峰值时间(ms)": pfc_time,
        "PFC最大左右差": float(lateral[lateral_index]),
        "PFC最大左右差时间(ms)": float(time_ms[lateral_index]),
    }


def save_summary(output_dir, rows):
    path = output_dir / "皮层响应汇总.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_mean_response(output_dir, title, cases, time_ms):
    fig, axes = plt.subplots(
        1, len(cases),
        figsize=(6 * len(cases), 4),
        squeeze=False,
    )

    for axis, (label, result) in zip(axes[0], cases):
        axis.plot(time_ms, result["v1_e"].mean(axis=0), label="V1")
        axis.plot(time_ms, result["it_e"].mean(axis=0), label="IT")
        axis.plot(time_ms, result["pfc_e"].mean(axis=0), label="PFC")
        axis.set_title(label)
        axis.set_xlabel("时间 (ms)")
        axis.set_ylabel("兴奋性群体响应")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "皮层平均响应.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pfc_response(output_dir, title, cases, time_ms):
    fig, axes = plt.subplots(
        1, len(cases),
        figsize=(6 * len(cases), 4),
        squeeze=False,
    )

    for axis, (label, result) in zip(axes[0], cases):
        axis.plot(time_ms, result["pfc_e"][0], label="PFC左")
        axis.plot(time_ms, result["pfc_e"][1], label="PFC右")
        axis.set_title(label)
        axis.set_xlabel("时间 (ms)")
        axis.set_ylabel("PFC响应")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "PFC左右响应.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def it_peak_map(result):
    return result["it_e"].max(axis=1).reshape(GRID, GRID)


def plot_it_map(output_dir, cases):
    maps = [it_peak_map(result) for _, result in cases]

    if len(cases) == 1:
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(maps[0])
        axis.set_title(cases[0][0])
        axes = [axis]
        fig.colorbar(image, ax=axis)
    else:
        difference = maps[0] - maps[1]
        vmax = max(float(maps[0].max()), float(maps[1].max()))
        dmax = max(float(np.abs(difference).max()), 1e-12)

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        axes[0].imshow(maps[0], vmin=0, vmax=vmax)
        axes[0].set_title(cases[0][0])
        axes[1].imshow(maps[1], vmin=0, vmax=vmax)
        axes[1].set_title(cases[1][0])
        image = axes[2].imshow(
            difference,
            cmap="seismic",
            vmin=-dmax,
            vmax=dmax,
        )
        axes[2].set_title(f"{cases[0][0]} − {cases[1][0]}")
        fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xticks(range(GRID), range(1, GRID + 1))
        axis.set_yticks(range(GRID), range(1, GRID + 1))

    fig.tight_layout()
    fig.savefig(output_dir / "IT空间峰值热图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stage(task, stage, stimuli, title):
    input_dir = LGN_ROOT / task / stage
    output_dir = OUTPUT_ROOT / task / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    time_ms = np.load(input_dir / "time_ms.npy")
    np.save(output_dir / "time_ms.npy", time_ms)

    cases = []

    for name, label in stimuli:
        lgn_total = np.load(input_dir / f"{name}_lgn_total.npy")
        result = simulate_cortex(lgn_total)
        save_result(output_dir, name, result)
        cases.append((label, result))

    save_summary(
        output_dir,
        [summary_row(label, result, time_ms) for label, result in cases],
    )

    plot_mean_response(output_dir, title, cases, time_ms)
    plot_pfc_response(output_dir, title, cases, time_ms)
    plot_it_map(output_dir, cases)


def main():
    process_stage(
        "Task1",
        "Stage1",
        [("left", "左提示"), ("right", "右提示")],
        "Task1 Stage1 皮层响应",
    )

    process_stage(
        "Task1",
        "Stage2",
        [("dots", "双圆点")],
        "Task1 Stage2 皮层响应",
    )

    process_stage(
        "Task2",
        "Stage1",
        [("left", "左提示"), ("right", "右提示")],
        "Task2 Stage1 皮层响应",
    )

    process_stage(
        "Task2",
        "Stage2",
        [("inward", "相向"), ("outward", "背向")],
        "Task2 Stage2 皮层响应",
    )

    print("皮层模型处理完成：", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
