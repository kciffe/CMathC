import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CORTEX_ROOT = SCRIPT_DIR / "output" / "03_cortex"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "04_simulated_eeg"

GRID = 4
CHANNELS = ["F3", "Fz", "F4"]

# 简化导联矩阵：6个皮层源 -> F3/Fz/F4
# 6个源依次为：
# V1左、V1右、IT左、IT右、PFC左空间、PFC右空间
# 这里只表示相对贡献，不是基于MRI计算的真实头模型导联场
LEAD_FIELD = np.array([
    [0.32, 0.24, 0.48, 0.36, 0.72, 0.58],
    [0.28, 0.28, 0.42, 0.42, 0.65, 0.65],
    [0.24, 0.32, 0.36, 0.48, 0.58, 0.72],
], dtype=np.float32)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def spatial_left_right(source):
    # V1有64通道：4方向 × 4×4空间区域
    if source.shape[0] == 64:
        spatial = source.reshape(
            4,
            GRID,
            GRID,
            source.shape[1]
        ).mean(axis=0)

    # IT或Task1 Stage2 V1为16个空间区域
    else:
        spatial = source.reshape(
            GRID,
            GRID,
            source.shape[1]
        )

    left = spatial[:, :2].mean(axis=(0, 1))
    right = spatial[:, 2:].mean(axis=(0, 1))

    return left, right


def build_sources(v1_source, it_source, pfc_source):
    v1_left, v1_right = spatial_left_right(v1_source)
    it_left, it_right = spatial_left_right(it_source)

    pfc_left = pfc_source[0]
    pfc_right = pfc_source[1]

    return np.stack([
        v1_left,
        v1_right,
        it_left,
        it_right,
        pfc_left,
        pfc_right,
    ]).astype(np.float32)


def map_to_eeg(source_matrix):
    return LEAD_FIELD @ source_matrix


def late_peak(signal, time_ms):
    mask = (
        (time_ms >= 250)
        & (time_ms <= 500)
    )

    values = signal[mask]
    times = time_ms[mask]

    index = int(np.argmax(values))

    return (
        float(values[index]),
        float(times[index])
    )


def summary_row(label, eeg, time_ms):
    row = {
        "条件": label
    }

    for i, channel in enumerate(CHANNELS):
        peak, peak_time = late_peak(
            eeg[i],
            time_ms
        )

        row[f"{channel}晚期峰值"] = peak
        row[f"{channel}晚期峰时间(ms)"] = peak_time

    asymmetry = eeg[2] - eeg[0]
    index = int(
        np.argmax(
            np.abs(asymmetry)
        )
    )

    row["F4-F3最大差"] = float(
        asymmetry[index]
    )

    row["F4-F3最大差时间(ms)"] = float(
        time_ms[index]
    )

    return row


def save_summary(output_dir, rows):
    path = (
        output_dir
        / "EEG响应汇总.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_eeg(
    output_dir,
    title,
    cases,
    time_ms
):
    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=(
            6 * len(cases),
            4
        ),
        squeeze=False,
        sharex=True,
        sharey=True
    )

    max_value = max(
        float(
            np.max(
                np.abs(eeg)
            )
        )
        for _, eeg in cases
    )

    max_value = max(
        max_value,
        1e-12
    )

    for axis, (
        label,
        eeg
    ) in zip(
        axes[0],
        cases
    ):

        for i, channel in enumerate(
            CHANNELS
        ):
            axis.plot(
                time_ms,
                eeg[i],
                label=channel
            )

        axis.axvspan(
            250,
            500,
            alpha=0.08
        )

        axis.axhline(
            0,
            linewidth=0.8
        )

        axis.set_title(
            label
        )

        axis.set_xlabel(
            "时间 (ms)"
        )

        axis.set_ylabel(
            "模拟电位（归一化单位）"
        )

        axis.set_ylim(
            -1.1 * max_value,
            1.1 * max_value
        )

        axis.grid(
            alpha=0.25
        )

        axis.legend()

    fig.suptitle(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "模拟EEG.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def process_stage(
    task,
    stage,
    stimuli,
    title
):
    input_dir = (
        CORTEX_ROOT
        / task
        / stage
    )

    output_dir = (
        OUTPUT_ROOT
        / task
        / stage
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cases = []
    rows = []

    for name, label in stimuli:
        v1_source = np.load(
            input_dir
            / f"{name}_v1_source.npy"
        )

        it_source = np.load(
            input_dir
            / f"{name}_it_source.npy"
        )

        pfc_source = np.load(
            input_dir
            / f"{name}_pfc_source.npy"
        )

        sources = build_sources(
            v1_source,
            it_source,
            pfc_source
        )

        eeg = map_to_eeg(
            sources
        ).astype(np.float32)

        time_ms = np.arange(
            eeg.shape[1],
            dtype=np.float32
        )

        np.save(
            output_dir
            / f"{name}_eeg.npy",
            eeg
        )

        rows.append(
            summary_row(
                label,
                eeg,
                time_ms
            )
        )

        cases.append(
            (
                label,
                eeg
            )
        )

    save_summary(
        output_dir,
        rows
    )

    plot_eeg(
        output_dir,
        title,
        cases,
        time_ms
    )


def main():
    process_stage(
        "Task1",
        "Stage1",
        [
            ("left", "左提示"),
            ("right", "右提示"),
        ],
        "Task1 Stage1 模拟 EEG",
    )

    process_stage(
        "Task1",
        "Stage2",
        [
            ("dots", "双圆点"),
        ],
        "Task1 Stage2 模拟 EEG",
    )

    process_stage(
        "Task2",
        "Stage1",
        [
            ("left", "左提示"),
            ("right", "右提示"),
        ],
        "Task2 Stage1 模拟 EEG",
    )

    process_stage(
        "Task2",
        "Stage2",
        [
            ("inward", "相向"),
            ("outward", "背向"),
        ],
        "Task2 Stage2 模拟 EEG",
    )

    print(
        "模拟 EEG 处理完成：",
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()
