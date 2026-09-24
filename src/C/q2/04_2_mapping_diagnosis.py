import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CORTEX_ROOT = SCRIPT_DIR / "output" / "03_cortex"
EEG_ROOT = SCRIPT_DIR / "output" / "04_simulated_eeg"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "04_2_mapping_diagnosis"

GRID = 4
CHANNELS = ["F3", "Fz", "F4"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def spatial_left_right(source):
    if source.shape[0] == 64:
        spatial = source.reshape(
            4, GRID, GRID, source.shape[1]
        ).mean(axis=0)
    else:
        spatial = source.reshape(
            GRID, GRID, source.shape[1]
        )

    left = spatial[:, :2].mean(axis=(0, 1))
    right = spatial[:, 2:].mean(axis=(0, 1))

    return left, right


def build_sources(v1_source, it_source, pfc_source):
    v1_left, v1_right = spatial_left_right(v1_source)
    it_left, it_right = spatial_left_right(it_source)

    return np.stack([
        v1_left,
        v1_right,
        it_left,
        it_right,
        pfc_source[0],
        pfc_source[1],
    ]).astype(np.float32)


def relative_l2(a, b):
    diff = np.linalg.norm(a - b)
    scale = 0.5 * (
        np.linalg.norm(a)
        + np.linalg.norm(b)
    )
    return float(diff / (scale + 1e-12))


def cosine_similarity(a, b):
    return float(
        np.dot(a, b)
        / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
            + 1e-12
        )
    )


def max_asymmetry(eeg, time_ms, start=None, end=None):
    diff = eeg[2] - eeg[0]

    if start is not None:
        mask = (
            (time_ms >= start)
            & (time_ms <= end)
        )
        values = diff[mask]
        times = time_ms[mask]
    else:
        values = diff
        times = time_ms

    index = int(
        np.argmax(
            np.abs(values)
        )
    )

    return (
        float(values[index]),
        float(times[index])
    )


def load_condition(task, stage, name):
    cortex_dir = CORTEX_ROOT / task / stage
    eeg_dir = EEG_ROOT / task / stage

    v1 = np.load(
        cortex_dir
        / f"{name}_v1_source.npy"
    )

    it = np.load(
        cortex_dir
        / f"{name}_it_source.npy"
    )

    pfc = np.load(
        cortex_dir
        / f"{name}_pfc_source.npy"
    )

    eeg = np.load(
        eeg_dir
        / f"{name}_eeg.npy"
    )

    sources = build_sources(
        v1,
        it,
        pfc
    )

    return sources, eeg


def recover_lead_field(all_sources, all_eeg):
    source_matrix = np.concatenate(
        all_sources,
        axis=1
    )

    eeg_matrix = np.concatenate(
        all_eeg,
        axis=1
    )

    return (
        eeg_matrix
        @ np.linalg.pinv(source_matrix)
    )


def save_lead_field(output_dir, lead_field):
    path = output_dir / "导联矩阵诊断.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as stream:

        writer = csv.writer(stream)

        writer.writerow([
            "电极",
            "V1左",
            "V1右",
            "IT左",
            "IT右",
            "PFC左",
            "PFC右",
        ])

        for channel, row in zip(
            CHANNELS,
            lead_field
        ):
            writer.writerow(
                [channel]
                + [
                    float(value)
                    for value in row
                ]
            )


def save_diagnosis(output_dir, rows):
    path = output_dir / "映射诊断.csv"

    fieldnames = [
        "类型",
        "任务阶段",
        "条件",
        "指标",
        "数值",
        "时间(ms)",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_lead_field(output_dir, lead_field):
    fig, axis = plt.subplots(
        figsize=(8, 4)
    )

    image = axis.imshow(
        lead_field,
        aspect="auto"
    )

    axis.set_xticks(
        range(6),
        [
            "V1左",
            "V1右",
            "IT左",
            "IT右",
            "PFC左",
            "PFC右",
        ]
    )

    axis.set_yticks(
        range(3),
        CHANNELS
    )

    axis.set_title(
        "当前有效导联矩阵"
    )

    fig.colorbar(
        image,
        ax=axis,
        label="相对权重"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "导联矩阵热图.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    conditions = [
        (
            "Task1",
            "Stage1",
            "left",
            "左提示"
        ),
        (
            "Task1",
            "Stage1",
            "right",
            "右提示"
        ),
        (
            "Task1",
            "Stage2",
            "dots",
            "双圆点"
        ),
        (
            "Task2",
            "Stage1",
            "left",
            "左提示"
        ),
        (
            "Task2",
            "Stage1",
            "right",
            "右提示"
        ),
        (
            "Task2",
            "Stage2",
            "inward",
            "相向"
        ),
        (
            "Task2",
            "Stage2",
            "outward",
            "背向"
        ),
    ]

    data = {}
    all_sources = []
    all_eeg = []

    rows = []

    for task, stage, name, label in conditions:
        sources, eeg = load_condition(
            task,
            stage,
            name
        )

        key = (
            task,
            stage,
            name
        )

        data[key] = (
            sources,
            eeg
        )

        all_sources.append(
            sources
        )

        all_eeg.append(
            eeg
        )

        time_ms = np.arange(
            eeg.shape[1],
            dtype=np.float32
        )

        full_value, full_time = max_asymmetry(
            eeg,
            time_ms
        )

        late_value, late_time = max_asymmetry(
            eeg,
            time_ms,
            250,
            500
        )

        spread = np.mean(
            np.std(
                eeg,
                axis=0
            )
        )

        scale = np.sqrt(
            np.mean(
                eeg ** 2
            )
        )

        rows.append({
            "类型": "单条件",
            "任务阶段": f"{task}/{stage}",
            "条件": label,
            "指标": "三电极相对空间分离度",
            "数值": float(
                spread
                / (scale + 1e-12)
            ),
            "时间(ms)": "",
        })

        rows.append({
            "类型": "单条件",
            "任务阶段": f"{task}/{stage}",
            "条件": label,
            "指标": "F4-F3全时段最大差",
            "数值": full_value,
            "时间(ms)": full_time,
        })

        rows.append({
            "类型": "单条件",
            "任务阶段": f"{task}/{stage}",
            "条件": label,
            "指标": "F4-F3晚期窗最大差",
            "数值": late_value,
            "时间(ms)": late_time,
        })

    lead_field = recover_lead_field(
        all_sources,
        all_eeg
    )

    save_lead_field(
        OUTPUT_ROOT,
        lead_field
    )

    plot_lead_field(
        OUTPUT_ROOT,
        lead_field
    )

    for i, j in [
        (0, 1),
        (0, 2),
        (1, 2),
    ]:
        rows.append({
            "类型": "导联矩阵",
            "任务阶段": "全部条件",
            "条件": f"{CHANNELS[i]} vs {CHANNELS[j]}",
            "指标": "导联权重余弦相似度",
            "数值": cosine_similarity(
                lead_field[i],
                lead_field[j]
            ),
            "时间(ms)": "",
        })

    pairs = [
        (
            "Task1",
            "Stage1",
            "left",
            "right",
            "左提示 vs 右提示"
        ),
        (
            "Task2",
            "Stage1",
            "left",
            "right",
            "左提示 vs 右提示"
        ),
        (
            "Task2",
            "Stage2",
            "inward",
            "outward",
            "相向 vs 背向"
        ),
    ]

    for task, stage, name_a, name_b, label in pairs:
        source_a, eeg_a = data[
            (task, stage, name_a)
        ]

        source_b, eeg_b = data[
            (task, stage, name_b)
        ]

        source_diff = relative_l2(
            source_a,
            source_b
        )

        eeg_diff = relative_l2(
            eeg_a,
            eeg_b
        )

        rows.append({
            "类型": "条件比较",
            "任务阶段": f"{task}/{stage}",
            "条件": label,
            "指标": "六源相对L2条件差异",
            "数值": source_diff,
            "时间(ms)": "",
        })

        rows.append({
            "类型": "条件比较",
            "任务阶段": f"{task}/{stage}",
            "条件": label,
            "指标": "三电极相对L2条件差异",
            "数值": eeg_diff,
            "时间(ms)": "",
        })

        rows.append({
            "类型": "条件比较",
            "任务阶段": f"{task}/{stage}",
            "条件": label,
            "指标": "条件差异保留比",
            "数值": float(
                eeg_diff
                / (source_diff + 1e-12)
            ),
            "时间(ms)": "",
        })

    save_diagnosis(
        OUTPUT_ROOT,
        rows
    )

    print(
        "04_2 映射诊断完成：",
        OUTPUT_ROOT
    )

    print(
        "F3-Fz 导联相似度：",
        f"{cosine_similarity(lead_field[0], lead_field[1]):.4f}"
    )

    print(
        "Fz-F4 导联相似度：",
        f"{cosine_similarity(lead_field[1], lead_field[2]):.4f}"
    )

    print(
        "F3-F4 导联相似度：",
        f"{cosine_similarity(lead_field[0], lead_field[2]):.4f}"
    )


if __name__ == "__main__":
    main()
