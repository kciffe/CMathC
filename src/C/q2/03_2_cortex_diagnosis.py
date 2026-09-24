import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CORTEX_ROOT = SCRIPT_DIR / "output" / "03_cortex"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "03_2_cortex_diagnosis"

GRID = 4

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def relative_l2(a, b):
    diff = np.linalg.norm(a - b)
    scale = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    return float(diff / (scale + 1e-12))


def mirror_signal(signal, level):
    if level == "V1":
        # 64通道 = 4方向 × 4行 × 4列
        data = signal.reshape(4, GRID, GRID, signal.shape[1])
        # 上游方向顺序为 [0°, 45°, 90°, 135°]；水平镜像交换 45° 和 135°。
        data = data[[0, 3, 2, 1], :, :, :]
        return data[:, :, ::-1, :].reshape(signal.shape)

    if level == "IT":
        data = signal.reshape(GRID, GRID, signal.shape[1])
        return data[:, ::-1, :].reshape(signal.shape)

    # PFC两个通道表示视觉空间左/右，镜像时交换
    return signal[::-1]


def spatial_peak_map(signal, level):
    peak = np.max(np.abs(signal), axis=1)

    if level == "V1":
        return peak.reshape(4, GRID, GRID).mean(axis=0)

    return peak.reshape(GRID, GRID)


def diagnose_pair(task, stage, name_a, label_a, name_b, label_b, check_mirror=False):
    input_dir = CORTEX_ROOT / task / stage
    output_dir = OUTPUT_ROOT / task / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    time_ms = np.arange(
        np.load(input_dir / f"{name_a}_v1_source.npy").shape[1],
        dtype=np.float32
    )

    rows = []
    signals = {}

    for level, key in [
        ("V1", "v1_source"),
        ("IT", "it_source"),
        ("PFC", "pfc_source"),
    ]:
        a = np.load(input_dir / f"{name_a}_{key}.npy")
        b = np.load(input_dir / f"{name_b}_{key}.npy")

        diff_curve = np.mean(np.abs(a - b), axis=0)
        peak_index = int(np.argmax(diff_curve))

        mirror_residual = ""
        if check_mirror:
            mirror_residual = relative_l2(
                a,
                mirror_signal(b, level)
            )

        rows.append({
            "层级": level,
            "相对L2条件差异": relative_l2(a, b),
            "平均绝对差峰值": float(diff_curve[peak_index]),
            "差异峰值时间(ms)": float(time_ms[peak_index]),
            "镜像交换残差": mirror_residual,
        })

        signals[level] = {
            "a": a,
            "b": b,
            "diff_curve": diff_curve,
        }

    with (output_dir / "皮层条件差异诊断.csv").open(
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

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 7)
    )

    for col, level in enumerate(["V1", "IT", "PFC"]):
        axes[0, col].plot(
            time_ms,
            signals[level]["diff_curve"]
        )
        axes[0, col].set_title(
            f"{level} 条件间平均绝对差"
        )
        axes[0, col].set_xlabel("时间 (ms)")
        axes[0, col].set_ylabel("差异幅值")
        axes[0, col].grid(alpha=0.25)

    it_a_map = spatial_peak_map(
        signals["IT"]["a"],
        "IT"
    )

    it_b_map = spatial_peak_map(
        signals["IT"]["b"],
        "IT"
    )

    it_diff = it_a_map - it_b_map

    vmax = max(
        float(it_a_map.max()),
        float(it_b_map.max()),
        1e-12
    )

    axes[1, 0].imshow(
        it_a_map,
        vmin=0,
        vmax=vmax
    )
    axes[1, 0].set_title(
        f"IT空间峰值：{label_a}"
    )

    axes[1, 1].imshow(
        it_b_map,
        vmin=0,
        vmax=vmax
    )
    axes[1, 1].set_title(
        f"IT空间峰值：{label_b}"
    )

    dmax = max(
        float(np.abs(it_diff).max()),
        1e-12
    )

    image = axes[1, 2].imshow(
        it_diff,
        cmap="seismic",
        vmin=-dmax,
        vmax=dmax
    )

    axes[1, 2].set_title(
        f"IT差异：{label_a} − {label_b}"
    )

    fig.colorbar(
        image,
        ax=axes[1, 2],
        fraction=0.046,
        pad=0.04
    )

    for axis in axes[1]:
        axis.set_xticks(
            range(GRID),
            range(1, GRID + 1)
        )
        axis.set_yticks(
            range(GRID),
            range(1, GRID + 1)
        )

    fig.suptitle(
        f"{task} {stage} 皮层条件差异诊断"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "皮层条件差异诊断.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\n{task} {stage}: {label_a} vs {label_b}")
    for row in rows:
        print(
            row["层级"],
            "相对L2差异 =",
            f'{row["相对L2条件差异"]:.4f}',
            "差异峰值时间 =",
            f'{row["差异峰值时间(ms)"]:.0f} ms'
        )

    if check_mirror:
        print(
            "Stage1镜像残差越小，说明左右提示主要保持镜像关系，"
            "而不是在皮层级联中被压成完全相同。"
        )


def main():
    # Task1 Stage1：左提示 vs 右提示
    diagnose_pair(
        "Task1",
        "Stage1",
        "left",
        "左提示",
        "right",
        "右提示",
        check_mirror=True
    )

    # Task2 Stage1：左提示 vs 右提示
    diagnose_pair(
        "Task2",
        "Stage1",
        "left",
        "左提示",
        "right",
        "右提示",
        check_mirror=True
    )

    # Task2 Stage2：相向 vs 背向
    diagnose_pair(
        "Task2",
        "Stage2",
        "inward",
        "相向",
        "outward",
        "背向",
        check_mirror=False
    )

    print(
        "\n诊断完成：",
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()
