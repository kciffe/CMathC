import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output", "01_gabor_frontend")
TASK1_STAGE1_DIR = os.path.join(OUTPUT_ROOT, "Task1", "Stage1")
TASK1_STAGE2_DIR = os.path.join(OUTPUT_ROOT, "Task1", "Stage2")
TASK2_STAGE1_DIR = os.path.join(OUTPUT_ROOT, "Task2", "Stage1")
TASK2_STAGE2_DIR = os.path.join(OUTPUT_ROOT, "Task2", "Stage2")

ANGLES = [0, 45, 90, 135]
GRID = 4

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_img(name):
    return np.load(os.path.join(INPUT_DIR, name)).astype(np.float32) / 255.0


def make_gabor_kernel(theta_deg, size=31, sigma=4.0, lambd=10.0, gamma=0.5, psi=0.0):
    half = size // 2
    x, y = np.meshgrid(
        np.arange(-half, half + 1, dtype=np.float32),
        np.arange(-half, half + 1, dtype=np.float32)
    )

    theta = np.deg2rad(theta_deg)
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    kernel = np.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2))
    kernel *= np.cos(2 * np.pi * x_theta / lambd + psi)

    kernel -= kernel.mean()
    kernel /= np.sqrt(np.sum(kernel ** 2)) + 1e-12
    return kernel


def gabor_response(img, theta_deg):
    kernel_cos = make_gabor_kernel(theta_deg, psi=0.0)
    kernel_sin = make_gabor_kernel(theta_deg, psi=np.pi / 2)

    response_cos = fftconvolve(img, kernel_cos, mode="same")
    response_sin = fftconvolve(img, kernel_sin, mode="same")

    return np.sqrt(response_cos ** 2 + response_sin ** 2)


def calc_gabor(img):
    return np.stack([gabor_response(img, angle) for angle in ANGLES])


def spatial_energy(response, grid=4):
    h, w = response.shape
    values = []

    for row in range(grid):
        for col in range(grid):
            y1 = row * h // grid
            y2 = (row + 1) * h // grid
            x1 = col * w // grid
            x2 = (col + 1) * w // grid
            values.append(float(response[y1:y2, x1:x2].mean()))

    return values


def save_pair_csv(output_dir, name, label_a, label_b, response_a, response_b):
    global_rows = []
    spatial_rows = []

    for i, angle in enumerate(ANGLES):
        energy_a = float(response_a[i].mean())
        energy_b = float(response_b[i].mean())

        global_rows.append({
            "方向(°)": angle,
            f"{label_a}平均响应": energy_a,
            f"{label_b}平均响应": energy_b,
            f"{label_a}-{label_b}": energy_a - energy_b,
            "绝对差": abs(energy_a - energy_b)
        })

        grid_a = spatial_energy(response_a[i], GRID)
        grid_b = spatial_energy(response_b[i], GRID)

        for region in range(GRID * GRID):
            row = region // GRID + 1
            col = region % GRID + 1

            spatial_rows.append({
                "方向(°)": angle,
                "区域": f"R{row}C{col}",
                "行": row,
                "列": col,
                f"{label_a}平均响应": grid_a[region],
                f"{label_b}平均响应": grid_b[region],
                f"{label_a}-{label_b}": grid_a[region] - grid_b[region],
                "绝对差": abs(grid_a[region] - grid_b[region])
            })

    pd.DataFrame(global_rows).to_csv(
        os.path.join(output_dir, f"{name}_Gabor全局方向响应.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(spatial_rows).to_csv(
        os.path.join(output_dir, f"{name}_Gabor空间方向响应.csv"),
        index=False,
        encoding="utf-8-sig"
    )


def save_single_csv(output_dir, name, label, responses):
    rows = []

    for i, angle in enumerate(ANGLES):
        rows.append({
            "方向(°)": angle,
            f"{label}平均响应": float(responses[i].mean())
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, f"{name}_Gabor方向响应.csv"),
        index=False,
        encoding="utf-8-sig"
    )


def plot_pair_overview(output_dir, name, title_a, title_b, img_a, img_b, response_a, response_b):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    axes[0, 0].imshow(img_a, cmap="gray")
    axes[0, 0].set_title(title_a)
    axes[1, 0].imshow(img_b, cmap="gray")
    axes[1, 0].set_title(title_b)

    for i, angle in enumerate(ANGLES):
        vmax = max(response_a[i].max(), response_b[i].max())

        axes[0, i + 1].imshow(response_a[i], cmap="viridis", vmin=0, vmax=vmax)
        axes[0, i + 1].set_title(f"{title_a} {angle}°")

        axes[1, i + 1].imshow(response_b[i], cmap="viridis", vmin=0, vmax=vmax)
        axes[1, i + 1].set_title(f"{title_b} {angle}°")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_Gabor方向响应总览.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_diff(output_dir, name, label_a, label_b, response_a, response_b):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for i, angle in enumerate(ANGLES):
        diff = response_a[i] - response_b[i]
        vmax = np.max(np.abs(diff)) + 1e-12

        im = axes[i].imshow(diff, cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"{angle}° {label_a}-{label_b}")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_Gabor差异.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_spatial(output_dir, name, title_a, title_b, response_a, response_b):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))

    for i, angle in enumerate(ANGLES):
        grid_a = np.array(spatial_energy(response_a[i], GRID)).reshape(GRID, GRID)
        grid_b = np.array(spatial_energy(response_b[i], GRID)).reshape(GRID, GRID)
        vmax = max(grid_a.max(), grid_b.max())

        axes[0, i].imshow(grid_a, cmap="viridis", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"{title_a} {angle}°")

        axes[1, i].imshow(grid_b, cmap="viridis", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"{title_b} {angle}°")

        for r in range(GRID):
            for c in range(GRID):
                axes[0, i].text(c, r, f"{grid_a[r, c]:.3f}", ha="center", va="center", fontsize=7)
                axes[1, i].text(c, r, f"{grid_b[r, c]:.3f}", ha="center", va="center", fontsize=7)

    for ax in axes.ravel():
        ax.set_xticks(range(GRID))
        ax.set_yticks(range(GRID))
        ax.set_xticklabels([1, 2, 3, 4])
        ax.set_yticklabels([1, 2, 3, 4])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_Gabor空间响应热图.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_task1_spatial(output_dir, contrast):
    values = np.array(spatial_energy(contrast, GRID)).reshape(GRID, GRID)

    rows = []
    for r in range(GRID):
        for c in range(GRID):
            rows.append({
                "区域": f"R{r + 1}C{c + 1}",
                "行": r + 1,
                "列": c + 1,
                "平均刺激增量": values[r, c]
            })

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "stage2_task1_空间亮度特征.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(values, cmap="viridis")
    plt.title("第二阶段 Task-1 双圆点空间刺激增量")

    for r in range(GRID):
        for c in range(GRID):
            plt.text(c, r, f"{values[r, c]:.3f}", ha="center", va="center", fontsize=8)

    plt.xticks(range(GRID), [1, 2, 3, 4])
    plt.yticks(range(GRID), [1, 2, 3, 4])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage2_task1_空间亮度热图.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # 第一阶段：圆圈基线 -> 左/右三角提示
    stage1_baseline = load_img("stage1_baseline_circle.npy")
    cue_left = load_img("stage1_cue_left.npy")
    cue_right = load_img("stage1_cue_right.npy")

    # Gabor使用正的刺激对比；有符号亮度变化另存，后面LGN ON/OFF会用到
    stage1_left_signed = cue_left - stage1_baseline
    stage1_right_signed = cue_right - stage1_baseline
    stage1_left_contrast = np.clip(stage1_baseline - cue_left, 0, None)
    stage1_right_contrast = np.clip(stage1_baseline - cue_right, 0, None)

    stage1_left_gabor = calc_gabor(stage1_left_contrast)
    stage1_right_gabor = calc_gabor(stage1_right_contrast)
    stage1_diff = stage1_left_gabor - stage1_right_gabor

    for output_dir in (TASK1_STAGE1_DIR, TASK2_STAGE1_DIR):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "stage1_delta_left_signed.npy"), stage1_left_signed)
        np.save(os.path.join(output_dir, "stage1_delta_right_signed.npy"), stage1_right_signed)
        np.save(os.path.join(output_dir, "stage1_contrast_left.npy"), stage1_left_contrast)
        np.save(os.path.join(output_dir, "stage1_contrast_right.npy"), stage1_right_contrast)
        np.save(os.path.join(output_dir, "stage1_gabor_left.npy"), stage1_left_gabor)
        np.save(os.path.join(output_dir, "stage1_gabor_right.npy"), stage1_right_gabor)
        np.save(os.path.join(output_dir, "stage1_gabor_diff.npy"), stage1_diff)

        save_pair_csv(output_dir, "stage1", "左提示", "右提示", stage1_left_gabor, stage1_right_gabor)
        plot_pair_overview(
            output_dir, "stage1", "左提示增量", "右提示增量",
            stage1_left_contrast, stage1_right_contrast,
            stage1_left_gabor, stage1_right_gabor
        )
        plot_diff(output_dir, "stage1", "左提示", "右提示", stage1_left_gabor, stage1_right_gabor)
        plot_spatial(output_dir, "stage1", "左提示", "右提示", stage1_left_gabor, stage1_right_gabor)

    # 第二阶段公共白色基线
    stage2_baseline = load_img("stage2_baseline_blank.npy")

    # Task-1：双圆点。主要保留空间亮度特征，Gabor只作为描述性结果
    os.makedirs(TASK1_STAGE2_DIR, exist_ok=True)
    target_dots = load_img("stage2_task1_target_dots.npy")
    dots_signed = target_dots - stage2_baseline
    dots_contrast = np.clip(stage2_baseline - target_dots, 0, None)
    dots_gabor = calc_gabor(dots_contrast)

    np.save(os.path.join(TASK1_STAGE2_DIR, "stage2_task1_delta_signed.npy"), dots_signed)
    np.save(os.path.join(TASK1_STAGE2_DIR, "stage2_task1_contrast.npy"), dots_contrast)
    np.save(os.path.join(TASK1_STAGE2_DIR, "stage2_task1_gabor_dots.npy"), dots_gabor)

    save_single_csv(TASK1_STAGE2_DIR, "stage2_task1", "双圆点", dots_gabor)
    plot_task1_spatial(TASK1_STAGE2_DIR, dots_contrast)

    # Task-2：相向 / 背向双三角，比较空间方向差异
    os.makedirs(TASK2_STAGE2_DIR, exist_ok=True)
    target_inward = load_img("stage2_task2_target_inward.npy")
    target_outward = load_img("stage2_task2_target_outward.npy")

    inward_signed = target_inward - stage2_baseline
    outward_signed = target_outward - stage2_baseline
    inward_contrast = np.clip(stage2_baseline - target_inward, 0, None)
    outward_contrast = np.clip(stage2_baseline - target_outward, 0, None)

    inward_gabor = calc_gabor(inward_contrast)
    outward_gabor = calc_gabor(outward_contrast)
    task2_diff = inward_gabor - outward_gabor

    np.save(os.path.join(TASK2_STAGE2_DIR, "stage2_task2_delta_inward_signed.npy"), inward_signed)
    np.save(os.path.join(TASK2_STAGE2_DIR, "stage2_task2_delta_outward_signed.npy"), outward_signed)
    np.save(os.path.join(TASK2_STAGE2_DIR, "stage2_task2_contrast_inward.npy"), inward_contrast)
    np.save(os.path.join(TASK2_STAGE2_DIR, "stage2_task2_contrast_outward.npy"), outward_contrast)
    np.save(os.path.join(TASK2_STAGE2_DIR, "stage2_task2_gabor_inward.npy"), inward_gabor)
    np.save(os.path.join(TASK2_STAGE2_DIR, "stage2_task2_gabor_outward.npy"), outward_gabor)
    np.save(os.path.join(TASK2_STAGE2_DIR, "stage2_task2_gabor_diff.npy"), task2_diff)

    save_pair_csv(TASK2_STAGE2_DIR, "stage2_task2", "相向", "背向", inward_gabor, outward_gabor)
    plot_pair_overview(
        TASK2_STAGE2_DIR, "stage2_task2", "相向增量", "背向增量",
        inward_contrast, outward_contrast,
        inward_gabor, outward_gabor
    )
    plot_diff(TASK2_STAGE2_DIR, "stage2_task2", "相向", "背向", inward_gabor, outward_gabor)
    plot_spatial(TASK2_STAGE2_DIR, "stage2_task2", "相向", "背向", inward_gabor, outward_gabor)

    print("Gabor视觉前端处理完成")
    print("第一阶段：左/右提示刺激增量 -> Gabor空间方向差异")
    print("第二阶段 Task-1：双圆点刺激增量 -> 空间亮度特征 + 描述性Gabor响应")
    print("第二阶段 Task-2：相向/背向双三角刺激增量 -> Gabor空间方向差异")
    print("输出目录:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
