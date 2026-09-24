import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


LEFT_PATH = r"D:\8\Desktop\CMathc\src\C\q2\input\img_left.npy"
RIGHT_PATH = r"D:\8\Desktop\CMathc\src\C\q2\input\img_right.npy"
OUTPUT_DIR = r"D:\8\Desktop\CMathc\src\C\q2\output"

ANGLES = [0, 45, 90, 135]
GRID = 4

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 白色背景转为0，刺激越深数值越大
    img_left = 1.0 - np.load(LEFT_PATH).astype(np.float32) / 255.0
    img_right = 1.0 - np.load(RIGHT_PATH).astype(np.float32) / 255.0

    left_responses = []
    right_responses = []

    for angle in ANGLES:
        left_responses.append(gabor_response(img_left, angle))
        right_responses.append(gabor_response(img_right, angle))

    left_responses = np.stack(left_responses)
    right_responses = np.stack(right_responses)

    np.save(os.path.join(OUTPUT_DIR, "gabor_left.npy"), left_responses)
    np.save(os.path.join(OUTPUT_DIR, "gabor_right.npy"), right_responses)

    global_rows = []
    spatial_rows = []

    for i, angle in enumerate(ANGLES):
        left_energy = float(left_responses[i].mean())
        right_energy = float(right_responses[i].mean())

        global_rows.append({
            "方向(°)": angle,
            "左刺激平均响应": left_energy,
            "右刺激平均响应": right_energy,
            "左-右": left_energy - right_energy,
            "绝对差": abs(left_energy - right_energy)
        })

        left_grid = spatial_energy(left_responses[i], GRID)
        right_grid = spatial_energy(right_responses[i], GRID)

        for region in range(GRID * GRID):
            row = region // GRID + 1
            col = region % GRID + 1
            spatial_rows.append({
                "方向(°)": angle,
                "区域": f"R{row}C{col}",
                "行": row,
                "列": col,
                "左刺激平均响应": left_grid[region],
                "右刺激平均响应": right_grid[region],
                "左-右": left_grid[region] - right_grid[region],
                "绝对差": abs(left_grid[region] - right_grid[region])
            })

    pd.DataFrame(global_rows).to_csv(
        os.path.join(OUTPUT_DIR, "Gabor全局方向响应.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    spatial_df = pd.DataFrame(spatial_rows)
    spatial_df.to_csv(
        os.path.join(OUTPUT_DIR, "Gabor空间方向响应.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 总览图：原始刺激 + 四方向响应
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    axes[0, 0].imshow(img_left, cmap="gray")
    axes[0, 0].set_title("左刺激")
    axes[1, 0].imshow(img_right, cmap="gray")
    axes[1, 0].set_title("右刺激")

    for i, angle in enumerate(ANGLES):
        vmax = max(left_responses[i].max(), right_responses[i].max())

        axes[0, i + 1].imshow(left_responses[i], cmap="viridis", vmin=0, vmax=vmax)
        axes[0, i + 1].set_title(f"左刺激 {angle}°")

        axes[1, i + 1].imshow(right_responses[i], cmap="viridis", vmin=0, vmax=vmax)
        axes[1, i + 1].set_title(f"右刺激 {angle}°")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Gabor方向响应总览.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 左右差异图
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for i, angle in enumerate(ANGLES):
        diff = left_responses[i] - right_responses[i]
        vmax = np.max(np.abs(diff))
        im = axes[i].imshow(diff, cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"{angle}° 左-右")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Gabor左右差异.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4×4空间响应热图
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))

    for i, angle in enumerate(ANGLES):
        left_grid = np.array(spatial_energy(left_responses[i], GRID)).reshape(GRID, GRID)
        right_grid = np.array(spatial_energy(right_responses[i], GRID)).reshape(GRID, GRID)
        vmax = max(left_grid.max(), right_grid.max())

        im1 = axes[0, i].imshow(left_grid, cmap="viridis", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"左刺激 {angle}°")

        axes[1, i].imshow(right_grid, cmap="viridis", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"右刺激 {angle}°")

        for r in range(GRID):
            for c in range(GRID):
                axes[0, i].text(c, r, f"{left_grid[r, c]:.3f}", ha="center", va="center", fontsize=7)
                axes[1, i].text(c, r, f"{right_grid[r, c]:.3f}", ha="center", va="center", fontsize=7)

    for ax in axes.ravel():
        ax.set_xticks(range(GRID))
        ax.set_yticks(range(GRID))
        ax.set_xticklabels([1, 2, 3, 4])
        ax.set_yticklabels([1, 2, 3, 4])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Gabor空间响应热图.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Gabor视觉前端处理完成")
    print("输出目录:", OUTPUT_DIR)
    print("gabor_left.npy 形状:", left_responses.shape)
    print("gabor_right.npy 形状:", right_responses.shape)
    print("空间特征数:", len(spatial_df))


if __name__ == "__main__":
    main()
