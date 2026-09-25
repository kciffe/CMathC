"""Generate the two targeted v3 diagnosis figures used by output/修改2.md."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import config
from frontend import load_stimulus, simulate_frontend
from model import ModelParams, project_frontend, simulate_forward, _wc_populations
from observation import model_curve_to_q1_grid


def main():
    root = config.Q2_ROOT
    model_dir = config.OUTPUT_ROOT
    out = root / "output" / "revision_v3_validation"
    out.mkdir(parents=True, exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9,
                         "legend.fontsize": 8, "figure.dpi": 120,
                         "savefig.dpi": 300})

    fits = json.loads((model_dir / "heldout_fit_details.json").read_text(encoding="utf-8"))
    fit = fits["VisualCogA_Task-1"]
    params = ModelParams.from_any(fit["parameters"])
    rows = list(csv.DictReader((model_dir / "drive_scales.csv").open(encoding="utf-8-sig")))
    scales = np.asarray([float(row["reference_rms_scale"]) for row in rows], dtype=np.float32).reshape(3, 2)
    time_ms = np.arange(0.0, 3001.0, 1.0)
    frontend = simulate_frontend(load_stimulus("Stage1", "left"),
                                 params={"tau_a": params.tau_a}, resolution=128,
                                 time_ms=time_ms, feature_stride_ms=4.0)
    drives = project_frontend(frontend, drive_scales=scales)
    excitatory, inhibitory, delayed_drive = _wc_populations(drives, time_ms, params)
    result = simulate_forward(frontend, params=params, time_ms=time_ms,
                              amplitude=float(fit["amplitude"]), drive_scales=scales)

    # Figure 1: normalized population states and the declining external drive.
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True)
    colors = {"E": "#2878B5", "I": "#D9822B"}
    names = ("早期视野群", "空间构型群", "三角模板偏好群")
    for index, ax in enumerate(axes[:3]):
        ax.plot(time_ms, excitatory[index].mean(axis=0), color=colors["E"], lw=1.8, label="兴奋群体 E")
        ax.plot(time_ms, inhibitory[index].mean(axis=0), color=colors["I"], lw=1.8,
                ls="--", label="抑制群体 I")
        ax.set_ylabel("归一化状态")
        ax.set_ylim(-0.025, 0.56)
        ax.set_title(names[index], loc="left", pad=3)
        ax.grid(alpha=0.2)
        if index == 0:
            ax.legend(ncol=2, frameon=False, loc="upper right")
    ax = axes[3]
    drive_mean = delayed_drive.mean(axis=(0, 1))
    ax.plot(time_ms, drive_mean, color="#4C566A", lw=1.7, label="群体平均外部驱动")
    ax.set_ylabel("归一化驱动")
    ax.set_xlabel("提示出现后时间 (ms)")
    ax.set_ylim(-0.02, max(0.8, float(drive_mean.max()) * 1.1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.axvspan(0, 203.125, color="#89B4A7", alpha=0.12, lw=0)
        ax.axvline(203.125, color="#3C7163", lw=1, ls=":")
        ax.axvline(2200, color="#888888", lw=1, ls=":")
    fig.suptitle("提示后群体活动拖尾\n代表记录 VisualCogA_Task-1；留出拟合未收敛；2200 ms 目标未纳入模拟",
                 fontsize=11)
    fig.savefig(out / "persistent_activity_decay.png", facecolor="white")
    plt.close(fig)

    # Figure 2: same zero-phase Q1 operator, with and without the post-800-ms tail.
    truncated_signal = result.eeg_scaled.copy()
    truncated_signal[:, time_ms > 800.0] = 0.0
    full_obs, obs_time = model_curve_to_q1_grid(result.eeg_scaled, time_ms)
    cut_obs, cut_time = model_curve_to_q1_grid(truncated_signal, time_ms)
    window = (obs_time >= 0.0) & (obs_time <= 796.875)
    difference = full_obs[:, window] - cut_obs[:, window]
    relative_rmse = float(np.sqrt(np.mean(difference ** 2)) /
                          max(float(np.sqrt(np.mean(full_obs[:, window] ** 2))), 1e-12))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.7), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    channels = ("F3", "Fz", "F4")
    channel_colors = ("#2878B5", "#D9822B", "#3A8F6B")
    ax = axes[0]
    for index, (channel, color) in enumerate(zip(channels, channel_colors)):
        ax.plot(obs_time[window], full_obs[index, window], color=color, lw=1.6,
                label=f"{channel}：保留自然尾部")
        ax.plot(cut_time[window], cut_obs[index, window], color=color, lw=1.2, ls="--",
                alpha=0.8, label=f"{channel}：800 ms 后补零")
    ax.axvline(800, color="#555555", ls=":", lw=1)
    ax.set_ylabel("模拟电位（相对单位）")
    ax.set_title("相同零相位滤波下，800 ms 后补零会改变分析窗内曲线", loc="left")
    ax.grid(alpha=0.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="#555555", lw=1.6, label="保留自然尾部"),
                       Line2D([0], [0], color="#555555", lw=1.4, ls="--", label="800 ms 后补零")],
              ncol=2, frameon=False, fontsize=7.5, loc="upper left")
    ax = axes[1]
    for index, (channel, color) in enumerate(zip(channels, channel_colors)):
        ax.plot(obs_time[window], difference[index], color=color, lw=1.4, label=channel)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("提示出现后时间 (ms)")
    ax.set_ylabel("自然尾部 − 补零")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    fig.suptitle(f"相同零相位滤波下，800 ms 后补零会改变分析窗内曲线\n"
                 f"代表记录 VisualCogA_Task-1 相对 RMSE={relative_rmse:.3f}；8 个记录×条件中位数=0.149（0.123–0.166）",
                 fontsize=10.5)
    fig.savefig(out / "zero_padding_filter_effect.png", facecolor="white")
    plt.close(fig)

    print(f"Saved {out / 'persistent_activity_decay.png'}")
    print(f"Saved {out / 'zero_padding_filter_effect.png'}")
    print(f"representative_relative_rmse={relative_rmse:.6f}")
    print(f"E800={np.round(excitatory[:, :, 800].mean(axis=1), 6).tolist()}")
    print(f"I800={np.round(inhibitory[:, :, 800].mean(axis=1), 6).tolist()}")
    print(f"drive800={float(delayed_drive[:, :, 800].mean()):.8f}")
    print(f"E2000={np.round(excitatory[:, :, 2000].mean(axis=1), 6).tolist()}")


if __name__ == "__main__":
    main()
