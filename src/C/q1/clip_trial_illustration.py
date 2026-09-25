"""Create a paper-ready illustration of the raw EEG clipping rule."""

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parents[2]
INPUT_PATH = PROJECT_DIR / "output" / "VisualCogA_Task-2_sliced_with_drop.mat"
OUTPUT_DIR = PROJECT_DIR / "output" / "clip_trial_illustration"

DATASET = "VisualCogA_Task-2"
TRIAL_NUMBER = 33  # one-based, matching the paper's Trial numbering
CHANNEL_NAMES = ("Fz", "F3", "F4")
CLIP_THRESHOLD = 999.0
CHANNEL_COLORS = {"Fz": "#0072B2", "F3": "#0072B2", "F4": "#0072B2"}
CLIP_COLOR = "#D55E00"
THRESHOLD_COLOR = "#E69F00"

sys.path.insert(0, str(PROJECT_ROOT / "math-model-agent" / "code"))
from algorithms.sci_figures import paper_figure_rc_params, publication_size

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.io import loadmat


def load_selected_trial():
    mat = loadmat(INPUT_PATH)
    trial_data = np.asarray(mat["trial_data"], dtype=float)
    time_by_trial = np.asarray(mat["relative_time"], dtype=float)
    drop = np.asarray(mat["drop"]).reshape(-1).astype(bool)
    sample_rate = int(np.asarray(mat["SampleRate"]).squeeze())

    index = TRIAL_NUMBER - 1
    if index < 0 or index >= trial_data.shape[0]:
        raise IndexError(f"Trial {TRIAL_NUMBER} is outside the available data")

    trial = trial_data[index, :3]
    time = time_by_trial[index]
    clip_masks = np.abs(trial) >= CLIP_THRESHOLD
    trigger_channels = [i for i, mask in enumerate(clip_masks) if mask.any()]
    if not drop[index] or trigger_channels != [1]:
        raise ValueError(
            "The selected example must be drop=1 with F3 as the only channel "
            "crossing the clipping threshold"
        )

    clip_indices = np.flatnonzero(clip_masks[1])
    if clip_indices.size == 0 or np.any(np.diff(clip_indices) != 1):
        raise ValueError("The selected F3 clipping region is not one contiguous run")
    return trial, time, clip_masks, clip_indices, sample_rate


def make_figure(trial, time, clip_masks, clip_indices, sample_rate):
    width, height = publication_size("double", 104.0)
    rc = paper_figure_rc_params()
    rc.update(
        {
            "font.family": "DengXian",
            "font.sans-serif": [
                "DengXian", "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    dt = 1.0 / sample_rate
    clip_start = float(time[clip_indices[0]])
    clip_end = float(time[clip_indices[-1]])
    clip_duration = float(clip_indices.size / sample_rate)

    with mpl.rc_context(rc):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(width, height),
            sharex=True,
            sharey=True,
        )
        fig.suptitle(
            f"削顶 Trial 原始波形示例：{DATASET}，Trial {TRIAL_NUMBER}",
            fontsize=10,
            y=0.985,
        )
        fig.text(
            0.5,
            0.945,
            "任一 Fz、F3、F4 采样点满足 |x| ≥ 999，即将整条 Trial 标记为削顶（drop=1）",
            ha="center",
            va="center",
            fontsize=7.5,
        )

        for channel_index, (ax, channel_name) in enumerate(zip(axes, CHANNEL_NAMES)):
            signal = trial[channel_index]
            ax.plot(
                time,
                signal,
                color=CHANNEL_COLORS[channel_name],
                linewidth=0.85,
                zorder=2,
            )
            ax.axhline(
                CLIP_THRESHOLD,
                color=THRESHOLD_COLOR,
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )
            ax.axhline(
                -CLIP_THRESHOLD,
                color=THRESHOLD_COLOR,
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )
            ax.set_title(channel_name, loc="left", pad=2)
            ax.set_ylim(-1100, 1100)
            ax.set_yticks([-1000, 0, 1000])
            ax.set_yticklabels(["-1000", "0", "1000"])
            ax.grid(axis="both", color="#D9DDE1", linewidth=0.45, alpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)

            if channel_name == "F3":
                ax.axvspan(
                    clip_start - dt / 2,
                    clip_end + dt / 2,
                    facecolor=CLIP_COLOR,
                    alpha=0.13,
                    zorder=0,
                )
                ax.scatter(
                    time[clip_masks[channel_index]],
                    signal[clip_masks[channel_index]],
                    s=4,
                    color=CLIP_COLOR,
                    linewidths=0,
                    zorder=3,
                    rasterized=True,
                )
                ax.axvline(clip_start, color=CLIP_COLOR, linewidth=0.7, alpha=0.8)
                ax.axvline(clip_end, color=CLIP_COLOR, linewidth=0.7, alpha=0.8)
                note = (
                    f"越阈点：{clip_indices.size} 个\n"
                    f"{clip_start:.3f}–{clip_end:.3f} s\n"
                    f"持续约 {clip_duration:.3f} s"
                )
                note_position = (0.5, 0.50)
                note_alignment = "center"
            else:
                max_abs = float(np.max(np.abs(signal)))
                note = f"max |x|={max_abs:.1f}，未达到 999 阈值"
                note_position = (0.99, 0.94)
                note_alignment = "right"

            ax.text(
                *note_position,
                note,
                transform=ax.transAxes,
                ha=note_alignment,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
            )

        axes[-1].set_xlim(-1.0, 3.0)
        x_ticks = np.arange(-1.0, 3.01, 0.5)
        axes[-1].set_xticks(x_ticks)
        axes[-1].set_xticklabels([f"{value:.1f}" for value in x_ticks])
        axes[-1].set_xlabel("相对提示 onset 的时间（s）")
        fig.supylabel("幅值（原始数据单位）", x=0.027, fontsize=7.5)

        legend_handles = [
            Line2D([0], [0], color="#0072B2", linewidth=1.0, label="原始 EEG"),
            Line2D(
                [0], [0], color=THRESHOLD_COLOR, linewidth=0.9, linestyle="--",
                label="削顶阈值 ±999",
            ),
            Patch(
                facecolor=CLIP_COLOR,
                edgecolor=CLIP_COLOR,
                alpha=0.18,
                label="F3 削顶采样范围",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=3,
            frameon=False,
            handlelength=2.0,
        )
        fig.subplots_adjust(left=0.105, right=0.98, bottom=0.12, top=0.86, hspace=0.28)
    return fig, rc


def main():
    trial, time, clip_masks, clip_indices, sample_rate = load_selected_trial()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, rc = make_figure(trial, time, clip_masks, clip_indices, sample_rate)

    stem = f"{DATASET}_Trial{TRIAL_NUMBER}_削顶示意图"
    with mpl.rc_context(rc):
        for extension in ("png", "pdf", "svg"):
            fig.savefig(
                OUTPUT_DIR / f"{stem}.{extension}",
                dpi=450,
                facecolor="white",
            )
    plt.close(fig)

    clip_start = float(time[clip_indices[0]])
    clip_end = float(time[clip_indices[-1]])
    metadata = {
        "claim": "A single F3 channel crossing |x| >= 999 marks the whole trial as clipped.",
        "figure_role": "representative raw-trial clipping diagnostic",
        "source_data": str(INPUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "dataset": DATASET,
        "trial_number_1based": TRIAL_NUMBER,
        "channel_order": list(CHANNEL_NAMES),
        "sample_rate_hz": sample_rate,
        "clip_threshold_absolute": CLIP_THRESHOLD,
        "triggering_channels": [CHANNEL_NAMES[i] for i, mask in enumerate(clip_masks) if mask.any()],
        "clip_sample_count_f3": int(clip_indices.size),
        "clip_start_sample_time_s": clip_start,
        "clip_end_sample_time_s": clip_end,
        "clip_sample_span_s": float(clip_indices.size / sample_rate),
        "drop": 1,
        "filtering_or_resampling_for_figure": "none; raw sliced trial before filtering/downsampling",
        "outputs": [f"{stem}.png", f"{stem}.pdf", f"{stem}.svg"],
    }
    (OUTPUT_DIR / f"{stem}.figure.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote figure outputs to {OUTPUT_DIR}")
    print(
        f"{DATASET} Trial {TRIAL_NUMBER}: F3 clipped at {clip_indices.size} samples, "
        f"t={clip_start:.6f}..{clip_end:.6f}s; drop=1"
    )


if __name__ == "__main__":
    main()
