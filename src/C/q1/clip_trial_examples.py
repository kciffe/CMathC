"""Generate three raw EEG clipping examples from distinct Q1 recordings."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.io import loadmat

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_DIR / "output" / "clip_trial_examples"

# Trial numbers are one-based positions in the sequence of complete VisCue
# events, matching 6drop_clipped.py. Each example uses a different recording.
EXAMPLES = (
    {"dataset": "VisualCogA_Task-1", "trial_number": 39},
    {"dataset": "VisualCogA_Task-2", "trial_number": 33},
    {"dataset": "VisualCogB_Task-1", "trial_number": 92},
)

CHANNEL_NAMES = ("Fz", "F3", "F4")
CHANNEL_COLORS = {"Fz": "#0072B2", "F3": "#D55E00", "F4": "#009E73"}
CLIP_THRESHOLD = 999.0
PRE_TIME_S = 1.0
POST_TIME_S = 3.0
FIGURE_SIZE = (10.5, 3.55)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}


def find_source(dataset: str) -> Path:
    matches = list(DATA_DIR.rglob(f"{dataset}.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one source MAT for {dataset}, found {len(matches)}"
        )
    return matches[0]


def select_trial(dataset: str, trial_number: int) -> dict:
    source_path = find_source(dataset)
    mat = loadmat(source_path)
    data = np.asarray(mat["data"], dtype=float)
    sample_rate = int(np.asarray(mat["SampleRate"]).squeeze())
    cue = data[7]
    timestamp = data[9]

    event_indices = np.flatnonzero((cue != 0) & (np.r_[0, cue[:-1]] == 0))
    pre = int(PRE_TIME_S * sample_rate)
    post = int(POST_TIME_S * sample_rate)
    event_indices = event_indices[
        (event_indices >= pre) & (event_indices + post <= data.shape[1])
    ]

    index = trial_number - 1
    if index < 0 or index >= len(event_indices):
        raise IndexError(
            f"{dataset}: Trial {trial_number} is outside 1..{len(event_indices)}"
        )

    onset_index = int(event_indices[index])
    start_index = onset_index - pre
    end_index = onset_index + post
    trial = data[:3, start_index:end_index].copy()
    time = timestamp[start_index:end_index].copy()
    clip_masks = np.abs(trial) >= CLIP_THRESHOLD

    if not clip_masks.any():
        raise ValueError(f"{dataset} Trial {trial_number} is not clipped")

    # Require each displayed clipped run to be fully contained in the trial.
    for channel_mask in clip_masks:
        clipped_indices = np.flatnonzero(channel_mask)
        if clipped_indices.size and (
            clipped_indices[0] == 0 or clipped_indices[-1] == channel_mask.size - 1
        ):
            raise ValueError(
                f"{dataset} Trial {trial_number} has clipping at the display edge"
            )

    condition = "左刺激" if cue[onset_index] < 0 else "右刺激"
    condition_ordinal = int(np.count_nonzero(cue[event_indices[: index + 1]] == cue[onset_index]))
    relative_time = time - float(timestamp[onset_index])

    return {
        "dataset": dataset,
        "trial_number": trial_number,
        "condition": condition,
        "condition_ordinal": condition_ordinal,
        "sample_rate": sample_rate,
        "source_path": source_path,
        "trial": trial,
        "time": time,
        "relative_time": relative_time,
        "cue_timestamp": float(timestamp[onset_index]),
        "clip_masks": clip_masks,
    }


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    starts = indices[np.r_[True, np.diff(indices) > 1]]
    ends = indices[np.r_[np.diff(indices) > 1, True]]
    return list(zip(starts.tolist(), ends.tolist()))


def plot_example(example: dict) -> plt.Figure:
    trial = example["trial"]
    time = example["time"]
    relative_time = example["relative_time"]
    clip_masks = example["clip_masks"]
    sample_rate = example["sample_rate"]
    any_clip = clip_masks.any(axis=0)
    clip_runs = contiguous_runs(any_clip)

    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.suptitle(
            f"{example['dataset']}  |  Trial {example['trial_number']}（第 {example['condition_ordinal']} 个{example['condition']}）",
            y=0.99,
            fontsize=10,
        )
        for start, end in clip_runs:
            ax.axvspan(
                time[start] - 0.5 / sample_rate,
                time[end] + 0.5 / sample_rate,
                color="#E69F00",
                alpha=0.17,
                zorder=0,
            )

        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            color = CHANNEL_COLORS[channel_name]
            ax.plot(
                time,
                trial[channel_index],
                color=color,
                linewidth=1.15,
                label=channel_name,
                zorder=2,
            )
            hit = clip_masks[channel_index]
            if hit.any():
                ax.scatter(
                    time[hit],
                    trial[channel_index, hit],
                    color="#7A3E00",
                    s=7,
                    linewidths=0,
                    zorder=4,
                )

        ax.axhline(
            CLIP_THRESHOLD,
            color="#8A6D00",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )
        ax.axhline(
            -CLIP_THRESHOLD,
            color="#8A6D00",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )
        ax.axvline(
            example["cue_timestamp"],
            color="#0072B2",
            linestyle=":",
            linewidth=1.0,
            zorder=1,
        )
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-1120, 1120)
        ax.set_yticks([-1000, -500, 0, 500, 1000])
        tick_start = np.ceil(time[0] * 2) / 2
        tick_end = np.floor(time[-1] * 2) / 2
        ax.set_xticks(np.arange(tick_start, tick_end + 0.25, 0.5))
        ax.set_xlabel("TimeStamp (s)")
        ax.set_ylabel("EEG 幅值（原始数据单位）")
        ax.grid(True, color="#D9DDE1", linewidth=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

        handles = [
            Line2D([0], [0], color=CHANNEL_COLORS[name], linewidth=1.3, label=name)
            for name in CHANNEL_NAMES
        ]
        handles.extend(
            [
                Line2D([0], [0], color="#8A6D00", linestyle="--", linewidth=0.9, label="削顶阈值 ±999"),
                Patch(facecolor="#E69F00", edgecolor="none", alpha=0.17, label="至少一个通道越阈区间"),
                Line2D([0], [0], color="#0072B2", linestyle=":", linewidth=1, label="VisCue onset"),
            ]
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.86),
            ncol=6,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.25,
        )
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.77)

    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in EXAMPLES:
        example = select_trial(item["dataset"], item["trial_number"])
        fig = plot_example(example)
        stem = f"{item['dataset']}_Trial{item['trial_number']}_削顶异常"

        with mpl.rc_context(RC):
            fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=450, facecolor="white")
        plt.close(fig)
        print(
            f"{item['dataset']} Trial {item['trial_number']} "
            f"({example['condition']}，第{example['condition_ordinal']}个) -> "
            f"{OUTPUT_DIR / (stem + '.png')}"
        )


if __name__ == "__main__":
    main()
