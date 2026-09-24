"""Plot SQI-boundary EEG trials without changing the denoising outputs."""

import importlib.util
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.io import loadmat


DATASETS = (
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
)
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "output" / "7filter_downsample"
SQI_DIR = PROJECT_DIR / "output" / "8riemann_denoise"
RESULT_DIR = PROJECT_DIR / "output" / "9check_riemann_drop"

TRIAL_LIMIT = 5
WINDOW_START, WINDOW_END = -0.2, 3.0
CUE_P300_WINDOW = (0.25, 0.50)
TARGET_ONSET = 2.20
TARGET_P300_WINDOW = (2.45, 2.70)
P300_CUE_COLOR = "#AFC6E9"
P300_TARGET_COLOR = "#C8B6E2"
CHANNELS = (("F3", 1), ("Fz", 0), ("F4", 2))
TRIAL_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
GROUPS = (
    ("lowest_sqi", "SQI 最低试次"),
    ("closest_below", "阈值下方最近试次"),
    ("closest_above", "阈值侧最近试次（含阈值）"),
)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


@lru_cache(maxsize=1)
def load_sqi_knee_function():
    """Load the production threshold helper rather than duplicating its rule."""
    source_path = PROJECT_DIR / "8riemann_denoise.py"
    spec = importlib.util.spec_from_file_location("q1_riemann_denoise", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the SQI threshold implementation from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sqi_knee_threshold


def calculate_threshold(sqi_values, is_group_a):
    """Reproduce the threshold used by the denoising step for one dataset."""
    values = np.asarray(sqi_values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("SQI threshold input must contain only finite values")
    knee = float(load_sqi_knee_function()(values))
    return knee if is_group_a else max(0.05, 0.6 * knee)


def select_trial_groups(metrics, threshold, limit=TRIAL_LIMIT):
    """Select up to ``limit`` unclipped trials for each SQI diagnostic group."""
    required = {"Trial", "ClippedDrop", "SQI", "RiemannDrop"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"SQI CSV is missing required columns: {sorted(missing)}")
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    table = metrics.loc[:, sorted(required)].copy()
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    if table["Trial"].isna().any() or table["Trial"].duplicated().any():
        raise ValueError("Trial values in the SQI CSV must be present and unique")
    if not np.equal(table["Trial"], np.floor(table["Trial"])).all():
        raise ValueError("Trial values in the SQI CSV must be integers")
    for column in ("ClippedDrop", "RiemannDrop"):
        if table[column].isna().any() or not table[column].isin((0, 1)).all():
            raise ValueError(f"{column} values in the SQI CSV must be 0 or 1")

    valid = table.loc[(table["ClippedDrop"] == 0) & table["SQI"].notna()].copy()
    if valid.empty:
        raise ValueError("No unclipped trials with an SQI value are available")
    if not np.isfinite(valid["SQI"]).all():
        raise ValueError("Unclipped trials must have finite SQI values")

    rejected_mask = valid["SQI"].to_numpy() < threshold
    recorded_drop_mask = valid["RiemannDrop"].to_numpy() == 1
    if not np.array_equal(rejected_mask, recorded_drop_mask):
        raise ValueError(
            "SQI values and RiemannDrop flags do not match the production threshold"
        )
    rejected = valid.loc[rejected_mask]
    retained = valid.loc[~rejected_mask]

    groups = {
        "lowest_sqi": valid.sort_values(["SQI", "Trial"]).head(limit),
        "closest_below": rejected.sort_values(
            ["SQI", "Trial"], ascending=[False, True]
        ).head(limit),
        "closest_above": retained.sort_values(["SQI", "Trial"]).head(limit),
    }
    return {name: rows.reset_index(drop=True) for name, rows in groups.items()}, threshold


def load_dataset(dataset_name):
    mat_path = INPUT_DIR / f"{dataset_name}_filtered_downsample.mat"
    csv_path = SQI_DIR / f"{dataset_name}_SQI指标.csv"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing filtered MAT input: {mat_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing SQI CSV input: {csv_path}")

    source = loadmat(mat_path)
    required = {"trial_data", "relative_time", "drop"}
    missing = required - source.keys()
    if missing:
        raise KeyError(f"{mat_path.name} is missing fields: {sorted(missing)}")

    trials = np.asarray(source["trial_data"], dtype=float)
    relative_time = np.asarray(source["relative_time"], dtype=float)
    clipped_drop = np.asarray(source["drop"]).reshape(-1).astype(bool)
    metrics = pd.read_csv(csv_path, encoding="utf-8-sig")

    if trials.ndim != 3 or trials.shape[1] < 3:
        raise ValueError(f"Expected trials x at least 3 channels x samples, got {trials.shape}")
    if relative_time.shape != (trials.shape[0], trials.shape[2]):
        raise ValueError(
            f"relative_time shape {relative_time.shape} does not match trial data {trials.shape}"
        )
    if clipped_drop.size != trials.shape[0]:
        raise ValueError(f"drop length does not match trial count in {mat_path.name}")
    if len(metrics) != trials.shape[0]:
        raise ValueError(
            f"SQI row count {len(metrics)} does not match trial count {trials.shape[0]}"
        )

    required_csv_columns = {"Trial", "ClippedDrop", "SQI", "RiemannDrop"}
    missing_csv_columns = required_csv_columns - set(metrics.columns)
    if missing_csv_columns:
        raise ValueError(f"SQI CSV is missing required columns: {sorted(missing_csv_columns)}")
    metrics["Trial"] = pd.to_numeric(metrics["Trial"], errors="coerce")
    metrics_by_trial = metrics.set_index("Trial")
    expected_trials = set(range(trials.shape[0]))
    if set(metrics_by_trial.index) != expected_trials:
        raise ValueError("SQI CSV Trial indices do not match the MAT trial indices")
    csv_clipped = pd.to_numeric(metrics_by_trial["ClippedDrop"], errors="coerce")
    if csv_clipped.isna().any() or not np.array_equal(
        csv_clipped.sort_index().to_numpy(dtype=np.uint8), clipped_drop.astype(np.uint8)
    ):
        raise ValueError("ClippedDrop flags in SQI CSV do not match the MAT drop field")

    sqi_values = pd.to_numeric(metrics["SQI"], errors="coerce")
    valid_sqi = sqi_values.loc[(csv_clipped.to_numpy() == 0) & sqi_values.notna()]
    threshold = calculate_threshold(
        valid_sqi.to_numpy(dtype=float), is_group_a=dataset_name.startswith("VisualCogA")
    )
    groups, threshold = select_trial_groups(metrics, threshold)

    return trials, relative_time, groups, threshold, mat_path, csv_path


def format_trial_sqi_legend(trial_index, sqi):
    """Format a trial legend entry with Chinese descriptors."""
    return f"第 {int(trial_index)} 个试次 · 信号质量指数 {float(sqi):.3f}"


def plot_dataset(dataset_name, trials, relative_time, groups, threshold):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey="col")
    row_legends = []

    for row, (group_name, group_label) in enumerate(GROUPS):
        selected = groups[group_name]
        row_label = f"{group_label}（{len(selected)}/{TRIAL_LIMIT}）"
        for column, (channel_name, channel_index) in enumerate(CHANNELS):
            ax = axes[row, column]
            for trial_position, (_, trial) in enumerate(selected.iterrows()):
                trial_index = int(trial["Trial"])
                time = relative_time[trial_index]
                window = (time >= WINDOW_START) & (time <= WINDOW_END)
                if not window.any():
                    raise ValueError(
                        f"Trial {trial_index} has no samples in "
                        f"{WINDOW_START}–{WINDOW_END} s"
                    )
                ax.plot(
                    time[window],
                    trials[trial_index, channel_index, window],
                    color=TRIAL_COLORS[trial_position],
                    linewidth=0.9,
                    alpha=0.85,
                    label=format_trial_sqi_legend(trial_index, trial["SQI"]),
                )

            ax.axvline(0, color="tab:blue", linewidth=1, linestyle="--")
            ax.axvline(TARGET_ONSET, color="tab:blue", linewidth=1, linestyle=":")
            ax.axvspan(
                *CUE_P300_WINDOW,
                color=P300_CUE_COLOR,
                alpha=0.55,
                zorder=0,
            )
            ax.axvspan(
                *TARGET_P300_WINDOW,
                color=P300_TARGET_COLOR,
                alpha=0.55,
                zorder=0,
            )
            ax.set_title(channel_name)
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(True, alpha=0.2)
            if column == 0:
                ax.set_ylabel(f"{row_label}\nEEG 振幅（原始数据单位）")
                if selected.empty:
                    row_legends.append(None)
                else:
                    row_legends.append(ax.get_legend_handles_labels())
            if row == len(GROUPS) - 1:
                ax.set_xlabel("时间 (s)")
            ax.set_xlim(WINDOW_START, WINDOW_END)

            if selected.empty:
                ax.text(
                    0.5, 0.5, "无候选 Trial", transform=ax.transAxes,
                    ha="center", va="center", color="#666666",
                )

    fig.suptitle(
        f"{dataset_name}：SQI 阈值边界 Trial 对比（阈值 = {threshold:.6f}）",
        fontsize=14,
        y=0.985,
    )
    fig.tight_layout(rect=(0.04, 0, 0.81, 0.90))
    fig.legend(
        handles=[
            Patch(
                facecolor=P300_CUE_COLOR,
                edgecolor="none",
                alpha=0.55,
                label="提示后 P300 候选时窗（250–500 ms）",
            ),
            Patch(
                facecolor=P300_TARGET_COLOR,
                edgecolor="none",
                alpha=0.55,
                label="目标后 P300 候选时窗（250–500 ms）",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.42, 0.945),
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    for row, legend in enumerate(row_legends):
        if legend is None:
            continue
        handles, labels = legend
        axes_position = axes[row, 2].get_position()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.83, (axes_position.y0 + axes_position.y1) / 2),
            fontsize=8,
            frameon=False,
        )
    output_path = RESULT_DIR / f"{dataset_name}_SQI边界Trial对比.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def process_dataset(dataset_name):
    trials, relative_time, groups, threshold, mat_path, csv_path = load_dataset(dataset_name)
    output_path = plot_dataset(dataset_name, trials, relative_time, groups, threshold)

    print(
        f"{dataset_name}: threshold={threshold:.6f}; "
        "sources=output/7filter_downsample MAT + output/8riemann_denoise SQI CSV"
    )
    for group_name, _ in GROUPS:
        selected = groups[group_name]
        candidates = ", ".join(
            f"Trial {int(row.Trial)} (SQI={row.SQI:.4f})"
            for row in selected.itertuples(index=False)
        ) or "none"
        print(f"  {group_name} [{len(selected)}/{TRIAL_LIMIT}]: {candidates}")
    print(f"  figure saved under output/9check_riemann_drop")


def main():
    for dataset_name in DATASETS:
        process_dataset(dataset_name)


if __name__ == "__main__":
    main()
