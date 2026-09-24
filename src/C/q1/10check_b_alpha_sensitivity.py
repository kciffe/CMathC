"""Check B-group SQI-threshold sensitivity without changing formal outputs."""

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


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "output" / "7filter_downsample"
SQI_DIR = PROJECT_DIR / "output" / "8riemann_denoise"
RESULT_DIR = PROJECT_DIR / "output" / "10check_b_alpha_sensitivity"
DATASETS = ("VisualCogB_Task-1", "VisualCogB_Task-2")
ALPHAS = (0.6, 0.7, 0.8, 1.0)
CONDITIONS = ((-1, "左条件"), (1, "右条件"))
CHANNELS = ((1, "F3"), (0, "Fz"), (2, "F4"))
BASELINE_WINDOW = (-0.2, 0.0)
CANDIDATE_WINDOWS = (
    ("cue", 0.25, 0.50),
    ("target", 2.45, 2.70),
)
TARGET_ONSET_S = 2.2
P300_CUE_COLOR = "#AFC6E9"
P300_TARGET_COLOR = "#C8B6E2"
ERP_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@lru_cache(maxsize=1)
def load_sqi_knee_function():
    """Use the production SQI knee implementation, not a copied rule."""
    source_path = PROJECT_DIR / "8riemann_denoise.py"
    spec = importlib.util.spec_from_file_location("q1_riemann_denoise", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the SQI threshold implementation from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sqi_knee_threshold


def threshold_for_alpha(knee, alpha):
    """Apply the production B-group threshold floor to one sensitivity value."""
    knee = float(knee)
    alpha = float(alpha)
    if not np.isfinite(knee) or knee < 0:
        raise ValueError("knee must be a finite non-negative value")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive value")
    return max(0.05, alpha * knee)


def retained_mask_for_threshold(sqi_values, clipped_drop, threshold):
    """Apply only the SQI cut while keeping the pre-existing clip mask fixed."""
    sqi = np.asarray(sqi_values, dtype=float).reshape(-1)
    clipped = np.asarray(clipped_drop, dtype=bool).reshape(-1)
    if sqi.shape != clipped.shape:
        raise ValueError("SQI and clipping arrays must have equal lengths")
    if not np.isfinite(sqi[~clipped]).all():
        raise ValueError("Unclipped trials must have finite SQI values")
    return (~clipped) & (sqi >= float(threshold))


def validate_formal_alpha06(sqi_values, clipped_drop, formal_final_drop, threshold):
    """Return the retained mask only when alpha=.6 exactly matches formal output."""
    sqi = np.asarray(sqi_values, dtype=float).reshape(-1)
    clipped = np.asarray(clipped_drop, dtype=bool).reshape(-1)
    formal = np.asarray(formal_final_drop, dtype=bool).reshape(-1)
    if sqi.shape != clipped.shape or sqi.shape != formal.shape:
        raise ValueError("SQI, clipping, and formal drop arrays must have equal lengths")
    if not np.isfinite(sqi[~clipped]).all():
        raise ValueError("Unclipped trials must have finite SQI values")

    keep = retained_mask_for_threshold(sqi, clipped, threshold)
    sqi_drop = (~clipped) & ~keep
    expected_final_drop = clipped | sqi_drop
    if not np.array_equal(expected_final_drop, formal):
        mismatch = np.flatnonzero(expected_final_drop != formal).tolist()
        raise ValueError(
            "alpha=0.6 does not exactly match the formal FinalDrop mask; "
            f"mismatched Trial indices: {mismatch}"
        )
    return keep


def baseline_correct_trials(trials, relative_time, baseline_window=BASELINE_WINDOW):
    """Subtract each trial/channel's mean over the half-open baseline interval."""
    data = np.asarray(trials, dtype=float)
    time = np.asarray(relative_time, dtype=float)
    if data.ndim != 3:
        raise ValueError("trials must have shape (trials, channels, samples)")
    if time.ndim == 1:
        if time.size != data.shape[2]:
            raise ValueError("relative_time length does not match trial samples")
        time_by_trial = np.broadcast_to(time, (data.shape[0], data.shape[2]))
    elif time.ndim == 2 and time.shape == (data.shape[0], data.shape[2]):
        time_by_trial = time
    else:
        raise ValueError("relative_time must have shape (samples,) or (trials, samples)")

    start, end = map(float, baseline_window)
    if not np.isfinite((start, end)).all() or start >= end:
        raise ValueError("baseline_window must have finite start < end")
    corrected = data.copy()
    for trial_index, trial_time in enumerate(time_by_trial):
        baseline_mask = (trial_time >= start) & (trial_time < end)
        if not baseline_mask.any():
            raise ValueError(f"Trial {trial_index} has no samples in the baseline window")
        baseline = data[trial_index][:, baseline_mask].mean(axis=-1, keepdims=True)
        corrected[trial_index] = data[trial_index] - baseline
    return corrected


def average_by_condition(corrected_trials, cue_type, keep_mask):
    """Average retained, baseline-corrected trials separately by left/right cue."""
    data = np.asarray(corrected_trials, dtype=float)
    cues = np.asarray(cue_type).reshape(-1)
    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if data.ndim != 3 or data.shape[0] != cues.size or cues.size != keep.size:
        raise ValueError("trial, cue_type, and keep_mask lengths must match")
    erps = {}
    for condition, _ in CONDITIONS:
        selected = keep & (cues == condition)
        if not selected.any():
            raise ValueError(f"No retained trials remain for condition {condition}")
        erps[condition] = data[selected].mean(axis=0)
    return erps


def extract_candidate_peaks(fz_erp, relative_time):
    """Extract max-positive Fz peaks in the two predefined candidate windows."""
    waveform = np.asarray(fz_erp, dtype=float).reshape(-1)
    time = np.asarray(relative_time, dtype=float).reshape(-1)
    if waveform.size != time.size:
        raise ValueError("Fz ERP and relative_time must have equal lengths")
    metrics = {}
    for name, start, end in CANDIDATE_WINDOWS:
        window_mask = (time >= start) & (time <= end)
        if not window_mask.any():
            raise ValueError(f"No Fz ERP samples in {name} candidate window [{start}, {end}] s")
        sample_indices = np.flatnonzero(window_mask)
        peak_index = sample_indices[np.argmax(waveform[window_mask])]
        peak_amplitude = float(waveform[peak_index])
        latency = float(time[peak_index])
        metrics[f"{name}_peak_amplitude"] = peak_amplitude
        metrics[f"{name}_latency_relative_to_cue_s"] = latency
        if name == "target":
            metrics["target_latency_relative_to_target_s"] = latency - TARGET_ONSET_S
    return metrics


def load_dataset(dataset_name):
    """Load immutable filtered trials and formal SQI metadata for one B dataset."""
    mat_path = INPUT_DIR / f"{dataset_name}_filtered_downsample.mat"
    csv_path = SQI_DIR / f"{dataset_name}_SQI指标.csv"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing filtered MAT input: {mat_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing formal SQI CSV: {csv_path}")

    source = loadmat(mat_path)
    required = {"trial_data", "relative_time", "cue_type", "drop"}
    missing = required - source.keys()
    if missing:
        raise KeyError(f"{mat_path.name} is missing fields: {sorted(missing)}")
    trials = np.asarray(source["trial_data"], dtype=float)
    time = np.asarray(source["relative_time"], dtype=float)
    cue_type = np.asarray(source["cue_type"], dtype=float).reshape(-1)
    clipped = np.asarray(source["drop"], dtype=bool).reshape(-1)
    if trials.ndim != 3 or trials.shape[1] < 3:
        raise ValueError(f"Expected trials x at least 3 channels x samples, got {trials.shape}")
    if time.shape != (trials.shape[0], trials.shape[2]):
        raise ValueError(f"relative_time shape {time.shape} does not match {trials.shape}")
    if cue_type.size != trials.shape[0] or clipped.size != trials.shape[0]:
        raise ValueError("cue_type/drop length does not match trial count")
    if not np.isin(cue_type, (-1, 1)).all():
        raise ValueError("cue_type must contain only -1 (left) and +1 (right)")
    if not np.allclose(time, time[0], rtol=0, atol=1e-10):
        raise ValueError("relative_time differs across trials; ERP samples cannot be averaged directly")

    metrics = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_columns = {"Trial", "CueType", "ClippedDrop", "SQI", "RiemannDrop", "FinalDrop"}
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise ValueError(f"SQI CSV is missing columns: {sorted(missing_columns)}")
    metrics["Trial"] = pd.to_numeric(metrics["Trial"], errors="coerce")
    if metrics["Trial"].isna().any() or metrics["Trial"].duplicated().any():
        raise ValueError("SQI CSV Trial indices must be present and unique")
    metrics = metrics.set_index("Trial").reindex(np.arange(trials.shape[0]))
    if metrics["SQI"].isna().all() or metrics["FinalDrop"].isna().any():
        raise ValueError("SQI CSV does not cover every MAT trial")

    csv_cues = pd.to_numeric(metrics["CueType"], errors="coerce").to_numpy(dtype=float)
    csv_clipped = pd.to_numeric(metrics["ClippedDrop"], errors="coerce").to_numpy(dtype=float)
    riemann_drop = pd.to_numeric(metrics["RiemannDrop"], errors="coerce").to_numpy(dtype=float)
    formal_final_drop = pd.to_numeric(metrics["FinalDrop"], errors="coerce").to_numpy(dtype=float)
    if not np.array_equal(csv_cues, cue_type):
        raise ValueError("CueType values in SQI CSV do not match the MAT input")
    if not np.isfinite(csv_clipped).all() or not np.array_equal(csv_clipped.astype(bool), clipped):
        raise ValueError("ClippedDrop flags in SQI CSV do not match the MAT drop field")
    if not np.isin(riemann_drop, (0, 1)).all() or not np.isin(formal_final_drop, (0, 1)).all():
        raise ValueError("RiemannDrop and FinalDrop values must be 0 or 1")
    formal_final_drop = formal_final_drop.astype(bool)
    riemann_drop = riemann_drop.astype(bool)
    if not np.array_equal(formal_final_drop, clipped | riemann_drop):
        raise ValueError("Formal FinalDrop is not the union of clipping and Riemann drops")

    sqi = pd.to_numeric(metrics["SQI"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(sqi[~clipped]).all():
        raise ValueError("Unclipped trials must have finite SQI values")
    knee = float(load_sqi_knee_function()(sqi[~clipped]))
    formal_threshold = threshold_for_alpha(knee, 0.6)
    formal_keep = validate_formal_alpha06(sqi, clipped, formal_final_drop, formal_threshold)
    expected_riemann_drop = (~clipped) & (sqi < formal_threshold)
    if not np.array_equal(expected_riemann_drop, riemann_drop):
        mismatch = np.flatnonzero(expected_riemann_drop != riemann_drop).tolist()
        raise ValueError(
            "alpha=0.6 SQI-drop mask does not match formal RiemannDrop; "
            f"mismatched Trial indices: {mismatch}"
        )

    return {
        "name": dataset_name,
        "trials": trials,
        "time_by_trial": time,
        "time": time[0],
        "cue_type": cue_type,
        "clipped_drop": clipped,
        "sqi": sqi,
        "knee": knee,
        "formal_threshold": formal_threshold,
        "formal_keep": formal_keep,
    }


def summarize_dataset(dataset, corrected_trials):
    """Compute trial averages and Fz candidate peaks for each alpha/condition."""
    records = []
    curves = {}
    for alpha in ALPHAS:
        threshold = threshold_for_alpha(dataset["knee"], alpha)
        if alpha == 0.6:
            keep = dataset["formal_keep"].copy()
        else:
            keep = retained_mask_for_threshold(
                dataset["sqi"], dataset["clipped_drop"], threshold
            )
        erps = average_by_condition(corrected_trials, dataset["cue_type"], keep)
        curves[alpha] = erps
        for condition, condition_label in CONDITIONS:
            count = int(np.count_nonzero(keep & (dataset["cue_type"] == condition)))
            peak_metrics = extract_candidate_peaks(erps[condition][0], dataset["time"])
            records.append({
                "dataset": dataset["name"],
                "alpha": alpha,
                "sqi_threshold": threshold,
                "retained_trial_count": int(keep.sum()),
                "left_retained_trial_count": int(np.count_nonzero(keep & (dataset["cue_type"] == -1))),
                "right_retained_trial_count": int(np.count_nonzero(keep & (dataset["cue_type"] == 1))),
                "condition": condition_label,
                "condition_retained_trial_count": count,
                "cue_candidate_p300_peak_amplitude": peak_metrics["cue_peak_amplitude"],
                "cue_candidate_p300_latency_relative_to_cue_s": peak_metrics["cue_latency_relative_to_cue_s"],
                "target_candidate_p300_peak_amplitude": peak_metrics["target_peak_amplitude"],
                "target_candidate_p300_latency_relative_to_cue_s": peak_metrics["target_latency_relative_to_cue_s"],
                "target_candidate_p300_latency_relative_to_target_s": peak_metrics["target_latency_relative_to_target_s"],
            })
    return records, curves


def format_alpha_legend(alpha, keep_count):
    """Format an alpha-series legend entry using Chinese sample-count text."""
    return f"α={alpha:g}（保留 {int(keep_count)} 个试次）"


def install_external_legend(fig, source_axis, fontsize=8):
    """Place alpha and candidate-window keys in one figure-level legend."""
    handles, labels = source_axis.get_legend_handles_labels()
    window_handles = [
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
    ]
    handles.extend(window_handles)
    labels.extend(handle.get_label() for handle in window_handles)
    fig.tight_layout(rect=(0, 0, 1.0, 0.84))
    return fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        fontsize=fontsize,
        frameon=True,
        columnspacing=1.5,
    )


def plot_dataset(dataset, curves):
    """Plot the alpha-only ERP comparison; source filtering/SQI stay fixed."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="col")
    for row, (condition, condition_label) in enumerate(CONDITIONS):
        for column, (channel_index, channel_name) in enumerate(CHANNELS):
            ax = axes[row, column]
            for alpha_index, alpha in enumerate(ALPHAS):
                keep_count = next(
                    record["condition_retained_trial_count"]
                    for record in dataset["records"]
                    if record["alpha"] == alpha and record["condition"] == condition_label
                )
                ax.plot(
                    dataset["time"],
                    curves[alpha][condition][channel_index],
                    color=ERP_COLORS[alpha_index],
                    linewidth=1.25,
                    label=format_alpha_legend(alpha, keep_count),
                )
            ax.axvspan(
                *CANDIDATE_WINDOWS[0][1:],
                color=P300_CUE_COLOR,
                alpha=0.55,
                zorder=0,
            )
            ax.axvspan(
                *CANDIDATE_WINDOWS[1][1:],
                color=P300_TARGET_COLOR,
                alpha=0.55,
                zorder=0,
            )
            ax.axvline(0, color="tab:blue", linewidth=1, linestyle="--")
            ax.axvline(TARGET_ONSET_S, color="tab:blue", linewidth=1, linestyle=":")
            ax.set_title(f"{condition_label} · {channel_name}")
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(-0.5, 3.0)
            if column == 0:
                ax.set_ylabel("基线校正后 ERP（原始数据单位）")
            if row == 1:
                ax.set_xlabel("相对提示 onset 的时间 (s)")
    fig.suptitle(
        f"{dataset['name']}：B 组 SQI 阈值敏感性（仅改变 α 与保留集合）",
        fontsize=13,
        y=0.985,
    )
    install_external_legend(fig, axes[0, 0], fontsize=8)
    output_path = RESULT_DIR / f"{dataset['name']}_alpha_ERP_sensitivity.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    # Complete the strict alpha=.6 check for both datasets before producing any output.
    datasets = [load_dataset(name) for name in DATASETS]
    for dataset in datasets:
        dataset["corrected_trials"] = baseline_correct_trials(
            dataset["trials"], dataset["time_by_trial"]
        )

    all_records = []
    for dataset in datasets:
        records, curves = summarize_dataset(dataset, dataset["corrected_trials"])
        dataset["records"] = records
        dataset["curves"] = curves
        all_records.extend(records)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_DIR / "b_alpha_sensitivity_summary.csv"
    pd.DataFrame(all_records).to_csv(summary_path, index=False, encoding="utf-8-sig")
    for dataset in datasets:
        plot_path = plot_dataset(dataset, dataset["curves"])
        print(
            f"{dataset['name']}: alpha=0.6 matches formal FinalDrop exactly; "
            f"formal threshold={dataset['formal_threshold']:.6f}, "
            f"retained={int(dataset['formal_keep'].sum())}/{len(dataset['formal_keep'])}; "
            f"wrote {plot_path.name}"
        )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
