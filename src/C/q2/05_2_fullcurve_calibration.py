import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LGN_ROOT = SCRIPT_DIR / "output" / "02_lgn"
SIM_ROOT = SCRIPT_DIR / "output" / "04_simulated_eeg"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "05_2_fullcurve_calibration"
SCALE_GRID = np.round(np.arange(0.5, 2.5001, 0.05), 2)
REFERENCE_SCALE = 1.5
MIN_TRIALS = 10
TIME_MS = np.arange(801, dtype=float)
RECORD_NAMES = {
    "VisualCogA_Task-1": "A1",
    "VisualCogA_Task-2": "A2",
    "VisualCogB_Task-1": "B1",
    "VisualCogB_Task-2": "B2",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
})


def load_script_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load_script_module("compare05", SCRIPT_DIR / "05_simulated_real_eeg_compare.py")
cortex = load_script_module("cortex03", SCRIPT_DIR / "03_cortex.py")
eeg_map = load_script_module("eeg04", SCRIPT_DIR / "04_simulated_eeg.py")


def load_real_cases():
    cases = {}
    task_by_dataset = {
        dataset: task
        for task, datasets in compare.DATASETS.items()
        for dataset in datasets
    }
    for dataset, task in task_by_dataset.items():
        data = compare.load_dataset(dataset)
        masks, metadata = compare.stage_conditions(task, "Stage1", data)
        for stage in ("Stage1", "Stage2"):
            if stage == "Stage1":
                onset_s = 0.0
                baseline_window_s = (-0.2, 0.0)
                conditions = ("left", "right")
            else:
                masks, metadata = compare.stage_conditions(task, stage, data)
                onset_s = compare.TARGET_ONSET_S
                baseline_window_s = (2.0, compare.TARGET_ONSET_S)
                conditions = ("dots",) if task == "Task1" else ("inward", "outward")

            corrected = compare.baseline_correct(
                data["eeg"], data["time_s"], baseline_window_s
            )
            relative_ms = (data["time_s"] - onset_s) * 1000.0
            valid = (relative_ms >= 0.0) & (relative_ms <= 800.0)
            time_ms = relative_ms[valid]
            # The calibration objective uses the complete simulated interval.
            # The late peak window remains stage-specific as a separate check.
            compare_window = (0.0, 800.0)
            late_window = compare.STAGE_WINDOWS_MS[stage]["late"]

            for condition in conditions:
                mask = masks[condition]
                n_trials = int(mask.sum())
                if n_trials < MIN_TRIALS:
                    continue
                real = np.nanmean(corrected[mask][:, :, valid], axis=0)
                cases[(dataset, stage, condition)] = {
                    "dataset": dataset,
                    "task": task,
                    "stage": stage,
                    "condition": condition,
                    "n_trials": n_trials,
                    "time_ms": time_ms,
                    "real": real,
                    "compare_window": compare_window,
                    "late_window": late_window,
                }
    return cases


def model_inputs():
    inputs = {}
    for task in ("Task1", "Task2"):
        for stage in ("Stage1", "Stage2"):
            if stage == "Stage1":
                conditions = ("left", "right")
            else:
                conditions = ("dots",) if task == "Task1" else ("inward", "outward")
            for condition in conditions:
                path = LGN_ROOT / task / stage / f"{condition}_lgn_total.npy"
                inputs[(task, stage, condition)] = np.load(path)
    return inputs


def generate_predictions(inputs):
    predictions = {}
    original_scale = cortex.ASSOCIATION_TIME_SCALE
    try:
        for scale in SCALE_GRID:
            cortex.ASSOCIATION_TIME_SCALE = float(scale)
            for key, lgn in inputs.items():
                result = cortex.simulate_cortex(lgn)
                source = eeg_map.build_sources(
                    result["v1_source"],
                    result["it_source"],
                    result["pfc_source"],
                )
                predictions[(float(scale),) + key] = eeg_map.map_to_eeg(source)
    finally:
        cortex.ASSOCIATION_TIME_SCALE = original_scale
    return predictions


def sample_on_real_grid(signal, time_ms):
    return np.vstack(
        [np.interp(time_ms, TIME_MS, signal[channel]) for channel in range(3)]
    )


def score_case(signal, case):
    time_ms = case["time_ms"]
    sim = sample_on_real_grid(signal, time_ms)
    real = case["real"]
    low, high = case["compare_window"]
    compare_mask = (time_ms >= low) & (time_ms <= high)
    channel_r = [
        compare.pearson_correlation(sim[i, compare_mask], real[i, compare_mask])
        for i in range(3)
    ]
    late_low, late_high = case["late_window"]
    late_mask = (time_ms >= late_low) & (time_ms <= late_high)
    sim_peak, sim_peak_ms, sim_edge = compare.positive_late_peak(
        sim[1], time_ms, case["late_window"]
    )
    real_peak, real_peak_ms, real_edge = compare.positive_late_peak(
        real[1], time_ms, case["late_window"]
    )
    sim_scale = float(np.max(np.abs(sim)))
    real_scale = float(np.max(np.abs(real)))
    if sim_scale > 0 and real_scale > 0 and np.any(late_mask):
        sim_topo = sim[:, late_mask].mean(axis=1) / sim_scale
        real_topo = real[:, late_mask].mean(axis=1) / real_scale
        denom = np.linalg.norm(sim_topo) * np.linalg.norm(real_topo)
        topo_cosine = float(np.dot(sim_topo, real_topo) / denom) if denom else float("nan")
    else:
        topo_cosine = float("nan")
    return {
        "mean_full_curve_r": float(np.mean(channel_r)),
        "f3_full_curve_r": channel_r[0],
        "fz_full_curve_r": channel_r[1],
        "f4_full_curve_r": channel_r[2],
        "fz_sim_late_peak_ms": sim_peak_ms,
        "fz_real_late_peak_ms": real_peak_ms,
        "fz_peak_abs_error_ms": abs(sim_peak_ms - real_peak_ms),
        "fz_sim_peak_at_window_edge": sim_edge,
        "fz_real_peak_at_window_edge": real_edge,
        "fz_sim_late_peak": sim_peak,
        "fz_real_late_peak": real_peak,
        "late_topography_cosine": topo_cosine,
    }


def record_stage_scores(cases, score_rows, dataset, scale):
    result = {}
    for stage in ("Stage1", "Stage2"):
        rows = [
            row
            for row in score_rows
            if row["dataset"] == dataset
            and row["stage"] == stage
            and row["association_time_scale"] == scale
        ]
        if rows:
            result[stage] = float(np.mean([row["mean_full_curve_r"] for row in rows]))
    return result


def fit_leave_one_record_out(cases, predictions):
    datasets = list(compare.DATASETS["Task1"] + compare.DATASETS["Task2"])
    case_scores = []
    for (dataset, stage, condition), case in cases.items():
        task = case["task"]
        for scale in SCALE_GRID:
            signal = predictions[(float(scale), task, stage, condition)]
            metrics = score_case(signal, case)
            case_scores.append({
                "dataset": dataset,
                "task": task,
                "stage": stage,
                "condition": condition,
                "n_trials": case["n_trials"],
                "association_time_scale": float(scale),
                **metrics,
            })

    fold_rows = []
    sweep_rows = []
    for heldout in datasets:
        training = [dataset for dataset in datasets if dataset != heldout]
        objectives = {}
        boundary_counts = {}
        for scale in SCALE_GRID:
            record_scores = []
            training_cases = [row for row in case_scores if row["dataset"] in training
                              and row["association_time_scale"] == float(scale)]
            for dataset in training:
                stage_scores = record_stage_scores(cases, case_scores, dataset, float(scale))
                if stage_scores:
                    record_scores.append(float(np.mean(list(stage_scores.values()))))
            objectives[float(scale)] = float(np.mean(record_scores))
            boundary_counts[float(scale)] = sum(
                row["fz_sim_peak_at_window_edge"] for row in training_cases
            )
            sweep_rows.append({
                "heldout_record": heldout,
                "scale": float(scale),
                "training_full_curve_r": objectives[float(scale)],
                "training_sim_peak_boundary_cases": boundary_counts[float(scale)],
                "selected_unconstrained": False,
                "selected_boundary_constrained": False,
                "reference_scale": REFERENCE_SCALE,
            })

        unconstrained_scale = max(
            SCALE_GRID,
            key=lambda value: (
                objectives[float(value)],
                -abs(float(value) - REFERENCE_SCALE),
                -float(value),
            ),
        )
        valid_scales = [scale for scale in SCALE_GRID if boundary_counts[float(scale)] == 0]
        constrained_scale = max(
            valid_scales,
            key=lambda value: (
                objectives[float(value)],
                -abs(float(value) - REFERENCE_SCALE),
                -float(value),
            ),
        )
        for row in sweep_rows:
            if row["heldout_record"] == heldout and row["scale"] == float(unconstrained_scale):
                row["selected_unconstrained"] = True
            if row["heldout_record"] == heldout and row["scale"] == float(constrained_scale):
                row["selected_boundary_constrained"] = True

        for stage in ("Stage1", "Stage2"):
            for scale, role in (
                (REFERENCE_SCALE, "fixed_baseline"),
                (float(unconstrained_scale), "unconstrained_fullcurve"),
                (float(constrained_scale), "boundary_constrained_fullcurve"),
            ):
                rows = [
                    row for row in case_scores
                    if row["dataset"] == heldout
                    and row["stage"] == stage
                    and row["association_time_scale"] == scale
                ]
                if not rows:
                    continue
                fold_rows.append({
                    "heldout_record": heldout,
                    "record_code": RECORD_NAMES[heldout],
                    "stage": stage,
                    "fit_role": role,
                    "association_time_scale": float(scale),
                    "cv_unconstrained_scale": float(unconstrained_scale),
                    "cv_boundary_constrained_scale": float(constrained_scale),
                    "training_objective_at_scale": objectives[float(scale)],
                    "training_sim_peak_boundary_cases": boundary_counts[float(scale)],
                    "n_conditions": len(rows),
                    "n_trials_total": sum(row["n_trials"] for row in rows),
                    "mean_full_curve_r": float(np.mean([row["mean_full_curve_r"] for row in rows])),
                    "f3_mean_r": float(np.mean([row["f3_full_curve_r"] for row in rows])),
                    "fz_mean_r": float(np.mean([row["fz_full_curve_r"] for row in rows])),
                    "f4_mean_r": float(np.mean([row["f4_full_curve_r"] for row in rows])),
                    "fz_peak_abs_error_ms": float(np.mean([row["fz_peak_abs_error_ms"] for row in rows])),
                    "fz_peak_error_uncensored_n": sum(
                        not row["fz_sim_peak_at_window_edge"] and not row["fz_real_peak_at_window_edge"]
                        for row in rows
                    ),
                    "late_topography_cosine": float(np.nanmean([row["late_topography_cosine"] for row in rows])),
                })

    heldout_case_rows = []
    for heldout in datasets:
        selected = [row for row in sweep_rows if row["heldout_record"] == heldout]
        unconstrained = next(row["scale"] for row in selected if row["selected_unconstrained"])
        constrained = next(row["scale"] for row in selected if row["selected_boundary_constrained"])
        for (dataset, stage, condition), _case in cases.items():
            if dataset != heldout:
                continue
            for scale, role in (
                (REFERENCE_SCALE, "fixed_baseline"),
                (unconstrained, "unconstrained_fullcurve"),
                (constrained, "boundary_constrained_fullcurve"),
            ):
                row = next(
                    item for item in case_scores
                    if item["dataset"] == dataset and item["stage"] == stage
                    and item["condition"] == condition
                    and item["association_time_scale"] == float(scale)
                ).copy()
                row["fit_role"] = role
                row["cv_unconstrained_scale"] = float(unconstrained)
                row["cv_boundary_constrained_scale"] = float(constrained)
                heldout_case_rows.append(row)
    return heldout_case_rows, fold_rows, sweep_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(fold_rows, sweep_rows):
    fig = plt.figure(figsize=(7.2, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    scales = SCALE_GRID
    for record in RECORD_NAMES:
        values = [
            next(row["training_full_curve_r"] for row in sweep_rows
                 if row["heldout_record"] == record and row["scale"] == float(scale))
            for scale in scales
        ]
        axes[0].plot(scales, values, color="#9aa3ad", alpha=0.55, linewidth=0.9)
    mean_objective = [
        np.mean([row["training_full_curve_r"] for row in sweep_rows if row["scale"] == float(scale)])
        for scale in scales
    ]
    axes[0].plot(scales, mean_objective, color="#087e8b", linewidth=2.0, label="留一记录训练目标均值")
    axes[0].axvline(REFERENCE_SCALE, color="#555555", linestyle="--", linewidth=1, label="当前基线 1.5")
    for record in RECORD_NAMES:
        for flag, color, marker in (("selected_boundary_constrained", "#087e8b", "o"),
                                    ("selected_unconstrained", "#d95f02", "x")):
            selected_row = next(row for row in sweep_rows
                                if row["heldout_record"] == record and row[flag])
            axes[0].scatter(selected_row["scale"], selected_row["training_full_curve_r"],
                            color=color, marker=marker, s=24, zorder=3)
    axes[0].set(xlabel="IT/PFC 时间尺度倍数", ylabel="平均 r", title="A  训练折参数扫描（圆点：峰值在窗内；叉号：无约束）")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=7, frameon=False)

    x = np.arange(len(RECORD_NAMES))
    for ax, stage in zip(axes[1:], ("Stage1", "Stage2")):
        base = []
        calibrated = []
        for record in RECORD_NAMES:
            base.append(next(row["mean_full_curve_r"] for row in fold_rows
                             if row["heldout_record"] == record and row["stage"] == stage
                             and row["fit_role"] == "fixed_baseline"))
            calibrated.append(next(row["mean_full_curve_r"] for row in fold_rows
                                   if row["heldout_record"] == record and row["stage"] == stage
                                   and row["fit_role"] == "boundary_constrained_fullcurve"))
        for i, (before, after) in enumerate(zip(base, calibrated)):
            ax.plot([i - 0.07, i + 0.07], [before, after], color="#aab2bd", linewidth=1)
        ax.scatter(x - 0.07, base, color="#667085", s=22, label="固定基线")
        ax.scatter(x + 0.07, calibrated, color="#087e8b", s=24, label="留一记录全波形校准（峰值未截断）")
        ax.set_xticks(x, [RECORD_NAMES[name] for name in RECORD_NAMES])
        ax.set(title=f"{stage} 留出记录", ylabel="三通道全波形平均 r", ylim=(-0.5, 1.0))
        ax.grid(axis="y", alpha=0.22)
        ax.set_xlabel("MAT 记录")
    axes[1].legend(fontsize=7, frameon=False)
    fig.savefig(OUTPUT_ROOT / "全波形留一记录比较.png", dpi=450)
    fig.savefig(OUTPUT_ROOT / "全波形留一记录比较.pdf")
    plt.close(fig)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = load_real_cases()
    inputs = model_inputs()
    predictions = generate_predictions(inputs)

    # The in-memory scale-1.5 simulation must reproduce the existing 04 baseline.
    baseline_differences = []
    for (task, stage, condition), signal in inputs.items():
        rerun = predictions[(REFERENCE_SCALE, task, stage, condition)]
        existing = np.load(SIM_ROOT / task / stage / f"{condition}_eeg.npy")
        baseline_differences.append(float(np.max(np.abs(rerun - existing))))
    if max(baseline_differences) > 1e-5:
        raise RuntimeError(f"The scale-1.5 rerun differs from the saved 04 baseline: {max(baseline_differences)}")

    case_scores, fold_rows, sweep_rows = fit_leave_one_record_out(cases, predictions)
    write_csv(OUTPUT_ROOT / "逐条件全波形指标.csv", case_scores)
    write_csv(OUTPUT_ROOT / "留一记录验证汇总.csv", fold_rows)
    write_csv(OUTPUT_ROOT / "训练参数扫描.csv", sweep_rows)
    plot_results(fold_rows, sweep_rows)

    print(f"Eligible real ERP conditions: {len(cases)} (n >= {MIN_TRIALS})")
    print(f"Scale-1.5 reproduction max absolute difference: {max(baseline_differences):.3g}")
    for record in RECORD_NAMES:
        chosen = next(row["scale"] for row in sweep_rows if row["heldout_record"] == record and row["selected_boundary_constrained"])
        unconstrained = next(row["scale"] for row in sweep_rows if row["heldout_record"] == record and row["selected_unconstrained"])
        for stage in ("Stage1", "Stage2"):
            rows = [row for row in fold_rows if row["heldout_record"] == record and row["stage"] == stage]
            before = next(row["mean_full_curve_r"] for row in rows if row["fit_role"] == "fixed_baseline")
            after = next(row["mean_full_curve_r"] for row in rows if row["fit_role"] == "boundary_constrained_fullcurve")
            unconstrained_row = next(row["mean_full_curve_r"] for row in rows if row["fit_role"] == "unconstrained_fullcurve")
            print(f"{RECORD_NAMES[record]} {stage}: valid scale={chosen:.2f}, held-out r {before:.3f} -> {after:.3f} ({after-before:+.3f}); unconstrained scale={unconstrained:.2f}, r={unconstrained_row:.3f}")
    print(f"Results: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
