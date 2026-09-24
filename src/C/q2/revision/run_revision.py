"""Run from any directory: python src/C/q2/revision/run_revision.py."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Q2_ROOT, OUTPUT_ROOT, STAGES
from model import STIMULI, simulate, load_contrast, cortex_from_v1
from diagnostics import run_diagnostics


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item()), encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            w.writeheader()
            w.writerows(rows)


def baseline_hashes():
    paths = list(Q2_ROOT.glob("*.py"))
    for p in (Q2_ROOT / "output").rglob("*"):
        if p.is_file() and "revision" not in p.relative_to(Q2_ROOT / "output").parts:
            paths.append(p)
    return {str(p.relative_to(Q2_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def plot_mechanism(predictions, out):
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    a = predictions[("Stage1", "left", 40)]
    b = predictions[("Stage2", "dots", 40)]
    for ax, result, title in ((axes[0, 0], a, "Cue: onset 0 ms, offset 200 ms"), (axes[0, 1], b, "Target: continuously present")):
        ax.plot(result["time_ms"], result["lgn_mean"][0], label="LGN ON", color="#007C91")
        ax.plot(result["time_ms"], result["lgn_mean"][1], label="LGN OFF", color="#B46A38")
        ax.set(title=title, xlabel="Time from onset (ms)", ylabel="Mean population rate (relative)")
        ax.legend(frameon=False)
    axes[0, 0].axvline(200, color="0.5", linestyle=":")
    left = a["v1_joint_feature"]
    right = predictions[("Stage1", "right", 40)]["v1_joint_feature"]
    axes[1, 0].plot([0, 45, 90, 135], left.mean(axis=(1, 2)), "o-", label="Left cue", color="#007C91")
    axes[1, 0].plot([0, 45, 90, 135], right.mean(axis=(1, 2)), "x--", label="Right cue", color="#B46A38")
    axes[1, 0].set(title="Global orientation energy overlaps", xlabel="Gabor angle (degrees)", ylabel="Mean activity (relative)")
    axes[1, 0].legend(frameon=False)
    delta = (left - right).reshape(4, 64)
    scale = np.max(np.abs(delta))
    im = axes[1, 1].imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    axes[1, 1].set(title="Joint position-orientation difference", xlabel="Spatial cell (row-major)", ylabel="Gabor angle index")
    fig.colorbar(im, ax=axes[1, 1], label="Left - right (relative)")
    fig.savefig(out / "mechanism.png", dpi=180)
    fig.savefig(out / "mechanism.pdf")
    plt.close(fig)


def plot_real_cases(cases, predictions, out):
    channels = ("F3", "Fz", "F4")
    for record in sorted({c["dataset"] for c in cases}):
        for stage in STAGES:
            selected = [c for c in cases if c["dataset"] == record and c["stage"] == stage]
            if not selected:
                continue
            fig, axes = plt.subplots(len(selected), 3, figsize=(10, 2.4 * len(selected)), squeeze=False, constrained_layout=True)
            for row, case in enumerate(selected):
                real = case["real"]
                real_scale = max(float(np.sqrt(np.mean(real ** 2))), 1e-15)
                for i, channel in enumerate(channels):
                    ax = axes[row, i]
                    ax.plot(case["time_ms"], real[i] / real_scale, color="0.3", label="Real ERP")
                    key = (stage, case["condition"], 40)
                    if key in predictions:
                        revised = predictions[key]
                        rs = max(float(np.sqrt(np.mean(revised["eeg"] ** 2))), 1e-15)
                        ax.plot(revised["time_ms"], revised["eeg"][i] / rs, color="#007C91", label="Revision fixed 40")
                        baseline = np.load(Q2_ROOT / "output" / "04_simulated_eeg" / case["task"] / stage / f'{case["condition"]}_eeg.npy')
                        bs = max(float(np.sqrt(np.mean(baseline ** 2))), 1e-15)
                        ax.plot(np.arange(baseline.shape[1]), baseline[i] / bs, "--", color="#B46A38", label="Baseline 1.50")
                    ax.set(title=f'{case["condition"]} / {channel} / n={case["n_trials"]}', xlabel="Time from onset (ms)", ylabel="Each series / pooled RMS")
                    if stage == "Stage1":
                        ax.axvline(200, color="0.7", linestyle=":")
                    ax.grid(alpha=.15)
            axes[0, 0].legend(fontsize=8, frameon=False)
            fig.suptitle(f"{record} / {stage} (shape view; not amplitude fitting)")
            fig.savefig(out / f"{record}_{stage}.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    before = baseline_hashes()
    predictions = {}
    for stage, condition in STIMULI:
        print(f"simulate {stage}/{condition}", flush=True)
        result = simulate(load_contrast(stage, condition), stage)
        predictions[(stage, condition, 40)] = result
        for tau in (30, 50, 60):
            candidate = cortex_from_v1(result["v1_drive"], result["time_ms"], tau)
            candidate["time_ms"] = result["time_ms"]
            predictions[(stage, condition, tau)] = candidate
        for tau in (30, 40, 50, 60):
            saved = {k: v for k, v in predictions[(stage, condition, tau)].items() if k != "v1_drive"}
            np.savez_compressed(OUTPUT_ROOT / f"{stage}_{condition}_tau{tau}.npz", **saved)
    diagnostics = run_diagnostics(predictions)
    write_json(OUTPUT_ROOT / "mechanism_diagnostics.json", diagnostics)
    plot_mechanism(predictions, OUTPUT_ROOT)
    if not args.simulate_only:
        from real_data import load_cases_with_audit
        from validation import evaluate
        cases, audit = load_cases_with_audit()
        write_csv(OUTPUT_ROOT / "event_audit.csv", audit)
        for c in cases:
            np.savez_compressed(OUTPUT_ROOT / f'{c["dataset"]}_{c["stage"]}_{c["condition"]}_real.npz', time_ms=c["time_ms"], erp=c["real"], trials=c["trials"])
        summary = evaluate(cases, predictions, OUTPUT_ROOT)
        summary["structural_pass"] = diagnostics["structural_pass"]
        write_json(OUTPUT_ROOT / "validation_summary.json", summary)
        plot_real_cases(cases, predictions, OUTPUT_ROOT)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.item()), flush=True)
    after = baseline_hashes()
    if before != after:
        raise RuntimeError("Baseline artifacts changed during revision run")
    source_paths = list(Path(__file__).parent.glob("*.py")) + list((Q2_ROOT / "input").glob("*.npy"))
    write_json(OUTPUT_ROOT / "run_manifest.json", {
        "baseline_unchanged": True, "baseline_file_count": len(before),
        "baseline_hashes": before, "assumptions": STAGES,
        "parameter_grid_ms": [30, 40, 50, 60], "fixed_tau_ms": 40,
        "numpy_version": np.__version__, "seed": 20260924,
        "source_hashes": {str(p.relative_to(Q2_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}})
    print(f"Artifacts: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
