# -*- coding: utf-8 -*-
"""Run the final Q1 analysis pipeline in order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


Q1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Q1_DIR.parents[2]
OUTPUT_DIR = Q1_DIR / "output"

PIPELINE = (
    "1clear_data.py",
    "6drop_clipped.py",
    "7filter_downsample.py",
    "8riemann_denoise.py",
    "9check_riemann_drop.py",
    "10check_b_alpha_sensitivity.py",
    "11erp_extract.py",
    "12check_a1_cleaning_sensitivity.py",
    "14erp_spline_fit.py",
    "clip_trial_examples.py",
)


def remove_tree_contents(path: Path) -> int:
    """Remove a generated Q1 directory without touching paths outside output/."""
    output_root = OUTPUT_DIR.resolve()
    target = path.resolve()
    if not target.is_relative_to(output_root) or target == output_root:
        raise ValueError(f"Refusing to clean a path outside Q1 output: {target}")
    if not target.exists():
        return 0

    files = [item for item in target.rglob("*") if item.is_file() or item.is_symlink()]
    for item in files:
        item.unlink()

    directories = sorted(
        (item for item in target.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        directory.rmdir()
    target.rmdir()
    return len(files)


def clean_intermediates() -> None:
    """Keep final figures and result tables while removing regenerated inputs."""
    removed = 0

    for path in OUTPUT_DIR.glob("*_sliced_with_drop.mat"):
        path.unlink()
        removed += 1

    removed += remove_tree_contents(OUTPUT_DIR / "7filter_downsample")

    sqi_dir = OUTPUT_DIR / "8riemann_denoise"
    if sqi_dir.exists():
        for path in sqi_dir.iterdir():
            if path.is_file() and (
                path.name.endswith("_clean.mat") or path.suffix.lower() == ".png"
            ):
                path.unlink()
                removed += 1

    waveform_points = OUTPUT_DIR / "14erp_spline_fit" / "ERP样条拟合波形.csv"
    if waveform_points.is_file():
        waveform_points.unlink()
        removed += 1

    print(f"Cleaned {removed} regenerated intermediate files; final results were kept.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all final Q1 analysis scripts in sequence."
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the generated stage 6–8 MAT files and diagnostic plots.",
    )
    args = parser.parse_args()

    child_env = os.environ.copy()
    child_env["MPLBACKEND"] = "Agg"

    pipeline_start = time.perf_counter()
    for index, script_name in enumerate(PIPELINE, start=1):
        script_path = Q1_DIR / script_name
        if not script_path.is_file():
            print(f"Missing pipeline script: {script_path}", file=sys.stderr)
            return 2

        print(f"[{index}/{len(PIPELINE)}] Running {script_name}", flush=True)
        step_start = time.perf_counter()
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT,
                env=child_env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"Stopped at {script_name} (exit code {exc.returncode}); "
                "intermediate files were kept for diagnosis.",
                file=sys.stderr,
            )
            return exc.returncode or 1

        elapsed = time.perf_counter() - step_start
        print(f"Finished {script_name} in {elapsed:.1f}s", flush=True)

    if not args.keep_intermediates:
        clean_intermediates()

    print(f"Q1 pipeline finished in {time.perf_counter() - pipeline_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
