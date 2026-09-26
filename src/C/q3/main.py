"""按顺序运行 Q3 的事件审计与动态验证流程。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


Q3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Q3_DIR.parents[2]
PIPELINE = (
    "09_event_time_semantics.py",
    "dynamic_validation.py",
)


def main() -> int:
    """先生成事件时间表，再运行 Q2-Q3 留记录验证。"""
    child_env = os.environ.copy()
    child_env["MPLBACKEND"] = "Agg"
    pipeline_start = time.perf_counter()

    for index, script_name in enumerate(PIPELINE, start=1):
        script_path = Q3_DIR / script_name
        if not script_path.is_file():
            print(f"缺少流程脚本：{script_path}", file=sys.stderr)
            return 2

        print(f"[{index}/{len(PIPELINE)}] 正在运行 {script_name}", flush=True)
        step_start = time.perf_counter()
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT,
                env=child_env,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            print(
                f"流程在 {script_name} 处停止，退出码：{error.returncode}",
                file=sys.stderr,
            )
            return error.returncode or 1
        print(f"{script_name} 完成，用时 {time.perf_counter() - step_start:.1f} 秒", flush=True)

    print(f"Q3 流程完成，用时 {time.perf_counter() - pipeline_start:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
