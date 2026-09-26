"""按顺序运行问题二的完整分析与论文生成流程。"""

import os
from pathlib import Path
import subprocess
import sys
import time


Q2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Q2_DIR.parents[2]
PIPELINE = (
    ("revision_v3/run_v3.py", "Q2 模型、拟合、留出验证与结果审计"),
    ("reports/build_q2_paper.py", "论文文件生成与版式检查"),
)


def main():
    child_env = os.environ.copy()
    child_env["MPLBACKEND"] = "Agg"
    pipeline_start = time.perf_counter()

    for index, (relative_path, description) in enumerate(PIPELINE, start=1):
        script_path = Q2_DIR / relative_path
        if not script_path.is_file():
            print(f"缺少流程脚本：{script_path}", file=sys.stderr)
            return 2

        print(f"[{index}/{len(PIPELINE)}] 开始：{description}", flush=True)
        step_start = time.perf_counter()
        try:
            subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT,
                           env=child_env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"流程在 {relative_path} 处停止，退出码：{exc.returncode}",
                  file=sys.stderr)
            return exc.returncode or 1
        print(f"完成：{description}，耗时 {time.perf_counter() - step_start:.1f} 秒。",
              flush=True)

    print(f"Q2 完整流程结束，总耗时 {time.perf_counter() - pipeline_start:.1f} 秒。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
