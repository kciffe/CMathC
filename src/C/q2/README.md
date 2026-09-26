# 问题二：视觉到脑电的正向模型

当前项目只保留 revision_v3 作为正式模型。模型将视觉刺激依次映射为 LGN ON/OFF 响应、空间特征、皮层兴奋/抑制活动和 F3/Fz/F4 观测代理。

## 目录

- `revision_v3/`：模型、数据读取、拟合、评价和核心测试；模型子流程由 `run_v3.py` 执行，并由上层 `main.py` 调用。
- `input/`：模型使用的刺激矩阵与预览图。
- `output/revision_v3/`：当前正式结果、审计表和图件。
- `documents/问题二文档.md`：模型结构、参数与实现说明。
- `documents/figures/`：论文和说明文档引用的正式图件。
- `reports/`：最终论文文件及论文生成器；生成契约输入保存在 `reports/paper_audit/input/`。

真实 EEG 读取自第一问清洗结果 `src/C/q1/output/8riemann_denoise/`。四份 MAT 文件按记录留出，不能据此视为四名独立受试者。

## 运行

统一入口参照 Q1 的 `main.py`，按脚本顺序运行正式模型、拟合、留出验证、结果审计和论文生成。从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe src/C/q2/main.py
```

入口固定运行模型分析、留出验证、结果审计和论文生成，不提供分支选项。正式结果写入 `output/revision_v3/`，论文文件写入 `reports/`。
