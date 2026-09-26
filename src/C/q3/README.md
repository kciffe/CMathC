# Q3 动态认知模型

本目录保留问题三当前使用的动态宏观模型、事件时间审计代码、输入表和核心验证结果。模型调用问题二的前向视觉链生成候选场景，再构造 V/H/P 功能状态，并按完整记录留出评估 F3、Fz、F4 的 EEG 预测。

## 运行

在仓库根目录执行：

```powershell
python src/C/q3/09_event_time_semantics.py
python src/C/q3/dynamic_validation.py
```

原始 MAT 记录需位于仓库根目录的 `data/` 数据目录；Q2 前向模型及导联矩阵需保留在 `src/C/q2/revision_v3/` 和 `src/C/q2/output/revision_v3/`。

第一步生成逐试次事件时间表 `output/continuation_audit/event_timing_by_trial.csv`，第二步读取该表并生成动态留出验证结果。模型将候选目标时刻和刺激类型作为敏感性场景；通道 9 的首个零到非零边沿只用于定义候选应答时刻和评分窗口，不作为 EEG 特征或状态输入。

## 输入与结果

- `input/`：Q3 外部事件、记录映射和试次真值表。
- `output/continuation_audit/event_timing_by_trial.csv`：动态验证使用的逐试次事件时间表。
- `output/dynamic_heldout_validation/`：核心验证摘要、固定窗与应答前窗指标、逐试次审计、结果图和运行摘要。
- `tests/`：当前动态模型、验证流程和事件解码的测试代码。

候选刺激和目标时刻尚未由逐试次标记确认，跨记录验证也不等同于独立受试者验证；结果不支持把 V/H/P 解释为具体脑区活动。
