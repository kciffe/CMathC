# `10check_b_alpha_sensitivity.py` 代码实现说明

## 1. 用途与处理边界

该脚本对 B 组 `VisualCogB_Task-1`、`VisualCogB_Task-2` 做 SQI 阈值系数 `α` 的敏感性分析，观察不同保留集合下左右条件 ERP 的变化。它是独立诊断分支，不回写正式清洗结果。

```text
第 7 阶段 filtered_downsample MAT ──> 固定的 Trial 波形、条件和削顶标记
第 8 阶段 SQI 指标 CSV ─────────────> 固定的 SQI 与正式剔除标记
                                      │
                                      └─> 改变 B 组 α 阈值与保留集合
                                           ├─> 左右条件 ERP 对比图
                                           └─> Fz 候选峰值汇总 CSV
```

分析中只改变 B 组 SQI 阈值及由此得到的 Trial 保留集合。削顶标记、SQI 数值、滤波与降采样后的波形均固定；脚本不会重跑 SQI、滤波或削顶检测，也不会改写 `output/8riemann_denoise` 中的正式文件。

## 2. 输入与 α=.6 一致性门槛

每个 B 组数据读取：

1. `output/7filter_downsample/{dataset}_filtered_downsample.mat`：`trial_data`、`relative_time`、`cue_type`、`drop`。
2. `output/8riemann_denoise/{dataset}_SQI指标.csv`：`Trial`、`CueType`、`ClippedDrop`、`SQI`、`RiemannDrop`、`FinalDrop`。

脚本检查 MAT 与 CSV 的 Trial 编号、提示条件和削顶标记对应，并确认正式 `FinalDrop` 等于削顶标记与 Riemann/SQI 剔除标记的并集。原始 knee 阈值通过动态加载 `8riemann_denoise.py` 中的 `sqi_knee_threshold()` 计算，不另写一套 knee 算法。

在产生任何敏感性结果前，脚本会对两个 B 组都执行 α=.6 的逐 Trial 校验：用当前 SQI、削顶标记和 α=.6 阈值重建保留集合，要求其与正式 `FinalDrop` 完全一致，同时要求 `RiemannDrop` 也逐条一致。任何一组不一致都会报错并停止，避免在口径不同的情况下继续比较。

## 3. 阈值与保留集合

候选 α 固定为：

```text
α ∈ {0.6, 0.7, 0.8, 1.0}
threshold = max(0.05, α × knee)
```

对每个 α，未削顶且满足 `SQI >= threshold` 的 Trial 保留。削顶 Trial 在所有 α 下均保持排除，不参与 α 引起的变化。这里采用 `SQI < threshold` 剔除，因此 SQI 恰等于阈值时保留。

α=.6 通过正式结果校验后使用正式保留掩码；其余 α 使用同一份 SQI 和削顶标记，仅重新应用阈值，确保比较只反映保留集合变化。

## 4. ERP 计算口径

### 4.1 逐 Trial 基线校正

每个 Trial、每个 EEG 通道分别计算 `−0.2 <= t < 0 s` 的样本均值，并从该 Trial 对应该通道的全部时间点减去该均值。区间左端包含、右端不包含，因此提示 onset 的 `t=0` 样本不参与基线均值。

### 4.2 分条件平均

完成逐 Trial 基线校正后，根据 `cue_type` 对当前 α 保留的 Trial 分别平均：

- `cue_type = −1`：左条件；
- `cue_type = +1`：右条件。

通道索引使用项目数据约定：Fz 为 0、F3 为 1、F4 为 2。图中显示三个通道；候选峰值指标只从左右条件的 Fz 平均 ERP 提取。

## 5. 候选 P300 指标

只在 Fz 条件平均波形上，对以下两个闭区间内的样本寻找最大正值：

| 窗口 | 时间范围（相对提示 onset） | 输出潜伏期 |
|---|---:|---|
| 提示后候选窗 | `0.25～0.50 s` | 相对提示 onset |
| 目标后候选窗 | `2.45～2.70 s` | 同时报告相对提示 onset、相对目标 onset |

目标 onset 按 `2.2 s` 计算，故目标后潜伏期换算为 `峰值时间 − 2.2 s`。每个条件、每个 α 都输出窗口最大正峰幅值和对应采样时刻。文中及字段名使用“候选 P300 峰值/潜伏期”，不据此直接断言它是真实 P300。

当前实现会返回窗口内最大值，但不自动判定该值是否恰在窗口边界。若潜伏期落在窗口端点，或接近端点，应把它视为窗口受限的候选指标，不能解释成已可靠定位的生理峰值；需要结合完整 ERP 图进一步检查。

## 6. 输出文件

输出目录：

```text
output/10check_b_alpha_sensitivity/
```

脚本写出：

1. `b_alpha_sensitivity_summary.csv`：每个数据集 × α × 左右条件各一行，共 16 行。包含 α、SQI 阈值、总保留数、左右条件保留数、当前条件 Trial 数、Fz 提示后与目标后候选峰幅值及潜伏期。CSV 使用 UTF-8 BOM 编码。
2. `{dataset}_alpha_ERP_sensitivity.png`：每个 B 组一张 2×3 图。行是左右条件，列是 F3/Fz/F4；不同颜色表示 α，图例附该条件保留 Trial 数。阴影标出两个候选窗口，参考线标出提示 onset `t=0` 和目标 onset `t=2.2 s`。图中若多个 α 的保留集合相同，其曲线会完全重合，视觉上表现为少于四条不同曲线。

当前一次运行的 α=.6 一致性及保留数快照：

| 数据集 | α=.6 正式一致 | α=.6 保留数 | α=.7 | α=.8 | α=1.0 |
|---|---|---:|---:|---:|---:|
| B_Task-1 | 是 | 83 | 81 | 81 | 79 |
| B_Task-2 | 是 | 80 | 78 | 77 | 76 |

以上数量是当前输出数据的运行记录；若输入 MAT 或 SQI CSV 变化，应重新运行脚本并以新输出为准。

## 7. 运行与测试

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe src/C/q1/10check_b_alpha_sensitivity.py
```

单元测试：

```powershell
.\.venv\Scripts\python.exe -m unittest src.C.q1.test_10check_b_alpha_sensitivity -v
```

测试覆盖 α 阈值系数与下限、不同阈值下削顶 Trial 始终不保留、逐 Trial 基线校正、左右条件 ERP 平均、候选峰潜伏期换算，以及 α=.6 与正式剔除结果不一致时必须报错。

## 8. 解读限制

敏感性曲线用于检查结果对 B 组阈值选择的依赖，不用于自动选出“效果最好”的 α。某个 α 产生更高的候选峰，不等于该 α 更正确；尤其不能因为某些 Trial 会降低 ERP 峰值就反向选择更严格的阈值。α=.6 是与当前正式输出严格对齐的基准，其余 α 仅作对照。
