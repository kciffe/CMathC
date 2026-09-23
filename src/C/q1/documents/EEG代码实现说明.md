# EEG 数据处理代码实现说明

> 本文按当前代码和已生成文件整理，供报告编写时引用。重点处理链路为 `6drop_clipped.py` → `7filter_downsample.py`，阶段标注由 `7plot_trial.py` 绘制。目录中另有清洗、异常分析和切片脚本，文中将其作为独立分支说明，避免混淆处理结果。

## 1. 数据结构

原始数据以 MATLAB `.mat` 文件保存，主要字段为 `data`、`SampleRate` 和 `DataLabel`。`data` 的每一行为一个通道，当前数据的通道对应关系如下。

| MATLAB 行号 | Python 索引 | 通道 | 用途 |
|---:|---:|---|---|
| 1 | 0 | Fz | EEG |
| 2 | 1 | F3 | EEG |
| 3 | 2 | F4 | EEG |
| 4–6 | 3–5 | FzDecon、F3Decon、F4Decon | 设备处理通道 |
| 7 | 6 | ECG | 心电 |
| 8 | 7 | VisCue | 视觉提示事件 |
| 9 | 8 | Action / TgtAct | 动作事件；Task-2 原始标签为 `TgtAct` |
| 10 | 9 | TimeStamp | 时间戳 |

采样率由每个 `.mat` 文件的 `SampleRate` 字段读取；当前四份数据的采样率均为 256 Hz，每份记录均检测到 100 个有效提示事件。

## 2. 当前采用的处理流程

```text
四份原始 MAT 文件
        │
        ├── 1clear_data.py：原始连续信号阶段与 P300 候选窗示例图
        │
        └── 6drop_clipped.py：按 VisCue 切成 Trial，并生成削顶弃置标记
                    │
                    └── 7filter_downsample.py：对四份数据滤波、降采样并分别输出比较图
```

`1clear_data.py` 用于展示原始连续 EEG 的提示前阶段、提示、等待、目标和动作事件；实际切片及 `drop` 标记由 `6drop_clipped.py` 生成。`7filter_downsample.py` 逐一读取四份 `*_sliced_with_drop.mat`，不再只处理 Task-1 A。

## 3. Trial 切片与削顶标记

### 3.1 事件检测与切片

程序在 `VisCue` 通道上检测从 0 变为非零的采样点：

\[
\mathrm{onset}_t = (\mathrm{VisCue}_t \ne 0) \land (\mathrm{VisCue}_{t-1}=0).
\]

以提示起点为 0 秒，切取提示前 1 秒至提示后 3 秒。对于 256 Hz 数据，切片使用前 256 点和后 768 点，得到每个 Trial 1024 个采样点；时间轴由 `arange(-256, 768) / 256` 生成，范围为 `−1.000` 至 `2.9961 s`，右端不包含 3 秒。

Trial 保留全部 10 个通道，并按提示方向保存 `cue_type`（左刺激为 −1，右刺激为 +1）。

### 3.2 削顶判定

`6drop_clipped.py` 检查每个 Trial 的 Fz、F3、F4 三个 EEG 通道。只要任一采样点满足

\[
|x| \ge 999,
\]

该 Trial 的 `drop` 就标为 1，否则标为 0。标记类型为 `uint8`。这一脚本保留所有 Trial 行，不插值，也不物理删除 Trial；后续分析可据 `drop` 字段排除相应试次。

四份切片文件的统计如下：

| 数据集 | Trial 数 | `drop=1` | `drop=0` |
|---|---:|---:|---:|
| VisualCogA_Task-1 | 100 | 17 | 83 |
| VisualCogA_Task-2 | 100 | 15 | 85 |
| VisualCogB_Task-1 | 100 | 15 | 85 |
| VisualCogB_Task-2 | 100 | 17 | 83 |
| 合计 | 400 | 64 | 336 |

每个 `*_sliced_with_drop.mat` 包含 `trial_data`、`relative_time`、`cue_type`、`drop`、`SampleRate` 和 `DataLabel`。其中 `trial_data` 的形状为 `(100, 10, 1024)`，`relative_time` 的形状为 `(100, 1024)`。

## 4. 滤波与降采样

`7filter_downsample.py` 处理四份切片数据，主要参数如下：

| 参数 | 设置 |
|---|---:|
| 高通截止频率 | 0.2 Hz |
| 低通截止频率 | 24 Hz |
| Butterworth 设计阶数 | 4 |
| 滤波方式 | `scipy.signal.filtfilt`，前后向零相位滤波 |
| 输出采样率 | 128 Hz |

滤波器按读取到的采样率设计，通带为 0.2–24 Hz。代码逐 Trial 对 Fz、F3、F4 三个 EEG 通道独立执行四阶 Butterworth 前后向滤波；其余通道在滤波阶段原样保留。随后将连续信号通道（Fz、F3、F4、FzDecon、F3Decon、F4Decon、ECG，以及可能存在的其他连续通道）用 `scipy.signal.resample` 重采样。`VisCue` 和 `Action` 是离散事件通道，不做 Fourier 插值；每个降采样时间格保留其中首个非零事件值，避免单采样点事件在抽取时丢失。`TimeStamp` 保留对应原始采样点的时间戳。当前 256 Hz 输入按 2:1 降采样至 128 Hz，1024 点变为 512 点；`relative_time` 从原始时间轴按相同间隔抽取。

每个 `*_filtered_downsample.mat` 只保留切片源文件原有的六个字段：`trial_data`、`relative_time`、`cue_type`、`drop`、`SampleRate` 和 `DataLabel`，不增加中间数据或滤波参数字段。`trial_data` 替换为降采样后的完整多通道数据，形状为 `(100, 10, 512)`；`relative_time` 更新为降采样时间轴，`SampleRate` 更新为 128 Hz。未处理的原始 Trial 数据仍保存在对应的 `*_sliced_with_drop.mat` 中。`drop` 只作为弃置标记保存，当前处理仍保留所有 Trial 行。

## 5. 阶段标注与比较图

`7plot_trial.py` 按 `cue_type` 分别选取最先出现的 5 个左刺激 Trial 和 5 个右刺激 Trial，绘制成 5 行 × 2 列图。每个子图显示 F3、Fz、F4 三条曲线。阶段时间从提示开始计，按 `1clear_data.py` 的定义映射到 Trial 相对时间轴：

| 阶段或事件 | Trial 相对时间 |
|---|---:|
| 提示前阶段 | −0.5 至 0 s |
| 三角形提示 | 0 至 0.2 s |
| 提示后等待 | 0.2 至 2.2 s |
| 目标开始 | 2.2 s |
| 提示后 P300 候选窗 | 0.25 至 0.50 s |
| 目标后 P300 候选窗 | 2.45 至 2.70 s |
| 动作/目标动作事件 | 从第 9 个 MATLAB 通道（Python 索引 8；具体名称读取 `DataLabel`）提取提示后的首个 0→非零跳变 |

阶段区间使用浅色背景，两个 P300 候选窗使用较深背景；提示前、提示、等待、目标和动作起点分别使用虚线标注。绘图使用 Microsoft YaHei、SimHei 等字体显示中文。

每份数据均生成四张比较图：滤波前、滤波后、滤波后且降采样前、滤波并降采样后。`before_downsample` 图展示的是已滤波的数据，只是重采样前状态。每个子图根据自己的信号自动设置纵轴范围，因此跨图比较振幅时应读取各自纵轴刻度。Task-1 A 保留原有文件名；其他三份数据的文件名前加数据集名称，以便同目录区分。

输出位于 `src/C/q1/output/7filter_downsample/`：

- `before_filter/before_filter.png`
- `after_filter/after_filter.png`
- `before_downsample/before_downsample.png`
- `after_downsample/after_downsample.png`
- `VisualCogA_Task-1_filtered_downsample.mat`

其余数据集分别输出 `VisualCogA_Task-2_filtered_downsample.mat`、`VisualCogB_Task-1_filtered_downsample.mat` 和 `VisualCogB_Task-2_filtered_downsample.mat`；各阶段图位于相同阶段子目录，文件名为 `<数据集名称>_<阶段>.png`。四份数据共生成 16 张比较图。

## 6. 相关脚本的职责与边界

| 脚本 | 当前实现职责 |
|---|---|
| `1clear_data.py` | 读取 A 组 Task-1 原始连续数据，展示前 5 个左刺激和前 5 个右刺激 Trial 的阶段及 P300 候选窗；横轴使用原始时间戳。 |
| `2fault_analyse.py` | 读取 `EEG异常检测指标.csv`，按 Trial 汇总削顶、跳变和基线指标，分级并绘制统计图；该脚本不修改 EEG 波形。 |
| `3wrong3_data_fix.py` | 独立的清洗实现：按 `|x|≥999` 检测削顶，短段插值、长段最终设为 NaN；检测并修复短时跳变，并执行 0.1 Hz 高通去漂移。它不属于 `6drop_clipped.py` → `7filter_downsample.py` 的直接数据链路。 |
| `4clip_data.py` | 从已清洗的 `_clean.mat` 读取数据，按 VisCue 和时间戳切分 Trial，并统计 NaN、动作和长度信息。 |
| `5EEG_slice.py` | 从四份原始 MAT 直接切取 `[-1, 3)` 秒 Trial，生成 `_sliced.mat`；该版本不写削顶 `drop` 标记。 |
| `6drop_clipped.py` | 当前四数据集切片与削顶标记实现，输出 `*_sliced_with_drop.mat`。 |
| `7filter_downsample.py` | 对四份数据滤波、降采样，保存全通道和源字段，并为每份数据生成四阶段比较图。 |
| `7plot_trial.py` | 供滤波脚本调用的 Trial 绘图函数，统一画信号、阶段阴影、事件线和图例。 |
| `EEG_clean_with_compare.py` | 独立的清洗比较脚本：对短削顶段和跳变点做插值并绘图；当前版本不含慢漂移修复。 |

报告描述最终结果时，应明确采用了哪条链路。`3wrong3_data_fix.py`、`4clip_data.py`、`5EEG_slice.py` 与 `6drop_clipped.py` 是不同的处理入口或版本，输出字段和处理规则并不相同。

## 7. 当前实现的复现注意事项

1. `drop` 是弃置标记，不是物理删行操作。当前滤波结果和比较图仍包含 `drop=1` 的试次；四个结果 MAT 均保留对应的 `drop` 字段。后续统计时需显式按该字段筛除不纳入分析的试次。
2. `7filter_downsample.py` 在切片后逐 Trial 调用 `filtfilt`，不是在完整连续 EEG 上滤波后再切 Trial。复现实验时应保持该实现描述准确。
3. 降采样对连续信号使用 Fourier 重采样；对事件通道和时间戳采用保持语义的抽取规则，不能将这些通道当作普通 EEG 曲线进行插值。
4. 旧文件 `EEG数据清洗说明_削顶_跳变_慢漂移.md` 记录的是清洗算法说明；若报告讨论当前的削顶弃置标记及 0.2–24 Hz 滤波，应以本说明和对应脚本为准，不要把两条处理链路的参数混写。

## 8. 可用于报告正文的实现描述

> 本研究首先读取四份包含 Fz、F3、F4、设备处理通道、ECG、VisCue、动作事件和时间戳的多通道 EEG 数据。依据 VisCue 从零到非零的跳变定位提示起点，并以提示起点为零时刻提取前 1 s 至后 3 s 的定长 Trial。对每个 Trial 的 Fz、F3、F4 通道进行削顶检查，当任一采样点绝对值达到 999 时，将该 Trial 标记为待剔除试次，保留数据并通过二值字段记录标记。随后，以四阶 Butterworth 带通滤波器分别处理四份数据的三个 EEG 通道，通带设为 0.2–24 Hz，采用前后向滤波降低相位偏移；连续信号通道重采样至 128 Hz，VisCue、动作事件和时间戳则按离散事件/时间戳语义保留对应信息。滤波降采样结果沿用切片 MAT 的原有字段，保存完整 10 通道 `trial_data`、`drop` 与 `DataLabel`；未处理的原始 Trial 仍在原切片文件中。最后，按左、右提示类别各展示前五个 Trial，并在比较图中标注提示前阶段、提示呈现、等待阶段、目标起点、动作事件起点及两段 P300 候选时窗。四份数据共标记 64 个 Trial，其中 336 个未触发削顶阈值。

> 注：正文中的“待剔除”对应 `drop=1` 标记；当前滤波降采样脚本尚未按该标记实际删除 Trial。
