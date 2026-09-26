# Q3 动态模型本次运行摘要

## 运行配置与数据

- 纳入 EEG epoch：355/400；记录数：4。
- 通道9操作性应答标记相对 cue 的总体中位数：2.2148 s；5%–95% 分位数：2.2148–2.2188 s。
- 应答前 EEG 端点定义为每个试次自己的 (t_act - 0.100 s)。只有 EEG 质控通过且 cue 区间内恰有一个通道9边沿的试次进入应答前评分。
- 目标持续时长候选：0.2 s；所有 target onset 与类型仍是敏感性场景，不代表已核实的真实试次事件。
- Q2 仿真分辨率：128；拟合与模型预测不使用通道9。

## 固定晚期窗口 [0.8, 2.8) s

下表为每个预处理×模型跨留出记录的晚期 NRMSE 汇总。NRMSE 越低，预测误差相对该留出 EEG 的标准差越小。

| 预处理 | 模型 | 平均 NRMSE | 记录间标准差 | 留出记录数 |
|---|---|---:|---:|---:|
| causal | Q2_plus_memory | 1.6474 | 0.7606 | 4 |
| causal | Q2_plus_memory_control | 1.6834 | 0.8317 | 4 |
| causal | Q2_visual_only | 1.6479 | 0.7586 | 4 |
| causal | Training_mean_template | 2.1489 | 1.6467 | 4 |
| none | Q2_plus_memory | 2.3511 | 0.9131 | 4 |
| none | Q2_plus_memory_control | 2.3699 | 0.9331 | 4 |
| none | Q2_visual_only | 2.3328 | 0.8938 | 4 |
| none | Training_mean_template | 2.4744 | 0.9864 | 4 |
| zero_phase | Q2_plus_memory | 1.6356 | 0.6674 | 4 |
| zero_phase | Q2_plus_memory_control | 1.6183 | 0.6549 | 4 |
| zero_phase | Q2_visual_only | 1.6144 | 0.6406 | 4 |
| zero_phase | Training_mean_template | 2.0805 | 1.4277 | 4 |

## 逐试次应答前窗口评分

每个试次先按自身端点分别截取累计窗 [0, t_act - 0.1) 与末 500 ms 窗，再在留出“记录×cue 侧”组内拼接截取后的 EEG 样本，计算 pooled RMSE、NRMSE 与相关系数。表中数值对候选场景和留出组作描述性平均，场景并非独立样本。

| 预处理 | 模型 | 窗口 | 平均 NRMSE | NRMSE行间标准差 | 场景/留出组数 | 每组计分试次数中位数 |
|---|---|---|---:|---:|---:|---:|
| causal | Q2_plus_memory | t_act_pre_response_cumulative | 1.0266 | 0.0420 | 152 | 44.5 |
| causal | Q2_plus_memory | t_act_pre_response_terminal_500ms | 1.0583 | 0.0922 | 152 | 44.5 |
| causal | Q2_plus_memory_control | t_act_pre_response_cumulative | 1.0283 | 0.0444 | 152 | 44.5 |
| causal | Q2_plus_memory_control | t_act_pre_response_terminal_500ms | 1.0565 | 0.0898 | 152 | 44.5 |
| causal | Q2_visual_only | t_act_pre_response_cumulative | 1.0264 | 0.0419 | 152 | 44.5 |
| causal | Q2_visual_only | t_act_pre_response_terminal_500ms | 1.0591 | 0.0936 | 152 | 44.5 |
| causal | Training_mean_template | t_act_pre_response_cumulative | 1.0409 | 0.0652 | 152 | 44.5 |
| causal | Training_mean_template | t_act_pre_response_terminal_500ms | 1.0291 | 0.0313 | 152 | 44.5 |
| none | Q2_plus_memory | t_act_pre_response_cumulative | 1.0537 | 0.0513 | 152 | 44.5 |
| none | Q2_plus_memory | t_act_pre_response_terminal_500ms | 1.0494 | 0.0518 | 152 | 44.5 |
| none | Q2_plus_memory_control | t_act_pre_response_cumulative | 1.0523 | 0.0501 | 152 | 44.5 |
| none | Q2_plus_memory_control | t_act_pre_response_terminal_500ms | 1.0492 | 0.0517 | 152 | 44.5 |
| none | Q2_visual_only | t_act_pre_response_cumulative | 1.0557 | 0.0534 | 152 | 44.5 |
| none | Q2_visual_only | t_act_pre_response_terminal_500ms | 1.0505 | 0.0526 | 152 | 44.5 |
| none | Training_mean_template | t_act_pre_response_cumulative | 1.0596 | 0.0537 | 152 | 44.5 |
| none | Training_mean_template | t_act_pre_response_terminal_500ms | 1.0606 | 0.0598 | 152 | 44.5 |
| zero_phase | Q2_plus_memory | t_act_pre_response_cumulative | 1.0206 | 0.0233 | 152 | 44.5 |
| zero_phase | Q2_plus_memory | t_act_pre_response_terminal_500ms | 1.0284 | 0.0240 | 152 | 44.5 |
| zero_phase | Q2_plus_memory_control | t_act_pre_response_cumulative | 1.0239 | 0.0293 | 152 | 44.5 |
| zero_phase | Q2_plus_memory_control | t_act_pre_response_terminal_500ms | 1.0251 | 0.0218 | 152 | 44.5 |
| zero_phase | Q2_visual_only | t_act_pre_response_cumulative | 1.0212 | 0.0229 | 152 | 44.5 |
| zero_phase | Q2_visual_only | t_act_pre_response_terminal_500ms | 1.0306 | 0.0262 | 152 | 44.5 |
| zero_phase | Training_mean_template | t_act_pre_response_cumulative | 1.0491 | 0.0922 | 152 | 44.5 |
| zero_phase | Training_mean_template | t_act_pre_response_terminal_500ms | 1.0309 | 0.0249 | 152 | 44.5 |

各试次的实际端点与每个留出组的计分试次数见 `t_act_endpoint_cv_metrics.csv`。CSV 中 `window_start_s`、`window_stop_s` 是各试次窗口界限的中位数摘要；评分本身逐试次使用各自界限。

## 解释边界

通道9首次零到非零边沿作为操作性 (t_{act})，但是否等于实际动作启动时刻仍需外部同步资料核实。通道9只定义评分窗口，不作为 EEG 特征、模型状态输入或观测回归预测变量。固定晚期窗口是统一 cue 锁定窗；候选目标时刻 2.2 s 和 2.4 s 晚于总体中位数应答前端点约 2.115 s，因此这些候选场景下应答前窗不含候选目标后的加工。结果支持模型流程已运行并接受留出检验，不能单独证明模型较好解释脑电。

详细条件行、应答前窗口指标、逐试次审计和分析图分别保存在 `dynamic_cv_metrics.csv`、`t_act_endpoint_cv_metrics.csv`、`observed_trial_audit.csv` 与 `figures/`。
