# revision_v2：形状对向读出与固定观测模式

## 用途

`revision_v2` 是问题二正向模型的一条独立实验分支，用来检验两项改动是否改善解释力：

1. 左、右三角模板分别读取，不做左右镜像不变池化；共同形状响应与左右方向对向响应分别进入低维 Wilson–Cowan 群体模型。
2. 将 F3/Fz/F4 转换到固定的正交传感器模式 `u0/u1/u2`，不拟合自由导联权重。

它保留 `revision_v1` 作为基线，不覆盖旧脚本或旧结果。`u0/u1/u2` 是电极信号的坐标变换，不是解剖脑区或真实源定位。模型单位为相对单位，不能解释为微伏。

## 运行

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe src/C/q2/revision_v2/run_revision.py --max-nfev 35
```

结果写入 `src/C/q2/output/revision_v2/`。数据来自第一问 `output/8riemann_denoise` 清洗后的 MAT 文件。主拟合仅使用 Stage1 左/右提示；Stage2 未知布局不重新贴标签。四份 MAT 按记录留一，训练记录估计参数，留出记录只用于评价。

## 实现边界

- 视觉路径为 signed contrast → LGN ON/OFF → Gabor → 共同形状/左右对向响应 → 三组 Wilson–Cowan 兴奋/抑制群体。
- 前端空间网格按核能量二阶矩审计自动选择；本次选择 128×128。特征以 4 ms 步长计算，再插值至 1 ms 动力学网格；与 1 ms 参考比较后通过预设误差阈值。
- 留出拟合参数为 `tau_s`、`g_i`、`tau_a` 及一个共享有符号增益。增益仅一个，不逐条件或逐电极拟合。少数参数触及搜索边界，因此估计不应解释为生理参数真值。
- 独立报告真实 EEG 的 6/9 维特征留一记录 LDA、记录内置换和分半稳定性。分类结果仅表示给定特征的可分性，不证明前向神经机制成立。
- 固定参数消融比较 opponent/legacy 前端路由与 rank-3/rank-2 观测。消融不重新拟合参数。

## 本次运行结果（2026-09-25）

- 模板方向检查通过：左右镜像刺激交换左右模板分数，说明构型前端能表达方向信息。
- 真实 EEG 留一 MAT 的平衡准确率：6 维 `0.453`、9 维 `0.479`；AUC 分别为 `0.421`、`0.448`。记录内 9 维置换检验经四记录 Holm 校正后均未达到 0.05。分半侧向模式稳定性不一致。
- opponent rank-3 正向模型的三电极留出 NRMSE 中位数为 `0.9998`、均值为 `1.0341`；零预测基线 NRMSE 为 `1.0000`。按整体归一化平方误差计算，模型相对零预测的平均 skill 为 `-0.077`。因此本次没有证据表明 v2 改善了绝对 ERP 拟合。
- v2 对左右 ERP 差异的幅值保留比例中位数约 `0.065`，四份记录间的差值波形相关方向不一致。固定 rank-2 与 rank-3 结果几乎相同，第三观测模式本次没有明显改善拟合。
- v1 与 v2 都没有稳定解释跨记录的左右波形差异。前端机制读出正确，不等于真实 EEG 中存在足够稳定的对应信号；不能据此声称已解释真实神经机制。

## 主要产物

- `../output/revision_v2/heldout_ERP_comparison.png`：留出条件实测与模拟 ERP。
- `../output/revision_v2/heldout_left_right_difference.png`：左右差值波形及模型保留比例。
- `../output/revision_v2/real_eeg_separability.png`：真实数据分类与分半稳定性。
- `../output/revision_v2/shape_opponent_template_responses.png`：左右模板方向响应检查。
- `../output/revision_v2/heldout_forward_metrics.csv`、`heldout_left_right_difference.csv`：留出拟合和左右差异指标。
- `../output/revision_v2/real_eeg_LO_record_LDA.csv`、`within_record_permutation.csv`、`split_half_stability.csv`：真实 EEG 可分性分析。
- `../output/revision_v2/spatial_resolution_audit.json`、`frontend_temporal_stride_audit.csv`：空间尺度与时间抽样审计。
- `../output/revision_v2/manifest.json`：数据、模型、参数范围和代码哈希清单。

## 结论

本分支完成了所提替代架构的可复现实现与留出评价，但没有解决“模型解释真实 EEG 左右差异”的问题。现有证据显示，方向模板编码是有效的；真实 EEG 的左右可分性跨记录不稳定，前向模型也未优于零预测基线。下一步应依据预先定义的信号质量和事件语义检查，判断可用 EEG 是否含足够信息，而不继续通过扩大模型自由度或按留出结果调参来追求拟合。
