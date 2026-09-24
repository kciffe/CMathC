# Revision v3：问题二第一小问计算结果

## 实现内容

本版本用标准化刺激矩阵作为确定性输入，经 LGN ON/OFF 与 Gabor/形状模板前端、三组 E/I 群体动力学、E-I 突触滤波得到六个源代理，再用固定几何近似导联场映射到 F3/Fz/F4。模型曲线与真实数据统一通过第一问的 0.2–24 Hz 双向四阶滤波、128 Hz 重采样和逐通道基线校正。拟合按留一 MAT 记录进行，尺度由固定左右参考刺激预先计算，不读取留出记录确定尺度。

## 本次设置

- 空间网格：128 × 128；前端时间特征步长：4 ms。
- 输入群体 RMS 尺度（early/shape/orientation，每个含左右通道）：`[[0.0007239999831654131, 0.0007239999831654131], [0.0005709999823011458, 0.0005709999823011458], [0.023274999111890793, 0.023274999111890793]]`。
- 导联场：规范化 10–20 位置、均匀导体点偶极近似、假定双乳突平均参考。它是可复现的示意映射，非个体头模；增益没有 μV 物理标定。
- Stage1 包含 0 ms 提示出现与 200 ms 提示消失；晚期成分不直接称为 P300。

## 留一记录拟合

- `VisualCogA_Task-1`: tau_s=95.75 ms, g_i=0.634, tau_a=60.0 ms, gain=-10.9, status=`profile_grid_best_candidate_optimizer_not_converged`
- `VisualCogA_Task-2`: tau_s=76.14 ms, g_i=0.649, tau_a=80.0 ms, gain=-3.851, status=`profile_grid_best_candidate_optimizer_not_converged`
- `VisualCogB_Task-1`: tau_s=100.00 ms, g_i=0.510, tau_a=40.0 ms, gain=-7.903, status=`profile_grid_best_candidate_optimizer_not_converged`
- `VisualCogB_Task-2`: tau_s=100.00 ms, g_i=0.500, tau_a=40.0 ms, gain=-6.291, status=`profile_grid_converged`

对左右 × 四份 MAT 的整体 held-out NRMSE 均值为 **1.805**（按实测 ERP RMS 归一）。若该值接近 1，正向模型的绝对波形解释力弱；幅度增益不能掩盖这一点。
右减左波形相关系数的四记录均值为 **0.150**。相关系数只描述波形相似，不代表幅度或机制正确。

## 特征区分验证

固定 6 维特征（u0 与 u1 各自的 80–200、250–450、450–700 ms 均值）在留一 MAT 记录 shrinkage-LDA 的平衡准确率为 **0.453**，AUC 为 **0.421**。四份 MAT 是记录，不应等同四个独立被试；这项跨记录结果若接近机会水平，说明当前固定特征没有稳健的记录间泛化。

## 针对性机制对照

去除空间位置后保留的模型左右差异比例均值：**0.000**。仅保留 cue onset、删除 200 ms offset 后的比例：**1.399**。对照使用同一折拟合参数和同一增益，不重新拟合。比例用于描述模型生成差异的来源，不用于声称真实 EEG 的因果效应。

## 解释边界

1. 本模型给出从输入到传感器代理的显式正向计算链，可用于解释模型内部的差异来源；三电极与四份记录不足以识别真实脑内源分布。
2. Stage1 的左右模板响应是形状方向编码的模型假设。必须同时报告实测右减左波形和留出结果；分类或拟合指标差不能单独证明生理机制。
3. 使用第一问已清洗数据，因此本版本只检查共同观测处理的一致性，不重新裁定第一问数据清洗的医学/实验有效性。
4. 原始的 v1/v2 代码和结果均未覆盖或删除。

## 主要输出

- `heldout_erp.png`：三通道左右条件的实测与留出模型 ERP。
- `heldout_difference_modes.png`：右减左的 common/lateral/shape 模态。
- `fitted_cascade.png`：拟合参数中位数下的 E/I 群体和观测代理。
- `heldout_metrics.csv`、`left_right_difference.csv`、`fixed_feature_lda.csv`、`mechanism_controls.csv`。
