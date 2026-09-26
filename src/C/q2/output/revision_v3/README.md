# Revision v3：问题二第一小问计算结果

## 实现内容

本版本用标准化刺激矩阵作为确定性输入，经 LGN ON/OFF 与 Gabor/形状模板前端、三组 E/I 群体动力学得到五个语义明确的源代理：左右视野分别投射到对侧半球的早期视觉与构型源，以及由左右模板偏好差形成的双侧中线对手源。模板偏好组接收左右视觉视野早期活动的固定等权平均；中线对手源是低维建模假设，不是解剖定位结论。电极/参考在外表面，源坐标在头内；再以均匀无限导体点偶极近似映射到 F3/Fz/F4。该映射是规范近似，不是有限球体或个体头模型。模型曲线与真实数据统一通过第一问的 0.2–24 Hz 双向四阶滤波、128 Hz 重采样和逐通道基线校正。拟合按留一 MAT 记录进行，尺度由固定左右参考刺激预先计算，不读取留出记录确定尺度。

## 本次设置

- 空间网格：128 × 128；前端时间特征步长：4 ms。
- 输入群体 RMS 尺度（early/shape/orientation，每个含左右通道）：`[[0.0007239999831654131, 0.0007239999831654131], [0.0005709999823011458, 0.0005709999823011458], [0.023274999111890793, 0.023274999111890793]]`。
- 导联场：规范化 10–20 位置、均匀导体点偶极近似、假定双乳突平均参考。它是可复现的示意映射，非个体头模；增益没有 μV 物理标定。
- Stage1 包含 0 ms 提示出现与 200 ms 提示消失；晚期成分不直接称为 P300。

## 留一记录拟合

- `VisualCogA_Task-1`: tau_s=78.87 ms, g_i=0.786, tau_a=100.0 ms, gain=-63.82, status=`profile_grid_best_candidate_optimizer_not_converged`
- `VisualCogA_Task-2`: tau_s=76.30 ms, g_i=0.629, tau_a=120.0 ms, gain=-34.67, status=`profile_grid_best_candidate_optimizer_not_converged`
- `VisualCogB_Task-1`: tau_s=76.67 ms, g_i=0.909, tau_a=120.0 ms, gain=-60.36, status=`profile_grid_best_candidate_optimizer_not_converged`
- `VisualCogB_Task-2`: tau_s=75.14 ms, g_i=1.096, tau_a=160.0 ms, gain=-56.79, status=`profile_grid_best_candidate_optimizer_not_converged`
- 四折中 `tau_s` 到达 100 ms 上界：0/4；达到设定的 12 次目标评估上限而未收敛：4/4。报告这些参数时应视为边界候选值，不能当作已识别的时间常数。

对左右 × 四份 MAT 的整体 held-out NRMSE 均值为 **1.810**（按实测 ERP RMS 归一）。若该值接近 1，正向模型的绝对波形解释力弱；幅度增益不能掩盖这一点。
右减左波形相关系数的四记录均值为 **-0.147**。相关系数只描述波形相似，不代表幅度或机制正确。
几何映射下模型左右差异的侧化能量占比均值为 **0.278**；实测为 **0.168**（单记录范围 0.001–0.345）。这表明当前对称源/导联假设把差异限制在侧化方向，不能解释实测中出现的共同/形状模态差异。

## 特征区分验证

固定 6 维特征（u0 与 u1 各自的 80–200、250–450、450–700 ms 均值）在留一 MAT 记录 shrinkage-LDA 的平衡准确率为 **0.453**，AUC 为 **0.421**。四份 MAT 是记录，不应等同四个独立被试；这项跨记录结果若接近机会水平，说明当前固定特征没有稳健的记录间泛化。

## 针对性机制对照

去除空间位置后保留的模型左右差异比例均值：**0.000**。仅保留 cue onset、删除 200 ms offset 后的比例：**1.317**。对照使用同一折拟合参数和同一增益，不重新拟合。比例用于描述模型生成差异的来源，不用于声称真实 EEG 的因果效应。

## 解释边界

1. 本模型给出从输入到传感器代理的显式正向计算链，可用于解释模型内部的差异来源；三电极与四份记录不足以识别真实脑内源分布。
2. Stage1 的左右模板响应是形状方向编码的模型假设。必须同时报告实测右减左波形和留出结果；分类或拟合指标差不能单独证明生理机制。
3. The cue-only model is simulated naturally through the complete 0-3000 ms epoch; no target response at 2.2 s is inserted. The measured context may contain a target-related component that leaks backward through zero-phase filtering, so absolute ERP fit remains conditional on the unverified event context.
4. 真实 EEG 使用第一问已清洗数据，模型在 ERP 前执行逐试次基线均值校正；这不等于线性基线回归，也不重新裁定第一问数据清洗的有效性。
5. 原始的 v1/v2 代码和结果均未覆盖或删除；25 次预算结果另存于 `output/revision_v3_max25/`。

## 主要输出

- `heldout_erp.png`：三通道左右条件的实测与留出模型 ERP。
- `heldout_difference_modes.png`：右减左的 common/lateral/shape 模态。
- `fitted_cascade.png`：拟合参数中位数下的 E/I 群体和观测代理。
- `lgn_spatiotemporal.png`：代表性左 cue 下 LGN ON/OFF 空间通道随时间的响应图。
- `cortical_ei_responses.png`：三组 E/I 响应，保留视野位置和模板偏好通道标签。
- `source_to_electrode_contributions.png`：五源对 F3/Fz/F4 的逐时贡献；代码检查其和等于电极预测。
- `lgn_spatiotemporal.png`：同一次拟合代表刺激下，8×8 pooled LGN ON/OFF 通道的时空活动。
- `cortical_ei_responses.png`：三组 E/I 群体逐通道响应；模板偏好通道与视觉视野通道分开标注。
- `source_to_electrode_contributions.png`：五个源代理对 F3/Fz/F4 的逐源贡献；图中校验各贡献之和等于电极预测。
- `heldout_metrics.csv`、`left_right_difference.csv`、`fixed_feature_lda.csv`、`mechanism_controls.csv`。

## Fixed nine-dimensional electrode ERP feature validation

The predeclared feature consists of F3, Fz, and F4 means in [100,250), [250,500), and [500,800) ms. Leave-one-MAT-out shrinkage-LDA balanced accuracy is **0.466**, macro AUC is **0.451** over 4 held-out records. The windows and channels were fixed before examining these fold results; scaling was estimated from training records only.
Four MAT files are recordings, not verified independent participants. This result is evidence about this fixed feature under record holdout, not evidence that the forward mechanism is correct.
The associated model population/source differences use the same fitted parameters, drive scales, gain, and observation operator for the left/right pair in each held-out fold. Five-source additivity is checked after filtering, resampling, and baseline correction.
New outputs: `model_population_right_minus_left.png`, `model_source_electrode_right_minus_left.png`, `measured_vs_model_candidate_features.png`, `heldout_9d_electrode_lda.png`, `heldout_9d_electrode_lda_folds.csv`, `measured_candidate_features_exploratory.csv`, and `model_source_contribution_summary.csv`.
