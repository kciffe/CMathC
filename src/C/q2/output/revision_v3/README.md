# 问题二正式模型结果

本目录保存唯一正式模型 `revision_v3` 的结果。模型由标准化刺激矩阵驱动，依次计算 LGN ON/OFF、空间特征、三组兴奋/抑制群体、五个功能源代理以及 F3/Fz/F4 观测。源代理和导联场用于表达模型内部的正向映射，不代表个体解剖定位；输出幅度也没有微伏标定。

## 验证摘要

- 四份 MAT 记录按整份记录留出。模拟 ERP 的平均 NRMSE 为 **1.810**，右减左波形相关均值为 **-0.147**。
- 固定九维电极特征的记录留出平衡准确率为 **0.466**，宏平均 AUC 为 **0.451**。
- 四折拟合优化器均未报告收敛。四份 MAT 是记录，不应视为四名独立受试者。
- 模型只输入提示刺激，未在约 2.2 秒处加入目标输入；晚期拟合受真实事件上下文限制。

这些结果支持检查模型内部的差异来源，但没有显示稳定的跨记录波形解释或解码。详细指标见 CSV 与 `manifest.json`。

## 正式输出

- `heldout_fit_summary.csv`、`heldout_metrics.csv`、`heldout_predictions.csv`：留出拟合参数和 ERP 指标。
- `heldout_9d_electrode_lda_summary.csv`、`heldout_9d_electrode_lda_folds.csv`：固定九维特征的分类结果。
- `left_right_difference.csv`、`mechanism_controls.csv`：左右差异和模型内机制对照。
- `source_additivity_audit.csv`、`leadfield.csv`、`leadfield_geometry.json`：源到电极映射及求和审计。
- `heldout_erp.png`、`heldout_difference_modes.png`、`fitted_cascade.png`、`lgn_spatiotemporal.png`、`cortical_ei_responses.png`、`source_to_electrode_contributions.png`：正式模型图件。

从仓库根目录运行 `python src/C/q2/revision_v3/run_v3.py` 可重建模型结果。代码和入口说明见 `src/C/q2/README.md`。
