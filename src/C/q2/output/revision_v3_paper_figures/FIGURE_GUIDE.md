# revision_v3：01–05 步图件索引

主分析顺序为 VisualCogA_Task-2、VisualCogB_Task-1、VisualCogB_Task-2；
VisualCogA_Task-1 按第一问质量结论标注为低质量参考，并保留在真实数据对照中。

01–04 为使用三个重点记录已保存留出拟合中位数的确定性 cue-only 正向示意；不重新优化参数。
05/05b 使用各记录自己的 saved leave-one-MAT-out 预测与真实 trial-average ERP。

空间图按输入图像坐标显示：数组第 0 行位于图像上沿；Stage1 补充图均直接读取 V3 前端响应。
stage1_Gabor* 图名沿用原图习惯，但数值来自 revision_v3，不是旧版 scipy Gabor 结果。

v3 拟合的 tau_s 候选达到上界，优化状态未收敛；图中的代表参数只用于展示链路，
不能写作已识别的生理时间常数。传感器增益也没有标定为 μV。

## 01_v3_gabor_frontend

V3 前端将左右 cue 映射为空间化的 Gabor 方向能量响应。

文件：`output\revision_v3_paper_figures\01_v3_gabor_frontend.pdf`, `output\revision_v3_paper_figures\01_v3_gabor_frontend.svg`, `output\revision_v3_paper_figures\01_v3_gabor_frontend.png`, `output\revision_v3_paper_figures\01_v3_gabor_frontend.figure.json`

## stage1_raw_stimulus_preprocessing

Stage1 cue 原始图像含共同圆形基线；逐像素相减后共同圆形抵消，仅三角区域保留亮度变化。

文件：`output\revision_v3_paper_figures\stage1_raw_stimulus_preprocessing.pdf`, `output\revision_v3_paper_figures\stage1_raw_stimulus_preprocessing.svg`, `output\revision_v3_paper_figures\stage1_raw_stimulus_preprocessing.png`, `output\revision_v3_paper_figures\stage1_raw_stimulus_preprocessing.figure.json`

## stage1_Gabor方向响应总览

Stage1 左右 cue 在 100 ms 的 V3 Gabor 方向能量空间分布互为镜像。

文件：`output\revision_v3_paper_figures\stage1_Gabor方向响应总览.pdf`, `output\revision_v3_paper_figures\stage1_Gabor方向响应总览.svg`, `output\revision_v3_paper_figures\stage1_Gabor方向响应总览.png`, `output\revision_v3_paper_figures\stage1_Gabor方向响应总览.figure.json`

## stage1_Gabor差异

V3 Gabor 各方向响应的空间差异显示左右 cue 的镜像编码。

文件：`output\revision_v3_paper_figures\stage1_Gabor差异.pdf`, `output\revision_v3_paper_figures\stage1_Gabor差异.svg`, `output\revision_v3_paper_figures\stage1_Gabor差异.png`, `output\revision_v3_paper_figures\stage1_Gabor差异.figure.json`

## stage1_Gabor空间响应热图

V3 Gabor 能量在刺激上方的 4×4 前端网格中呈现左右镜像空间分布。

文件：`output\revision_v3_paper_figures\stage1_Gabor空间响应热图.pdf`, `output\revision_v3_paper_figures\stage1_Gabor空间响应热图.svg`, `output\revision_v3_paper_figures\stage1_Gabor空间响应热图.png`, `output\revision_v3_paper_figures\stage1_Gabor空间响应热图.figure.json`

## 02_v3_lgn_response

镜像左右 cue 的 LGN 整体 ON/OFF 平均响应近似相同，但空间化中继响应呈镜像分布。

文件：`output\revision_v3_paper_figures\02_v3_lgn_response.pdf`, `output\revision_v3_paper_figures\02_v3_lgn_response.svg`, `output\revision_v3_paper_figures\02_v3_lgn_response.png`, `output\revision_v3_paper_figures\02_v3_lgn_response.figure.json`

## 03_v3_population_dynamics

V3 将视觉输入传入早期、形状和方向三个功能群体，内部左右通道响应可在时间上分化。

文件：`output\revision_v3_paper_figures\03_v3_population_dynamics.pdf`, `output\revision_v3_paper_figures\03_v3_population_dynamics.svg`, `output\revision_v3_paper_figures\03_v3_population_dynamics.png`, `output\revision_v3_paper_figures\03_v3_population_dynamics.figure.json`

## 04_v3_simulated_eeg

V3 的几何导联场将六个源代理映射为 F3/Fz/F4 相对传感器曲线。

文件：`output\revision_v3_paper_figures\04_v3_simulated_eeg.pdf`, `output\revision_v3_paper_figures\04_v3_simulated_eeg.svg`, `output\revision_v3_paper_figures\04_v3_simulated_eeg.png`, `output\revision_v3_paper_figures\04_v3_simulated_eeg.figure.json`

## 导联矩阵热图

V3 用固定规范头部几何导联矩阵将六个源代理映射到 F3/Fz/F4。

文件：`output\revision_v3_paper_figures\导联矩阵热图.pdf`, `output\revision_v3_paper_figures\导联矩阵热图.svg`, `output\revision_v3_paper_figures\导联矩阵热图.png`, `output\revision_v3_paper_figures\导联矩阵热图.figure.json`

## 05_v3_heldout_erp

V3 的左右 cue 波形拟合在三个重点记录与低质量参考记录间存在异质性。

文件：`output\revision_v3_paper_figures\05_v3_heldout_erp.pdf`, `output\revision_v3_paper_figures\05_v3_heldout_erp.svg`, `output\revision_v3_paper_figures\05_v3_heldout_erp.png`, `output\revision_v3_paper_figures\05_v3_heldout_erp.figure.json`

## 05b_v3_heldout_lr_modes

V3 预测的左右差异主要落在侧化模态；真实右减左波形在三种观测模态中的幅度和方向依记录而异。

文件：`output\revision_v3_paper_figures\05b_v3_heldout_lr_modes.pdf`, `output\revision_v3_paper_figures\05b_v3_heldout_lr_modes.svg`, `output\revision_v3_paper_figures\05b_v3_heldout_lr_modes.png`, `output\revision_v3_paper_figures\05b_v3_heldout_lr_modes.figure.json`
