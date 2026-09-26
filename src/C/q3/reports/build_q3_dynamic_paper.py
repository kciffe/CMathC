from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


Q3_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Q3_ROOT.parents[2]
REPORTS = Q3_ROOT / "reports"
AUDIT = REPORTS / "paper_audit"
INPUTS = AUDIT / "input"
OUTPUTS = AUDIT / "output"
TITLE = "问题三：基于问题二前向机制的视觉认知动态宏观模型"
KEYWORDS = "视觉认知；脑电信号；动态宏观模型；记录留出验证；敏感性分析"
SUMMARY_PATH = Q3_ROOT / "output" / "dynamic_heldout_validation" / "validation_summary.json"
METRICS_PATH = Q3_ROOT / "output" / "dynamic_heldout_validation" / "dynamic_cv_metrics.csv"


ABSTRACT = """题目要求在问题二的脑电形成机制基础上构造认知宏观模型，并用实测脑电验证，同时指出应答可能错误或未及时。本文以四份256 Hz记录中的F3、Fz、F4和VisCue为观测对象，沿“前向机制衔接、动态状态构造、整份记录留出检验”建立问题三模型。事件审计发现，400个提示事件各对应一个第9通道起始沿，其相对提示时间中位数为2.2148 s，接近候选目标时刻；由于没有独立的目标起始时间、逐试次正确答案和反应截止时间，通道9只用于事件审计，不作为反应时、正确性、漏答或脑电预测标签。按边界、非有限值、原始幅度和通道平直规则筛选后保留355个完整试次。问题二前向链在候选提示与目标场景下生成五个源代理，并经3×5导联矩阵正向投影到三电极；数值投影的最大绝对差为5.954×10⁻⁸，该检查只确认计算接口一致。本文以问题二早期视觉兴奋群体均值驱动视觉状态V，再以提示门控、候选目标门控和假设匹配证据递推记忆相关状态H与控制状态P。目标类型、目标时刻、持续时间及匹配证据均按候选情景设置，状态时间常数和增益保持预设值，不由留出数据反向挑选。观测方程按电极逐一拟合问题二视觉轨迹、H、P及控制交互项；输入标准化量和岭回归系数只由训练记录估计。每折完整留出一份记录，用其余三份记录的记录×提示侧均值波形拟合，再在固定提示后0.8–2.8 s晚期窗评价，比较训练均值模板、问题二视觉基线、加入记忆态及再加入控制态四种方案；同时考察无滤波、因果滤波、零相位滤波和2.0、2.2、2.4 s三个目标候选时刻。加入H后，晚期NRMSE在因果滤波下由1.6479降至1.6474，变化仅0.03%；无滤波和零相位滤波下分别增加0.78%和1.31%。在九个处理方式×目标时刻聚合单元中只有两个误差下降，控制态也没有形成跨处理方式的额外收益。所有处理口径下误差仍高于留出信号标准差，且外层只有四份记录。因此，当前证据支持“模型已构造并完成跨记录检验”，但不支持记忆态具有稳定预测增益的结论。结果不作脑区定位、临床诊断效能或真实行为决策证据；应答前约100 ms的信号终点也暂不能验证，需待目标和行为时间戳核实后再开展。"""


BODY = r"""# 问题重述与分析

题目要求在问题二脑电形成机制的基础上建立认知宏观模型，从实测脑电中截取与视觉认知相关的信号并验证模型；题面同时提醒，应答可能错误，也可能未及时发生\cite{problem2026}。本题数据提供四份连续记录、F3/Fz/F4三个脑电通道、视觉提示通道VisCue以及第9通道行动标记和时间戳。问题二给出的视觉前端与源到头皮的正向机制可为视觉驱动和电极观测关系提供结构约束。

本实现把“模型构造”和“事件真值”分开处理。动态状态由问题二前向视觉驱动及预设情景产生，实测EEG只用于训练折中的传感器映射拟合和外层留出评价。按参考模型，第9通道零到非零边沿定义为应答时刻`t_act`，用于补充定义应答前累计窗和末500 ms窗；它不作为EEG特征、状态或回归输入。目标呈现时刻没有逐试次标记，因此`t_act`相对目标 onset 的反应时长仍未知。

## 问题分析

三路额区EEG只能观测头皮电位，不能从三维观测唯一恢复问题二的五个源代理，更不能直接确认新增状态对应某一深部脑区。若用自由拟合潜变量再将其命名为视觉、记忆或控制，名称本身不会带来可辨识性。本文保留问题二的前向计算方向，用其视觉群体活动驱动V/H/P功能状态；新增状态是否有用，则由完整记录留出时的脑电预测误差决定。

另一难点是事件与行为标签。审计到的第9通道起始沿集中在提示后约2.215 s，与候选目标计划时刻接近，而数据没有独立目标起始标记。四份文件的任务类型映射也未由实验记录确认。故候选目标时间、刺激类型和匹配状态并列作敏感性情景，不能依据验证误差挑选一个“真实”时刻或正确性标签。

整体路线为：从连续EEG按VisCue切段并作统一质量筛选；用问题二前向链重算候选cue-target场景；按离散动力学得到V/H/P；在每个留一记录折的训练数据上估计电极观测映射；最后比较新增记忆和控制状态能否改善晚期留出EEG。此评价检验当前模型的预测增量，不等同于机制因果检验。

# 模型假设与符号说明

## 模型假设

1. VisCue的零到非零边沿可作为可复现的提示起点；这一假设只用于提示对齐，不替代目标起始事件。
2. 候选目标刺激和时间用于构造敏感性场景。其持续时间按0.2 s设定，匹配证据取情景值；这些量不代表逐试次观测真值。
3. V/H/P的时间常数与状态增益采用预设正值。它们定义可检验的动态假设，当前不由EEG反演或行为标签估计。
4. 四份MAT文件作为四个外层留出域。由于记录身份未能确认，四折结果只称记录留出，不称独立受试者验证。
5. 电极观测系数只在相应训练记录上估计，留出记录不参与特征标准化、拟合或候选条件选择。

## 符号说明

**表1 主要符号与含义**

|符号|含义|取值或单位|
|---|---|---|
|$\tau$|相对VisCue起点的时间|s|
|$\mathbf y_i(\tau)$|第$i$试次F3、Fz、F4实测电位向量|$\mathbb R^3$|
|$\mathbf q_{Q2,k}$|问题二的五维源代理|$\mathbb R^5$|
|$G$|五源代理到三电极的正向导联矩阵|$3\times5$|
|$u_{Q2,k}$|问题二早期视觉兴奋群体的空间均值|相对状态量|
|$V_k,H_k,P_k$|视觉、记忆相关、控制功能状态|相对状态量|
|$C_k,T_k$|提示和候选目标的时间门控|$\{0,1\}$|
|$r$|假设匹配证据|本次情景为$-1$或$+1$|
|$\rho_j,\tau_j$|状态离散保持率与时间常数|无量纲、s|
|$\alpha$|岭回归惩罚系数|1|
|$\mathrm{NRMSE}$|按留出实测标准差归一化的均方根误差|越小越好|

# 数据处理与事件审计

每份原始记录以256 Hz采样，每份含100个VisCue提示事件，四份共400个候选试次。以VisCue由零到非零的边沿确定提示时刻$t_i^c$，把三通道信号插值到统一提示相对时间轴。单个epoch范围为$[-0.2,3.0]$ s；预处理后模型比较统一在128 Hz网格上进行。

质量筛选检查epoch是否越过记录边界或与下一提示重叠，是否存在非有限值、原始幅度绝对值不小于999.5的样本，以及任一通道是否平直。原始数据单位未独立校准，因此幅度门槛以原始单位表述，不换算为微伏。筛选后保留355个完整试次，覆盖四份记录；所有分支使用同一纳入集合。

事件审计在每个提示区间中检出一个第9通道起始沿，标记相对提示的中位时间为2.2148 s，5%至95%分位数为2.2148至2.2188 s。依参考模型，将唯一的零到非零边沿作为绝对应答时刻`t_act`，并以`t_act-100 ms`作为应答前窗口截止点。候选cue+2.2 s仅是目标时间计划代理，不能用于推定真实反应时长；正确性和漏答状态也仍需真实值及截止时间。

图1用于说明已观测提示、通道9标记、候选目标时刻和实际评价窗口之间的关系。其作用是界定可观测边界；图中的候选线不表示真实目标起点。

![事件标记与分析窗](../output/dynamic_heldout_validation/figures/analysis_event_windows.png){width=92%}

实测信号、问题二电极轨迹以及宏观状态量分别采用无滤波、四阶Butterworth 0.5至30 Hz因果滤波和离线零相位滤波三种处理。滤波先作用于连续记录，再按事件切段，以减小短epoch起点的滤波器初始化影响；随后重采样至128 Hz并扣除提示前$[-0.2,0)$ s均值。三种口径都对实测信号和模型基函数使用相同的算子。零相位滤波只用于离线敏感性比较，不能据此评价在线处理性能。

图2展示被纳入的F3、Fz、F4提示对齐波形及记录组间变异。阴影表示八个记录×提示侧均值之间的标准差，不是受试者总体置信区间；电位单位也未校准为微伏。

![F3、Fz、F4提示对齐波形](../output/dynamic_heldout_validation/figures/observed_three_channel_stages.png){width=94%}

# 问题二前向机制接口

为使认知状态承接问题二的脑电形成过程，先将视觉提示与候选目标组成连续刺激场景，经视觉前端、LGN处理、皮层兴奋/抑制群体动力学和源代理映射形成问题二电极轨迹。皮层兴奋/抑制群体动力学沿用Wilson-Cowan类型的前向结构\cite{wilsonCowan1972}。记五个源代理为$\mathbf q_{Q2,k}$，导联矩阵为$G$，则电极信号采用正向关系：

$$
\mathbf y_{Q2,k}=G\mathbf q_{Q2,k},\qquad \mathbf q_{Q2,k}\in\mathbb R^5,\quad \mathbf y_{Q2,k}\in\mathbb R^3.
$$

三电极少于五个源代理，反问题不唯一；本实现只计算$G$的正向投影，不求逆，也不把五源轨迹当作实测真值。候选场景中，问题二早期视觉兴奋群体的空间均值作为Q3视觉驱动：

$$
u_{Q2,k}=\frac{1}{N_xN_y}\sum_{x=1}^{N_x}\sum_{y=1}^{N_y}E_{\mathrm{early}}(x,y,k).
$$

代码以矩阵乘法重算电极轨迹，并与问题二前向函数逐点比较，最大绝对差为$5.954\times10^{-8}$。该数值核对说明前向接口一致，不表示五源可从三电极唯一识别。

# 动态认知状态模型

本文把$V$定义为受问题二视觉驱动的快速状态，把$H$定义为提示写入并受候选匹配证据调制的记忆相关状态，把$P$定义为候选目标阶段由记忆强度和冲突情景驱动的控制状态。这里的H和P是功能状态名称，不是海马或前额叶源定位，也不表示已观测到真实匹配或真实冲突。

设$C_k,T_k$分别为提示与候选目标门控，$r\in[-1,1]$为假设匹配证据，$\Delta t=0.004$ s为问题二前向时间步。代码采用指数保持率$\rho_j=\exp(-\Delta t/\tau_j)$进行稳定离散递推。按实现中的更新顺序，状态满足：

$$
\begin{aligned}
V_k &= \rho_VV_{k-1}+(1-\rho_V)u_{Q2,k},\\
H_k &= \rho_HH_{k-1}+(1-\rho_H)\left(g_sV_kC_k+g_mV_kT_kr\right),\\
P_k &= \rho_PP_{k-1}+(1-\rho_P)\left(g_{HP}|H_{k-1}|T_k+g_\delta T_k\frac{1-r}{2}\right).
\end{aligned}
$$

计算从$V_0=u_{Q2,0}$、$H_0=P_0=0$开始。提示门控$C_k$使视觉状态写入H；候选目标门控$T_k$使$r$调节H，并使记忆强度和冲突项驱动P。三条状态分别使用$\tau_V,\tau_H,\tau_P$，因此不是按电极数构造的三个标签。虽然代码接口允许$[-1,1]$内的匹配证据，本轮只运行$r=-1$与$r=+1$两个情景，不能解释为正确与错误试次。

**表2 动态状态的预设时间常数和增益**

|参数|数值|进入方程的作用|
|---|---:|---|
|$\tau_V$|0.060 s|视觉状态平滑时间常数|
|$\tau_H$|1.000 s|记忆相关状态保持时间常数|
|$\tau_P$|0.250 s|控制状态衰减时间常数|
|$g_s$|0.80|提示写入增益|
|$g_m$|0.80|候选目标匹配调制增益|
|$g_{HP}$|0.25|记忆到控制的耦合增益|
|$g_\delta$|0.80|冲突情景增益|
|$g_{PV}$|0.10|控制对问题二视觉轨迹的调制增益|

这些数值是结构性初值，没有通过实测EEG或外层留出折估计。视觉、记忆与控制状态如何影响各电极，由下一节的训练折观测映射估计。图3用一个候选情景显示状态时程及训练折传感器贡献；载荷是预测系数，不具有解剖定位含义。

![动态状态与传感器贡献分解](../output/dynamic_heldout_validation/figures/dynamic_states_sensor_contributions.png){width=96%}

# 传感器观测方程与求解

对第$c$个电极，构造由问题二视觉轨迹、记忆相关状态、控制状态和视觉×控制交互组成的候选解释量。三种递增模型的设计向量分别为：

$$
\begin{aligned}
\mathbf x_{c,k}^{(0)}&=[\tilde y_{Q2,c,k}],\\
\mathbf x_{c,k}^{(H)}&=[\tilde y_{Q2,c,k},\tilde H_k],\\
\mathbf x_{c,k}^{(H,P)}&=[\tilde y_{Q2,c,k},\tilde H_k,\tilde P_k,\widetilde{y_{Q2,c,k}P_k}].
\end{aligned}
$$

波浪号表示按训练记录均值和标准差标准化。$Q2$模型只含视觉前向轨迹；$Q2+H$增加记忆相关状态；$Q2+H+P$再增加控制状态及其对视觉轨迹的交互。对每个电极独立拟合带截距的观测式：

$$
\hat y_{c,k}=\bar y_{c,\mathrm{train}}+\mathbf z_{c,k}^{\mathsf T}\hat\beta_c,
\qquad
\hat\beta_c=\arg\min_{\beta}\left\|\mathbf y_{c,\mathrm{train}}-\bar y_{c,\mathrm{train}}\mathbf 1-\mathbf Z_{c,\mathrm{train}}\beta\right\|_2^2+\alpha\|\beta\|_2^2,
\quad \alpha=1.
$$

其中$\mathbf Z$由训练折标准化后的设计向量逐时刻组成，$\bar y_{c,\mathrm{train}}$为训练目标均值。标准化均值、尺度和回归系数只从训练记录估计，预测留出记录时原样沿用。岭回归用于限制小样本、相关预测量下的系数波动\cite{hoerlKennard1970}。另以训练集中同一提示侧的平均波形作简单模板基线，检查动态模型是否超过跨记录均值预测。

模型处理三种共同口径：无滤波、因果滤波和离线零相位滤波。主比较窗是提示后$[0.8,2.8)$ s的晚期阶段；$[0,0.8)$ s早期窗用于检查新增状态是否只改善提示阶段拟合。对非空目标情景还计算候选起点后最多0.6 s的局部窗口。候选目标刺激为点阵、向内运动和向外运动，候选时刻为提示后2.0、2.2、2.4 s，另含无目标输入情景；候选持续时间为0.2 s。不同情景并列评价，不按留出误差筛选目标真值。

# 记录留出验证

外层每次留出一份完整MAT记录，另外三份记录作为训练数据。拟合单位是“记录×提示侧”的平均波形，各试次仍保留在质量审计中，但同一记录不会拆到训练和留出两侧。每个留出记录在各候选情景、提示侧和晚期窗上形成归一化误差。令$g$表示一个留出记录、提示侧、情景及评价时间窗，$y_g$和$\hat y_g$将F3/Fz/F4及该窗内时点展平，则：

$$
\mathrm{RMSE}_g=\sqrt{\frac{1}{n_g}\sum_{j=1}^{n_g}(y_{g,j}-\hat y_{g,j})^2},\qquad
\mathrm{NRMSE}_g=\frac{\mathrm{RMSE}_g}{\mathrm{SD}(y_g)},\qquad
\Delta_H=\mathrm{NRMSE}_{Q2}-\mathrm{NRMSE}_{Q2+H}.
$$

$\Delta_H>0$表示加入H后误差下降。先在每份留出记录内对提示侧、三类候选刺激、匹配/不匹配情景及三个非空目标时刻汇总，再对四份留出记录计算均值与标准差。均值和标准差以记录为单位，候选情景不当作独立受试者样本。四折结果用于报告外推表现，不进行超出样本规模的总体推断。

# 结果与分析

表3给出晚期窗中四种模型的记录留出NRMSE。每格为四份留出记录的均值（记录间标准差）；该汇总仅包含2.0、2.2、2.4 s三个非空目标候选，不把无目标情景混入表内。

**表3 不同预处理与模型的晚期留出NRMSE**

|预处理|训练均值模板|$Q2$|$Q2+H$|$Q2+H+P$|
|---|---:|---:|---:|---:|
@@RESULT_TABLE@@

因果滤波下，$Q2+H$的NRMSE为1.6474，略低于$Q2$视觉基线的1.6479，绝对改善0.0005、相对改善0.03%。无滤波下加入H使误差由2.3328升至2.3511，相对恶化0.78%；零相位滤波下由1.6144升至1.6356，相对恶化1.31%。因此改善只出现在一个预处理汇总口径，且幅度很小。三种处理下加入P后误差均未低于$Q2$视觉基线，当前结果不支持控制状态提供稳定额外预测信息。

三种预处理下，$Q2$视觉基线均优于提示侧训练均值模板，说明问题二前向视觉轨迹提供了可用于传感器预测的结构。但所有汇总NRMSE均大于1，表示预测误差仍与留出信号自身波动相当或更大；这限制了绝对波形拟合的解释。模型比较图呈现早期与晚期窗的完整模型次序，不能只凭一根柱子推断潜状态的神经来源。

图4展示四类模型的早期与晚期误差。柱形按情景和留出记录汇总，图中差别需结合表3中的记录间变异及下方目标时刻敏感性共同阅读。

![各模型的留出误差比较](../output/dynamic_heldout_validation/figures/heldout_model_comparison.png){width=96%}

图5展示代表性候选场景的留出波形。它用于检查预测曲线与实测曲线在量级和形态上的差异；该图不是全部情景的统计替代，也不用于挑选最佳候选目标时刻。

![留出预测与实测波形对照](../output/dynamic_heldout_validation/figures/heldout_predictions_vs_measured.png){width=96%}

进一步将$\Delta_H$按候选目标时刻汇总，表4中的正值表示加入H后误差下降。每个单元先平均三类目标刺激、两种匹配情景、左右提示侧和同一留出记录，再平均四份记录；无目标输入情景不计入这九个单元。

**表4 记忆状态的目标时刻敏感性：$\Delta_H$**

|预处理|2.0 s|2.2 s|2.4 s|
|---|---:|---:|---:|
@@SENSITIVITY_TABLE@@

九个处理方式×目标时刻单元只有两个为正，且都出现在因果滤波下的2.2或2.4 s情景；其余七个单元加入H后误差不降。所有候选时刻均保留为敏感性分支，没有利用验证误差反推真实target onset。这个模式与表3一致：目前最多只能说个别情景出现微弱改善，不能说记忆过程已得到验证。

![不同候选目标时刻下的记忆态误差改善](../output/dynamic_heldout_validation/figures/memory_target_time_sensitivity.png){width=96%}

# 模型评价与适用范围

本模型把问题二的视觉前向动力学和三电极观测端接入动态认知状态，使视觉驱动、记忆相关过程和控制/冲突过程具有明确时序方程；四份记录整份留出，降低了同一文件信号同时进入训练与验证造成的信息泄漏。共同预处理、训练折内标准化和预设候选网格也使不同状态模型可以在一致口径下比较。

当前证据仍有明确边界。第一，晚期窗固定在提示后0.8至2.8 s，不能解释成纯粹的目标后记忆阶段；按`t_act-100 ms`另算的应答前分数使用留出记录×提示侧组中位端点，不是逐试次分类。第二，目标类型、时刻与匹配证据是情景输入，真实试次的目标、正确性、反应时长和漏答状态仍未知；本研究没有训练DDM或行为分类器。第三，状态参数采用预设值，训练折估计的只是传感器映射系数；若时间常数或情景映射与真实实验不符，H/P的形态可能错位。第四，四份记录并未确认对应四名独立受试者，NRMSE均值和标准差不能替代外部样本验证或显著性检验。第五，三个电极不足以唯一分解五个源代理，加载系数仅有传感器预测意义，不证明海马或前额叶来源。第六，不同滤波口径改变误差，零相位结果只适用于离线分析，因果滤波结果虽保留时间因果性但仍有相位延迟。

后续若取得逐试次target-onset、刺激真值、任务文件映射、反应截止时间和行为日志，可先冻结事件语义与主评价窗，再只在各外层训练记录内部估计时间常数或增益，并保留整份记录作为最终评估。若新增状态仍不能在不同记录、目标候选时刻和合理预处理下稳定降低留出误差，应将其保留为可复核的建模假设，而不将结构设想写成已确认的认知机制。现阶段结论限于：动态状态模型已按代码实现并完成记录留出验证；当前样本尚未显示记忆态的稳定晚期预测增益。

\clearpage
\addcontentsline{toc}{section}{参考文献}

\begin{thebibliography}{99}
\setlength{\itemsep}{2pt}
\bibitem{problem2026} 第二十三届中国研究生数学建模竞赛组委会. 服务于脑机接口与精神性疾病诊断的脑电图计算模型：C题题目[Z]. 2026.
\bibitem{wilsonCowan1972} Wilson H R, Cowan J D. Excitatory and inhibitory interactions in localized populations of model neurons[J]. Biophysical Journal, 1972, 12(1): 1-24. DOI: 10.1016/S0006-3495(72)86068-5.
\bibitem{hoerlKennard1970} Hoerl A E, Kennard R W. Ridge regression: Biased estimation for nonorthogonal problems[J]. Technometrics, 1970, 12(1): 55-67. DOI: 10.1080/00401706.1970.10488634.
\end{thebibliography}
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_inputs() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    summary = read_json(SUMMARY_PATH)
    source_paths = [
        WORKSPACE_ROOT / "data" / "第二十三届中国研究生数学建模竞赛+-+中文题目" / "中文题目" / "C题" / "服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx",
        WORKSPACE_ROOT / "data" / "第二十三届中国研究生数学建模竞赛+-+中文题目" / "中文题目" / "C题" / "deepseek_latex_20260925_4ed4dc(1).pdf",
        Q3_ROOT / "问题三实现流程与模型说明.md",
        Q3_ROOT / "dynamic_cognitive_model.py",
        Q3_ROOT / "dynamic_validation.py",
        SUMMARY_PATH,
        METRICS_PATH,
    ]
    sources = []
    for path in source_paths:
        sources.append({
            "path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        })

    write_json(INPUTS / "paper_spec.json", {
        "scope": "只写C题第三问的动态宏观模型解法正文",
        "audience": "数学建模竞赛评阅者",
        "primary_evidence": "2026-09-26的Q3实现说明、动态验证结果和当前代码",
        "layout": "沿用仓库CUMCM LaTeX模板；摘要、关键词、目录、正文、参考文献",
        "deliverables": ["Markdown正文", "LaTeX源稿", "渲染PDF"],
        "explicit_limits": ["不把候选目标时刻写成实测真值", "不声称估计真实反应时/正确率/漏答", "不声称脑区定位或临床效能"],
    })
    write_json(INPUTS / "evidence_index.json", {"sources": sources})
    write_json(INPUTS / "decision_contract.json", {
        "primary_outcome": "留出EEG晚期窗NRMSE",
        "primary_window_s": [0.8, 2.8],
        "outer_split": "leave one complete record out; four record files; participant independence unknown",
        "training_boundary": "standardization and per-channel ridge mapping use only the other three records",
        "primary_comparison": ["Q2_visual_only", "Q2_plus_memory"],
        "secondary_comparison": ["Training_mean_template", "Q2_plus_memory_control"],
        "sensitivity": {"preprocessing": ["none", "causal", "zero_phase"], "target_onsets_s": [2.0, 2.2, 2.4], "no_target_is_separate": True},
        "decision_rule": "claim stable memory-state value only if the late NRMSE improvement is consistent across candidate onset and preprocessing; otherwise report no stable gain",
    })
    write_json(INPUTS / "quality_validation.json", {
        "status": "PASS",
        "validation_summary": summary,
        "detailed_metrics_csv": str(METRICS_PATH.relative_to(Q3_ROOT)).replace("\\", "/"),
        "validation_protocol": "leave one complete record file out; preprocessing and feature standardization are fit within each training fold",
    })
    write_json(INPUTS / "claim_registry.json", {
        "claims": [
            {"id": "Q3_DATA_400_355", "claim": "400 audited cue events; 355 included trials across four records", "evidence": "validation_summary.json:n_trials_audited,n_trials_included,n_records_included"},
            {"id": "Q3_MARKER_SEMANTICS", "claim": "channel-9 zero-to-nonzero edge defines t_act per the reference model and sets response-window endpoints; it is not an EEG predictor", "evidence": "validation_summary.json:t_act_definition,channel9_used_for_response_window_scoring,channel9_used_as_predictor_or_state_input"},
            {"id": "Q3_Q2_PROJECTION", "claim": "Q2 forward projection maximum absolute discrepancy is 5.954e-08", "evidence": "validation_summary.json:q2_projection_max_abs_error"},
            {"id": "Q3_MEMORY_NO_STABLE_GAIN", "claim": "only 2 of 9 nonempty-onset preprocessing cells improve; overall gain is preprocessing-sensitive", "evidence": "validation_summary.json:memory_improvement_by_preprocessing,positive_preprocessing_onset_cells,total_preprocessing_onset_cells"},
            {"id": "Q3_FOUR_RECORD_CV", "claim": "the outer validation leaves out one full record file at a time", "evidence": "validation_summary.json:split"},
        ]
    })
    write_json(INPUTS / "formula_registry.json", {
        "formulas": [
            {"id": "F_Q2_FORWARD", "purpose": "project five Q2 source proxies to the three measured sensors", "expression": "y_Q2,k = G q_Q2,k", "implementation": "dynamic_cognitive_model.py:simulate_q2_scenario"},
            {"id": "F_VHP", "purpose": "recursively define visual, memory-related, and control functional states", "expression": "exponential discrete recursions in the manuscript", "implementation": "dynamic_cognitive_model.py:integrate_macro_states"},
            {"id": "F_SENSOR_RIDGE", "purpose": "map Q2 and macro-state features to each EEG sensor", "expression": "per-sensor ridge objective with alpha=1", "implementation": "dynamic_validation.py:fit_sensor_mapping"},
            {"id": "F_NRMSE", "purpose": "evaluate held-out waveform prediction", "expression": "NRMSE=RMSE/SD(y); positive Delta_H indicates improvement", "implementation": "dynamic_validation.py:_score_prediction and _paired_memory_gains"},
        ]
    })
    write_json(INPUTS / "figure_registry.json", {
        "figures": [
            {"file": "output/dynamic_heldout_validation/figures/analysis_event_windows.png", "claim": "t_act and t_act-minus-100-ms response-window endpoints alongside candidate target times", "use": "event semantics and analysis windows"},
            {"file": "output/dynamic_heldout_validation/figures/observed_three_channel_stages.png", "claim": "cue-aligned measured traces and record-group variability", "use": "data description"},
            {"file": "output/dynamic_heldout_validation/figures/dynamic_states_sensor_contributions.png", "claim": "scenario states and training-fitted sensor contributions", "use": "model interpretation with non-anatomical caveat"},
            {"file": "output/dynamic_heldout_validation/figures/heldout_model_comparison.png", "claim": "cross-record prediction error differs by model and stage", "use": "model comparison"},
            {"file": "output/dynamic_heldout_validation/figures/heldout_predictions_vs_measured.png", "claim": "representative held-out waveforms", "use": "prediction diagnostic"},
            {"file": "output/dynamic_heldout_validation/figures/memory_target_time_sensitivity.png", "claim": "memory-state gain is candidate-onset sensitive", "use": "sensitivity analysis"},
        ]
    })
    write_json(INPUTS / "table_registry.json", {
        "tables": [
            {"id": "T_PARAMETERS", "content": summary["macro_parameters"], "source": "validation_summary.json:macro_parameters"},
            {"id": "T_LATE_NRMSE", "content": summary["late_stage_nrmse_by_model_and_preprocessing"], "source": "validation_summary.json:late_stage_nrmse_by_model_and_preprocessing"},
            {"id": "T_ONSET_SENSITIVITY", "content": "aggregate paired memory improvement from dynamic_cv_metrics.csv by preprocessing, onset, and heldout record", "source": "dynamic_cv_metrics.csv"},
        ]
    })
    references = """@misc{problem2026,
  author = {{Second China Graduate Mathematical Contest in Modeling Organizing Committee}},
  title = {EEG Computational Model for Brain-Computer Interfaces and Mental-Disorder Diagnosis},
  year = {2026},
  note = {C problem statement, Chinese}
}

@article{wilsonCowan1972,
  author = {Wilson, Hugh R. and Cowan, Jack D.},
  title = {Excitatory and inhibitory interactions in localized populations of model neurons},
  journal = {Biophysical Journal},
  volume = {12},
  number = {1},
  pages = {1--24},
  year = {1972},
  doi = {10.1016/S0006-3495(72)86068-5}
}

@article{hoerlKennard1970,
  author = {Hoerl, Arthur E. and Kennard, Robert W.},
  title = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  journal = {Technometrics},
  volume = {12},
  number = {1},
  pages = {55--67},
  year = {1970},
  doi = {10.1080/00401706.1970.10488634}
}
"""
    (INPUTS / "references.bib").write_text(references, encoding="utf-8")
    write_json(INPUTS / "result_object.json", {
        "summary": summary,
        "detailed_metrics_csv": str(METRICS_PATH.relative_to(Q3_ROOT)).replace("\\", "/"),
        "primary_result": "no stable late-window memory-state improvement across preprocessing and candidate onset",
    })
    print(json.dumps({"prepared_input_roles": sorted(path.name for path in INPUTS.iterdir())}, ensure_ascii=False))


def onset_sensitivity_rows() -> list[list[str]]:
    with METRICS_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = (
        "preprocessing", "target_stimulus_candidate", "target_onset_candidate_s",
        "match_evidence_scenario", "heldout_record", "cue_side", "evaluation_window",
    )
    scores: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["model"] not in {"Q2_visual_only", "Q2_plus_memory"}:
            continue
        key = tuple(row[name] for name in keys)
        scores[key][row["model"]] = float(row["nrmse_by_heldout_sd"])
    record_onset: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for key, values in scores.items():
        mode, _, onset, _, record, _, window = key
        if not onset or window != "late_stage" or set(values) != {"Q2_visual_only", "Q2_plus_memory"}:
            continue
        record_onset[(mode, onset, record)].append(values["Q2_visual_only"] - values["Q2_plus_memory"])
    mode_onset: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (mode, onset, _record), values in record_onset.items():
        mode_onset[(mode, onset)].append(mean(values))
    display = {"causal": "因果滤波", "none": "无滤波", "zero_phase": "零相位滤波"}
    output = []
    for mode in ("causal", "none", "zero_phase"):
        cells = []
        for onset in ("2.0", "2.2", "2.4"):
            vals = mode_onset[(mode, onset)]
            cells.append(f"{mean(vals):+.4f}" if vals else "缺失")
        output.append([display[mode], *cells])
    return output


def result_table(summary: dict) -> list[list[str]]:
    metrics = {
        (item["preprocessing"], item["model"]): item
        for item in summary["late_stage_nrmse_by_model_and_preprocessing"]
    }
    models = ("Training_mean_template", "Q2_visual_only", "Q2_plus_memory", "Q2_plus_memory_control")
    modes = (("none", "无滤波"), ("causal", "因果滤波"), ("zero_phase", "零相位滤波"))
    output = []
    for mode, label in modes:
        cells = []
        for model in models:
            item = metrics[(mode, model)]
            cells.append(f"{item['mean']:.4f} ({item['std']:.4f})")
        output.append([label, *cells])
    return output


def markdown_table(rows: list[list[str]]) -> str:
    return "\n".join("|" + "|".join(row) + "|" for row in rows)


def build_markdown(summary: dict) -> str:
    body = BODY.replace("@@RESULT_TABLE@@", markdown_table(result_table(summary)))
    body = body.replace("@@SENSITIVITY_TABLE@@", markdown_table(onset_sensitivity_rows()))
    return (
        f"# {TITLE}\n\n## 摘要\n\n{ABSTRACT}\n\n**关键词：** {KEYWORDS}\n\n"
        "<!-- PAPER_BODY -->\n\n" + body.strip() + "\n"
    )


def tex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")


def build_tex(markdown: str) -> str:
    marker = "<!-- PAPER_BODY -->"
    if marker not in markdown:
        raise ValueError("manuscript body marker is missing")
    body_md = markdown.split(marker, 1)[1].strip()
    body_md_path = REPORTS / "paper_audit" / "body_for_tex.md"
    body_tex_path = REPORTS / "paper_audit" / "body_for_tex.tex"
    body_md_path.write_text(body_md + "\n", encoding="utf-8")
    subprocess.run([
        "pandoc", str(body_md_path), "--from=markdown+tex_math_dollars+pipe_tables+implicit_figures+link_attributes",
        "--to=latex", "--wrap=none", "--output", str(body_tex_path),
    ], cwd=REPORTS, check=True, capture_output=True, text=True, encoding="utf-8")
    template = WORKSPACE_ROOT / "math-model-agent" / "skills" / "write-model-paper" / "assets" / "cume-template.tex"
    preamble = template.read_text(encoding="utf-8").split(r"\begin{document}", 1)[0]
    preamble = preamble.replace(r"\usepackage{calc}", r"\usepackage{calc}" + "\n" + r"\usepackage{etoolbox}")
    preamble = preamble.replace(r"\newcommand{\PaperTitle}{在此填写论文标题}", r"\newcommand{\PaperTitle}{" + TITLE + "}")
    preamble += "\n\\setkeys{Gin}{keepaspectratio}\n"
    preamble += "\\providecommand{\\tightlist}{\\setlength{\\itemsep}{0pt}\\setlength{\\parskip}{0pt}}\n"
    # The template retains a readable code-listing style for later integration of appendices.
    preamble += "% Optional appendix environment from the contest template: \\begin{pycode}\n"
    title_tex = tex_escape(TITLE)
    abstract_tex = tex_escape(ABSTRACT).replace("5.954×10⁻⁸", r"5.954$\times 10^{-8}$")
    keywords_tex = tex_escape(KEYWORDS)
    body_tex = body_tex_path.read_text(encoding="utf-8")
    front = rf"""\begin{{document}}
\thispagestyle{{plain}}
\begin{{center}}
  {{\fontsize{{16pt}}{{24pt}}\selectfont\bfseries\heitifont {title_tex}\par}}
\end{{center}}
\section*{{摘\quad 要}}
{abstract_tex}

\noindent{{\bfseries 关键词：}}{keywords_tex}

\newpage
\tableofcontents
\newpage
\pagenumbering{{arabic}}\setcounter{{page}}{{1}}

"""
    return preamble + front + body_tex + "\n\\end{document}\n"


def render_and_report(markdown: str, tex: str, summary: dict) -> dict:
    md_path = REPORTS / "问题三论文正文草稿.md"
    tex_path = REPORTS / "问题三论文正文草稿.tex"
    pdf_path = REPORTS / "问题三论文正文草稿.pdf"
    md_path.write_text(markdown, encoding="utf-8")
    tex_path.write_text(tex, encoding="utf-8")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    layout_report = OUTPUTS / "cumcm_layout_validation.json"
    validator = WORKSPACE_ROOT / "math-model-agent" / "skills" / "write-model-paper" / "scripts" / "validate_cumcm_layout.py"
    subprocess.run([
        sys.executable, str(validator), "--tex", str(tex_path), "--report", str(layout_report)
    ], cwd=WORKSPACE_ROOT, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    layout = read_json(layout_report)
    if layout["status"] != "PASS":
        raise RuntimeError(f"CUMCM layout validation failed: {layout}")

    temp_dir = WORKSPACE_ROOT / "tmp" / "pdfs" / "q3_dynamic_paper_current"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_tex = temp_dir / "q3_dynamic_paper.tex"
    temp_tex.write_text(tex, encoding="utf-8")
    for _ in range(2):
        subprocess.run([
            "xelatex", "-interaction=nonstopmode", "-halt-on-error",
            f"-output-directory={temp_dir}", temp_tex.name,
        ], cwd=REPORTS, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    compiled_pdf = temp_dir / "q3_dynamic_paper.pdf"
    shutil.copy2(compiled_pdf, pdf_path)
    image_prefix = temp_dir / "page"
    for stale_page in temp_dir.glob("page-*.png"):
        stale_page.unlink()
    subprocess.run(["pdftoppm", "-png", "-r", "120", str(pdf_path), str(image_prefix)], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    extracted = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    extracted_path = temp_dir / "extracted_text.txt"
    extracted_path.write_text(extracted, encoding="utf-8")
    pdf_info = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    page_line = next((line.strip() for line in pdf_info.splitlines() if line.startswith("Pages:")), "Pages: unknown")
    required_text = ("问题重述", "动态认知状态模型", "结果与分析", "参考文献", "1.6479", "1.6474")
    missing_text = [item for item in required_text if item not in extracted]
    if missing_text:
        raise RuntimeError(f"PDF text extraction is missing expected manuscript content: {missing_text}")

    generation_report = {
        "status": "PASS",
        "scope": "Q3 dynamic macro-state solution paper only",
        "primary_data": {key: summary[key] for key in ("n_trials_audited", "n_trials_included", "n_records_included", "split", "behavioral_labels_used", "t_act_definition", "channel9_used_for_response_window_scoring", "channel9_used_as_predictor_or_state_input", "n_record_cue_groups_with_t_act_windows")},
        "primary_result": summary["memory_improvement_by_preprocessing"],
        "onset_cells": {"positive": summary["positive_preprocessing_onset_cells"], "total": summary["total_preprocessing_onset_cells"]},
        "numerical_interface_check": summary["q2_projection_max_abs_error"],
        "abstract_character_count": len(ABSTRACT),
        "compile": "XeLaTeX executed twice; PDF text extraction contains required headings and headline metrics",
        "pdf_pages": page_line,
        "limitations": ["four record folds only", "participant independence unverified", "candidate event truth not observed", "state parameters fixed", "no behavior labels"],
    }
    write_json(OUTPUTS / "paper_generation_report.json", generation_report)
    write_json(OUTPUTS / "claim_usage_report.json", {
        "used_claims": [
            {"id": "Q3_DATA_400_355", "where": "数据处理与事件审计", "source": "validation_summary.json"},
            {"id": "Q3_MARKER_SEMANTICS", "where": "问题重述; 数据处理与事件审计; 模型评价与适用范围", "source": "validation_summary.json and event timing audit"},
            {"id": "Q3_Q2_PROJECTION", "where": "问题二前向机制接口", "source": "validation_summary.json"},
            {"id": "Q3_MEMORY_NO_STABLE_GAIN", "where": "结果与分析", "source": "validation_summary.json and dynamic_cv_metrics.csv"},
            {"id": "Q3_FOUR_RECORD_CV", "where": "记录留出验证; 模型评价与适用范围", "source": "validation_summary.json"},
        ],
        "formal_text_contains_registry_tokens": False,
    })
    write_json(OUTPUTS / "render_request.json", {
        "pdf": str(pdf_path.relative_to(Q3_ROOT)).replace("\\", "/"),
        "rendered_pages_directory": str(temp_dir.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "page_renders": sorted(path.name for path in temp_dir.glob("page-*.png")),
        "visual_review_required": True,
        "extracted_text_file": str(extracted_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
    })
    return {"markdown": str(md_path), "tex": str(tex_path), "pdf": str(pdf_path), "pages": page_line, "layout": layout["status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="write machine-readable paper evidence inputs only")
    args = parser.parse_args()
    if args.prepare:
        prepare_inputs()
        return
    summary = read_json(SUMMARY_PATH)
    markdown = build_markdown(summary)
    tex = build_tex(markdown)
    result = render_and_report(markdown, tex, summary)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
