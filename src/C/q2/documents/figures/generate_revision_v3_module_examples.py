from pathlib import Path
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


HERE = Path(__file__).resolve().parent
Q2_ROOT = HERE.parents[1]
REVISION = Q2_ROOT / "revision_v3"
INPUT = Q2_ROOT / "input"
REAL_ROOT = Q2_ROOT.parent / "q1" / "output" / "8riemann_denoise"
sys.path.insert(0, str(REVISION))

from frontend import load_stimulus, simulate_frontend
from model import ModelParams, simulate_forward
from head_model import SOURCE_LABELS, geometry_manifest
from observation import model_curve_to_q1_grid
from real_data import load_dataset


font_path = r"C:\Windows\Fonts\msyh.ttc"
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})

stim = load_stimulus("Stage1", "left")
front = simulate_frontend(stim, resolution=64, capture_spatial_audit=True)
model = simulate_forward(front, params=ModelParams(), amplitude=1.0)
observed, observed_time_ms = model_curve_to_q1_grid(model.eeg_scaled, model.time_ms)

# Actual trial-data input and the output of real_data.py's cue-aligned ERP path.
record_path = REAL_ROOT / "VisualCogA_Task-1_clean.mat"
data = load_dataset(record_path)
time_s = data["time_s"]
baseline_mask = (time_s >= -0.2) & (time_s < 0.0)
corrected = data["eeg"] - data["eeg"][:, :, baseline_mask].mean(axis=2, keepdims=True)
cue_time_ms = time_s * 1000.0
epoch_mask = (cue_time_ms >= 0) & (cue_time_ms <= 800)
left_mask = data["cue_type"] == -1
right_mask = data["cue_type"] == 1
left_trials = corrected[left_mask][:, :, epoch_mask]
right_trials = corrected[right_mask][:, :, epoch_mask]
erp_time = cue_time_ms[epoch_mask]

colors = {"left": "#356B9A", "right": "#C66A4A"}
channels = ("F3", "Fz", "F4")

# Figure 1: actual intermediate products of the current revision_v3 frontend.
fig = plt.figure(figsize=(13.8, 8.8), dpi=300, layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
fig.suptitle("revision_v3 前端模块：实际输入与中间输出示例", fontsize=17, fontweight="bold")

ax = fig.add_subplot(gs[0, 0])
sub = ax.inset_axes([0.02, 0.1, 0.43, 0.8])
sub.imshow(stim.image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
sub.set_title("cue image\n256×256", fontsize=11)
sub.axis("off")
sub = ax.inset_axes([0.54, 0.1, 0.43, 0.8])
im = sub.imshow(stim.signed_contrast, cmap="coolwarm", vmin=-0.5, vmax=0.5,
                interpolation="nearest")
sub.set_title("signed contrast\n(image − baseline)/255", fontsize=11)
sub.axis("off")
ax.set_title("输入：标准刺激矩阵与刺激增量", fontsize=13, fontweight="bold")
ax.axis("off")
fig.colorbar(im, ax=ax, shrink=0.72, label="相对对比度")

ax = fig.add_subplot(gs[0, 1])
ax.plot(front["time_ms"], front["lgn_off_mean"], color="#356B9A", label="OFF relay")
ax.plot(front["time_ms"], front["lgn_on_mean"], color="#C66A4A", label="ON relay")
ax.axvline(0, color="0.45", lw=0.9)
ax.axvline(200, color="0.45", lw=0.9, ls="--", label="cue offset")
ax.set(xlim=(0, 800), xlabel="提示后时间 (ms)", ylabel="群体平均活动（相对单位）")
ax.set_title("frontend.py 输出：LGN ON/OFF 时间序列", fontsize=13, fontweight="bold")
ax.grid(alpha=0.2)
ax.legend(frameon=False, ncol=3, fontsize=10)

audit = front["audit_maps"][100.0]
ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.set_title("frontend.py 输出：100 ms 的四方向 Gabor 图", fontsize=13, fontweight="bold")
gabor = audit["V1"]
labels = ("0°", "45°", "90°", "135°")
for i in range(4):
    sub = ax.inset_axes([0.03 + (i % 2) * 0.49, 0.05 + (1 - i // 2) * 0.48, 0.44, 0.42])
    sub.imshow(gabor[i], cmap="magma", vmin=0, vmax=max(float(gabor.max()), 1e-8),
               interpolation="nearest")
    sub.set_title(labels[i], fontsize=10)
    sub.axis("off")

ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
ax.set_title("frontend.py 输出：形状共同图与左右对手图（100 ms）", fontsize=13, fontweight="bold")
maps = [(audit["shape"], "shape = H_L + H_R", "magma", 0.0, None),
        (audit["direction"], "direction = H_R − H_L", "coolwarm", None, None)]
for i, (values, title, cmap, vmin, vmax) in enumerate(maps):
    sub = ax.inset_axes([0.04 + i * 0.49, 0.06, 0.44, 0.82])
    if title.startswith("direction"):
        limit = max(float(np.max(np.abs(values))), 1e-8)
        vmin, vmax = -limit, limit
    else:
        vmax = max(float(values.max()), 1e-8)
    sub.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    sub.set_title(title, fontsize=10)
    sub.axis("off")

fig.text(0.5, -0.01,
         f"示例为 Stage1 左提示；前端输出 gabor={front['gabor'].shape}，"
         f"LGN pooled={front['lgn_on'].shape}，空间图分辨率={front['resolution']}×{front['resolution']}。",
         ha="center", fontsize=10.5, color="#405261")
fig.savefig(HERE / "revision_v3_前端模块I_O示例.png", dpi=450, bbox_inches="tight")
fig.savefig(HERE / "revision_v3_前端模块I_O示例.svg", bbox_inches="tight")
plt.close(fig)

# Figure 2: actual trial-derived EEG I/O and the current model's downstream states.
fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.3), dpi=300, layout="constrained")
fig.suptitle("revision_v3 数据、皮层与观测模块：实际 I/O 示例", fontsize=17, fontweight="bold")

ax = axes[0, 0]
for values, label, color in ((left_trials, "左提示", colors["left"]),
                             (right_trials, "右提示", colors["right"])):
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    for ch in range(3):
        ax.plot(erp_time, mean[ch], color=color, lw=1.6,
                ls=("-", "--", ":")[ch], label=f"{label} · {channels[ch]}")
ax.axvline(0, color="0.45", lw=0.8)
ax.axhline(0, color="0.65", lw=0.7)
ax.set(xlim=(0, 800), xlabel="提示后时间 (ms)", ylabel="基线校正 EEG（源数据单位）")
ax.set_title("real_data.py 输出：VisualCogA_Task-1 Stage1 条件 ERP", fontsize=13, fontweight="bold")
ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=9.5, ncol=2)

ax = axes[0, 1]
population_names = ("early visual field", "spatial configuration", "triangle-template preference")
pop_colors = ("#356B9A", "#53836A", "#A66C9A")
for p, (name, color) in enumerate(zip(population_names, pop_colors)):
    ax.plot(model.time_ms, model.excitatory[p].mean(axis=0), color=color, lw=1.7,
            label=f"{name} E")
    ax.plot(model.time_ms, model.inhibitory[p].mean(axis=0), color=color, lw=1.3,
            ls="--", label=f"{name} I")
state_max = max(float(model.excitatory.max()), float(model.inhibitory.max()))
ax.set(xlim=(0, 800), ylim=(0, max(state_max * 1.1, 0.02)),
       xlabel="提示后时间 (ms)", ylabel="群体活动（局部放大；状态上限 1）")
ax.set_title("model.py 输出：三组 E/I 群体状态", fontsize=13, fontweight="bold")
ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=9, ncol=2)

ax = axes[1, 0]
source_names = (
    "早期：左半球←右视野",
    "早期：右半球←左视野",
    "构型：左半球←右视野",
    "构型：右半球←左视野",
    "模板偏好差：双侧中线",
)
if len(source_names) != len(SOURCE_LABELS) or model.source_proxy.shape[0] != len(SOURCE_LABELS):
    raise ValueError("source plot labels do not match the current source mapping")
source_colors = ("#356B9A", "#78A6C8", "#53836A", "#90B39A", "#A66C9A")
for row, (name, color) in enumerate(zip(source_names, source_colors)):
    ax.plot(model.time_ms, model.source_proxy[row], color=color, lw=1.3, label=name)
ax.axhline(0, color="0.65", lw=0.7)
ax.set(xlim=(0, 800), xlabel="提示后时间 (ms)", ylabel="源代理（相对单位）")
ax.set_title("model.py 输出：五路语义路由后的滤波 E−I 源代理", fontsize=13, fontweight="bold")
ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=9.2, ncol=2)

ax = axes[1, 1]
sensor_names = ("F3", "Fz", "F4")
sensor_colors = ("#356B9A", "#53836A", "#C66A4A")
obs_mask = (observed_time_ms >= 0) & (observed_time_ms <= 800)
for ch, (name, color) in enumerate(zip(sensor_names, sensor_colors)):
    ax.plot(observed_time_ms[obs_mask], observed[ch, obs_mask], color=color,
            lw=1.6, label=name)
ax.axhline(0, color="0.65", lw=0.7)
ax.set(xlim=(0, 800), xlabel="提示后时间 (ms)", ylabel="模拟观测（相对单位）")
ax.set_title("head_model + observation 输出：三电极模拟波形", fontsize=13, fontweight="bold")
ax.grid(alpha=0.2)
ax.legend(frameon=False, ncol=3, fontsize=9.5)

fig.text(0.5, -0.01,
         "源顺序：早期/构型左视野→右半球、右视野→左半球；左右模板偏好差→双侧中线对手源。"
         "模板偏好组接收左右视野早期活动的固定等权平均；中线对手源为建模假设。"
         "实测 ERP 来自 VisualCogA_Task-1_clean.mat；模型曲线为默认参数示例，不是拟合或留出成绩。",
         ha="center", fontsize=10.5, color="#405261")
fig.savefig(HERE / "revision_v3_真实数据与皮层观测模块I_O示例.png", dpi=450, bbox_inches="tight")
fig.savefig(HERE / "revision_v3_真实数据与皮层观测模块I_O示例.svg", bbox_inches="tight")
plt.close(fig)

metadata = {
    "claim": "展示 revision_v3 当前视觉前端从标准刺激到 LGN、方向 Gabor 与形状/方向编码的实际 I/O。",
    "sources": [
        "q2/input/stage1_cue_left.npy",
        "q2/revision_v3/frontend.py",
    ],
    "scenarios": ["Stage1 left cue"],
    "model_parameters": {"frontend_tau_adaptation_ms": 80.0, "resolution": 64},
    "processing": "Deterministic frontend response; no ERP fit or classifier is used.",
    "randomness": "none",
    "dimensions": {
        "image": list(stim.image.shape),
        "frontend_gabor": list(front["gabor"].shape),
        "frontend_lgn_on_pooled": list(front["lgn_on"].shape),
    },
    "notes": "Gabor and LGN outputs are model features in relative units, not measured neural activity.",
}
(HERE / "revision_v3_模块I_O示例.figure.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

metadata = {
    "claim": "以当前五源 revision_v3 正向模型展示真实 ERP 输入、三组 E/I 群体、视野/模板偏好分离的语义路由源代理与简化 F3/Fz/F4 观测。",
    "sources": [
        "q1/output/8riemann_denoise/VisualCogA_Task-1_clean.mat",
        "q2/revision_v3/model.py",
        "q2/revision_v3/head_model.py",
        "q2/revision_v3/observation.py",
        "q2/revision_v3/real_data.py",
    ],
    "scenarios": ["VisualCogA_Task-1 Stage1 observed ERP", "Stage1 left cue default-parameter model illustration"],
    "model_parameters": {"tau_s_ms": 40.0, "g_i": 1.0, "tau_a_ms": 80.0, "amplitude": 1.0},
    "source_mapping_schema": "contralateral_visual_field_5source_midline_opponent_bilateral_ff_v2",
    "source_labels": list(SOURCE_LABELS),
    "head_geometry": geometry_manifest(),
    "processing": "Real trial ERP is baseline-corrected per trial over [-200,0) ms. Model observation uses revision_v3 observation.py with Q1-style filter/resample/baseline operator.",
    "notes": "Model curves use default parameters and are illustrative only; no fit or held-out score is shown. Shape-preference channels receive the same fixed equal-weight bilateral early-field pool. The bilateral midline opponent source is a modeling hypothesis. Source proxies are relative units; leadfield uses a homogeneous infinite-conductor point-dipole approximation.",
    "randomness": "none",
    "dimensions": {
        "mat_trial_data": list(data["eeg"].shape[:1]) + [10, 512],
        "selected_eeg": list(data["eeg"].shape),
        "frontend_gabor": list(front["gabor"].shape),
        "frontend_lgn_on_pooled": list(front["lgn_on"].shape),
        "model_e_i": list(model.excitatory.shape),
        "source_proxy": list(model.source_proxy.shape),
        "sensor_eeg": list(model.eeg_scaled.shape),
        "observation_grid": list(observed.shape),
    },
}
(HERE / "revision_v3_真实数据与皮层观测模块I_O示例.figure.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("generated:", HERE / "revision_v3_前端模块I_O示例.png")
print("generated:", HERE / "revision_v3_真实数据与皮层观测模块I_O示例.png")
print("real-data trials:", data["eeg"].shape, "left/right:", left_trials.shape[0], right_trials.shape[0])
print("frontend gabor / LGN:", front["gabor"].shape, front["lgn_on"].shape)
print("model E/I / source / eeg:", model.excitatory.shape, model.source_proxy.shape, model.eeg.shape)
print("observed grid:", observed.shape)
