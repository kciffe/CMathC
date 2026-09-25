from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "问题二第一问_模块输入输出与验证流程.png"
OUT_SVG = ROOT / "问题二第一问_模块输入输出与验证流程.svg"
OUT_META = ROOT / "问题二第一问_模块输入输出与验证流程.figure.json"

font_path = r"C:\Windows\Fonts\msyh.ttc"
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

fig, ax = plt.subplots(figsize=(16, 8.2), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 15.4)
ax.set_ylim(0, 8.2)
ax.axis("off")

ink = "#243746"
blue = "#E9F1F8"
blue_edge = "#52799B"
green = "#EAF4EF"
green_edge = "#53836A"
amber = "#FFF4DF"
amber_edge = "#C18A32"
muted = "#60717F"


def box(x, y, w, h, title, body, face, edge, title_size=12, body_size=10.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=1.5, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=ink, linespacing=1.15)
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center",
            fontsize=body_size, color=ink, linespacing=1.35)


def arrow(x1, y1, x2, y2, color=muted, dashed=False, lw=1.7, curve=0.0):
    style = "dashed" if dashed else "solid"
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, linestyle=style,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)


ax.text(0.35, 7.72, "A  正向模型：视觉刺激到模拟头皮 EEG", fontsize=15,
        fontweight="bold", color=ink, va="center")

top_boxes = [
    (0.35, "视觉刺激输入", "PNG / NPY\n256 × 256 灰度矩阵"),
    (2.84, "frontend.py", "ON/OFF-LGN、Gabor\n形状与空间位置编码"),
    (5.33, "model.py", "3 组 E/I 功能群体\nWilson–Cowan 风格动力学"),
    (7.82, "源代理", "5 × T\n四路视野/构型对侧源\n+ 一路双侧偏好差中线源"),
    (10.31, "观测映射", "G：3行 × 5列\n头内源→表面电极\n无限导体点偶极近似"),
    (12.80, "模拟 EEG 输出", "3 × T\nF3 / Fz / F4"),
]
for x, title, body in top_boxes:
    box(x, 5.32, 2.05, 1.48, title, body, blue, blue_edge)
for i in range(len(top_boxes) - 1):
    x1 = top_boxes[i][0] + 2.05
    x2 = top_boxes[i + 1][0]
    arrow(x1 + 0.035, 6.06, x2 - 0.035, 6.06)

ax.text(0.35, 4.46, "B  实测数据与块留出验证：训练阶段和测试阶段严格隔离",
        fontsize=15, fontweight="bold", color=ink, va="center")

bottom_boxes = [
    (0.55, "第一问清洗数据", "*_clean.mat\ntrial_data + cue_type"),
    (3.50, "real_data.py", "提取 F3 / Fz / F4\n事件、标签与基线处理"),
    (6.45, "按时间分块", "每折留一块测试\n其余块只作训练"),
    (9.40, "训练块内估计", "拟合参数与 ERP 模板\n选择仅依据训练数据"),
    (12.35, "测试块评价", "ERP 与左右差异波\nRMSE / NRMSE / 相关"),
]
for x, title, body in bottom_boxes:
    box(x, 1.88, 2.35, 1.58, title, body, green, green_edge, title_size=11.5)
for i in range(len(bottom_boxes) - 1):
    x1 = bottom_boxes[i][0] + 2.35
    x2 = bottom_boxes[i + 1][0]
    arrow(x1 + 0.04, 2.67, x2 - 0.04, 2.67, color=green_edge)

# Model predictions are scored against the held-out real-data ERP.
arrow(13.83, 5.28, 13.62, 3.51, color=blue_edge, dashed=True, lw=1.8, curve=-0.12)
ax.text(14.28, 4.34, "模型预测\n只在测试块评分", ha="center", va="center",
        fontsize=9.5, color=blue_edge, linespacing=1.25)

warning = FancyBboxPatch(
    (0.55, 0.48), 14.15, 0.88,
    boxstyle="round,pad=0.03,rounding_size=0.08",
    linewidth=1.2, edgecolor=amber_edge, facecolor=amber,
)
ax.add_patch(warning)
ax.text(0.78, 0.92, "当前需审计", fontsize=11, fontweight="bold", color=amber_edge,
        ha="left", va="center")
ax.text(2.15, 0.92,
        "旧版拟合尚未按五源映射重跑；旧曲线与触边参数不代表当前实现，滤波上下文仍需审计。",
        fontsize=10.5, color=ink, ha="left", va="center")

fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
plt.close(fig)

metadata = {
    "claim": "展示修正后五源模型的模块输入输出，以及建议的训练块/测试块验证路径；区分左右视野与模板偏好。",
    "source_mapping_schema": "contralateral_visual_field_5source_midline_opponent_v1",
    "source": [
        "q2/revision_v3/config.py",
        "q2/revision_v3/frontend.py",
        "q2/revision_v3/model.py",
        "q2/revision_v3/head_model.py",
        "q2/revision_v3/observation.py",
        "q2/revision_v3/real_data.py",
        "q2/revision_v3/fit.py",
        "q2/revision_v3/evaluate.py",
    ],
    "units": "Source proxies are model-relative currents; T denotes time samples. The 3x5 leadfield is a homogeneous infinite-conductor approximation, not a finite spherical head solution.",
    "method": "Hand-laid module and validation flow diagram; boxes reflect current interfaces and proposed blocked validation.",
    "randomness": "none",
    "outputs": [str(OUT_PNG), str(OUT_SVG)],
}
OUT_META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(OUT_PNG)
print(OUT_SVG)
