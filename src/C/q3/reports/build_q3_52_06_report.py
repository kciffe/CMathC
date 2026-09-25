from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


Q3 = Path(__file__).resolve().parents[1]
REPORT_DIR = Q3 / "reports"
FIG_DIR = REPORT_DIR / "figures"
BASE = Q3 / "baselines" / "20260925_initial" / "output"
EXP = Q3 / "output" / "experiments" / "20260925_model_search_v1"
FEATURE_CSV = BASE / "trial_features.csv"
SUMMARY_JSON = EXP / "summary.json"
SELECTED_CSV = EXP / "nested_selected_fold_metrics.csv"
BASE_CANDIDATE_CSV = EXP / "outer_candidate_metrics.csv"
BASE_PRED_CSV = EXP / "baseline_oof_predictions.csv"
SELECTED_PRED_CSV = EXP / "nested_selected_oof_predictions.csv"
SUBGROUP_CSV = EXP / "quality_subgroup_metrics.csv"
MODEL_SEARCH = Q3 / "experiments" / "model_search.py"

RECORDS = ["VisualCogA_Task-1", "VisualCogA_Task-2", "VisualCogB_Task-1", "VisualCogB_Task-2"]
RECORD_LABELS = ["A·任务1", "A·任务2", "B·任务1", "B·任务2"]
NAVY = "#174A6E"
BLUE = "#3F83B5"
CYAN = "#69A9B9"
ORANGE = "#E58B45"
RED = "#C85C5C"
GREEN = "#4F9676"
GRAY = "#77838B"
LIGHT = "#EAF1F5"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 15,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig, code: str, title: str, purpose: str, sources: list[Path], notes: dict | None = None):
    png = FIG_DIR / f"{code}.png"
    svg = FIG_DIR / f"{code}.svg"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {
        "figure": code,
        "title": title,
        "purpose": purpose,
        "data_sources": [str(path.resolve()) for path in sources],
        "source_sha256": {str(path.resolve()): source_hash(path) for path in sources if path.is_file()},
        "render": {"png_dpi": 320, "vector_master": svg.name, "background": "white"},
        "notes": notes or {},
    }
    (FIG_DIR / f"{code}.figure.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def add_box(ax, xy, w, h, text, face, edge=NAVY, fs=10, color="#1E2B33", weight="normal"):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=face,
                         edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center", fontsize=fs,
            color=color, weight=weight, wrap=True)


def arrow(ax, start, end, color=GRAY, style="-|>", lw=1.4, ls="-"):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": style, "color": color,
                "lw": lw, "linestyle": ls, "shrinkA": 2, "shrinkB": 2})


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(13.2, 4.5))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 4.5); ax.axis("off")
    xs = [0.25, 2.35, 4.45, 6.55, 8.65, 10.75]
    texts = [
        "四份连续 EEG\nF3 / Fz / F4\n256 Hz",
        "事件对齐\nVisCue + 时间戳\n提示锁定",
        "滤波与质量控制\n0.5–30 / 1–80 Hz\n保留有效 trial",
        "13 维 trial 特征\nERP 3 + 功率 9\nα不对称 1",
        "49 个预设候选\n内层按 BA 选择\n训练折内标准化",
        "留一记录外测\n四个外层记录\n与固定基线比较",
    ]
    colors = ["#EDF3F7", "#E6F0F5", "#EAF3F0", "#F7F0E7", "#EEF0F8", "#EAF1F5"]
    for x, t, c in zip(xs, texts, colors):
        add_box(ax, (x, 1.65), 1.7, 1.3, t, c, fs=9.4)
    for i in range(5):
        arrow(ax, (xs[i]+1.72, 2.3), (xs[i+1]-0.05, 2.3))
    ax.text(6.6, 3.85, "52.06% 对应的独立验证分支：提示出现后 0–500 ms 的左右提示解码",
            ha="center", va="center", color=NAVY, fontsize=13, weight="bold")
    ax.text(6.6, 0.72,
            "边界：目标是 VisCue 左/右标签；不是正确率、疾病诊断率，也不是完整 V/H/P + DDM 总模型分数。",
            ha="center", va="center", fontsize=10, color=RED,
            bbox={"boxstyle":"round,pad=0.45", "facecolor":"#FBF1EF", "edgecolor":"#E6C3BD"})
    save_figure(fig, "F01_pipeline", "分析流程与结果边界", "展示从原始 EEG 到外层验证的步骤及 52.06% 结果所对应的任务范围。",
                [FEATURE_CSV, SUMMARY_JSON, SELECTED_CSV])


def figure_counts(features: pd.DataFrame):
    cue = features[(features.stage == "cue_locked") & features.qc_valid.fillna(False).astype(bool)].copy()
    target = features[(features.stage == "target_locked") & features.qc_valid.fillna(False).astype(bool)].copy()
    q1_cue = cue[cue.q1_quality_pass.fillna(False).astype(bool)]
    q1_target = target[target.q1_quality_pass.fillna(False).astype(bool)]
    series = [
        ("Cue · raw-QC", cue, NAVY), ("Cue · Q1 matched", q1_cue, CYAN),
        ("Target · raw-QC", target, ORANGE), ("Target · Q1 matched", q1_target, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    width = .19
    x = np.arange(len(RECORDS))
    for k, (label, frame, color) in enumerate(series):
        counts = frame.groupby("record").size().reindex(RECORDS, fill_value=0).to_numpy()
        positions = x + (k-1.5)*width
        bars = ax.bar(positions, counts, width, label=label, color=color, edgecolor="white", linewidth=.6)
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x, RECORD_LABELS)
    ax.set_ylabel("通过条件的试次数")
    ax.set_ylim(0, 125)
    ax.grid(axis="y", alpha=.22)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.17))
    fig.suptitle("trial 筛选结果按记录与事件阶段分布", y=.99, color=NAVY, weight="bold")
    fig.subplots_adjust(top=.77)
    save_figure(fig, "F02_trial_counts", "Trial 数量与筛选口径", "比较 cue/target 阶段的 raw-QC 与 Q1 质量匹配样本数量。",
                [FEATURE_CSV, SUMMARY_JSON], {"raw_qc_rules": "finite values; no absolute value >=999.5; no channel flatline; trial feature completeness",
                 "q1_matched": "Q1 time/quality mapping only; Q1 EEG samples are not the feature source"})


def figure_windows():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.8, 6.6), gridspec_kw={"height_ratios":[1.3,1]}, constrained_layout=True)
    ax.set_xlim(-.2, 3.15); ax.set_ylim(-.35, 3.25); ax.set_yticks([2.6, 1.7, .8], ["Cue epoch", "Target epoch", "Response-anchored window"])
    ax.axvline(0, color=NAVY, lw=1.5)
    ax.text(0, 3.0, "事件锚点", ha="center", color=NAVY, fontsize=9)
    ax.barh(2.6, .6, left=-.1, height=.32, color=BLUE, alpha=.8)
    ax.barh(2.6, .25, left=.25, height=.48, color=ORANGE, alpha=.9)
    ax.text(.2, 2.6, "cue epoch [-0.10, 0.50] s", ha="center", va="center", color="white", fontsize=8)
    ax.text(.375, 2.12, "ERP 均值窗\n[0.25, 0.50) s", ha="center", va="center", color="#6C3C12", fontsize=8)
    # target anchor is cue + 2.2 seconds; display a compressed interval with a break.
    ax.plot([.9, 1.55], [1.7,1.7], color=GRAY, lw=1.5, ls="--")
    ax.text(1.22, 1.84, "时间压缩", ha="center", color=GRAY, fontsize=8)
    ax.axvline(1.7, color=ORANGE, lw=1.3, ls="--")
    ax.barh(1.7, .9, left=1.6, height=.32, color=ORANGE, alpha=.65)
    ax.text(2.05, 1.7, "target epoch [-0.10, 0.80] s", ha="center", va="center", color="#573817", fontsize=8)
    ax.plot([1.55, 1.62], [1.7,1.7], color=GRAY, lw=1.5)
    ax.text(1.7, 1.98, "cue+2.2 s\n(排程假设)", ha="center", color=ORANGE, fontsize=8)
    ax.plot([0, 2.8], [.8,.8], color=GREEN, lw=5, solid_capstyle="round")
    ax.plot([2.8, 2.8], [.57,1.03], color=RED, lw=2)
    ax.text(1.35, .98, "从提示到应答标记前 100 ms（长度因 trial 而异）", ha="center", color=GREEN, fontsize=8.5)
    ax.text(2.8, .45, "t_act − 0.10 s", ha="center", color=RED, fontsize=8)
    ax.set_xlabel("相对提示的时间（s；target 区间为示意）")
    ax.grid(axis="x", alpha=.18)
    bx.set_xlim(0, 82); bx.set_ylim(0, 2.8); bx.set_yticks([2.1,1.25,.4], ["ERP 分支", "功率分支", "Welch 功率积分带"])
    bx.barh(2.1, 29.5, left=.5, height=.35, color=BLUE)
    bx.text(15.25,2.1,"0.5–30 Hz · Butterworth 4阶 · 零相位",ha="center",va="center",color="white",fontsize=8.3)
    bx.barh(1.25, 79, left=1, height=.35, color=CYAN)
    bx.text(40.5,1.25,"1–80 Hz · Butterworth 4阶 · 零相位",ha="center",va="center",color="white",fontsize=8.3)
    bands=[("θ 4–8",4,4,BLUE),("α 8–13",8,5,ORANGE),("β 13–30",13,17,GREEN),("γ 30–80*",30,50,GRAY)]
    for label,left,w,c in bands:
        bx.barh(.4,w,left=left,height=.28,color=c,alpha=.85)
        bx.text(left+w/2,.4,label,ha="center",va="center",color="white",fontsize=7.7)
    bx.set_xlabel("频率（Hz）")
    bx.grid(axis="x",alpha=.18)
    bx.text(.99,.02,"*γ功率虽被提取，但不在本次13维主模型中。",transform=bx.transAxes,ha="right",color=GRAY,fontsize=8)
    save_figure(fig, "F03_windows_filters", "事件窗口与信号分支", "对齐 cue/target 和应答终点窗口，并展示实际滤波与频段定义。",
                [Q3 / "config.py", Q3 / "02b_preprocess_raw.py", Q3 / "signal_processing.py"],
                {"primary_52_06_window": "cue epoch [-0.10,0.50] s; ERP feature [0.25,0.50); power [0,0.50)",
                 "target_anchor": "cue + 2.2 s schedule assumption", "response_window": "not used for the 52.06% classifier"})


def figure_feature_schema():
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(6,5.65,"主分类输入：每个有效 cue trial 一行、13 个预先定义的 EEG 数值特征",ha="center",fontsize=13,weight="bold",color=NAVY)
    cols=["F3","Fz","F4"]
    rows=["ERP 均值 · 250–500 ms","θ功率 · 4–8 Hz","α功率 · 8–13 Hz","β功率 · 13–30 Hz"]
    startx=2.6; starty=4.6; cw=1.5; ch=.72; gapx=.25; gapy=.18
    for i,row in enumerate(rows):
        y=starty-i*(ch+gapy)
        ax.text(2.35,y+ch/2,row,ha="right",va="center",fontsize=9,color="#39464F")
        for j,c in enumerate(cols):
            x=startx+j*(cw+gapx)
            color=["#D9EAF4","#E4F1ED","#F8E8D8","#E8EAF5"][i]
            add_box(ax,(x,y),cw,ch,f"{row.split(' · ')[0]}_{c}",color,edge="#B8C8D1",fs=8.3)
            ax.text(x+cw/2,y-.13,"log power" if i>0 else "baseline-corrected",ha="center",va="top",fontsize=7,color=GRAY)
    add_box(ax,(8.7,3.0),2.25,.8,"AIα = logPα(F4) − logPα(F3)","#F7E7DA",edge=ORANGE,fs=9)
    ax.text(9.82,2.55,"与三通道 α 功率存在代数冗余",ha="center",fontsize=8,color=RED)
    ax.text(9.82,2.18,"full 13维：包含 AIα",ha="center",fontsize=9,color=NAVY)
    ax.text(9.82,1.85,"ablation 12维：删除 AIα",ha="center",fontsize=9,color=NAVY)
    ax.text(6,0.55,"γ功率、ERP峰值/潜伏期、不对称指数、通道9应答码均未进入本次候选特征矩阵。",
            ha="center",fontsize=9,color=GRAY,bbox={"boxstyle":"round,pad=.4","facecolor":"#F5F7F8","edgecolor":"#D8E0E4"})
    save_figure(fig,"F04_feature_schema","13维 EEG 特征设计","明确展示进入49候选模型的输入列、频段与冗余特征处理。",
                [SUMMARY_JSON, Q3 / "signal_processing.py", MODEL_SEARCH],
                {"selected_feature_families": {"ERP_mean": 3,"log_bandpower_theta_alpha_beta": 9,"alpha_asymmetry": 1},
                 "excluded": ["gamma power", "peak latency", "response marker", "pre-response features"]})


def figure_feature_contrast(features: pd.DataFrame):
    cue=features[(features.stage=="cue_locked") & features.qc_valid.fillna(False).astype(bool)].copy()
    feature_names=["erp_mean_F3","erp_mean_Fz","erp_mean_F4"]+[f"log_{band}_power_{ch}" for band in ("theta","alpha","beta") for ch in ("F3","Fz","F4")]+["AI_alpha"]
    labels=["ERP F3","ERP Fz","ERP F4"]+[f"{band.upper()} {ch}" for band in ("theta","alpha","beta") for ch in ("F3","Fz","F4")]+["α asym"]
    effect=[]
    for col in feature_names:
        a=cue.loc[cue.cue_side==-1,col].dropna().to_numpy(float)
        b=cue.loc[cue.cue_side==1,col].dropna().to_numpy(float)
        pooled=math.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
        effect.append((np.mean(b)-np.mean(a))/pooled if pooled>0 else 0.)
    fig,ax=plt.subplots(figsize=(9.5,5.8))
    y=np.arange(len(labels))
    colors=[BLUE if x<0 else ORANGE for x in effect]
    ax.barh(y,effect,color=colors,alpha=.88)
    ax.axvline(0,color="#4F5C64",lw=.8)
    ax.set_yticks(y,labels); ax.invert_yaxis()
    ax.set_xlabel("标准化均值差 d（右提示 − 左提示）")
    ax.set_title("全 raw-QC cue 样本中的特征差异（描述性分析）")
    ax.grid(axis="x",alpha=.2)
    save_figure(fig,"F05_feature_contrast","输入特征的左右提示描述差异","检查主特征矩阵是否存在明显的单变量左右提示分离；该图是事后描述性证据。",
                [FEATURE_CSV],{"metric":"Cohen pooled-SD standardized mean difference; right minus left",
                 "use_in_model_selection":False,"sample":"387 raw-QC cue-locked trials"})


def load_candidate_grid():
    spec=importlib.util.spec_from_file_location("q3_model_search",MODEL_SEARCH)
    module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.candidate_grid()


def figure_candidate_grid(candidates):
    counts=Counter(c.model for c in candidates)
    order=["logistic","lda","linear_svm","rbf_svm"]
    names=["Logistic","LDA","Linear SVM","RBF SVM"]
    values=[counts.get(x,0) for x in order]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={"width_ratios":[.9,1.35]})
    bars=ax.bar(names,values,color=[NAVY,CYAN,ORANGE,GREEN])
    ax.bar_label(bars,fontsize=10,padding=3)
    ax.set_ylim(0,max(values)+5); ax.set_ylabel("候选配置数"); ax.set_title(f"预设搜索空间：{len(candidates)} 个候选")
    ax.tick_params(axis="x",rotation=0,labelsize=8.5); ax.grid(axis="y",alpha=.18)
    bx.axis("off")
    grid=[
        ["维度","预设值 / 选项"],
        ["特征集","full-13；去冗余12；ERP-only 3；bandpower-only 9"],
        ["Logistic L2","0.1、10、100；固定基线 λ=1"],
        ["Robust Logistic L2","0.1、1、10；full-13 / 去冗余12"],
        ["Linear SVM C","0.01、0.1、1、10；full-13 / 去冗余12"],
        ["RBF SVM C","0.1、1、10；full-13 / 去冗余12"],
        ["RBF SVM γ","scale、0.03、0.1"],
        ["LDA","solver=lsqr；shrinkage=auto；4种特征集"],
    ]
    table=bx.table(cellText=grid[1:],colLabels=grid[0],loc="center",cellLoc="left",colWidths=[.28,.72])
    table.auto_set_font_size(False); table.set_fontsize(8.2); table.scale(1,1.65)
    for (r,c),cell in table.get_celld().items():
        if r==0: cell.set_facecolor(NAVY); cell.set_text_props(color="white",weight="bold")
        elif r%2: cell.set_facecolor("#F2F6F8")
        else: cell.set_facecolor("white")
        cell.set_edgecolor("#D5DFE4")
    bx.set_title("参数在进入外层测试前预先限定",pad=12)
    fig.suptitle("候选模型与参数网格",y=1.03,color=NAVY,weight="bold")
    save_figure(fig,"F06_candidate_grid","模型族与超参数搜索网格","呈现49个候选的组成与取值，便于论文复述参数设计。",
                [MODEL_SEARCH, SUMMARY_JSON],{"candidate_count":len(candidates),"family_count":dict(counts),"selection_metric":"inner-fold mean balanced accuracy"})


def figure_nested_cv():
    fig,ax=plt.subplots(figsize=(12.2,5.4))
    ax.set_xlim(0,12.2); ax.set_ylim(0,5.4); ax.axis("off")
    ax.text(6.1,5.05,"每个外层折：整份记录只作为最终测试；内层三折只负责挑选配置",ha="center",fontsize=13,weight="bold",color=NAVY)
    # Outer fold cards
    add_box(ax,(.35,2.4),2.1,1.5,"外层测试记录\n1份完整文件\n约93–100 trial","#FBEBDD",edge=ORANGE,fs=10)
    add_box(ax,(3.05,2.4),2.35,1.5,"外层训练集合\n其余3份完整文件\n约287–294 trial","#E7F0F6",edge=BLUE,fs=10)
    arrow(ax,(2.5,3.15),(3.0,3.15),style="<->",color=GRAY)
    add_box(ax,(6.0,3.45),2.15,1.05,"内层折 1\n留出记录 A","#EAF3F0",edge=GREEN,fs=9)
    add_box(ax,(6.0,2.1),2.15,1.05,"内层折 2\n留出记录 B","#EAF3F0",edge=GREEN,fs=9)
    add_box(ax,(6.0,.75),2.15,1.05,"内层折 3\n留出记录 C","#EAF3F0",edge=GREEN,fs=9)
    arrow(ax,(5.45,3.15),(5.92,3.98)); arrow(ax,(5.45,3.15),(5.92,2.62)); arrow(ax,(5.45,3.15),(5.92,1.27))
    add_box(ax,(8.8,2.25),2.75,1.7,"49候选逐个评估\n训练折拟合缩放 + 分类器\n平均3个内层 BA\n选择 BA 最高者","#EEF0F8",edge=NAVY,fs=9.5)
    arrow(ax,(8.2,3.98),(8.75,3.35)); arrow(ax,(8.2,2.62),(8.75,3.1)); arrow(ax,(8.2,1.27),(8.75,2.8))
    arrow(ax,(7.95,.42),(9.55,.42),color=RED)
    ax.text(8.75,.18,"定参后在完整外层训练集重拟合，再对留出记录评分一次",ha="center",fontsize=8.6,color=RED)
    ax.text(.45,.55,"四个外层记录轮流留出；主结果 = 四个外层 balanced accuracy 的算术平均。",
            ha="left",fontsize=9.4,color=NAVY)
    save_figure(fig,"F07_nested_validation","嵌套留一记录验证结构","展示外层评估和内层模型选择的隔离方式，及训练折内缩放规则。",
                [MODEL_SEARCH, SELECTED_CSV],{"outer":"leave one recording out; 4 groups",
                 "inner":"leave one of three outer-training recordings out",
                 "primary_aggregation":"unweighted mean of four outer-fold balanced accuracies",
                 "threshold_tuning":False,"scaling":"fit on train split only"})


def figure_primary_result(summary, selected):
    s=summary["selections_and_deltas"]
    item=next(x for x in s if x["sample"]=="raw_qc_pass" and x["stage"]=="cue_locked")
    b=[100*item["baseline_fold_balanced_accuracy"][r] for r in RECORDS]
    z=[100*item["selected_fold_balanced_accuracy"][r] for r in RECORDS]
    d=[100*item["delta_by_fold_percentage_points"][r] for r in RECORDS]
    fig,ax=plt.subplots(figsize=(10.5,5.6))
    x=np.arange(4); w=.34
    bars1=ax.bar(x-w/2,b,w,label="固定 Logistic 基线",color=GRAY)
    bars2=ax.bar(x+w/2,z,w,label="内层选出的模型",color=BLUE)
    ax.bar_label(bars1,fmt="%.1f",padding=2,fontsize=8)
    ax.bar_label(bars2,fmt="%.1f",padding=2,fontsize=8)
    for i,delta in enumerate(d):
        y=max(b[i],z[i])+5.5
        ax.annotate(f"Δ {delta:+.2f} pp",xy=(i,y),ha="center",fontsize=9,color=GREEN,weight="bold")
    base_mean=100*item["baseline_mean_balanced_accuracy"]
    sel_mean=100*item["nested_selected_mean_balanced_accuracy"]
    ax.axhline(base_mean,color=GRAY,ls="--",lw=1.2,label=f"基线折均值 {base_mean:.2f}%")
    ax.axhline(sel_mean,color=ORANGE,ls="--",lw=1.6,label=f"嵌套选择折均值 {sel_mean:.2f}%")
    ax.set_xticks(x,RECORD_LABELS); ax.set_ylim(30,72); ax.set_ylabel("Balanced accuracy（%）")
    ax.set_title("raw-QC cue 左右提示解码：各外层记录均优于固定基线")
    ax.grid(axis="y",alpha=.2); ax.legend(ncol=2,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.19),fontsize=8.5)
    ax.text(.99,.03,"主指标：四个外层记录 BA 等权平均；不是将387条trial直接合并后的pooled BA。",
            transform=ax.transAxes,ha="right",fontsize=8.3,color=RED)
    save_figure(fig,"F08_primary_scores","52.06%主结果及四折配对比较","显示固定基线与嵌套模型选择程序在四个外层记录上的BA，以及+6.05个百分点总体变化。",
                [SUMMARY_JSON, SELECTED_CSV, BASE_CANDIDATE_CSV],{"baseline_fold_mean_percent":base_mean,
                 "nested_selected_fold_mean_percent":sel_mean,"delta_percentage_points":item["delta_percentage_points"],
                 "fold_deltas_percentage_points":d,"n_trials":item["n_trials"],"outer_folds":4})


def figure_selected_configs(selected):
    data=selected.query("sample=='raw_qc_pass' and stage=='cue_locked'").set_index("held_out_record").reindex(RECORDS)
    rows=[]
    for rec,row in data.iterrows():
        if row.model=="rbf_svm":
            param=f"RBF SVM · C={row.parameter:g}, γ={row.gamma}"
        elif row.model=="logistic":
            param=f"Logistic · λ={row.parameter:g}"
        else: param=f"{row.model} · {row.parameter}"
        feat={"full_baseline_13":"13维","drop_redundant_alpha_asym_12":"去冗余12维","erp_only_3":"ERP 3维","bandpower_only_9":"功率9维"}.get(row.feature_set,row.feature_set)
        rows.append([RECORD_LABELS[RECORDS.index(rec)],str(row.model),feat,param,f"{100*row.balanced_accuracy:.2f}%",f"{100*row.balanced_accuracy_delta:+.2f} pp"])
    fig,ax=plt.subplots(figsize=(11,3.5)); ax.axis("off")
    t=ax.table(cellText=rows,colLabels=["外层留出记录","模型族","特征集","外层训练选出的参数","BA","相对基线"],
               loc="center",cellLoc="center",colWidths=[.18,.14,.19,.28,.09,.12])
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1,1.7)
    for (r,c),cell in t.get_celld().items():
        if r==0: cell.set_facecolor(NAVY); cell.set_text_props(color="white",weight="bold")
        else:
            cell.set_facecolor("#F1F6F8" if r%2 else "white")
            if c in (4,5): cell.set_text_props(weight="bold",color=GREEN)
        cell.set_edgecolor("#D5DFE4")
    ax.set_title("每个外层折单独用内层结果选择模型；不存在唯一的全局最优模型",pad=20,color=NAVY,weight="bold")
    save_figure(fig,"F09_selected_models","四个外层折的模型选择结果","列出每个留出记录对应的模型族、特征集、参数和外层评分。",
                [SELECTED_CSV],{"selection_scope":"model is selected independently within each outer-training split"})


def get_predictions(path):
    return pd.read_csv(path,encoding="utf-8-sig")


def figure_confusion(baseline_pred, selected_pred):
    data=[]
    for df in [baseline_pred,selected_pred]:
        d=df.query("sample=='raw_qc_pass' and stage=='cue_locked'")
        data.append(confusion_matrix(d.true_cue_side,d.predicted_cue_side,labels=[-1,1]))
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.4),constrained_layout=True)
    v=max(int(m.max()) for m in data)
    for ax,m,title in zip(axes,data,["固定 Logistic 基线","嵌套选定模型"]):
        im=ax.imshow(m,cmap="Blues",vmin=0,vmax=v)
        ax.set_xticks([0,1],["预测左","预测右"]); ax.set_yticks([0,1],["实际左","实际右"])
        for i in range(2):
            for j in range(2): ax.text(j,i,str(m[i,j]),ha="center",va="center",fontsize=14,
                                                color="white" if m[i,j]>v*.55 else NAVY,weight="bold")
        ax.set_title(title)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label="trial 数")
    fig.suptitle("合并387条外层留出预测的混淆矩阵（计数）",color=NAVY,weight="bold")
    fig.text(.5,-.02,"混淆矩阵合并计数仅作错误分布描述；主报告BA是四折等权平均，见图8。",
             ha="center",fontsize=8.5,color=RED)
    save_figure(fig,"F10_confusion_matrices","外层留出预测的混淆矩阵","比较基线与嵌套所选模型的左右错误计数；不以合并矩阵替代主 BA。",
                [BASE_PRED_CSV, SELECTED_PRED_CSV],{"labels":["left=-1","right=+1"],
                 "baseline_matrix":data[0].tolist(),"selected_matrix":data[1].tolist(),"pooled_metric_not_primary":True})


def figure_sensitivity(summary):
    selections=summary["selections_and_deltas"]
    order=[("raw_qc_pass","cue_locked","Raw-QC · cue"),("q1_quality_matched","cue_locked","Q1匹配 · cue"),
           ("raw_qc_pass","target_locked","Raw-QC · target"),("q1_quality_matched","target_locked","Q1匹配 · target")]
    b=[];s=[];n=[]
    for sample,stage,label in order:
        r=next(x for x in selections if x["sample"]==sample and x["stage"]==stage)
        b.append(100*r["baseline_mean_balanced_accuracy"]);s.append(100*r["nested_selected_mean_balanced_accuracy"]);n.append(r["n_trials"])
    fig,ax=plt.subplots(figsize=(10.5,5.4));x=np.arange(4);w=.34
    a=ax.bar(x-w/2,b,w,color=GRAY,label="固定基线");c=ax.bar(x+w/2,s,w,color=BLUE,label="嵌套选择")
    ax.bar_label(a,fmt="%.2f",padding=2,fontsize=8);ax.bar_label(c,fmt="%.2f",padding=2,fontsize=8)
    for i in range(4):
        delta=s[i]-b[i]; color=GREEN if delta>=0 else RED
        ax.text(i,max(b[i],s[i])+4,f"{delta:+.2f} pp\nn={n[i]}",ha="center",fontsize=8.6,color=color,weight="bold")
    ax.set_xticks(x,[x[2] for x in order]);ax.set_ylim(35,67);ax.set_ylabel("四个外层记录的平均 balanced accuracy（%）")
    ax.set_title("52.06% 的提升没有在全部样本口径与事件阶段复现",color=NAVY)
    ax.grid(axis="y",alpha=.2);ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.15))
    save_figure(fig,"F11_scope_sensitivity","样本口径与事件阶段敏感性","检查提示分类提升是否能迁移到Q1匹配子集与target锁定事件。",
                [SUMMARY_JSON,SELECTED_CSV],{"samples":{"raw_qc_cue":387,"q1_cue":297,"raw_qc_target":396,"q1_target":297},
                 "q1_is_sensitivity_analysis":True})


def figure_quality_subgroups(subgroup_csv: Path):
    df=pd.read_csv(subgroup_csv,encoding="utf-8-sig")
    def val(scope, stage, q1_pass, model):
        row=df[(df.training_scope==scope)&(df.stage==stage)&
               (df.evaluated_q1_quality_pass.astype(str).str.lower()==str(q1_pass).lower())&
               (df.model==model)]
        if len(row)!=1:
            raise ValueError(f"Expected one subgroup row for {(scope,stage,q1_pass,model)}, found {len(row)}")
        return float(row.iloc[0].mean_record_balanced_accuracy),int(row.iloc[0].n_trials)
    pairs=[
        ("cue_locked",True,"Cue · Q1 retained"),
        ("cue_locked",False,"Cue · Q1 excluded\n(raw-QC passed)"),
        ("target_locked",True,"Target · Q1 retained"),
        ("target_locked",False,"Target · Q1 excluded\n(raw-QC passed)"),
    ]
    vals=[(val("raw_qc_pass",stage,q1_pass,"fixed_baseline"),
           val("raw_qc_pass",stage,q1_pass,"nested_selected")) for stage,q1_pass,_ in pairs]
    base=[100*x[0][0] for x in vals];sel=[100*x[1][0] for x in vals]
    counts=[x[1][1] for x in vals]
    fig,ax=plt.subplots(figsize=(9.8,5.2))
    # The summarized values are fixed from the checked experiment report to keep figure labels explicit.
    labels=[x[2] for x in pairs]
    x=np.arange(4);w=.34
    p=ax.bar(x-w/2,base,w,color=GRAY,label="基线");q=ax.bar(x+w/2,sel,w,color=BLUE,label="嵌套选择")
    ax.bar_label(p,fmt="%.1f",padding=2,fontsize=8);ax.bar_label(q,fmt="%.1f",padding=2,fontsize=8)
    for i in range(4):
        delta=sel[i]-base[i];ax.text(i,max(base[i],sel[i])+4,f"Δ {delta:+.1f} pp\nn={counts[i]}",ha="center",fontsize=8,color=GREEN if delta>=0 else RED)
    ax.set_xticks(x,labels);ax.set_ylim(30,70);ax.set_ylabel("Balanced accuracy（%）")
    ax.set_title("按 held-out 质量标签分组的诊断性结果",color=NAVY);ax.grid(axis="y",alpha=.2);ax.legend(frameon=False,ncol=2)
    save_figure(fig,"F12_quality_subgroups","held-out 质量子组诊断","展示 raw-QC 训练时提升集中于 Q1保留子组、且在排除子组方向相反的现象。",
                [subgroup_csv,EXP/"quality_subgroup_summary.json"],
                {"cue_q1_retained":{"n":counts[0],"baseline_BA":base[0]/100,"nested_BA":sel[0]/100},
                 "cue_q1_excluded_raw_qc_pass":{"n":counts[1],"baseline_BA":base[1]/100,"nested_BA":sel[1]/100},
                 "diagnostic_only":True})


def build_markdown(summary):
    doc = r'''---
title: "问题三实测脑电提示方向解码的模型验证报告（52.06%结果版）"
date: "2026年9月25日"
lang: zh-CN
---

> **使用边界。** 本报告复述当前代码中得到 52.06% balanced accuracy 的一条 EEG 验证分支。它预测的是 cue 阶段的 VisCue 左/右提示标签，使用提示后 0–500 ms 的 EEG 特征。它不是诊断准确率、应答正确率，也不是《第三.pdf》中完整的视觉—海马—前额叶潜在状态模型与 DDM 的总成绩。正式论文宜将其放在“实测 EEG 提示方向解码/补充验证”小节，并与 V/H/P 状态建模、应答前 100 ms 分析、行为模型结果分别报告。

# 摘要

针对第三问中“利用实测脑电验证视觉认知模型”的要求，本文整理一项可复现的提示方向解码实验。研究以四份连续记录中的 F3、Fz、F4 原始 EEG 为输入，按 VisCue 事件对齐；经零相位滤波与试次级质量控制后，cue 锁定阶段保留 387 个试次。每个试次构造 13 维特征，包括 250–500 ms 的基线校正 ERP 均值、θ/α/β 三频带在三通道上的对数功率，以及 F4−F3 α 功率对比。为了检验有限记录下的模型配置选择，设置 49 个预先限定的 Logistic、收缩 LDA、线性 SVM 与 RBF SVM 候选，并采用嵌套留一记录交叉验证；标准化参数只由相应训练折估计。

以固定的 13 维 Logistic 回归（L2 系数 1）为基线，raw-QC cue 任务的四个外层记录 balanced accuracy 等权平均由 46.01% 提高至 52.06%，变化为 +6.05 个百分点，四个外层记录的变化均为正。该结果仍接近二分类机会水平，且 49 个候选在不同外层折中选出不同模型。对 Q1 质量匹配子集重新训练后，cue 提升消失（−0.02 个百分点）；target 锁定分析也未显示提升。因此，52.06% 应视为开发阶段、对样本口径敏感的探索性结果，不支持稳定泛化或临床诊断结论。

**关键词：** 实测 EEG；视觉提示方向；balanced accuracy；嵌套交叉验证；模型选择；质量敏感性

# 1 研究范围与第三问的衔接

《第三.pdf》提出的主体路线是：从实测 EEG 提取认知特征，建立视觉前馈状态 (V_i)、海马样记忆匹配状态 (H_i) 与前额叶样控制状态 (P_i)，并在数据允许时以试次级漂移扩散模型验证行为差异；认知信号终点约为应答前 100 ms。52.06% 的结果来自另一项已完成的分类实验：它以提示方向 (y_i\in\{-1,+1\}) 为标签，使用提示出现后的 cue 锁定 EEG 特征预测左/右提示。

因此，本报告可以为完整模型提供一项**观测特征层面的解码证据**，用以回答“这组三通道 EEG 特征是否包含可泛化到未见记录的提示方向信息”。它不直接估计 (V_i,H_i,P_i)，不拟合观测矩阵 (C)，也不拟合 DDM 漂移率或反应时。它的时间窗也不是应答标记前 100 ms 的可变窗口。论文应分别报告两种时间窗与两个目标，不能把本报告的 52.06% 称为完整第三问模型的准确率。

![图1 分析流程与结果边界](figures/F01_pipeline.png)

*图1　当前 52.06% 结果的处理链与解释边界。问题二的 EEG 形成机制可作为生理解释背景，但本分类代码使用实测 EEG，不调用问题二的仿真输出。*

# 2 数据、事件与试次筛选

## 2.1 数据单元与标签

分析对象为四份记录：VisualCogA_Task-1、VisualCogA_Task-2、VisualCogB_Task-1、VisualCogB_Task-2。源信号以 256 Hz 采样，分类只取 F3、Fz、F4 三个脑电通道。VisCue 事件提供提示发生时间和左右提示标签；时间戳用于将事件映射到 EEG 采样点。行为/应答通道仅作为 trial 元数据供其他分析使用，本分类的标签是 VisCue 提示侧，而不是第9通道选择侧、正确/错误或漏答标签。

总共有 400 个候选 cue 试次。raw-QC cue 分析保留 387 个；其中左提示 200 个、右提示 187 个。为检查前序 Q1 清理口径对结论的影响，另按 Q1 试次映射保留 297 个质量匹配 cue 试次。target 锁定分析因窗口边缘与数据有效性不同而保留 396 个 raw-QC 试次；Q1 target 子集同为 297 个。Q1 的 EEG 波形不进入第三问特征计算，它只提供时间映射和质量标签。

## 2.2 质量控制

事件窗口采用 raw 连续 EEG。ERP 分支与功率分支在连续信号上分别滤波后再切 epoch，以减少逐 trial 滤波带来的边缘效应。在滤波后的试次 epoch 中，任何通道出现非有限样本、绝对值达到 999.5，或通道峰峰值不超过 $10^{-8}$ 的 flatline，都会使对应特征分支的该 trial 判为无效。质量检查标记异常，不对样本插值或改写。由于只使用三路前额 EEG 且没有独立眼电参考，本流程不报告 ICA 去伪迹；仍可能有肌电、眼动与残余运动伪迹。

| 样本口径 | Cue 锁定 | Target 锁定 | 用途 |
|---|---:|---:|---|
| raw-QC | 387 | 396 | 主分类比较与事件阶段敏感性 |
| Q1 质量匹配 | 297 | 297 | 样本口径敏感性分析 |

![图2 trial 数量与筛选口径](figures/F02_trial_counts.png)

*图2　四份记录在 cue/target 阶段的有效 trial 数。target 事件时刻按 cue+2.2 s 的任务排程假设对齐，不能视为独立检测到的真实目标 marker。*

# 3 信号预处理与事件窗

原始连续 EEG 以 256 Hz 读取，信号沿通道维组织为 ([F3,Fz,F4]^T)。ERP 分支经四阶 Butterworth 0.5–30 Hz 带通，功率分支经四阶 Butterworth 1–80 Hz 带通；均以 SOS 形式作前向—反向滤波，得到零相位结果。cue 锁定 epoch 为 ([-0.10,0.50]) s，基线为 ([-0.10,0)) s；ERP 候选均值取 ([0.25,0.50)) s。频带功率仅从事件后部分计算，即 cue 的 ([0,0.50)) s。target 锁定窗口为 ([-0.10,0.80]) s，时间锚点暂取 cue+2.2 s，仅用于次要敏感性分析。

任务稿中的“应答前约 100 ms”信号另以应答标记时间 (t_{act}) 为锚，截取从 cue 到 (t_{act}-0.10) s 的可变长度片段；但这条可变长度分支没有进入 52.06% 模型。对于无应答 marker 的试次，也不能仅靠 marker 缺失断定行为漏答，除非实验时限和事件语义已验证。

![图3 事件窗口与信号分支](figures/F03_windows_filters.png)

*图3　cue 分类使用的 cue 锁定窗口、ERP 与功率频段；应答终点窗口用于第三问其他分支，不是 52.06% 分类器的输入。*

# 4 特征构造

## 4.1 13 维主特征

对 trial (i)，基线校正后的通道 ERP 均值定义为

\[
E_{i,c}=\bar{x}^{ERP}_{i,c}(W_E)-\bar{x}^{ERP}_{i,c}(W_B),
\]

其中 (c\in\{F3,Fz,F4\})，(W_B=[-0.10,0)) s 为基线窗，(W_E=[0.25,0.50)) s 为事件后均值窗，共 3 个 ERP 均值特征。本文将其称为**额区 ERP 均值候选特征**，不把它单独等同于典型顶区 P300：目前只有三个额区电极，缺少完整头皮拓扑证据。

对功率分支，每个 trial 的事件后片段先减去片段均值，再用 Welch 方法估计功率谱密度。实现设置 `nperseg=片段长度`、`noverlap=0`、constant detrend 与 density scaling，并沿用默认 Hann 窗；在频带内对谱密度作梯形积分，最终取自然对数：

\[
L_{i,c,b}=\log\left(\int_{f_{min}(b)}^{f_{max}(b)}\widehat S_{i,c}(f)\,df\right).
\]

三个频带乘三个通道得到 9 个特征。由于 cue 功率窗长仅 0.5 s，离散频率间隔约为 2 Hz；θ频带的积分精度因此有限，不能过度解释窄频峰。

最后增加一个前额 α 功率对比：

\[
AI_{\alpha,i}=L_{i,F4,\alpha}-L_{i,F3,\alpha}.
\]

三类特征合计 13 维。由于 (AI_\alpha) 是 F3/F4 α 对数功率的线性组合，候选空间同时包含删去该冗余项的 12 维版本。γ功率、ERP 峰值与潜伏期、ERP 不对称指数、试次级应答窗统计量未进入该次候选模型；通道9也未进入 EEG 特征矩阵。

![图4 13维 EEG 特征设计](figures/F04_feature_schema.png)

*图4　主特征的组成及 α 不对称项的冗余消融。*

![图5 左右提示的描述性特征差异](figures/F05_feature_contrast.png)

*图5　右提示与左提示的单变量标准化均值差。该图以最终 raw-QC cue 样本作事后描述，不用于筛选特征、拟合缩放参数或挑选模型。*

# 5 分类器与参数设计

## 5.1 固定基线

比较基线为固定的 13 维 Logistic 回归，标准化后取 L2 惩罚系数 \(\lambda=1\)，概率阈值 0.5。实现优化的目标可写为

\[
\mathcal L(\beta)=\sum_i\{\log(1+e^{\eta_i})-y_i\eta_i\}+\frac{\lambda}{2}\|\beta_{1:p}\|_2^2,
\quad \eta_i=\beta_0+x_i^T\beta_{1:p},
\]

截距 \(\beta_0\) 不受 L2 惩罚。该固定模型与被冻结的初始流程一致，因此可以做同折配对比较。

## 5.2 49 个预设候选

搜索空间包含 4 个特征集：完整 13 维、删除冗余 α 对比后的 12 维、3 维 ERP-only，以及 9 维 bandpower-only。分类器候选由标准 Logistic、收缩 LDA、robust-scaled Logistic、线性 SVM 与 RBF SVM 组成。全体候选在内层验证的平均 BA 上排序；不在外层测试记录上挑选配置。

| 候选族 | 参数设置 | 设计目的 |
|---|---|---|
| 标准 Logistic | \(\lambda\in\{0.1,10,100\}\)，另含固定基线 \(\lambda=1\) | 调整正则化强度，并比较四种特征集 |
| 收缩 LDA | `solver=lsqr`, `shrinkage=auto` | 小样本下进行协方差收缩 |
| Robust Logistic | robust scaler；\(\lambda\in\{0.1,1,10\}\)；完整或去冗余特征 | 检查中位数/IQR 缩放对异常值敏感性的影响 |
| 线性 SVM | 标准 scaler；\(C\in\{0.01,0.1,1,10\}\)；完整或去冗余特征 | 比较线性间隔与正则化强度 |
| RBF SVM | 标准 scaler；\(C\in\{0.1,1,10\}\)，\(\gamma\in\{\text{scale},0.03,0.1\}\)；完整或去冗余特征 | 检查有限的非线性边界候选 |

`standard` 缩放使用训练折均值和标准差；`robust` 缩放使用训练折中位数与四分位距。若训练折某列尺度非有限或近零，则将该列尺度设为 1。验证数据只使用训练折参数变换。概率型模型采用 0.5 阈值，SVM 决策函数采用 0 阈值；未调阈值。候选同分时按事先列出的候选顺序优先。

![图6 模型族与超参数网格](figures/F06_candidate_grid.png)

*图6　49 个候选配置的模型族数量和超参数取值。阈值、特征频段及搜索范围未通过外层测试得分反向调整。*

# 6 嵌套留一记录验证

普通随机 trial 切分会使同一文件中的记录同时进入训练和测试，难以反映对新记录的迁移。这里将完整文件作为分组单位。外层依次留出四份记录之一，其余三份作为训练数据。对每个外层训练集合，内层再依次留出其中一份记录，对 49 个候选各自评估 3 个内层 BA，并取算术平均。平均内层 BA 最高的候选配置被选中；随后用三份外层训练记录重新估计缩放参数与模型，最后只在未见的外层记录上评分一次。

主指标为记录级 balanced accuracy：

\[
BA_r=\frac{1}{2}\left(\frac{TP_{left,r}}{N_{left,r}}+\frac{TP_{right,r}}{N_{right,r}}\right),\qquad
\overline{BA}=\frac{1}{4}\sum_{r=1}^{4}BA_r.
\]

即每个外层记录先对左右两类召回率取平均，再对四份记录等权平均。这是 52.06% 的计算口径；将 387 个外层预测直接汇总再计算 pooled BA 会得到另一个数值。BA 能降低每个记录内轻微类别数差异对评分的影响，但不能弥补只有四个记录组所带来的估计不确定性。

![图7 嵌套留一记录验证结构](figures/F07_nested_validation.png)

*图7　每份记录在外层只用于一次测试；超参数选择只查看剩余三份记录的内层验证结果。*

# 7 主结果与模型解释

## 7.1 52.06% 的来源

| 指标（四外层折均值） | 固定 Logistic 基线 | 嵌套选择程序 | 变化 |
|---|---:|---:|---:|
| Balanced accuracy（主指标） | 46.01% | **52.06%** | **+6.05 pp** |
| Accuracy | 46.73% | 51.63% | +4.91 pp |
| ROC AUC（逐外层折后等权平均） | 48.34% | 52.52% | +4.18 pp |
| Macro-F1 | 44.90% | 47.25% | +2.35 pp |

“+6.05 pp”表示 52.06%−46.01%=6.05 个百分点，不是相对增长 6.05%。四个外层记录的 BA 分别从 49.07%、41.67%、47.95%、45.35% 变为 54.35%、52.78%、52.22%、48.91%，折内增量均为正（+5.27、+11.11、+4.26、+3.56 pp）。分折一致性是这一轮搜索的积极迹象；但四个文件有限，仍无法提供稳定的总体泛化推断。

![图8 52.06%主结果及四折配对比较](figures/F08_primary_scores.png)

*图8　每个柱为一个外层记录 BA；虚线为四折等权均值。误差线未绘制，因为仅四折且记录并非充分随机抽样的独立大量样本。*

## 7.2 每折所选模型

| 留出记录 | 内层所选配置 | 外层 BA | 相对基线 |
|---|---|---:|---:|
| A-1 | RBF SVM；12维；\(C=10,\gamma=0.03\) | 54.35% | +5.27 pp |
| A-2 | RBF SVM；13维；\(C=10,\gamma=\text{scale}\) | 52.78% | +11.11 pp |
| B-1 | RBF SVM；12维；\(C=1,\gamma=0.1\) | 52.22% | +4.26 pp |
| B-2 | Logistic；ERP 3维；\(\lambda=100\) | 48.91% | +3.56 pp |

其中 A-1/A-2/B-1/B-2 分别表示 VisualCogA_Task-1、VisualCogA_Task-2、VisualCogB_Task-1、VisualCogB_Task-2。各外层训练集分别包含 287、291、289、294 个 trial；对应留出集分别为 100、96、98、93 个 trial。

选择结果以 RBF SVM 为主，但第四折选出强正则化的 ERP-only Logistic。因而这套流程没有得到一个稳定、唯一的分类器；若后续需要部署单一模型，应在新增训练记录上预先锁定模型与参数，然后另留独立测试记录。

![图9 各外层折选出的分类器与参数](figures/F09_selected_models.png)

*图9　配置按外层训练折分别选择。该表不能被解释为从测试集上逐折挑选出的“最佳模型”；选择依据始终是对应的内层 BA。*

## 7.3 错误类型的辅助观察

将所有外层预测按 trial 合并得到的计数矩阵为：基线 \(\begin{bmatrix}113&87\\119&68\end{bmatrix}\)，嵌套选择 \(\begin{bmatrix}135&65\\122&65\end{bmatrix}\)；行依次为实际左/右，列为预测左/右。嵌套选择减少了左提示误判为右的计数，但右提示召回并未相应改善。应答类别的漏判模式仍不平衡；这也是 pooled 分类准确率与 BA 需要同时解释的原因。

![图10 外层留出预测的混淆矩阵](figures/F10_confusion_matrices.png)

*图10　基线与嵌套选择程序的外层留出预测计数。混淆矩阵用于看错误构成；主结论仍依据四折等权的 BA。*

# 8 稳健性与样本口径敏感性

为评估提升是否依赖某一种数据定义或事件阶段，在同一模型搜索协议下比较 raw-QC 与 Q1 质量匹配样本，并比较 cue/target 锁定分析。Q1 子集上的模型重新在 Q1 训练数据中进行内层选择，不能与 raw-QC 模型在 Q1 保留试次上的子组诊断混为一谈。

| 分析口径 | 有效 trial 数 | 基线 BA | 嵌套选择 BA | 变化 |
|---|---:|---:|---:|---:|
| raw-QC cue（主结果） | 387 | 46.01% | 52.06% | +6.05 pp |
| Q1 匹配 cue | 297 | 48.68% | 48.65% | −0.02 pp |
| raw-QC target | 396 | 52.25% | 50.10% | −2.15 pp |
| Q1 匹配 target | 297 | 51.04% | 49.70% | −1.34 pp |

![图11 样本口径与事件阶段敏感性](figures/F11_scope_sensitivity.png)

*图11　只有 raw-QC cue 口径出现正向提升；Q1 匹配 cue 的嵌套选择与基线几乎相同，target 两种口径均未改善。*

进一步将 raw-QC 训练模型在留出记录上的预测按 Q1 质量标签分组，只作诊断，不影响候选选择。cue 的 Q1 保留组（297 trial）BA 从 44.2% 增至 54.2%；Q1 未保留但通过 raw-QC 的 90 trial BA 从 52.0% 降至 47.3%。这说明主结果的改善并未一致覆盖所有 EEG 质量子组。另一方面，重新限定训练样本后 Q1 cue 提升消失，提示训练样本构成和模型选择都可能影响这个结果。

![图12 held-out 质量子组诊断](figures/F12_quality_subgroups.png)

*图12　质量子组只在外层留出预测后分层；它是描述性诊断，样本量和分组差异不足以作因果解释。*

# 9 结论、论文写法与局限

## 9.1 可报告结论

在当前四份 EEG 记录和预设候选空间下，嵌套模型选择程序对 raw-QC cue 左右提示分类的外层 BA 均值为 52.06%，较固定 Logistic 基线 46.01% 增加 6.05 个百分点，且四个外层记录的单折变化均为正。这个结果支持“部分 cue 锁定额区 EEG 特征携带有限的左右提示相关信息”这一谨慎表述。由于 Q1 匹配与 target 锁定比较没有复现同等提升，结论应限定于当前 raw-QC cue 分析口径。

## 9.2 应避免的表述

不得将 52.06% 写成第三问完整 V/H/P 状态模型的识别准确率、DDM 行为预测准确率、任务正确率或精神疾病诊断率；不得称 13 维输入中的 ERP 均值已确认是标准 P300；不得声称该模型对新个体或新实验室具有稳定泛化。尤其注意，Q1 匹配 cue 的独立嵌套结果没有提升，而 raw-QC 训练模型在 Q1 排除子组上表现下降。

## 9.3 限制与下一步

1. **独立记录数仅为4。** trial 数不能替代真正独立的记录/受试者数；当前外层 BA 的估计方差未知。
2. **多轮开发比较共用同一批外层记录。** 嵌套 CV 防止了本轮 49 个候选的直接外层调参泄漏，但不能消除团队跨多轮特征/模型探索造成的总体选择偏差。
3. **样本口径敏感。** raw-QC cue 增益在重新限定到 Q1 子集后消失。
4. **标签语义有限。** 预测的是 VisCue 提示方向，不等于应答方向是否正确；应答 marker 编码、反应时和任务目标真值要单独审计。
5. **EEG 空间信息有限。** 只有 F3/Fz/F4，不能定位海马或 LGN，也无法确认 P300 的经典拓扑。
6. **频带估计受短窗影响。** cue 功率窗口约 0.5 s，低频分辨率有限；短窗 Welch 积分不宜作精细谱峰解释。
7. **目标事件锚点为排程假设。** cue+2.2 s target 结果只作为次要敏感性分析；主要 52.06% cue 结论不受该锚点影响。
8. **当前分支未检验完整认知状态与 DDM。** 后续应在独立流程中报告 (V/H/P) 估计、观测重建或消融、应答前 100 ms 特征、正确/错误/漏答分类以及行为 DDM 的拟合与预测检验。

建议最终论文将本结果写成一段补充验证，并将 52.06%、46.01% 和 +6.05 pp 同表展示；同时注明 `n=387`、4 个留出文件、Q1 cue 敏感性结果为 −0.02 pp。若要保留显著性或总体推广结论，需新增独立记录或受试者后再作推断。

# 10 可复现性与结果追溯

本报告的主结果来自冻结初始输入特征与第一轮嵌套模型搜索，不覆盖 baseline snapshot。主要文件如下：

- 冻结特征与基线：`baselines/20260925_initial/output/trial_features.csv`
- 搜索实现：`experiments/model_search.py`
- 第一轮实验输出目录：`output/experiments/20260925_model_search_v1/`
- 逐外层折结果：上述目录中的 `nested_selected_fold_metrics.csv`
- 基线与嵌套选择逐 trial 预测：上述目录中的 `baseline_oof_predictions.csv`、`nested_selected_oof_predictions.csv`
- 参数、特征集与主要汇总：上述目录中的 `summary.json`
- 本报告图形的逐图数据来源、哈希和参数：同目录 `figures/*.figure.json`

该结果的主要算式是：

\[
\overline{BA}_{nested}=\frac{0.543478+0.527778+0.522157+0.489130}{4}=0.520636.
\]

\[
\overline{BA}_{baseline}=\frac{0.490741+0.416667+0.479515+0.453515}{4}=0.460109.
\]

两者之差为 0.060526，即 +6.0526 个百分点。图和表中的百分数均由逐折文件重新读取，渲染图另保存 PNG、SVG 和 JSON 数据说明。

# 参考资料

1. 第二十三届中国研究生数学建模竞赛 C 题：《服务于脑机接口与精神性疾病诊断的脑电图计算模型》，用户提供的题目文件。
2. 用户提供的《第三.pdf》（2026年9月24日版本），用于界定完整第三问的 V/H/P 状态建模、应答前窗口及 DDM 扩展分析范围。
3. 项目计算记录：冻结基线 `20260925_initial` 与嵌套搜索 `20260925_model_search_v1`。具体文件路径见第10节，可在相应 CSV/JSON 与源代码中追溯。

---

**稿件状态：** 论文写作参考稿；仍需与第三问完整 V/H/P 模型及 DDM 的实际输出合并，并在新增独立记录上确认泛化后，方可形成最终结论。
'''
    doc = doc.replace(r"\[", "$$").replace(r"\]", "$$")
    doc = doc.replace(r"\(", "$").replace(r"\)", "$")
    path=REPORT_DIR/"Q3_52_06_paper_reference_report.md"
    path.write_text(doc,encoding="utf-8")
    return path


def build_docx(markdown: Path):
    output=REPORT_DIR/"Q3_52_06_paper_reference_report.docx"
    # Figures already have explicit Chinese captions immediately after each image.
    # Disable implicit_figures so Pandoc does not add a second caption from image alt text.
    cmd=["pandoc",str(markdown),"--from=markdown+tex_math_dollars+pipe_tables-implicit_figures",
         "--to=docx","--standalone",
         "--resource-path="+str(REPORT_DIR),"--output",str(output)]
    subprocess.run(cmd,check=True,cwd=REPORT_DIR)
    return output


def main():
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    FIG_DIR.mkdir(parents=True,exist_ok=True)
    features=pd.read_csv(FEATURE_CSV,encoding="utf-8-sig")
    summary=load_json(SUMMARY_JSON)
    selected=pd.read_csv(SELECTED_CSV,encoding="utf-8-sig")
    base_pred=get_predictions(BASE_PRED_CSV)
    selected_pred=get_predictions(SELECTED_PRED_CSV)
    candidates=load_candidate_grid()
    assert len(candidates)==49, f"Expected 49 candidates, found {len(candidates)}"
    figure_pipeline()
    figure_counts(features)
    figure_windows()
    figure_feature_schema()
    figure_feature_contrast(features)
    figure_candidate_grid(candidates)
    figure_nested_cv()
    figure_primary_result(summary,selected)
    figure_selected_configs(selected)
    figure_confusion(base_pred,selected_pred)
    figure_sensitivity(summary)
    figure_quality_subgroups(SUBGROUP_CSV)
    markdown=build_markdown(summary)
    docx=build_docx(markdown)
    inputs=[FEATURE_CSV,SUMMARY_JSON,SELECTED_CSV,BASE_CANDIDATE_CSV,BASE_PRED_CSV,SELECTED_PRED_CSV,SUBGROUP_CSV,MODEL_SEARCH]
    manifest={
        "report_title":"问题三实测脑电提示方向解码的模型验证报告（52.06%结果版）",
        "generated_files":[str(markdown.resolve()),str(docx.resolve())],
        "input_sha256":{str(path.resolve()):source_hash(path) for path in inputs},
        "figure_count":12,
        "main_result":{"balanced_accuracy":0.5206359160163508,"baseline":0.4601094303048512,"delta_percentage_points":6.052648571149966,"n_cue_raw_qc":387,"outer_groups":4},
        "scope_warning":"Cue-side decoding only; not the full V/H/P + DDM model score.",
    }
    (REPORT_DIR/"report_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"markdown":str(markdown),"docx":str(docx),"figures":12,"docx_bytes":docx.stat().st_size},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
