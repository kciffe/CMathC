import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.io import loadmat, savemat
from scipy.interpolate import PchipInterpolator


# ============================================================
# 路径
# ============================================================

data_dir = Path(
    r"D:\8\Desktop\CMathc\data\第二十三届中国研究生数学建模竞赛+-+中文题目\中文题目\C题"
)

output_dir = Path(
    r"D:\8\Desktop\CMathc\src\C\q1\output"
)

output_dir.mkdir(exist_ok=True)

DATASETS = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]


# ============================================================
# 参数
# ============================================================

CHANNELS = ["Fz", "F3", "F4"]

CLIP_THRESHOLD = 999
MAX_CLIP_LENGTH = 10

JUMP_SIGMA = 6

PRE_TIME = 1.0
POST_TIME = 3.0


# ============================================================
# 基础工具
# ============================================================

def pchip_fill(x, mask):
    y = x.copy()
    if mask.sum() == 0:
        return y

    idx = np.arange(len(y))
    valid = ~mask

    if valid.sum() < 2:
        return y

    f = PchipInterpolator(idx[valid], y[valid])
    y[mask] = f(idx[mask])
    return y


def find_segments(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return groups


# ============================================================
# 异常检测与修复
# ============================================================

def detect_clip(x):
    return np.abs(x) >= CLIP_THRESHOLD


def repair_clip(x):
    raw_mask = detect_clip(x)
    repaired_mask = np.zeros_like(raw_mask, dtype=bool)

    groups = find_segments(raw_mask)
    for g in groups:
        if len(g) <= MAX_CLIP_LENGTH:
            repaired_mask[g] = True

    y = pchip_fill(x, repaired_mask)
    return y, raw_mask, repaired_mask


def detect_jump(x):
    diff = np.diff(x)
    mad = np.median(np.abs(diff - np.median(diff)))
    if mad == 0:
        return np.zeros_like(x, dtype=bool)

    mask = np.r_[False, np.abs(diff) > JUMP_SIGMA * mad]
    return mask


def repair_jump(x):
    jump_mask = detect_jump(x)
    y = pchip_fill(x, jump_mask)
    return y, jump_mask


# ============================================================
# 选代表片段
# ============================================================

def pick_top_clip_cases(mask, top_k=2):
    groups = find_segments(mask)
    if len(groups) == 0:
        return []

    groups = sorted(groups, key=lambda g: len(g), reverse=True)
    centers = [int(g[len(g)//2]) for g in groups[:top_k]]
    return centers


def pick_top_jump_cases(x, jump_mask, top_k=2):
    idx = np.where(jump_mask)[0]
    if len(idx) == 0:
        return []

    diff = np.abs(np.diff(x))
    jump_strength = [(i, diff[i-1] if i > 0 else 0) for i in idx]
    jump_strength = sorted(jump_strength, key=lambda t: t[1], reverse=True)

    centers = []
    used = []
    for i, _ in jump_strength:
        if all(abs(i - u) > 256 for u in used):
            centers.append(i)
            used.append(i)
        if len(centers) >= top_k:
            break
    return centers


# ============================================================
# 绘图
# ============================================================

def plot_before_after_cases(
    before_3ch,
    after_3ch,
    centers,
    fs,
    save_dir,
    prefix,
    window_sec=2,
    xlabel="Relative time (s)"
):
    save_dir.mkdir(parents=True, exist_ok=True)

    for k, center in enumerate(centers, start=1):
        start = max(0, center - int(window_sec * fs))
        end = min(before_3ch.shape[1], center + int(window_sec * fs))

        t = (np.arange(start, end) - center) / fs

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        for i, ch in enumerate(CHANNELS):
            axes[i].plot(t, before_3ch[i, start:end], label="Before", linewidth=1)
            axes[i].plot(t, after_3ch[i, start:end], label="After", linewidth=1)
            axes[i].set_ylabel(ch)
            axes[i].grid(alpha=0.3)

        axes[0].legend()
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(f"{prefix} case {k}")

        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix.lower()}_case_{k}_before_after.png", dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================
# Trial切片
# ============================================================

def extract_trials(data_clean, fs):
    cue = data_clean[7]

    events = np.where(
        (cue != 0) &
        (np.r_[0, cue[:-1]] == 0)
    )[0]

    trial_len = int((PRE_TIME + POST_TIME) * fs)

    trial_data = []
    relative_time = []
    cue_type = []

    for idx in events:
        t0 = data_clean[9, idx]

        start_idx = np.searchsorted(data_clean[9], t0 - PRE_TIME)
        end_idx = np.searchsorted(data_clean[9], t0 + POST_TIME)

        seg = data_clean[:, start_idx:end_idx].copy()

        if seg.shape[1] < trial_len:
            pad = np.full((data_clean.shape[0], trial_len - seg.shape[1]), np.nan)
            seg = np.hstack([seg, pad])
        else:
            seg = seg[:, :trial_len]

        trial_data.append(seg)
        relative_time.append(np.arange(trial_len) / fs - PRE_TIME)
        cue_type.append(data_clean[7, idx])

    return (
        np.asarray(trial_data, dtype=float),
        np.asarray(relative_time, dtype=float),
        np.asarray(cue_type, dtype=float)
    )


# ============================================================
# 主流程
# ============================================================

for dataset_name in DATASETS:
    print(f"\n处理: {dataset_name}")

    raw_path = data_dir / f"{dataset_name}.mat"
    mat = loadmat(raw_path)

    data = np.asarray(mat["data"], dtype=float)
    fs = int(np.asarray(mat["SampleRate"]).squeeze())

    eeg_raw = data[:3].copy()

    # ---------- Clip ----------
    clip_after = eeg_raw.copy()
    clip_masks = []
    clip_repaired_masks = []

    for i in range(3):
        y, clip_mask, repaired_mask = repair_clip(clip_after[i])
        clip_after[i] = y
        clip_masks.append(clip_mask)
        clip_repaired_masks.append(repaired_mask)

    clip_masks = np.asarray(clip_masks)
    clip_repaired_masks = np.asarray(clip_repaired_masks)

    # ---------- Jump ----------
    jump_before = clip_after.copy()
    jump_after = jump_before.copy()
    jump_masks = []

    for i in range(3):
        y, jump_mask = repair_jump(jump_after[i])
        jump_after[i] = y
        jump_masks.append(jump_mask)

    jump_masks = np.asarray(jump_masks)

    # ---------- 保存clean ----------
    data_clean = data.copy()
    data_clean[:3] = jump_after

    savemat(
        output_dir / f"{dataset_name}_clean.mat",
        {
            "data_clean": data_clean,
            "SampleRate": fs,
            "DataLabel": mat["DataLabel"]
        }
    )

    # ---------- QC ----------
    qc_rows = []
    for i, ch in enumerate(CHANNELS):
        qc_rows.append({
            "channel": ch,
            "clip_points": int(clip_masks[i].sum()),
            "clip_repaired_points": int(clip_repaired_masks[i].sum()),
            "jump_points": int(jump_masks[i].sum())
        })

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(
        output_dir / f"{dataset_name}_QC.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ---------- 异常图 ----------
    fig_root = output_dir / "figures" / dataset_name

    # Clip 图：优先从F3/F4中找
    clip_centers = []
    for ch_i in [1, 2, 0]:
        centers = pick_top_clip_cases(clip_masks[ch_i], top_k=2)
        clip_centers.extend(centers)
        if len(clip_centers) >= 2:
            break
    clip_centers = clip_centers[:2]

    if len(clip_centers) > 0:
        plot_before_after_cases(
            eeg_raw,
            clip_after,
            clip_centers,
            fs,
            fig_root,
            prefix="clip",
            window_sec=2
        )

    # Jump 图
    jump_centers = []
    for ch_i in [1, 2, 0]:
        centers = pick_top_jump_cases(jump_before[ch_i], jump_masks[ch_i], top_k=2)
        jump_centers.extend(centers)
        if len(jump_centers) >= 2:
            break
    jump_centers = jump_centers[:2]

    if len(jump_centers) > 0:
        plot_before_after_cases(
            jump_before,
            jump_after,
            jump_centers,
            fs,
            fig_root,
            prefix="jump",
            window_sec=1
        )

    # ---------- Trial切片 ----------
    trial_data, relative_time, cue_type = extract_trials(data_clean, fs)

    savemat(
        output_dir / f"{dataset_name}_trials.mat",
        {
            "trial_data": trial_data,
            "relative_time": relative_time,
            "cue_type": cue_type,
            "fs": fs
        }
    )

    print(f"完成: {dataset_name}")
    print(f"  Trial shape: {trial_data.shape}")
    print(f"  Figure dir: {fig_root}")


print("\n全部处理完成")
