import numpy as np
import pandas as pd

from pathlib import Path

from scipy.io import loadmat, savemat
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt


# ============================================================
# 路径
# ============================================================

input_dir = Path(
    r"D:\8\Desktop\CMathc\data"
    r"\第二十三届中国研究生数学建模竞赛+-+中文题目"
    r"\中文题目\C题"
)

DATASET_NAMES = (
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
)

output_dir = Path(
    r"D:\8\Desktop\CMathc\src\C\q1\output"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 参数
# ============================================================

# EEG通道在data中的位置
EEG_CHANNELS = {
    "Fz": 0,
    "F3": 1,
    "F4": 2,
}

# 削顶判断
CLIP_THRESHOLD = 999.0

# <= 50 ms 的削顶允许局部重构
MAX_SHORT_CLIP_S = 0.05

# 瞬时跳变最多修复30 ms
MAX_JUMP_REPAIR_S = 0.03

# Hampel / MAD 窗口
JUMP_WINDOW_S = 0.125

# 瞬时跳变阈值
JUMP_N_SIGMA = 6.0

# 慢漂移高通截止频率
HIGH_PASS_HZ = 0.1

# 高通滤波器阶数
HIGH_PASS_ORDER = 4


# ============================================================
# 查找连续True区间
# ============================================================

def find_runs(mask):
    """
    返回True连续区间：
    [(start, end), ...]
    start/end均包含。
    """

    mask = np.asarray(
        mask,
        dtype=bool
    )

    padded = np.r_[
        False,
        mask,
        False
    ]

    diff = np.diff(
        padded.astype(int)
    )

    starts = np.where(
        diff == 1
    )[0]

    ends = (
        np.where(
            diff == -1
        )[0]
        - 1
    )

    return list(
        zip(starts, ends)
    )


# ============================================================
# PCHIP局部修复
# ============================================================

def repair_short_runs(
    signal,
    bad_mask,
    max_run_samples,
    context_samples=12,
):
    """
    仅修复长度 <= max_run_samples 的异常段。

    长异常段不处理。
    """

    y = np.asarray(
        signal,
        dtype=float
    ).copy()

    repaired_mask = np.zeros(
        len(y),
        dtype=bool
    )

    runs = find_runs(
        bad_mask
    )

    for start, end in runs:

        run_length = (
            end - start + 1
        )

        if run_length > max_run_samples:
            continue

        left = max(
            0,
            start - context_samples
        )

        right = min(
            len(y),
            end + context_samples + 1
        )

        local_idx = np.arange(
            left,
            right
        )

        local_bad = bad_mask[
            left:right
        ]

        good_idx = local_idx[
            ~local_bad
        ]

        if len(good_idx) < 2:
            continue

        repair_idx = np.arange(
            start,
            end + 1
        )

        # 至少4个正常邻点时用PCHIP
        if len(good_idx) >= 4:

            interpolator = PchipInterpolator(
                good_idx,
                y[good_idx],
                extrapolate=True,
            )

            y[repair_idx] = (
                interpolator(
                    repair_idx
                )
            )

        else:

            y[repair_idx] = np.interp(
                repair_idx,
                good_idx,
                y[good_idx],
            )

        repaired_mask[
            start:end + 1
        ] = True

    return (
        y,
        repaired_mask
    )


# ============================================================
# 长异常段临时桥接
#
# 仅为保证滤波器能运行。
# 最终这些位置会重新设为NaN。
# ============================================================

def temporary_bridge(
    signal,
    invalid_mask,
):

    y = np.asarray(
        signal,
        dtype=float
    ).copy()

    if not invalid_mask.any():
        return y

    idx = np.arange(
        len(y)
    )

    good = ~invalid_mask

    if good.sum() < 2:
        return y

    y[invalid_mask] = np.interp(
        idx[invalid_mask],
        idx[good],
        y[good],
    )

    return y


# ============================================================
# 瞬时跳变检测
#
# 在一阶差分上使用局部MAD
# ============================================================

def detect_jumps(
    signal,
    fs,
    window_s=0.125,
    n_sigma=6.0,
):

    y = np.asarray(
        signal,
        dtype=float
    )

    diff = np.diff(
        y,
        prepend=y[0]
    )

    window = int(
        round(
            window_s * fs
        )
    )

    # 必须为奇数
    if window % 2 == 0:
        window += 1

    window = max(
        window,
        5
    )

    s = pd.Series(
        diff
    )

    local_median = (
        s
        .rolling(
            window,
            center=True,
            min_periods=1,
        )
        .median()
    )

    abs_dev = (
        s - local_median
    ).abs()

    local_mad = (
        abs_dev
        .rolling(
            window,
            center=True,
            min_periods=1,
        )
        .median()
    )

    # MAD -> sigma
    local_sigma = (
        1.4826
        *
        local_mad.to_numpy()
    )

    # 防止某些局部窗口MAD为0
    global_mad = np.median(
        np.abs(
            diff
            -
            np.median(diff)
        )
    )

    global_sigma = (
        1.4826
        *
        global_mad
    )

    min_sigma = max(
        global_sigma * 0.25,
        1e-9
    )

    local_sigma = np.maximum(
        local_sigma,
        min_sigma
    )

    jump_mask = (
        np.abs(
            diff
            -
            local_median.to_numpy()
        )
        >
        n_sigma
        *
        local_sigma
    )

    # 跳变通常会影响相邻采样点
    expanded = jump_mask.copy()

    expanded[1:] |= (
        jump_mask[:-1]
    )

    expanded[:-1] |= (
        jump_mask[1:]
    )

    return expanded


# ============================================================
# 高通去除慢漂移
# ============================================================

def remove_slow_drift(
    signal,
    fs,
    cutoff=0.1,
    order=4,
):

    sos = butter(
        order,
        cutoff,
        btype="highpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(
        sos,
        signal
    )


# ============================================================
# 单通道清洗
# ============================================================

def clean_channel(
    signal,
    fs,
):

    raw = np.asarray(
        signal,
        dtype=float
    )

    # --------------------------------------------------------
    # 1. 削顶检测
    # --------------------------------------------------------

    clip_mask = (
        np.abs(raw)
        >=
        CLIP_THRESHOLD
    )

    max_short_clip_samples = int(
        round(
            MAX_SHORT_CLIP_S * fs
        )
    )

    # 短削顶修复
    x, clip_repaired_mask = (
        repair_short_runs(
            raw,
            clip_mask,
            max_short_clip_samples,
        )
    )


    # --------------------------------------------------------
    # 找长削顶
    # --------------------------------------------------------

    long_clip_mask = np.zeros(
        len(raw),
        dtype=bool
    )

    for start, end in find_runs(
        clip_mask
    ):

        duration_samples = (
            end - start + 1
        )

        if (
            duration_samples
            >
            max_short_clip_samples
        ):

            long_clip_mask[
                start:end + 1
            ] = True


    # --------------------------------------------------------
    # 为后续滤波临时连接长削顶区域
    # --------------------------------------------------------

    x = temporary_bridge(
        x,
        long_clip_mask
    )


    # --------------------------------------------------------
    # 2. 瞬时跳变检测
    # --------------------------------------------------------

    jump_mask = detect_jumps(
        x,
        fs,
        window_s=JUMP_WINDOW_S,
        n_sigma=JUMP_N_SIGMA,
    )

    # 削顶区域不重复算成jump
    jump_mask[
        clip_mask
    ] = False


    max_jump_samples = int(
        round(
            MAX_JUMP_REPAIR_S * fs
        )
    )


    # --------------------------------------------------------
    # 修复短时跳变
    # --------------------------------------------------------

    x, jump_repaired_mask = (
        repair_short_runs(
            x,
            jump_mask,
            max_jump_samples,
        )
    )


    # --------------------------------------------------------
    # 3. 高通去慢漂移
    # --------------------------------------------------------

    x = remove_slow_drift(
        x,
        fs,
        cutoff=HIGH_PASS_HZ,
        order=HIGH_PASS_ORDER,
    )


    # --------------------------------------------------------
    # 4. 长削顶不能可靠恢复
    #
    # 最终重新设置为NaN
    # 后续ERP分析应排除这些位置
    # --------------------------------------------------------

    x[
        long_clip_mask
    ] = np.nan


    return {
        "clean":
            x,

        "clip_mask":
            clip_mask,

        "clip_repaired_mask":
            clip_repaired_mask,

        "long_clip_mask":
            long_clip_mask,

        "jump_mask":
            jump_mask,

        "jump_repaired_mask":
            jump_repaired_mask,
    }


# ============================================================
# 读取MAT
# ============================================================

def process_dataset(dataset_name):
    input_path = input_dir / f"{dataset_name}.mat"
    clean_mat_path = output_dir / f"{dataset_name}_clean.mat"
    qc_csv_path = output_dir / f"{dataset_name}_clean_QC.csv"

    mat = loadmat(input_path)
    fs = int(np.asarray(mat["SampleRate"]).squeeze())
    data = np.asarray(mat["data"], dtype=float)

    print(f"\n{'=' * 60}")
    print(f"处理数据集: {dataset_name}")
    print(f"采样率: {fs} Hz")
    print(f"数据维度: {data.shape}")

    data_clean = data.copy()
    mask_shape = (len(EEG_CHANNELS), data.shape[1])
    clip_mask_all = np.zeros(mask_shape, dtype=np.uint8)
    clip_repaired_all = np.zeros(mask_shape, dtype=np.uint8)
    long_clip_all = np.zeros(mask_shape, dtype=np.uint8)
    jump_mask_all = np.zeros(mask_shape, dtype=np.uint8)
    jump_repaired_all = np.zeros(mask_shape, dtype=np.uint8)
    qc_records = []

    for mask_row, (channel, channel_idx) in enumerate(EEG_CHANNELS.items()):
        print(f"处理 {channel} ...")
        result = clean_channel(data[channel_idx], fs)
        data_clean[channel_idx] = result["clean"]

        clip_mask_all[mask_row] = result["clip_mask"].astype(np.uint8)
        clip_repaired_all[mask_row] = result["clip_repaired_mask"].astype(np.uint8)
        long_clip_all[mask_row] = result["long_clip_mask"].astype(np.uint8)
        jump_mask_all[mask_row] = result["jump_mask"].astype(np.uint8)
        jump_repaired_all[mask_row] = result["jump_repaired_mask"].astype(np.uint8)

        qc_records.append(
            {
                "channel": channel,
                "clip_samples": int(result["clip_mask"].sum()),
                "clip_seconds": result["clip_mask"].sum() / fs,
                "clip_runs": len(find_runs(result["clip_mask"])),
                "short_clip_repaired_samples": int(
                    result["clip_repaired_mask"].sum()
                ),
                "long_clip_runs": len(find_runs(result["long_clip_mask"])),
                "long_clip_invalid_seconds": result["long_clip_mask"].sum() / fs,
                "jump_runs": len(find_runs(result["jump_mask"])),
                "jump_detected_samples": int(result["jump_mask"].sum()),
                "jump_repaired_samples": int(
                    result["jump_repaired_mask"].sum()
                ),
                "highpass_hz": HIGH_PASS_HZ,
            }
        )

    qc_df = pd.DataFrame(qc_records)
    print("\n清洗结果:")
    print(qc_df.to_string(index=False))
    qc_df.to_csv(qc_csv_path, index=False, encoding="utf-8-sig")

    output_mat = {
        "SampleRate": mat["SampleRate"],
        "DataLabel": mat["DataLabel"],
        "data_clean": data_clean,
        "clean_channel_labels": np.array(
            list(EEG_CHANNELS),
            dtype=object,
        ),
        "clip_mask": clip_mask_all,
        "clip_repaired_mask": clip_repaired_all,
        "long_clip_invalid_mask": long_clip_all,
        "jump_mask": jump_mask_all,
        "jump_repaired_mask": jump_repaired_all,
        "processing_highpass_hz": np.array([[HIGH_PASS_HZ]]),
        "processing_clip_threshold": np.array([[CLIP_THRESHOLD]]),
        "processing_max_short_clip_s": np.array([[MAX_SHORT_CLIP_S]]),
        "processing_max_jump_repair_s": np.array(
            [[MAX_JUMP_REPAIR_S]]
        ),
    }

    savemat(clean_mat_path, output_mat, do_compression=True)

    print("处理完成:")
    print("原始文件:", input_path)
    print("清洗文件:", clean_mat_path)
    print("QC统计:", qc_csv_path)

    return clean_mat_path, qc_csv_path


def main():
    for dataset_name in DATASET_NAMES:
        process_dataset(dataset_name)


if __name__ == "__main__":
    main()
