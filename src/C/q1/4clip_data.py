import numpy as np
from scipy.io import loadmat, savemat
from pathlib import Path


# ============================================================
# 路径
# ============================================================

data_dir = Path(
    r"D:\8\Desktop\CMathc"
    r"\第二十三届中国研究生数学建模竞赛+-+中文题目"
    r"\中文题目\C题"
)

output_dir = Path(
    r"D:\8\Desktop\CMathc\src\C\q1\output"
)
output_dir.mkdir(exist_ok=True)


# ============================================================
# 数据文件
# ============================================================

mat_files = [
    "VisualCogA_Task-1.mat",
    "VisualCogA_Task-2.mat",
    "VisualCogB_Task-1.mat",
    "VisualCogB_Task-2.mat",
]


# ============================================================
# Trial参数
# ============================================================

PRE_TIME = 1.0
POST_TIME = 3.0


CHANNEL_LABELS = [
    "Fz",
    "F3",
    "F4",
    "FzDecon",
    "F3Decon",
    "F4Decon",
    "ECG",
    "VisCue",
    "Action",
    "TimeStamp",
]


# ============================================================
# Trial切片
# ============================================================

for file_name in mat_files:

    print("\n============================")
    print("处理:", file_name)


    clean_path = output_dir / file_name.replace(
        ".mat",
        "_clean.mat"
    )


    trial_path = output_dir / file_name.replace(
        ".mat",
        "_trials.mat"
    )


    mat = loadmat(clean_path)


    data = np.asarray(
        mat["data_clean"],
        dtype=float
    )


    fs = int(
        np.asarray(
            mat["SampleRate"]
        ).squeeze()
    )


    expected_length = int(
        (PRE_TIME + POST_TIME) * fs
    )


    # --------------------------------------------------------
    # VisCue检测
    # --------------------------------------------------------

    viscue = data[7]

    cue_onsets = np.where(
        (viscue != 0)
        &
        (np.r_[0, viscue[:-1]] == 0)
    )[0]


    print(
        "Trial数量:",
        len(cue_onsets)
    )


    # 保存矩阵
    trial_data_list = []
    relative_time_list = []
    timestamp_list = []

    cue_type_list = []
    action_exist_list = []

    nan_ratio_list = []
    length_warning_list = []


    # --------------------------------------------------------
    # Trial循环
    # --------------------------------------------------------

    for trial_id, idx in enumerate(
        cue_onsets,
        start=1
    ):

        t0 = data[9, idx]


        start_time = t0 - PRE_TIME
        end_time = t0 + POST_TIME


        start_idx = np.searchsorted(
            data[9],
            start_time
        )

        end_idx = np.searchsorted(
            data[9],
            end_time
        )


        trial_data = data[
            :,
            start_idx:end_idx
        ].copy()


        # -----------------------------
        # 统一长度
        # -----------------------------

        if trial_data.shape[1] < expected_length:

            pad = np.full(
                (
                    trial_data.shape[0],
                    expected_length - trial_data.shape[1]
                ),
                np.nan
            )

            trial_data = np.hstack(
                [
                    trial_data,
                    pad
                ]
            )

        else:

            trial_data = trial_data[
                :,
                :expected_length
            ]


        relative_time = np.arange(
            expected_length
        ) / fs - PRE_TIME


        timestamp = np.full(
            expected_length,
            np.nan
        )

        valid_len = min(
            len(data[9, start_idx:end_idx]),
            expected_length
        )

        timestamp[:valid_len] = (
            data[9,start_idx:start_idx+valid_len]
        )


        eeg_data = trial_data[:3]


        nan_ratio = (
            np.isnan(eeg_data).sum()
            /
            eeg_data.size
        )


        trial_data_list.append(
            trial_data
        )

        relative_time_list.append(
            relative_time
        )

        timestamp_list.append(
            timestamp
        )

        cue_type_list.append(
            data[7, idx]
        )

        action_exist_list.append(
            np.any(
                trial_data[8] != 0
            )
        )

        nan_ratio_list.append(
            nan_ratio
        )

        length_warning_list.append(
            nan_ratio > 0
        )


    # --------------------------------------------------------
    # 保存矩阵结构
    # --------------------------------------------------------

    savemat(
        trial_path,
        {

            "trial_data":
                np.asarray(
                    trial_data_list,
                    dtype=float
                ),

            "relative_time":
                np.asarray(
                    relative_time_list
                ),

            "timestamp":
                np.asarray(
                    timestamp_list
                ),

            "cue_type":
                np.asarray(
                    cue_type_list
                ),

            "action_exist":
                np.asarray(
                    action_exist_list
                ),

            "nan_ratio":
                np.asarray(
                    nan_ratio_list
                ),

            "length_warning":
                np.asarray(
                    length_warning_list
                ),

            "fs":
                fs,

            "window":
                np.array(
                    [
                        -PRE_TIME,
                        POST_TIME
                    ]
                ),

            "channel_labels":
                CHANNEL_LABELS,

        },

        do_compression=True
    )


    print(
        "保存:",
        trial_path.name
    )

print("\n全部Trial切片完成")
