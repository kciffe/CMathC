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


output_dir = data_dir / "output"

output_dir.mkdir(
    exist_ok=True
)


# 四个数据文件

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



# ============================================================
# 通道
# ============================================================

labels = [
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
# 批量处理
# ============================================================

for file_name in mat_files:


    print("\n====================")
    print(
        "正在处理:",
        file_name
    )


    clean_path = (
        output_dir
        /
        file_name.replace(
            ".mat",
            "_clean.mat"
        )
    )


    trial_path = (
        output_dir
        /
        file_name.replace(
            ".mat",
            "_trials.mat"
        )
    )


    # ----------------------------------
    # 读取clean数据
    # ----------------------------------

    mat = loadmat(
        clean_path
    )


    data = np.asarray(
        mat["data_clean"],
        dtype=float
    )


    fs = int(
        np.asarray(
            mat["SampleRate"]
        ).squeeze()
    )


    print(
        "采样率:",
        fs
    )

    print(
        "数据:",
        data.shape
    )



    # ----------------------------------
    # VisCue检测
    # ----------------------------------

    viscue = data[7]


    cue_onsets = np.where(
        (viscue != 0)
        &
        (np.r_[0,viscue[:-1]]==0)
    )[0]


    print(
        "检测Trial:",
        len(cue_onsets)
    )



    # ----------------------------------
    # Trial切分
    # ----------------------------------

    trials = []


    for trial_id, idx in enumerate(
        cue_onsets,
        start=1
    ):


        t0 = data[
            9,
            idx
        ]


        start_time = (
            t0
            -
            PRE_TIME
        )


        end_time = (
            t0
            +
            POST_TIME
        )


        start_idx = np.searchsorted(
            data[9],
            start_time
        )


        end_idx = np.searchsorted(
            data[9],
            end_time
        )


        start_idx = max(
            start_idx,
            0
        )

        end_idx = min(
            end_idx,
            data.shape[1]
        )



        trial_data = (
            data[:,start_idx:end_idx]
            .copy()
        )


        relative_time = (
            trial_data[9]
            -
            t0
        )


        trial = {


            "trial_id":
                trial_id,


            "data":
                trial_data,


            "relative_time":
                relative_time,


            "cue_type":
                int(
                    data[7,idx]
                ),


            "channel_labels":
                labels,


            "action_exist":
                bool(
                    np.any(
                        trial_data[8]!=0
                    )
                )

        }


        trials.append(
            trial
        )



    # ----------------------------------
    # 保存
    # ----------------------------------

    savemat(
        trial_path,
        {

            "trials":
                np.array(
                    trials,
                    dtype=object
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


            "source_file":
                file_name

        },

        do_compression=True
    )


    print(
        "保存:",
        trial_path.name
    )


print("\n全部Trial切片完成")