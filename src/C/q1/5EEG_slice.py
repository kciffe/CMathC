from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

DATA_DIR = Path(
    r"D:\8\Desktop\CMathc\data\第二十三届中国研究生数学建模竞赛+-+中文题目\中文题目\C题"
)
OUTPUT_DIR = Path(r"D:\8\Desktop\CMathc\src\C\q1\output")

FILES = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]

PRE_TIME = 1.0
POST_TIME = 3.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name in FILES:
    mat = loadmat(DATA_DIR / f"{name}.mat")
    data = np.asarray(mat["data"], dtype=float)
    fs = int(np.asarray(mat["SampleRate"]).squeeze())

    cue = data[7]  # VisCue
    events = np.where((cue != 0) & (np.r_[0, cue[:-1]] == 0))[0]

    pre = int(PRE_TIME * fs)
    post = int(POST_TIME * fs)
    events = events[(events >= pre) & (events + post <= data.shape[1])]

    trials = np.stack([data[:, i - pre:i + post] for i in events])
    time = np.arange(-pre, post) / fs
    cue_type = cue[events]

    savemat(
        OUTPUT_DIR / f"{name}_sliced.mat",
        {
            "trials": trials,
            "time": time,
            "cue_type": cue_type,
            "event_index": events,
            "SampleRate": fs,
        },
    )

    print(f"{name}: {len(events)} trials, shape={trials.shape}")
