from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


DATASETS = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]

CLIP_THRESHOLD = 999.0
PRE_TIME = 1.0
POST_TIME = 3.0

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def slice_trials_with_drop(data, fs):
    data = np.asarray(data, dtype=float)
    pre = int(PRE_TIME * fs)
    post = int(POST_TIME * fs)

    cue = data[7]
    events = np.flatnonzero((cue != 0) & (np.r_[0, cue[:-1]] == 0))
    events = events[(events >= pre) & (events + post <= data.shape[1])]

    trial_data = np.stack([data[:, i - pre:i + post] for i in events])
    one_time_axis = np.arange(-pre, post) / fs
    relative_time = np.tile(one_time_axis, (len(events), 1))
    drop = np.any(np.abs(trial_data[:, :3]) >= CLIP_THRESHOLD, axis=(1, 2)).astype(np.uint8)

    return {
        "trial_data": trial_data,
        "relative_time": relative_time,
        "cue_type": cue[events],
        "drop": drop,
    }


def find_dataset_path(dataset_name):
    matches = list(DATA_ROOT.rglob(f"{dataset_name}.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one source file for {dataset_name}, found {len(matches)}"
        )
    return matches[0]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        source = loadmat(find_dataset_path(dataset_name))
        data = np.asarray(source["data"], dtype=float)
        fs = int(np.asarray(source["SampleRate"]).squeeze())
        result = slice_trials_with_drop(data, fs)

        savemat(
            OUTPUT_DIR / f"{dataset_name}_sliced_with_drop.mat",
            {
                **result,
                "SampleRate": fs,
                "DataLabel": source["DataLabel"],
            },
        )

        dropped = int(result["drop"].sum())
        print(f"{dataset_name}: {len(result['drop'])} trials, drop={dropped}")


if __name__ == "__main__":
    main()
