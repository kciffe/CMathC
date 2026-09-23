from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, resample


DATASETS = [
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
]

INPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = INPUT_DIR / "7filter_downsample"
LEGACY_PLOT_DATASET = DATASETS[0]

EEG_CHANNELS = (0, 1, 2)  # Fz, F3, F4
EVENT_CHANNELS = (7, 8)  # VisCue, Action
TIMESTAMP_CHANNEL = 9

HIGH_PASS_HZ = 0.2
LOW_PASS_HZ = 24.0
FILTER_ORDER = 4
OUTPUT_SAMPLE_RATE = 128


def load_plotter():
    plotter_path = Path(__file__).resolve().with_name("7plot_trial.py")
    spec = importlib.util.spec_from_file_location("q1_plot_trial", plotter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plot helper: {plotter_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_trials


def filter_eeg_channels(trial_data, sample_rate):
    """Bandpass only the three EEG channels; retain all other channels."""
    if not 0 < HIGH_PASS_HZ < LOW_PASS_HZ < sample_rate / 2:
        raise ValueError(f"Invalid filter band for sample rate {sample_rate} Hz")

    b, a = butter(
        FILTER_ORDER,
        [HIGH_PASS_HZ, LOW_PASS_HZ],
        btype="bandpass",
        fs=sample_rate,
    )
    filtered = np.asarray(trial_data, dtype=np.float64).copy()
    for channel in EEG_CHANNELS:
        filtered[:, channel, :] = filtfilt(b, a, filtered[:, channel, :], axis=-1)
    return filtered


def downsample_trial_data(filtered_trial_data, original_trial_data, sample_rate):
    """Downsample continuous signals and preserve discrete event markers."""
    if sample_rate % OUTPUT_SAMPLE_RATE != 0:
        raise ValueError(
            f"Sample rate {sample_rate} Hz is not an integer multiple of "
            f"{OUTPUT_SAMPLE_RATE} Hz"
        )

    factor = sample_rate // OUTPUT_SAMPLE_RATE
    n_trials, n_channels, n_samples = filtered_trial_data.shape
    if n_samples % factor != 0:
        raise ValueError(
            f"Trial length {n_samples} is not divisible by downsample factor {factor}"
        )

    new_length = n_samples // factor
    downsampled = np.empty((n_trials, n_channels, new_length), dtype=np.float64)

    for channel in range(n_channels):
        if channel in EVENT_CHANNELS:
            blocks = original_trial_data[:, channel, :].reshape(
                n_trials, new_length, factor
            )
            nonzero = blocks != 0
            first_nonzero = np.argmax(nonzero, axis=-1)
            picked = np.take_along_axis(
                blocks, first_nonzero[..., np.newaxis], axis=-1
            )[..., 0]
            downsampled[:, channel, :] = np.where(nonzero.any(axis=-1), picked, 0)
        elif channel == TIMESTAMP_CHANNEL:
            downsampled[:, channel, :] = original_trial_data[:, channel, ::factor]
        else:
            downsampled[:, channel, :] = resample(
                filtered_trial_data[:, channel, :], new_length, axis=-1
            )

    return downsampled


def get_action_onsets(trial_data, relative_time):
    action_onsets = np.full(trial_data.shape[0], np.nan, dtype=np.float64)
    if trial_data.shape[1] <= EVENT_CHANNELS[1]:
        return action_onsets

    for trial_index, action in enumerate(trial_data[:, EVENT_CHANNELS[1], :]):
        onset_indices = np.flatnonzero(
            (action != 0) & np.r_[True, action[:-1] == 0]
        )
        after_cue = onset_indices[relative_time[trial_index, onset_indices] >= 0]
        if after_cue.size:
            action_onsets[trial_index] = relative_time[trial_index, after_cue[0]]
    return action_onsets


def event_label_for_dataset(data_label):
    labels = np.asarray(data_label, dtype=object).reshape(-1)
    if labels.size <= EVENT_CHANNELS[1]:
        return "事件开始（通道9）"

    label = labels[EVENT_CHANNELS[1]]
    while isinstance(label, np.ndarray) and label.size == 1:
        label = label.item()
    channel_name = str(label).split(":", maxsplit=1)[0]
    display_name = {"Action": "动作", "TgtAct": "目标动作"}.get(
        channel_name, channel_name
    )
    return f"{display_name}事件开始（通道9）"


def plot_dataset(
    plot_trials,
    dataset_name,
    original_trial_data,
    filtered_trial_data,
    downsampled_trial_data,
    relative_time,
    downsampled_relative_time,
    cue_type,
    event_label,
):
    action_onsets = get_action_onsets(original_trial_data, relative_time)
    plot_stages = [
        ("before_filter", original_trial_data, relative_time, "滤波前"),
        ("after_filter", filtered_trial_data, relative_time, "滤波后"),
        ("before_downsample", filtered_trial_data, relative_time, "降采样前"),
        (
            "after_downsample",
            downsampled_trial_data,
            downsampled_relative_time,
            "降采样后",
        ),
    ]

    for stage, data, time_axis, stage_title in plot_stages:
        plot_dir = OUTPUT_DIR / stage
        plot_dir.mkdir(parents=True, exist_ok=True)
        if dataset_name == LEGACY_PLOT_DATASET:
            filename = f"{stage}.png"
        else:
            filename = f"{dataset_name}_{stage}.png"

        plot_trials(
            data,
            time_axis,
            cue_type,
            action_onsets,
            plot_dir / filename,
            f"{dataset_name} {stage_title}",
            event_label,
        )


def process_dataset(dataset_name, plot_trials):
    input_path = INPUT_DIR / f"{dataset_name}_sliced_with_drop.mat"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing sliced input: {input_path}")

    source = loadmat(input_path)
    original_trial_data = np.asarray(source["trial_data"], dtype=np.float64)
    original_relative_time = np.asarray(source["relative_time"], dtype=np.float64)
    source_sample_rate = int(np.asarray(source["SampleRate"]).squeeze())
    cue_type = np.asarray(source["cue_type"]).squeeze()

    if original_trial_data.ndim != 3 or original_trial_data.shape[1] < 10:
        raise ValueError(
            f"Expected trials x at least 10 channels x samples, got "
            f"{original_trial_data.shape} in {input_path.name}"
        )
    if original_relative_time.shape != (
        original_trial_data.shape[0],
        original_trial_data.shape[2],
    ):
        raise ValueError(
            f"relative_time shape {original_relative_time.shape} does not match "
            f"trial_data shape {original_trial_data.shape}"
        )
    if cue_type.size != original_trial_data.shape[0]:
        raise ValueError(
            f"cue_type has {cue_type.size} entries for "
            f"{original_trial_data.shape[0]} trials"
        )

    filtered_trial_data = filter_eeg_channels(
        original_trial_data, source_sample_rate
    )
    downsampled_trial_data = downsample_trial_data(
        filtered_trial_data, original_trial_data, source_sample_rate
    )
    factor = source_sample_rate // OUTPUT_SAMPLE_RATE
    downsampled_relative_time = original_relative_time[:, ::factor].copy()

    # Start from every source field so drop, DataLabel, and any existing
    # annotations survive. Replace only the canonical data/time/rate fields;
    # the original sliced MAT remains the source of unmodified trial samples.
    result = {
        key: value
        for key, value in source.items()
        if not key.startswith("__")
    }
    result.update(
        {
            "trial_data": downsampled_trial_data,
            "relative_time": downsampled_relative_time,
            "SampleRate": np.array([[OUTPUT_SAMPLE_RATE]], dtype=np.int32),
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{dataset_name}_filtered_downsample.mat"
    savemat(output_path, result, do_compression=True)

    plot_dataset(
        plot_trials,
        dataset_name,
        original_trial_data,
        filtered_trial_data,
        downsampled_trial_data,
        original_relative_time,
        downsampled_relative_time,
        cue_type,
        event_label_for_dataset(source["DataLabel"]),
    )

    dropped = int(np.asarray(source.get("drop", [])).sum())
    print(
        f"{dataset_name}: trials={original_trial_data.shape[0]}, "
        f"drop={dropped}, channels={original_trial_data.shape[1]}, "
        f"samples={original_trial_data.shape[2]}->{downsampled_trial_data.shape[2]}, "
        f"output={output_path}"
    )


def main():
    plot_trials = load_plotter()
    for dataset_name in DATASETS:
        process_dataset(dataset_name, plot_trials)


if __name__ == "__main__":
    main()
