"""Shared paths and analysis settings for the independent Q3 pipeline."""

from pathlib import Path


Q3_DIR = Path(__file__).resolve().parent
REPO_ROOT = Q3_DIR.parents[2]
RAW_DATA_DIR = (
    REPO_ROOT
    / "data"
    / "第二十三届中国研究生数学建模竞赛+-+中文题目"
    / "中文题目"
    / "C题"
)
Q1_CLEAN_DIR = REPO_ROOT / "src" / "C" / "q1" / "output" / "8riemann_denoise"
OUTPUT_DIR = Q3_DIR / "output"
INPUT_DIR = Q3_DIR / "input"
BEHAVIOR_LABELS_PATH = INPUT_DIR / "behavior_labels.csv"

RECORDS = (
    "VisualCogA_Task-1",
    "VisualCogA_Task-2",
    "VisualCogB_Task-1",
    "VisualCogB_Task-2",
)

# Q3 deliberately selects only these signals from the MAT files. The ninth
# channel (action/target-action marker) is not read or used.
EEG_CHANNELS = ("F3", "Fz", "F4")
EVENT_CHANNELS = ("VisCue", "TimeStamp")
ALLOWED_CHANNELS = frozenset((*EEG_CHANNELS, *EVENT_CHANNELS))

RAW_SAMPLE_RATE_HZ = 256.0
Q1_SAMPLE_RATE_HZ = 128.0
FILTER_BAND_HZ = (0.2, 24.0)

# The target time is an experimental schedule assumption. Since the action
# marker is excluded, this offset cannot be checked against an event channel.
TARGET_OFFSET_S = 2.2
TARGET_OFFSET_SOURCE = "protocol_schedule_assumption_unverified_from_allowed_channels"

# Epoch-relative windows used on Q1's uniformly filtered and downsampled trials.
ERP_BASELINE_WINDOW_S = (-0.20, 0.0)
ERP_CANDIDATE_WINDOW_S = (0.25, 0.50)
ERP_PEAK_WINDOW_S = (0.25, 0.60)
POWER_WINDOW_S = (0.20, 1.00)
BANDS_HZ = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 24.0),
}

RANDOM_SEED = 20260925
