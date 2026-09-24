"""Shared event assumptions and paths for the parallel Q2 revision."""
from pathlib import Path
import numpy as np

Q2_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Q2_ROOT / "output" / "revision"
DT = 1.0
TIME_MS = np.arange(801, dtype=float)
MIN_TRIALS = 10
SEED = 20260924
STAGES = {
    "Stage1": {"onset_s": 0.0, "baseline": (-0.2, 0.0), "late": (450, 700)},
    "Stage2": {"onset_s": 2.2, "baseline": (2.0, 2.2), "late": (250, 500)},
}
