"""Fixed paths, timing, stimulus, and model constants for revision v3."""
from pathlib import Path
import json

import numpy as np

Q2_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Q2_ROOT / "input"
REAL_ROOT = Q2_ROOT.parent / "q1" / "output" / "8riemann_denoise"
OUTPUT_ROOT = Q2_ROOT / "output" / "revision_v3"
SOURCE_MAPPING_SCHEMA = "contralateral_visual_field_5source_midline_opponent_bilateral_ff_v2"
SEED = 20260924

TIME_MS = np.arange(801, dtype=np.float64)
STAGES = {
    "Stage1": {"onset_s": 0.0, "baseline": (-0.2, 0.0), "late": (450.0, 700.0), "offset_ms": 200.0},
    "Stage2": {"onset_s": 2.2, "baseline": (2.0, 2.2), "late": (250.0, 500.0), "offset_ms": None},
}
MIN_TRIALS = 10
CHANNELS = ("F3", "Fz", "F4")
DATASETS = {
    "Task1": ("VisualCogA_Task-1", "VisualCogA_Task-2"),
    "Task2": ("VisualCogB_Task-1", "VisualCogB_Task-2"),
}
STIMULI = {
    ("Stage1", "left"): ("stage1_cue_left", "stage1_baseline_circle"),
    ("Stage1", "right"): ("stage1_cue_right", "stage1_baseline_circle"),
    ("Stage2", "dots"): ("stage2_task1_target_dots", "stage2_baseline_blank"),
    ("Stage2", "inward"): ("stage2_task2_target_inward", "stage2_baseline_blank"),
    ("Stage2", "outward"): ("stage2_task2_target_outward", "stage2_baseline_blank"),
}

# Spatial parameters are defined in original 256x256 pixel units.
PIXEL_PARAMS = {"dog_center_sigma": 3.0, "dog_surround_sigma": 8.0,
                "gabor_sigma": 4.0, "gabor_wavelength": 10.0,
                "configuration_blur": 2.0}
SCALES = (64, 128, 256)
POOL_SIZE = 8
GABOR_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)
FRONTEND_FEATURE_STRIDE_MS = 4.0
TAU_ADAPT_DEFAULT = 80.0
LGN_TAU_MS = {"tcr": 18.0, "in": 12.0, "trn": 25.0}
WC_FIXED = {"w_ee": 1.4, "w_ei": 1.1, "w_ie": 1.0, "w_ii": 0.8,
            "g_p": 2.2, "g_q": 1.2, "tau_e_early": 18.0,
            "tau_i_early": 10.0, "delay_early": 8.0,
            "tau_i_ratio": 0.55, "delay_shape": 33.0,
            "w_feedforward_e": 0.9, "w_feedforward_i": 0.45}
# The template-preference population receives the same fixed mean of the two
# early visual-field channels; L/R here denotes template preference, not field.
SHAPE_FEEDFORWARD_FIELD_WEIGHTS = (0.5, 0.5)
PARAM_BOUNDS = {"tau_s": (20.0, 100.0), "g_i": (0.5, 1.5), "tau_a": (40.0, 160.0)}
PARAM_DEFAULTS = {"tau_s": 40.0, "g_i": 1.0, "tau_a": 80.0}
FIT_STARTS = (np.array([40.0, 1.0, 80.0]), np.array([28.0, 0.7, 55.0]), np.array([75.0, 1.3, 135.0]))
MAX_NFEV_PER_START = 60
# Profile the expensive visual adaptation parameter once per candidate, then
# optimize only the two inexpensive Wilson-Cowan parameters at each profile.
FIT_TAU_A_GRID = np.arange(40.0, 160.0 + 1e-9, 20.0)
# L-BFGS-B's default 1e-8 finite-difference step is below float32 state
# resolution. These steps apply to tau_s (ms) and g_i, respectively.
FIT_FINITE_DIFF_STEPS = np.array([0.5, 0.01], dtype=float)

U0 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
U1 = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
U2 = np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)
# Rows are the orthonormal F3/Fz/F4 sensor modes; columns map modes to sensors.
U_OBS = np.stack([U0, U1, U2])
U_MATRIX = U_OBS.T
FRONTEND_ROUTE = "opponent"
OBSERVATION_RANK = 3

FEATURE_WINDOWS_MS = ((80.0, 200.0), (250.0, 450.0), (450.0, 700.0))
LDA_SHRINKAGE = 0.1
SPLIT_HALF_SEED = 20260925
SPLIT_HALF_REPEATS = 500
PERMUTATION_SEED = 20260926
PERMUTATION_REPEATS = 1999


def require_current_source_mapping_manifest(path=None):
    """Refuse downstream use of fits made with incompatible routing semantics."""
    manifest_path = OUTPUT_ROOT / "manifest.json" if path is None else Path(path)
    if not manifest_path.exists():
        raise RuntimeError(
            f"current revision_v3 manifest not found at {manifest_path}; "
            "rerun revision_v3/run_v3.py before downstream analysis")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_mapping_schema") != SOURCE_MAPPING_SCHEMA:
        raise RuntimeError(
            "saved revision_v3 fits use older source or feedforward routing semantics; "
            "rerun revision_v3/run_v3.py before consuming them")
    return manifest
