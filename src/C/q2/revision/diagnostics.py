"""Mechanism controls separate from fitting real EEG."""
import numpy as np
from model import LEAD_FIELD, load_contrast, simulate, cortex_from_v1


def relative_difference(a, b):
    return float(2 * np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-15))


def run_diagnostics(predictions):
    left = predictions[("Stage1", "left", 40)]
    right = predictions[("Stage1", "right", 40)]
    inward = predictions[("Stage2", "inward", 40)]
    outward = predictions[("Stage2", "outward", 40)]
    permutation = [0, 3, 2, 1]
    metrics = {"lead_field_rank": int(np.linalg.matrix_rank(LEAD_FIELD)),
               "lead_field_singular_values": np.linalg.svd(LEAD_FIELD, compute_uv=False).tolist(),
               "observable_modes": ["common=(F3+Fz+F4)/3", "lateral=(F4-F3)/2"],
               "unmodelled_topology": "F3-2*Fz+F4",
               "v1_mirror_relative_error": relative_difference(right["v1_joint_feature"], left["v1_joint_feature"][permutation, :, ::-1])}
    for name, a, b in (("cue", left, right), ("layout", inward, outward)):
        for key in ("v1_joint_feature", "it_joint_feature", "it_configuration_feature", "eeg"):
            metrics[f"{name}_{key}_relative_difference"] = relative_difference(a[key], b[key])
        metrics[f"{name}_global_orientation_relative_difference"] = relative_difference(
            a["v1_joint_feature"].mean(axis=(1, 2)), b["v1_joint_feature"].mean(axis=(1, 2)))
    # Fixed spatial permutation preserves total energy for each direction exactly.
    rng = np.random.default_rng(20260924)
    drive = left["v1_drive"]
    shuffle = rng.permutation(64)
    shuffled = drive.reshape(4, 64, -1)[:, shuffle].reshape(drive.shape)
    control = cortex_from_v1(shuffled, left["time_ms"])
    metrics["shuffle_global_energy_max_error"] = float(np.max(np.abs(drive.sum(axis=(1, 2)) - shuffled.sum(axis=(1, 2)))))
    metrics["shuffle_configuration_relative_difference"] = relative_difference(left["it_configuration_feature"], control["it_configuration_feature"])
    # Sum over positions retains orientation-pair/displacement identity.
    metrics["shuffle_position_pooled_configuration_difference"] = relative_difference(
        left["it_configuration_feature"].sum(axis=(1, 2)), control["it_configuration_feature"].sum(axis=(1, 2)))
    contrast = load_contrast("Stage1", "left")
    half = simulate(contrast, "Stage1", dt_ms=.5)
    no_offset = simulate(contrast, "Stage1", include_offset=False)
    metrics["dt_half_eeg_relative_error"] = relative_difference(left["eeg"], half["eeg"][:, ::2])
    metrics["offset_ablation_eeg_relative_difference"] = relative_difference(left["eeg"], no_offset["eeg"])
    metrics["stage2_drive_final_mean"] = float(predictions[("Stage2", "dots", 40)]["lgn_drive_mean"][:, -1].sum())
    metrics["parameter_sensitivity"] = []
    for tau in (30, 40, 50, 60):
        a = predictions[("Stage1", "left", tau)]
        b = predictions[("Stage1", "right", tau)]
        metrics["parameter_sensitivity"].append({"tau_it_ms": tau,
            "cue_configuration_difference": relative_difference(a["it_configuration_feature"], b["it_configuration_feature"]),
            "eeg_change_from_40": relative_difference(a["eeg"], left["eeg"])})
    metrics["structural_pass"] = bool(metrics["v1_mirror_relative_error"] < 1e-5
        and metrics["cue_it_configuration_feature_relative_difference"] > .01
        and metrics["shuffle_position_pooled_configuration_difference"] > .01
        and metrics["dt_half_eeg_relative_error"] < .05
        and metrics["stage2_drive_final_mean"] > 0)
    return metrics
