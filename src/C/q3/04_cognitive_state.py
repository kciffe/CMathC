"""Build transparent trial-level V/H/P functional-state proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from common import ensure_output_dir, write_csv, write_json
from config import EEG_CHANNELS


def main() -> None:
    output_dir = ensure_output_dir()
    feature_path = output_dir / "trial_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError("Run 03_extract_features.py first to create trial_features.csv")
    features = pd.read_csv(feature_path, encoding="utf-8-sig")

    erp_columns = [f"erp_mean_{channel}" for channel in EEG_CHANNELS]
    theta_columns = [f"log_theta_power_{channel}" for channel in EEG_CHANNELS]
    beta_columns = [f"log_beta_power_{channel}" for channel in EEG_CHANNELS]
    features["V_visual_like_raw"] = features[erp_columns].mean(axis=1)
    features["H_memory_like_raw"] = features[theta_columns].mean(axis=1)
    features["P_control_like_raw"] = features[beta_columns].mean(axis=1)

    state_sources = {
        "V_visual_like": ("V_visual_like_raw", "mean frontal ERP candidate amplitude"),
        "H_memory_like": ("H_memory_like_raw", "mean frontal log theta power"),
        "P_control_like": ("P_control_like_raw", "mean frontal log beta power"),
    }
    scale_rows: list[dict] = []
    for stage, row_indices in features.groupby("stage").groups.items():
        for score_name, (source_name, definition) in state_sources.items():
            values = pd.to_numeric(features.loc[row_indices, source_name], errors="coerce")
            center = float(values.mean())
            scale = float(values.std(ddof=0))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                scale = 1.0
            score_column = f"{score_name}_score_z"
            features.loc[row_indices, score_column] = (values - center) / scale
            scale_rows.append(
                {
                    "stage": stage,
                    "state_proxy": score_name,
                    "source_feature": source_name,
                    "definition": definition,
                    "pooled_center": center,
                    "pooled_scale_sd": scale,
                    "scope": "descriptive scaling within event stage; not a predictive train/test transform",
                }
            )

    features["u_cue_side"] = features["cue_side"].astype(float)
    features["u_stage_code"] = features["stage"].map(
        {"cue_locked": 0.0, "target_locked": 1.0}
    )
    features["task_type"] = "unresolved_from_allowed_channels"
    write_csv(features, output_dir / "state_scores.csv")
    write_csv(pd.DataFrame(scale_rows), output_dir / "state_proxy_definitions.csv")

    model_description = {
        "observation_unit": "one retained trial at one fixed event-locked analysis stage",
        "proxy_vector": ["V_visual_like", "H_memory_like", "P_control_like"],
        "proxy_definitions": {
            "V_visual_like": "frontal mean ERP candidate amplitude across F3/Fz/F4",
            "H_memory_like": "frontal mean log theta power across F3/Fz/F4",
            "P_control_like": "frontal mean log beta power across F3/Fz/F4",
        },
        "additional_feature": "AI_alpha = log(P_alpha,F4) - log(P_alpha,F3)",
        "inputs_u": ["VisCue direction", "event stage"],
        "task_type": "unresolved; not inferred from recording filenames or response-channel coding",
        "latent_factor_model_fitted": False,
        "reason": "three-channel trial-level observations do not identify unconstrained latent loadings; use predefined feature composites",
        "state_transition_model_fitted": False,
        "interpretation_limit": "functional feature proxies only; they do not identify LGN, hippocampal, or PFC sources",
    }
    write_json(model_description, output_dir / "cognitive_state_model.json")

    target = features.loc[features["stage"] == "target_locked"]
    score_columns = [f"{name}_score_z" for name in state_sources]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, column in zip(axes, score_columns):
        values = [
            target.loc[target["cue_side"] == side, column].dropna().to_numpy()
            for side in (-1, 1)
        ]
        ax.boxplot(values, tick_labels=["Left cue", "Right cue"], showfliers=False)
        ax.set_title(column.replace("_score_z", ""))
        ax.set_ylabel("Descriptive z score")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Target-locked trial-level functional state proxies")
    fig.savefig(output_dir / "state_proxy_scores.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {len(features)} trial-stage state proxy rows; no latent dynamics were fitted.")


if __name__ == "__main__":
    main()
