from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round1_validation as rv
from frontend import _simulate_lgn


def test_stage1_timeline_has_a_pre_cue_state_and_exact_offset():
    time = np.array([-1.0, 0.0, 199.0, 200.0, 2199.0, 2200.0, 2999.0])
    scene = rv.build_scene_timeline(time, target_onset_ms=2200.0)
    assert scene.tolist() == [0, 1, 1, 0, 0, 2, 2]


def test_effective_mapping_recovers_synthetic_sensor_mix():
    rng = np.random.default_rng(10)
    source = rng.normal(size=(2, 6, 80))
    true_map = rng.normal(size=(3, 6))
    observed = np.einsum("rs,cst->crt", true_map, source)
    fitted = rv.fit_effective_mapping(source, observed, np.zeros((3, 6)), alpha=1e-10)
    np.testing.assert_allclose(fitted, true_map, atol=1e-8, rtol=1e-8)


def test_shape_preference_channels_are_not_named_as_cortical_hemispheres():
    labels = rv.source_channel_labels()
    assert labels[4:] == ("triangle_left_preference", "triangle_right_preference")
    assert not any("cortex_left" in label or "cortex_right" in label for label in labels)


def test_dynamic_lgn_accepts_warmed_state_and_switches_full_scenes():
    base = np.zeros((8, 8), dtype=np.float32)
    base[2:6, 2:6] = -0.4
    cue = base.copy()
    cue[3:5, 1:3] = -0.8
    target = np.zeros_like(base)
    target[1:3, 1:3] = -0.5
    warm = _simulate_lgn(base, "Stage1", np.arange(101.0), 80.0,
                         include_offset=False, return_state=True)
    state = warm[-1]
    out = _simulate_lgn({"scene_contrasts": np.stack([base, cue, target]),
                         "scene_index": np.array([0, 1, 0, 2, 2])},
                        "Stage1", np.arange(5.0), 80.0,
                        initial_state=state, return_state=True)
    assert out[0].shape == (8, 8, 5)
    assert np.isfinite(out[0]).all()
    assert out[-1]["tcr"].shape == (2, 8, 8)
    assert np.max(np.abs(out[-1]["adaptation"])) > 0.0
