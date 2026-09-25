"""Regression checks for visual-field, shape-preference, and source geometry semantics."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from head_model import SOURCE_LABELS, build_sensor_leadfield, geometry_manifest
from config import SOURCE_MAPPING_SCHEMA, require_current_source_mapping_manifest
from model import (_route_early_feedforward, map_population_to_source_channels,
                   simulate_forward)


def test_shape_preference_feedforward_is_same_fixed_bilateral_visual_field_pool():
    early_activity = np.zeros((2, 5), dtype=np.float32)
    early_activity[0, 4] = 2.0  # left visual field
    early_activity[1, 4] = 6.0  # right visual field

    field_route, preference_route = _route_early_feedforward(
        early_activity, step=4, delay_samples=0.0)

    np.testing.assert_array_equal(field_route, [2.0, 6.0])
    np.testing.assert_array_equal(preference_route, [4.0, 4.0])


def test_visual_field_halves_route_to_contralateral_early_and_configuration_sources():
    populations = np.zeros((3, 2, 1), dtype=np.float32)
    populations[0, 0, 0] = 1.0  # left visual field
    populations[0, 1, 0] = 2.0  # right visual field
    populations[1, 0, 0] = 3.0
    populations[1, 1, 0] = 4.0

    sources = map_population_to_source_channels(populations)

    np.testing.assert_array_equal(sources[:, 0], [2.0, 1.0, 4.0, 3.0, 0.0])


def test_shape_preference_pair_becomes_one_signed_midline_source_not_two_hemispheres():
    populations = np.zeros((3, 2, 2), dtype=np.float32)
    populations[2, 0] = [5.0, 1.0]  # left-triangle-preferring population
    populations[2, 1] = [2.0, 4.0]  # right-triangle-preferring population

    sources = map_population_to_source_channels(populations)

    np.testing.assert_array_equal(sources[4], [3.0, -3.0])
    np.testing.assert_array_equal(sources[:4], np.zeros((4, 2), dtype=np.float32))
    assert SOURCE_LABELS[4] == "bilateral_shape_preference_opponent_midline"
    assert len(SOURCE_LABELS) == 5


def test_sensor_and_reference_are_on_outer_surface_while_sources_remain_inside():
    geometry = geometry_manifest()
    assert geometry["coordinate_axes"] == {
        "x": "negative=left, positive=right",
        "y": "negative=posterior, positive=anterior",
        "z": "positive=superior",
    }
    assert len(geometry["sources"]) == 5
    np.testing.assert_allclose(geometry["sensor_radii_mm"], [90.0, 90.0, 90.0], atol=1e-9)
    np.testing.assert_allclose(geometry["reference_radii_mm"], [90.0, 90.0], atol=1e-9)
    assert max(geometry["source_radii_mm"]) < 80.0
    assert min(geometry["source_depths_mm"]) > 10.0
    assert geometry["forward_approximation"] == "homogeneous infinite conductor point dipoles; not a finite spherical head solution"


def test_leadfield_and_full_forward_output_use_five_explicit_source_channels():
    lead = build_sensor_leadfield()
    assert lead.shape == (3, 5)
    assert np.linalg.matrix_rank(lead) == 3

    n = 51
    b = np.zeros((8, 8, n), dtype=np.float32)
    shape = np.zeros_like(b)
    front = {"time_ms": np.arange(n, dtype=float), "B": b, "shape": shape,
             "H_L": shape.copy(), "H_R": shape.copy(),
             "dir_left": np.zeros(n, dtype=np.float32),
             "dir_right": np.zeros(n, dtype=np.float32)}
    result = simulate_forward(front, drive_scales=np.ones((3, 2)))
    assert result.source_proxy.shape == (5, n)
    assert result.eeg.shape == (3, n)
    np.testing.assert_allclose(result.eeg, lead @ result.source_proxy, rtol=1e-6, atol=1e-10)


def test_downstream_results_reject_pre_correction_six_source_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"revision":"revision_v3"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="rerun revision_v3/run_v3.py"):
        require_current_source_mapping_manifest(manifest)


def test_downstream_results_accept_current_source_mapping_schema(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"source_mapping_schema":"' + SOURCE_MAPPING_SCHEMA + '"}', encoding="utf-8")

    result = require_current_source_mapping_manifest(manifest)

    assert result["source_mapping_schema"] == SOURCE_MAPPING_SCHEMA
