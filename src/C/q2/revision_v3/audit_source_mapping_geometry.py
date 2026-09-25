"""Write a small reproducible proof bundle for source semantics and geometry."""
import csv
import json
from pathlib import Path

import numpy as np

try:
    from . import config
    from .head_model import SOURCE_LABELS, build_sensor_leadfield, geometry_manifest
    from .model import _route_early_feedforward, map_population_to_source_channels
except ImportError:
    import config
    from head_model import SOURCE_LABELS, build_sensor_leadfield, geometry_manifest
    from model import _route_early_feedforward, map_population_to_source_channels


def _write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    out = config.OUTPUT_ROOT.parent / "revision_v3_source_mapping_audit"
    out.mkdir(parents=True, exist_ok=True)

    # One unit impulse per population input channel gives the exact linear
    # routing matrix from the six neural-population channels to five sources.
    route_rows = []
    routing = np.zeros((len(SOURCE_LABELS), 6), dtype=float)
    input_labels = (
        "early.left_visual_field", "early.right_visual_field",
        "configuration.left_visual_field", "configuration.right_visual_field",
        "shape_preference.left_triangle_template", "shape_preference.right_triangle_template",
    )
    for input_index, input_label in enumerate(input_labels):
        populations = np.zeros((3, 2, 1), dtype=np.float32)
        populations.reshape(6, 1)[input_index, 0] = 1.0
        output = map_population_to_source_channels(populations)[:, 0]
        routing[:, input_index] = output
        for source_index, source_label in enumerate(SOURCE_LABELS):
            route_rows.append({"input_channel": input_label, "source": source_label,
                               "coefficient": float(output[source_index])})

    # Hand-specified expected routes; these do not reuse the implementation.
    expected = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ], dtype=float)
    # Complete the literal routing matrix's remaining nonzero columns.
    expected[0, 1] = 1
    expected[2, 3] = 1
    expected[4, 5] = -1
    if not np.array_equal(routing, expected):
        raise AssertionError(f"population/source routing differs from declared map:\n{routing}")

    feedforward_rows = []
    for field_i, field_name in enumerate(("left_visual_field", "right_visual_field")):
        impulse = np.zeros((2, 1), dtype=np.float32)
        impulse[field_i, 0] = 1.0
        field_route, preference_route = _route_early_feedforward(impulse, 0, 0.0)
        expected_preference = np.full(2, config.SHAPE_FEEDFORWARD_FIELD_WEIGHTS[field_i])
        if not np.allclose(field_route, impulse[:, 0]) or not np.allclose(
                preference_route, expected_preference):
            raise AssertionError("early-to-template feedforward routing is inconsistent")
        feedforward_rows.append({
            "unit_input": field_name,
            "configuration_left_field_channel": float(field_route[0]),
            "configuration_right_field_channel": float(field_route[1]),
            "template_left_preference_channel": float(preference_route[0]),
            "template_right_preference_channel": float(preference_route[1]),
        })

    lead = build_sensor_leadfield()
    geometry = geometry_manifest()
    sensor_radii = np.asarray(geometry["sensor_radii_mm"], dtype=float)
    reference_radii = np.asarray(geometry["reference_radii_mm"], dtype=float)
    source_radii = np.asarray(geometry["source_radii_mm"], dtype=float)
    depths = np.asarray(geometry["source_depths_mm"], dtype=float)
    mirrored = np.allclose(lead[0, 0], lead[2, 1], rtol=1e-10, atol=1e-12)
    if (lead.shape != (3, 5) or np.linalg.matrix_rank(lead) != 3
            or not np.allclose(sensor_radii, 90.0, atol=1e-9)
            or not np.allclose(reference_radii, 90.0, atol=1e-9)
            or not np.all(source_radii < 90.0) or not np.all(depths > 0.0)
            or not mirrored):
        raise AssertionError("source/sensor geometry checks failed")

    route_columns = {f"source_{i + 1}": label for i, label in enumerate(SOURCE_LABELS)}
    fit_manifest_path = config.OUTPUT_ROOT / "manifest.json"
    fit_manifest = (json.loads(fit_manifest_path.read_text(encoding="utf-8"))
                    if fit_manifest_path.exists() else {})
    saved_fit_schema = fit_manifest.get("source_mapping_schema")
    saved_fits_current = saved_fit_schema == config.SOURCE_MAPPING_SCHEMA
    _write_csv(out / "population_to_source_routing.csv", [
        {"input_channel": name, **{f"source_{i + 1}": float(routing[i, j])
                                    for i in range(len(SOURCE_LABELS))}}
        for j, name in enumerate(input_labels)])
    _write_csv(out / "early_to_template_preference_routing.csv", feedforward_rows)
    _write_csv(out / "leadfield.csv", [
        {"sensor": sensor, **{f"source_{i + 1}": float(value)
                              for i, value in enumerate(row)}}
        for sensor, row in zip(config.CHANNELS, lead)])

    proof = {
        "status": "PASS",
        "code_version": "revision_v3_bilateral_preference_feedforward_and_source_geometry_fix",
        "current_source_mapping_schema": config.SOURCE_MAPPING_SCHEMA,
        "saved_fit_manifest_schema": saved_fit_schema,
        "saved_fit_outputs_match_current_schema": saved_fits_current,
        "source_labels": list(SOURCE_LABELS),
        "source_column_meanings": route_columns,
        "routing_matrix_rows_sources_columns_inputs": routing.tolist(),
        "routing_matrix_input_order": list(input_labels),
        "semantic_checks": {
            "left_visual_field_to_right_hemisphere_source": bool(routing[1, 0] == 1.0),
            "right_visual_field_to_left_hemisphere_source": bool(routing[0, 1] == 1.0),
            "configuration_halves_use_same_contralateral_route": bool(
                routing[3, 2] == 1.0 and routing[2, 3] == 1.0),
            "left_minus_right_template_preference_to_single_midline_source": bool(
                routing[4, 4] == 1.0 and routing[4, 5] == -1.0),
            "template_preference_has_no_hemisphere_source_columns": bool(
                np.all(routing[:4, 4:] == 0.0)),
            "configuration_group_retains_field_aligned_feedforward": True,
            "both_template_preference_channels_receive_same_bilateral_pool": bool(
                np.allclose([r["template_left_preference_channel"] for r in feedforward_rows],
                            [r["template_right_preference_channel"] for r in feedforward_rows])),
            "template_feedforward_weights_left_right":
                list(config.SHAPE_FEEDFORWARD_FIELD_WEIGHTS),
        },
        "geometry_checks": {
            "outer_head_radius_mm": geometry["head_outer_radius_mm"],
            "sensor_radii_mm": sensor_radii.tolist(),
            "reference_radii_mm": reference_radii.tolist(),
            "source_radii_mm": source_radii.tolist(),
            "source_depths_below_outer_surface_mm": depths.tolist(),
            "all_sources_strictly_inside_surface": bool(np.all(source_radii < 90.0)),
            "all_sensor_and_reference_points_on_surface": bool(
                np.allclose(sensor_radii, 90.0, atol=1e-9)
                and np.allclose(reference_radii, 90.0, atol=1e-9)),
            "leadfield_shape": list(lead.shape),
            "leadfield_rank": int(np.linalg.matrix_rank(lead)),
            "mirrored_hemisphere_leadfield_symmetry_F3_LH_equals_F4_RH": bool(mirrored),
            "leadfield_V_per_A_m": lead.tolist(),
            "forward_approximation": geometry["forward_approximation"],
        },
        "limitations": geometry["limitations"],
        "note": "This proves the implemented routing and coordinates, not anatomical localization or EEG model validity. The bilateral midline opponent source remains a modeling hypothesis.",
    }
    (out / "source_semantics_geometry_audit.json").write_text(
        json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_rows = []
    for label, xyz, radius, depth in zip(
            geometry["sources"], geometry["source_coordinates_mm"], source_radii, depths):
        source_rows.append(
            f"| `{label}` | {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f} | "
            f"{radius:.2f} | {depth:.2f} |")
    stale_note = ("Fit outputs under `output/revision_v3` have the current mapping schema."
                  if saved_fits_current else
                  "**Existing fitted outputs are stale:** their manifest has no current source-mapping schema. "
                  "Downstream scripts now reject them; rerun `python src/C/q2/revision_v3/run_v3.py` "
                  "before using or regenerating fitted curves and figures.")
    readme = [
        "# Source semantics and geometry audit", "",
        "This deterministic audit checks the six population input channels, their mapping into five source proxies, and the source/electrode coordinates.",
        "The detailed machine-readable evidence is in `source_semantics_geometry_audit.json`; `population_to_source_routing.csv`, `early_to_template_preference_routing.csv`, and `leadfield.csv` contain the literal routing tables and matrix.",
        "", "## Semantic routing", "",
        "| Input meaning | Source interpretation |", "|---|---|",
        "| Early left visual-field half | Right-hemisphere early visual proxy |",
        "| Early right visual-field half | Left-hemisphere early visual proxy |",
        "| Configuration left visual-field half | Right-hemisphere configuration proxy |",
        "| Configuration right visual-field half | Left-hemisphere configuration proxy |",
        "| Left-triangle preference minus right-triangle preference | One signed bilateral midline opponent proxy; no hemisphere is inferred from preference |",
        "| Early left/right visual field to the template-preference group | Both preference channels receive the same pre-fixed 0.5/0.5 field average; template-specific drives carry the preference distinction |",
        "", "## Geometry and leadfield", "",
        "Electrodes and approximate linked-mastoid reference points lie on the 90 mm outer surface. The five canonical source proxies are internal:",
        "", "| Source proxy | Coordinate (x, y, z) mm | Radius mm | Depth below surface mm |", "|---|---:|---:|---:|"]
    readme.extend(source_rows)
    readme += [
        "", f"Leadfield shape/rank: `{list(lead.shape)}` / `{int(np.linalg.matrix_rank(lead))}`.",
        "The leadfield uses a homogeneous infinite-conductor point-dipole approximation, **not** a finite spherical head solution. Coordinates and radial dipole orientations are canonical assumptions, not MRI-localized anatomy; source currents are relative proxies, not calibrated A·m.",
        "", "## Saved result status", "", stale_note,
        "", "This audit proves the implemented channel routing, matrix dimensions, and geometric radii. It does not prove anatomical localization or physiological validity.", ""]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return proof


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
