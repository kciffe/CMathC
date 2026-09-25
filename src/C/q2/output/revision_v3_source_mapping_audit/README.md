# Source semantics and geometry audit

This deterministic audit checks the six population input channels, their mapping into five source proxies, and the source/electrode coordinates.
The detailed machine-readable evidence is in `source_semantics_geometry_audit.json`; `population_to_source_routing.csv` and `leadfield.csv` contain the literal matrices.

## Semantic routing

| Input meaning | Source interpretation |
|---|---|
| Early left visual-field half | Right-hemisphere early visual proxy |
| Early right visual-field half | Left-hemisphere early visual proxy |
| Configuration left visual-field half | Right-hemisphere configuration proxy |
| Configuration right visual-field half | Left-hemisphere configuration proxy |
| Left-triangle preference minus right-triangle preference | One signed bilateral midline opponent proxy; no hemisphere is inferred from preference |

## Geometry and leadfield

Electrodes and approximate linked-mastoid reference points lie on the 90 mm outer surface. The five canonical source proxies are internal:

| Source proxy | Coordinate (x, y, z) mm | Radius mm | Depth below surface mm |
|---|---:|---:|---:|
| `early_visual_left_hemisphere_from_right_visual_field` | -10.0, -58.0, 45.0 | 74.09 | 15.91 |
| `early_visual_right_hemisphere_from_left_visual_field` | 10.0, -58.0, 45.0 | 74.09 | 15.91 |
| `configuration_left_hemisphere_from_right_visual_field` | -36.0, -25.0, 50.0 | 66.49 | 23.51 |
| `configuration_right_hemisphere_from_left_visual_field` | 36.0, -25.0, 50.0 | 66.49 | 23.51 |
| `bilateral_shape_preference_opponent_midline` | 0.0, -34.0, 56.0 | 65.51 | 24.49 |

Leadfield shape/rank: `[3, 5]` / `3`.
The leadfield uses a homogeneous infinite-conductor point-dipole approximation, **not** a finite spherical head solution. Coordinates and radial dipole orientations are canonical assumptions, not MRI-localized anatomy; source currents are relative proxies, not calibrated A·m.

## Saved result status

**Existing fitted outputs are stale:** their manifest has no current source-mapping schema. Downstream scripts now reject them; rerun `python src/C/q2/revision_v3/run_v3.py` before using or regenerating fitted curves and figures.

This audit proves the implemented channel routing, matrix dimensions, and geometric radii. It does not prove anatomical localization or physiological validity.
