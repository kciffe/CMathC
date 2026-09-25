"""Transparent point-dipole approximation with internal sources and scalp sensors.

Coordinates use a head-centred Cartesian frame (x<0 left, y<0 posterior,
z>0 superior). Electrodes and linked-mastoid references lie on a 90-mm outer
surface. Functional source proxies remain at explicitly defined internal
locations. The conductor is still approximated as homogeneous and infinite;
this is not a finite spherical or individualized head solution.
"""
import numpy as np

CONDUCTIVITY_S_PER_M = 0.33
HEAD_RADIUS_M = 0.09
HEAD_RADIUS_MM = HEAD_RADIUS_M * 1000.0

_ELECTRODES_MM = np.array([
    [-35.0, 55.0, 62.0],  # F3
    [0.0, 66.0, 61.0],    # Fz
    [35.0, 55.0, 62.0],   # F4
])
_REFERENCE_MM = np.array([
    [-70.0, -50.0, 5.0],
    [70.0, -50.0, 5.0],
])

# These are canonical functional-source proxies, not MRI-localized anatomy.
# x<0 is the left cerebral hemisphere; x>0 is the right hemisphere.
# The first two source pairs are hemisphere-specific. The last source is a
# single bilateral shape-preference opponent current, so template preference
# is never interpreted as a hemispheric location.
SOURCE_LABELS = (
    "early_visual_left_hemisphere_from_right_visual_field",
    "early_visual_right_hemisphere_from_left_visual_field",
    "configuration_left_hemisphere_from_right_visual_field",
    "configuration_right_hemisphere_from_left_visual_field",
    "bilateral_shape_preference_opponent_midline",
)
_SOURCE_MM = np.array([
    [-10.0, -58.0, 45.0],  # early visual: left hemisphere, radius 74.1 mm
    [10.0, -58.0, 45.0],   # early visual: right hemisphere
    [-36.0, -25.0, 50.0],  # configuration: left hemisphere, radius 66.5 mm
    [36.0, -25.0, 50.0],   # configuration: right hemisphere
    [0.0, -34.0, 56.0],    # bilateral preference opponent, midline, radius 65.5 mm
])


def _project_to_outer_surface(points_mm):
    """Place electrode/reference coordinates on the 90-mm outer shell."""
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("surface points must have shape [point, xyz]")
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("surface coordinates must be nonzero")
    return HEAD_RADIUS_M * points / norms


def _source_positions_m():
    """Return internal source coordinates without projecting them to scalp."""
    sources = _SOURCE_MM / 1000.0
    radii = np.linalg.norm(sources, axis=1)
    if np.any(radii <= 0.0) or np.any(radii >= HEAD_RADIUS_M):
        raise ValueError("all functional sources must lie strictly inside the head surface")
    return sources


def build_sensor_leadfield(conductivity=CONDUCTIVITY_S_PER_M):
    """Return V/(A m) leadfield for F3/Fz/F4 and five functional sources.

    Uses the homogeneous infinite-conductor point-dipole approximation
    ``V(r)=p·(r-r_s)/(4*pi*sigma*|r-r_s|^3)``. Dipole directions are radial
    from the head centre. Electrode and reference points are on the outer
    shell; source points remain at their declared internal depths.
    """
    if not np.isfinite(conductivity) or conductivity <= 0:
        raise ValueError("conductivity must be positive and finite")

    electrodes = _project_to_outer_surface(_ELECTRODES_MM)
    references = _project_to_outer_surface(_REFERENCE_MM)
    sources = _source_positions_m()
    normals = sources / np.linalg.norm(sources, axis=1, keepdims=True)

    def potential_at(points):
        delta = points[:, None, :] - sources[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        return np.einsum("sc,esc->es", normals, delta) / (
            4.0 * np.pi * conductivity * distance ** 3)

    leadfield = potential_at(electrodes) - potential_at(references).mean(axis=0, keepdims=True)
    if leadfield.shape != (3, len(SOURCE_LABELS)) or not np.isfinite(leadfield).all():
        raise FloatingPointError("invalid source-to-sensor leadfield")
    if np.linalg.matrix_rank(leadfield) != 3:
        raise FloatingPointError("fixed source geometry does not span three sensors")
    return leadfield


def geometry_manifest():
    """Return the complete, machine-readable geometry and semantic mapping."""
    electrode_surface = _project_to_outer_surface(_ELECTRODES_MM) * 1000.0
    reference_surface = _project_to_outer_surface(_REFERENCE_MM) * 1000.0
    source_mm = _SOURCE_MM.copy()
    source_radii = np.linalg.norm(source_mm, axis=1)
    return {
        "sensors": ["F3", "Fz", "F4"],
        "coordinate_axes": {"x": "negative=left, positive=right",
                            "y": "negative=posterior, positive=anterior",
                            "z": "positive=superior"},
        "head_outer_radius_mm": HEAD_RADIUS_MM,
        "sensor_coordinates_mm": electrode_surface.tolist(),
        "sensor_radii_mm": np.linalg.norm(electrode_surface, axis=1).tolist(),
        "reference": "average of approximate left/right mastoid coordinates",
        "reference_coordinates_mm": reference_surface.tolist(),
        "reference_radii_mm": np.linalg.norm(reference_surface, axis=1).tolist(),
        "sources": list(SOURCE_LABELS),
        "source_coordinates_mm": source_mm.tolist(),
        "source_radii_mm": source_radii.tolist(),
        "source_depths_mm": (HEAD_RADIUS_MM - source_radii).tolist(),
        "source_orientation": "radial from head centre; canonical approximation",
        "visual_field_to_hemisphere": {
            "left_visual_field": "right_hemisphere (contralateral)",
            "right_visual_field": "left_hemisphere (contralateral)"},
        "shape_preference_to_source": (
            "left-template preference minus right-template preference -> one bilateral midline opponent source"),
        "population_channel_semantics": {
            "early": ["left_visual_field", "right_visual_field"],
            "configuration": ["left_visual_field", "right_visual_field"],
            "shape_preference": ["left_triangle_template_preference", "right_triangle_template_preference"]},
        "conductivity_S_per_m": CONDUCTIVITY_S_PER_M,
        "leadfield_units": "V/(A m)",
        "forward_approximation": (
            "homogeneous infinite conductor point dipoles; not a finite spherical head solution"),
        "limitations": ["canonical coordinates, not MRI-localized anatomy",
                        "radial dipole directions are an explicit approximation",
                        "assumed linked-mastoid reference; recorded reference metadata not used",
                        "source currents are relative proxies, not calibrated A m"]}
