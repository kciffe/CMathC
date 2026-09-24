"""Transparent point-dipole forward approximation for three EEG sensors.

Coordinates are canonical approximate 10-20 positions. The medium is treated
as homogeneous; reference is an assumed linked-mastoid average. This is a
reproducible teaching model, not an individualized head model.
"""
import numpy as np

CONDUCTIVITY_S_PER_M = 0.33
HEAD_RADIUS_M = 0.09
_ELECTRODES_MM = np.array([[-35.0, 55.0, 62.0], [0.0, 66.0, 61.0], [35.0, 55.0, 62.0]])
_REFERENCE_MM = np.array([[-70.0, -50.0, 5.0], [70.0, -50.0, 5.0]])
_SOURCE_MM = np.array([[-12.0, -62.0, 58.0], [12.0, -62.0, 58.0],
                       [-42.0, -20.0, 60.0], [42.0, -20.0, 60.0],
                       [-30.0, 18.0, 68.0], [30.0, 18.0, 68.0]])


def _on_sphere(points_mm):
    points = np.asarray(points_mm, dtype=float) / 1000.0
    return HEAD_RADIUS_M * points / np.linalg.norm(points, axis=-1, keepdims=True)


def build_sensor_leadfield(conductivity=CONDUCTIVITY_S_PER_M):
    """Return V/(A m) leadfield for rows F3/Fz/F4 and six fixed dipoles.

    Uses V(r)=p·(r-r_s)/(4*pi*sigma*|r-r_s|^3) in a homogeneous conductor;
    subtracts the mean potential at the two assumed mastoid references.
    """
    if not np.isfinite(conductivity) or conductivity <= 0:
        raise ValueError("conductivity must be positive and finite")
    electrodes, references, sources = map(_on_sphere, (_ELECTRODES_MM, _REFERENCE_MM, _SOURCE_MM))
    normals = sources / np.linalg.norm(sources, axis=1, keepdims=True)

    def potential_at(points):
        delta = points[:, None, :] - sources[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        return np.einsum("sc,esc->es", normals, delta) / (
            4.0 * np.pi * conductivity * distance ** 3)

    leadfield = potential_at(electrodes) - potential_at(references).mean(axis=0, keepdims=True)
    if leadfield.shape != (3, 6) or not np.isfinite(leadfield).all():
        raise FloatingPointError("invalid source-to-sensor leadfield")
    if np.linalg.matrix_rank(leadfield) != 3:
        raise FloatingPointError("fixed source geometry does not span three sensors")
    return leadfield


def geometry_manifest():
    return {"sensors": ["F3", "Fz", "F4"], "sensor_coordinates_mm": _ELECTRODES_MM.tolist(),
            "reference": "average of approximate left/right mastoid coordinates",
            "reference_coordinates_mm": _REFERENCE_MM.tolist(),
            "sources": ["early_visual_left", "early_visual_right", "shape_left", "shape_right",
                        "orientation_left", "orientation_right"],
            "source_coordinates_mm": _SOURCE_MM.tolist(), "source_orientation": "radial",
            "conductivity_S_per_m": CONDUCTIVITY_S_PER_M, "leadfield_units": "V/(A m)",
            "limitations": ["canonical coordinates", "homogeneous conductor",
                            "assumed linked-mastoid reference", "not an individualized head model"]}
