"""Headless math for editor transform gizmos.

The module is intentionally UI-free: axes, pivots, snapping and readouts are
plain functions so they can be unit-tested without GLFW/OpenGL or Tk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRANSLATE = "translate"
ROTATE = "rotate"
SCALE = "scale"
GIZMO_MODES = (TRANSLATE, ROTATE, SCALE)

GLOBAL = "global"
LOCAL = "local"
GIZMO_SPACES = (GLOBAL, LOCAL)

PIVOT_CENTER = "center"
PIVOT_INDIVIDUAL = "individual"
PIVOT_CURSOR = "cursor"
PIVOT_MODES = (PIVOT_CENTER, PIVOT_INDIVIDUAL, PIVOT_CURSOR)

AXIS_NAMES = ("x", "y", "z")
AXIS_COLORS = (
    (0.95, 0.25, 0.25),
    (0.35, 0.9, 0.35),
    (0.35, 0.55, 0.98),
)


def _positive_finite(value, label):
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number.")
    return number


@dataclass(frozen=True)
class GizmoSettings:
    """Active gizmo configuration shared by editor handlers and tests."""

    mode: str = TRANSLATE
    space: str = GLOBAL
    pivot: str = PIVOT_CENTER
    snap_translate: bool = True
    snap_rotate: bool = True
    snap_scale: bool = True
    translate_step: float = 0.25
    rotate_step_degrees: float = 15.0
    scale_step: float = 0.1
    cursor: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.mode not in GIZMO_MODES:
            raise ValueError(f"Unknown gizmo mode: {self.mode!r}")
        if self.space not in GIZMO_SPACES:
            raise ValueError(f"Unknown gizmo space: {self.space!r}")
        if self.pivot not in PIVOT_MODES:
            raise ValueError(f"Unknown gizmo pivot: {self.pivot!r}")
        _positive_finite(self.translate_step, "translate_step")
        _positive_finite(self.rotate_step_degrees, "rotate_step_degrees")
        _positive_finite(self.scale_step, "scale_step")
        cursor = np.asarray(self.cursor, dtype=np.float64)
        if cursor.shape != (3,) or not np.isfinite(cursor).all():
            raise ValueError("cursor must be a finite 3D vector.")

    @property
    def rotate_step_radians(self) -> float:
        return math.radians(float(self.rotate_step_degrees))


def rotation_matrix(rotation) -> np.ndarray:
    """Euler XYZ rotation matrix matching dec_engine.scene._rotation_matrix."""

    x, y, z = (float(value) for value in rotation)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def euler_from_matrix(matrix) -> tuple[float, float, float]:
    """Inverse of rotation_matrix for a proper orthonormal 3x3 matrix."""

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("rotation matrix must be orthonormal.")
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return (roll, pitch, yaw)


def axis_vectors(rotation, space: str = GLOBAL) -> tuple[np.ndarray, ...]:
    """Unit axis vectors for a gizmo in global or local space."""

    if space == GLOBAL:
        return tuple(np.eye(3, dtype=np.float64))
    if space != LOCAL:
        raise ValueError(f"Unknown gizmo space: {space!r}")
    basis = rotation_matrix(rotation)
    return tuple(basis[:, index].copy() for index in range(3))


def axis_angle_quaternion(axis, angle: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for an axis-angle rotation."""

    vector = np.asarray(axis, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("axis must be a 3D vector.")
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("axis must not be zero-length.")
    unit = vector / length
    half = float(angle) * 0.5
    sin_half = math.sin(half)
    return np.array(
        [math.cos(half), unit[0] * sin_half, unit[1] * sin_half, unit[2] * sin_half]
    )


def quaternion_to_matrix(quaternion) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def snap_value(value: float, step: float) -> float:
    """Round a scalar to the nearest multiple of a positive step."""

    if step <= 0.0 or not math.isfinite(step):
        raise ValueError("step must be a positive finite number.")
    return round(float(value) / step) * step


def snap_angle(angle: float, step_degrees: float) -> float:
    """Round an angle in radians to the nearest step expressed in degrees."""

    step = math.radians(_positive_finite(step_degrees, "step_degrees"))
    return snap_value(float(angle), step)


def translate_position(position, axis, distance, settings: GizmoSettings | None = None):
    """Move a position along an axis with optional stepped snapping."""

    base = np.asarray(position, dtype=np.float64)
    direction = np.asarray(axis, dtype=np.float64)
    if base.shape != (3,) or direction.shape != (3,):
        raise ValueError("position and axis must be 3D vectors.")
    amount = float(distance)
    if settings is not None and settings.snap_translate:
        amount = snap_value(amount, settings.translate_step)
    return base + direction * amount


def rotate_rotation(rotation, axis, angle, settings: GizmoSettings | None = None):
    """Compose an axis-angle rotation with an existing Euler XYZ rotation."""

    amount = float(angle)
    if settings is not None and settings.snap_rotate:
        amount = snap_angle(amount, settings.rotate_step_degrees)
    current = rotation_matrix(rotation)
    delta = quaternion_to_matrix(axis_angle_quaternion(axis, amount))
    return euler_from_matrix(delta @ current)


def scale_axes(scale_xyz, axis_index: int, factor: float, settings=None):
    """Scale one axis by a factor with snapping and a positive floor."""

    values = np.asarray(scale_xyz, dtype=np.float64).copy()
    if values.shape != (3,):
        raise ValueError("scale_xyz must be a 3D vector.")
    if axis_index not in (0, 1, 2):
        raise ValueError("axis_index must be 0, 1 or 2.")
    result = float(values[axis_index]) * float(factor)
    if settings is not None and settings.snap_scale:
        result = snap_value(result, settings.scale_step)
    values[axis_index] = max(result, 1e-4)
    return tuple(float(value) for value in values)


def pivot_point(positions, settings: GizmoSettings):
    """Return the pivot for a selection according to the pivot mode."""

    points = [np.asarray(position, dtype=np.float64) for position in positions]
    if not points:
        raise ValueError("pivot requires at least one position.")
    for point in points:
        if point.shape != (3,):
            raise ValueError("positions must be 3D vectors.")
    if settings.pivot == PIVOT_CURSOR:
        return np.asarray(settings.cursor, dtype=np.float64)
    if settings.pivot == PIVOT_INDIVIDUAL:
        return tuple(point.copy() for point in points)
    return np.mean(points, axis=0)


def measurement(mode: str, value: float, axis: str | None = None) -> str:
    """Human-readable readout for the active gizmo drag."""

    suffix = f" по оси {str(axis).upper()}" if axis else ""
    if mode == TRANSLATE:
        return f"Смещение: {float(value):+.3f}{suffix}"
    if mode == ROTATE:
        return f"Поворот: {math.degrees(float(value)):+.2f}°{suffix}"
    if mode == SCALE:
        return f"Масштаб: ×{float(value):.3f}{suffix}"
    raise ValueError(f"Unknown gizmo mode: {mode!r}")


def snap_to_vertices(point, vertices, axis_index=None, threshold=0.25):
    """Snap a point to the nearest vertex coordinate or to a full vertex."""

    target = np.asarray(point, dtype=np.float64)
    candidates = np.asarray(vertices, dtype=np.float64)
    if target.shape != (3,):
        raise ValueError("point must be 3D and vertices must be Nx3.")
    if candidates.size == 0:
        return target.copy()
    if candidates.ndim != 2 or candidates.shape[1:] != (3,):
        raise ValueError("point must be 3D and vertices must be Nx3.")
    if axis_index is not None:
        if axis_index not in (0, 1, 2):
            raise ValueError("axis_index must be 0, 1 or 2.")
        nearest = candidates[
            np.argmin(np.abs(candidates[:, axis_index] - target[axis_index]))
        ]
        if abs(float(nearest[axis_index]) - float(target[axis_index])) <= threshold:
            snapped = target.copy()
            snapped[axis_index] = nearest[axis_index]
            return snapped
        return target.copy()
    distances = np.linalg.norm(candidates - target, axis=1)
    index = int(np.argmin(distances))
    if float(distances[index]) <= threshold:
        return candidates[index].copy()
    return target.copy()


def _line_segment(start, end):
    return (
        tuple(float(value) for value in start),
        tuple(float(value) for value in end),
    )


def gizmo_line_segments(mode: str = TRANSLATE, size: float = 0.75):
    """Coloured line segments for a 3D gizmo, usable without GL context."""

    length = _positive_finite(size, "size")
    segments = []
    if mode == TRANSLATE:
        for index, color in enumerate(AXIS_COLORS):
            start = np.zeros(3)
            end = np.zeros(3)
            end[index] = length
            segments.append((_line_segment(start, end), color))
    elif mode == SCALE:
        for index, color in enumerate(AXIS_COLORS):
            start = np.zeros(3)
            end = np.zeros(3)
            end[index] = length
            box = np.zeros(3)
            box[index] = length
            segments.append((_line_segment(start, end), color))
            segments.append(
                (
                    _line_segment(box - 0.06, box + 0.06),
                    color,
                )
            )
    elif mode == ROTATE:
        angles = np.linspace(0.0, 2.0 * math.pi, 33)
        for index, color in enumerate(AXIS_COLORS):
            for start_angle, end_angle in zip(angles, angles[1:]):
                first = np.zeros(3)
                second = np.zeros(3)
                first[(index + 1) % 3] = math.cos(start_angle) * length
                first[(index + 2) % 3] = math.sin(start_angle) * length
                second[(index + 1) % 3] = math.cos(end_angle) * length
                second[(index + 2) % 3] = math.sin(end_angle) * length
                segments.append((_line_segment(first, second), color))
    else:
        raise ValueError(f"Unknown gizmo mode: {mode!r}")
    return tuple(segments)
