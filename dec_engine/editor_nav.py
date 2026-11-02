"""Headless helpers for editor grid, framing and rectangle selection.

Pure numpy functions so adaptivity, framing and selection can be tested
without Tk, GLFW or OpenGL contexts.
"""

from __future__ import annotations

import math

import numpy as np


def _positive(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number.")
    return number


def adaptive_grid_step(
    view_distance: float,
    base: float = 1.0,
    *,
    min_step: float = 0.1,
    max_step: float = 1000.0,
) -> float:
    """Pick a power-of-two grid step so view_distance/step stays in [10, 20)."""

    distance = float(view_distance)
    base_value = _positive(base, "base")
    minimum = _positive(min_step, "min_step")
    maximum = _positive(max_step, "max_step")
    if minimum > maximum:
        raise ValueError("min_step must not exceed max_step.")
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("view_distance must be a finite non-negative number.")
    if distance == 0.0:
        return float(minimum)
    exponent = math.floor(math.log2(max(distance, base_value) / base_value / 10.0))
    step = base_value * 2.0**exponent
    return float(min(max(step, minimum), maximum))


def grid_label_positions(
    start: float,
    end: float,
    step: float,
    *,
    max_labels: int = 64,
) -> tuple[float, ...]:
    """World positions of grid labels between start and end inclusive."""

    first_input = float(start)
    last_input = float(end)
    step_value = _positive(step, "step")
    if not math.isfinite(first_input) or not math.isfinite(last_input):
        raise ValueError("start and end must be finite numbers.")
    if max_labels < 1:
        raise ValueError("max_labels must be positive.")
    low, high = sorted((first_input, last_input))
    first = math.ceil(low / step_value - 1e-9) * step_value
    if first > high + 1e-9:
        return ()
    count = int(math.floor((high - first) / step_value + 1e-9)) + 1
    if count > max_labels:
        raise ValueError(
            f"Too many grid labels ({count}); increase step or max_labels."
        )
    return tuple(first + index * step_value for index in range(count))


def scene_bounds(scene) -> tuple[tuple[float, float, float], float]:
    """World-space AABB center and radius for meshes and lights."""

    chunks = []
    for obj in scene.objects:
        matrix = np.asarray(scene.world_matrix(obj), dtype=np.float64)
        vertices = np.asarray(obj.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            continue
        homogeneous = np.concatenate(
            (vertices, np.ones((len(vertices), 1), dtype=np.float64)), axis=1
        )
        chunks.append((homogeneous @ matrix.T)[:, :3])
    for light in scene.lights:
        chunks.append(np.asarray([scene.world_position(light)], dtype=np.float64))
    if not chunks:
        return (0.0, 0.0, 0.0), 0.0
    points = np.concatenate(chunks, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.linalg.norm(maximum - center))
    return tuple(float(value) for value in center), radius


def frame_distance(radius: float, fov_radians: float, *, margin: float = 1.1) -> float:
    """Camera distance that fits a bounding sphere into the given FOV."""

    radius_value = float(radius)
    if not math.isfinite(radius_value) or radius_value < 0.0:
        raise ValueError("radius must be a finite non-negative number.")
    fov = _positive(fov_radians, "fov_radians")
    if fov >= math.pi:
        fov = math.pi - 1e-6
    margin_value = _positive(margin, "margin")
    return radius_value / math.sin(fov * 0.5) * margin_value


def normalized_rect(start, end) -> tuple[float, float, float, float]:
    """Order two screen-space corners into (x0, y0, x1, y1)."""

    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    if first.shape != (2,) or second.shape != (2,):
        raise ValueError("rectangle corners must be 2D points.")
    return (
        float(min(first[0], second[0])),
        float(min(first[1], second[1])),
        float(max(first[0], second[0])),
        float(max(first[1], second[1])),
    )


def point_in_rect(point, rect) -> bool:
    x, y = float(point[0]), float(point[1])
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def project_points(points, view_projection, viewport_size):
    """Project world points to screen pixels; behind-camera points become None."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("points must be an Nx3 array.")
    matrix = np.asarray(view_projection, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("view_projection must be 4x4.")
    width = _positive(viewport_size[0], "viewport width")
    height = _positive(viewport_size[1], "viewport height")
    homogeneous = np.concatenate(
        (values, np.ones((len(values), 1), dtype=np.float64)), axis=1
    )
    clip = homogeneous @ matrix.T
    projected = []
    for x, y, _z, w in clip:
        if w <= 1e-9:
            projected.append(None)
            continue
        ndc_x = x / w
        ndc_y = y / w
        projected.append(
            (
                (ndc_x + 1.0) * 0.5 * width,
                (1.0 - (ndc_y + 1.0) * 0.5) * height,
            )
        )
    return tuple(projected)


def nodes_in_screen_rect(scene, view_projection, rect, viewport_size):
    """IDs of meshes with at least one vertex inside the screen rectangle."""

    x0, y0, x1, y1 = normalized_rect((rect[0], rect[1]), (rect[2], rect[3]))
    ordered = (x0, y0, x1, y1)
    selected = []
    for obj in scene.objects:
        matrix = np.asarray(scene.world_matrix(obj), dtype=np.float64)
        vertices = np.asarray(obj.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            continue
        homogeneous = np.concatenate(
            (vertices, np.ones((len(vertices), 1), dtype=np.float64)), axis=1
        )
        world = (homogeneous @ matrix.T)[:, :3]
        projected = project_points(world, view_projection, viewport_size)
        if any(
            point is not None and point_in_rect(point, ordered) for point in projected
        ):
            selected.append(obj.id)
    return tuple(selected)


def cull_world_objects(positions, radii, camera_position, max_distance):
    """Indices of objects whose bounding sphere is within max_distance."""

    points = np.asarray(positions, dtype=np.float64)
    sphere_radii = np.asarray(radii, dtype=np.float64)
    camera = np.asarray(camera_position, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("positions must be an Nx3 array.")
    if sphere_radii.shape != (len(points),):
        raise ValueError("radii must match the number of positions.")
    if camera.shape != (3,):
        raise ValueError("camera_position must be a 3D point.")
    limit = _positive(max_distance, "max_distance")
    distances = np.linalg.norm(points - camera, axis=1)
    return tuple(
        int(index) for index in np.where(distances - sphere_radii <= limit)[0]
    )


def zoom_towards(camera_position, target, factor: float):
    """Move the camera towards a target; factor < 1 zooms in."""

    camera = np.asarray(camera_position, dtype=np.float64)
    focus = np.asarray(target, dtype=np.float64)
    if camera.shape != (3,) or focus.shape != (3,):
        raise ValueError("camera_position and target must be 3D points.")
    ratio = _positive(factor, "factor")
    return tuple(float(value) for value in focus + (camera - focus) * ratio)

