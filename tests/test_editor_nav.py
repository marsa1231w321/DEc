import math
import unittest

import numpy as np

from dec_engine import editor_nav
from dec_engine.scene import MeshInstance, Scene


def make_mesh(name, position):
    return MeshInstance(
        name=name,
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32
        ),
        normals=np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float32),
        uvs=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        indices=np.asarray((0, 1, 2), dtype=np.int32),
        color=(1.0, 1.0, 1.0),
        position=position,
    )


class AdaptiveGridTests(unittest.TestCase):
    def test_step_keeps_ratio_between_ten_and_twenty(self):
        for distance in (5.0, 10.0, 80.0, 160.0):
            with self.subTest(distance=distance):
                step = editor_nav.adaptive_grid_step(distance)
                ratio = distance / step
                self.assertGreaterEqual(ratio, 10.0 - 1e-9)
                self.assertLess(ratio, 20.0 + 1e-9)

    def test_zero_distance_returns_minimum_step(self):
        self.assertEqual(editor_nav.adaptive_grid_step(0.0), 0.1)

    def test_clamps_to_limits(self):
        self.assertEqual(
            editor_nav.adaptive_grid_step(1e9, min_step=0.5, max_step=8.0), 8.0
        )
        self.assertEqual(
            editor_nav.adaptive_grid_step(1e-9, min_step=0.5, max_step=8.0), 0.5
        )

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "base"):
            editor_nav.adaptive_grid_step(10.0, base=0.0)
        with self.assertRaisesRegex(ValueError, "view_distance"):
            editor_nav.adaptive_grid_step(-1.0)
        with self.assertRaisesRegex(ValueError, "min_step"):
            editor_nav.adaptive_grid_step(10.0, min_step=5.0, max_step=1.0)


class GridLabelTests(unittest.TestCase):
    def test_labels_include_both_ends(self):
        labels = editor_nav.grid_label_positions(0.0, 10.0, 2.0)

        self.assertEqual(labels, (0.0, 2.0, 4.0, 6.0, 8.0, 10.0))

    def test_labels_align_to_step_from_offset(self):
        labels = editor_nav.grid_label_positions(1.0, 9.0, 2.0)

        self.assertEqual(labels, (2.0, 4.0, 6.0, 8.0))

    def test_labels_respect_maximum(self):
        with self.assertRaisesRegex(ValueError, "Too many grid labels"):
            editor_nav.grid_label_positions(0.0, 100.0, 1.0, max_labels=8)

    def test_reversed_range_is_supported(self):
        labels = editor_nav.grid_label_positions(10.0, 0.0, 5.0)

        self.assertEqual(labels, (0.0, 5.0, 10.0))


class SceneBoundsTests(unittest.TestCase):
    def test_bounds_cover_two_meshes(self):
        scene = Scene()
        scene.add_mesh(make_mesh("A", (0.0, 0.0, 0.0)))
        scene.add_mesh(make_mesh("B", (2.0, 0.0, 0.0)))

        center, radius = editor_nav.scene_bounds(scene)

        np.testing.assert_allclose(center, (1.5, 0.5, 0.0), atol=1e-6)
        self.assertAlmostEqual(radius, math.sqrt(2.5), places=6)

    def test_empty_scene_returns_origin(self):
        center, radius = editor_nav.scene_bounds(Scene())

        self.assertEqual(center, (0.0, 0.0, 0.0))
        self.assertEqual(radius, 0.0)

    def test_frame_distance_respects_fov_and_margin(self):
        distance = editor_nav.frame_distance(1.0, math.pi / 2.0, margin=1.0)

        self.assertAlmostEqual(distance, math.sqrt(2.0), places=6)

    def test_frame_distance_rejects_bad_input(self):
        with self.assertRaisesRegex(ValueError, "radius"):
            editor_nav.frame_distance(-1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "fov_radians"):
            editor_nav.frame_distance(1.0, 0.0)


class RectangleTests(unittest.TestCase):
    def test_normalized_rect_orders_corners(self):
        rect = editor_nav.normalized_rect((10.0, 20.0), (0.0, 5.0))

        self.assertEqual(rect, (0.0, 5.0, 10.0, 20.0))

    def test_point_in_rect_matches_bounds(self):
        rect = (0.0, 0.0, 10.0, 10.0)

        self.assertTrue(editor_nav.point_in_rect((5.0, 5.0), rect))
        self.assertFalse(editor_nav.point_in_rect((11.0, 5.0), rect))

    def test_rect_rejects_non_2d_points(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            editor_nav.normalized_rect((0.0,), (0.0, 0.0))


class ProjectionTests(unittest.TestCase):
    def test_identity_projection_maps_to_viewport_center(self):
        projected = editor_nav.project_points(
            ((0.0, 0.0, 0.0),), np.eye(4), (100.0, 100.0)
        )

        self.assertAlmostEqual(projected[0][0], 50.0)
        self.assertAlmostEqual(projected[0][1], 50.0)

    def test_points_behind_camera_become_none(self):
        view_projection = np.eye(4)
        view_projection[3, 3] = -1.0

        projected = editor_nav.project_points(
            ((0.0, 0.0, 0.0),), view_projection, (100.0, 100.0)
        )

        self.assertIsNone(projected[0])

    def test_projection_validates_inputs(self):
        with self.assertRaisesRegex(ValueError, "Nx3"):
            editor_nav.project_points(((0.0, 0.0),), np.eye(4), (10.0, 10.0))
        with self.assertRaisesRegex(ValueError, "4x4"):
            editor_nav.project_points(((0.0, 0.0, 0.0),), np.eye(3), (10.0, 10.0))


class SelectionTests(unittest.TestCase):
    def _scene(self):
        scene = Scene()
        first = make_mesh("A", (0.0, 0.0, 0.0))
        second = make_mesh("B", (5.0, 0.0, 0.0))
        scene.add_mesh(first)
        scene.add_mesh(second)
        return scene, first, second

    def test_rect_selects_only_visible_meshes(self):
        scene, first, _second = self._scene()

        selected = editor_nav.nodes_in_screen_rect(
            scene, np.eye(4), (40.0, 0.0, 110.0, 60.0), (100.0, 100.0)
        )

        self.assertEqual(selected, (first.id,))

    def test_empty_rect_selects_nothing(self):
        scene, _first, _second = self._scene()

        selected = editor_nav.nodes_in_screen_rect(
            scene, np.eye(4), (200.0, 200.0, 220.0, 220.0), (100.0, 100.0)
        )

        self.assertEqual(selected, ())

    def test_culling_keeps_objects_within_distance(self):
        positions = np.asarray(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)))
        radii = np.asarray((1.0, 1.0))

        kept = editor_nav.cull_world_objects(positions, radii, (0.0, 0.0, 0.0), 10.0)

        self.assertEqual(kept, (0,))

    def test_culling_validates_shapes(self):
        with self.assertRaisesRegex(ValueError, "radii"):
            editor_nav.cull_world_objects(
                ((0.0, 0.0, 0.0),), (1.0, 2.0), (0.0, 0.0, 0.0), 5.0
            )
        with self.assertRaisesRegex(ValueError, "max_distance"):
            editor_nav.cull_world_objects(
                ((0.0, 0.0, 0.0),), (1.0,), (0.0, 0.0, 0.0), 0.0
            )

    def test_zoom_towards_moves_camera_to_target(self):
        moved = editor_nav.zoom_towards((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), 0.5)

        np.testing.assert_allclose(moved, (0.0, 0.0, 5.0))

    def test_zoom_towards_rejects_bad_factor(self):
        with self.assertRaisesRegex(ValueError, "factor"):
            editor_nav.zoom_towards((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
