import math
import unittest

import numpy as np

from dec_engine import gizmo


class GizmoSettingsTests(unittest.TestCase):
    def test_defaults_are_translate_global_center(self):
        settings = gizmo.GizmoSettings()

        self.assertEqual(settings.mode, gizmo.TRANSLATE)
        self.assertEqual(settings.space, gizmo.GLOBAL)
        self.assertEqual(settings.pivot, gizmo.PIVOT_CENTER)
        self.assertAlmostEqual(settings.rotate_step_radians, math.radians(15.0))

    def test_unknown_mode_space_and_pivot_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown gizmo mode"):
            gizmo.GizmoSettings(mode="shear")
        with self.assertRaisesRegex(ValueError, "Unknown gizmo space"):
            gizmo.GizmoSettings(space="view")
        with self.assertRaisesRegex(ValueError, "Unknown gizmo pivot"):
            gizmo.GizmoSettings(pivot="median")

    def test_non_positive_steps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "translate_step"):
            gizmo.GizmoSettings(translate_step=0.0)
        with self.assertRaisesRegex(ValueError, "rotate_step_degrees"):
            gizmo.GizmoSettings(rotate_step_degrees=-5.0)
        with self.assertRaisesRegex(ValueError, "scale_step"):
            gizmo.GizmoSettings(scale_step=0.0)

    def test_non_finite_cursor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cursor"):
            gizmo.GizmoSettings(cursor=(0.0, float("nan"), 0.0))


class RotationMathTests(unittest.TestCase):
    def test_identity_matrix_maps_axes_to_canonical_basis(self):
        matrix = gizmo.rotation_matrix((0.0, 0.0, 0.0))

        np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)

    def test_rotation_about_z_maps_x_to_y(self):
        matrix = gizmo.rotation_matrix((0.0, 0.0, math.pi / 2.0))

        np.testing.assert_allclose(matrix[:, 0], (0.0, 1.0, 0.0), atol=1e-9)

    def test_euler_round_trip_recovers_input_rotation(self):
        rotation = (0.3, -0.4, 1.1)

        recovered = gizmo.euler_from_matrix(gizmo.rotation_matrix(rotation))

        np.testing.assert_allclose(
            gizmo.rotation_matrix(recovered),
            gizmo.rotation_matrix(rotation),
            atol=1e-9,
        )

    def test_euler_from_matrix_rejects_non_orthonormal_input(self):
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            gizmo.euler_from_matrix(np.diag((2.0, 1.0, 1.0)))

    def test_euler_from_matrix_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "3x3"):
            gizmo.euler_from_matrix(np.eye(4))


class AxisVectorTests(unittest.TestCase):
    def test_global_axes_are_canonical(self):
        axes = gizmo.axis_vectors((0.2, 0.5, -0.1), gizmo.GLOBAL)

        np.testing.assert_allclose(axes[0], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(axes[1], (0.0, 1.0, 0.0))
        np.testing.assert_allclose(axes[2], (0.0, 0.0, 1.0))

    def test_local_axes_follow_rotation(self):
        axes = gizmo.axis_vectors((0.0, 0.0, math.pi / 2.0), gizmo.LOCAL)

        np.testing.assert_allclose(axes[0], (0.0, 1.0, 0.0), atol=1e-9)
        np.testing.assert_allclose(axes[2], (0.0, 0.0, 1.0), atol=1e-9)

    def test_unknown_space_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown gizmo space"):
            gizmo.axis_vectors((0.0, 0.0, 0.0), "view")


class SnappingTests(unittest.TestCase):
    def test_snap_value_rounds_to_step(self):
        self.assertAlmostEqual(gizmo.snap_value(0.31, 0.25), 0.25)
        self.assertAlmostEqual(gizmo.snap_value(0.39, 0.25), 0.5)
        self.assertAlmostEqual(gizmo.snap_value(-0.31, 0.25), -0.25)

    def test_snap_value_rejects_bad_step(self):
        with self.assertRaisesRegex(ValueError, "step"):
            gizmo.snap_value(1.0, 0.0)

    def test_snap_angle_rounds_degrees(self):
        self.assertAlmostEqual(
            gizmo.snap_angle(math.radians(17.0), 15.0),
            math.radians(15.0),
        )
        self.assertAlmostEqual(
            gizmo.snap_angle(math.radians(23.0), 15.0),
            math.radians(30.0),
        )


class TransformApplicationTests(unittest.TestCase):
    def setUp(self):
        self.settings = gizmo.GizmoSettings()

    def test_translate_without_snapping_uses_raw_distance(self):
        result = gizmo.translate_position((1.0, 2.0, 3.0), (0.0, 1.0, 0.0), 0.37, None)

        np.testing.assert_allclose(result, (1.0, 2.37, 3.0))

    def test_translate_with_snapping_rounds_distance(self):
        result = gizmo.translate_position(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.37, self.settings
        )

        np.testing.assert_allclose(result, (0.25, 0.0, 0.0))

    def test_translate_rejects_wrong_shapes(self):
        with self.assertRaisesRegex(ValueError, "3D vectors"):
            gizmo.translate_position((0.0, 0.0), (1.0, 0.0, 0.0), 1.0)

    def test_rotate_about_z_by_ninety_degrees(self):
        rotation = gizmo.rotate_rotation(
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 2.0, None
        )

        np.testing.assert_allclose(
            gizmo.rotation_matrix(rotation),
            gizmo.rotation_matrix((0.0, 0.0, math.pi / 2.0)),
            atol=1e-9,
        )

    def test_rotate_with_snapping_rounds_to_step(self):
        rotation = gizmo.rotate_rotation(
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.radians(17.0), self.settings
        )

        np.testing.assert_allclose(
            gizmo.rotation_matrix(rotation),
            gizmo.rotation_matrix((0.0, 0.0, math.radians(15.0))),
            atol=1e-9,
        )

    def test_rotate_rejects_zero_axis(self):
        with self.assertRaisesRegex(ValueError, "zero-length"):
            gizmo.rotate_rotation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)

    def test_scale_with_snapping_rounds_result(self):
        result = gizmo.scale_axes((1.0, 1.0, 1.0), 0, 1.37, self.settings)

        self.assertAlmostEqual(result[0], 1.4)
        self.assertAlmostEqual(result[1], 1.0)
        self.assertAlmostEqual(result[2], 1.0)

    def test_scale_floors_at_positive_minimum(self):
        result = gizmo.scale_axes((1.0, 1.0, 1.0), 1, -3.0, None)

        self.assertGreater(result[1], 0.0)

    def test_scale_rejects_bad_axis(self):
        with self.assertRaisesRegex(ValueError, "axis_index"):
            gizmo.scale_axes((1.0, 1.0, 1.0), 3, 1.0)


class PivotTests(unittest.TestCase):
    def test_center_pivot_is_mean_of_selection(self):
        settings = gizmo.GizmoSettings(pivot=gizmo.PIVOT_CENTER)

        pivot = gizmo.pivot_point(((0.0, 0.0, 0.0), (2.0, 4.0, 6.0)), settings)

        np.testing.assert_allclose(pivot, (1.0, 2.0, 3.0))

    def test_cursor_pivot_uses_settings_cursor(self):
        settings = gizmo.GizmoSettings(
            pivot=gizmo.PIVOT_CURSOR, cursor=(5.0, -1.0, 2.0)
        )

        pivot = gizmo.pivot_point(((0.0, 0.0, 0.0),), settings)

        np.testing.assert_allclose(pivot, (5.0, -1.0, 2.0))

    def test_individual_pivot_returns_each_position(self):
        settings = gizmo.GizmoSettings(pivot=gizmo.PIVOT_INDIVIDUAL)

        pivots = gizmo.pivot_point(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), settings)

        self.assertEqual(len(pivots), 2)
        np.testing.assert_allclose(pivots[1], (1.0, 1.0, 1.0))

    def test_empty_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one position"):
            gizmo.pivot_point((), gizmo.GizmoSettings())


class MeasurementTests(unittest.TestCase):
    def test_translate_readout_has_axis_and_units(self):
        text = gizmo.measurement(gizmo.TRANSLATE, 0.25, "x")

        self.assertIn("+0.250", text)
        self.assertIn("X", text)

    def test_rotate_readout_uses_degrees(self):
        text = gizmo.measurement(gizmo.ROTATE, math.radians(15.0), "z")

        self.assertIn("15.00", text)
        self.assertIn("°", text)

    def test_scale_readout_uses_multiplier(self):
        text = gizmo.measurement(gizmo.SCALE, 1.1)

        self.assertIn("×1.100", text)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown gizmo mode"):
            gizmo.measurement("shear", 1.0)


class VertexSnappingTests(unittest.TestCase):
    def setUp(self):
        self.vertices = ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-1.0, 0.5, 0.0))

    def test_full_vertex_snapping_inside_threshold(self):
        result = gizmo.snap_to_vertices((0.9, 2.1, 2.95), self.vertices, threshold=0.2)

        np.testing.assert_allclose(result, (1.0, 2.0, 3.0))

    def test_full_vertex_snapping_outside_threshold_keeps_point(self):
        result = gizmo.snap_to_vertices((4.0, 4.0, 4.0), self.vertices, threshold=0.2)

        np.testing.assert_allclose(result, (4.0, 4.0, 4.0))

    def test_axis_snapping_changes_only_that_axis(self):
        result = gizmo.snap_to_vertices(
            (0.1, 0.55, 0.1), self.vertices, axis_index=1, threshold=0.3
        )

        np.testing.assert_allclose(result, (0.1, 0.5, 0.1))

    def test_empty_vertices_return_original_point(self):
        result = gizmo.snap_to_vertices((1.0, 2.0, 3.0), (), threshold=1.0)

        np.testing.assert_allclose(result, (1.0, 2.0, 3.0))

    def test_invalid_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "axis_index"):
            gizmo.snap_to_vertices((0.0, 0.0, 0.0), self.vertices, axis_index=5)


class GizmoGeometryTests(unittest.TestCase):
    def test_translate_geometry_has_three_coloured_axes(self):
        segments = gizmo.gizmo_line_segments(gizmo.TRANSLATE, size=1.0)

        self.assertEqual(len(segments), 3)
        for index, (points, color) in enumerate(segments):
            self.assertEqual(len(points), 2)
            self.assertEqual(color, gizmo.AXIS_COLORS[index])
            np.testing.assert_allclose(points[1][index], 1.0)

    def test_scale_geometry_adds_handle_boxes(self):
        segments = gizmo.gizmo_line_segments(gizmo.SCALE, size=1.0)

        self.assertEqual(len(segments), 6)

    def test_rotate_geometry_is_closed_circle_per_axis(self):
        segments = gizmo.gizmo_line_segments(gizmo.ROTATE, size=1.0)

        self.assertEqual(len(segments), 3 * 32)
        for index in range(3):
            first = segments[index * 32][0][0]
            last = segments[index * 32 + 31][0][1]
            np.testing.assert_allclose(first, last, atol=1e-9)

    def test_unknown_mode_and_bad_size_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown gizmo mode"):
            gizmo.gizmo_line_segments("shear")
        with self.assertRaisesRegex(ValueError, "size"):
            gizmo.gizmo_line_segments(gizmo.TRANSLATE, size=0.0)


if __name__ == "__main__":
    unittest.main()
