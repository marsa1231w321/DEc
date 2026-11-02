import unittest

import numpy as np

from dec_engine.scene import MeshInstance, Scene
from dec_engine.scene_observers import SceneEventConsumers
from dec_engine.validation import (
    BLOCKER,
    WARNING,
    ValidationIssue,
    blocking_issues,
    collect_validation_issues,
    has_blockers,
    validate_scene,
)


def make_mesh(name="Mesh", position=(0.0, 0.0, 0.0)):
    return MeshInstance(
        name=name,
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        ),
        normals=np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float32),
        uvs=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        indices=np.asarray((0, 1, 2), dtype=np.int32),
        color=(1.0, 1.0, 1.0),
        position=position,
    )


class ValidationSeverityTests(unittest.TestCase):
    def test_valid_scene_has_no_issues(self):
        scene = Scene()
        scene.add_mesh(make_mesh())

        self.assertEqual(collect_validation_issues(scene), [])
        self.assertEqual(validate_scene(scene), [])
        self.assertFalse(has_blockers(scene))

    def test_broken_vertices_are_blockers_with_object_context(self):
        scene = Scene()
        mesh = make_mesh()
        mesh.vertices = np.asarray(
            ((np.nan, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        )
        scene.add_mesh(mesh)

        issues = collect_validation_issues(scene)

        self.assertTrue(issues)
        blocker = issues[0]
        self.assertEqual(blocker.severity, BLOCKER)
        self.assertTrue(blocker.is_blocker)
        self.assertEqual(blocker.object_id, mesh.id)
        self.assertEqual(blocker.object_name, mesh.name)
        self.assertEqual(blocker.property, "vertices")
        self.assertTrue(has_blockers(scene))

    def test_duplicate_names_are_warnings_and_do_not_block(self):
        scene = Scene()
        scene.add_mesh(make_mesh("Same"))
        scene.add_mesh(make_mesh("Same", position=(2.0, 0.0, 0.0)))

        issues = collect_validation_issues(scene)

        self.assertTrue(issues)
        self.assertTrue(all(issue.severity == WARNING for issue in issues))
        self.assertFalse(has_blockers(scene))
        self.assertEqual(blocking_issues(scene), [])

    def test_color_range_is_a_warning(self):
        scene = Scene()
        mesh = make_mesh()
        mesh.color = (2.0, 1.0, 1.0)
        scene.add_mesh(mesh)

        issues = collect_validation_issues(scene)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, WARNING)
        self.assertEqual(issues[0].property, "color")

    def test_missing_logic_target_is_a_blocker(self):
        scene = Scene()
        scene.add_mesh(make_mesh("Target"))
        scene.logic_actions.append(
            {
                "trigger": "on_start",
                "target": "Missing",
                "action": "toggle_visibility",
            }
        )

        issues = collect_validation_issues(scene)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, BLOCKER)
        self.assertEqual(issues[0].property, "target")
        self.assertIn("целевой объект отсутствует", issues[0].message)
        self.assertEqual(validate_scene(scene), [issue.message for issue in issues])

    def test_validate_scene_preserves_legacy_message_list(self):
        scene = Scene()
        scene.add_mesh(make_mesh("Same"))
        scene.add_mesh(make_mesh("Same", position=(2.0, 0.0, 0.0)))

        self.assertEqual(
            validate_scene(scene),
            [issue.message for issue in collect_validation_issues(scene)],
        )
        self.assertTrue(
            all(isinstance(problem, str) for problem in validate_scene(scene))
        )


class ValidationIssueTypeTests(unittest.TestCase):
    def test_issue_defaults_to_blocker_and_is_frozen(self):
        issue = ValidationIssue("boom")

        self.assertEqual(issue.severity, BLOCKER)
        self.assertTrue(issue.is_blocker)
        with self.assertRaises(Exception):
            setattr(issue, "message", "changed")


class SceneEventConsumerValidationTests(unittest.TestCase):
    def test_consumers_cache_structured_issues_and_messages(self):
        scene = Scene()
        mesh = make_mesh()
        mesh.vertices = np.asarray(
            ((np.nan, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        )
        scene.add_mesh(mesh)
        consumers = SceneEventConsumers(scene)
        self.addCleanup(consumers.close)

        self.assertEqual(consumers.validation_issues[0].property, "vertices")
        self.assertEqual(
            consumers.validation_problems,
            tuple(issue.message for issue in consumers.validation_issues),
        )


if __name__ == "__main__":
    unittest.main()
