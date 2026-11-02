import unittest
from unittest.mock import patch

import numpy as np

from dec_engine.logic import LogicRuntime
from dec_engine.playtest import PlaytestController
from dec_engine.scene import MeshInstance, Scene, TransformNode
from dec_engine.validation import validate_scene


def make_mesh(name, position=(0.0, 0.0, 0.0)):
    return MeshInstance(
        name=name,
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
            dtype=np.float32,
        ),
        normals=np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float32),
        uvs=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        indices=np.asarray((0, 1, 2), dtype=np.int32),
        color=(1.0, 1.0, 1.0),
        position=position,
    )


class HeadlessRuntimeTests(unittest.TestCase):
    def test_proximity_logic_uses_world_position_and_fires_on_entry(self):
        scene = Scene()
        parent = scene.add_node(TransformNode(name="Parent", position=(5.0, 0.0, 0.0)))
        target = make_mesh("Target", position=(1.0, 0.0, 0.0))
        scene.add_mesh(target)
        scene.reparent_node(target, parent)
        scene.logic_actions.append(
            {
                "trigger": "on_player_near",
                "target": "Target",
                "radius": 1.0,
                "action": "toggle_visibility",
            }
        )
        runtime = LogicRuntime()

        with patch("dec_engine.logic.glfw.get_key", return_value=0):
            runtime.update(scene, 1 / 60, None, player_position=(6.0, 0.0, 0.0))
            self.assertFalse(target.visible)

            runtime.update(scene, 1 / 60, None, player_position=(6.0, 0.0, 0.0))
            self.assertFalse(target.visible)

            runtime.update(scene, 1 / 60, None, player_position=(10.0, 0.0, 0.0))
            runtime.update(scene, 1 / 60, None, player_position=(6.0, 0.0, 0.0))
        self.assertTrue(target.visible)

    def test_playtest_collision_uses_parent_world_transform(self):
        scene = Scene()
        parent = scene.add_node(TransformNode(name="Parent", position=(5.0, 0.0, 0.0)))
        blocker = make_mesh("Blocker")
        scene.add_mesh(blocker)
        scene.reparent_node(blocker, parent)
        controller = PlaytestController()

        self.assertTrue(controller._collides(scene, 5.5, 0.1, 0.5))
        self.assertFalse(controller._collides(scene, 2.0, 0.1, 0.5))

    def test_scene_validation_runs_without_graphics_context(self):
        scene = Scene()
        target = make_mesh("Target")
        scene.add_mesh(target)
        scene.logic_actions.append(
            {
                "trigger": "on_player_near",
                "target": target.name,
                "radius": 2.0,
                "action": "toggle_visibility",
            }
        )

        self.assertEqual(validate_scene(scene), [])

        scene.logic_actions[0]["target"] = "Missing"
        self.assertTrue(
            any(
                "целевой объект отсутствует" in problem
                for problem in validate_scene(scene)
            )
        )


if __name__ == "__main__":
    unittest.main()
