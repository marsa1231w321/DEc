import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dec_engine.commands import (
    AddNodeCommand,
    CommandStack,
    RemoveNodeCommand,
    SetNodePropertyCommand,
    SetScenePropertyCommand,
    build_scene_delta,
)
from dec_engine.project import load_project, save_project
from dec_engine.scene import MeshInstance, Scene, TransformNode


def make_mesh():
    return MeshInstance(
        name="Mesh",
        vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32),
        normals=np.asarray(((0, 0, 1),) * 3, dtype=np.float32),
        uvs=np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float32),
        indices=np.asarray((0, 1, 2), dtype=np.int32),
        color=(1.0, 1.0, 1.0),
    )


class CommandTests(unittest.TestCase):
    def test_command_stack_undoes_and_redoes_node_and_scene_properties(self):
        scene = Scene()
        mesh = make_mesh()
        scene.add_mesh(mesh)
        stack = CommandStack(scene)

        with stack.transaction("Transform and logic") as transaction:
            transaction.execute(
                SetNodePropertyCommand(
                    mesh.id, "position", mesh.position, (2.0, 3.0, 4.0)
                )
            )
            transaction.execute(
                SetScenePropertyCommand(
                    "logic_actions",
                    [],
                    [{"trigger": "on_start", "action": "set_position"}],
                )
            )

        self.assertTrue(stack.can_undo)
        self.assertEqual(mesh.position, (2.0, 3.0, 4.0))
        self.assertEqual(len(scene.logic_actions), 1)

        self.assertTrue(stack.undo())
        self.assertEqual(mesh.position, (0.0, 0.0, 0.0))
        self.assertEqual(scene.logic_actions, [])
        self.assertTrue(stack.redo())
        self.assertEqual(mesh.position, (2.0, 3.0, 4.0))
        self.assertEqual(len(scene.logic_actions), 1)

    def test_add_and_remove_node_commands_preserve_scene_graph(self):
        scene = Scene()
        stack = CommandStack(scene)
        node = TransformNode(name="Parent")
        stack.execute(AddNodeCommand(node))
        self.assertIsNotNone(scene.graph.get_node(node.id))

        stack.undo()
        self.assertIsNone(scene.graph.get_node(node.id))
        stack.redo()
        self.assertIsNotNone(scene.graph.get_node(node.id))

        stack.execute(RemoveNodeCommand(node.id))
        self.assertIsNone(scene.graph.get_node(node.id))
        stack.undo()
        restored = scene.graph.get_node(node.id)
        if restored is None:
            self.fail("Undo did not restore the removed node.")
        self.assertEqual(restored.name, "Parent")

    def test_failed_transaction_rolls_back_without_recording_history(self):
        scene = Scene()
        mesh = make_mesh()
        scene.add_mesh(mesh)
        stack = CommandStack(scene)

        with self.assertRaisesRegex(RuntimeError, "abort"):
            with stack.transaction("Rollback") as transaction:
                transaction.execute(
                    SetNodePropertyCommand(
                        mesh.id, "position", mesh.position, (4.0, 0.0, 0.0)
                    )
                )
                raise RuntimeError("abort")

        self.assertEqual(mesh.position, (0.0, 0.0, 0.0))
        self.assertFalse(stack.can_undo)

    def test_invalid_persisted_history_is_rejected(self):
        for history in (
            {"undo": [{"type": "unknown"}], "redo": []},
            [],
        ):
            with self.subTest(history=history):
                scene = Scene()
                with self.assertRaisesRegex(ValueError, "Invalid command history"):
                    CommandStack(scene, history=history)

    def test_command_history_syncs_after_record_and_clear(self):
        scene = Scene()
        stack = CommandStack(scene)
        command = SetScenePropertyCommand("ambient_intensity", 1.0, 0.5)
        command.do(scene)

        self.assertTrue(stack.record_applied(command))
        self.assertEqual(scene.command_history, stack.to_record())

        stack.clear()
        self.assertEqual(scene.command_history, {"undo": [], "redo": []})

    def test_command_over_memory_limit_is_not_kept(self):
        scene = Scene()
        stack = CommandStack(scene, max_memory_bytes=8)
        command = SetScenePropertyCommand("ambient_intensity", 1.0, 0.5)
        command.do(scene)

        self.assertFalse(stack.record_applied(command))
        self.assertFalse(stack.can_undo)
        self.assertEqual(scene.command_history, {"undo": [], "redo": []})

    def test_persisted_history_rejects_object_arrays(self):
        scene = Scene()
        scene.command_history = {
            "undo": [
                {
                    "type": "set_scene_property",
                    "property": "ambient_intensity",
                    "before": {"$array": "", "dtype": "O", "shape": [1]},
                    "after": 0.5,
                }
            ],
            "redo": [],
        }

        with self.assertRaisesRegex(ValueError, "Invalid command history"):
            CommandStack.from_scene(scene)

    def test_undo_history_survives_project_save_and_load(self):
        scene = Scene()
        mesh = make_mesh()
        scene.add_mesh(mesh)
        before = pickle.dumps(scene.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
        mesh.position = (3.0, 2.0, 1.0)
        scene.logic_actions.append({"trigger": "on_start", "action": "set_position"})
        child = TransformNode(name="Added child", parent_id=mesh.id)
        scene.add_node(child, parent=mesh)
        after = pickle.dumps(scene.__dict__, protocol=pickle.HIGHEST_PROTOCOL)

        stack = CommandStack(scene)
        stack.record_applied(build_scene_delta(before, after, "Combined edit"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.DE"
            save_project(path, scene)
            restored_scene = load_project(path)

        restored_stack = CommandStack.from_scene(restored_scene)
        self.assertTrue(restored_stack.can_undo)
        self.assertEqual(restored_scene.objects[0].position, (3.0, 2.0, 1.0))
        self.assertEqual(len(restored_scene.logic_actions), 1)
        self.assertIsNotNone(restored_scene.graph.get_node(child.id))

        self.assertTrue(restored_stack.undo())
        restored_mesh = restored_scene.graph.get_node(mesh.id)
        if not isinstance(restored_mesh, MeshInstance):
            self.fail("The mesh disappeared while undoing the saved command.")
        self.assertEqual(restored_mesh.position, (0.0, 0.0, 0.0))
        self.assertEqual(restored_scene.logic_actions, [])
        self.assertIsNone(restored_scene.graph.get_node(child.id))

        self.assertTrue(restored_stack.redo())
        self.assertEqual(restored_mesh.position, (3.0, 2.0, 1.0))
        self.assertEqual(len(restored_scene.logic_actions), 1)
        self.assertIsNotNone(restored_scene.graph.get_node(child.id))

    def test_mesh_node_commands_round_trip_array_payloads(self):
        scene = Scene()
        mesh = make_mesh()
        stack = CommandStack(scene)
        stack.execute(AddNodeCommand(mesh))
        stack.execute(RemoveNodeCommand(mesh.id))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh-history.DE"
            save_project(path, scene)
            restored_scene = load_project(path)

        restored_stack = CommandStack.from_scene(restored_scene)
        self.assertTrue(restored_stack.undo())
        restored_mesh = restored_scene.graph.get_node(mesh.id)
        if not isinstance(restored_mesh, MeshInstance):
            self.fail("Undo did not restore the serialized mesh command.")
        np.testing.assert_array_equal(restored_mesh.vertices, mesh.vertices)

        self.assertTrue(restored_stack.undo())
        self.assertIsNone(restored_scene.graph.get_node(mesh.id))

    def test_scene_delta_reparents_children_before_removing_parent(self):
        scene = Scene()
        parent = TransformNode(name="Parent")
        child = TransformNode(name="Child")
        scene.add_node(parent)
        scene.add_node(child, parent=parent)
        before = pickle.dumps(scene.__dict__, protocol=pickle.HIGHEST_PROTOCOL)

        scene.reparent_node(child, None)
        scene.remove_node(parent)
        after = pickle.dumps(scene.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
        stack = CommandStack(scene)
        stack.record_applied(build_scene_delta(before, after, "Delete parent"))

        self.assertTrue(stack.undo())
        restored_child = scene.graph.get_node(child.id)
        if restored_child is None:
            self.fail("Undo did not restore the child.")
        self.assertEqual(restored_child.parent_id, parent.id)
        self.assertIsNotNone(scene.graph.get_node(parent.id))

        self.assertTrue(stack.redo())
        self.assertIsNone(scene.graph.get_node(parent.id))
        remaining_child = scene.graph.get_node(child.id)
        if remaining_child is None:
            self.fail("Redo unexpectedly removed the child.")
        self.assertIsNone(remaining_child.parent_id)


if __name__ == "__main__":
    unittest.main()
