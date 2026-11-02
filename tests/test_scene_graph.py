import tempfile
import unittest
import zipfile
import json
from pathlib import Path

import numpy as np

from dec_engine.project import load_project, save_project
from dec_engine.playtest import PlaytestController
from dec_engine.scene import Light, MeshInstance, Scene, TransformNode


def make_mesh(name, position=(0.0, 0.0, 0.0)):
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


class SceneGraphTests(unittest.TestCase):
    def test_world_transform_composes_parent_and_child(self):
        scene = Scene()
        parent = scene.add_node(
            TransformNode(
                name="Parent",
                position=(2.0, 0.0, 0.0),
                rotation=(0.0, np.pi / 2.0, 0.0),
            )
        )
        child = make_mesh("Child", position=(0.0, 0.0, -1.0))
        scene.add_mesh(child)
        scene.reparent_node(child, parent)

        self.assertTrue(
            np.allclose(scene.world_position(child), (1.0, 0.0, 0.0), atol=1e-6)
        )

    def test_set_world_position_converts_through_parent(self):
        scene = Scene()
        parent = scene.add_node(
            TransformNode(
                name="Parent",
                position=(3.0, 1.0, -2.0),
                rotation=(0.0, np.pi / 2.0, 0.0),
                local_scale=(2.0, 1.0, 2.0),
            )
        )
        child = make_mesh("Child")
        scene.add_mesh(child)
        scene.reparent_node(child, parent)

        scene.set_world_position(child, (8.0, 4.0, 1.0))

        self.assertTrue(
            np.allclose(scene.world_position(child), (8.0, 4.0, 1.0), atol=1e-6)
        )

    def test_playtest_bounds_include_parent_world_transform(self):
        scene = Scene()
        parent = scene.add_node(TransformNode(name="Parent", position=(5.0, 0.0, 0.0)))
        mesh = make_mesh("Collider")
        scene.add_mesh(mesh)
        scene.reparent_node(mesh, parent)

        minimum, maximum = PlaytestController._object_bounds(scene, mesh)

        self.assertTrue(np.allclose(minimum, (5.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(maximum, (6.0, 1.0, 0.0)))

    def test_reparent_rejects_cycles_without_changing_parent(self):
        scene = Scene()
        parent = scene.add_node(TransformNode(name="Parent"))
        child = scene.add_node(TransformNode(name="Child"), parent=parent)

        with self.assertRaisesRegex(ValueError, "cycle"):
            scene.reparent_node(parent, child)

        self.assertIsNone(parent.parent_id)
        self.assertEqual(child.parent_id, parent.id)

    def test_parent_id_and_scene_views_are_read_only(self):
        scene = Scene()
        parent = scene.add_node(TransformNode(name="Parent"))
        child = scene.add_node(TransformNode(name="Child"), parent=parent)

        with self.assertRaises(AttributeError):
            setattr(child, "parent_id", None)

        self.assertIsInstance(scene.objects, tuple)
        self.assertIsInstance(scene.lights, tuple)
        self.assertEqual(scene.graph.nodes, (scene.lights[0], parent, child))

    def test_graph_load_handles_parent_inserted_after_child(self):
        scene = Scene()
        child = make_mesh("Child")
        scene.add_mesh(child)
        parent = scene.add_node(TransformNode(name="Parent"))
        scene.reparent_node(child, parent)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ordered.DE"
            save_project(path, scene)
            loaded = load_project(path)

        loaded_parent = loaded.graph.get_node(parent.id)
        loaded_child = next(node for node in loaded.objects if node.name == "Child")
        assert loaded_parent is not None
        self.assertIsNone(loaded_parent.parent_id)
        self.assertEqual(loaded_child.parent_id, parent.id)

    def test_remove_requires_explicit_cascade_for_parent(self):
        scene = Scene()
        parent = scene.add_node(TransformNode(name="Parent"))
        child = scene.add_node(TransformNode(name="Child"), parent=parent)

        with self.assertRaisesRegex(ValueError, "cascade=True"):
            scene.remove_node(parent)

        self.assertIs(scene.graph.get_node(parent.id), parent)
        retained_light_id = scene.lights[0].id
        removed = scene.remove_node(parent, cascade=True)
        self.assertEqual({node.id for node in removed}, {parent.id, child.id})
        self.assertEqual({node.id for node in scene.graph.nodes}, {retained_light_id})

    def test_query_filters_by_component_parent_groups_tags_and_layer(self):
        scene = Scene()
        parent = scene.add_node(
            TransformNode(name="Group", groups={"environment"}, tags={"root"})
        )
        mesh = make_mesh("Floor")
        mesh.groups.add("environment")
        mesh.tags.add("walkable")
        mesh.layer = "world"
        scene.add_mesh(mesh)
        scene.reparent_node(mesh, parent)

        self.assertEqual(
            scene.graph.query(
                node_type=MeshInstance,
                parent=parent,
                group="environment",
                tag="walkable",
                layer="world",
            ),
            (mesh,),
        )
        self.assertEqual(scene.graph.query(name="not-found"), ())

    def test_save_load_preserves_ids_hierarchy_metadata_and_world_transform(self):
        scene = Scene()
        parent = scene.add_node(
            TransformNode(
                name="Root",
                position=(4.0, 1.0, -2.0),
                groups={"actors"},
                tags={"persistent"},
                layer="editor",
            )
        )
        mesh = make_mesh("Child", position=(1.0, 2.0, 3.0))
        scene.add_mesh(mesh)
        scene.reparent_node(mesh, parent)
        light = Light(name="Child light", position=(-2.0, 1.0, 0.0))
        scene.add_light(light)
        scene.reparent_node(light, parent)
        expected_world = scene.world_matrix(mesh)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scene.DE"
            save_project(path, scene)
            loaded = load_project(path)

        loaded_mesh = next(item for item in loaded.objects if item.name == "Child")
        loaded_parent = loaded.graph.get_node(parent.id)
        loaded_light = next(
            item for item in loaded.lights if item.name == "Child light"
        )
        assert loaded_parent is not None
        self.assertEqual(loaded_mesh.id, mesh.id)
        self.assertEqual(loaded_mesh.parent_id, parent.id)
        self.assertEqual(loaded_light.parent_id, parent.id)
        self.assertEqual(loaded_parent.groups, {"actors"})
        self.assertEqual(loaded_parent.tags, {"persistent"})
        self.assertEqual(loaded_parent.layer, "editor")
        self.assertTrue(np.allclose(loaded.world_matrix(loaded_mesh), expected_world))

    def test_loader_accepts_legacy_version_six_without_graph_extension(self):
        scene = Scene()
        mesh = make_mesh("Legacy", position=(3.0, 0.0, -1.0))
        scene.add_mesh(mesh)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy.DE"
            save_project(path, scene)
            with zipfile.ZipFile(path, "r") as source:
                entries = {name: source.read(name) for name in source.namelist()}
            manifest = json.loads(entries["scene.json"])
            manifest.pop("scene_graph")
            for item in manifest["objects"]:
                item.pop("node_id", None)
            for item in manifest["lights"]:
                item.pop("node_id", None)
            entries["scene.json"] = json.dumps(manifest).encode("utf-8")
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as target:
                for name, data in entries.items():
                    target.writestr(name, data)
            loaded = load_project(path)

        loaded_mesh = next(item for item in loaded.objects if item.name == "Legacy")
        self.assertIsNone(loaded_mesh.parent_id)
        self.assertNotEqual(loaded_mesh.id, mesh.id)
        self.assertTrue(np.allclose(loaded.world_position(loaded_mesh), mesh.position))


if __name__ == "__main__":
    unittest.main()
