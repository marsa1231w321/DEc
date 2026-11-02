import unittest

import numpy as np

from dec_engine.events import SceneChanged
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


class SceneSearchTests(unittest.TestCase):
    def setUp(self):
        self.scene = Scene()
        self.alpha = make_mesh("Alpha")
        self.alpha.tags.add("enemy")
        self.alpha.layer = "gameplay"
        self.scene.add_mesh(self.alpha)
        self.beta = make_mesh("Beta")
        self.beta.groups.add("props")
        self.scene.add_mesh(self.beta)
        self.group = self.scene.add_node(TransformNode(name="PropsGroup"))

    def test_search_is_case_insensitive_substring(self):
        matches = self.scene.search_nodes("ALP")
        self.assertEqual([node.id for node in matches], [self.alpha.id])

    def test_search_without_text_returns_all_nodes(self):
        self.assertEqual(len(self.scene.search_nodes()), len(self.scene.graph.nodes))

    def test_search_combines_name_tag_and_layer_filters(self):
        self.assertEqual(
            [
                node.id
                for node in self.scene.search_nodes(
                    "alpha", tag="enemy", layer="gameplay"
                )
            ],
            [self.alpha.id],
        )
        self.assertEqual(self.scene.search_nodes("alpha", layer="default"), ())

    def test_search_filters_by_node_type_and_group(self):
        meshes = self.scene.search_nodes(node_type=MeshInstance)
        self.assertEqual({node.id for node in meshes}, {self.alpha.id, self.beta.id})
        grouped = self.scene.search_nodes(group="props")
        self.assertEqual([node.id for node in grouped], [self.beta.id])
        lights = self.scene.search_nodes(node_type=Light)
        self.assertEqual(len(lights), 1)

    def test_rename_updates_logic_and_animation_references(self):
        self.scene.logic_actions = [
            {"trigger": "on-start", "target": "Alpha", "action": "set-color"},
            {"trigger": "timer", "target": "Other", "action": "set-color"},
        ]
        self.scene.animation_tracks = [
            {
                "target_type": "object",
                "target": "Alpha",
                "property": "position",
                "keys": [],
            },
            {
                "target_type": "bone",
                "target": "Alpha::arm",
                "property": "rotation",
                "keys": [],
            },
            {
                "target_type": "object",
                "target": "Beta",
                "property": "position",
                "keys": [],
            },
        ]
        events = []
        self.scene.events.subscribe(SceneChanged, events.append)

        self.scene.rename_node(self.alpha, "Hero")

        self.assertEqual(self.alpha.name, "Hero")
        self.assertEqual(self.scene.logic_actions[0]["target"], "Hero")
        self.assertEqual(self.scene.logic_actions[1]["target"], "Other")
        self.assertEqual(self.scene.animation_tracks[0]["target"], "Hero")
        self.assertEqual(self.scene.animation_tracks[1]["target"], "Hero::arm")
        self.assertEqual(self.scene.animation_tracks[2]["target"], "Beta")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].operation, "updated")
        self.assertEqual(events[0].node_id, self.alpha.id)
        self.assertEqual(events[0].fields, ("name",))

    def test_rename_accepts_node_id_and_ignores_same_name(self):
        events = []
        self.scene.events.subscribe(SceneChanged, events.append)
        self.scene.rename_node(self.alpha.id, "Alpha")
        self.assertEqual(self.alpha.name, "Alpha")
        self.assertEqual(events, [])

    def test_rename_rejects_empty_name_and_unknown_node(self):
        with self.assertRaises(ValueError):
            self.scene.rename_node(self.alpha, "   ")
        with self.assertRaises(KeyError):
            self.scene.rename_node("missing-node", "Name")


if __name__ == "__main__":
    unittest.main()
