import pickle
import unittest

import numpy as np

from dec_engine.events import (
    EventBus,
    ResourceChanged,
    SceneChanged,
    SelectionChanged,
)
from dec_engine.scene import MeshInstance, Scene, TransformNode


def make_mesh(name):
    return MeshInstance(
        name=name,
        vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32),
        normals=np.asarray(((0, 0, 1),) * 3, dtype=np.float32),
        uvs=np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float32),
        indices=np.asarray((0, 1, 2), dtype=np.int32),
        color=(1.0, 1.0, 1.0),
    )


class EventBusTests(unittest.TestCase):
    def test_delivery_is_ordered_and_scoped_to_exact_event_type(self):
        bus = EventBus()
        delivered = []
        bus.subscribe(
            SceneChanged, lambda event: delivered.append(("scene", event.node_id))
        )
        bus.subscribe(
            ResourceChanged, lambda _event: delivered.append(("resource", None))
        )
        bus.subscribe(
            SceneChanged, lambda event: delivered.append(("second", event.node_id))
        )

        bus.publish(SceneChanged("added", "node-1", "MeshInstance"))

        self.assertEqual(delivered, [("scene", "node-1"), ("second", "node-1")])

    def test_unsubscribe_is_idempotent_and_releases_listener(self):
        bus = EventBus()
        delivered = []
        subscription = bus.subscribe(SelectionChanged, delivered.append)

        subscription.close()
        subscription.close()
        bus.publish(SelectionChanged((), None, "none"))

        self.assertEqual(bus.listener_count(), 0)
        self.assertEqual(delivered, [])
        self.assertFalse(subscription.active)

    def test_publish_uses_stable_snapshot_when_listener_subscribes(self):
        bus = EventBus()
        delivered = []

        def late_listener(_event):
            delivered.append("late")

        def first_listener(event):
            delivered.append("first")
            bus.subscribe(SceneChanged, late_listener)

        bus.subscribe(SceneChanged, first_listener)
        event = SceneChanged("updated", "node-1", "MeshInstance")
        bus.publish(event)
        self.assertEqual(delivered, ["first"])

        bus.publish(event)
        self.assertEqual(delivered, ["first", "first", "late"])

    def test_scene_publishes_add_reparent_remove_and_world_transform_events(self):
        scene = Scene()
        received = []
        scene.events.subscribe(SceneChanged, received.append)
        parent = scene.add_node(TransformNode(name="Parent"))
        mesh = make_mesh("Mesh")
        scene.add_mesh(mesh)
        scene.reparent_node(mesh, parent)
        scene.set_world_position(mesh, (2.0, 0.0, 0.0))
        scene.remove_node(mesh)

        self.assertEqual(
            [event.operation for event in received],
            ["added", "added", "reparented", "updated", "removed"],
        )
        self.assertEqual(received[2].previous_parent_id, None)
        self.assertEqual(received[2].parent_id, parent.id)
        self.assertEqual(received[3].fields, ("position",))

    def test_resource_and_selection_events_are_typed_and_publishable(self):
        bus = EventBus()
        received = []
        bus.subscribe(ResourceChanged, received.append)
        bus.subscribe(SelectionChanged, received.append)

        bus.publish(ResourceChanged("texture", "assigned", "stone.png", "mesh-1"))
        bus.publish(SelectionChanged(("mesh-1",), "mesh-1", "object_list"))

        self.assertEqual(received[0].resource_kind, "texture")
        self.assertEqual(received[1].active_id, "mesh-1")

    def test_event_subscriptions_are_not_copied_into_scene_snapshots(self):
        scene = Scene()
        received = []
        scene.events.subscribe(SceneChanged, received.append)

        snapshot = pickle.loads(pickle.dumps(scene.__dict__))
        restored_bus = snapshot["events"]
        self.assertEqual(restored_bus.listener_count(), 0)
        self.assertEqual(scene.events.listener_count(), 1)


if __name__ == "__main__":
    unittest.main()
