from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar
from uuid import uuid4

import numpy as np

from .events import EventBus, SceneChanged


def _rotation_matrix(rotation):
    x, y, z = rotation
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


@dataclass(kw_only=True)
class SceneNode:
    name: str = "Node"
    id: str = field(default_factory=lambda: str(uuid4()))
    _parent_id: str | None = field(default=None, repr=False)
    groups: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    layer: str = "default"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _local_scale: tuple[float, float, float] = field(
        default=(1.0, 1.0, 1.0), repr=False
    )

    @property
    def parent_id(self) -> str | None:
        return self._parent_id

    @property
    def local_scale(self) -> tuple[float, float, float]:
        return self._local_scale

    @local_scale.setter
    def local_scale(self, value: tuple[float, float, float]) -> None:
        self._local_scale = (
            float(value[0]),
            float(value[1]),
            float(value[2]),
        )

    def local_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = _rotation_matrix(self.rotation) @ np.diag(self.local_scale)
        matrix[:3, 3] = self.position
        return matrix


@dataclass(init=False)
class TransformNode(SceneNode):
    def __init__(
        self,
        *,
        name: str = "Node",
        id: str | None = None,
        parent_id: str | None = None,
        groups: set[str] | None = None,
        tags: set[str] | None = None,
        layer: str = "default",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__(
            name=name,
            id=id or str(uuid4()),
            _parent_id=parent_id,
            groups=set(groups or ()),
            tags=set(tags or ()),
            layer=layer,
            position=position,
            rotation=rotation,
            _local_scale=local_scale,
        )


@dataclass
class MeshInstance(SceneNode):
    vertices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    color: tuple[float, float, float]
    scale: float = 1.0
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    source_path: str | None = None
    primitive: str | None = None
    visible: bool = True
    texture_path: str | None = None
    texture_repeat: tuple[float, float] = (1.0, 1.0)
    audio_path: str | None = None
    sound_radius: float = 12.0
    audio_loop: bool = True
    npc_behavior: str = "idle"
    rig: dict | None = None

    @property
    def local_scale(self) -> tuple[float, float, float]:
        return (
            float(self.scale_xyz[0] * self.scale),
            float(self.scale_xyz[1] * self.scale),
            float(self.scale_xyz[2] * self.scale),
        )

    def model_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = (
            _rotation_matrix(self.rotation) @ np.diag(self.scale_xyz) * self.scale
        )
        matrix[:3, 3] = self.position
        return matrix

    def local_matrix(self) -> np.ndarray:
        return self.model_matrix()


@dataclass
class Light(SceneNode):
    name: str = "Sun"
    kind: str = "directional"
    direction: tuple[float, float, float] = (-0.8, -1.0, -0.7)
    color: tuple[float, float, float] = (1.0, 0.96, 0.88)
    intensity: float = 1.0
    enabled: bool = True
    cast_shadows: bool = True


NodeType = TypeVar("NodeType", bound=SceneNode)


class SceneGraph:
    def __init__(self):
        self._nodes: dict[str, SceneNode] = {}

    @property
    def nodes(self) -> tuple[SceneNode, ...]:
        return tuple(self._nodes.values())

    @property
    def root_nodes(self) -> tuple[SceneNode, ...]:
        return tuple(node for node in self._nodes.values() if node.parent_id is None)

    def add(self, node: NodeType, parent: SceneNode | str | None = None) -> NodeType:
        if not isinstance(node, SceneNode):
            raise TypeError("SceneGraph accepts SceneNode instances.")
        if not node.id or not isinstance(node.id, str):
            raise ValueError("Scene node ID must be a non-empty string.")
        if node.id in self._nodes:
            raise ValueError(f"Scene node ID already exists: {node.id}")
        if parent is not None:
            parent_node = self._resolve(parent)
            node._parent_id = parent_node.id
        elif node._parent_id is not None and node._parent_id not in self._nodes:
            raise ValueError(f"Parent node does not exist: {node._parent_id}")
        self._nodes[node.id] = node
        return node

    def insert_at(self, node: NodeType, index: int) -> NodeType:
        added = self.add(node)
        nodes = list(self._nodes.items())
        nodes.pop()
        nodes.insert(min(max(index, 0), len(nodes)), (added.id, added))
        self._nodes = dict(nodes)
        return added

    def index_of(self, node_id: str) -> int:
        try:
            return tuple(self._nodes).index(node_id)
        except ValueError as error:
            raise ValueError(f"Scene node does not exist: {node_id}") from error

    def clear(self) -> None:
        self._nodes.clear()

    def get_node(self, node: SceneNode | str) -> SceneNode | None:
        node_id = node.id if isinstance(node, SceneNode) else node
        return self._nodes.get(node_id)

    def children_of(self, node: SceneNode | str) -> tuple[SceneNode, ...]:
        parent = self._resolve(node)
        return tuple(
            child for child in self._nodes.values() if child.parent_id == parent.id
        )

    def remove(
        self, node: SceneNode | str, *, cascade: bool = False
    ) -> tuple[SceneNode, ...]:
        target = self._resolve(node)
        children = self.children_of(target)
        if children and not cascade:
            raise ValueError("Cannot remove a node with children unless cascade=True.")
        removed = []
        if cascade:
            for child in children:
                removed.extend(self.remove(child, cascade=True))
        removed.append(target)
        del self._nodes[target.id]
        return tuple(removed)

    def reparent(
        self,
        node: SceneNode | str,
        parent: SceneNode | str | None,
    ) -> SceneNode:
        target = self._resolve(node)
        parent_node = self._resolve(parent) if parent is not None else None
        if parent_node is target:
            raise ValueError("A scene node cannot be its own parent.")
        ancestor = parent_node
        while ancestor is not None:
            if ancestor is target:
                raise ValueError("Reparenting would create a scene graph cycle.")
            ancestor = (
                self._nodes.get(ancestor.parent_id)
                if ancestor.parent_id is not None
                else None
            )
        target._parent_id = parent_node.id if parent_node is not None else None
        return target

    def query(
        self,
        *,
        node_id: str | None = None,
        name: str | None = None,
        node_type: type[SceneNode] | None = None,
        parent: SceneNode | str | None = None,
        group: str | None = None,
        tag: str | None = None,
        layer: str | None = None,
    ) -> tuple[SceneNode, ...]:
        parent_id = parent.id if isinstance(parent, SceneNode) else parent
        return tuple(
            node
            for node in self._nodes.values()
            if (node_id is None or node.id == node_id)
            and (name is None or node.name == name)
            and (node_type is None or isinstance(node, node_type))
            and (parent_id is None or node.parent_id == parent_id)
            and (group is None or group in node.groups)
            and (tag is None or tag in node.tags)
            and (layer is None or node.layer == layer)
        )

    def world_matrix(self, node: SceneNode | str) -> np.ndarray:
        current = self._resolve(node)
        chain = []
        while current is not None:
            chain.append(current)
            current = (
                self._nodes.get(current.parent_id)
                if current.parent_id is not None
                else None
            )
        matrix = np.eye(4, dtype=np.float32)
        for ancestor in reversed(chain):
            matrix = matrix @ ancestor.local_matrix()
        return matrix

    def world_position(self, node: SceneNode | str) -> tuple[float, float, float]:
        position = self.world_matrix(node)[:3, 3]
        return float(position[0]), float(position[1]), float(position[2])

    def set_world_position(
        self,
        node: SceneNode | str,
        position: tuple[float, float, float],
    ) -> None:
        target = self._resolve(node)
        desired = np.asarray(position, dtype=np.float64)
        if desired.shape != (3,) or not np.isfinite(desired).all():
            raise ValueError("World position must be a finite 3D vector.")
        if target.parent_id is not None:
            parent_world = self.world_matrix(target.parent_id).astype(np.float64)
            desired = np.linalg.inv(parent_world) @ np.append(desired, 1.0)
            desired = desired[:3]
        target.position = (
            float(desired[0]),
            float(desired[1]),
            float(desired[2]),
        )

    def world_transform(self, node: SceneNode | str) -> dict:
        matrix = self.world_matrix(node)
        basis = matrix[:3, :3].astype(np.float64)
        scale = np.linalg.norm(basis, axis=0)
        if np.any(scale < 1e-8):
            raise ValueError("World transform has a zero scale component.")
        rotation = basis / scale
        if np.linalg.det(rotation) < 0:
            scale[0] *= -1.0
            rotation[:, 0] *= -1.0
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(
                "World transform contains shear and cannot be represented as TRS."
            )
        pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
        if abs(np.cos(pitch)) > 1e-6:
            roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        else:
            roll = 0.0
            yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
        return {
            "position": tuple(float(value) for value in matrix[:3, 3]),
            "rotation": (roll, pitch, yaw),
            "scale": tuple(float(value) for value in scale),
        }

    def world_direction(
        self,
        node: SceneNode | str,
        direction: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        basis = self.world_matrix(node)[:3, :3].astype(np.float64)
        left, _, right = np.linalg.svd(basis)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, 0] *= -1.0
            rotation = left @ right
        rotated = rotation @ np.asarray(direction, dtype=np.float64)
        length = float(np.linalg.norm(rotated))
        if length < 1e-8:
            return (0.0, -1.0, 0.0)
        normalized = rotated / length
        return (
            float(normalized[0]),
            float(normalized[1]),
            float(normalized[2]),
        )

    def _resolve(self, node: SceneNode | str) -> SceneNode:
        node_id = node.id if isinstance(node, SceneNode) else node
        resolved = self._nodes.get(node_id)
        if resolved is None:
            raise KeyError(f"Scene node does not exist: {node_id}")
        if isinstance(node, SceneNode) and resolved is not node:
            raise ValueError("Scene node instance does not belong to this graph.")
        return resolved


class Scene:
    def __init__(self):
        self.events = EventBus()
        self.graph = SceneGraph()
        self.add_light(Light())
        self.logic_actions: list[dict] = []
        self.camera_keyframes: list[dict] = []
        self.animation_tracks: list[dict] = []
        self.command_history: dict = {"undo": [], "redo": []}
        self.ambient_intensity = 0.25
        self.shadows_enabled = True
        self.shadow_map_size = 2048

    @property
    def objects(self) -> tuple[MeshInstance, ...]:
        return tuple(
            node for node in self.graph.nodes if isinstance(node, MeshInstance)
        )

    @property
    def lights(self) -> tuple[Light, ...]:
        return tuple(node for node in self.graph.nodes if isinstance(node, Light))

    @property
    def light_direction(self):
        light = next(
            (light for light in self.lights if light.kind == "directional"), None
        )
        return light.direction if light else (0.0, -1.0, 0.0)

    @light_direction.setter
    def light_direction(self, value):
        light = next(
            (light for light in self.lights if light.kind == "directional"), None
        )
        if light is None:
            light = Light()
            self.add_light(light)
        light.direction = tuple(value)

    @property
    def light_color(self):
        light = next(
            (light for light in self.lights if light.kind == "directional"), None
        )
        return light.color if light else (1.0, 1.0, 1.0)

    @light_color.setter
    def light_color(self, value):
        light = next(
            (light for light in self.lights if light.kind == "directional"), None
        )
        if light is None:
            light = Light()
            self.add_light(light)
        light.color = tuple(value)

    @property
    def light_intensity(self):
        light = next(
            (light for light in self.lights if light.kind == "directional"), None
        )
        return light.intensity if light else 0.0

    @light_intensity.setter
    def light_intensity(self, value):
        light = next(
            (light for light in self.lights if light.kind == "directional"), None
        )
        if light is None:
            light = Light()
            self.add_light(light)
        light.intensity = float(value)

    def add_mesh(self, instance: MeshInstance) -> None:
        self.add_node(instance)

    def add_light(self, light: Light) -> None:
        self.add_node(light)

    def add_node(
        self, node: SceneNode, parent: SceneNode | str | None = None
    ) -> SceneNode:
        added = self.graph.add(node, parent)
        self.events.publish(
            SceneChanged(
                operation="added",
                node_id=added.id,
                node_kind=type(added).__name__,
                parent_id=added.parent_id,
            )
        )
        return added

    def remove_node(self, node: SceneNode | str, *, cascade: bool = False):
        removed = self.graph.remove(node, cascade=cascade)
        for removed_node in removed:
            self.events.publish(
                SceneChanged(
                    operation="removed",
                    node_id=removed_node.id,
                    node_kind=type(removed_node).__name__,
                    parent_id=removed_node.parent_id,
                )
            )
        return removed

    def reparent_node(self, node: SceneNode | str, parent: SceneNode | str | None):
        target = self.graph.get_node(node)
        if target is None:
            raise KeyError(f"Scene node does not exist: {node!r}")
        previous_parent_id = target.parent_id
        updated = self.graph.reparent(node, parent)
        self.events.publish(
            SceneChanged(
                operation="reparented",
                node_id=updated.id,
                node_kind=type(updated).__name__,
                parent_id=updated.parent_id,
                previous_parent_id=previous_parent_id,
            )
        )
        return updated

    def rename_node(self, node: SceneNode | str, new_name: str) -> SceneNode:
        """Переименовать узел и синхронизировать ссылки логики/анимации."""

        target = self.graph.get_node(node)
        if target is None:
            raise KeyError(f"Scene node does not exist: {node!r}")
        name = str(new_name)
        if not name.strip():
            raise ValueError("Scene node name cannot be empty.")
        if name == target.name:
            return target
        previous_name = target.name
        target.name = name
        self._sync_name_references(previous_name, name)
        self.events.publish(
            SceneChanged(
                operation="updated",
                node_id=target.id,
                node_kind=type(target).__name__,
                parent_id=target.parent_id,
                fields=("name",),
            )
        )
        return target

    def _sync_name_references(self, previous_name: str, new_name: str) -> None:
        if not previous_name or previous_name == new_name:
            return
        for rule in self.logic_actions:
            if isinstance(rule, dict) and rule.get("target") == previous_name:
                rule["target"] = new_name
        for track in self.animation_tracks:
            if not isinstance(track, dict):
                continue
            if (
                track.get("target_type") == "object"
                and track.get("target") == previous_name
            ):
                track["target"] = new_name
            elif track.get("target_type") == "bone" and str(
                track.get("target", "")
            ).startswith(f"{previous_name}::"):
                track["target"] = (
                    f"{new_name}::{str(track['target']).split('::', 1)[1]}"
                )

    def search_nodes(
        self,
        text: str = "",
        *,
        node_type=None,
        tag: str | None = None,
        group: str | None = None,
        layer: str | None = None,
    ) -> tuple[SceneNode, ...]:
        """Поиск узлов по подстроке имени с фильтрами графа."""

        needle = str(text).strip().lower()
        matches = self.graph.query(
            node_type=node_type, tag=tag, group=group, layer=layer
        )
        if not needle:
            return matches
        return tuple(node for node in matches if needle in node.name.lower())

    def notify_changed(self, *fields: str) -> None:
        self.events.publish(
            SceneChanged(
                operation="updated",
                node_id=None,
                node_kind="Scene",
                fields=tuple(fields),
            )
        )

    def world_matrix(self, node: SceneNode | str) -> np.ndarray:
        return self.graph.world_matrix(node)

    def world_position(self, node: SceneNode | str) -> tuple[float, float, float]:
        return self.graph.world_position(node)

    def set_world_position(
        self,
        node: SceneNode | str,
        position: tuple[float, float, float],
    ) -> None:
        updated = self.graph.get_node(node)
        if updated is None:
            raise KeyError(f"Scene node does not exist: {node!r}")
        self.graph.set_world_position(node, position)
        self.events.publish(
            SceneChanged(
                operation="updated",
                node_id=updated.id,
                node_kind=type(updated).__name__,
                fields=("position",),
            )
        )

    def add_ground(self, size=20.0):
        plane = np.array(
            [
                [-size, 0.0, -size, 0.0, 1.0, 0.0, 0.0, 0.0],
                [-size, 0.0, size, 0.0, 1.0, 0.0, 0.0, 1.0],
                [size, 0.0, size, 0.0, 1.0, 0.0, 1.0, 1.0],
                [-size, 0.0, -size, 0.0, 1.0, 0.0, 0.0, 0.0],
                [size, 0.0, size, 0.0, 1.0, 0.0, 1.0, 1.0],
                [size, 0.0, -size, 0.0, 1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.add_mesh(
            MeshInstance(
                name="ground",
                vertices=plane[:, :3],
                normals=plane[:, 3:6],
                uvs=plane[:, 6:8],
                indices=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
                color=(0.35, 0.35, 0.38),
                position=(0.0, 0.0, 0.0),
                primitive="ground",
            )
        )
