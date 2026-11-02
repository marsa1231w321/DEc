from __future__ import annotations

import base64
import json
import math
import pickle
from copy import deepcopy
from dataclasses import fields
from typing import Protocol

import numpy as np

from .scene import Light, MeshInstance, Scene, SceneNode, TransformNode


class Command(Protocol):
    def do(self, scene: Scene) -> None: ...

    def undo(self, scene: Scene) -> None: ...

    def to_record(self) -> dict: ...


def _encode_value(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Command records do not support object arrays.")
        return {
            "$array": base64.b64encode(value.tobytes()).decode("ascii"),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return _encode_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, tuple):
        return {"$tuple": [_encode_value(item) for item in value]}
    if isinstance(value, set):
        return {"$set": [_encode_value(item) for item in sorted(value, key=repr)]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Command records only support string-keyed dictionaries.")
        return {key: _encode_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported command value: {type(value).__name__}")


def _decode_value(value):
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$array" in value:
        raw = base64.b64decode(value["$array"], validate=True)
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Command records do not support object arrays.")
        array = np.frombuffer(raw, dtype=dtype)
        return array.copy().reshape(tuple(value["shape"]))
    if "$tuple" in value:
        return tuple(_decode_value(item) for item in value["$tuple"])
    if "$set" in value:
        return set(_decode_value(item) for item in value["$set"])
    if "$float" in value:
        return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}[
            value["$float"]
        ]
    return {key: _decode_value(item) for key, item in value.items()}


def _node_to_record(node: SceneNode) -> dict:
    return {
        "type": type(node).__name__,
        "fields": {
            field.name: _encode_value(getattr(node, field.name))
            for field in fields(node)
        },
    }


def _node_from_record(record: dict) -> SceneNode:
    if not isinstance(record, dict) or not isinstance(record.get("fields"), dict):
        raise ValueError("Invalid command node record.")
    node_type = record.get("type")
    values = {
        name: _decode_value(value) for name, value in record["fields"].items()
    }
    if node_type == "TransformNode":
        return TransformNode(
            name=values["name"],
            id=values["id"],
            parent_id=values["_parent_id"],
            groups=values["groups"],
            tags=values["tags"],
            layer=values["layer"],
            position=values["position"],
            rotation=values["rotation"],
            local_scale=values["_local_scale"],
        )
    node_types = {
        "SceneNode": SceneNode,
        "MeshInstance": MeshInstance,
        "Light": Light,
    }
    node_class = node_types.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown command node type: {node_type!r}")
    return node_class(**values)


def _insert_node(scene: Scene, node: SceneNode, index: int) -> None:
    graph = scene.graph
    if graph.get_node(node.id) is not None:
        raise ValueError(f"Command node already exists: {node.id}")
    graph.insert_at(deepcopy(node), index)


def _remove_node(scene: Scene, node_id: str) -> SceneNode:
    node = scene.graph.get_node(node_id)
    if node is None:
        raise ValueError(f"Command node does not exist: {node_id}")
    return scene.graph.remove(node)[0]


class SetNodePropertyCommand:
    def __init__(
        self, node_id: str, property_name: str, before, after, label: str | None = None
    ):
        self.node_id = node_id
        self.property_name = property_name
        self.before = deepcopy(before)
        self.after = deepcopy(after)
        self.label = label or f"Изменение {property_name}"

    def _set(self, scene: Scene, value) -> None:
        node = scene.graph.get_node(self.node_id)
        if node is None:
            raise ValueError(f"Command node does not exist: {self.node_id}")
        allowed = {field.name for field in fields(node)} - {"id"}
        if self.property_name not in allowed:
            raise ValueError(f"Unsupported node property: {self.property_name}")
        setattr(node, self.property_name, deepcopy(value))

    def do(self, scene: Scene) -> None:
        self._set(scene, self.after)

    def undo(self, scene: Scene) -> None:
        self._set(scene, self.before)

    def to_record(self) -> dict:
        return {
            "type": "set_node_property",
            "node_id": self.node_id,
            "property": self.property_name,
            "before": _encode_value(self.before),
            "after": _encode_value(self.after),
            "label": self.label,
        }


class SetScenePropertyCommand:
    def __init__(
        self, property_name: str, before, after, label: str | None = None
    ):
        self.property_name = property_name
        self.before = deepcopy(before)
        self.after = deepcopy(after)
        self.label = label or f"Изменение {property_name}"

    def _set(self, scene: Scene, value) -> None:
        if self.property_name not in scene.__dict__ or self.property_name in {
            "events",
            "graph",
            "command_history",
        }:
            raise ValueError(f"Unsupported scene property: {self.property_name}")
        setattr(scene, self.property_name, deepcopy(value))

    def do(self, scene: Scene) -> None:
        self._set(scene, self.after)

    def undo(self, scene: Scene) -> None:
        self._set(scene, self.before)

    def to_record(self) -> dict:
        return {
            "type": "set_scene_property",
            "property": self.property_name,
            "before": _encode_value(self.before),
            "after": _encode_value(self.after),
            "label": self.label,
        }


class AddNodeCommand:
    def __init__(self, node: SceneNode, index: int | None = None):
        self.node = deepcopy(node)
        self.index = index
        self.label = f"Добавление узла «{node.name}»"

    def do(self, scene: Scene) -> None:
        index = len(scene.graph.nodes) if self.index is None else self.index
        _insert_node(scene, self.node, index)
        self.index = index

    def undo(self, scene: Scene) -> None:
        _remove_node(scene, self.node.id)

    def to_record(self) -> dict:
        if self.index is None:
            raise ValueError("An unapplied node command cannot be serialized.")
        return {
            "type": "add_node",
            "node": _node_to_record(self.node),
            "index": self.index,
        }


class RemoveNodeCommand:
    def __init__(
        self, node_id: str, node: SceneNode | None = None, index: int | None = None
    ):
        self.node_id = node_id
        self.node = deepcopy(node) if node is not None else None
        self.index = index
        self.label = f"Удаление узла {node_id}"

    def do(self, scene: Scene) -> None:
        current = scene.graph.get_node(self.node_id)
        if current is None:
            raise ValueError(f"Command node does not exist: {self.node_id}")
        if self.node is None:
            self.node = deepcopy(current)
        if self.index is None:
            self.index = scene.graph.index_of(self.node_id)
        _remove_node(scene, self.node_id)

    def undo(self, scene: Scene) -> None:
        if self.node is None or self.index is None:
            raise ValueError("Node removal has not been applied.")
        _insert_node(scene, self.node, self.index)

    def to_record(self) -> dict:
        if self.node is None or self.index is None:
            raise ValueError("An unapplied node command cannot be serialized.")
        return {
            "type": "remove_node",
            "node": _node_to_record(self.node),
            "index": self.index,
        }


class TransactionCommand:
    def __init__(self, label: str, commands: tuple[Command, ...]):
        self.label = label
        self.commands = commands

    def do(self, scene: Scene) -> None:
        completed = []
        try:
            for command in self.commands:
                command.do(scene)
                completed.append(command)
        except Exception:
            for command in reversed(completed):
                command.undo(scene)
            raise

    def undo(self, scene: Scene) -> None:
        for command in reversed(self.commands):
            command.undo(scene)

    def to_record(self) -> dict:
        return {
            "type": "transaction",
            "label": self.label,
            "commands": [command.to_record() for command in self.commands],
        }


def command_from_record(record: dict) -> Command:
    if not isinstance(record, dict):
        raise ValueError("Invalid command record.")
    command_type = record.get("type")
    if command_type == "set_node_property":
        return SetNodePropertyCommand(
            str(record["node_id"]),
            str(record["property"]),
            _decode_value(record["before"]),
            _decode_value(record["after"]),
            str(record.get("label") or ""),
        )
    if command_type == "set_scene_property":
        return SetScenePropertyCommand(
            str(record["property"]),
            _decode_value(record["before"]),
            _decode_value(record["after"]),
            str(record.get("label") or ""),
        )
    if command_type in ("add_node", "remove_node"):
        node = _node_from_record(record["node"])
        index = int(record["index"])
        if command_type == "add_node":
            return AddNodeCommand(node, index)
        return RemoveNodeCommand(node.id, node, index)
    if command_type == "transaction":
        children = record.get("commands")
        if not isinstance(children, list):
            raise ValueError("Invalid transaction command list.")
        commands = tuple(command_from_record(item) for item in children)
        return TransactionCommand(str(record["label"]), commands)
    raise ValueError(f"Unknown command type: {command_type!r}")


def build_scene_delta(before: bytes, after: bytes, label: str) -> TransactionCommand:
    old_state = pickle.loads(before)
    new_state = pickle.loads(after)
    old_node_list = old_state["graph"].nodes
    new_node_list = new_state["graph"].nodes
    old_nodes = {node.id: node for node in old_node_list}
    new_nodes = {node.id: node for node in new_node_list}
    commands: list[Command] = []

    old_ids = [node.id for node in old_node_list]
    new_ids = [node.id for node in new_node_list]
    old_indices = {node_id: index for index, node_id in enumerate(old_ids)}
    new_indices = {node_id: index for index, node_id in enumerate(new_ids)}

    for node_id in old_ids:
        if node_id not in new_nodes:
            continue
        old_node = old_nodes[node_id]
        new_node = new_nodes[node_id]
        for field in fields(old_node):
            if field.name == "id":
                continue
            old_value = getattr(old_node, field.name)
            new_value = getattr(new_node, field.name)
            if pickle.dumps(old_value, protocol=pickle.HIGHEST_PROTOCOL) != pickle.dumps(
                new_value, protocol=pickle.HIGHEST_PROTOCOL
            ):
                commands.append(
                    SetNodePropertyCommand(
                        node_id, field.name, old_value, new_value, label
                    )
                )

    for field_name, old_value in old_state.items():
        if field_name in {"events", "graph", "command_history"}:
            continue
        new_value = new_state[field_name]
        if pickle.dumps(old_value, protocol=pickle.HIGHEST_PROTOCOL) != pickle.dumps(
            new_value, protocol=pickle.HIGHEST_PROTOCOL
        ):
            commands.append(
                SetScenePropertyCommand(field_name, old_value, new_value, label)
            )

    for node_id in reversed(old_ids):
        if node_id not in new_nodes:
            commands.append(
                RemoveNodeCommand(node_id, old_nodes[node_id], old_indices[node_id])
            )

    for node_id in new_ids:
        if node_id not in old_nodes:
            commands.append(AddNodeCommand(new_nodes[node_id], new_indices[node_id]))

    return TransactionCommand(label, tuple(commands))


class CommandTransaction:
    def __init__(self, stack: CommandStack, label: str):
        self.stack = stack
        self.label = label
        self.commands: list[Command] = []

    def __enter__(self) -> CommandTransaction:
        return self

    def execute(self, command: Command) -> None:
        command.do(self.stack.scene)
        self.commands.append(command)

    def __exit__(self, error_type, error, traceback) -> bool:
        if error_type is not None:
            for command in reversed(self.commands):
                command.undo(self.stack.scene)
            return False
        if self.commands:
            self.stack.record_applied(
                TransactionCommand(self.label, tuple(self.commands))
            )
        return False


class CommandStack:
    def __init__(
        self,
        scene: Scene,
        history: object | None = None,
        max_history: int = 30,
        max_memory_bytes: int = 128 * 1024 * 1024,
    ):
        self.scene = scene
        self.max_history = max_history
        self.max_memory_bytes = max_memory_bytes
        if history is None:
            history = {"undo": [], "redo": []}
        if not isinstance(history, dict):
            raise ValueError("Invalid command history.")
        undo_records = history.get("undo", [])
        redo_records = history.get("redo", [])
        if (
            not isinstance(undo_records, list)
            or not isinstance(redo_records, list)
            or not all(isinstance(item, dict) for item in undo_records + redo_records)
        ):
            raise ValueError("Invalid command history.")
        try:
            self._undo = [command_from_record(item) for item in undo_records]
            self._redo = [command_from_record(item) for item in redo_records]
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError("Invalid command history.") from error
        self._trim()
        self._sync_history()

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs) -> CommandStack:
        return cls(scene, history=scene.command_history, **kwargs)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def execute(self, command: Command) -> None:
        command.do(self.scene)
        self.record_applied(command)

    def record_applied(self, command: Command) -> bool:
        self._undo.append(command)
        self._redo.clear()
        recorded = self._trim(required_command=command)
        self._sync_history()
        self.scene.notify_changed("command")
        return recorded

    def undo(self) -> bool:
        if not self._undo:
            return False
        command = self._undo[-1]
        command.undo(self.scene)
        self._undo.pop()
        self._redo.append(command)
        self._trim()
        self._sync_history()
        self.scene.notify_changed("undo")
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        command = self._redo[-1]
        command.do(self.scene)
        self._redo.pop()
        self._undo.append(command)
        self._trim()
        self._sync_history()
        self.scene.notify_changed("redo")
        return True

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._trim()
        self._sync_history()

    def transaction(self, label: str) -> CommandTransaction:
        return CommandTransaction(self, label)

    def to_record(self) -> dict:
        return {
            "undo": [command.to_record() for command in self._undo],
            "redo": [command.to_record() for command in self._redo],
        }

    def _trim(self, required_command: Command | None = None) -> bool:
        while len(self._undo) > self.max_history:
            del self._undo[0]
        while len(self._redo) > self.max_history:
            del self._redo[0]
        try:
            record = self.to_record()
            while (
                len(
                    json.dumps(
                        record, separators=(",", ":"), ensure_ascii=True
                    ).encode("utf-8")
                )
                > self.max_memory_bytes
            ):
                if self._undo:
                    del self._undo[0]
                elif self._redo:
                    del self._redo[0]
                else:
                    self._history_record = {"undo": [], "redo": []}
                    return False
                record = self.to_record()
            self._history_record = record
            return required_command is None or any(
                item is required_command for item in self._undo + self._redo
            )
        except (TypeError, ValueError):
            self._undo.clear()
            self._redo.clear()
            self._history_record = {"undo": [], "redo": []}
            return False

    def _sync_history(self) -> None:
        self.scene.command_history = self._history_record
