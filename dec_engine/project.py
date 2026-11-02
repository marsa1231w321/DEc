import hashlib
import io
import json
import os
import tempfile
import zipfile
from copy import deepcopy
from uuid import uuid4

import numpy as np

from .scene import Light, MeshInstance, Scene, TransformNode

_PROJECT_ASSET_CACHE = tempfile.TemporaryDirectory(prefix="dec-project-assets-")
CURRENT_PROJECT_FORMAT_VERSION = 6
_SUPPORTED_PROJECT_FORMAT_VERSIONS = frozenset(
    range(1, CURRENT_PROJECT_FORMAT_VERSION + 1)
)


def _advance_manifest_version(manifest, next_version):
    manifest["format_version"] = next_version
    return manifest


def _migrate_v3_to_v4(manifest):
    source_version = manifest["source_version"]
    if source_version >= 4:
        return _advance_manifest_version(manifest, 4)

    lighting = manifest.get("lighting", {})
    if not isinstance(lighting, dict):
        raise ValueError("Invalid legacy lighting data in project.")

    if source_version < 3:
        lights = []
    else:
        lights = manifest.get("lights", [])
        if not isinstance(lights, list):
            raise ValueError("Invalid legacy light list in project.")

    directional = next(
        (
            light
            for light in lights
            if isinstance(light, dict) and light.get("kind") == "directional"
        ),
        None,
    )
    legacy_light_fields = ("direction", "color", "intensity")
    if directional is None and any(field in lighting for field in legacy_light_fields):
        directional = {
            "name": "Sun",
            "kind": "directional",
            "position": [0.0, 0.0, 0.0],
            "direction": [-0.8, -1.0, -0.7],
            "color": [1.0, 0.96, 0.88],
            "intensity": 1.0,
            "enabled": True,
            "cast_shadows": True,
        }
        lights.append(directional)
    if directional is not None:
        for field in legacy_light_fields:
            if field in lighting:
                directional[field] = lighting[field]
    manifest["lights"] = lights
    return _advance_manifest_version(manifest, 4)


_PROJECT_MIGRATIONS = {
    version: (
        lambda manifest, target=version + 1: _advance_manifest_version(manifest, target)
    )
    for version in range(1, CURRENT_PROJECT_FORMAT_VERSION)
}
_PROJECT_MIGRATIONS[3] = _migrate_v3_to_v4


def migrate_project_manifest(old):
    if not isinstance(old, dict):
        raise ValueError("Project manifest must be a JSON object.")

    original = deepcopy(old)
    declared_version = original.get("format_version", original.get("version"))
    if (
        not isinstance(declared_version, int)
        or isinstance(declared_version, bool)
        or declared_version not in _SUPPORTED_PROJECT_FORMAT_VERSIONS
    ):
        raise ValueError(f"Unsupported project version: {declared_version!r}")

    source_version = original.get("source_version", declared_version)
    if (
        not isinstance(source_version, int)
        or isinstance(source_version, bool)
        or source_version not in _SUPPORTED_PROJECT_FORMAT_VERSIONS
    ):
        raise ValueError(f"Unsupported source project version: {source_version!r}")

    if "format_version" in original and "version" in original:
        if (
            original["format_version"] != original["version"]
            and "source_version" not in original
        ):
            raise ValueError("Project version markers do not match.")

    if source_version >= 4 and original.get("format") != "DEcProject":
        raise ValueError("This .DE archive is not a DEc map.")

    manifest = original
    manifest["source_version"] = source_version
    version = declared_version
    while version < CURRENT_PROJECT_FORMAT_VERSION:
        migrate_step = _PROJECT_MIGRATIONS.get(version)
        if migrate_step is None:
            raise ValueError(f"No project migration exists for version {version}.")
        manifest = migrate_step(manifest)
        version = manifest["format_version"]

    manifest["format_version"] = CURRENT_PROJECT_FORMAT_VERSION
    manifest["version"] = CURRENT_PROJECT_FORMAT_VERSION
    manifest["source_version"] = source_version
    return manifest


def _legacy_world_transform(scene, node):
    matrix = scene.world_matrix(node).astype(np.float64)
    basis = matrix[:3, :3]
    scale = np.linalg.norm(basis, axis=0)
    safe_scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = basis / safe_scale
    left, _, right = np.linalg.svd(normalized)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, 0] *= -1.0
        rotation = left @ right
        scale[0] *= -1.0
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    return (
        tuple(float(value) for value in matrix[:3, 3]),
        (roll, pitch, yaw),
        tuple(float(value) for value in scale),
        rotation,
    )


def _scene_graph_manifest(scene):
    nodes = []
    for node in scene.graph.nodes:
        if isinstance(node, MeshInstance):
            kind = "mesh"
            local = {
                "position": list(node.position),
                "rotation": list(node.rotation),
                "scale": node.scale,
                "scale_xyz": list(node.scale_xyz),
            }
        elif isinstance(node, Light):
            kind = "light"
            local = {
                "position": list(node.position),
                "rotation": list(node.rotation),
                "local_scale": list(node.local_scale),
                "direction": list(node.direction),
            }
        else:
            kind = "transform"
            local = {
                "position": list(node.position),
                "rotation": list(node.rotation),
                "local_scale": list(node.local_scale),
            }
        nodes.append(
            {
                "id": node.id,
                "kind": kind,
                "name": node.name,
                "parent_id": node.parent_id,
                "groups": sorted(node.groups),
                "tags": sorted(node.tags),
                "layer": node.layer,
                "local": local,
            }
        )
    return {"nodes": nodes}


def save_project(path, scene: Scene) -> None:
    manifest = {
        "format": "DEcProject",
        "version": CURRENT_PROJECT_FORMAT_VERSION,
        "format_version": CURRENT_PROJECT_FORMAT_VERSION,
        "objects": [],
        "lights": [
            {
                "node_id": light.id,
                "name": light.name,
                "kind": light.kind,
                "position": list(
                    light.position
                    if light.parent_id is None
                    else _legacy_world_transform(scene, light)[0]
                ),
                "direction": list(
                    light.direction
                    if light.parent_id is None
                    else _legacy_world_transform(scene, light)[3]
                    @ np.asarray(light.direction, dtype=np.float64)
                ),
                "color": list(light.color),
                "intensity": light.intensity,
                "enabled": light.enabled,
                "cast_shadows": light.cast_shadows,
            }
            for light in scene.lights
        ],
        "logic_actions": scene.logic_actions,
        "camera_keyframes": scene.camera_keyframes,
        "animation_tracks": scene.animation_tracks,
        "shadows": {
            "enabled": scene.shadows_enabled,
            "map_size": scene.shadow_map_size,
        },
        "lighting": {
            "direction": list(scene.light_direction),
            "color": list(scene.light_color),
            "intensity": scene.light_intensity,
            "ambient_intensity": scene.ambient_intensity,
        },
        "scene_graph": _scene_graph_manifest(scene),
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, obj in enumerate(scene.objects):
            world_position, world_rotation, world_scale, _ = _legacy_world_transform(
                scene, obj
            )
            if obj.parent_id is None:
                legacy_scale = obj.scale
                legacy_scale_xyz = obj.scale_xyz
            else:
                legacy_scale = 1.0
                legacy_scale_xyz = world_scale
            mesh_path = f"meshes/{index}.npz"
            arrays = io.BytesIO()
            mesh_data = {
                "vertices": obj.vertices,
                "normals": obj.normals,
                "uvs": obj.uvs,
                "indices": obj.indices,
            }
            if obj.rig:
                mesh_data["rig_base_vertices"] = obj.rig["base_vertices"]
                mesh_data["rig_weights"] = obj.rig["weights"]
            np.savez_compressed(arrays, **mesh_data)
            archive.writestr(mesh_path, arrays.getvalue())
            texture_entry = None
            if obj.texture_path:
                if not os.path.isfile(obj.texture_path):
                    raise ValueError(f"Texture file is missing: {obj.texture_path}")
                extension = os.path.splitext(obj.texture_path)[1].lower()
                texture_entry = f"assets/textures/{index}{extension}"
                archive.write(obj.texture_path, texture_entry)
            audio_entry = None
            if obj.audio_path:
                if not os.path.isfile(obj.audio_path):
                    raise ValueError(f"Audio file is missing: {obj.audio_path}")
                extension = os.path.splitext(obj.audio_path)[1].lower()
                audio_entry = f"assets/sounds/{index}{extension}"
                archive.write(obj.audio_path, audio_entry)
            manifest["objects"].append(
                {
                    "node_id": obj.id,
                    "mesh": mesh_path,
                    "name": obj.name,
                    "color": list(obj.color),
                    "position": list(world_position),
                    "rotation": list(world_rotation),
                    "scale": legacy_scale,
                    "scale_xyz": list(legacy_scale_xyz),
                    "source_path": obj.source_path,
                    "primitive": obj.primitive,
                    "visible": obj.visible,
                    "texture_path": f"archive://{texture_entry}"
                    if texture_entry
                    else None,
                    "texture_repeat": list(obj.texture_repeat),
                    "audio_path": f"archive://{audio_entry}" if audio_entry else None,
                    "sound_radius": obj.sound_radius,
                    "audio_loop": obj.audio_loop,
                    "npc_behavior": obj.npc_behavior,
                    "rig": ({"bones": obj.rig["bones"]} if obj.rig else None),
                }
            )
        archive.writestr(
            "scene.json",
            json.dumps(manifest, indent=2, ensure_ascii=True),
        )


def load_project(path) -> Scene:
    with zipfile.ZipFile(path, "r") as archive:
        try:
            manifest = migrate_project_manifest(json.loads(archive.read("scene.json")))
        except KeyError as error:
            raise ValueError("Project file is missing scene.json") from error
        scene = Scene()
        scene.graph.clear()
        for item in manifest.get("lights", []):
            scene.add_light(
                Light(
                    id=str(item.get("node_id") or uuid4()),
                    name=str(item["name"]),
                    kind=str(item["kind"]),
                    position=tuple(float(value) for value in item["position"]),
                    direction=tuple(float(value) for value in item["direction"]),
                    color=tuple(float(value) for value in item["color"]),
                    intensity=float(item["intensity"]),
                    enabled=bool(item.get("enabled", True)),
                    cast_shadows=bool(item.get("cast_shadows", True)),
                )
            )
        scene.logic_actions = manifest.get("logic_actions", [])
        scene.camera_keyframes = manifest.get("camera_keyframes", [])
        scene.animation_tracks = manifest.get("animation_tracks", [])
        if scene.camera_keyframes and not any(
            track.get("target_type") == "camera" for track in scene.animation_tracks
        ):
            scene.animation_tracks.append(
                {
                    "target_type": "camera",
                    "target": "Camera",
                    "interpolation": "linear",
                    "keys": [
                        {
                            "time": float(frame["time"]),
                            "values": {
                                "position": list(frame["position"]),
                                "yaw": float(frame["yaw"]),
                                "pitch": float(frame["pitch"]),
                                "roll": float(frame.get("roll", 0.0)),
                            },
                        }
                        for frame in scene.camera_keyframes
                    ],
                }
            )
        shadows = manifest.get("shadows", {})
        scene.shadows_enabled = bool(shadows.get("enabled", True))
        scene.shadow_map_size = int(shadows.get("map_size", 2048))
        if scene.shadow_map_size not in (512, 1024, 2048, 4096):
            raise ValueError(f"Unsupported shadow map size: {scene.shadow_map_size}")
        lighting = manifest.get("lighting", {})
        scene.ambient_intensity = float(
            lighting.get("ambient_intensity", scene.ambient_intensity)
        )
        cache_key = hashlib.sha256(
            os.path.abspath(os.fspath(path)).encode("utf-8")
        ).hexdigest()
        for item in manifest.get("objects", []):
            try:
                with np.load(
                    io.BytesIO(archive.read(item["mesh"])), allow_pickle=False
                ) as data:
                    vertices = np.asarray(data["vertices"], dtype=np.float32)
                    normals = np.asarray(data["normals"], dtype=np.float32)
                    uvs = np.asarray(data["uvs"], dtype=np.float32)
                    indices = np.asarray(data["indices"], dtype=np.int32)
                    rig_data = item.get("rig")
                    if rig_data:
                        rig_base_vertices = np.asarray(
                            data["rig_base_vertices"], dtype=np.float32
                        )
                        rig_weights = np.asarray(data["rig_weights"], dtype=np.float32)
                    else:
                        rig_base_vertices = None
                        rig_weights = None
                if (
                    vertices.ndim != 2
                    or vertices.shape[1] != 3
                    or normals.shape != vertices.shape
                    or uvs.shape != (len(vertices), 2)
                    or indices.ndim != 1
                    or len(indices) % 3
                    or (
                        len(indices)
                        and (indices.min() < 0 or indices.max() >= len(vertices))
                    )
                ):
                    raise ValueError(
                        f"Invalid mesh data for object {item.get('name', '<unnamed>')!r}"
                    )
                texture_path = item.get("texture_path")
                if texture_path and texture_path.startswith("archive://"):
                    texture_entry = texture_path[len("archive://") :].replace("\\", "/")
                    if not texture_entry.startswith(
                        "assets/textures/"
                    ) or ".." in texture_entry.split("/"):
                        raise ValueError(
                            f"Unsafe texture path in project: {texture_path!r}"
                        )
                    texture_path = os.path.join(
                        _PROJECT_ASSET_CACHE.name,
                        cache_key,
                        *texture_entry.split("/"),
                    )
                    os.makedirs(os.path.dirname(texture_path), exist_ok=True)
                    with (
                        archive.open(texture_entry, "r") as source,
                        open(texture_path, "wb") as target,
                    ):
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                audio_path = item.get("audio_path")
                if audio_path and audio_path.startswith("archive://"):
                    audio_entry = audio_path[len("archive://") :].replace("\\", "/")
                    if not audio_entry.startswith(
                        "assets/sounds/"
                    ) or ".." in audio_entry.split("/"):
                        raise ValueError(
                            f"Unsafe audio path in project: {audio_path!r}"
                        )
                    audio_path = os.path.join(
                        _PROJECT_ASSET_CACHE.name,
                        cache_key,
                        *audio_entry.split("/"),
                    )
                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                    with (
                        archive.open(audio_entry, "r") as source,
                        open(audio_path, "wb") as target,
                    ):
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                rig = None
                if rig_data:
                    bones = rig_data.get("bones", [])
                    if (
                        not bones
                        or rig_base_vertices.shape != vertices.shape
                        or rig_weights.shape != (len(vertices), len(bones))
                        or not np.isfinite(rig_base_vertices).all()
                        or not np.isfinite(rig_weights).all()
                        or np.any(rig_weights < 0)
                        or not np.allclose(rig_weights.sum(axis=1), 1.0, atol=1e-4)
                    ):
                        raise ValueError(
                            f"Invalid rig data for object {item.get('name', '<unnamed>')!r}"
                        )
                    for bone in bones:
                        if (
                            not isinstance(bone, dict)
                            or not bone.get("name")
                            or np.asarray(bone.get("head"), dtype=np.float32).shape
                            != (3,)
                            or np.asarray(bone.get("tail"), dtype=np.float32).shape
                            != (3,)
                            or np.asarray(bone.get("rotation"), dtype=np.float32).shape
                            != (3,)
                        ):
                            raise ValueError(
                                f"Invalid rig bone for object {item.get('name', '<unnamed>')!r}"
                            )
                    rig = {
                        "bones": bones,
                        "base_vertices": rig_base_vertices,
                        "weights": rig_weights,
                    }
                scene.add_mesh(
                    MeshInstance(
                        id=str(item.get("node_id") or uuid4()),
                        name=str(item["name"]),
                        vertices=vertices,
                        normals=normals,
                        uvs=uvs,
                        indices=indices,
                        color=tuple(float(channel) for channel in item["color"]),
                        position=tuple(float(value) for value in item["position"]),
                        rotation=tuple(float(value) for value in item["rotation"]),
                        scale=float(item["scale"]),
                        scale_xyz=tuple(
                            float(value)
                            for value in item.get("scale_xyz", (1.0, 1.0, 1.0))
                        ),
                        source_path=item.get("source_path"),
                        primitive=item.get("primitive"),
                        visible=bool(item.get("visible", True)),
                        texture_path=texture_path,
                        texture_repeat=tuple(
                            float(value)
                            for value in item.get("texture_repeat", (1.0, 1.0))
                        ),
                        audio_path=audio_path,
                        sound_radius=float(item.get("sound_radius", 12.0)),
                        audio_loop=bool(item.get("audio_loop", True)),
                        npc_behavior=str(item.get("npc_behavior", "idle")),
                        rig=rig,
                    )
                )
            except KeyError as error:
                raise ValueError(
                    f"Invalid project object: missing {error.args[0]}"
                ) from error

        graph_data = manifest.get("scene_graph")
        if graph_data is not None:
            if not isinstance(graph_data, dict) or not isinstance(
                graph_data.get("nodes"), list
            ):
                raise ValueError("Invalid scene_graph data in project.")
            components = {node.id: node for node in scene.graph.nodes}
            ordered_nodes = []
            for descriptor in graph_data["nodes"]:
                if not isinstance(descriptor, dict):
                    raise ValueError("Invalid scene graph node descriptor.")
                node_id = descriptor.get("id")
                kind = descriptor.get("kind")
                if not isinstance(node_id, str) or not node_id:
                    raise ValueError("Scene graph node is missing a valid ID.")
                node = components.get(node_id)
                if kind == "transform":
                    if node is not None:
                        raise ValueError(f"Duplicate scene graph node ID: {node_id}")
                    node = TransformNode(
                        name=str(descriptor.get("name", "Node")), id=node_id
                    )
                elif kind == "mesh" and isinstance(node, MeshInstance):
                    pass
                elif kind == "light" and isinstance(node, Light):
                    pass
                else:
                    raise ValueError(
                        f"Scene graph node {node_id!r} has an unknown or mismatched kind."
                    )
                parent_id = descriptor.get("parent_id")
                if parent_id is not None and not isinstance(parent_id, str):
                    raise ValueError(f"Invalid parent ID for scene node {node_id!r}.")
                groups = descriptor.get("groups", [])
                tags = descriptor.get("tags", [])
                if (
                    not isinstance(groups, list)
                    or not all(isinstance(value, str) for value in groups)
                    or not isinstance(tags, list)
                    or not all(isinstance(value, str) for value in tags)
                ):
                    raise ValueError(
                        f"Invalid groups or tags for scene node {node_id!r}."
                    )
                local = descriptor.get("local", {})
                if not isinstance(local, dict):
                    raise ValueError(
                        f"Invalid local transform for scene node {node_id!r}."
                    )
                node.name = str(descriptor.get("name", node.name))
                node.groups = set(groups)
                node.tags = set(tags)
                node.layer = str(descriptor.get("layer", "default"))
                node._parent_id = None
                if "position" in local:
                    node.position = tuple(float(value) for value in local["position"])
                if "rotation" in local:
                    node.rotation = tuple(float(value) for value in local["rotation"])
                if isinstance(node, MeshInstance):
                    node.scale = float(local.get("scale", node.scale))
                    node.scale_xyz = tuple(
                        float(value) for value in local.get("scale_xyz", node.scale_xyz)
                    )
                else:
                    node.local_scale = tuple(
                        float(value)
                        for value in local.get("local_scale", node.local_scale)
                    )
                if isinstance(node, Light) and "direction" in local:
                    node.direction = tuple(float(value) for value in local["direction"])
                ordered_nodes.append((node, parent_id))
            if len(components) + sum(
                1 for node, _ in ordered_nodes if isinstance(node, TransformNode)
            ) != len(ordered_nodes):
                raise ValueError("Scene graph contains duplicate nodes.")
            scene.graph.clear()
            for node, parent_id in ordered_nodes:
                scene.graph.add(node)
            for node, parent_id in ordered_nodes:
                if parent_id is not None:
                    if scene.graph.get_node(parent_id) is None:
                        raise ValueError(
                            f"Scene graph parent does not exist: {parent_id!r}."
                        )
                    scene.graph.reparent(node, parent_id)
            if set(components) - {node.id for node, _ in ordered_nodes}:
                raise ValueError("Scene graph is missing object or light nodes.")

    return scene
