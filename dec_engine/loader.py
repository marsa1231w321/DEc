import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from .logging_config import get_logger
from .materials import ALPHA_MODES, Material, default_material

_logger = get_logger(__name__)


@dataclass
class MeshData:
    name: str
    vertices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    color: tuple
    texture_image: Image.Image | None = None
    materials: tuple[Material, ...] = ()
    material_indices: np.ndarray | None = None
    material_report: tuple[str, ...] = field(default_factory=tuple)


def _vector3_factor(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    if array.size == 1:
        array = np.repeat(array, 3)
    array = array[:3]
    if np.max(np.abs(array)) > 1.0:
        array = array / 255.0
    return tuple(float(min(max(channel, 0.0), 1.0)) for channel in array)


def _scalar(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _alpha_mode(value: Any) -> str:
    text = str(value or "OPAQUE").upper()
    return text if text in ALPHA_MODES else "OPAQUE"


def extract_material(source, fallback_name: str, report: list[str]) -> Material:
    """Convert a trimesh material into a validated :class:`Material`."""

    if source is None:
        report.append(
            f"{fallback_name}: материал не найден, применён цвет по умолчанию."
        )
        return default_material(fallback_name)
    name = str(getattr(source, "name", "") or fallback_name)
    base_color = _vector3_factor(getattr(source, "baseColorFactor", None))
    if base_color is None:
        base_color = _vector3_factor(getattr(source, "main_color", None))
    if base_color is None:
        base_color = (1.0, 1.0, 1.0)
    metallic = _scalar(
        getattr(source, "metallicFactor", None),
        _scalar(getattr(source, "metallic", None), 0.0),
    )
    roughness = _scalar(
        getattr(source, "roughnessFactor", None),
        _scalar(getattr(source, "roughness", None), 1.0),
    )
    emissive = _vector3_factor(getattr(source, "emissiveFactor", None)) or (
        0.0,
        0.0,
        0.0,
    )
    emissive_strength = _scalar(getattr(source, "emissiveStrength", None), 1.0)
    for attribute, label in (
        ("normalTexture", "normal map"),
        ("occlusionTexture", "occlusion map"),
        ("metallicRoughnessTexture", "metallic/roughness map"),
        ("emissiveTexture", "emissive map"),
    ):
        if getattr(source, attribute, None) is not None:
            report.append(
                f"{name}: {label} не сохранён — путь к файлу недоступен."
            )
    try:
        return Material(
            name=name,
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            emissive=emissive,
            emissive_strength=emissive_strength,
            alpha_mode=_alpha_mode(getattr(source, "alphaMode", "OPAQUE")),
        )
    except ValueError as error:
        report.append(f"{name}: параметры материала сброшены ({error}).")
        return default_material(name, base_color)


def geometry_part(geometry, index: int, report: list[str]) -> dict | None:
    """Flatten one trimesh geometry into arrays plus its material."""

    if not isinstance(geometry, trimesh.Trimesh):
        report.append(f"Часть {index}: не является сеткой и пропущена.")
        return None
    vertices = np.asarray(geometry.vertices, dtype=np.float32)
    faces = np.asarray(geometry.faces, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        report.append(f"Часть {index}: пустая геометрия пропущена.")
        return None
    normals = np.asarray(geometry.vertex_normals, dtype=np.float32)
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        normals = np.zeros_like(vertices, dtype=np.float32)
    visual = getattr(geometry, "visual", None)
    uvs = getattr(visual, "uv", None)
    if uvs is not None and len(uvs) == len(vertices):
        uvs = np.asarray(uvs, dtype=np.float32)
        if not np.isfinite(uvs).all():
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    else:
        uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    part_name = str(getattr(geometry, "name", "") or f"Part {index}")
    source = getattr(visual, "material", None)
    material = extract_material(source, part_name, report)
    image = getattr(source, "image", None) if source is not None else None
    if not isinstance(image, Image.Image):
        image = None
    return {
        "vertices": vertices,
        "normals": normals,
        "uvs": uvs,
        "faces": faces,
        "material": material,
        "texture_image": image,
    }
def combine_geometry_parts(
    parts,
    name: str,
    color=None,
    report: list[str] | None = None,
) -> MeshData:
    """Merge geometry parts into one mesh, keeping per-face material indices."""

    usable = [part for part in parts if part]
    if not usable:
        raise ValueError(f"Model has no renderable geometry: {name}")
    vertices = np.concatenate([part["vertices"] for part in usable])
    normals = np.concatenate([part["normals"] for part in usable])
    uvs = np.concatenate([part["uvs"] for part in usable])
    materials = tuple(part["material"] for part in usable)
    material_indices = []
    faces = []
    offset = 0
    for index, part in enumerate(usable):
        part_faces = np.asarray(part["faces"], dtype=np.int32)
        faces.append(part_faces + offset)
        material_indices.append(np.full(len(part_faces), index, dtype=np.int32))
        offset += len(part["vertices"])
    indices = np.concatenate(faces).reshape(-1).astype(np.int32)
    textures = [
        part["texture_image"]
        for part in usable
        if part.get("texture_image") is not None
    ]
    texture_image = textures[0] if textures else None
    if texture_image is not None:
        texture_image = texture_image.copy()
        texture_image.load()
        if report is not None:
            if len(textures) > 1:
                report.append(
                    f"{name}: найдено {len(textures)} встроенных текстур, "
                    "сохранена только первая как основная текстура объекта."
                )
            else:
                report.append(
                    f"{name}: встроенная текстура сохранена как основная текстура "
                    "объекта."
                )
    base_color = color
    if base_color is None:
        base_color = materials[0].base_color
    if texture_image is not None and color is None:
        base_color = (1.0, 1.0, 1.0)
    data = MeshData(
        name=name,
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        indices=indices,
        color=tuple(float(channel) for channel in base_color),
        texture_image=texture_image,
        materials=materials,
        material_indices=np.concatenate(material_indices),
    )
    if report is not None:
        data.material_report = tuple(report)
    return data


class ModelLoader:
    @staticmethod
    def supported_formats() -> list[str]:
        return sorted(trimesh.available_formats())

    @staticmethod
    def load(path: str, color=None) -> MeshData:
        path = os.fspath(path)
        extension = os.path.splitext(path)[1].lower()
        if extension not in ModelLoader.supported_formats():
            supported = ", ".join(ModelLoader.supported_formats())
            raise ValueError(
                f"Unsupported model format '{extension or '(no extension)'}'. "
                f"Formats available in this installation: {supported}"
            )

        loaded = trimesh.load(path, force="scene")
        report: list[str] = []
        if isinstance(loaded, trimesh.Scene):
            geometries = [loaded.geometry[key] for key in loaded.geometry]
        else:
            geometries = [loaded]
        if not geometries:
            raise ValueError(f"Scene is empty: {path}")
        parts = []
        for index, geometry in enumerate(geometries):
            part = geometry_part(geometry, index, report)
            if part is not None:
                parts.append(part)
        result = combine_geometry_parts(
            parts, os.path.basename(path), color=color, report=report
        )
        _logger.bind(
            operation="model_import",
            file=os.path.basename(path),
            vertex_count=len(result.vertices),
            face_count=len(result.indices) // 3,
            material_count=len(result.materials),
            material_issues=len(result.material_report),
        ).info("Model imported")
        for message in result.material_report:
            _logger.bind(operation="model_import", file=result.name).warning(message)
        return result


def load_model(path: str, color=None) -> MeshData:
    return ModelLoader.load(path, color=color)

