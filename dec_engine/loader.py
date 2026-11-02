import os
from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image

from .logging_config import get_logger


_logger = get_logger(__name__)


@dataclass
class MeshData:
    name: str
    vertices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    color: tuple[float, float, float]
    texture_image: Image.Image | None = None


class ModelLoader:
    @staticmethod
    def supported_formats() -> tuple[str, ...]:
        return tuple(
            sorted(
                f".{extension}" for extension in trimesh.exchange.load.mesh_formats()
            )
        )

    @staticmethod
    def detect_format(path: str) -> str:
        return os.path.splitext(path)[1].lower()

    @staticmethod
    def load(path: str, color=None) -> MeshData:
        extension = ModelLoader.detect_format(path)
        if extension not in ModelLoader.supported_formats():
            supported = ", ".join(ModelLoader.supported_formats())
            raise ValueError(
                f"Unsupported model format '{extension or '(no extension)'}'. "
                f"Formats available in this installation: {supported}"
            )

        loaded = trimesh.load(path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) == 0:
                raise ValueError(f"Scene is empty: {path}")
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if isinstance(mesh, trimesh.Trimesh):
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise ValueError(f"Model has no renderable triangles: {path}")

            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.int32)
            normals = np.zeros_like(vertices)
            face_vectors = np.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]],
                vertices[faces[:, 2]] - vertices[faces[:, 0]],
            )
            for corner in range(3):
                np.add.at(normals, faces[:, corner], face_vectors)
            normal_lengths = np.linalg.norm(normals, axis=1)
            np.divide(
                normals,
                normal_lengths[:, np.newaxis],
                out=normals,
                where=normal_lengths[:, np.newaxis] > 1e-8,
            )
            visual = mesh.visual
            material = getattr(visual, "material", None)
            visual_uvs = getattr(visual, "uv", None)
            if visual_uvs is not None and len(visual_uvs) == len(vertices):
                uvs = np.asarray(visual_uvs, dtype=np.float32)
            else:
                uvs = np.zeros((len(vertices), 2), dtype=np.float32)
            indices = faces.reshape(-1)
            texture = getattr(material, "image", None)
            if texture is None:
                texture = getattr(material, "baseColorTexture", None)
            texture_image = None
            if isinstance(texture, Image.Image):
                texture_image = texture.copy()
                texture_image.load()
            if color is None:
                material_color = getattr(material, "main_color", None)
                if material_color is None:
                    material_color = getattr(visual, "main_color", None)
                if material_color is None:
                    material_color = getattr(material, "baseColorFactor", None)
                if material_color is None:
                    imported_color = np.ones(3, dtype=np.float32)
                else:
                    imported_color = np.asarray(material_color[:3], dtype=np.float32)
                    if np.max(imported_color) > 1.0:
                        imported_color /= 255.0
                if texture_image is not None:
                    imported_color = np.ones(3, dtype=np.float32)
                color = tuple(float(channel) for channel in imported_color)
            result = MeshData(
                os.path.basename(path),
                vertices,
                normals,
                uvs,
                indices,
                tuple(color),
                texture_image,
            )
            _logger.bind(
                operation="model_import",
                file=os.path.basename(path),
                vertex_count=len(vertices),
                face_count=len(faces),
            ).info("Model imported")
            return result

        raise TypeError(f"Unsupported trimesh payload for {path}: {type(mesh)!r}")


def load_model(path: str, color=None) -> MeshData:
    return ModelLoader.load(path, color=color)
