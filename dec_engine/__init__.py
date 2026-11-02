"""DEc 3D engine and editor."""

from .camera import Camera
from .loader import ModelLoader, load_model
from .scene import Scene, MeshInstance
from .renderer import Renderer

__all__ = ["Camera", "ModelLoader", "load_model", "Scene", "MeshInstance", "Renderer"]
__version__ = "0.1.0"
