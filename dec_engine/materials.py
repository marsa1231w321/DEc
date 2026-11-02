"""PBR material model shared by the loader, scene and project format.

Materials are intentionally plain data (no PIL/GL objects) so they can be
validated, serialized and unit-tested without a running editor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")


def _vector3(value, label, *, upper=None):
    vector = tuple(float(channel) for channel in value)
    if len(vector) != 3:
        raise ValueError(f"{label} must have exactly 3 channels.")
    for channel in vector:
        if not math.isfinite(channel) or channel < 0.0:
            raise ValueError(f"{label} channels must be finite and non-negative.")
        if upper is not None and channel > upper:
            raise ValueError(f"{label} channels must be in the 0..{upper} range.")
    return vector


def _unit(value, label):
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite value in the 0..1 range.")
    return number


@dataclass(frozen=True)
class Material:
    """A single PBR material with optional texture references."""

    name: str = "Material"
    base_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_strength: float = 1.0
    alpha_mode: str = "OPAQUE"
    base_color_texture: str | None = None
    normal_texture: str | None = None
    metallic_roughness_texture: str | None = None
    emissive_texture: str | None = None

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("Material name cannot be empty.")
        object.__setattr__(self, "base_color", _vector3(self.base_color, "base_color", upper=1.0))
        object.__setattr__(self, "emissive", _vector3(self.emissive, "emissive"))
        object.__setattr__(self, "metallic", _unit(self.metallic, "metallic"))
        object.__setattr__(self, "roughness", _unit(self.roughness, "roughness"))
        strength = float(self.emissive_strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("emissive_strength must be finite and non-negative.")
        object.__setattr__(self, "emissive_strength", strength)
        if self.alpha_mode not in ALPHA_MODES:
            raise ValueError(f"Unknown alpha_mode: {self.alpha_mode!r}")
        for field_name in (
            "base_color_texture",
            "normal_texture",
            "metallic_roughness_texture",
            "emissive_texture",
        ):
            value = getattr(self, field_name)
            if value is not None and not str(value):
                raise ValueError(f"{field_name} must be a non-empty string or None.")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_color": list(self.base_color),
            "metallic": self.metallic,
            "roughness": self.roughness,
            "emissive": list(self.emissive),
            "emissive_strength": self.emissive_strength,
            "alpha_mode": self.alpha_mode,
            "base_color_texture": self.base_color_texture,
            "normal_texture": self.normal_texture,
            "metallic_roughness_texture": self.metallic_roughness_texture,
            "emissive_texture": self.emissive_texture,
        }

    @classmethod
    def from_dict(cls, data) -> "Material":
        if not isinstance(data, dict):
            raise ValueError("Material data must be a mapping.")
        try:
            return cls(
                name=str(data.get("name", "Material")),
                base_color=tuple(data.get("base_color", (1.0, 1.0, 1.0))),
                metallic=float(data.get("metallic", 0.0)),
                roughness=float(data.get("roughness", 1.0)),
                emissive=tuple(data.get("emissive", (0.0, 0.0, 0.0))),
                emissive_strength=float(data.get("emissive_strength", 1.0)),
                alpha_mode=str(data.get("alpha_mode", "OPAQUE")),
                base_color_texture=data.get("base_color_texture"),
                normal_texture=data.get("normal_texture"),
                metallic_roughness_texture=data.get("metallic_roughness_texture"),
                emissive_texture=data.get("emissive_texture"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid material data: {error}") from error

    def with_texture(self, field_name: str, path: str | None) -> "Material":
        if field_name not in (
            "base_color_texture",
            "normal_texture",
            "metallic_roughness_texture",
            "emissive_texture",
        ):
            raise ValueError(f"Unknown material texture field: {field_name!r}")
        return replace(self, **{field_name: path})


def default_material(name: str = "Material", color=(1.0, 1.0, 1.0)) -> Material:
    return Material(name=name, base_color=color)
