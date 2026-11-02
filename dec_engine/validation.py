import os
from dataclasses import dataclass

import numpy as np

from .logic import KEY_CODES

LOGIC_TRIGGERS = {"on_start", "on_timer", "on_key", "on_player_near"}
LOGIC_ACTIONS = {
    "set_position",
    "add_rotation",
    "set_color",
    "toggle_visibility",
    "set_scale",
}

BLOCKER = "blocker"
WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """Единичная находка валидации с уровнем и контекстом перехода."""

    message: str
    severity: str = BLOCKER
    object_id: str | None = None
    object_name: str | None = None
    property_name: str | None = None

    @property
    def is_blocker(self) -> bool:
        return self.severity == BLOCKER

    @property
    def property(self) -> str | None:
        return self.property_name


def _check_vector(
    issues,
    value,
    length,
    label,
    *,
    severity=BLOCKER,
    object_id=None,
    object_name=None,
    property_name=None,
):
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        issues.append(
            ValidationIssue(
                f"{label}: ожидался вектор из {length} чисел.",
                severity,
                object_id,
                object_name,
                property_name,
            )
        )
        return None
    if vector.shape != (length,) or not np.isfinite(vector).all():
        issues.append(
            ValidationIssue(
                f"{label}: нужен конечный вектор из {length} чисел.",
                severity,
                object_id,
                object_name,
                property_name,
            )
        )
        return None
    return vector


def _node_issue(issues, node, message, severity, property_name):
    issues.append(
        ValidationIssue(
            message,
            severity,
            getattr(node, "id", None),
            getattr(node, "name", None),
            property_name,
        )
    )


def collect_validation_issues(scene) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    object_names = set()

    for index, obj in enumerate(scene.objects, start=1):
        label = f"Объект {index} «{obj.name}»"

        def add(message, severity=BLOCKER, property_name=None, obj=obj):
            _node_issue(issues, obj, message, severity, property_name)

        name = str(obj.name).strip()
        if not name:
            add(f"{label}: имя не может быть пустым.", WARNING, "name")
        elif name in object_names:
            add(f"{label}: имя объекта повторяется.", WARNING, "name")
        object_names.add(name)

        try:
            vertices = np.asarray(obj.vertices, dtype=np.float64)
            indices = np.asarray(obj.indices)
            normals = np.asarray(obj.normals, dtype=np.float64)
            uvs = np.asarray(obj.uvs, dtype=np.float64)
        except (TypeError, ValueError):
            add(
                f"{label}: геометрия содержит данные неверного формата.",
                BLOCKER,
                "geometry",
            )
            continue
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            add(f"{label}: нет корректных вершин формата X/Y/Z.", BLOCKER, "vertices")
            continue
        if not np.isfinite(vertices).all():
            add(
                f"{label}: координаты вершин содержат NaN или бесконечность.",
                BLOCKER,
                "vertices",
            )
        if normals.shape != vertices.shape or not np.isfinite(normals).all():
            add(
                f"{label}: нормали отсутствуют, имеют неверный размер или не конечны.",
                BLOCKER,
                "normals",
            )
        if uvs.shape != (len(vertices), 2) or not np.isfinite(uvs).all():
            add(
                f"{label}: UV-координаты имеют неверный размер или не конечны.",
                BLOCKER,
                "uvs",
            )
        if (
            indices.ndim != 1
            or not len(indices)
            or len(indices) % 3
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(vertices))
        ):
            add(
                f"{label}: индексы должны описывать треугольники существующих вершин.",
                BLOCKER,
                "indices",
            )

        position = _check_vector(
            issues,
            obj.position,
            3,
            f"{label}, позиция",
            object_id=obj.id,
            object_name=obj.name,
            property_name="position",
        )
        rotation = _check_vector(
            issues,
            obj.rotation,
            3,
            f"{label}, поворот",
            object_id=obj.id,
            object_name=obj.name,
            property_name="rotation",
        )
        color = _check_vector(
            issues,
            obj.color,
            3,
            f"{label}, цвет",
            severity=WARNING,
            object_id=obj.id,
            object_name=obj.name,
            property_name="color",
        )
        scale_xyz = _check_vector(
            issues,
            obj.scale_xyz,
            3,
            f"{label}, размеры",
            object_id=obj.id,
            object_name=obj.name,
            property_name="scale_xyz",
        )
        try:
            scale = float(obj.scale)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError
        except (TypeError, ValueError):
            add(
                f"{label}: общий масштаб должен быть конечным и больше нуля.",
                BLOCKER,
                "scale",
            )
        if scale_xyz is not None and np.any(scale_xyz < 0):
            add(
                f"{label}: размеры по осям не могут быть отрицательными.",
                BLOCKER,
                "scale_xyz",
            )
        if color is not None and np.any((color < 0) | (color > 1)):
            add(
                f"{label}: каналы цвета должны находиться в диапазоне 0–1.",
                WARNING,
                "color",
            )
        try:
            repeat = np.asarray(obj.texture_repeat, dtype=np.float64)
            if (
                repeat.shape != (2,)
                or not np.isfinite(repeat).all()
                or np.any(repeat <= 0)
            ):
                raise ValueError
        except (TypeError, ValueError):
            add(
                f"{label}: повтор текстуры должен содержать два положительных числа.",
                WARNING,
                "texture_repeat",
            )
        if obj.primitive == "sound_source":
            try:
                radius = float(obj.sound_radius)
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                add(
                    f"{label}: радиус источника звука должен быть положительным.",
                    WARNING,
                    "sound_radius",
                )
        if position is None or rotation is None:
            continue

        for resource_name, path, property_name in (
            ("текстура", obj.texture_path, "texture_path"),
            ("звук", obj.audio_path, "audio_path"),
        ):
            if path and not os.path.isfile(path):
                add(
                    f"{label}: файл {resource_name} не найден: {path}",
                    WARNING,
                    property_name,
                )

    light_names = set()
    for index, light in enumerate(scene.lights, start=1):
        label = f"Источник света {index} «{light.name}»"
        name = str(light.name).strip()
        if not name:
            _node_issue(
                issues, light, f"{label}: имя не может быть пустым.", WARNING, "name"
            )
        elif name in light_names:
            _node_issue(
                issues,
                light,
                f"{label}: имя источника света повторяется.",
                WARNING,
                "name",
            )
        light_names.add(name)
        if light.kind not in ("directional", "point"):
            _node_issue(
                issues, light, f"{label}: неизвестный тип источника.", BLOCKER, "kind"
            )
        position = _check_vector(
            issues,
            light.position,
            3,
            f"{label}, позиция",
            object_id=light.id,
            object_name=light.name,
            property_name="position",
        )
        direction = _check_vector(
            issues,
            light.direction,
            3,
            f"{label}, направление",
            object_id=light.id,
            object_name=light.name,
            property_name="direction",
        )
        color = _check_vector(
            issues,
            light.color,
            3,
            f"{label}, цвет",
            severity=WARNING,
            object_id=light.id,
            object_name=light.name,
            property_name="color",
        )
        if (
            light.kind == "directional"
            and direction is not None
            and np.linalg.norm(direction) < 1e-8
        ):
            _node_issue(
                issues,
                light,
                f"{label}: направление не может быть нулевым.",
                BLOCKER,
                "direction",
            )
        if color is not None and np.any((color < 0) | (color > 1)):
            _node_issue(
                issues,
                light,
                f"{label}: каналы цвета должны находиться в диапазоне 0–1.",
                WARNING,
                "color",
            )
        try:
            intensity = float(light.intensity)
            if not np.isfinite(intensity) or not 0 <= intensity <= 4:
                raise ValueError
        except (TypeError, ValueError):
            _node_issue(
                issues,
                light,
                f"{label}: яркость должна быть конечным числом от 0 до 4.",
                WARNING,
                "intensity",
            )

    try:
        ambient = float(scene.ambient_intensity)
        if not np.isfinite(ambient) or not 0 <= ambient <= 2:
            raise ValueError
    except (TypeError, ValueError):
        issues.append(
            ValidationIssue(
                "Общее заполнение света должно быть конечным числом от 0 до 2.",
                WARNING,
                property_name="ambient_intensity",
            )
        )
    if scene.shadow_map_size not in (512, 1024, 2048, 4096):
        issues.append(
            ValidationIssue(
                "Размер карты теней должен быть 512, 1024, 2048 или 4096.",
                BLOCKER,
                property_name="shadow_map_size",
            )
        )

    for index, node in enumerate(scene.logic_actions, start=1):
        label = f"Правило логики {index}"
        if not isinstance(node, dict):
            issues.append(
                ValidationIssue(
                    f"{label}: ожидается объект с настройками правила.",
                    BLOCKER,
                    property_name="rule",
                )
            )
            continue
        trigger = node.get("trigger")
        action = node.get("action")
        if not isinstance(trigger, str) or trigger not in LOGIC_TRIGGERS:
            issues.append(
                ValidationIssue(
                    f"{label}: неизвестное событие.",
                    BLOCKER,
                    property_name="trigger",
                )
            )
        if not isinstance(action, str) or action not in LOGIC_ACTIONS:
            issues.append(
                ValidationIssue(
                    f"{label}: неизвестное действие.",
                    BLOCKER,
                    property_name="action",
                )
            )
        target = node.get("target")
        if not isinstance(target, str) or target not in object_names:
            issues.append(
                ValidationIssue(
                    f"{label}: целевой объект отсутствует на карте.",
                    BLOCKER,
                    property_name="target",
                )
            )
        if trigger == "on_key" and str(node.get("key", "E")).upper() not in KEY_CODES:
            issues.append(
                ValidationIssue(
                    f"{label}: клавиша события не поддерживается.",
                    WARNING,
                    property_name="key",
                )
            )
        if trigger in ("on_timer", "on_player_near"):
            field = "delay" if trigger == "on_timer" else "radius"
            try:
                value = float(node.get(field, 1.0 if field == "delay" else 3.0))
                if (
                    not np.isfinite(value)
                    or value < 0
                    or (field == "radius" and value == 0)
                ):
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        f"{label}: параметр «{field}» имеет некорректное значение.",
                        WARNING,
                        property_name=field,
                    )
                )
        if action != "toggle_visibility":
            value = _check_vector(
                issues,
                node.get("value"),
                3,
                f"{label}, значение",
                property_name="value",
            )
            if value is not None and action == "set_color":
                if np.any((value < 0) | (value > 1)):
                    issues.append(
                        ValidationIssue(
                            f"{label}: каналы цвета должны находиться в диапазоне 0–1.",
                            WARNING,
                            property_name="value",
                        )
                    )
            if value is not None and action == "set_scale" and np.any(value <= 0):
                issues.append(
                    ValidationIssue(
                        f"{label}: размер по каждой оси должен быть больше нуля.",
                        WARNING,
                        property_name="value",
                    )
                )

    return issues


def validate_scene(scene) -> list[str]:
    """Обратно совместимый список сообщений валидации."""

    return [issue.message for issue in collect_validation_issues(scene)]


def blocking_issues(scene) -> list[ValidationIssue]:
    return [issue for issue in collect_validation_issues(scene) if issue.is_blocker]


def has_blockers(scene) -> bool:
    return any(issue.is_blocker for issue in collect_validation_issues(scene))
