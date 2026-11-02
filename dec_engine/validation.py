import os

import numpy as np

from .logic import KEY_CODES

LOGIC_TRIGGERS = {"on_start", "on_timer", "on_key", "on_player_near"}
LOGIC_ACTIONS = {
    "set_position", "add_rotation", "set_color", "toggle_visibility", "set_scale",
}


def validate_scene(scene):
    problems = []
    object_names = set()

    def check_vector(value, length, label):
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            problems.append(f"{label}: ожидался вектор из {length} чисел.")
            return None
        if vector.shape != (length,) or not np.isfinite(vector).all():
            problems.append(f"{label}: нужен конечный вектор из {length} чисел.")
            return None
        return vector

    for index, obj in enumerate(scene.objects, start=1):
        label = f"Объект {index} «{obj.name}»"
        name = str(obj.name).strip()
        if not name:
            problems.append(f"{label}: имя не может быть пустым.")
        elif name in object_names:
            problems.append(f"{label}: имя объекта повторяется.")
        object_names.add(name)

        try:
            vertices = np.asarray(obj.vertices, dtype=np.float64)
            indices = np.asarray(obj.indices)
            normals = np.asarray(obj.normals, dtype=np.float64)
            uvs = np.asarray(obj.uvs, dtype=np.float64)
        except (TypeError, ValueError):
            problems.append(f"{label}: геометрия содержит данные неверного формата.")
            continue
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            problems.append(f"{label}: нет корректных вершин формата X/Y/Z.")
            continue
        if not np.isfinite(vertices).all():
            problems.append(f"{label}: координаты вершин содержат NaN или бесконечность.")
        if normals.shape != vertices.shape or not np.isfinite(normals).all():
            problems.append(f"{label}: нормали отсутствуют, имеют неверный размер или не конечны.")
        if uvs.shape != (len(vertices), 2) or not np.isfinite(uvs).all():
            problems.append(f"{label}: UV-координаты имеют неверный размер или не конечны.")
        if (
            indices.ndim != 1
            or not len(indices)
            or len(indices) % 3
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(vertices))
        ):
            problems.append(f"{label}: индексы должны описывать треугольники существующих вершин.")

        position = check_vector(obj.position, 3, f"{label}, позиция")
        rotation = check_vector(obj.rotation, 3, f"{label}, поворот")
        color = check_vector(obj.color, 3, f"{label}, цвет")
        scale_xyz = check_vector(obj.scale_xyz, 3, f"{label}, размеры")
        try:
            scale = float(obj.scale)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"{label}: общий масштаб должен быть конечным и больше нуля.")
        if scale_xyz is not None and np.any(scale_xyz < 0):
            problems.append(f"{label}: размеры по осям не могут быть отрицательными.")
        if color is not None and np.any((color < 0) | (color > 1)):
            problems.append(f"{label}: каналы цвета должны находиться в диапазоне 0–1.")
        try:
            repeat = np.asarray(obj.texture_repeat, dtype=np.float64)
            if repeat.shape != (2,) or not np.isfinite(repeat).all() or np.any(repeat <= 0):
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"{label}: повтор текстуры должен содержать два положительных числа.")
        if obj.primitive == "sound_source":
            try:
                radius = float(obj.sound_radius)
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                problems.append(f"{label}: радиус источника звука должен быть положительным.")
        if position is None or rotation is None:
            continue

        for resource_name, path in (
            ("текстура", obj.texture_path),
            ("звук", obj.audio_path),
        ):
            if path and not os.path.isfile(path):
                problems.append(f"{label}: файл {resource_name} не найден: {path}")

    light_names = set()
    for index, light in enumerate(scene.lights, start=1):
        label = f"Источник света {index} «{light.name}»"
        name = str(light.name).strip()
        if not name:
            problems.append(f"{label}: имя не может быть пустым.")
        elif name in light_names:
            problems.append(f"{label}: имя источника света повторяется.")
        light_names.add(name)
        if light.kind not in ("directional", "point"):
            problems.append(f"{label}: неизвестный тип источника.")
        check_vector(light.position, 3, f"{label}, позиция")
        direction = check_vector(light.direction, 3, f"{label}, направление")
        color = check_vector(light.color, 3, f"{label}, цвет")
        if light.kind == "directional" and direction is not None and np.linalg.norm(direction) < 1e-8:
            problems.append(f"{label}: направление не может быть нулевым.")
        if color is not None and np.any((color < 0) | (color > 1)):
            problems.append(f"{label}: каналы цвета должны находиться в диапазоне 0–1.")
        try:
            intensity = float(light.intensity)
            if not np.isfinite(intensity) or not 0 <= intensity <= 4:
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"{label}: яркость должна быть конечным числом от 0 до 4.")

    try:
        ambient = float(scene.ambient_intensity)
        if not np.isfinite(ambient) or not 0 <= ambient <= 2:
            raise ValueError
    except (TypeError, ValueError):
        problems.append("Общее заполнение света должно быть конечным числом от 0 до 2.")
    if scene.shadow_map_size not in (512, 1024, 2048, 4096):
        problems.append("Размер карты теней должен быть 512, 1024, 2048 или 4096.")

    for index, node in enumerate(scene.logic_actions, start=1):
        label = f"Правило логики {index}"
        if not isinstance(node, dict):
            problems.append(f"{label}: ожидается объект с настройками правила.")
            continue
        trigger = node.get("trigger")
        action = node.get("action")
        if not isinstance(trigger, str) or trigger not in LOGIC_TRIGGERS:
            problems.append(f"{label}: неизвестное событие.")
        if not isinstance(action, str) or action not in LOGIC_ACTIONS:
            problems.append(f"{label}: неизвестное действие.")
        target = node.get("target")
        if not isinstance(target, str) or target not in object_names:
            problems.append(f"{label}: целевой объект отсутствует на карте.")
        if trigger == "on_key" and str(node.get("key", "E")).upper() not in KEY_CODES:
            problems.append(f"{label}: клавиша события не поддерживается.")
        if trigger in ("on_timer", "on_player_near"):
            field = "delay" if trigger == "on_timer" else "radius"
            try:
                value = float(node.get(field, 1.0 if field == "delay" else 3.0))
                if not np.isfinite(value) or value < 0 or (
                    field == "radius" and value == 0
                ):
                    raise ValueError
            except (TypeError, ValueError):
                problems.append(f"{label}: параметр «{field}» имеет некорректное значение.")
        if action != "toggle_visibility":
            value = check_vector(node.get("value"), 3, f"{label}, значение")
            if value is not None and action == "set_color":
                if np.any((value < 0) | (value > 1)):
                    problems.append(f"{label}: каналы цвета должны находиться в диапазоне 0–1.")
            if value is not None and action == "set_scale" and np.any(value <= 0):
                problems.append(f"{label}: размер по каждой оси должен быть больше нуля.")

    return problems
