import math


def _interpolate(left, right, amount, mode, previous=None, following=None):
    if mode == "smooth":
        amount = amount * amount * (3.0 - 2.0 * amount)
    if mode == "catmull_rom" and previous is not None and following is not None:
        return 0.5 * (
            2.0 * left
            + (-previous + right) * amount
            + (2.0 * previous - 5.0 * left + 4.0 * right - following) * amount**2
            + (-previous + 3.0 * left - 3.0 * right + following) * amount**3
        )
    return left + (right - left) * amount


def _unwrap_angle(value, reference):
    return reference + (value - reference + math.pi) % (2.0 * math.pi) - math.pi


def evaluate_track(track, time_seconds):
    keys = sorted(track.get("keys", []), key=lambda key: float(key["time"]))
    if not keys:
        return None
    if len(keys) == 1 or time_seconds <= float(keys[0]["time"]):
        return keys[0]["values"].copy()
    if time_seconds >= float(keys[-1]["time"]):
        return keys[-1]["values"].copy()

    for index, (left_key, right_key) in enumerate(zip(keys, keys[1:])):
        start = float(left_key["time"])
        end = float(right_key["time"])
        if start <= time_seconds <= end:
            amount = (time_seconds - start) / max(end - start, 1e-8)
            mode = track.get("interpolation", "linear")
            previous = keys[max(0, index - 1)]["values"]
            following = keys[min(len(keys) - 1, index + 2)]["values"]
            result = {}
            for name, left_value in left_key["values"].items():
                if name not in right_key["values"]:
                    continue
                right_value = right_key["values"][name]
                if isinstance(left_value, list):
                    left_vector = left_value
                    right_vector = right_value
                    prev_vector = previous.get(name, left_vector)
                    next_vector = following.get(name, right_vector)
                    result[name] = [
                        float(_interpolate(
                            float(a), float(b), amount, mode,
                            float(prev_vector[axis]), float(next_vector[axis]),
                        ))
                        for axis, (a, b) in enumerate(zip(left_vector, right_vector))
                    ]
                elif isinstance(left_value, (int, float)):
                    left_number = float(left_value)
                    right_number = float(right_value)
                    previous_number = float(previous.get(name, left_value))
                    following_number = float(following.get(name, right_value))
                    if name == "yaw":
                        right_number = _unwrap_angle(right_number, left_number)
                        previous_number = _unwrap_angle(previous_number, left_number)
                        following_number = _unwrap_angle(following_number, right_number)
                    result[name] = float(_interpolate(
                        left_number,
                        right_number,
                        amount,
                        mode,
                        previous_number,
                        following_number,
                    ))
                else:
                    result[name] = left_value if amount < 0.5 else right_value
            return result
    return keys[-1]["values"].copy()


def capture_values(target_type, target):
    if target_type == "camera":
        return {
            "position": [float(value) for value in target.position],
            "yaw": float(target.yaw),
            "pitch": float(target.pitch),
            "roll": float(target.roll),
        }
    if target_type == "object":
        return {
            "position": list(target.position),
            "rotation": list(target.rotation),
            "scale": float(target.scale),
            "scale_xyz": list(target.scale_xyz),
            "color": list(target.color),
        }
    if target_type == "light":
        return {
            "position": list(target.position),
            "direction": list(target.direction),
            "color": list(target.color),
            "intensity": float(target.intensity),
        }
    if target_type == "bone":
        return {"rotation": [float(value) for value in target["rotation"]]}
    raise ValueError(f"Unknown animation target type: {target_type}")


def apply_animation_tracks(scene, camera, time_seconds):
    for track in scene.animation_tracks:
        values = evaluate_track(track, time_seconds)
        if values is None:
            continue
        target_type = track.get("target_type")
        target_name = track.get("target")
        if target_type == "camera":
            target = camera
        elif target_type == "object":
            target = next((obj for obj in scene.objects if obj.name == target_name), None)
        elif target_type == "light":
            target = next((light for light in scene.lights if light.name == target_name), None)
        elif target_type == "bone":
            object_name, separator, bone_name = str(target_name).rpartition("::")
            obj = next((item for item in scene.objects if item.name == object_name), None)
            target = (
                next(
                    (bone for bone in obj.rig["bones"] if bone["name"] == bone_name),
                    None,
                )
                if separator and obj is not None and obj.rig else None
            )
        else:
            continue
        if target is None:
            continue
        for field in ("position", "rotation", "direction", "color", "scale_xyz"):
            if field in values and hasattr(target, field):
                vector = tuple(float(value) for value in values[field])
                if field == "direction" and sum(value * value for value in vector) < 1e-8:
                    continue
                setattr(target, field, vector)
        for field in ("yaw", "pitch", "roll", "scale", "intensity"):
            if field in values and hasattr(target, field):
                value = float(values[field])
                if field == "scale":
                    value = max(value, 0.001)
                setattr(target, field, value)
        if target_type == "bone" and "rotation" in values:
            target["rotation"] = [float(value) for value in values["rotation"]]


def sample_track_curve(track, field="position", axis=0, samples=80):
    keys = track.get("keys", [])
    if len(keys) < 2:
        return []
    start = min(float(key["time"]) for key in keys)
    end = max(float(key["time"]) for key in keys)
    if end <= start:
        return []
    result = []
    for time_seconds in (
            start + (end - start) * index / (samples - 1)
            for index in range(samples)
    ):
        value = evaluate_track(track, time_seconds).get(field, [0.0])
        if isinstance(value, list):
            value = value[axis]
        result.append((time_seconds, value))
    return result
