import math


def capture_camera(camera, time_seconds):
    return {
        "time": float(time_seconds),
        "position": [float(value) for value in camera.position],
        "yaw": float(camera.yaw),
        "pitch": float(camera.pitch),
        "roll": float(camera.roll),
    }


def apply_camera_keyframes(camera, keyframes, time_seconds):
    if not keyframes:
        return

    frames = sorted(keyframes, key=lambda frame: float(frame["time"]))
    if time_seconds <= float(frames[0]["time"]):
        first = frames[0]
        camera.position[:] = first["position"]
        camera.yaw = float(first["yaw"])
        camera.pitch = float(first["pitch"])
        camera.roll = float(first.get("roll", 0.0))
        return
    if time_seconds >= float(frames[-1]["time"]):
        last = frames[-1]
        camera.position[:] = last["position"]
        camera.yaw = float(last["yaw"])
        camera.pitch = float(last["pitch"])
        camera.roll = float(last.get("roll", 0.0))
        return

    for left, right in zip(frames, frames[1:]):
        start = float(left["time"])
        end = float(right["time"])
        if start <= time_seconds <= end:
            amount = (time_seconds - start) / max(end - start, 1e-8)
            camera.position[:] = [
                a + (b - a) * amount
                for a, b in zip(left["position"], right["position"])
            ]
            yaw_delta = (float(right["yaw"]) - float(left["yaw"]) + math.pi) % (2 * math.pi) - math.pi
            camera.yaw = float(left["yaw"]) + yaw_delta * amount
            camera.pitch = float(left["pitch"]) + (
                float(right["pitch"]) - float(left["pitch"])
            ) * amount
            camera.roll = float(left.get("roll", 0.0)) + (
                float(right.get("roll", 0.0)) - float(left.get("roll", 0.0))
            ) * amount
            return
