import glfw
import numpy as np


KEY_CODES = {
    "W": glfw.KEY_W,
    "A": glfw.KEY_A,
    "S": glfw.KEY_S,
    "D": glfw.KEY_D,
    "SPACE": glfw.KEY_SPACE,
    "E": glfw.KEY_E,
    "F": glfw.KEY_F,
}


class LogicRuntime:
    def __init__(self):
        self._scene = None
        self._elapsed = 0.0
        self._fired = set()
        self._previous_keys = set()
        self._near_last = set()

    def update(self, scene, delta_time, window, player_position=None):
        if scene is not self._scene:
            self._scene = scene
            self._elapsed = 0.0
            self._fired.clear()
            self._previous_keys.clear()
            self._near_last.clear()

        self._elapsed += delta_time
        pressed = {
            key for key, code in KEY_CODES.items()
            if glfw.get_key(window, code) == glfw.PRESS
        }
        newly_pressed = pressed - self._previous_keys
        near_now = set()

        for index, node in enumerate(scene.logic_actions):
            trigger = node.get("trigger")
            target = next(
                (obj for obj in scene.objects if obj.name == node.get("target")),
                None,
            )
            player_near = (
                trigger == "on_player_near"
                and player_position is not None
                and target is not None
                and np.linalg.norm(
                    np.asarray(player_position) - np.asarray(scene.world_position(target))
                )
                <= max(0.0, float(node.get("radius", 3.0)))
            )
            if player_near:
                near_now.add(index)
            should_fire = (
                trigger == "on_start" and index not in self._fired
            ) or (
                trigger == "on_timer"
                and index not in self._fired
                and self._elapsed >= max(0.0, float(node.get("delay", 1.0)))
            ) or (
                trigger == "on_key"
                and str(node.get("key", "E")).upper() in newly_pressed
            ) or (
                trigger == "on_player_near"
                and player_near
                and index not in self._near_last
            )
            if should_fire:
                self._apply(scene, node)
                if trigger != "on_key":
                    self._fired.add(index)
        self._previous_keys = pressed
        self._near_last = near_now

    @staticmethod
    def _apply(scene, node):
        target = next(
            (obj for obj in scene.objects if obj.name == node.get("target")),
            None,
        )
        if target is None:
            return

        action = node.get("action")
        value = node.get("value", [])
        if action == "set_position" and len(value) == 3:
            target.position = tuple(float(part) for part in value)
        elif action == "add_rotation" and len(value) == 3:
            target.rotation = tuple(
                current + float(offset)
                for current, offset in zip(target.rotation, value)
            )
        elif action == "set_color" and len(value) == 3:
            target.color = tuple(
                max(0.0, min(1.0, float(channel))) for channel in value
            )
        elif action == "set_scale" and len(value) == 3:
            vertices = np.asarray(target.vertices, dtype=np.float64)
            base_dimensions = (
                vertices.max(axis=0) - vertices.min(axis=0)
            ) * target.scale
            target.scale_xyz = tuple(
                float(max(0.01, float(size) / base)) if base > 1e-8 else current
                for size, base, current in zip(value, base_dimensions, target.scale_xyz)
            )
        elif action == "toggle_visibility":
            target.visible = not target.visible
