import math
import os
import time

import glfw
import numpy as np

from .camera import Camera
from .audio import AudioPlayer
from .input_state import application_has_focus, key_down
from .scene import Scene


class PlaytestController:
    def __init__(self):
        self.active = False
        self.eye_height = 1.65
        self.player_radius = 0.32
        self.vertical_velocity = 0.0
        self.speed = 5.0
        self.jump_speed = 5.5
        self.sensitivity = 0.0025
        self._previous_mouse = None
        self._space_was_down = False
        self._grounded = False
        self._active_audio_path = None
        self.audio_player = AudioPlayer()
        self.audio_error = None
        self._last_mouse_check = 0.0
        self.mouse_sensitivity = 0.0025
        self.capture_state = "Мышь свободна"

    def start(self, scene: Scene, camera: Camera, window=None):
        self.active = True
        if window is not None:
            glfw.focus_window(window)
            if os.name == "nt":
                import ctypes

                user32 = ctypes.windll.user32
                hwnd = glfw.get_win32_window(window)
                root_hwnd = user32.GetAncestor(hwnd, 2)
                user32.SetForegroundWindow(root_hwnd or hwnd)
                user32.SetFocus(hwnd)
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            if glfw.raw_mouse_motion_supported():
                glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)
            self.capture_state = "Тест: мышь захвачена; WASD — движение, Space — прыжок, Esc — выход"
        else:
            self.capture_state = "Тест запущен без окна ввода"
        self.vertical_velocity = 0.0
        self._grounded = False
        self._space_was_down = False
        spawn = next((obj for obj in scene.objects if obj.primitive == "spawn"), None)
        if spawn is not None:
            spawn_position = scene.world_position(spawn)
            camera.position[:] = (
                spawn_position[0],
                spawn_position[1] + self.eye_height,
                spawn_position[2],
            )
        else:
            camera.position[:] = (0.0, self.eye_height, 6.0)
        camera.yaw = 0.0
        camera.pitch = 0.0
        camera.roll = 0.0
        self._previous_mouse = None

    def stop(self, window=None):
        self.active = False
        self._previous_mouse = None
        self._stop_audio()
        self.capture_state = "Мышь освобождена"
        if window is not None:
            try:
                if glfw.raw_mouse_motion_supported():
                    glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.FALSE)
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
            finally:
                if os.name == "nt":
                    import ctypes

                    ctypes.windll.user32.ClipCursor(None)

    @staticmethod
    def _object_bounds(scene, obj):
        vertices = np.asarray(obj.vertices, dtype=np.float32)
        homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float32)))
        world = (scene.world_matrix(obj) @ homogeneous.T).T[:, :3]
        return world.min(axis=0), world.max(axis=0)

    def _collides(self, scene, x, feet_y, z):
        player_min = (x - self.player_radius, feet_y, z - self.player_radius)
        player_max = (x + self.player_radius, feet_y + self.eye_height, z + self.player_radius)
        for obj in scene.objects:
            if not obj.visible or obj.primitive in (
                "ground", "terrain", "spawn", "sound_source", "logic_trigger", "npc",
            ):
                continue
            minimum, maximum = self._object_bounds(scene, obj)
            if all(
                player_min[axis] < maximum[axis] and player_max[axis] > minimum[axis]
                for axis in range(3)
            ):
                return True
        return False

    def update(self, scene: Scene, camera: Camera, delta_time: float, window):
        if not self.active:
            return
        if time.monotonic() - self._last_mouse_check >= 0.5:
            self._last_mouse_check = time.monotonic()
            focused = application_has_focus(window)
            viewport_focused = glfw.get_window_attrib(window, glfw.FOCUSED) == glfw.TRUE
            cursor_mode = glfw.get_input_mode(window, glfw.CURSOR)
            if not focused:
                self.stop(window)
                return
            if not viewport_focused:
                glfw.focus_window(window)
                if os.name == "nt":
                    import ctypes

                    ctypes.windll.user32.SetFocus(glfw.get_win32_window(window))
                self.capture_state = "Фокус игрового вида восстановлен"
                return "Фокус игрового вида восстановлен."
            if cursor_mode != glfw.CURSOR_DISABLED:
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                if glfw.raw_mouse_motion_supported():
                    glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)
                self.capture_state = "Захват мыши восстановлен"
                return "Захват мыши восстановлен."

        forward = np.array((math.sin(camera.yaw), 0.0, -math.cos(camera.yaw)), dtype=np.float32)
        right = np.array((math.cos(camera.yaw), 0.0, math.sin(camera.yaw)), dtype=np.float32)
        move = np.zeros(3, dtype=np.float32)
        for key, direction in (
            (glfw.KEY_W, forward),
            (glfw.KEY_S, -forward),
            (glfw.KEY_D, right),
            (glfw.KEY_A, -right),
        ):
            if key_down(window, key):
                move += direction
        if np.linalg.norm(move) > 0:
            move /= np.linalg.norm(move)
        if key_down(window, glfw.KEY_Q):
            camera.roll += 1.5 * delta_time
        if key_down(window, glfw.KEY_E):
            camera.roll -= 1.5 * delta_time

        feet_y = float(camera.position[1] - self.eye_height)
        step = self.speed * delta_time
        proposed_x = float(camera.position[0] + move[0] * step)
        if not self._collides(scene, proposed_x, feet_y, float(camera.position[2])):
            camera.position[0] = proposed_x
        proposed_z = float(camera.position[2] + move[2] * step)
        if not self._collides(scene, float(camera.position[0]), feet_y, proposed_z):
            camera.position[2] = proposed_z

        space_down = key_down(window, glfw.KEY_SPACE)
        if space_down and not self._space_was_down and self._grounded:
            self.vertical_velocity = self.jump_speed
            self._grounded = False
        self._space_was_down = space_down

        floor_y = 0.0
        for obj in scene.objects:
            if not obj.visible or obj.primitive in ("spawn", "sound_source", "logic_trigger"):
                continue
            if obj.primitive == "terrain":
                terrain_height = self._terrain_height(
                    scene, obj, float(camera.position[0]), float(camera.position[2])
                )
                if terrain_height is not None and terrain_height <= feet_y + 0.2:
                    floor_y = max(floor_y, terrain_height)
                continue
            minimum, maximum = self._object_bounds(scene, obj)
            overlaps_xz = (
                camera.position[0] + self.player_radius > minimum[0]
                and camera.position[0] - self.player_radius < maximum[0]
                and camera.position[2] + self.player_radius > minimum[2]
                and camera.position[2] - self.player_radius < maximum[2]
            )
            if overlaps_xz and maximum[1] <= feet_y + 0.05:
                floor_y = max(floor_y, float(maximum[1]))

        self.vertical_velocity -= 9.81 * delta_time
        next_feet_y = feet_y + self.vertical_velocity * delta_time
        if next_feet_y <= floor_y:
            next_feet_y = floor_y
            self.vertical_velocity = 0.0
            self._grounded = True
        else:
            self._grounded = False
        camera.position[1] = next_feet_y + self.eye_height
        self._update_npcs(scene, camera, delta_time)
        self._update_audio(scene, camera)

    def handle_mouse_motion(self, camera, delta_x, delta_y, invert_y=False):
        if not self.active:
            return
        camera.yaw += delta_x * self.mouse_sensitivity
        camera.pitch = max(
            -1.5,
            min(1.5, camera.pitch - delta_y * self.mouse_sensitivity * (-1 if invert_y else 1)),
        )

    @staticmethod
    def _terrain_height(scene, obj, x, z):
        vertices = np.asarray(obj.vertices, dtype=np.float32)
        homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float32)))
        vertices = (scene.world_matrix(obj) @ homogeneous.T).T[:, :3]
        if (
            x < vertices[:, 0].min() or x > vertices[:, 0].max()
            or z < vertices[:, 2].min() or z > vertices[:, 2].max()
        ):
            return None
        distances = (vertices[:, 0] - x) ** 2 + (vertices[:, 2] - z) ** 2
        return float(vertices[int(np.argmin(distances)), 1])

    def _update_npcs(self, scene, camera, delta_time):
        for obj in scene.objects:
            if obj.primitive != "npc" or obj.npc_behavior != "follow":
                continue
            direction = np.asarray(camera.position, dtype=np.float32) - np.asarray(
                scene.world_position(obj), dtype=np.float32
            )
            direction[1] = 0.0
            distance = float(np.linalg.norm(direction))
            if distance <= 1.1:
                continue
            direction /= distance
            step = min(1.8 * delta_time, distance - 1.1)
            world_position = scene.world_position(obj)
            next_x = float(world_position[0] + direction[0] * step)
            next_z = float(world_position[2] + direction[2] * step)
            feet_y = float(world_position[1])
            if not self._collides(scene, next_x, feet_y, next_z):
                scene.set_world_position(obj, (next_x, feet_y, next_z))

    def _update_audio(self, scene, camera):
        candidates = [
            obj for obj in scene.objects
            if obj.primitive == "sound_source"
            and obj.audio_path
            and np.linalg.norm(
                np.asarray(camera.position) - np.asarray(scene.world_position(obj))
            )
            <= obj.sound_radius
        ]
        source = min(
            candidates,
            key=lambda obj: float(
                np.linalg.norm(
                    np.asarray(camera.position) - np.asarray(scene.world_position(obj))
                )
            ),
            default=None,
        )
        audio_path = source.audio_path if source is not None else None
        if audio_path == self._active_audio_path:
            try:
                self.audio_player.update()
            except RuntimeError as error:
                self.audio_error = str(error)
            return
        self.audio_error = None
        self._stop_audio()
        if source is not None:
            try:
                self.audio_player.play(audio_path, loop=source.audio_loop)
            except (OSError, ValueError, RuntimeError) as error:
                self.audio_error = str(error)
            else:
                self._active_audio_path = audio_path

    def _stop_audio(self):
        if self._active_audio_path is None:
            return
        self.audio_player.stop()
        self._active_audio_path = None
