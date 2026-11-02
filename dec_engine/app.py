import argparse
import math
import time

import glfw
import moderngl
import numpy as np
from PIL import Image, ImageTk

from .camera import Camera
from .loader import load_model
from .scene import MeshInstance, Scene
from .renderer import Renderer
from .playtest import PlaytestController
from .input_state import (
    ViewportInputState,
    application_has_focus,
    key_down,
    mouse_button_down,
    poll_viewport_input,
)
from .logging_config import configure_logging, get_logger


_logger = get_logger(__name__)


def make_box(size=1.0, color=(0.8, 0.45, 0.28), position=(0.0, 0.0, 0.0)):
    h = size / 2.0
    vertices = [
        # position xyz, normal xyz, uv xy
        (-h, -h, h, 0, 0, 1, 0, 0),
        (h, -h, h, 0, 0, 1, 1, 0),
        (h, h, h, 0, 0, 1, 1, 1),
        (-h, -h, h, 0, 0, 1, 0, 0),
        (h, h, h, 0, 0, 1, 1, 1),
        (-h, h, h, 0, 0, 1, 0, 1),
        (-h, -h, -h, 0, 0, -1, 0, 0),
        (-h, h, -h, 0, 0, -1, 1, 0),
        (h, h, -h, 0, 0, -1, 1, 1),
        (-h, -h, -h, 0, 0, -1, 0, 0),
        (h, h, -h, 0, 0, -1, 1, 1),
        (h, -h, -h, 0, 0, -1, 0, 1),
        (-h, h, -h, 0, 1, 0, 0, 0),
        (-h, h, h, 0, 1, 0, 1, 0),
        (h, h, h, 0, 1, 0, 1, 1),
        (-h, h, -h, 0, 1, 0, 0, 0),
        (h, h, h, 0, 1, 0, 1, 1),
        (h, h, -h, 0, 1, 0, 0, 1),
        (-h, -h, -h, 0, -1, 0, 0, 0),
        (h, -h, -h, 0, -1, 0, 1, 0),
        (h, -h, h, 0, -1, 0, 1, 1),
        (-h, -h, -h, 0, -1, 0, 0, 0),
        (h, -h, h, 0, -1, 0, 1, 1),
        (-h, -h, h, 0, -1, 0, 0, 1),
        (h, -h, -h, 1, 0, 0, 0, 0),
        (h, h, -h, 1, 0, 0, 1, 0),
        (h, h, h, 1, 0, 0, 1, 1),
        (h, -h, -h, 1, 0, 0, 0, 0),
        (h, h, h, 1, 0, 0, 1, 1),
        (h, -h, h, 1, 0, 0, 0, 1),
        (-h, -h, -h, -1, 0, 0, 0, 0),
        (-h, h, h, -1, 0, 0, 1, 0),
        (-h, h, -h, -1, 0, 0, 1, 1),
        (-h, -h, -h, -1, 0, 0, 0, 0),
        (-h, -h, h, -1, 0, 0, 1, 1),
        (-h, h, h, -1, 0, 0, 0, 1),
    ]
    vertices = np.array(vertices, dtype=np.float32)
    mesh = MeshInstance(
        name="box",
        vertices=vertices[:, :3],
        normals=vertices[:, 3:6],
        uvs=vertices[:, 6:8],
        indices=np.arange(len(vertices), dtype=np.int32),
        color=color,
        position=tuple(position),
        scale=1.0,
        primitive="box",
    )
    return mesh


def build_demo_scene(model_path=None):
    scene = Scene()
    scene.add_ground()

    if model_path:
        mesh = load_model(model_path)
        scene.add_mesh(
            MeshInstance(
                name="loaded_model",
                vertices=mesh.vertices,
                normals=mesh.normals,
                uvs=mesh.uvs,
                indices=mesh.indices,
                color=mesh.color,
                position=(5.0, 1.0, 0.0),
                scale=1.0,
                source_path=model_path,
            )
        )

    boxes = [
        (0.0, 1.2, 0.0, (0.83, 0.45, 0.28)),
        (2.4, 0.9, -0.8, (0.5, 0.7, 0.9)),
        (-2.7, 1.0, 1.2, (0.7, 0.68, 0.58)),
        (1.4, 1.8, 2.2, (0.78, 0.79, 0.82)),
    ]
    for index, (x, y, z, color) in enumerate(boxes, start=1):
        mesh = make_box(1.7, color=color, position=(x, y, z))
        mesh.name = f"Box {index}"
        scene.add_mesh(mesh)
    return scene


def run_app(model_path=None):
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(1280, 720, "DEc | 3D-просмотр", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(0)

    ctx = moderngl.create_context()
    renderer = Renderer(ctx)
    camera = Camera(position=(8.0, 6.0, 8.0), yaw=-math.pi / 4, pitch=-0.7)
    scene = build_demo_scene(model_path=model_path)
    from .editor import EditorUI

    editor = EditorUI(scene, camera, viewport_window=window)
    playtest = PlaytestController()

    was_playtesting = False
    escape_was_down = False
    last_output_frame = 0.0
    viewport_input = ViewportInputState()
    previous_application_focus = None

    def handle_window_focus(focused):
        nonlocal was_playtesting
        if not focused:
            viewport_input.clear()
        if not focused and editor.playtest_active:
            editor.playtest_active = False
            was_playtesting = False
            playtest.stop(window)
            editor.playtest_button.configure(text="Играть / Стоп")
            editor.status.set("Тест остановлен: окно потеряло фокус.")

    def handle_playtest_mouse(delta_x, delta_y):
        if editor.playtest_active:
            if key_down(window, glfw.KEY_LEFT_SHIFT):
                camera.roll += delta_x * camera.sensitivity
            else:
                playtest.handle_mouse_motion(
                    camera, delta_x, delta_y, editor.invert_mouse_y.get()
                )
        elif mouse_button_down(window, glfw.MOUSE_BUTTON_RIGHT):
            if key_down(window, glfw.KEY_LEFT_SHIFT):
                camera.roll += delta_x * camera.sensitivity
            else:
                camera.yaw += delta_x * camera.sensitivity
                vertical_direction = -1.0 if editor.invert_mouse_y.get() else 1.0
                camera.pitch = max(
                    -1.5,
                    min(
                        1.5,
                        camera.pitch
                        - delta_y * camera.sensitivity * vertical_direction,
                    ),
                )

    def clear_viewport_input(event):
        if event.widget is not None:
            viewport_input.clear()

    editor.root.bind_all("<Button-1>", clear_viewport_input, add="+")
    editor.root.bind_all("<Button-3>", clear_viewport_input, add="+")

    previous_frame = time.perf_counter()
    try:
        while not glfw.window_should_close(window) and not editor.closing:
            now = time.perf_counter()
            delta_time = min(now - previous_frame, 0.1)
            previous_frame = now
            glfw.poll_events()
            application_focused = application_has_focus(window)
            if application_focused != previous_application_focus:
                handle_window_focus(application_focused)
                previous_application_focus = application_focused

            width, height = glfw.get_framebuffer_size(window)
            if width <= 0 or height <= 0:
                glfw.wait_events_timeout(0.05)
                continue

            if application_focused:
                mouse_delta_x, mouse_delta_y = poll_viewport_input(
                    window, viewport_input
                )
                handle_playtest_mouse(mouse_delta_x, mouse_delta_y)

            editor.update(delta_time, window)
            if editor.closing:
                break
            if editor.playtest_active != was_playtesting:
                if editor.playtest_active:
                    viewport_input.reset_cursor()
                    playtest.start(editor.scene, camera, window)
                    editor.status.set(playtest.capture_state)
                else:
                    playtest.stop(window)
                    viewport_input.reset_cursor()
                    editor.root.focus_force()
                was_playtesting = editor.playtest_active

            escape_down = key_down(window, glfw.KEY_ESCAPE)
            if escape_down and not escape_was_down:
                if editor.playtest_active:
                    editor.toggle_playtest()
                else:
                    glfw.set_window_should_close(window, True)
            escape_was_down = escape_down

            if editor.playtest_active:
                capture_status = playtest.update(
                    editor.scene, camera, delta_time, window
                )
                if capture_status:
                    editor.status.set(capture_status)
                if playtest.audio_error:
                    editor.status.set(f"Ошибка аудио: {playtest.audio_error}")
                if not playtest.active:
                    editor.playtest_active = False
                    editor.playtest_button.configure(text="Играть / Стоп")
                    was_playtesting = False
                    editor.root.focus_force()
                    editor.status.set("Тест остановлен: окно потеряло фокус.")
            else:
                move = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                if viewport_input.active:
                    forward = camera.forward()
                    right = camera.right()
                    if key_down(window, glfw.KEY_W):
                        move += forward
                    if key_down(window, glfw.KEY_S):
                        move -= forward
                    if key_down(window, glfw.KEY_D):
                        move += right
                    if key_down(window, glfw.KEY_A):
                        move -= right
                if viewport_input.active and key_down(window, glfw.KEY_SPACE):
                    move += np.array([0.0, 1.0, 0.0], dtype=np.float32)
                if viewport_input.active and key_down(window, glfw.KEY_LEFT_SHIFT):
                    move += np.array([0.0, -1.0, 0.0], dtype=np.float32)
                if np.linalg.norm(move) > 0:
                    move /= np.linalg.norm(move)
                    camera.position += move * camera.speed * delta_time
                if viewport_input.active and key_down(window, glfw.KEY_Q):
                    camera.roll += 1.5 * delta_time
                if viewport_input.active and key_down(window, glfw.KEY_E):
                    camera.roll -= 1.5 * delta_time
            editor._update_output_camera(delta_time)

            renderer.render(
                editor.scene,
                camera,
                width,
                height,
                show_editor_gizmos=not editor.playtest_active,
                preview_object=editor.preview_object
                if not editor.playtest_active
                else None,
            )
            if editor.output_visible and now - last_output_frame >= 0.05:
                last_output_frame = now
                output_canvas = editor.active_output_canvas()
                if output_canvas is not None and output_canvas.winfo_exists():
                    output_width = min(max(output_canvas.winfo_width(), 240), 1280)
                    output_height = min(max(output_canvas.winfo_height(), 135), 720)
                    output_framebuffer, output_texture = renderer.render_target(
                        output_width, output_height
                    )
                    output_camera = editor.output_camera_for_render(camera)
                    renderer.render(
                        editor.scene,
                        output_camera,
                        output_width,
                        output_height,
                        show_editor_gizmos=False,
                        framebuffer=output_framebuffer,
                    )
                    output_image = Image.frombytes(
                        "RGBA",
                        (output_width, output_height),
                        output_texture.read(alignment=1),
                    ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    editor._output_photo = ImageTk.PhotoImage(output_image)
                    output_canvas.delete("all")
                    output_canvas.create_image(
                        output_canvas.winfo_width() / 2,
                        output_canvas.winfo_height() / 2,
                        image=editor._output_photo,
                    )
                    ctx.screen.use()
            glfw.swap_buffers(window)
            time.sleep(max(0.0, 1.0 / 120.0 - (time.perf_counter() - now)))
    finally:
        try:
            playtest.stop(window)
        finally:
            try:
                editor.close()
            finally:
                try:
                    renderer.release()
                finally:
                    glfw.destroy_window(window)
                    glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description="DEc 3D engine and level editor")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=None,
        help="Developer log level (overrides DEC_LOG_LEVEL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a model file supported by trimesh",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    _logger.info(
        "DEc editor started",
        extra={"dec_context": {"operation": "startup"}},
    )
    run_app(model_path=args.model)
    _logger.info(
        "DEc editor stopped",
        extra={"dec_context": {"operation": "shutdown"}},
    )


if __name__ == "__main__":
    main()
