import ctypes
import os

import glfw


class ViewportInputState:
    def __init__(self):
        self.active = False
        self.left_button_down = False
        self.right_button_down = False
        self._cursor_position = None

    def clear(self):
        self.active = False
        self.left_button_down = False
        self.right_button_down = False
        self._cursor_position = None

    def reset_cursor(self):
        self._cursor_position = None

    def update(self, x, y, left_button_down, right_button_down):
        button_pressed = (
            left_button_down and not self.left_button_down
        ) or (right_button_down and not self.right_button_down)
        if button_pressed:
            self.active = True
            self.reset_cursor()
        if self.right_button_down and not right_button_down:
            self.reset_cursor()

        if self._cursor_position is None:
            delta_x = 0.0
            delta_y = 0.0
        else:
            delta_x = x - self._cursor_position[0]
            delta_y = y - self._cursor_position[1]
        self._cursor_position = (x, y)
        self.left_button_down = left_button_down
        self.right_button_down = right_button_down
        return delta_x, delta_y


def poll_viewport_input(window, state):
    x, y = glfw.get_cursor_pos(window)
    left_button_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    right_button_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    return state.update(x, y, left_button_down, right_button_down)


_KEY_VIRTUAL_CODES = {
    glfw.KEY_A: 0x41,
    glfw.KEY_D: 0x44,
    glfw.KEY_E: 0x45,
    glfw.KEY_ESCAPE: 0x1B,
    glfw.KEY_Q: 0x51,
    glfw.KEY_S: 0x53,
    glfw.KEY_W: 0x57,
    glfw.KEY_SPACE: 0x20,
    glfw.KEY_LEFT_SHIFT: 0xA0,
}


def key_down(window, key):
    if os.name == "nt" and key in _KEY_VIRTUAL_CODES:
        return bool(ctypes.windll.user32.GetAsyncKeyState(_KEY_VIRTUAL_CODES[key]) & 0x8000)
    return glfw.get_key(window, key) == glfw.PRESS


def mouse_button_down(window, button):
    if os.name == "nt" and button == glfw.MOUSE_BUTTON_RIGHT:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x02) & 0x8000)
    return glfw.get_mouse_button(window, button) == glfw.PRESS


def application_has_focus(window):
    if os.name != "nt":
        return glfw.get_window_attrib(window, glfw.FOCUSED) == glfw.TRUE
    hwnd = glfw.get_win32_window(window)
    user32 = ctypes.windll.user32
    root = user32.GetAncestor(hwnd, 2)
    return bool(root and user32.GetForegroundWindow() == root)
