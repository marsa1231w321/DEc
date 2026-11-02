import ctypes
import os
import time

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


_WIN32_PUMP = None
_PM_REMOVE = 0x0001


def _win32_message_pump():
    """Ленивая настройка PeekMessage/TranslateMessage/DispatchMessage."""

    global _WIN32_PUMP
    if _WIN32_PUMP is None:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.PeekMessageW.argtypes = (
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        )
        user32.PeekMessageW.restype = wintypes.BOOL
        user32.TranslateMessage.argtypes = (ctypes.POINTER(wintypes.MSG),)
        user32.TranslateMessage.restype = wintypes.BOOL
        user32.DispatchMessageW.argtypes = (ctypes.POINTER(wintypes.MSG),)
        user32.DispatchMessageW.restype = ctypes.c_ssize_t
        _WIN32_PUMP = (user32, wintypes)
    return _WIN32_PUMP


def pump_platform_events(platform=None):
    """Обработать события ОС, не вызывая GIL-отпускающую обёртку pyGLFW.

    На Windows встроенный GLFW-viewport перекачивает сообщения через Win32
    SendMessage напрямую: pyGLFW на Python 3.15 может фатально упасть в
    ``glfw.poll_events`` с ``PyEval_RestoreThread``/``thread state is NULL``.
    WndProc GLFW всё равно получает WM_SIZE/WM_MOUSEMOVE/WM_KEYDOWN, поэтому
    размеры, курсор и клавиши продолжают обновляться.
    """

    if (platform or os.name) != "nt":
        glfw.poll_events()
        return
    user32, wintypes = _win32_message_pump()
    message = wintypes.MSG()
    while user32.PeekMessageW(ctypes.byref(message), None, 0, 0, _PM_REMOVE):
        user32.TranslateMessage(ctypes.byref(message))
        user32.DispatchMessageW(ctypes.byref(message))


def wait_platform_events(timeout, platform=None):
    """Ожидать события, не используя ``glfw.wait_events_timeout`` на Windows."""

    if (platform or os.name) != "nt":
        glfw.wait_events_timeout(timeout)
        return
    pump_platform_events(platform)
    time.sleep(max(0.0, float(timeout)))

