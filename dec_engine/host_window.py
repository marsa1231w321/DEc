from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from glfw import _GLFWwindowPointerT


class HostWindowBackend(Protocol):
    def embed(
        self, _viewport_window: _GLFWwindowPointerT, _host_handle: int
    ) -> None: ...

    def resize(
        self, _viewport_window: _GLFWwindowPointerT, _width: int, _height: int
    ) -> None: ...


class DetachedWindowBackend:
    def embed(self, _viewport_window: _GLFWwindowPointerT, _host_handle: int) -> None:
        import glfw

        glfw.show_window(_viewport_window)

    def resize(
        self, _viewport_window: _GLFWwindowPointerT, _width: int, _height: int
    ) -> None:
        return


class Win32HostWindowBackend:
    def embed(
        self, _viewport_window: _GLFWwindowPointerT, _host_handle: int
    ) -> None:
        import ctypes
        from ctypes import wintypes

        import glfw

        hwnd = int(glfw.get_win32_window(_viewport_window))
        user32 = ctypes.windll.user32
        user32.SetParent.argtypes = (wintypes.HWND, wintypes.HWND)
        user32.SetParent.restype = wintypes.HWND
        user32.GetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowLongPtrW.argtypes = (
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_ssize_t,
        )
        user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowPos.argtypes = (
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.SetParent(ctypes.c_void_p(hwnd), ctypes.c_void_p(_host_handle))
        style = user32.GetWindowLongPtrW(ctypes.c_void_p(hwnd), -16)
        style = (style & ~0x00C00000 & ~0x00040000) | 0x40000000 | 0x10000000
        user32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), -16, style)
        user32.SetWindowPos(ctypes.c_void_p(hwnd), None, 0, 0, 1, 1, 0x0040 | 0x0010)

    def resize(
        self, _viewport_window: _GLFWwindowPointerT, _width: int, _height: int
    ) -> None:
        import ctypes
        from ctypes import wintypes

        import glfw

        user32 = ctypes.windll.user32
        user32.MoveWindow.argtypes = (
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.BOOL,
        )
        user32.MoveWindow.restype = wintypes.BOOL
        hwnd = int(glfw.get_win32_window(_viewport_window))
        if not user32.MoveWindow(hwnd, 0, 0, _width, _height, True):
            raise ctypes.WinError()


def create_host_window_backend(platform: str | None = None) -> HostWindowBackend:
    if (platform or os.name) == "nt":
        return Win32HostWindowBackend()
    return DetachedWindowBackend()
