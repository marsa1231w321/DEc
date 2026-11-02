import ctypes
import unittest
from unittest.mock import patch

import glfw

from dec_engine.host_window import (
    DetachedWindowBackend,
    Win32HostWindowBackend,
    create_host_window_backend,
)


class HostWindowBackendTests(unittest.TestCase):
    def test_non_windows_backend_keeps_viewport_detached(self):
        backend = create_host_window_backend(platform="posix")

        self.assertIsInstance(backend, DetachedWindowBackend)
        viewport = ctypes.POINTER(glfw._GLFWwindow)()
        with patch("glfw.show_window") as show_window:
            backend.embed(viewport, 100)
        show_window.assert_called_once_with(viewport)
        backend.resize(viewport, 640, 480)

    def test_windows_backend_is_selected_without_loading_win32_apis(self):
        backend = create_host_window_backend(platform="nt")

        self.assertIsInstance(backend, Win32HostWindowBackend)


if __name__ == "__main__":
    unittest.main()
