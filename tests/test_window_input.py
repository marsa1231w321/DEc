import ctypes
import unittest
import tkinter as tk
from unittest.mock import patch

import glfw

from dec_engine.editor import EditorUI
from dec_engine.input_state import (
    ViewportInputState,
    poll_viewport_input,
    pump_platform_events,
    wait_platform_events,
)


class ViewportInputTests(unittest.TestCase):
    def test_mouse_drag_activates_viewport_and_reports_relative_motion(self):
        state = ViewportInputState()

        self.assertEqual(state.update(20.0, 30.0, True, False), (0.0, 0.0))
        self.assertTrue(state.active)
        self.assertEqual(state.update(25.0, 34.0, True, False), (5.0, 4.0))

    def test_releasing_right_button_resets_cursor_baseline(self):
        state = ViewportInputState()
        state.update(10.0, 10.0, False, True)
        self.assertEqual(state.update(15.0, 13.0, False, True), (5.0, 3.0))

        self.assertEqual(state.update(500.0, 400.0, False, False), (0.0, 0.0))
        self.assertEqual(state.update(502.0, 401.0, False, False), (2.0, 1.0))

    def test_clear_deactivates_viewport_and_resets_mouse_state(self):
        state = ViewportInputState()
        state.update(10.0, 10.0, True, False)
        state.clear()

        self.assertFalse(state.active)
        self.assertEqual(state.update(300.0, 200.0, False, False), (0.0, 0.0))

    def test_input_poll_reads_glfw_state_without_registered_callbacks(self):
        state = ViewportInputState()
        with (
            patch(
                "dec_engine.input_state.glfw.get_cursor_pos", return_value=(8.0, 9.0)
            ),
            patch(
                "dec_engine.input_state.glfw.get_mouse_button",
                side_effect=(glfw.PRESS, glfw.RELEASE),
            ),
        ):
            delta = poll_viewport_input(object(), state)

        self.assertEqual(delta, (0.0, 0.0))
        self.assertTrue(state.active)
        self.assertTrue(state.left_button_down)

    def test_viewport_resize_events_are_coalesced(self):
        class FakeRoot(tk.Tk):
            def __init__(self):
                self.scheduled = []

            def after(self, delay, callback):
                self.scheduled.append((delay, callback))
                return "resize-id"

        fake_root = FakeRoot()
        editor = object.__new__(EditorUI)
        editor.viewport_window = object()
        editor.closing = False
        editor._viewport_resize_after_id = None
        editor.root = fake_root

        with patch("dec_engine.editor.os.name", "nt"):
            editor._resize_viewport()
            editor._resize_viewport()

        self.assertEqual(len(fake_root.scheduled), 1)
        self.assertEqual(fake_root.scheduled[0][0], 16)
        self.assertEqual(editor._viewport_resize_after_id, "resize-id")


class PlatformEventPumpTests(unittest.TestCase):
    def test_non_windows_pump_delegates_to_glfw(self):
        with patch("dec_engine.input_state.glfw.poll_events") as poll:
            pump_platform_events("posix")

        poll.assert_called_once_with()

    def test_windows_pump_dispatches_win32_messages(self):
        calls = []

        class FakeUser32:
            def __init__(self):
                self.results = iter((1, 0))

            def PeekMessageW(self, _message, _hwnd, _low, _high, _remove):
                calls.append("peek")
                return next(self.results)

            def TranslateMessage(self, _message):
                calls.append("translate")
                return 1

            def DispatchMessageW(self, _message):
                calls.append("dispatch")
                return 0

        class FakeWintypes:
            def MSG(self):
                return ctypes.c_int()

        with patch(
            "dec_engine.input_state._win32_message_pump",
            return_value=(FakeUser32(), FakeWintypes()),
        ):
            pump_platform_events("nt")

        self.assertEqual(calls, ["peek", "translate", "dispatch", "peek"])

    def test_non_windows_wait_delegates_to_glfw(self):
        with patch("dec_engine.input_state.glfw.wait_events_timeout") as wait:
            wait_platform_events(0.05, "posix")

        wait.assert_called_once_with(0.05)

    def test_windows_wait_pumps_then_sleeps(self):
        with (
            patch("dec_engine.input_state.pump_platform_events") as pump,
            patch("dec_engine.input_state.time.sleep") as sleep,
        ):
            wait_platform_events(0.05, "nt")

        pump.assert_called_once_with("nt")
        sleep.assert_called_once_with(0.05)


if __name__ == "__main__":
    unittest.main()
