import ctypes
import os
import pickle
import re
import shlex
import stat
import tempfile
import tkinter as tk
import uuid
import zipfile
from ctypes import wintypes
from functools import wraps
from pathlib import PurePosixPath
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .animation import apply_animation_tracks, capture_values, sample_track_curve
from .audio import AudioPlayer
from .asset_pack import (
    ANIMATION_EXTENSIONS,
    MODEL_EXTENSIONS,
    SOUND_EXTENSIONS,
    TEXTURE_EXTENSIONS,
    create_asset_pack,
    extract_asset,
    list_asset_pack,
)
from .camera import Camera
from .commands import CommandStack, build_scene_delta
from .events import ResourceChanged, SelectionChanged
from .loader import load_model
from .logic import KEY_CODES, LogicRuntime
from .project import load_project, save_project
from .scene import Light, MeshInstance, Scene
from .scene_observers import SceneEventConsumers
from .timeline import apply_camera_keyframes, capture_camera


def _is_float(value):
    try:
        float(value)
    except ValueError:
        return False
    return True


def _undoable(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self._history_ready or self._history_suspended:
            return method(self, *args, **kwargs)
        before = self._capture_history_state()
        result = method(self, *args, **kwargs)
        self._record_history_change(before, method.__name__)
        return result

    return wrapped


class EditorUI:
    def __init__(self, scene: Scene, camera: Camera, viewport_window=None):
        self.scene = scene
        self.camera = camera
        self.output_camera = Camera(
            position=tuple(float(value) for value in camera.position),
            yaw=camera.yaw,
            pitch=camera.pitch,
        )
        self.output_visible = False
        self._output_photo: ImageTk.PhotoImage | None = None
        self._output_drag = None
        self.output_window = None
        self.output_window_canvas = None
        self.curve_window = None
        self.detached_curve_canvas = None
        self._output_keys = set()
        self._output_follow_key = None
        self._output_follow_target = None
        self.viewport_window = viewport_window
        self._viewport_resize_after_id = None
        self._viewport_size = None
        self.logic = LogicRuntime()
        self.playhead = 0.0
        self.timeline_playing = False
        self.playtest_active = False
        self.project_path = None
        self.grid_zoom = 1.0
        self.grid_drag = None
        self.asset_pack_path = None
        self.asset_entries = []
        self.asset_cache = tempfile.TemporaryDirectory(prefix="dec-assets-")
        self.preview_object = None
        self.audio_preview = None
        self.audio_player = AudioPlayer()
        self.closing = False
        self._history_ready = False
        self._history_suspended = False
        self._history_limit = 30
        self._history_memory_limit = 128 * 1024 * 1024
        self._history_gesture_before = None
        self._dirty = False
        self._scene_revision = 0
        self._autosaved_revision = -1
        self._recovery_path = self._get_recovery_path()
        self._recovered_from_autosave = False
        self._recovery_is_current = False

        self.root = tk.Tk()
        self.root.title("DEc | Редактор")
        self.root.geometry("1420x900")
        self.root.minsize(960, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        self.root.bind_all(
            "<Control-z>",
            lambda event: self._handle_history_shortcut(event, self.undo),
            add="+",
        )
        self.root.bind_all(
            "<Control-y>",
            lambda event: self._handle_history_shortcut(event, self.redo),
            add="+",
        )
        self.asset_filter = tk.StringVar(master=self.root, value="All")
        self.asset_search = tk.StringVar(master=self.root, value="")
        self.layout_profile = tk.StringVar(master=self.root, value="DEc Studio")

        self.status = tk.StringVar(value="Готово")
        self._offer_recovery()
        self.command_stack = CommandStack.from_scene(
            self.scene,
            max_history=self._history_limit,
            max_memory_bytes=self._history_memory_limit,
        )
        if self._recovered_from_autosave:
            self.status.set("Автокопия восстановлена. Сохраните карту в новый файл.")
        self._build_ui()
        self._bind_scene_event_consumers()
        if self._recovered_from_autosave:
            self.scene.notify_changed("recovery_loaded")
        self._refresh_objects()
        self._refresh_lighting_fields()
        self._refresh_logic()
        self._set_animation_kind("camera")
        self._refresh_animation_targets()
        self._refresh_keyframes()
        self._history_ready = True
        self._autosave_after_id = self.root.after(30_000, self._autosave_tick)

    @_undoable
    def _set_track_interpolation(self):
        track = self._current_animation_track()
        if track is not None:
            track["interpolation"] = self._interpolation_mode()
            self._draw_animation_curve()

    def _capture_history_state(self):
        state = self.scene.__dict__.copy()
        state.pop("command_history", None)
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def _record_history_change(self, before: bytes, label: str) -> bool:
        after = self._capture_history_state()
        if before == after:
            return False
        command = build_scene_delta(before, after, label)
        if not command.commands:
            return False
        if not self.command_stack.record_applied(command):
            self.status.set(
                "Изменение выполнено, но слишком велико для истории отмены."
            )
        self._refresh_history_buttons()
        return True

    def _bind_scene_event_consumers(self):
        previous = getattr(self, "_scene_event_consumers", None)
        if previous is not None:
            previous.close()
        self.command_stack = CommandStack.from_scene(
            self.scene,
            max_history=self._history_limit,
            max_memory_bytes=self._history_memory_limit,
        )
        self._scene_event_consumers = SceneEventConsumers(
            self.scene, on_change=self._on_scene_event_change
        )

    def _on_scene_event_change(
        self, _revision: int, _problems: tuple[str, ...]
    ) -> None:
        self._dirty = True
        self._scene_revision += 1

    def _refresh_after_history_change(self, description):
        selected_object = self._selected_index()
        selected_light = self._selected_light_index()
        self._history_suspended = True
        try:
            object_index = (
                min(selected_object, len(self.scene.objects) - 1)
                if selected_object is not None and self.scene.objects
                else None
            )
            light_index = (
                min(selected_light, len(self.scene.lights) - 1)
                if selected_light is not None and self.scene.lights
                else None
            )
            self._refresh_objects(select=object_index)
            self._refresh_lighting_fields()
            if light_index is not None:
                self.light_list.selection_clear(0, tk.END)
                self.light_list.selection_set(light_index)
                self.light_list.event_generate("<<ListboxSelect>>")
            self._refresh_logic()
            self._refresh_keyframes()
        finally:
            self._history_suspended = False
        self.status.set(description)

    def undo(self):
        if not self.command_stack.undo():
            self.status.set("Нет действий для отмены.")
            return
        self._refresh_after_history_change("Последнее действие отменено.")
        self._refresh_history_buttons()

    def redo(self):
        if not self.command_stack.redo():
            self.status.set("Нет действий для повтора.")
            return
        self._refresh_after_history_change("Действие повторено.")
        self._refresh_history_buttons()

    def _clear_history(self):
        self.command_stack.clear()
        self._history_gesture_before = None
        self._refresh_history_buttons()

    @staticmethod
    def _get_recovery_path():
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = os.path.join(os.path.expanduser("~"), ".local", "share")
        return os.path.join(base, "DEc", "recovery", "last_session.DE")

    def _offer_recovery(self):
        if not os.path.isfile(self._recovery_path):
            return
        choice = messagebox.askyesnocancel(
            "Восстановление DEc",
            "Найдена автокопия предыдущей сессии.\n"
            "Да — восстановить её; Нет — удалить автокопию; Отмена — оставить её на потом.",
            parent=self.root,
        )
        if choice is None:
            return
        if not choice:
            try:
                os.remove(self._recovery_path)
                self._recovery_is_current = False
            except OSError as error:
                messagebox.showerror(
                    "Не удалось удалить автокопию", str(error), parent=self.root
                )
            return
        try:
            self.scene = load_project(self._recovery_path)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Не удалось восстановить автокопию", str(error), parent=self.root
            )
            return
        self._recovered_from_autosave = True
        self._recovery_is_current = True

    def _write_recovery_copy(self):
        os.makedirs(os.path.dirname(self._recovery_path), exist_ok=True)
        save_project(self._recovery_path, self.scene)
        self._autosaved_revision = self._scene_revision
        self._recovery_is_current = True

    def _remove_recovery_copy(self):
        if not self._recovery_is_current:
            return
        if not os.path.isfile(self._recovery_path):
            self._recovery_is_current = False
            return
        try:
            os.remove(self._recovery_path)
            self._recovery_is_current = False
        except OSError as error:
            self.status.set(
                f"Карта сохранена, но автокопию удалить не удалось: {error}"
            )

    def _autosave_tick(self):
        if self.closing:
            return
        if (
            self._dirty
            and self._autosaved_revision != self._scene_revision
            and not self.playtest_active
            and not self.timeline_playing
        ):
            try:
                self._write_recovery_copy()
                self.status.set("Автокопия изменений сохранена.")
            except (OSError, ValueError) as error:
                self.status.set(f"Ошибка автосохранения: {error}")
        self._autosave_after_id = self.root.after(30_000, self._autosave_tick)

    @staticmethod
    def _validation_summary(problems):
        visible = problems[:20]
        summary = "\n".join(f"- {problem}" for problem in visible)
        if len(problems) > len(visible):
            summary += f"\n... и ещё {len(problems) - len(visible)} проблем."
        return summary

    def _refresh_history_buttons(self):
        if not hasattr(self, "undo_button") or not hasattr(self, "command_stack"):
            return
        self.undo_button.configure(
            state="normal" if self.command_stack.can_undo else "disabled"
        )
        self.redo_button.configure(
            state="normal" if self.command_stack.can_redo else "disabled"
        )

    def _handle_history_shortcut(self, event, action):
        focused = self.root.focus_get()
        if isinstance(
            focused,
            (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Spinbox, ttk.Combobox),
        ):
            return None
        action()
        return "break"

    def _begin_history_gesture(self):
        if self._history_ready and not self._history_suspended:
            self._history_gesture_before = self._capture_history_state()

    def _finish_history_gesture(self):
        before = self._history_gesture_before
        self._history_gesture_before = None
        if before is not None:
            self._record_history_change(before)

    def _build_ui(self):
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("DEc.TFrame", background="#202631")
        style.configure("DEc.TLabel", background="#202631", foreground="#e7edf5")

        header = ttk.Frame(self.root, padding=(8, 6))
        header.pack(fill="x")
        ttk.Label(header, text="DEc", font=("Segoe UI", 13, "bold")).pack(
            side="left", padx=(2, 12)
        )
        ttk.Label(header, text="Шаблон интерфейса:").pack(side="left")
        self.undo_button = ttk.Button(
            header, text="Отменить", command=self.undo, state="disabled"
        )
        self.undo_button.pack(side="left", padx=(8, 2))
        self.redo_button = ttk.Button(
            header, text="Повторить", command=self.redo, state="disabled"
        )
        self.redo_button.pack(side="left", padx=2)
        profile_picker = ttk.Combobox(
            header,
            textvariable=self.layout_profile,
            values=("DEc Studio", "DEc Map", "DEc Motion"),
            state="readonly",
            width=12,
        )
        profile_picker.pack(side="left", padx=5)
        profile_picker.bind(
            "<<ComboboxSelected>>", lambda _event: self.apply_layout_profile()
        )
        ttk.Label(
            header,
            text="DEc Studio — все инструменты; DEc Map — карты; DEc Motion — камера и анимация. Панели можно менять по ширине.",
        ).pack(side="left", padx=8)
        ttk.Button(header, text="Открыть пакет .DE", command=self.open_asset_pack).pack(
            side="right", padx=3
        )
        ttk.Button(
            header, text="Создать пакет .DE", command=self.create_asset_pack_dialog
        ).pack(side="right", padx=3)
        ttk.Button(
            header,
            text="Показать дополнительный вывод",
            command=self.toggle_output_monitor,
        ).pack(side="right", padx=3)

        self.main_vertical = ttk.PanedWindow(self.root, orient="vertical")
        self.main_vertical.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        self.main_panes = ttk.PanedWindow(self.main_vertical, orient="horizontal")
        self.main_vertical.add(self.main_panes, weight=5)
        self.viewport_frame = ttk.Frame(self.main_panes)
        self.workspace_frame = ttk.Frame(self.main_panes, width=500)
        self.main_panes.add(self.viewport_frame, weight=3)
        self.main_panes.add(self.workspace_frame, weight=2)
        ttk.Label(
            self.viewport_frame,
            text="3D-просмотр | кликните в окно, WASD — камера | ПКМ + мышь — обзор | Shift + ПКМ — крен",
        ).pack(fill="x", padx=6, pady=4)
        self.viewport_host = tk.Frame(self.viewport_frame, background="#101319")
        self.viewport_host.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.viewport_host.bind("<Configure>", self._resize_viewport)
        self.output_panel = ttk.LabelFrame(
            self.main_vertical, text="Дополнительный вывод сцены"
        )
        self.output_controls = ttk.Frame(self.output_panel)
        self.output_controls.pack(fill="x", padx=4, pady=3)
        ttk.Label(self.output_controls, text="Источник").pack(side="left")
        self.output_source = tk.StringVar(value="Свободная камера")
        output_picker = ttk.Combobox(
            self.output_controls,
            textvariable=self.output_source,
            values=(
                "Свободная камера",
                "Камера редактора",
                "Камера теста",
                "Выбранный объект",
                "От выбранного света",
            ),
            state="readonly",
            width=22,
        )
        output_picker.pack(side="left", padx=4)
        output_picker.bind(
            "<<ComboboxSelected>>", lambda _event: self._sync_output_camera()
        )
        ttk.Button(
            self.output_controls,
            text="Отделить окно",
            command=self.detach_output_monitor,
        ).pack(side="right", padx=4)
        ttk.Label(
            self.output_controls,
            text="Свободный вид / слежение за объектом или светом: WASD, мышь; Shift + мышь или Q/E — крен",
        ).pack(side="left", padx=8)
        self.output_canvas = tk.Canvas(
            self.output_panel, height=180, background="#101319", highlightthickness=0
        )
        self.output_canvas.pack(fill="x", padx=4, pady=(0, 4))
        self.output_canvas.bind("<Button-1>", self._output_mouse_down)
        self.output_canvas.bind("<B1-Motion>", self._output_mouse_drag)
        self.output_canvas.bind(
            "<ButtonRelease-1>", lambda _event: setattr(self, "_output_drag", None)
        )
        self.output_canvas.bind("<KeyPress>", self._output_key_press)
        self.output_canvas.bind("<KeyRelease>", self._output_key_release)
        self.output_canvas.bind("<MouseWheel>", self._output_zoom)
        self.output_visible = False

        notebook = ttk.Notebook(self.workspace_frame)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self.scene_tab = ttk.Frame(notebook, padding=10)
        self.grid_tab = ttk.Frame(notebook, padding=10)
        self.lighting_tab = ttk.Frame(notebook, padding=10)
        self.logic_tab = ttk.Frame(notebook, padding=10)
        self.timeline_tab = ttk.Frame(notebook, padding=10)
        self.assets_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.scene_tab, text="Сцена")
        notebook.add(self.grid_tab, text="2D-сетка")
        notebook.add(self.lighting_tab, text="Свет")
        notebook.add(self.logic_tab, text="Логика")
        notebook.add(self.timeline_tab, text="Анимация")
        notebook.add(self.assets_tab, text="Ресурсы")

        self._build_scene_tab()
        self._build_grid_tab()
        self._build_lighting_tab()
        self._build_logic_tab()
        self._build_timeline_tab()
        self._build_assets_tab()
        self.curve_dock = ttk.LabelFrame(self.main_vertical, text="Кривые анимации")
        curve_header = ttk.Frame(self.curve_dock)
        curve_header.pack(fill="x", padx=4)
        ttk.Button(
            curve_header, text="Отделить кривую", command=self.detach_curve_editor
        ).pack(side="right")
        self.docked_curve_canvas = tk.Canvas(
            self.curve_dock,
            height=105,
            background="#20242c",
            highlightthickness=1,
            highlightbackground="#58616d",
        )
        self.curve_canvas = self.docked_curve_canvas
        self.docked_curve_canvas.pack(fill="x", padx=4, pady=4)
        self.curve_canvas.bind(
            "<Configure>", lambda _event: self._draw_animation_curve()
        )
        self.main_vertical.add(self.curve_dock, weight=1)
        ttk.Label(self.root, textvariable=self.status, anchor="w").pack(
            fill="x", padx=10, pady=(0, 5)
        )
        self._embed_viewport()
        self.apply_layout_profile()

    def _embed_viewport(self):
        if self.viewport_window is None:
            return
        if os.name != "nt":
            return
        self.root.update_idletasks()
        hwnd = int(__import__("glfw").get_win32_window(self.viewport_window))
        host_hwnd = self.viewport_host.winfo_id()
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
        user32.SetParent(ctypes.c_void_p(hwnd), ctypes.c_void_p(host_hwnd))
        style = user32.GetWindowLongPtrW(ctypes.c_void_p(hwnd), -16)
        style = (style & ~0x00C00000 & ~0x00040000) | 0x40000000 | 0x10000000
        user32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), -16, style)
        user32.SetWindowPos(ctypes.c_void_p(hwnd), None, 0, 0, 1, 1, 0x0040 | 0x0010)
        self._resize_viewport()

    def _resize_viewport(self, _event=None):
        if (
            self.viewport_window is None
            or os.name != "nt"
            or self.closing
            or self._viewport_resize_after_id is not None
        ):
            return
        self._viewport_resize_after_id = self.root.after(
            16, self._apply_viewport_resize
        )

    def _apply_viewport_resize(self):
        self._viewport_resize_after_id = None
        if self.viewport_window is None or os.name != "nt" or self.closing:
            return
        if not self.viewport_host.winfo_exists():
            return
        width = self.viewport_host.winfo_width()
        height = self.viewport_host.winfo_height()
        if width <= 1 or height <= 1:
            return
        if self._viewport_size == (width, height):
            return
        hwnd = int(__import__("glfw").get_win32_window(self.viewport_window))
        ctypes.windll.user32.MoveWindow.argtypes = (
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.BOOL,
        )
        ctypes.windll.user32.MoveWindow.restype = wintypes.BOOL
        if not ctypes.windll.user32.MoveWindow(hwnd, 0, 0, width, height, True):
            self.status.set(f"Ошибка изменения размера 3D-окна: {ctypes.WinError()}")
        else:
            self._viewport_size = (width, height)

    def apply_layout_profile(self):
        profile = self.layout_profile.get()
        tabs = {
            "DEc Studio": (
                (self.scene_tab, "Сцена"),
                (self.grid_tab, "2D-сетка"),
                (self.lighting_tab, "Свет"),
                (self.logic_tab, "Логика"),
                (self.timeline_tab, "Анимация"),
                (self.assets_tab, "Ресурсы"),
            ),
            "DEc Map": (
                (self.scene_tab, "Сцена"),
                (self.grid_tab, "2D-сетка"),
                (self.lighting_tab, "Свет"),
                (self.logic_tab, "Логика"),
                (self.assets_tab, "Ресурсы"),
            ),
            "DEc Motion": (
                (self.scene_tab, "Сцена"),
                (self.timeline_tab, "Анимация"),
                (self.lighting_tab, "Свет"),
                (self.assets_tab, "Ресурсы"),
            ),
        }
        for tab in self.notebook.tabs():
            self.notebook.hide(tab)
        for frame, title in tabs.get(profile, tabs["DEc Studio"]):
            self.notebook.tab(frame, text=title, state="normal")
        self.status.set(f"Шаблон интерфейса: {profile}")

    def _build_scene_tab(self):
        canvas = tk.Canvas(self.scene_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self.scene_tab, orient="vertical", command=canvas.yview
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, padding=(2, 0, 8, 0))
        canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(canvas_window, width=event.width),
        )
        self.root.bind_all(
            "<MouseWheel>",
            lambda event: self._scroll_scene_canvas(event, canvas, content),
            add="+",
        )

        toolbar = ttk.Frame(content)
        toolbar.pack(fill="x")
        for label, command in (
            ("Новая карта", self.new_project),
            ("Открыть", self.open_project),
            ("Сохранить", self.save_current_project),
            ("Играть / Стоп", self.toggle_playtest),
        ):
            button = ttk.Button(toolbar, text=label, command=command)
            button.pack(side="left", expand=True, fill="x", padx=2)
            if label == "Играть / Стоп":
                self.playtest_button = button

        ttk.Label(
            content,
            text="Создавайте карту, импортируйте ресурсы и настраивайте объекты. Режим «Играть» запускает проверку уровня от первого лица.",
            wraplength=460,
        ).pack(anchor="w", pady=(5, 2))
        tools = ttk.Frame(content)
        tools.pack(fill="x", pady=(8, 4))
        ttk.Button(tools, text="Добавить ящик", command=self.add_box).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(tools, text="Модель / ZIP", command=self.import_model).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(tools, text="Импорт папки", command=self.import_model_folder).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(tools, text="Точка игрока", command=self.add_spawn).pack(
            side="left", expand=True, fill="x", padx=2
        )
        entity_tools = ttk.Frame(content)
        entity_tools.pack(fill="x", pady=2)
        ttk.Button(entity_tools, text="Ландшафт", command=self.add_landscape).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(
            entity_tools, text="Источник звука", command=self.add_sound_source
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(entity_tools, text="NPC", command=self.add_npc).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(
            entity_tools, text="Логический триггер", command=self.add_logic_trigger
        ).pack(side="left", expand=True, fill="x", padx=2)

        ttk.Label(content, text="Объекты карты").pack(anchor="w", pady=(8, 2))
        self.object_list = tk.Listbox(content, height=10, exportselection=False)
        self.object_list.pack(fill="x")
        self.object_list.bind("<<ListboxSelect>>", self._select_object)
        ttk.Button(content, text="Удалить объект", command=self.delete_object).pack(
            fill="x", pady=(4, 10)
        )

        self.object_fields = {}
        self.object_axis_controls = {}
        self.object_color_swatch = None
        self.object_color_button = None
        for label, key, initial in (
            ("Имя", "name", ""),
            ("Позиция локальная X, Y, Z", "position", "0, 0, 0"),
            ("Поворот X, Y, Z (градусы)", "rotation", "0, 0, 0"),
            ("Масштаб", "scale", "1"),
            ("Цвет объекта", "color", "0.8, 0.8, 0.8"),
        ):
            row = ttk.Frame(content)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=27).pack(side="left")
            value = tk.StringVar(value=initial)
            if key == "rotation":
                for axis_name in ("X", "Y", "Z"):
                    ttk.Label(row, text=axis_name).pack(side="left", padx=(3, 0))
                    axis_value = tk.StringVar(value="0")
                    control = ttk.Spinbox(
                        row,
                        textvariable=axis_value,
                        from_=-360,
                        to=360,
                        increment=1,
                        width=7,
                        command=self._commit_object_rotation,
                    )
                    control.pack(side="left", fill="x", expand=True, padx=1)
                    control.bind("<Return>", self._commit_object_rotation)
                    control.bind("<FocusOut>", self._commit_object_rotation)
                    control.bind(
                        "<MouseWheel>",
                        lambda event, value=axis_value: self._spinbox_wheel(
                            event,
                            value,
                            1,
                            -360,
                            360,
                            self._commit_object_rotation,
                        ),
                    )
                    self.object_axis_controls[axis_name] = axis_value
                self.object_fields[key] = value
                continue
            if key == "color":
                self.object_color_swatch = tk.Label(
                    row,
                    text=" ",
                    width=4,
                    relief="sunken",
                )
                self.object_color_swatch.pack(side="left", padx=(4, 0))
                self.object_color_button = ttk.Button(
                    row,
                    text="Выбрать цвет",
                    command=self.choose_object_color,
                )
                self.object_color_button.pack(side="left", padx=4)
            else:
                control = ttk.Entry(row, textvariable=value)
                control.pack(side="left", fill="x", expand=True)
                if key == "scale":
                    control.bind("<Return>", self._commit_uniform_scale)
                    control.bind("<FocusOut>", self._commit_uniform_scale)
            self.object_fields[key] = value
        size_row = ttk.Frame(content)
        size_row.pack(fill="x", pady=2)
        ttk.Label(size_row, text="Размер X, Y, Z (единицы)", width=27).pack(side="left")
        self.object_size_controls = {}
        for axis_name in ("X", "Y", "Z"):
            ttk.Label(size_row, text=axis_name).pack(side="left", padx=(3, 0))
            axis_value = tk.StringVar(value="1")
            control = ttk.Spinbox(
                size_row,
                textvariable=axis_value,
                from_=0,
                to=1000,
                increment=0.1,
                width=7,
                command=self._commit_object_size,
            )
            control.pack(side="left", fill="x", expand=True, padx=1)
            control.bind("<Return>", self._commit_object_size)
            control.bind("<FocusOut>", self._commit_object_size)
            control.bind(
                "<MouseWheel>",
                lambda event, value=axis_value: self._spinbox_wheel(
                    event,
                    value,
                    0.1,
                    0,
                    1000,
                    self._commit_object_size,
                ),
            )
            self.object_size_controls[axis_name] = axis_value
        self.object_fields["scale_xyz"] = tk.StringVar(value="1, 1, 1")
        ttk.Button(content, text="Применить свойства", command=self.apply_object).pack(
            fill="x", pady=(8, 4)
        )
        self.source_label = ttk.Label(content, text="Объект не выбран", wraplength=420)
        self.source_label.pack(anchor="w", pady=4)
        self.rig_controls = ttk.Frame(content)
        self.rig_controls.pack(fill="x", pady=2)
        self.rig_button = ttk.Button(
            self.rig_controls, text="Создать авто-риг", command=self.auto_rig_selected
        )
        self.rig_button.pack(fill="x")
        self.texture_panel = ttk.LabelFrame(content, text="Материал и текстуры")
        self.texture_panel.pack(fill="x", pady=(5, 2))
        texture_row = ttk.Frame(self.texture_panel)
        texture_row.pack(fill="x", pady=(6, 2))
        ttk.Button(
            texture_row, text="Выбрать текстуру", command=self.choose_texture
        ).pack(side="left")
        ttk.Button(
            texture_row, text="Убрать текстуру", command=self.remove_texture
        ).pack(side="left", padx=4)
        self.texture_label = ttk.Label(
            self.texture_panel, text="Текстура не назначена", wraplength=440
        )
        self.texture_label.pack(anchor="w", pady=2)
        repeat_row = ttk.Frame(self.texture_panel)
        repeat_row.pack(fill="x", pady=2)
        ttk.Label(repeat_row, text="Повтор текстуры U / V").pack(side="left")
        self.texture_repeat_controls = {}
        for axis in ("U", "V"):
            ttk.Label(repeat_row, text=axis).pack(side="left", padx=(5, 1))
            value = tk.StringVar(value="1")
            control = ttk.Spinbox(
                repeat_row,
                textvariable=value,
                from_=0.1,
                to=100,
                increment=0.1,
                width=7,
                command=self._commit_texture_repeat,
            )
            control.pack(side="left", padx=2)
            control.bind("<Return>", self._commit_texture_repeat)
            control.bind("<FocusOut>", self._commit_texture_repeat)
            self.texture_repeat_controls[axis] = value
        ttk.Button(
            repeat_row, text="Применить", command=self._commit_texture_repeat
        ).pack(side="left", padx=4)
        sound_panel = ttk.LabelFrame(content, text="Источник звука")
        sound_panel.pack(fill="x", pady=(5, 2))
        sound_row = ttk.Frame(sound_panel)
        sound_row.pack(fill="x")
        ttk.Button(sound_row, text="Выбрать аудиофайл", command=self.choose_sound).pack(
            side="left"
        )
        ttk.Label(sound_row, text="Радиус").pack(side="left", padx=(6, 2))
        self.sound_radius = tk.StringVar(value="12")
        ttk.Entry(sound_row, textvariable=self.sound_radius, width=7).pack(side="left")
        self.audio_loop = tk.BooleanVar(value=True)
        ttk.Checkbutton(sound_row, text="Повтор", variable=self.audio_loop).pack(
            side="left", padx=4
        )
        self.audio_label = ttk.Label(
            sound_panel, text="Аудиофайл не назначен", wraplength=440
        )
        self.audio_label.pack(anchor="w", pady=2)
        ttk.Button(
            sound_panel,
            text="Применить настройки звука",
            command=self.apply_sound_settings,
        ).pack(fill="x", pady=(2, 4))
        self.sound_panel = sound_panel
        self.npc_panel = ttk.LabelFrame(content, text="NPC")
        self.npc_panel.pack(fill="x")
        ttk.Label(self.npc_panel, text="Поведение").pack(anchor="w")
        self.npc_behavior = tk.StringVar(value="Стоять")
        ttk.Combobox(
            self.npc_panel,
            textvariable=self.npc_behavior,
            values=("Стоять", "Следовать за игроком"),
            state="readonly",
        ).pack(fill="x", pady=(0, 3))
        self.invert_mouse_y = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            content,
            text="Инвертировать вертикальный обзор мышью",
            variable=self.invert_mouse_y,
        ).pack(anchor="w", pady=(6, 2))
        self.texture_controls = (texture_row, self.texture_label, repeat_row)

    def _scroll_scene_canvas(self, event, canvas, content):
        if self.notebook.select() != str(self.scene_tab):
            return
        if str(event.widget).startswith(str(content)):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

    def _build_grid_tab(self):
        ttk.Label(
            self.grid_tab,
            text="Три проекции видны одновременно. Перетаскивайте объект для перемещения; тяните квадратный угол для изменения размера. Shift + щелчок по выбранному ландшафту на виде сверху поднимает или опускает поверхность.",
            wraplength=460,
        ).pack(anchor="w", pady=(0, 6))
        controls = ttk.Frame(self.grid_tab)
        controls.pack(fill="x", pady=(0, 6))
        self.grid_snap = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Привязка",
            variable=self.grid_snap,
            command=self._draw_grid,
        ).pack(side="left", padx=8)
        self.grid_size = tk.StringVar(value="1")
        ttk.Label(controls, text="Шаг").pack(side="left")
        ttk.Entry(controls, textvariable=self.grid_size, width=6).pack(
            side="left", padx=4
        )
        self.terrain_brush_radius = tk.StringVar(value="2")
        ttk.Label(controls, text="Радиус кисти").pack(side="left", padx=(8, 2))
        ttk.Entry(controls, textvariable=self.terrain_brush_radius, width=5).pack(
            side="left"
        )
        self.terrain_brush_strength = tk.StringVar(value="0.5")
        ttk.Label(controls, text="Высота").pack(side="left", padx=(6, 2))
        ttk.Entry(controls, textvariable=self.terrain_brush_strength, width=5).pack(
            side="left"
        )
        ttk.Button(controls, text="Обновить", command=self._draw_grid).pack(
            side="right"
        )

        views = ttk.Frame(self.grid_tab)
        views.pack(fill="both", expand=True)
        views.columnconfigure((0, 1), weight=1, uniform="grid-view")
        views.rowconfigure((0, 1), weight=1, uniform="grid-view")
        self.grid_view_specs = {
            "XZ": ("Сверху — X / Z", (0, 2)),
            "XY": ("Спереди — X / Y", (0, 1)),
            "ZY": ("Сбоку — Z / Y", (2, 1)),
        }
        self.grid_views = {}
        for index, (view, (title, _axes)) in enumerate(self.grid_view_specs.items()):
            frame = ttk.LabelFrame(views, text=title, padding=3)
            frame.grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=3,
                pady=3,
            )
            canvas = tk.Canvas(
                frame,
                background="#20242c",
                highlightthickness=1,
                highlightbackground="#58616d",
            )
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda _event: self._draw_grid())
            canvas.bind(
                "<Button-1>",
                lambda event, view=view: self._grid_mouse_down(event, view),
            )
            canvas.bind(
                "<B1-Motion>",
                lambda event, view=view: self._grid_mouse_drag(event, view),
            )
            canvas.bind("<ButtonRelease-1>", self._finish_grid_drag)
            canvas.bind("<MouseWheel>", self._grid_mouse_wheel)
            self.grid_views[view] = canvas
        self.grid_canvas = self.grid_views["XZ"]
        help_panel = ttk.LabelFrame(views, text="Размеры выбранного объекта", padding=8)
        help_panel.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)
        ttk.Label(
            help_panel,
            text="Точный размер по X, Y, Z задаётся в полях «Сцена». Квадратный маркер меняет размеры по двум осям; маркеры света, появления и звука можно перемещать. Shift + щелчок на верхнем виде скульптит выбранный ландшафт.",
            wraplength=250,
            justify="left",
        ).pack(anchor="nw")

    def _build_lighting_tab(self):
        ttk.Label(
            self.lighting_tab,
            text="Источники света — объекты карты. Направленный свет имитирует солнце; точечный освещает сцену вокруг своей позиции.",
            wraplength=450,
        ).pack(anchor="w", pady=(0, 10))
        self.light_list = tk.Listbox(self.lighting_tab, height=6, exportselection=False)
        self.light_list.pack(fill="x")
        self.light_list.bind("<<ListboxSelect>>", self._select_light)
        light_buttons = ttk.Frame(self.lighting_tab)
        light_buttons.pack(fill="x", pady=4)
        ttk.Button(
            light_buttons,
            text="Добавить направленный",
            command=lambda: self.add_light("directional"),
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            light_buttons,
            text="Добавить точечный",
            command=lambda: self.add_light("point"),
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(light_buttons, text="Удалить свет", command=self.delete_light).pack(
            side="left", expand=True, fill="x", padx=2
        )
        self.light_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.lighting_tab, text="Свет включён", variable=self.light_enabled
        ).pack(anchor="w", pady=2)
        self.light_cast_shadows = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.lighting_tab,
            text="Источник отбрасывает тени",
            variable=self.light_cast_shadows,
        ).pack(anchor="w", pady=2)
        self.light_fields = {}
        self.light_color_swatch = None
        self.light_color_button = None
        for label, key, initial in (
            ("Имя", "name", "Солнце"),
            ("Тип", "kind", "Направленный"),
            ("Позиция локальная X, Y, Z", "position", "0, 5, 0"),
            ("Направление X, Y, Z", "direction", "-0.8, -1, -0.7"),
            ("Цвет источника", "color", "1, 0.96, 0.88"),
            ("Яркость (0-4)", "intensity", "1"),
            ("Общее заполнение (0-2)", "ambient", "0.25"),
        ):
            row = ttk.Frame(self.lighting_tab)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, width=29).pack(side="left")
            value = tk.StringVar(value=initial)
            if key == "kind":
                control = ttk.Combobox(
                    row,
                    textvariable=value,
                    values=("Направленный", "Точечный"),
                    state="readonly",
                )
                control.pack(side="left", fill="x", expand=True)
            elif key == "color":
                self.light_color_swatch = tk.Label(
                    row, text=" ", width=4, relief="sunken"
                )
                self.light_color_swatch.pack(side="left", padx=(4, 0))
                self.light_color_button = ttk.Button(
                    row,
                    text="Выбрать цвет",
                    command=self.choose_light_color,
                )
                self.light_color_button.pack(side="left", padx=4)
            else:
                control = ttk.Entry(row, textvariable=value)
                control.pack(side="left", fill="x", expand=True)
            self.light_fields[key] = value
        orientation_row = ttk.Frame(self.lighting_tab)
        orientation_row.pack(fill="x", pady=4)
        ttk.Label(orientation_row, text="Поворот направления: азимут / высота").pack(
            side="left"
        )
        self.light_orientation = {}
        for axis in ("Азимут", "Высота"):
            ttk.Label(orientation_row, text=axis).pack(side="left", padx=(6, 1))
            axis_value = tk.StringVar(value="0")
            control = ttk.Spinbox(
                orientation_row,
                textvariable=axis_value,
                from_=-180,
                to=180,
                increment=1,
                width=7,
                command=self._commit_light_orientation,
            )
            control.pack(side="left", padx=2)
            control.bind("<Return>", self._commit_light_orientation)
            control.bind("<FocusOut>", self._commit_light_orientation)
            control.bind(
                "<MouseWheel>",
                lambda event, value=axis_value: self._spinbox_wheel(
                    event,
                    value,
                    1,
                    -180,
                    180,
                    self._commit_light_orientation,
                ),
            )
            self.light_orientation[axis] = axis_value
        ttk.Button(
            self.lighting_tab,
            text="Применить параметры света",
            command=self.apply_lighting,
        ).pack(fill="x", pady=(8, 4))
        ttk.Separator(self.lighting_tab).pack(fill="x", pady=10)
        ttk.Label(
            self.lighting_tab,
            text="Карта теней строится по первому включённому направленному источнику. Точечные источники пока не создают тени. Высокое разрешение требует больше видеопамяти.",
            wraplength=450,
        ).pack(anchor="w")
        self.shadows_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.lighting_tab,
            text="Включить карты теней",
            variable=self.shadows_enabled,
            command=self.apply_shadow_settings,
        ).pack(anchor="w", pady=4)
        self.shadow_map_size = tk.StringVar(value="2048")
        shadow_row = ttk.Frame(self.lighting_tab)
        shadow_row.pack(fill="x")
        ttk.Label(shadow_row, text="Разрешение карты теней").pack(side="left")
        ttk.Combobox(
            shadow_row,
            textvariable=self.shadow_map_size,
            values=("512", "1024", "2048", "4096"),
            state="readonly",
            width=10,
        ).pack(side="right")
        ttk.Button(
            self.lighting_tab,
            text="Применить параметры теней",
            command=self.apply_shadow_settings,
        ).pack(fill="x", pady=5)

    def _build_logic_tab(self):
        ttk.Label(
            self.logic_tab,
            text="Настройте событие, цель и действие. «Когда игрок рядом» срабатывает при входе в заданный радиус; позиция и размер задаются в единицах сцены, поворот — в градусах.",
            wraplength=460,
        ).pack(anchor="w", pady=(0, 8))
        self.logic_list = tk.Listbox(self.logic_tab, height=12, exportselection=False)
        self.logic_list.pack(fill="x")
        ttk.Button(
            self.logic_tab, text="Удалить правило", command=self.delete_logic
        ).pack(fill="x", pady=(4, 12))

        self.logic_fields = {}
        self.logic_color_swatch = None
        self.logic_color_button = None
        self.logic_value_control = None
        for label, key, initial in (
            ("Событие", "trigger", "При запуске карты"),
            ("Действие", "action", "Задать позицию"),
            ("Объект-цель", "target", ""),
            ("Значение X, Y, Z / цвет RGB", "value", "0, 1, 0"),
            ("Клавиша для события", "key", "E"),
            ("Задержка таймера (сек)", "delay", "1"),
            ("Радиус для игрока рядом (ед.)", "radius", "3"),
        ):
            row = ttk.Frame(self.logic_tab)
            row.pack(fill="x", pady=3)
            label_widget = ttk.Label(row, text=label, width=27)
            label_widget.pack(side="left")
            if key == "value":
                self.logic_value_label = label_widget
            value = tk.StringVar(value=initial)
            if key == "trigger":
                control = ttk.Combobox(
                    row,
                    textvariable=value,
                    values=(
                        "При запуске карты",
                        "По таймеру",
                        "При нажатии клавиши",
                        "Когда игрок рядом",
                    ),
                    state="readonly",
                )
            elif key == "action":
                control = ttk.Combobox(
                    row,
                    textvariable=value,
                    values=(
                        "Задать позицию",
                        "Добавить поворот",
                        "Задать цвет",
                        "Переключить видимость",
                        "Задать размер",
                    ),
                    state="readonly",
                )
            elif key == "target":
                control = ttk.Combobox(row, textvariable=value, state="readonly")
                self.logic_target = control
            else:
                control = ttk.Entry(row, textvariable=value)
            control.pack(side="left", fill="x", expand=True)
            self.logic_fields[key] = value
            if key == "value":
                self.logic_value_control = control
            if key == "trigger":
                control.bind(
                    "<<ComboboxSelected>>", lambda _event: self._update_logic_controls()
                )
            elif key == "action":
                control.bind(
                    "<<ComboboxSelected>>", lambda _event: self._update_logic_controls()
                )
            if key == "value":
                self.logic_color_swatch = tk.Label(
                    row, text=" ", width=4, relief="sunken"
                )
                self.logic_color_button = ttk.Button(
                    row,
                    text="Выбрать цвет",
                    command=self.choose_logic_color,
                )
        self._update_logic_controls()
        ttk.Button(
            self.logic_tab, text="Добавить правило", command=self.add_logic
        ).pack(fill="x", pady=(8, 4))

    def _build_timeline_tab(self):
        ttk.Label(
            self.timeline_tab,
            text="Анимируйте камеру, объект или источник света. Выберите цель, измените её параметры и создайте ключи в разные моменты времени.",
            wraplength=460,
        ).pack(anchor="w", pady=(0, 8))
        self.timeline_value = tk.DoubleVar(value=0.0)
        self.timeline_scale = tk.Scale(
            self.timeline_tab,
            from_=0.0,
            to=30.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.timeline_value,
            command=self._set_playhead,
            label="Текущее время (секунды)",
        )
        self.timeline_scale.pack(fill="x")
        timing_row = ttk.Frame(self.timeline_tab)
        timing_row.pack(fill="x", pady=2)
        ttk.Label(timing_row, text="Точное время, сек").pack(side="left")
        self.key_time_value = tk.StringVar(value="0.00")
        ttk.Entry(timing_row, textvariable=self.key_time_value, width=9).pack(
            side="left", padx=4
        )
        ttk.Button(timing_row, text="Перейти", command=self.commit_playhead).pack(
            side="left"
        )
        ttk.Label(timing_row, text="Кадров/сек").pack(side="left", padx=(12, 2))
        self.timeline_fps = tk.StringVar(value="30")
        fps_picker = ttk.Combobox(
            timing_row,
            textvariable=self.timeline_fps,
            values=("24", "30", "60", "120"),
            state="readonly",
            width=5,
        )
        fps_picker.pack(side="left")
        fps_picker.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.timeline_scale.configure(
                resolution=1.0 / int(self.timeline_fps.get())
            ),
        )
        ttk.Button(
            timing_row,
            text="Изменить время выбранного ключа",
            command=self.move_selected_key,
        ).pack(side="right")
        controls = ttk.Frame(self.timeline_tab)
        controls.pack(fill="x", pady=4)
        ttk.Button(
            controls, text="Записать ключ", command=self.capture_selected_key
        ).pack(side="left", expand=True, fill="x", padx=2)
        self.play_button = ttk.Button(
            controls, text="Воспроизвести", command=self.toggle_playback
        )
        self.play_button.pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(controls, text="Удалить ключ", command=self.delete_camera_key).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(
            self.timeline_tab,
            text="Записать ключ через секунду",
            command=self.capture_selected_checkpoint,
        ).pack(fill="x", pady=(4, 0))
        target_row = ttk.Frame(self.timeline_tab)
        target_row.pack(fill="x", pady=(8, 2))
        ttk.Label(target_row, text="Цель").pack(side="left")
        self.animation_target_type = tk.StringVar(value="Камера")
        target_type_picker = ttk.Combobox(
            target_row,
            textvariable=self.animation_target_type,
            values=("Камера", "Объект", "Свет", "Кость"),
            state="readonly",
            width=12,
        )
        target_type_picker.pack(side="left", padx=4)
        target_type_picker.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_animation_targets()
        )
        self.animation_target_name = tk.StringVar(value="Camera")
        self.animation_target_picker = ttk.Combobox(
            target_row, textvariable=self.animation_target_name, state="readonly"
        )
        self.animation_target_picker.pack(side="left", fill="x", expand=True)
        self.animation_target_picker.bind(
            "<<ComboboxSelected>>", lambda _event: self._animation_target_changed()
        )
        self.bone_pose_row = ttk.Frame(self.timeline_tab)
        ttk.Label(self.bone_pose_row, text="Поворот кости, градусы").pack(side="left")
        self.bone_pose_controls = {}
        for axis in ("X", "Y", "Z"):
            ttk.Label(self.bone_pose_row, text=axis).pack(side="left", padx=(4, 1))
            value = tk.StringVar(value="0")
            control = ttk.Spinbox(
                self.bone_pose_row,
                textvariable=value,
                from_=-180,
                to=180,
                increment=1,
                width=7,
            )
            control.pack(side="left", padx=1)
            self.bone_pose_controls[axis] = value
        ttk.Button(
            self.bone_pose_row, text="Применить позу", command=self._apply_bone_pose
        ).pack(side="left", padx=4)
        interpolation_row = ttk.Frame(self.timeline_tab)
        interpolation_row.pack(fill="x", pady=2)
        ttk.Label(interpolation_row, text="Тип кривой").pack(side="left")
        self.animation_interpolation = tk.StringVar(value="Catmull–Rom")
        interpolation_picker = ttk.Combobox(
            interpolation_row,
            textvariable=self.animation_interpolation,
            values=("Линейная", "Плавная", "Catmull–Rom"),
            state="readonly",
            width=16,
        )
        interpolation_picker.pack(side="right")
        interpolation_picker.bind(
            "<<ComboboxSelected>>", lambda _event: self._set_track_interpolation()
        )
        ttk.Label(
            self.timeline_tab,
            text="Линейная — постоянная скорость; плавная — разгон и торможение; Catmull–Rom — кривая через ключи. Для объектов доступны позиция, поворот, масштаб и цвет; для света — позиция, направление, цвет и яркость.",
            wraplength=460,
        ).pack(anchor="w", pady=2)
        curve_row = ttk.Frame(self.timeline_tab)
        curve_row.pack(fill="x", pady=2)
        ttk.Label(curve_row, text="Канал кривой").pack(side="left")
        self.curve_channel = tk.StringVar(value="Позиция X")
        curve_picker = ttk.Combobox(
            curve_row,
            textvariable=self.curve_channel,
            values=(
                "Позиция X",
                "Позиция Y",
                "Позиция Z",
                "Поворот X",
                "Поворот Y",
                "Поворот Z",
                "Размер X",
                "Размер Y",
                "Размер Z",
                "Масштаб",
                "Цвет R",
                "Цвет G",
                "Цвет B",
                "Направление X",
                "Направление Y",
                "Направление Z",
                "Яркость",
                "Камера: yaw",
                "Камера: pitch",
                "Камера: roll",
            ),
            state="readonly",
            width=18,
        )
        curve_picker.pack(side="right")
        curve_picker.bind(
            "<<ComboboxSelected>>", lambda _event: self._draw_animation_curve()
        )
        self.keyframe_list = tk.Listbox(
            self.timeline_tab, height=14, exportselection=False
        )
        self.keyframe_list.pack(fill="both", expand=True, pady=(8, 0))
        self.keyframe_list.bind("<Button-1>", self._keyframe_list_click, add="+")
        self.keyframe_list.bind("<<ListboxSelect>>", self._select_keyframe)
        self._refresh_animation_targets()

    def _build_assets_tab(self):
        ttk.Label(
            self.assets_tab,
            text="Пакет .DE — сжатый архив моделей, изображений, анимаций и звуков. Фильтр и поиск помогают найти ресурс; двойной щелчок запускает предпросмотр.",
            wraplength=460,
        ).pack(anchor="w", pady=(0, 6))
        toolbar = ttk.Frame(self.assets_tab)
        toolbar.pack(fill="x", pady=3)
        ttk.Button(toolbar, text="Открыть пакет", command=self.open_asset_pack).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(
            toolbar, text="Создать пакет", command=self.create_asset_pack_dialog
        ).pack(side="left", expand=True, fill="x", padx=2)
        filter_row = ttk.Frame(self.assets_tab)
        filter_row.pack(fill="x", pady=3)
        ttk.Label(filter_row, text="Тип").pack(side="left")
        filter_picker = ttk.Combobox(
            filter_row,
            textvariable=self.asset_filter,
            values=("Все", "Модели", "Текстуры", "Анимации", "Звуки", "Прочее"),
            state="readonly",
            width=12,
        )
        filter_picker.pack(side="left", padx=4)
        filter_picker.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_asset_list()
        )
        ttk.Entry(filter_row, textvariable=self.asset_search).pack(
            side="left", fill="x", expand=True
        )
        self.asset_search.trace_add("write", lambda *_args: self._refresh_asset_list())
        self.asset_list = tk.Listbox(self.assets_tab, height=12, exportselection=False)
        self.asset_list.pack(fill="both", expand=True, pady=4)
        self.asset_list.bind("<<ListboxSelect>>", self.preview_selected_asset)
        self.asset_list.bind("<Double-Button-1>", self.import_selected_asset)
        self.asset_preview_label = ttk.Label(
            self.assets_tab,
            text="Выберите ресурс для предпросмотра. Модели показываются в окне 3D, изображения — ниже.",
            wraplength=460,
        )
        self.asset_preview_label.pack(fill="x", pady=3)
        self.asset_image_preview = ttk.Label(self.assets_tab)
        self.asset_image_preview.pack(pady=3)
        actions = ttk.Frame(self.assets_tab)
        actions.pack(fill="x")
        ttk.Button(
            actions, text="Импортировать модель", command=self.import_selected_asset
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            actions,
            text="Назначить выбранную текстуру",
            command=self.assign_selected_asset_texture,
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            actions,
            text="Назначить аудио источнику",
            command=self.assign_selected_asset_sound,
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            actions, text="Остановить предпросмотр", command=self.clear_asset_preview
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            actions, text="Прослушать аудио", command=self.play_selected_asset_sound
        ).pack(side="left", expand=True, fill="x", padx=2)

    def _selected_index(self):
        selection = self.object_list.curselection()
        return selection[0] if selection else None

    def create_asset_pack_dialog(self):
        files = filedialog.askopenfilenames(
            parent=self.root,
            title="Выберите модели, текстуры, анимации и звуки",
            filetypes=(
                (
                    "Ресурсы DEc",
                    "*.obj *.glb *.gltf *.stl *.ply *.off *.xyz *.png *.jpg *.jpeg *.bmp *.tga *.wav *.ogg *.mp3 *.flac *.bvh *.anim",
                ),
                ("Все файлы", "*.*"),
            ),
        )
        if not files:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить сжатый пакет ресурсов",
            defaultextension=".DE",
            filetypes=(("Пакет DEc", "*.DE"),),
        )
        if not path:
            return
        try:
            create_asset_pack(path, files)
            self.load_asset_pack(path)
            self.status.set(f"Пакет ресурсов создан: {path}")
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Не удалось создать пакет", str(error), parent=self.root
            )

    def open_asset_pack(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Открыть пакет ресурсов DEc",
            filetypes=(
                ("Пакет DEc", "*.DE"),
                ("Архивы", "*.zip"),
                ("Все файлы", "*.*"),
            ),
        )
        if not path:
            return
        try:
            self.load_asset_pack(path)
            self.notebook.select(self.assets_tab)
            self.status.set(f"Открыт пакет ресурсов: {path}")
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Не удалось открыть пакет", str(error), parent=self.root
            )

    def load_asset_pack(self, path):
        self.clear_asset_preview()
        self.asset_entries = list_asset_pack(path)
        self.asset_pack_path = path
        self.asset_filter.set("Все")
        self.asset_search.set("")
        self._refresh_asset_list()
        self.scene.events.publish(
            ResourceChanged(
                resource_kind="asset_pack",
                action="loaded",
                resource_id=os.fspath(path),
            )
        )

    def _refresh_asset_list(self):
        if not hasattr(self, "asset_list"):
            return
        categories = {
            "Все": None,
            "Модели": "Models",
            "Текстуры": "Textures",
            "Анимации": "Animations",
            "Звуки": "Sounds",
            "Прочее": "Other",
        }
        category = categories.get(self.asset_filter.get())
        query = self.asset_search.get().strip().casefold()
        self.filtered_asset_entries = [
            item
            for item in self.asset_entries
            if (category is None or item["category"] == category)
            and query in item["name"].casefold()
        ]
        self.asset_list.delete(0, tk.END)
        for item in self.filtered_asset_entries:
            self.asset_list.insert(tk.END, f"{item['category']}: {item['name']}")

    def _selected_asset(self):
        selection = self.asset_list.curselection()
        if not selection or selection[0] >= len(
            getattr(self, "filtered_asset_entries", [])
        ):
            return None
        if not self.asset_pack_path:
            return None
        entry = self.filtered_asset_entries[selection[0]]
        destination = os.path.join(self.asset_cache.name, *entry["path"].split("/"))
        extract_asset(self.asset_pack_path, entry["path"], destination)
        return entry, destination

    def preview_selected_asset(self, _event=None):
        try:
            selected = self._selected_asset()
            if selected is None:
                return
            entry, path = selected
            self.clear_asset_preview()
            extension = os.path.splitext(path)[1].lower()
            if extension in MODEL_EXTENSIONS:
                for packed_entry in self.asset_entries:
                    destination = os.path.join(
                        self.asset_cache.name, *packed_entry["path"].split("/")
                    )
                    extract_asset(
                        self.asset_pack_path, packed_entry["path"], destination
                    )
                imported = load_model(path)
                minimum = imported.vertices.min(axis=0)
                maximum = imported.vertices.max(axis=0)
                bounds = maximum - minimum
                vertices = imported.vertices - (minimum + maximum) / 2.0
                size = max(float(max(bounds)), 1e-4)
                preview = MeshInstance(
                    name="DEc asset preview",
                    vertices=vertices,
                    normals=imported.normals,
                    uvs=imported.uvs,
                    indices=imported.indices,
                    color=imported.color,
                    position=tuple(
                        self.camera.position
                        + self.camera.forward() * 4.0
                        + self.camera.right() * 2.0
                    ),
                    scale=3.0 / size,
                    source_path=path,
                    primitive="asset_preview",
                    texture_path=self._resolve_model_texture(
                        path, imported, self.asset_cache.name
                    ),
                )
                self.preview_object = preview
                self.asset_preview_label.configure(
                    text=f"3D-предпросмотр: {entry['name']} | Размеры: "
                    f"{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}"
                )
            elif extension in TEXTURE_EXTENSIONS:
                with Image.open(path) as image:
                    image.thumbnail((360, 180))
                    preview = image.copy()
                self._asset_photo = tk.PhotoImage(
                    master=self.root,
                    data=self._image_to_png_data(preview),
                    format="png",
                )
                self.asset_image_preview.configure(image=self._asset_photo)
                self.asset_preview_label.configure(
                    text=f"Текстура: {entry['name']} | {preview.width}×{preview.height} px"
                )
            elif extension in SOUND_EXTENSIONS:
                self.asset_preview_label.configure(
                    text=f"Аудио: {entry['name']} | нажмите «Прослушать», чтобы проверить звук"
                )
            elif extension in ANIMATION_EXTENSIONS:
                self.asset_preview_label.configure(
                    text=f"Анимационный ресурс: {entry['name']}. Файловый предпросмотр пока ограничен; "
                    "для анимации используйте дорожки Timeline."
                )
            else:
                self.asset_preview_label.configure(text=f"Ресурс: {entry['name']}")
        except (OSError, ValueError, TypeError) as error:
            self.clear_asset_preview()
            self.asset_preview_label.configure(text=f"Ошибка предпросмотра: {error}")

    @staticmethod
    def _image_to_png_data(image):
        import base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue())

    def import_selected_asset(self, _event=None):
        try:
            selected = self._selected_asset()
            if selected is None:
                return
            entry, path = selected
            if entry["category"] != "Models":
                self.status.set(
                    "Этот ресурс не является моделью. Для него доступен предпросмотр или назначение текстуры."
                )
                return
            for packed_entry in self.asset_entries:
                destination = os.path.join(
                    self.asset_cache.name, *packed_entry["path"].split("/")
                )
                extract_asset(self.asset_pack_path, packed_entry["path"], destination)
            self._import_model_file(
                path, self.asset_cache.name, display_name=entry["name"]
            )
        except (OSError, ValueError, TypeError) as error:
            messagebox.showerror("Ошибка импорта модели", str(error), parent=self.root)

    @_undoable
    def assign_selected_asset_texture(self):
        selected = self._selected_asset()
        object_index = self._selected_index()
        if selected is None or object_index is None:
            self.status.set("Выберите текстуру в пакете и объект на вкладке «Сцена».")
            return
        entry, path = selected
        if os.path.splitext(path)[1].lower() not in TEXTURE_EXTENSIONS:
            self.status.set("Выбранный ресурс не является текстурой.")
            return
        obj = self.scene.objects[object_index]
        if not self._supports_texture(obj):
            self.status.set(
                "У выбранного технического объекта нет отображаемого материала."
            )
            return
        obj.texture_path = path
        self.scene.events.publish(
            ResourceChanged(
                resource_kind="texture",
                action="assigned",
                resource_id=os.fspath(path),
                node_id=obj.id,
            )
        )
        self.texture_label.configure(text=entry["name"])
        self.status.set(f"Текстура назначена: {entry['name']}")

    def play_selected_asset_sound(self):
        selected = self._selected_asset()
        if selected is None:
            return
        entry, path = selected
        if os.path.splitext(path)[1].lower() not in SOUND_EXTENSIONS:
            self.status.set("Выберите поддерживаемый аудиофайл.")
            return
        try:
            self.audio_player.play(path)
        except (OSError, ValueError, RuntimeError) as error:
            self.status.set(f"Ошибка аудиопредпросмотра: {error}")
            return
        self.audio_preview = path
        self.status.set(f"Воспроизводится: {entry['name']}")

    def clear_asset_preview(self):
        if getattr(self, "preview_object", None) is not None:
            self.preview_object = None
        if hasattr(self, "asset_image_preview"):
            self.asset_image_preview.configure(image="")
        self._asset_photo = None
        if self.audio_preview:
            self.audio_player.stop()
            self.audio_preview = None

    def _refresh_objects(self, select=None):
        self.object_list.delete(0, tk.END)
        for obj in self.scene.objects:
            self.object_list.insert(tk.END, obj.name)
        self.logic_target["values"] = tuple(obj.name for obj in self.scene.objects)
        if hasattr(self, "animation_target_picker"):
            self._refresh_animation_targets()
        self._draw_grid()
        if select is not None and 0 <= select < len(self.scene.objects):
            self.object_list.selection_set(select)
            self.object_list.event_generate("<<ListboxSelect>>")
        else:
            self._configure_object_controls()

    def _unique_name(self, requested, ignored_index=None):
        used = {
            obj.name
            for index, obj in enumerate(self.scene.objects)
            if index != ignored_index
        }
        name = requested
        suffix = 2
        while name in used:
            name = f"{requested} ({suffix})"
            suffix += 1
        return name

    def _configure_object_controls(self, obj=None):
        if obj is None:
            self.texture_panel.pack_forget()
            self.sound_panel.pack_forget()
            self.npc_panel.pack_forget()
            self.rig_controls.pack_forget()
            return
        if obj.primitive in ("spawn", "sound_source", "logic_trigger"):
            self.texture_panel.pack_forget()
        else:
            self.texture_panel.pack(fill="x", pady=(5, 2))
        if obj.primitive == "sound_source":
            self.sound_panel.pack(fill="x", pady=(5, 2))
        else:
            self.sound_panel.pack_forget()
        if obj.primitive == "npc":
            self.npc_panel.pack(fill="x")
        else:
            self.npc_panel.pack_forget()
        if obj.primitive in ("spawn", "sound_source", "logic_trigger"):
            self.rig_controls.pack_forget()
        else:
            self.rig_controls.pack(fill="x", pady=2)
        self.rig_button.configure(
            text="Пересоздать авто-риг" if obj.rig else "Создать авто-риг"
        )

    @staticmethod
    def _supports_texture(obj):
        return obj.primitive not in ("spawn", "sound_source", "logic_trigger")

    def toggle_output_monitor(self):
        if self.output_window is not None:
            self.output_window.deiconify()
            self.output_window.lift()
            self.output_visible = True
            return
        self.output_visible = not self.output_visible
        if self.output_visible:
            self.main_vertical.insert(
                self.main_vertical.panes().index(str(self.curve_dock)),
                self.output_panel,
                weight=2,
            )
            self.output_canvas.focus_set()
        elif str(self.output_panel) in self.main_vertical.panes():
            self.main_vertical.forget(self.output_panel)

    def active_output_canvas(self):
        if not self.output_visible:
            return None
        return self.output_window_canvas or self.output_canvas

    def detach_output_monitor(self):
        if self.output_window is not None:
            self.output_window.deiconify()
            self.output_window.lift()
            return
        if str(self.output_panel) in self.main_vertical.panes():
            self.main_vertical.forget(self.output_panel)
        self.output_window = tk.Toplevel(self.root)
        self.output_window.title("DEc | Дополнительный вывод")
        self.output_window.geometry("960x600")
        self.output_window.minsize(360, 240)
        controls = ttk.Frame(self.output_window)
        controls.pack(fill="x", padx=4, pady=3)
        ttk.Label(controls, text="Источник").pack(side="left")
        picker = ttk.Combobox(
            controls,
            textvariable=self.output_source,
            values=(
                "Свободная камера",
                "Камера редактора",
                "Камера теста",
                "Выбранный объект",
                "От выбранного света",
            ),
            state="readonly",
            width=22,
        )
        picker.pack(side="left", padx=4)
        picker.bind("<<ComboboxSelected>>", lambda _event: self._sync_output_camera())
        ttk.Button(
            controls, text="Вернуть в редактор", command=self.dock_output_monitor
        ).pack(side="right")
        self.output_window_canvas = tk.Canvas(
            self.output_window, background="#101319", highlightthickness=0
        )
        self.output_window_canvas.pack(fill="both", expand=True)
        self._bind_output_camera(self.output_window_canvas)
        self.output_window.protocol("WM_DELETE_WINDOW", self.dock_output_monitor)
        self.output_visible = True
        self.status.set("Дополнительный вывод открыт в отдельном изменяемом окне.")

    def dock_output_monitor(self):
        if self.output_window is None:
            return
        self.output_visible = False
        self.output_window.destroy()
        self.output_window = None
        self.output_window_canvas = None
        self.output_visible = True
        self.main_vertical.insert(
            self.main_vertical.panes().index(str(self.curve_dock)),
            self.output_panel,
            weight=2,
        )

    def detach_curve_editor(self):
        if self.curve_window is not None:
            self.curve_window.deiconify()
            self.curve_window.lift()
            return
        self.main_vertical.forget(self.curve_dock)
        self.curve_window = tk.Toplevel(self.root)
        self.curve_window.title("DEc | Кривые анимации")
        self.curve_window.geometry("960x320")
        self.curve_window.minsize(420, 180)
        ttk.Button(
            self.curve_window,
            text="Вернуть кривую в редактор",
            command=self.dock_curve_editor,
        ).pack(anchor="e", padx=6, pady=4)
        self.detached_curve_canvas = tk.Canvas(
            self.curve_window,
            height=180,
            background="#20242c",
            highlightthickness=1,
            highlightbackground="#58616d",
        )
        self.detached_curve_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.detached_curve_canvas.bind(
            "<Configure>", lambda _event: self._draw_animation_curve()
        )
        self.curve_canvas = self.detached_curve_canvas
        self.curve_window.protocol("WM_DELETE_WINDOW", self.dock_curve_editor)
        self._draw_animation_curve()

    def dock_curve_editor(self):
        if self.curve_window is None:
            return
        self.curve_window.destroy()
        self.curve_window = None
        self.detached_curve_canvas = None
        self.curve_canvas = self.docked_curve_canvas
        self.main_vertical.add(self.curve_dock, weight=1)
        self._draw_animation_curve()

    def _bind_output_camera(self, canvas):
        canvas.bind("<Button-1>", self._output_mouse_down)
        canvas.bind("<B1-Motion>", self._output_mouse_drag)
        canvas.bind(
            "<ButtonRelease-1>", lambda _event: setattr(self, "_output_drag", None)
        )
        canvas.bind("<KeyPress>", self._output_key_press)
        canvas.bind("<KeyRelease>", self._output_key_release)
        canvas.bind("<MouseWheel>", self._output_zoom)

    def _sync_output_camera(self):
        mode = self.output_source.get()
        if mode in ("Свободная камера", "Камера редактора", "Камера теста"):
            self._output_follow_key = None
            self._output_follow_target = None
            return
        if mode == "Выбранный объект":
            index = self._selected_index()
            if index is None:
                self.status.set("Выберите объект для дополнительного вывода.")
                return
            obj = self.scene.objects[index]
            local_center = (
                np.asarray(obj.vertices).min(axis=0)
                + np.asarray(obj.vertices).max(axis=0)
            ) / 2
            target = (self.scene.world_matrix(obj) @ np.append(local_center, 1.0))[:3]
            follow_key = (mode, id(obj))
            if follow_key != self._output_follow_key:
                self.output_camera.position[:] = target + np.asarray((3.0, 2.0, 4.0))
                direction = target - self.output_camera.position
                self.output_camera.yaw = float(np.arctan2(direction[0], -direction[2]))
                self.output_camera.pitch = float(
                    np.arcsin(
                        np.clip(
                            direction[1] / (np.linalg.norm(direction) + 1e-8), -1, 1
                        )
                    )
                )
        else:
            index = self._selected_light_index()
            if index is None:
                self.status.set("Выберите источник света для дополнительного вывода.")
                return
            light = self.scene.lights[index]
            target = np.asarray(self.scene.world_position(light), dtype=np.float32)
            follow_key = (mode, id(light))
            if follow_key != self._output_follow_key:
                self.output_camera.position[:] = target
                direction = np.asarray(light.direction, dtype=np.float32)
                direction /= np.linalg.norm(direction) + 1e-8
                self.output_camera.yaw = float(np.arctan2(direction[0], -direction[2]))
                self.output_camera.pitch = float(
                    np.arcsin(np.clip(direction[1], -1, 1))
                )
        if (
            follow_key == self._output_follow_key
            and self._output_follow_target is not None
        ):
            self.output_camera.position += target - self._output_follow_target
        self._output_follow_key = follow_key
        self._output_follow_target = np.asarray(target, dtype=np.float32).copy()

    def output_camera_for_render(self, playtest_camera):
        mode = self.output_source.get()
        if mode == "Камера редактора":
            return self.camera
        if mode == "Камера теста":
            return playtest_camera
        if mode in ("Выбранный объект", "От выбранного света"):
            self._sync_output_camera()
            return self.output_camera
        return self.output_camera

    def _output_mouse_down(self, event):
        event.widget.focus_set()
        if self.output_source.get() not in ("Камера редактора", "Камера теста"):
            self._output_drag = (event.x, event.y)

    def _output_mouse_drag(self, event):
        if self._output_drag is None or self.output_source.get() in (
            "Камера редактора",
            "Камера теста",
        ):
            return
        previous_x, previous_y = self._output_drag
        if event.state & 0x0001:
            self.output_camera.roll += (
                event.x - previous_x
            ) * self.output_camera.sensitivity
        else:
            self.output_camera.yaw += (
                event.x - previous_x
            ) * self.output_camera.sensitivity
            self.output_camera.pitch = float(
                np.clip(
                    self.output_camera.pitch
                    - (event.y - previous_y) * self.output_camera.sensitivity,
                    -1.5,
                    1.5,
                )
            )
        self._output_drag = (event.x, event.y)

    def _output_key_press(self, event):
        if self.output_source.get() in ("Камера редактора", "Камера теста"):
            return
        self._output_keys.add(event.keysym.lower())

    def _output_key_release(self, event):
        self._output_keys.discard(event.keysym.lower())

    def _update_output_camera(self, delta_time):
        if not self.output_visible or self.output_source.get() in (
            "Камера редактора",
            "Камера теста",
        ):
            return
        forward = self.output_camera.forward()
        right = self.output_camera.right()
        move = np.zeros(3, dtype=np.float32)
        for key, direction in (
            ("w", forward),
            ("s", -forward),
            ("a", -right),
            ("d", right),
        ):
            if key in self._output_keys:
                move += direction
        if np.linalg.norm(move) > 0:
            self.output_camera.position += (
                move / np.linalg.norm(move) * self.output_camera.speed * delta_time
            )
        if "q" in self._output_keys:
            self.output_camera.roll += 1.5 * delta_time
        if "e" in self._output_keys:
            self.output_camera.roll -= 1.5 * delta_time

    def _output_zoom(self, event):
        if self.output_source.get() not in ("Камера редактора", "Камера теста"):
            self.output_camera.position += self.output_camera.forward() * (
                0.5 if event.delta > 0 else -0.5
            )

    def _select_object(self, _event=None):
        index = self._selected_index()
        if index is None:
            self.scene.events.publish(
                SelectionChanged(selected_ids=(), active_id=None, source="none")
            )
            return
        obj = self.scene.objects[index]
        self.scene.events.publish(
            SelectionChanged(
                selected_ids=(obj.id,),
                active_id=obj.id,
                source="object_list",
            )
        )
        self.object_fields["name"].set(obj.name)
        self.object_fields["position"].set(
            ", ".join(f"{part:.3f}" for part in obj.position)
        )
        rotation_degrees = tuple(
            part * 180.0 / 3.141592653589793 for part in obj.rotation
        )
        self.object_fields["rotation"].set(
            ", ".join(f"{part:.2f}" for part in rotation_degrees)
        )
        for axis, value in zip(("X", "Y", "Z"), rotation_degrees):
            self.object_axis_controls[axis].set(f"{value:.2f}")
        self.object_fields["scale"].set(f"{obj.scale:.3f}")
        dimensions = self._object_dimensions(obj)
        self.object_fields["scale_xyz"].set(
            ", ".join(f"{part:.3f}" for part in dimensions)
        )
        for axis, value in zip(("X", "Y", "Z"), dimensions):
            self.object_size_controls[axis].set(f"{value:.3f}")
        for axis, value in zip(("U", "V"), obj.texture_repeat):
            self.texture_repeat_controls[axis].set(f"{value:.2f}")
        self.object_fields["color"].set(", ".join(f"{part:.6f}" for part in obj.color))
        self._update_color_swatch(self.object_color_swatch, obj.color)
        self.source_label.configure(
            text=obj.source_path or f"Тип: {obj.primitive or 'модель'}"
        )
        self.texture_label.configure(text=obj.texture_path or "Текстура не назначена")
        self.audio_label.configure(text=obj.audio_path or "Аудиофайл не назначен")
        self.sound_radius.set(f"{obj.sound_radius:g}")
        self.audio_loop.set(obj.audio_loop)
        self.npc_behavior.set(
            "Следовать за игроком" if obj.npc_behavior == "follow" else "Стоять"
        )
        self._configure_object_controls(obj)
        self.logic_fields["target"].set(obj.name)
        self._set_animation_kind("object")
        self._refresh_animation_targets()
        self.animation_target_name.set(obj.name)
        self._animation_target_changed()
        self._draw_grid()

    @_undoable
    def _commit_object_rotation(self, _event=None):
        index = self._selected_index()
        if index is None:
            return
        try:
            degrees = tuple(
                float(self.object_axis_controls[axis].get()) for axis in ("X", "Y", "Z")
            )
        except ValueError:
            self.status.set("Введите угол поворота числом для каждой оси.")
            return
        if not all(np.isfinite(value) for value in degrees):
            self.status.set("Углы поворота должны быть конечными числами.")
            return
        self.scene.objects[index].rotation = tuple(
            value * np.pi / 180.0 for value in degrees
        )
        self.object_fields["rotation"].set(
            ", ".join(f"{value:.2f}" for value in degrees)
        )
        self._draw_grid()

    @_undoable
    def _commit_uniform_scale(self, _event=None):
        index = self._selected_index()
        if index is None:
            return
        try:
            scale = float(self.object_fields["scale"].get())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError
        except ValueError:
            self.status.set("Общий масштаб должен быть положительным числом.")
            return
        obj = self.scene.objects[index]
        obj.scale = scale
        dimensions = self._object_dimensions(obj)
        self.object_fields["scale_xyz"].set(
            ", ".join(f"{value:.3f}" for value in dimensions)
        )
        for axis, value in zip(("X", "Y", "Z"), dimensions):
            self.object_size_controls[axis].set(f"{value:.3f}")
        self._draw_grid()

    def _spinbox_wheel(self, event, variable, increment, minimum, maximum, commit):
        try:
            current = float(variable.get())
        except ValueError:
            self.status.set(
                "Сначала введите корректное число, затем используйте колесо мыши."
            )
            return "break"
        direction = 1 if event.delta > 0 else -1
        value = min(maximum, max(minimum, current + direction * increment))
        variable.set(f"{value:.3f}".rstrip("0").rstrip("."))
        commit()
        return "break"

    @_undoable
    def _commit_object_size(self, _event=None):
        index = self._selected_index()
        if index is None:
            return
        try:
            sizes = tuple(
                float(self.object_size_controls[axis].get()) for axis in ("X", "Y", "Z")
            )
            if any(not np.isfinite(value) or value < 0 for value in sizes):
                raise ValueError
        except ValueError:
            self.status.set("Размеры осей должны быть неотрицательными числами.")
            return
        obj = self.scene.objects[index]
        base_dimensions = self._object_base_dimensions(obj)
        if any(
            size > 1e-8 and base <= 1e-8 for size, base in zip(sizes, base_dimensions)
        ):
            self.status.set(
                "Нельзя задать размер по оси, у которой у модели нет геометрии."
            )
            return
        obj.scale_xyz = tuple(
            size / base if base > 1e-8 else current
            for size, base, current in zip(sizes, base_dimensions, obj.scale_xyz)
        )
        self.object_fields["scale_xyz"].set(
            ", ".join(f"{value:.3f}" for value in sizes)
        )
        self._draw_grid()

    @staticmethod
    def _object_base_dimensions(obj):
        vertices = np.asarray(obj.vertices, dtype=np.float64)
        return tuple(
            float(value) * obj.scale
            for value in vertices.max(axis=0) - vertices.min(axis=0)
        )

    def _object_dimensions(self, obj):
        return tuple(
            base * factor
            for base, factor in zip(self._object_base_dimensions(obj), obj.scale_xyz)
        )

    @staticmethod
    def _update_color_swatch(swatch, color):
        if swatch is None:
            return
        rgb = tuple(max(0, min(255, round(channel * 255))) for channel in color)
        swatch.configure(background="#%02x%02x%02x" % rgb)

    def _choose_color(self, variable, swatch, title):
        try:
            channels = self._parse_vector(variable.get(), "Цвет")
            initial = "#%02x%02x%02x" % tuple(
                max(0, min(255, round(channel * 255))) for channel in channels
            )
        except ValueError:
            initial = "#ffffff"
        _rgb, color = colorchooser.askcolor(
            color=initial,
            parent=self.root,
            title=title,
        )
        if color is None:
            return
        channels = tuple(
            int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5)
        )
        variable.set(", ".join(f"{channel:.6f}" for channel in channels))
        self._update_color_swatch(swatch, channels)

    def choose_object_color(self):
        self._choose_color(
            self.object_fields["color"], self.object_color_swatch, "Цвет объекта"
        )

    @_undoable
    def choose_texture(self):
        index = self._selected_index()
        if index is None:
            self.status.set("Сначала выберите объект для назначения текстуры.")
            return
        if not self._supports_texture(self.scene.objects[index]):
            self.status.set(
                "У выбранного технического объекта нет отображаемого материала."
            )
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите изображение для текстуры",
            filetypes=(
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tga *.tif *.tiff"),
                ("Все файлы", "*.*"),
            ),
        )
        if not path:
            return
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            messagebox.showerror("Некорректная текстура", str(error), parent=self.root)
            return
        self.scene.objects[index].texture_path = path
        self.scene.events.publish(
            ResourceChanged(
                resource_kind="texture",
                action="assigned",
                resource_id=os.fspath(path),
                node_id=self.scene.objects[index].id,
            )
        )
        self.texture_label.configure(text=path)
        self.status.set(f"Текстура назначена объекту: {self.scene.objects[index].name}")

    @_undoable
    def _commit_texture_repeat(self, _event=None):
        index = self._selected_index()
        if index is None:
            return
        try:
            repeat = tuple(
                float(self.texture_repeat_controls[axis].get()) for axis in ("U", "V")
            )
            if any(
                not np.isfinite(value) or not 0.0 < value <= 100.0 for value in repeat
            ):
                raise ValueError("Повтор текстуры должен быть от 0 до 100.")
        except ValueError as error:
            self.status.set(f"Некорректный повтор текстуры: {error}")
            return
        self.scene.objects[index].texture_repeat = repeat
        self.status.set(f"Повтор текстуры обновлён: U={repeat[0]:g}, V={repeat[1]:g}")

    @_undoable
    def choose_sound(self):
        index = self._selected_index()
        if index is None or self.scene.objects[index].primitive != "sound_source":
            self.status.set("Сначала выберите объект «Источник звука».")
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите аудиофайл",
            filetypes=(
                (
                    "Аудиофайлы",
                    " ".join(f"*{extension}" for extension in sorted(SOUND_EXTENSIONS)),
                ),
                ("Все файлы", "*.*"),
            ),
        )
        if not path:
            return
        try:
            self.audio_player.validate(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Некорректный аудиофайл", str(error), parent=self.root)
            return
        self.scene.objects[index].audio_path = path
        self.scene.events.publish(
            ResourceChanged(
                resource_kind="audio",
                action="assigned",
                resource_id=os.fspath(path),
                node_id=self.scene.objects[index].id,
            )
        )
        self.audio_label.configure(text=path)
        self.status.set(f"Аудио назначено источнику: {self.scene.objects[index].name}")

    @_undoable
    def apply_sound_settings(self):
        index = self._selected_index()
        if index is None or self.scene.objects[index].primitive != "sound_source":
            self.status.set("Выберите объект «Источник звука».")
            return
        try:
            radius = float(self.sound_radius.get())
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("Радиус должен быть положительным числом.")
        except ValueError as error:
            messagebox.showerror(
                "Некорректный источник звука", str(error), parent=self.root
            )
            return
        sound = self.scene.objects[index]
        sound.sound_radius = radius
        sound.audio_loop = self.audio_loop.get()
        self.status.set(f"Настройки звука обновлены: радиус {radius:g}")

    @_undoable
    def assign_selected_asset_sound(self):
        selected = self._selected_asset()
        object_index = self._selected_index()
        if selected is None or object_index is None:
            self.status.set(
                "Выберите аудиофайл в пакете и источник звука на вкладке «Сцена»."
            )
            return
        entry, path = selected
        sound = self.scene.objects[object_index]
        if (
            sound.primitive != "sound_source"
            or os.path.splitext(path)[1].lower() not in SOUND_EXTENSIONS
        ):
            self.status.set(
                "Выберите поддерживаемый аудиофайл и объект «Источник звука»."
            )
            return
        try:
            self.audio_player.validate(path)
        except (OSError, ValueError) as error:
            self.status.set(f"Ошибка аудиофайла: {error}")
            return
        sound.audio_path = path
        self.scene.events.publish(
            ResourceChanged(
                resource_kind="audio",
                action="assigned",
                resource_id=os.fspath(path),
                node_id=sound.id,
            )
        )
        self.audio_label.configure(text=entry["name"])
        self.status.set(f"Аудио назначено источнику: {entry['name']}")

    @_undoable
    def remove_texture(self):
        index = self._selected_index()
        if index is None:
            self.status.set("Сначала выберите объект.")
            return
        obj = self.scene.objects[index]
        if not self._supports_texture(obj):
            self.status.set(
                "У выбранного технического объекта нет отображаемого материала."
            )
            return
        previous_texture_path = obj.texture_path
        obj.texture_path = None
        self.scene.events.publish(
            ResourceChanged(
                resource_kind="texture",
                action="removed",
                resource_id=previous_texture_path or "",
                node_id=obj.id,
            )
        )
        self.texture_label.configure(text="Текстура не назначена")
        self.status.set(f"Текстура удалена у объекта: {obj.name}")

    def _selected_object_index(self):
        selection = self.object_list.curselection()
        return selection[0] if selection else None

    def _grid_axes(self, view="XZ"):
        return self.grid_view_specs[view][1]

    def _grid_step(self):
        try:
            step = float(self.grid_size.get())
        except ValueError as error:
            raise ValueError("Шаг сетки должен быть положительным числом.") from error
        if step <= 0:
            raise ValueError("Шаг сетки должен быть положительным числом.")
        return step

    def _grid_world(self, canvas, x, y):
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        scale = 32.0 * self.grid_zoom
        center_x, center_y = width / 2.0, height / 2.0
        return ((x - center_x) / scale, (center_y - y) / scale)

    def _draw_grid(self):
        if not hasattr(self, "grid_views"):
            return
        for view, canvas in self.grid_views.items():
            if canvas.winfo_exists():
                self._draw_grid_view(canvas, view)

    def _draw_grid_view(self, canvas, view):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        center_x, center_y = width / 2.0, height / 2.0
        scale = 32.0 * self.grid_zoom
        try:
            step = self._grid_step()
            interval = max(1, round(step * scale))
        except ValueError:
            interval = max(1, round(scale))
        x = center_x % interval
        while x < width:
            is_axis = abs(x - center_x) < 1
            canvas.create_line(
                x,
                0,
                x,
                height,
                fill="#8392a5"
                if is_axis
                else "#39424e"
                if round(x) % (interval * 5)
                else "#4b5664",
                width=2 if is_axis else 1,
            )
            x += interval
        y = center_y % interval
        while y < height:
            is_axis = abs(y - center_y) < 1
            canvas.create_line(
                0,
                y,
                width,
                y,
                fill="#8392a5"
                if is_axis
                else "#39424e"
                if round(y) % (interval * 5)
                else "#4b5664",
                width=2 if is_axis else 1,
            )
            y += interval
        axes = self._grid_axes(view)
        selected = self._selected_object_index()
        for index, obj in enumerate(self.scene.objects):
            is_editor_marker = obj.primitive in (
                "spawn",
                "sound_source",
                "logic_trigger",
            )
            if obj.primitive == "ground" or (not obj.visible and not is_editor_marker):
                continue
            minimum, maximum = self._object_grid_bounds(obj, axes)
            left = center_x + minimum[0] * scale
            right = center_x + maximum[0] * scale
            top = center_y - maximum[1] * scale
            bottom = center_y - minimum[1] * scale
            world_position = self.scene.world_position(obj)
            if right - left < 8:
                left, right = (
                    center_x + world_position[axes[0]] * scale - 4,
                    center_x + world_position[axes[0]] * scale + 4,
                )
            if bottom - top < 8:
                top, bottom = (
                    center_y - world_position[axes[1]] * scale - 4,
                    center_y - world_position[axes[1]] * scale + 4,
                )
            color = "#ffd166" if index == selected else "#59c3c3"
            canvas.create_rectangle(
                left,
                top,
                right,
                bottom,
                fill=color,
                outline="#111820",
                stipple="gray50",
            )
            canvas.create_text(
                left + 5,
                top - 8,
                text=obj.name,
                fill="#e7edf5",
                anchor="w",
            )
            if index == selected:
                canvas.create_rectangle(
                    right - 4,
                    bottom - 4,
                    right + 4,
                    bottom + 4,
                    fill="#ffffff",
                    outline="#111820",
                    tags=("resize-handle",),
                )
        selected_light = self._selected_light_index()
        for index, light in enumerate(self.scene.lights):
            world_position = self.scene.world_position(light)
            marker_x = center_x + world_position[axes[0]] * scale
            marker_y = center_y - world_position[axes[1]] * scale
            radius = 8 if index == selected_light else 6
            color = "#ffd166" if index == selected_light else "#ff8c69"
            canvas.create_polygon(
                marker_x,
                marker_y - radius,
                marker_x + radius,
                marker_y,
                marker_x,
                marker_y + radius,
                marker_x - radius,
                marker_y,
                fill=color,
                outline="#111820",
            )
            canvas.create_text(
                marker_x + 10,
                marker_y + 8,
                text=f"Свет: {light.name}",
                fill="#e7edf5",
                anchor="w",
            )
        canvas.create_text(
            8,
            8,
            text=f"{self.grid_view_specs[view][0]} | {1.0 / self.grid_zoom:.2f} ед. / 32 px",
            fill="#bdc7d4",
            anchor="nw",
        )

    def _object_grid_bounds(self, obj, axes):
        vertices = np.asarray(obj.vertices, dtype=np.float64)
        homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
        world = (self.scene.world_matrix(obj) @ homogeneous.T).T[:, :3]
        return world[:, axes].min(axis=0), world[:, axes].max(axis=0)

    def _grid_mouse_down(self, event, view="XZ"):
        canvas = event.widget
        axes = self._grid_axes(view)
        scale = 32.0 * self.grid_zoom
        canvas_center = (
            canvas.winfo_width() / 2.0,
            canvas.winfo_height() / 2.0,
        )
        hits = []
        if view == "XZ" and event.state & 0x0001:
            selected_index = self._selected_object_index()
            if selected_index is not None:
                terrain = self.scene.objects[selected_index]
                if terrain.primitive == "terrain":
                    world = self._grid_world(canvas, event.x, event.y)
                    self._sculpt_landscape(terrain, world[0], world[1])
                    return "break"
        for index, obj in enumerate(self.scene.objects):
            is_editor_marker = obj.primitive in (
                "spawn",
                "sound_source",
                "logic_trigger",
            )
            if obj.primitive == "ground" or (not obj.visible and not is_editor_marker):
                continue
            minimum, maximum = self._object_grid_bounds(obj, axes)
            left = canvas_center[0] + minimum[0] * scale
            right = canvas_center[0] + maximum[0] * scale
            top = canvas_center[1] - maximum[1] * scale
            bottom = canvas_center[1] - minimum[1] * scale
            world_position = self.scene.world_position(obj)
            center = (
                canvas_center[0] + world_position[axes[0]] * scale,
                canvas_center[1] - world_position[axes[1]] * scale,
            )
            if left <= event.x <= right and top <= event.y <= bottom:
                selected = index == self._selected_object_index()
                resize_hit = (
                    selected
                    and abs(event.x - right) <= 10
                    and abs(event.y - bottom) <= 10
                )
                distance = (event.x - center[0]) ** 2 + (event.y - center[1]) ** 2
                hits.append(
                    ("object", index, distance, obj, "resize" if resize_hit else "move")
                )
        for index, light in enumerate(self.scene.lights):
            world_position = self.scene.world_position(light)
            x = canvas_center[0] + world_position[axes[0]] * scale
            y = canvas_center[1] - world_position[axes[1]] * scale
            distance = (event.x - x) ** 2 + (event.y - y) ** 2
            if distance <= 13**2:
                hits.append(("light", index, distance, light, "move"))
        if hits:
            target_type = self._animation_kind()
            preferred = [hit for hit in hits if hit[0] == target_type]
            resize_handles = [hit for hit in hits if hit[4] == "resize"]
            selected_hits = [
                hit
                for hit in hits
                if hit[0] == "object" and hit[1] == self._selected_object_index()
            ]
            kind, index, _distance, target, mode = min(
                resize_handles or selected_hits or preferred or hits,
                key=lambda hit: hit[2],
            )
            self._begin_history_gesture()
            if kind == "light":
                self.light_list.selection_clear(0, tk.END)
                self.light_list.selection_set(index)
                self.light_list.event_generate("<<ListboxSelect>>")
                self.grid_drag = (
                    "light",
                    target,
                    axes,
                    self._grid_world(canvas, event.x, event.y),
                    self.scene.world_position(target),
                    None,
                    canvas,
                    view,
                )
                return "break"
            self.object_list.selection_clear(0, tk.END)
            self.object_list.selection_set(index)
            self.object_list.event_generate("<<ListboxSelect>>")
            self.grid_drag = (
                "object",
                target,
                axes,
                self._grid_world(canvas, event.x, event.y),
                self.scene.world_position(target),
                tuple(target.scale_xyz),
                canvas,
                view,
                mode,
            )
            return "break"
        self.grid_drag = None
        return

    def _finish_grid_drag(self, _event=None):
        self.grid_drag = None
        self._finish_history_gesture()

    def _grid_mouse_drag(self, event, view="XZ"):
        if self.grid_drag is None:
            return
        (
            target_type,
            obj,
            axes,
            start,
            original,
            original_scale,
            canvas,
            drag_view,
            *extra,
        ) = self.grid_drag
        if view != drag_view:
            return
        current = self._grid_world(canvas, event.x, event.y)
        if target_type == "object" and extra[0] == "resize":
            scale_xyz = list(original_scale)
            try:
                step = self._grid_step()
            except ValueError as error:
                self.status.set(str(error))
                return
            local_bounds = np.asarray(obj.vertices).max(axis=0) - np.asarray(
                obj.vertices
            ).min(axis=0)
            for plane_axis, world_axis in enumerate(axes):
                extent = float(local_bounds[world_axis]) * obj.scale
                if extent <= 1e-8:
                    continue
                delta = current[plane_axis] - start[plane_axis]
                if plane_axis == 1:
                    delta = -delta
                world_size = max(0.01, original_scale[world_axis] * extent + delta)
                if self.grid_snap.get():
                    world_size = max(step, round(world_size / step) * step)
                scale_xyz[world_axis] = world_size / extent
            obj.scale_xyz = tuple(float(value) for value in scale_xyz)
            dimensions = self._object_dimensions(obj)
            for axis, value in zip(("X", "Y", "Z"), dimensions):
                self.object_size_controls[axis].set(f"{value:.3f}")
            self.object_fields["scale_xyz"].set(
                ", ".join(f"{value:.3f}" for value in dimensions)
            )
            self._draw_grid()
            return
        position = list(original)
        try:
            step = self._grid_step()
        except ValueError as error:
            self.status.set(str(error))
            return
        for plane_axis, world_axis in enumerate(axes):
            value = original[world_axis] + current[plane_axis] - start[plane_axis]
            if self.grid_snap.get():
                value = round(value / step) * step
            position[world_axis] = value
        self.scene.set_world_position(obj, tuple(position))
        if target_type == "object":
            self._select_object()
        else:
            self._select_light()
        self._draw_grid()

    def _grid_mouse_wheel(self, event):
        self.grid_zoom = min(
            8.0, max(0.25, self.grid_zoom * (1.1 if event.delta > 0 else 1 / 1.1))
        )
        self._draw_grid()

    @_undoable
    def apply_lighting(self):
        try:
            index = self._selected_light_index()
            if index is None:
                raise ValueError("Сначала выберите или добавьте источник света.")
            light = self.scene.lights[index]
            name = self.light_fields["name"].get().strip()
            if not name:
                raise ValueError("Имя источника света не может быть пустым.")
            position = self._parse_vector(
                self.light_fields["position"].get(), "Позиция"
            )
            direction = self._parse_vector(
                self.light_fields["direction"].get(), "Направление"
            )
            color = self._parse_vector(self.light_fields["color"].get(), "Цвет света")
            kind = {"Направленный": "directional", "Точечный": "point"}.get(
                self.light_fields["kind"].get(), self.light_fields["kind"].get()
            )
            intensity = float(self.light_fields["intensity"].get())
            ambient = float(self.light_fields["ambient"].get())
            if sum(value * value for value in direction) < 1e-8:
                raise ValueError("Направление света не может быть нулевым.")
            if any(value < 0 or value > 1 for value in color):
                raise ValueError("Значения каналов цвета должны быть от 0 до 1.")
            if not 0 <= intensity <= 4 or not 0 <= ambient <= 2:
                raise ValueError(
                    "Яркость должна быть от 0 до 4, общее заполнение — от 0 до 2."
                )
            old_name = light.name
            if light.kind != kind:
                light.kind = kind
            light.name = name
            light.position = position
            light.direction = direction
            light.color = color
            light.intensity = intensity
            light.enabled = self.light_enabled.get()
            light.cast_shadows = self.light_cast_shadows.get()
            self.scene.ambient_intensity = ambient
            yaw, pitch = self._light_angles(direction)
            self.light_orientation["Азимут"].set(f"{yaw:.2f}")
            self.light_orientation["Высота"].set(f"{pitch:.2f}")
            for track in self.scene.animation_tracks:
                if (
                    track.get("target_type") == "light"
                    and track.get("target") == old_name
                ):
                    track["target"] = light.name
            self._refresh_lights(select=index)
            self.status.set(f"Параметры света обновлены: {name}")
        except ValueError as error:
            messagebox.showerror(
                "Некорректные параметры света", str(error), parent=self.root
            )

    def _refresh_lighting_fields(self):
        if not hasattr(self, "light_fields"):
            return
        self._refresh_lights(select=0 if self.scene.lights else None)
        self.light_fields["ambient"].set(f"{self.scene.ambient_intensity:.3f}")
        self.shadows_enabled.set(self.scene.shadows_enabled)
        self.shadow_map_size.set(str(self.scene.shadow_map_size))

    @_undoable
    def apply_shadow_settings(self):
        try:
            size = int(self.shadow_map_size.get())
        except ValueError:
            messagebox.showerror(
                "Некорректная карта теней",
                "Выберите разрешение 512, 1024, 2048 или 4096.",
                parent=self.root,
            )
            return
        if size not in (512, 1024, 2048, 4096):
            messagebox.showerror(
                "Некорректная карта теней",
                "Выберите разрешение 512, 1024, 2048 или 4096.",
                parent=self.root,
            )
            return
        self.scene.shadows_enabled = self.shadows_enabled.get()
        self.scene.shadow_map_size = size
        self.status.set(f"Разрешение карты теней: {size} × {size}")

    def _selected_light_index(self):
        selection = self.light_list.curselection()
        return selection[0] if selection else None

    def _refresh_lights(self, select=None):
        self.light_list.delete(0, tk.END)
        for light in self.scene.lights:
            kind = "направленный" if light.kind == "directional" else "точечный"
            state = "включён" if light.enabled else "выключен"
            self.light_list.insert(tk.END, f"{light.name} [{kind}, {state}]")
        if hasattr(self, "animation_target_picker"):
            self._refresh_animation_targets()
        if select is not None and 0 <= select < len(self.scene.lights):
            self.light_list.selection_set(select)
            self.light_list.event_generate("<<ListboxSelect>>")

    def _select_light(self, _event=None):
        index = self._selected_light_index()
        if index is None:
            self.scene.events.publish(
                SelectionChanged(selected_ids=(), active_id=None, source="none")
            )
            return
        light = self.scene.lights[index]
        self.scene.events.publish(
            SelectionChanged(
                selected_ids=(light.id,),
                active_id=light.id,
                source="light_list",
            )
        )
        self.light_fields["name"].set(light.name)
        self.light_fields["kind"].set(
            "Направленный" if light.kind == "directional" else "Точечный"
        )
        self.light_fields["position"].set(
            ", ".join(f"{value:.3f}" for value in light.position)
        )
        self.light_fields["direction"].set(
            ", ".join(f"{value:.3f}" for value in light.direction)
        )
        self.light_fields["color"].set(
            ", ".join(f"{value:.6f}" for value in light.color)
        )
        self._update_color_swatch(self.light_color_swatch, light.color)
        yaw, pitch = self._light_angles(light.direction)
        self.light_orientation["Азимут"].set(f"{yaw:.2f}")
        self.light_orientation["Высота"].set(f"{pitch:.2f}")
        self.light_fields["intensity"].set(f"{light.intensity:.3f}")
        self.light_fields["ambient"].set(f"{self.scene.ambient_intensity:.3f}")
        self.light_enabled.set(light.enabled)
        self.light_cast_shadows.set(light.cast_shadows)
        self._set_animation_kind("light")
        self._refresh_animation_targets()
        self._draw_grid()

    @staticmethod
    def _light_angles(direction):
        x, y, z = direction
        length = max(float(np.linalg.norm(direction)), 1e-8)
        return (
            float(np.degrees(np.arctan2(x, -z))),
            float(np.degrees(np.arcsin(np.clip(y / length, -1.0, 1.0)))),
        )

    @_undoable
    def _commit_light_orientation(self, _event=None):
        index = self._selected_light_index()
        if index is None:
            return
        try:
            yaw = float(self.light_orientation["Азимут"].get())
            pitch = float(self.light_orientation["Высота"].get())
        except ValueError:
            self.status.set("Введите числовые углы азимута и высоты.")
            return
        if not np.isfinite(yaw) or not np.isfinite(pitch):
            self.status.set("Углы азимута и высоты должны быть конечными числами.")
            return
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        direction = (
            float(np.sin(yaw) * np.cos(pitch)),
            float(np.sin(pitch)),
            float(-np.cos(yaw) * np.cos(pitch)),
        )
        light = self.scene.lights[index]
        light.direction = direction
        self.light_fields["direction"].set(
            ", ".join(f"{value:.3f}" for value in direction)
        )
        self._draw_grid()

    def choose_light_color(self):
        self._choose_color(
            self.light_fields["color"],
            self.light_color_swatch,
            "Цвет источника света",
        )

    @_undoable
    def add_light(self, kind):
        from .scene import Light

        number = len(self.scene.lights) + 1
        name = "Солнце" if kind == "directional" else "Точечный свет"
        light = Light(
            name=f"{name} {number}",
            kind=kind,
            position=(0.0, 3.0, 0.0),
        )
        self.scene.add_light(light)
        self._refresh_lights(select=len(self.scene.lights) - 1)
        kind_name = "направленный" if kind == "directional" else "точечный"
        self.status.set(f"Добавлен источник света: {kind_name}.")

    @_undoable
    def delete_light(self):
        index = self._selected_light_index()
        if index is None:
            return
        light = self.scene.lights[index]
        removed_nodes = self.scene.remove_node(light, cascade=True)
        self._remove_node_references(removed_nodes)
        self._refresh_lights(select=min(index, len(self.scene.lights) - 1))
        self._refresh_objects()
        self.status.set(f"Удалён источник света: {light.name}")

    def _remove_node_references(self, nodes):
        object_names = {node.name for node in nodes if isinstance(node, MeshInstance)}
        light_names = {node.name for node in nodes if isinstance(node, Light)}
        self.scene.animation_tracks = [
            track
            for track in self.scene.animation_tracks
            if not (
                track.get("target_type") == "object"
                and track.get("target") in object_names
                or track.get("target_type") == "light"
                and track.get("target") in light_names
                or track.get("target_type") == "bone"
                and any(
                    str(track.get("target", "")).startswith(f"{name}::")
                    for name in object_names
                )
            )
        ]
        self.scene.logic_actions = [
            rule
            for rule in self.scene.logic_actions
            if rule.get("target") not in object_names
        ]

    @staticmethod
    def _parse_vector(value, label):
        try:
            parts = tuple(float(part) for part in re.split(r"[\s,]+", value.strip()))
        except ValueError as error:
            raise ValueError(
                f"{label}: введите три числа через запятую или пробел"
            ) from error
        if len(parts) != 3:
            raise ValueError(f"{label}: требуется ровно три значения")
        if not all(np.isfinite(part) for part in parts):
            raise ValueError(f"{label}: значения должны быть конечными числами")
        return parts

    @_undoable
    def apply_object(self):
        index = self._selected_index()
        if index is None:
            self.status.set("Сначала выберите объект.")
            return
        try:
            obj = self.scene.objects[index]
            name = self.object_fields["name"].get().strip()
            if not name:
                raise ValueError("Имя объекта не может быть пустым.")
            position = self._parse_vector(
                self.object_fields["position"].get(), "Позиция"
            )
            rotation = self._parse_vector(
                self.object_fields["rotation"].get(), "Поворот"
            )
            dimensions = self._parse_vector(
                self.object_fields["scale_xyz"].get(), "Размер по осям"
            )
            color = self._parse_vector(self.object_fields["color"].get(), "Цвет")
            scale = float(self.object_fields["scale"].get())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Масштаб должен быть больше нуля.")
            if any(not np.isfinite(value) or value < 0 for value in dimensions):
                raise ValueError("Размеры по осям должны быть неотрицательными.")
            base_dimensions = self._object_base_dimensions(obj)
            if any(
                dimension > 1e-8 and base <= 1e-8
                for dimension, base in zip(dimensions, base_dimensions)
            ):
                raise ValueError(
                    "Нельзя задать размер по оси, у которой у модели нет геометрии."
                )
            if any(channel < 0 or channel > 1 for channel in color):
                raise ValueError("Значения каналов цвета должны быть от 0 до 1.")
            old_name = obj.name
            obj.name = self._unique_name(name, ignored_index=index)
            obj.position = position
            obj.rotation = tuple(
                value * 3.141592653589793 / 180.0 for value in rotation
            )
            obj.scale = scale
            obj.scale_xyz = tuple(
                dimension / max(base, 1e-8) if base > 1e-8 else current
                for dimension, base, current in zip(
                    dimensions, base_dimensions, obj.scale_xyz
                )
            )
            obj.color = color
            if obj.primitive == "npc":
                obj.npc_behavior = (
                    "follow"
                    if self.npc_behavior.get() == "Следовать за игроком"
                    else "idle"
                )
            for node in self.scene.logic_actions:
                if node.get("target") == old_name:
                    node["target"] = obj.name
            for track in self.scene.animation_tracks:
                if (
                    track.get("target_type") == "object"
                    and track.get("target") == old_name
                ):
                    track["target"] = obj.name
                elif track.get("target_type") == "bone" and str(
                    track.get("target", "")
                ).startswith(f"{old_name}::"):
                    track["target"] = (
                        f"{obj.name}::{str(track['target']).split('::', 1)[1]}"
                    )
            self._refresh_objects(select=index)
            for axis, value in zip(("X", "Y", "Z"), rotation):
                self.object_axis_controls[axis].set(f"{value:.2f}")
            for axis, value in zip(("X", "Y", "Z"), self._object_dimensions(obj)):
                self.object_size_controls[axis].set(f"{value:.3f}")
            self._update_color_swatch(self.object_color_swatch, color)
            self._refresh_logic()
            self.status.set(f"Объект обновлён: {name}")
        except ValueError as error:
            messagebox.showerror(
                "Некорректные параметры объекта", str(error), parent=self.root
            )

    @_undoable
    def add_box(self):
        from .app import make_box

        index = len(self.scene.objects) + 1
        while any(obj.name == f"Box {index}" for obj in self.scene.objects):
            index += 1
        mesh = make_box(position=(0.0, 0.85, 0.0))
        mesh.name = f"Box {index}"
        self.scene.add_mesh(mesh)
        self._refresh_objects(select=len(self.scene.objects) - 1)
        self.status.set("Ящик добавлен.")

    @_undoable
    def add_spawn(self):
        from .app import make_box

        mesh = make_box(0.1, color=(0.2, 0.85, 0.55), position=(0.0, 0.0, 6.0))
        mesh.name = self._unique_name("Точка появления игрока")
        mesh.primitive = "spawn"
        mesh.visible = False
        self.scene.add_mesh(mesh)
        self._refresh_objects(select=len(self.scene.objects) - 1)
        self.status.set(
            "Точка появления добавлена. Её центр задаёт положение ног игрока."
        )

    @_undoable
    def add_landscape(self):
        divisions = 24
        half_size = 12.0
        vertices = []
        uvs = []
        for z_index in range(divisions + 1):
            z = -half_size + 2.0 * half_size * z_index / divisions
            for x_index in range(divisions + 1):
                x = -half_size + 2.0 * half_size * x_index / divisions
                vertices.append((x, 0.0, z))
                uvs.append((x_index / divisions, z_index / divisions))
        indices = []
        stride = divisions + 1
        for z_index in range(divisions):
            for x_index in range(divisions):
                first = z_index * stride + x_index
                next_row = first + stride
                indices.extend(
                    (first, next_row, next_row + 1, first, next_row + 1, first + 1)
                )
        vertices = np.asarray(vertices, dtype=np.float32)
        mesh = MeshInstance(
            name=self._unique_name("Ландшафт"),
            vertices=vertices,
            normals=np.tile((0.0, 1.0, 0.0), (len(vertices), 1)).astype(np.float32),
            uvs=np.asarray(uvs, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.int32),
            color=(0.34, 0.48, 0.28),
            primitive="terrain",
        )
        self.scene.add_mesh(mesh)
        self._refresh_objects(select=len(self.scene.objects) - 1)
        self.status.set(
            "Ландшафт создан. Выберите его и используйте Shift + щелчок на верхней сетке."
        )

    @_undoable
    def auto_rig_selected(self):
        index = self._selected_index()
        if index is None:
            self.status.set("Сначала выберите модель для авто-риггинга.")
            return
        obj = self.scene.objects[index]
        if obj.primitive in ("spawn", "sound_source", "logic_trigger"):
            self.status.set("Технические объекты нельзя снабдить скелетом.")
            return
        vertices = np.asarray(obj.vertices, dtype=np.float32)
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        center = (minimum + maximum) / 2
        extent = maximum - minimum
        height = max(float(extent[1]), float(extent[0]), float(extent[2]), 1e-3)
        leg_offset = max(float(extent[0]) * 0.18, height * 0.035)
        hand_x = max(float(extent[0]) * 0.45, height * 0.12)
        torso_x = max(float(extent[0]) * 0.08, height * 0.025)
        z = float(center[2])
        y = float(minimum[1])
        bones = [
            {
                "name": "Корень",
                "parent": None,
                "head": [float(center[0]), y, z],
                "tail": [float(center[0]), y + height * 0.35, z],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "name": "Спина",
                "parent": "Корень",
                "head": [float(center[0]), y + height * 0.30, z],
                "tail": [float(center[0]), y + height * 0.72, z],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "name": "Голова",
                "parent": "Спина",
                "head": [float(center[0]), y + height * 0.68, z],
                "tail": [float(center[0]), y + height * 0.98, z],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "name": "Рука левая",
                "parent": "Спина",
                "head": [float(center[0]) - torso_x, y + height * 0.66, z],
                "tail": [float(center[0]) - hand_x, y + height * 0.61, z],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "name": "Рука правая",
                "parent": "Спина",
                "head": [float(center[0]) + torso_x, y + height * 0.66, z],
                "tail": [float(center[0]) + hand_x, y + height * 0.61, z],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "name": "Нога левая",
                "parent": "Корень",
                "head": [float(center[0]) - leg_offset, y + height * 0.32, z],
                "tail": [float(center[0]) - leg_offset, y, z],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "name": "Нога правая",
                "parent": "Корень",
                "head": [float(center[0]) + leg_offset, y + height * 0.32, z],
                "tail": [float(center[0]) + leg_offset, y, z],
                "rotation": [0.0, 0.0, 0.0],
            },
        ]
        scores = []
        for bone in bones:
            start = np.asarray(bone["head"], dtype=np.float32)
            end = np.asarray(bone["tail"], dtype=np.float32)
            segment = end - start
            amount = np.clip(
                ((vertices - start) @ segment) / max(float(segment @ segment), 1e-8),
                0.0,
                1.0,
            )
            closest = start + amount[:, None] * segment
            distance_squared = np.sum((vertices - closest) ** 2, axis=1)
            scores.append(1.0 / (distance_squared + (height * 0.08) ** 2))
        weight_matrix = np.stack(scores, axis=1)
        weight_matrix /= weight_matrix.sum(axis=1, keepdims=True)
        obj.rig = {
            "bones": bones,
            "base_vertices": vertices.copy(),
            "weights": weight_matrix.astype(np.float32),
        }
        self._refresh_animation_targets()
        self.status.set(
            f"Авто-риг создан для «{obj.name}»: 7 костей и веса вершин. Настраивайте кости через кривые."
        )

    @_undoable
    def add_sound_source(self):
        from .app import make_box

        mesh = make_box(0.1, color=(0.35, 0.75, 1.0))
        mesh.name = self._unique_name("Источник звука")
        mesh.primitive = "sound_source"
        mesh.visible = False
        self.scene.add_mesh(mesh)
        self._refresh_objects(select=len(self.scene.objects) - 1)
        self.status.set(
            "Добавлен источник звука. Назначьте аудиофайл и задайте радиус."
        )

    @_undoable
    def add_logic_trigger(self):
        from .app import make_box

        mesh = make_box(1.0, color=(1.0, 0.68, 0.25))
        mesh.name = self._unique_name("Триггер логики")
        mesh.primitive = "logic_trigger"
        mesh.visible = False
        self.scene.add_mesh(mesh)
        self._refresh_objects(select=len(self.scene.objects) - 1)
        self.status.set(
            "Добавлен технический объект. Используйте его как цель правила «Когда игрок рядом»."
        )

    @_undoable
    def add_npc(self):
        from .app import make_box

        parts = (
            make_box(0.48, position=(0.0, 0.85, 0.0)),
            make_box(0.34, position=(0.0, 1.32, 0.0)),
            make_box(0.2, position=(-0.34, 0.86, 0.0)),
            make_box(0.2, position=(0.34, 0.86, 0.0)),
            make_box(0.2, position=(-0.13, 0.28, 0.0)),
            make_box(0.2, position=(0.13, 0.28, 0.0)),
        )
        vertices = np.concatenate([part.vertices for part in parts])
        normals = np.concatenate([part.normals for part in parts])
        uvs = np.concatenate([part.uvs for part in parts])
        indices = np.concatenate(
            [
                part.indices + sum(len(previous.vertices) for previous in parts[:index])
                for index, part in enumerate(parts)
            ]
        )
        mesh = MeshInstance(
            name=self._unique_name("NPC"),
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            indices=indices.astype(np.int32),
            color=(0.68, 0.76, 0.88),
            position=(-4.0, 0.0, -4.0),
            primitive="npc",
            npc_behavior="follow",
        )
        self.scene.add_mesh(mesh)
        self._refresh_objects(select=len(self.scene.objects) - 1)
        self.status.set("NPC создан. В режиме теста по умолчанию следует за игроком.")

    @staticmethod
    def _recalculate_mesh_normals(mesh):
        triangles = np.asarray(mesh.vertices, dtype=np.float32)[
            np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
        ]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normals = np.zeros_like(mesh.vertices, dtype=np.float32)
        for corner in range(3):
            np.add.at(normals, mesh.indices.reshape(-1, 3)[:, corner], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-8
        normals[valid] /= lengths[valid, None]
        mesh.normals = normals

    @_undoable
    def _sculpt_landscape(self, obj, world_x, world_z):
        try:
            radius = float(self.terrain_brush_radius.get())
            strength = float(self.terrain_brush_strength.get())
            if (
                not np.isfinite(radius)
                or radius <= 0
                or not np.isfinite(strength)
                or strength == 0
            ):
                raise ValueError(
                    "Радиус должен быть положительным, высота — ненулевой."
                )
        except ValueError as error:
            self.status.set(f"Параметры кисти: {error}")
            return
        vertices = np.array(obj.vertices, dtype=np.float32, copy=True)
        try:
            inverse_world = np.linalg.inv(self.scene.world_matrix(obj))
        except np.linalg.LinAlgError:
            self.status.set(
                "Нельзя редактировать ландшафт с вырожденным преобразованием."
            )
            return
        local_brush_center = inverse_world @ np.asarray(
            (world_x, 0.0, world_z, 1.0), dtype=np.float32
        )
        local_x, local_z = local_brush_center[[0, 2]]
        world_scale = np.linalg.norm(self.scene.world_matrix(obj)[:3, :3], axis=0)
        local_radius = radius / max(
            float((world_scale[0] + world_scale[2]) * 0.5), 1e-8
        )
        distances = np.hypot(vertices[:, 0] - local_x, vertices[:, 2] - local_z)
        falloff = np.clip(1.0 - distances / local_radius, 0.0, 1.0) ** 2
        if not np.any(falloff):
            return
        vertices[:, 1] += (strength * falloff).astype(np.float32)
        obj.vertices = vertices
        self._recalculate_mesh_normals(obj)
        self.status.set(
            f"Рельеф изменён кистью: радиус {radius:g}, высота {strength:g}"
        )
        self._draw_grid()

    def toggle_playtest(self):
        if not self.playtest_active:
            problems = self._scene_event_consumers.validation_problems
            if problems:
                messagebox.showerror(
                    "Проверка карты не пройдена",
                    self._validation_summary(problems),
                    parent=self.root,
                )
                self.status.set(f"Тест не запущен: найдено проблем — {len(problems)}.")
                return
        self.playtest_active = not self.playtest_active
        self.timeline_playing = False
        self.play_button.configure(text="Воспроизвести")
        self.playtest_button.configure(
            text="Остановить" if self.playtest_active else "Играть / Стоп"
        )
        self.status.set(
            "Тест запущен: WASD — движение, мышь — обзор, Space — прыжок, Esc — выход."
            if self.playtest_active
            else "Тест остановлен."
        )

    def import_model(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Импорт модели или архива",
            filetypes=(
                (
                    "Модели и ZIP-архивы",
                    "*.obj *.glb *.gltf *.stl *.ply *.off *.xyz *.zip",
                ),
                ("Поддерживаемые модели", "*.obj *.glb *.gltf *.stl *.ply *.off *.xyz"),
                ("ZIP-архивы", "*.zip"),
            ),
        )
        if not path:
            return
        try:
            if os.path.splitext(path)[1].lower() == ".zip":
                root, models = self._extract_model_archive(path)
                selected = self._choose_model_path(models, root)
                if selected:
                    self._import_model_file(selected, root)
            else:
                self._import_model_file(path, os.path.dirname(path))
        except (OSError, ValueError, TypeError, zipfile.BadZipFile) as error:
            messagebox.showerror(
                "Не удалось импортировать модель", str(error), parent=self.root
            )

    def import_model_folder(self):
        root = filedialog.askdirectory(
            parent=self.root, title="Выберите папку с моделью и текстурами"
        )
        if not root:
            return
        models = self._find_model_files(root)
        if not models:
            messagebox.showerror(
                "Модель не найдена",
                "В выбранной папке и её подпапках не найдено поддерживаемых моделей.",
                parent=self.root,
            )
            return
        path = self._choose_model_path(models, root)
        if path:
            try:
                self._import_model_file(path, root)
            except (OSError, ValueError, TypeError) as error:
                messagebox.showerror(
                    "Не удалось импортировать модель", str(error), parent=self.root
                )

    @staticmethod
    def _find_model_files(root):
        models = []
        for directory, subdirectories, filenames in os.walk(root):
            subdirectories[:] = [
                name for name in subdirectories if name not in (".git", "__pycache__")
            ]
            models.extend(
                os.path.join(directory, filename)
                for filename in filenames
                if os.path.splitext(filename)[1].lower() in MODEL_EXTENSIONS
            )
        return sorted(models, key=str.casefold)

    def _extract_model_archive(self, archive_path):
        import_root = os.path.join(self.asset_cache.name, "imports", uuid.uuid4().hex)
        os.makedirs(import_root)
        models = []
        total_size = 0
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > 20000:
                raise ValueError("В архиве слишком много файлов.")
            for member in members:
                normalized = member.filename.replace("\\", "/")
                relative = PurePosixPath(normalized)
                if (
                    not normalized
                    or relative.is_absolute()
                    or any(part in ("", ".", "..") for part in relative.parts)
                    or (relative.parts and ":" in relative.parts[0])
                ):
                    raise ValueError(
                        f"В архиве обнаружен небезопасный путь: {member.filename}"
                    )
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(
                        f"Архив содержит ссылку, извлечение запрещено: {member.filename}"
                    )
                if member.is_dir():
                    continue
                total_size += member.file_size
                if total_size > 1024 * 1024 * 1024:
                    raise ValueError(
                        "Размер файлов в архиве превышает ограничение 1 ГБ."
                    )
                destination = os.path.abspath(
                    os.path.join(import_root, *relative.parts)
                )
                if os.path.commonpath(
                    (os.path.abspath(import_root), destination)
                ) != os.path.abspath(import_root):
                    raise ValueError(
                        f"В архиве обнаружен небезопасный путь: {member.filename}"
                    )
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with archive.open(member) as source, open(destination, "wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                if os.path.splitext(destination)[1].lower() in MODEL_EXTENSIONS:
                    models.append(destination)
        if not models:
            raise ValueError("В ZIP-архиве не найдена поддерживаемая 3D-модель.")
        return import_root, sorted(models, key=str.casefold)

    def _choose_model_path(self, models, root):
        if len(models) == 1:
            return models[0]
        dialog = tk.Toplevel(self.root)
        dialog.title("Выберите модель")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(dialog, text="В источнике найдено несколько моделей:").pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        listing = tk.Listbox(
            dialog, width=70, height=min(12, len(models)), exportselection=False
        )
        listing.pack(fill="both", expand=True, padx=10, pady=4)
        for model in models:
            listing.insert(tk.END, os.path.relpath(model, root))
        listing.selection_set(0)
        result = {"path": None}

        def choose(_event=None):
            selection = listing.curselection()
            if selection:
                result["path"] = models[selection[0]]
                dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Button(buttons, text="Импортировать", command=choose).pack(side="right")
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(
            side="right", padx=4
        )
        listing.bind("<Double-Button-1>", choose)
        self.root.wait_window(dialog)
        return result["path"]

    def _resolve_model_texture(self, model_path, imported, search_root):
        if imported.texture_image is not None:
            texture_directory = os.path.join(self.asset_cache.name, "imported-textures")
            os.makedirs(texture_directory, exist_ok=True)
            texture_path = os.path.join(texture_directory, f"{uuid.uuid4().hex}.png")
            imported.texture_image.convert("RGBA").save(texture_path)
            return texture_path

        referenced = self._find_material_texture(model_path, search_root)
        if referenced:
            return referenced
        model_stem = os.path.splitext(os.path.basename(model_path))[0].casefold()
        candidates = []
        for directory, _subdirectories, filenames in os.walk(search_root):
            candidates.extend(
                os.path.join(directory, filename)
                for filename in filenames
                if os.path.splitext(filename)[1].lower() in TEXTURE_EXTENSIONS
            )
        exact = [
            path
            for path in candidates
            if os.path.splitext(os.path.basename(path))[0].casefold() == model_stem
        ]
        if len(exact) == 1:
            return exact[0]
        related = [
            path
            for path in candidates
            if os.path.splitext(os.path.basename(path))[0]
            .casefold()
            .startswith(model_stem + "_")
            or os.path.splitext(os.path.basename(path))[0]
            .casefold()
            .startswith(model_stem + "-")
        ]
        if len(related) == 1:
            return related[0]
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _find_material_texture(model_path, search_root):
        if os.path.splitext(model_path)[1].lower() != ".obj":
            return None
        try:
            with open(
                model_path, "r", encoding="utf-8-sig", errors="replace"
            ) as source:
                material_libraries = [
                    line.partition("#")[0].strip().split(maxsplit=1)[1]
                    for line in source
                    if line.partition("#")[0].strip().lower().startswith("mtllib ")
                ]
        except OSError:
            return None

        root = os.path.abspath(search_root)
        for library in material_libraries:
            for library_name in shlex.split(library):
                material_path = os.path.join(os.path.dirname(model_path), library_name)
                if not os.path.isfile(material_path):
                    continue
                try:
                    with open(
                        material_path, "r", encoding="utf-8-sig", errors="replace"
                    ) as source:
                        maps = [
                            line.partition("#")[0].strip().split(maxsplit=1)[1]
                            for line in source
                            if line.partition("#")[0]
                            .strip()
                            .lower()
                            .startswith("map_kd ")
                        ]
                except OSError:
                    continue
                for map_value in maps:
                    tokens = shlex.split(map_value)
                    index = 0
                    while index < len(tokens) and tokens[index].startswith("-"):
                        option = tokens[index].lower()
                        index += 1
                        if option in ("-o", "-s", "-t"):
                            consumed = 0
                            while (
                                index < len(tokens)
                                and consumed < 3
                                and _is_float(tokens[index])
                            ):
                                index += 1
                                consumed += 1
                        else:
                            option_values = {
                                "-mm": 2,
                                "-blendu": 1,
                                "-blendv": 1,
                                "-boost": 1,
                                "-bm": 1,
                                "-cc": 1,
                                "-clamp": 1,
                                "-colorspace": 1,
                                "-imfchan": 1,
                                "-texres": 1,
                                "-type": 1,
                            }
                            index += option_values.get(option, 0)
                    if index >= len(tokens):
                        continue
                    texture_name = " ".join(tokens[index:])
                    candidates = (
                        os.path.abspath(
                            os.path.join(os.path.dirname(material_path), texture_name)
                        ),
                        os.path.abspath(
                            os.path.join(os.path.dirname(model_path), texture_name)
                        ),
                        os.path.abspath(os.path.join(root, texture_name)),
                    )
                    for candidate in candidates:
                        try:
                            is_inside_root = (
                                os.path.commonpath((root, candidate)) == root
                            )
                        except ValueError:
                            continue
                        if (
                            is_inside_root
                            and os.path.isfile(candidate)
                            and os.path.splitext(candidate)[1].lower()
                            in TEXTURE_EXTENSIONS
                        ):
                            return candidate
        return None

    @_undoable
    def _import_model_file(self, path, search_root, display_name=None):
        self.clear_asset_preview()
        imported = load_model(path)
        texture_path = self._resolve_model_texture(path, imported, search_root)
        name = self._unique_name(os.path.splitext(imported.name)[0])
        minimum = imported.vertices.min(axis=0)
        maximum = imported.vertices.max(axis=0)
        bounds = maximum - minimum
        vertices = imported.vertices - (minimum + maximum) / 2.0
        scale = 2.0 / max(float(max(bounds)), 1e-4)
        half_extents = bounds * scale / 2.0
        step = max(2.5, float(max(half_extents[[0, 2]])) * 2.0 + 1.0)
        candidates = [(0.0, 0.0)]
        for ring in range(1, 5):
            candidates.extend(
                (x * step, z * step)
                for x, z in (
                    (ring, 0),
                    (-ring, 0),
                    (0, ring),
                    (0, -ring),
                    (ring, ring),
                    (-ring, ring),
                    (ring, -ring),
                    (-ring, -ring),
                )
            )
        position_xz = (0.0, 0.0)
        for candidate in candidates:
            candidate_min = (
                candidate[0] - half_extents[0],
                candidate[1] - half_extents[2],
            )
            candidate_max = (
                candidate[0] + half_extents[0],
                candidate[1] + half_extents[2],
            )
            occupied = False
            for existing in self.scene.objects:
                if not existing.visible or existing.primitive in (
                    "ground",
                    "spawn",
                    "sound_source",
                ):
                    continue
                minimum_xz, maximum_xz = self._object_grid_bounds(existing, (0, 2))
                if (
                    candidate_min[0] < maximum_xz[0]
                    and candidate_max[0] > minimum_xz[0]
                    and candidate_min[1] < maximum_xz[1]
                    and candidate_max[1] > minimum_xz[1]
                ):
                    occupied = True
                    break
            if not occupied:
                position_xz = candidate
                break
        self.scene.add_mesh(
            MeshInstance(
                name=name,
                vertices=vertices,
                normals=imported.normals,
                uvs=imported.uvs,
                indices=imported.indices,
                color=imported.color,
                position=(
                    position_xz[0],
                    max(float(bounds[1]) * scale / 2.0, 0.1),
                    position_xz[1],
                ),
                scale=scale,
                source_path=path,
                texture_path=texture_path,
            )
        )
        imported_object = self.scene.objects[-1]
        self.scene.events.publish(
            ResourceChanged(
                resource_kind="model",
                action="loaded",
                resource_id=os.fspath(path),
                node_id=imported_object.id,
            )
        )
        self._refresh_objects(select=len(self.scene.objects) - 1)
        label = display_name or os.path.basename(path)
        if texture_path:
            self.status.set(f"Импортирована модель с текстурой: {label}")
        else:
            self.status.set(
                f"Модель импортирована: {label}. Совпадающая текстура не найдена."
            )

    @_undoable
    def delete_object(self):
        index = self._selected_index()
        if index is None:
            return
        removed_nodes = self.scene.remove_node(self.scene.objects[index], cascade=True)
        removed_names = [
            node.name for node in removed_nodes if isinstance(node, MeshInstance)
        ]
        self._remove_node_references(removed_nodes)
        self._refresh_objects(select=min(index, len(self.scene.objects) - 1))
        self._refresh_lights()
        self._refresh_logic()
        self.status.set(
            f"Удалён объект и его дочерние узлы: {', '.join(sorted(removed_names))}"
        )

    def new_project(self):
        if not messagebox.askyesno(
            "Новая карта", "Создать пустую карту?", parent=self.root
        ):
            return
        self.scene = Scene()
        self.scene.add_ground()
        self._bind_scene_event_consumers()
        self._clear_history()
        self.scene.notify_changed("new_project")
        self.project_path = None
        self.playhead = 0.0
        self.timeline_playing = False
        self._refresh_objects()
        self._refresh_lighting_fields()
        self._refresh_logic()
        self._refresh_keyframes()
        self.status.set("Создана новая карта.")

    def open_project(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Открыть карту",
            filetypes=(("Карта DEc", "*.DE"), ("Все файлы", "*.*")),
        )
        if not path:
            return
        try:
            self.scene = load_project(path)
            self._bind_scene_event_consumers()
            self._dirty = False
            self._autosaved_revision = self._scene_revision
            self.project_path = path
            self.playhead = 0.0
            self.timeline_playing = False
            self._refresh_objects()
            self._refresh_lighting_fields()
            self._refresh_logic()
            self._refresh_keyframes()
            self.status.set(f"Карта открыта: {path}")
            self._remove_recovery_copy()
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Не удалось открыть карту", str(error), parent=self.root
            )

    def save_current_project(self):
        problems = self._scene_event_consumers.validation_problems
        if problems and not messagebox.askyesno(
            "Проблемы карты",
            "Перед сохранением найдены проблемы:\n\n"
            f"{self._validation_summary(problems)}\n\n"
            "Продолжить сохранение?",
            parent=self.root,
        ):
            return
        path = self.project_path or filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить карту",
            defaultextension=".DE",
            filetypes=(("Карта DEc", "*.DE"),),
        )
        if not path:
            return
        try:
            save_project(path, self.scene)
            self.project_path = path
            self._dirty = False
            self._autosaved_revision = self._scene_revision
            self._scene_event_consumers.mark_saved()
            self.status.set(f"Карта сохранена: {path}")
            self._remove_recovery_copy()
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Не удалось сохранить карту", str(error), parent=self.root
            )

    def _refresh_logic(self):
        self.logic_list.delete(0, tk.END)
        triggers = {
            "on_start": "При запуске карты",
            "on_timer": "По таймеру",
            "on_key": "При нажатии клавиши",
            "on_player_near": "Когда игрок рядом",
        }
        actions = {
            "set_position": "Задать позицию",
            "add_rotation": "Добавить поворот",
            "set_color": "Задать цвет",
            "toggle_visibility": "Переключить видимость",
            "set_scale": "Задать размер",
        }
        for node in self.scene.logic_actions:
            trigger = triggers.get(node.get("trigger"), node.get("trigger"))
            action = actions.get(node.get("action"), node.get("action"))
            detail = f"{trigger} → {action} [{node.get('target')}]"
            if node.get("trigger") == "on_key":
                detail += f" | клавиша: {node.get('key', 'E')}"
            if node.get("trigger") == "on_timer":
                detail += f" | задержка: {node.get('delay', 1)} с"
            if node.get("trigger") == "on_player_near":
                detail += f" | радиус: {node.get('radius', 3)}"
            if node.get("action") != "toggle_visibility":
                value = node.get("value")
                if (
                    node.get("action") == "add_rotation"
                    and isinstance(value, (list, tuple))
                    and len(value) == 3
                ):
                    value = [round(float(angle) * 180.0 / np.pi, 2) for angle in value]
                detail += f" | значение: {value}"
            self.logic_list.insert(tk.END, detail)
        self.logic_target["values"] = tuple(obj.name for obj in self.scene.objects)

    def _update_logic_controls(self):
        if not hasattr(self, "logic_fields"):
            return
        action = self.logic_fields["action"].get()
        if action == "Задать цвет":
            self.logic_value_label.configure(text="Цвет объекта")
            self.logic_value_control.pack_forget()
            self.logic_color_swatch.pack(side="left", padx=(4, 0))
            self.logic_color_button.pack(side="left", padx=4)
            try:
                color = self._parse_vector(self.logic_fields["value"].get(), "Цвет")
            except ValueError:
                color = (1.0, 1.0, 1.0)
            self._update_color_swatch(self.logic_color_swatch, color)
        else:
            self.logic_color_swatch.pack_forget()
            self.logic_color_button.pack_forget()
            if not self.logic_value_control.winfo_manager():
                self.logic_value_control.pack(side="left", fill="x", expand=True)
            if action == "Добавить поворот":
                self.logic_value_label.configure(text="Углы X, Y, Z (градусы)")
            elif action == "Переключить видимость":
                self.logic_value_label.configure(text="Значение не используется")
            elif action == "Задать размер":
                self.logic_value_label.configure(text="Размер по X, Y, Z")
            else:
                self.logic_value_label.configure(text="Позиция X, Y, Z")

    def choose_logic_color(self):
        self._choose_color(
            self.logic_fields["value"],
            self.logic_color_swatch,
            "Цвет для правила логики",
        )

    @_undoable
    def add_logic(self):
        trigger = {
            "При запуске карты": "on_start",
            "По таймеру": "on_timer",
            "При нажатии клавиши": "on_key",
            "Когда игрок рядом": "on_player_near",
        }.get(self.logic_fields["trigger"].get())
        action = {
            "Задать позицию": "set_position",
            "Добавить поворот": "add_rotation",
            "Задать цвет": "set_color",
            "Переключить видимость": "toggle_visibility",
            "Задать размер": "set_scale",
        }.get(self.logic_fields["action"].get())
        target = self.logic_fields["target"].get()
        if trigger is None or action is None:
            messagebox.showerror(
                "Некорректное правило", "Выберите событие и действие.", parent=self.root
            )
            return
        if not any(obj.name == target for obj in self.scene.objects):
            messagebox.showerror(
                "Некорректное правило",
                "Выберите существующий объект-цель.",
                parent=self.root,
            )
            return
        node = {"trigger": trigger, "action": action, "target": target, "value": []}
        try:
            if trigger == "on_player_near":
                radius = float(self.logic_fields["radius"].get())
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError(
                        "Радиус срабатывания должен быть положительным числом."
                    )
                node["radius"] = radius
            if action != "toggle_visibility":
                value_label = "Цвет" if action == "set_color" else "Значение"
                value = self._parse_vector(
                    self.logic_fields["value"].get(), value_label
                )
                if action == "set_color" and any(
                    channel < 0 or channel > 1 for channel in value
                ):
                    raise ValueError("Значения каналов цвета должны быть от 0 до 1.")
                if action == "set_scale" and any(size <= 0 for size in value):
                    raise ValueError("Размер по каждой оси должен быть больше нуля.")
                if action == "add_rotation":
                    value = [float(np.radians(angle)) for angle in value]
                node["value"] = list(value)
            if trigger == "on_key":
                key = self.logic_fields["key"].get().strip().upper()
                if key not in KEY_CODES:
                    raise ValueError("Допустимые клавиши: W, A, S, D, E, F или SPACE.")
                node["key"] = key
            elif trigger == "on_timer":
                delay = float(self.logic_fields["delay"].get())
                if delay < 0:
                    raise ValueError("Задержка таймера не может быть отрицательной.")
                node["delay"] = delay
        except ValueError as error:
            messagebox.showerror("Некорректное правило", str(error), parent=self.root)
            return
        self.scene.logic_actions.append(node)
        self._refresh_logic()
        self.status.set("Правило логики добавлено.")

    @_undoable
    def delete_logic(self):
        selection = self.logic_list.curselection()
        if not selection:
            return
        del self.scene.logic_actions[selection[0]]
        self._refresh_logic()

    def _set_playhead(self, value):
        self.playhead = float(value)
        if hasattr(self, "key_time_value"):
            self.key_time_value.set(f"{self.playhead:.3f}")
        if not self.timeline_playing:
            self._apply_timeline_pose()

    def commit_playhead(self):
        try:
            time_seconds = float(self.key_time_value.get())
            fps = int(self.timeline_fps.get())
            if time_seconds < 0 or fps <= 0:
                raise ValueError("Время должно быть неотрицательным.")
            time_seconds = round(time_seconds * fps) / fps
        except ValueError as error:
            messagebox.showerror("Некорректное время", str(error), parent=self.root)
            return
        self.playhead = time_seconds
        self.timeline_value.set(time_seconds)
        self.key_time_value.set(f"{time_seconds:.3f}")
        self._apply_timeline_pose()

    @_undoable
    def move_selected_key(self):
        selection = self.keyframe_list.curselection()
        if not selection:
            self.status.set("Выберите ключ анимации.")
            return
        try:
            new_time = float(self.key_time_value.get())
            fps = int(self.timeline_fps.get())
            if new_time < 0 or fps <= 0:
                raise ValueError("Время должно быть неотрицательным.")
            new_time = round(new_time * fps) / fps
        except ValueError as error:
            messagebox.showerror("Некорректное время", str(error), parent=self.root)
            return
        track = self._current_animation_track()
        if track:
            keys = sorted(track.get("keys", []), key=lambda key: float(key["time"]))
            if selection[0] >= len(keys):
                return
            old_time = float(keys[selection[0]]["time"])
            if any(
                abs(float(key["time"]) - new_time) < 1e-5
                for index, key in enumerate(keys)
                if index != selection[0]
            ):
                messagebox.showerror(
                    "Время уже занято",
                    "В этом треке уже есть ключ на выбранном времени.",
                    parent=self.root,
                )
                return
            keys[selection[0]]["time"] = new_time
            track["keys"] = sorted(keys, key=lambda key: float(key["time"]))
            if track.get("target_type") == "camera":
                legacy = next(
                    (
                        frame
                        for frame in self.scene.camera_keyframes
                        if abs(float(frame["time"]) - old_time) < 1e-5
                    ),
                    None,
                )
                if legacy is not None:
                    legacy["time"] = new_time
                    self.scene.camera_keyframes.sort(
                        key=lambda frame: float(frame["time"])
                    )
        elif self._animation_kind() == "camera":
            frames = sorted(
                self.scene.camera_keyframes, key=lambda key: float(key["time"])
            )
            if selection[0] >= len(frames):
                return
            frames[selection[0]]["time"] = new_time
            self.scene.camera_keyframes = sorted(
                frames, key=lambda frame: float(frame["time"])
            )
        self.playhead = new_time
        self.timeline_value.set(new_time)
        self.key_time_value.set(f"{new_time:.3f}")
        self._refresh_keyframes()
        self.status.set(f"Время ключа изменено: {new_time:.3f} сек")

    def add_camera_key(self):
        self._set_animation_kind("camera")
        self._refresh_animation_targets()
        self.capture_selected_key()

    def add_camera_checkpoint(self):
        self.capture_selected_checkpoint()

    def _store_camera_key(self, time_seconds):
        captured = capture_camera(self.camera, time_seconds)
        self.scene.camera_keyframes = [
            frame
            for frame in self.scene.camera_keyframes
            if abs(float(frame["time"]) - time_seconds) > 1e-5
        ]
        self.scene.camera_keyframes.append(captured)
        self.scene.camera_keyframes.sort(key=lambda frame: float(frame["time"]))
        self._refresh_keyframes()

    def _animation_kind(self):
        return {
            "Камера": "camera",
            "Объект": "object",
            "Свет": "light",
            "Кость": "bone",
        }[self.animation_target_type.get()]

    def _interpolation_mode(self):
        return {
            "Линейная": "linear",
            "Плавная": "smooth",
            "Catmull–Rom": "catmull_rom",
        }[self.animation_interpolation.get()]

    def _set_animation_kind(self, kind):
        labels = {
            "camera": "Камера",
            "object": "Объект",
            "light": "Свет",
            "bone": "Кость",
        }
        self.animation_target_type.set(labels[kind])

    def _refresh_animation_targets(self):
        if not hasattr(self, "animation_target_picker"):
            return
        kind = self._animation_kind()
        if kind == "camera":
            values = ("Камера",)
        elif kind == "object":
            values = tuple(obj.name for obj in self.scene.objects)
        elif kind == "bone":
            values = tuple(
                f"{obj.name}::{bone['name']}"
                for obj in self.scene.objects
                if obj.rig
                for bone in obj.rig["bones"]
            )
        else:
            values = tuple(light.name for light in self.scene.lights)
        self.animation_target_picker["values"] = values
        current = self.animation_target_name.get()
        if current not in values:
            self.animation_target_name.set(values[0] if values else "")
        self._refresh_keyframes()
        self._refresh_bone_controls()

    def _animation_target_changed(self):
        self._refresh_keyframes()
        self._refresh_bone_controls()

    def _refresh_bone_controls(self):
        if not hasattr(self, "bone_pose_row"):
            return
        target = self._animation_target()
        if self._animation_kind() != "bone" or target is None:
            self.bone_pose_row.pack_forget()
            return
        self.bone_pose_row.pack(fill="x", pady=2)
        for axis, value in zip(("X", "Y", "Z"), target["rotation"]):
            self.bone_pose_controls[axis].set(f"{np.degrees(value):.2f}")

    def _apply_bone_pose(self):
        target = self._animation_target()
        if self._animation_kind() != "bone" or target is None:
            self.status.set("Выберите кость модели с авто-ригом.")
            return
        try:
            degrees = tuple(
                float(self.bone_pose_controls[axis].get()) for axis in ("X", "Y", "Z")
            )
            if not all(np.isfinite(value) for value in degrees):
                raise ValueError
        except ValueError:
            self.status.set("Введите конечные углы поворота кости.")
            return
        target["rotation"] = [float(np.radians(value)) for value in degrees]
        self._draw_animation_curve()
        self.status.set(
            "Поза кости применена. Запишите ключ, чтобы сохранить её в анимации."
        )

    def _animation_target(self):
        kind = self._animation_kind()
        name = self.animation_target_name.get()
        if kind == "camera":
            return self.camera
        if kind == "object":
            return next((obj for obj in self.scene.objects if obj.name == name), None)
        if kind == "bone":
            object_name, separator, bone_name = name.rpartition("::")
            obj = next(
                (item for item in self.scene.objects if item.name == object_name), None
            )
            if separator and obj is not None and obj.rig:
                return next(
                    (bone for bone in obj.rig["bones"] if bone["name"] == bone_name),
                    None,
                )
            return None
        if kind == "light":
            return next(
                (light for light in self.scene.lights if light.name == name), None
            )
        return None

    def _current_animation_track(self, create=False):
        kind = self._animation_kind()
        name = "Camera" if kind == "camera" else self.animation_target_name.get()
        track = next(
            (
                item
                for item in self.scene.animation_tracks
                if item.get("target_type") == kind and item.get("target") == name
            ),
            None,
        )
        if track is None and create and name:
            track = {
                "target_type": kind,
                "target": name,
                "interpolation": self._interpolation_mode(),
                "keys": [],
            }
            self.scene.animation_tracks.append(track)
        return track

    @_undoable
    def capture_selected_key(self):
        target = self._animation_target()
        if target is None:
            self.status.set("Выберите существующую цель анимации.")
            return
        track = self._current_animation_track(create=True)
        track["interpolation"] = self._interpolation_mode()
        captured = {
            "time": self.playhead,
            "values": capture_values(self._animation_kind(), target),
        }
        track["keys"] = [
            key
            for key in track["keys"]
            if abs(float(key["time"]) - self.playhead) > 1e-5
        ]
        track["keys"].append(captured)
        track["keys"].sort(key=lambda key: float(key["time"]))
        if self._animation_kind() == "camera":
            self._store_camera_key(self.playhead)
        self._refresh_keyframes()
        self.status.set(
            f"Ключ записан: {self.animation_target_type.get()} «{self.animation_target_name.get()}» "
            f"в {self.playhead:.2f} с"
        )

    def capture_selected_checkpoint(self):
        track = self._current_animation_track()
        last_time = (
            max(
                (float(key["time"]) for key in track.get("keys", [])),
                default=-1.0,
            )
            if track
            else -1.0
        )
        if self._animation_kind() == "camera":
            last_time = max(
                last_time,
                max(
                    (float(key["time"]) for key in self.scene.camera_keyframes),
                    default=-1.0,
                ),
            )
        self.playhead = max(self.playhead, last_time) + 1.0
        self.capture_selected_key()
        self.timeline_value.set(self.playhead)

    def _apply_timeline_pose(self):
        if not self._current_animation_track() and self._animation_kind() == "camera":
            apply_camera_keyframes(
                self.camera, self.scene.camera_keyframes, self.playhead
            )
        apply_animation_tracks(self.scene, self.camera, self.playhead)

    def _draw_animation_curve(self):
        if not hasattr(self, "curve_canvas") or not self.curve_canvas.winfo_exists():
            return
        canvas = self.curve_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 2)
        height = max(canvas.winfo_height(), 2)
        canvas.create_line(0, height - 20, width, height - 20, fill="#647080")
        canvas.create_line(28, 0, 28, height, fill="#647080")
        track = self._current_animation_track()
        if not track:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Добавьте минимум два ключа, чтобы увидеть кривую движения.",
                fill="#d5dde8",
            )
            return
        channel = self.curve_channel.get()
        channel_map = {
            "Позиция X": ("position", 0),
            "Позиция Y": ("position", 1),
            "Позиция Z": ("position", 2),
            "Поворот X": ("rotation", 0),
            "Поворот Y": ("rotation", 1),
            "Поворот Z": ("rotation", 2),
            "Размер X": ("scale_xyz", 0),
            "Размер Y": ("scale_xyz", 1),
            "Размер Z": ("scale_xyz", 2),
            "Масштаб": ("scale", 0),
            "Цвет R": ("color", 0),
            "Цвет G": ("color", 1),
            "Цвет B": ("color", 2),
            "Направление X": ("direction", 0),
            "Направление Y": ("direction", 1),
            "Направление Z": ("direction", 2),
            "Яркость": ("intensity", 0),
            "Камера: yaw": ("yaw", 0),
            "Камера: pitch": ("pitch", 0),
            "Камера: roll": ("roll", 0),
        }
        field, axis = channel_map[channel]
        samples = sample_track_curve(track, field, axis)
        if len(samples) < 2:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Добавьте ещё один ключ, чтобы построить кривую.",
                fill="#d5dde8",
            )
            return
        times = [sample[0] for sample in samples]
        values = [sample[1] for sample in samples]
        min_time, max_time = min(times), max(times)
        min_value, max_value = min(values), max(values)
        value_span = max(max_value - min_value, 1e-6)
        points = []
        for time_value, value in samples:
            x = 30 + (time_value - min_time) / max(max_time - min_time, 1e-6) * (
                width - 40
            )
            y = height - 22 - (value - min_value) / value_span * (height - 32)
            points.extend((x, y))
        canvas.create_line(*points, fill="#59c3c3", width=2, smooth=False)
        for key in track.get("keys", []):
            value = key["values"].get(field, [0.0])
            if isinstance(value, list):
                value = value[axis]
            x = 30 + (float(key["time"]) - min_time) / max(
                max_time - min_time, 1e-6
            ) * (width - 40)
            y = height - 22 - (value - min_value) / value_span * (height - 32)
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ffd166", outline="")
        canvas.create_text(
            34, 8, text=f"{channel} во времени", fill="#d5dde8", anchor="nw"
        )

    def _keyframe_list_click(self, event):
        row = self.keyframe_list.nearest(event.y)
        bounds = self.keyframe_list.bbox(row) if self.keyframe_list.size() else None
        if bounds is None or not bounds[1] <= event.y < bounds[1] + bounds[3]:
            self.keyframe_list.selection_clear(0, tk.END)
            return "break"

    def _refresh_keyframes(self):
        if not hasattr(self, "keyframe_list"):
            return
        self.keyframe_list.delete(0, tk.END)
        track = self._current_animation_track()
        if track:
            self.animation_interpolation.set(
                {
                    "linear": "Линейная",
                    "smooth": "Плавная",
                    "catmull_rom": "Catmull–Rom",
                }.get(track.get("interpolation", "linear"), "Линейная")
            )
        if track:
            frames = sorted(track.get("keys", []), key=lambda item: float(item["time"]))
        elif self._animation_kind() == "camera":
            frames = [
                {"time": frame["time"], "values": {"position": frame["position"]}}
                for frame in self.scene.camera_keyframes
            ]
        else:
            frames = []
        max_time = max(
            (
                float(key["time"])
                for item in self.scene.animation_tracks
                for key in item.get("keys", [])
            ),
            default=0.0,
        )
        max_time = max(
            max_time,
            max(
                (float(frame["time"]) for frame in self.scene.camera_keyframes),
                default=0.0,
            ),
        )
        self.timeline_scale.configure(to=max(30.0, max_time + 5.0))
        for frame in frames:
            values = frame.get("values", {})
            target_label = self.animation_target_name.get()
            track_type = self.animation_target_type.get()
            position = values.get("position", [])
            self.keyframe_list.insert(
                tk.END,
                f"{float(frame['time']):.2f} с  {track_type} {target_label}  {position}",
            )
        self._draw_animation_curve()

    def _select_keyframe(self, _event=None):
        selection = self.keyframe_list.curselection()
        if not selection:
            return
        track = self._current_animation_track()
        frames = (
            sorted(track.get("keys", []), key=lambda item: float(item["time"]))
            if track
            else []
        )
        if not frames and self._animation_kind() == "camera":
            frames = sorted(
                self.scene.camera_keyframes, key=lambda item: float(item["time"])
            )
        if selection[0] >= len(frames):
            return
        self.timeline_value.set(float(frames[selection[0]]["time"]))
        self._set_playhead(self.timeline_value.get())

    @_undoable
    def delete_camera_key(self):
        selection = self.keyframe_list.curselection()
        if not selection:
            return
        track = self._current_animation_track()
        if track:
            frames = sorted(track.get("keys", []), key=lambda item: float(item["time"]))
            if selection[0] < len(frames):
                deleted_time = float(frames[selection[0]]["time"])
                del frames[selection[0]]
                track["keys"] = frames
                if track.get("target_type") == "camera":
                    self.scene.camera_keyframes = [
                        key
                        for key in self.scene.camera_keyframes
                        if abs(float(key["time"]) - deleted_time) > 1e-5
                    ]
                if not frames:
                    self.scene.animation_tracks.remove(track)
        elif self._animation_kind() == "camera":
            frames = sorted(
                self.scene.camera_keyframes, key=lambda item: float(item["time"])
            )
            if selection[0] < len(frames):
                del frames[selection[0]]
                self.scene.camera_keyframes = frames
        self._refresh_keyframes()

    def toggle_playback(self):
        self.timeline_playing = not self.timeline_playing
        self.play_button.configure(
            text="Пауза" if self.timeline_playing else "Воспроизвести"
        )

    def update(self, delta_time, window):
        self.root.update_idletasks()
        self.root.update()
        try:
            self.audio_player.update()
        except RuntimeError as error:
            self.audio_preview = None
            self.status.set(f"Ошибка аудиопредпросмотра: {error}")
        self.logic.update(
            self.scene,
            delta_time,
            window,
            self.camera.position if self.playtest_active else None,
        )
        animated_times = [
            float(key["time"])
            for track in self.scene.animation_tracks
            for key in track.get("keys", [])
        ] + [float(frame["time"]) for frame in self.scene.camera_keyframes]
        if self.timeline_playing and animated_times:
            last_time = max(animated_times)
            if self.playhead >= last_time:
                self.playhead = 0.0
            else:
                self.playhead = min(last_time, self.playhead + delta_time)
            self.timeline_value.set(self.playhead)
            self._apply_timeline_pose()
        elif self.timeline_playing:
            self.timeline_playing = False
            self.play_button.configure(text="Воспроизвести")

    def request_close(self):
        self.closing = True

    def close(self):
        if self._viewport_resize_after_id is not None:
            self.root.after_cancel(self._viewport_resize_after_id)
            self._viewport_resize_after_id = None
        if (
            self._dirty
            and self._autosaved_revision != self._scene_revision
            and not self.playtest_active
        ):
            try:
                self._write_recovery_copy()
            except (OSError, ValueError) as error:
                self.status.set(f"Не удалось сохранить автокопию: {error}")
        self._scene_event_consumers.close()
        self.clear_asset_preview()
        self.audio_player.stop()
        self.asset_cache.cleanup()
        self.root.destroy()
