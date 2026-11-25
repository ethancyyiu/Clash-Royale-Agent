"""Windows desktop input, with a polling recorder that needs no extra package."""
from __future__ import annotations

import ctypes
import json
import sys
from ctypes import wintypes
from threading import Event, Thread
from time import monotonic_ns, sleep


class Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def require_windows():
    if sys.platform != "win32":
        raise OSError("Emulator mouse control currently requires Windows.")
    return ctypes.windll.user32


def cursor_position() -> tuple[int, int]:
    point = Point()
    if not require_windows().GetCursorPos(ctypes.byref(point)):
        raise OSError("GetCursorPos failed")
    return point.x, point.y


class MouseRecorder:
    def __init__(self, stream, region: dict[str, int]):
        self.stream = stream
        self.region = region
        self.stop = Event()
        self.thread = Thread(target=self._run, daemon=True)

    def __enter__(self):
        require_windows()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def _run(self):
        user32 = require_windows()
        previous = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
        while not self.stop.is_set():
            pressed = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
            if pressed != previous:
                x, y = cursor_position()
                row = {
                    "timestamp_ns": monotonic_ns(),
                    "event": "down" if pressed else "up",
                    "x": (x - self.region["left"]) / self.region["width"],
                    "y": (y - self.region["top"]) / self.region["height"],
                }
                self.stream.write(json.dumps(row) + "\n")
                previous = pressed
            sleep(0.005)


def click(x: int, y: int) -> None:
    user32 = require_windows()
    if not user32.SetCursorPos(x, y):
        raise OSError("SetCursorPos failed")
    # mouse_event coordinates come from the current cursor; avoids display scaling assumptions.
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    sleep(0.04)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
