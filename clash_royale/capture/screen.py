from dataclasses import dataclass
from time import monotonic_ns

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class Frame:
    bgr: np.ndarray
    captured_ns: int
    grab_ns: int


def select_region(
    screen: mss.MSS, monitor: int, region: list[int] | None
) -> dict[str, int]:
    monitors = screen.monitors
    if not 1 <= monitor < len(monitors):
        raise ValueError(f"Monitor {monitor} is unavailable. Use --list-monitors.")

    bounds = monitors[monitor]
    if region is None:
        return {key: bounds[key] for key in ("left", "top", "width", "height")}

    x, y, width, height = region
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > bounds["width"]
        or y + height > bounds["height"]
    ):
        raise ValueError("The region must have a positive size and fit inside the selected monitor.")

    return {
        "left": bounds["left"] + x,
        "top": bounds["top"] + y,
        "width": width,
        "height": height,
    }


def grab_frame(screen: mss.MSS, region: dict[str, int]) -> Frame:
    started_ns = monotonic_ns()
    screenshot = screen.grab(region)
    captured_ns = monotonic_ns()
    bgr = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_BGRA2BGR)
    return Frame(bgr, captured_ns, captured_ns - started_ns)
