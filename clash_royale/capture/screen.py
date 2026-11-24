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


def draw_region(screen: mss.MSS, monitor: int) -> list[int]:
    """Let the user select a monitor-relative capture region on a still screenshot."""
    bounds = select_region(screen, monitor, None)
    screenshot = screen.grab(bounds)
    image = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_BGRA2BGR)
    height, width = image.shape[:2]
    scale = min(1.0, 1100 / width, 760 / height)
    view_width = max(1, round(width * scale))
    view_height = max(1, round(height * scale))
    preview = cv2.resize(image, (view_width, view_height), interpolation=cv2.INTER_AREA)
    window = "Select capture region"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        x, y, w, h = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(window)
    if w == 0 or h == 0:
        raise ValueError("Capture region selection cancelled; the saved region was not changed.")
    left = round(x * width / view_width)
    top = round(y * height / view_height)
    right = round((x + w) * width / view_width)
    bottom = round((y + h) * height / view_height)
    return [left, top, right - left, bottom - top]


def grab_frame(screen: mss.MSS, region: dict[str, int]) -> Frame:
    started_ns = monotonic_ns()
    screenshot = screen.grab(region)
    captured_ns = monotonic_ns()
    bgr = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_BGRA2BGR)
    return Frame(bgr, captured_ns, captured_ns - started_ns)
