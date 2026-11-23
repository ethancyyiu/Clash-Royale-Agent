import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..dataset.replay_dataset import write_file

DEFAULT_CALIBRATION = Path("hud_data/calibration.json")
DEFAULT_REFERENCES = Path("hud_data/references")


@dataclass(frozen=True)
class Region:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        if not (0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1):
            raise ValueError("HUD regions must be nonempty and inside the normalized frame.")

    def crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1 = round(self.left * width), round(self.top * height)
        x2, y2 = round(self.right * width), round(self.bottom * height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("A calibrated HUD region is smaller than one pixel in this frame.")
        return frame[y1:y2, x1:x2]


@dataclass(frozen=True)
class Calibration:
    frame_width: int
    frame_height: int
    hand: tuple[Region, ...]
    elixir: Region
    next_card: Region | None = None

    def __post_init__(self):
        if (
            type(self.frame_width) is not int or type(self.frame_height) is not int
            or min(self.frame_width, self.frame_height) <= 0 or len(self.hand) != 4
        ):
            raise ValueError("Calibration needs valid frame dimensions and exactly four card slots.")

    def check_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        ratio = (width * self.frame_height) / (height * self.frame_width)
        if abs(ratio - 1) > 0.03:
            raise ValueError("Frame aspect ratio differs from HUD calibration. Use the original game crop or recalibrate.")

    def save(self, path: Path) -> None:
        write_file(path, (json.dumps(asdict(self), indent=2) + "\n").encode("utf-8"))

    @classmethod
    def load(cls, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                data["frame_width"], data["frame_height"], tuple(Region(**box) for box in data["hand"]),
                Region(**data["elixir"]), Region(**data["next_card"]) if data.get("next_card") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid calibration file {path}: {error}") from error


def select_calibration(frame: np.ndarray, include_next: bool) -> Calibration:
    height, width = frame.shape[:2]
    scale = min(1.0, 1100 / width, 760 / height)
    preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    view_height, view_width = preview.shape[:2]
    window = "Calibrate HUD"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def select(name: str, optional: bool = False) -> Region | None:
        print(f"Select {name}, then press Enter. Press C to cancel.", flush=True)
        x, y, w, h = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
        if w == 0 or h == 0:
            if optional:
                return None
            raise ValueError(f"Selection cancelled for {name}.")
        return Region(x / view_width, y / view_height, (x + w) / view_width, (y + h) / view_height)

    try:
        hand = tuple(select(f"card slot {i} (left to right)") for i in range(1, 5))
        elixir = select("the elixir bar and its visible digit")
        next_card = select("the next-card preview", optional=True) if include_next else None
        return Calibration(width, height, hand, elixir, next_card)
    finally:
        cv2.destroyWindow(window)
