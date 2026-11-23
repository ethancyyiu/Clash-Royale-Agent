import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

DEFAULT_CLASSES = [
    "Friendly troop", "Enemy troop", "Friendly tower",
    "Enemy tower", "Card", "Other / unknown",
]


@dataclass(frozen=True)
class Box:
    class_id: int
    left: int
    top: int
    right: int
    bottom: int

    def yolo_row(self, width: int, height: int) -> str:
        values = (
            (self.left + self.right) / (2 * width),
            (self.top + self.bottom) / (2 * height),
            (self.right - self.left) / width,
            (self.bottom - self.top) / height,
        )
        return f"{self.class_id} " + " ".join(f"{value:.10f}" for value in values)


def read_capture_timestamps(video: Path) -> dict[int, int]:
    path = video.with_suffix(".timestamps.csv")
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        if rows.fieldnames != ["frame_index", "capture_monotonic_ns"]:
            raise ValueError(f"Invalid timestamp CSV header in {path}")
        timestamps = {}
        for row in rows:
            try:
                index = int(row["frame_index"])
                stamp = int(row["capture_monotonic_ns"])
                if index < 0 or stamp < 0 or index in timestamps:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid timestamp in {path} at line {rows.line_num}") from error
            timestamps[index] = stamp
        return timestamps


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=path.parent, delete=False)
    temp = Path(file.name)
    try:
        with file:
            file.write(data)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_classes(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if not names or not all(names) or len(set(names)) != len(names):
        raise ValueError(f"Use one unique, nonempty class name per line in {path}")
    return names


class Dataset:
    def __init__(
        self, root: Path, video: Path, classes_file: Path | None, match: str | None
    ) -> None:
        self.root = root
        classes_path = root / "classes.txt"
        requested = read_classes(classes_file) if classes_file else None
        self.classes = read_classes(classes_path) if classes_path.exists() else requested or DEFAULT_CLASSES
        if requested is not None and requested != self.classes:
            raise ValueError("The class list differs from this dataset's classes.txt. Use a new dataset folder.")
        if match is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", match):
            raise ValueError("--match must start with a letter or digit and use letters, digits, _ or -")

        self.capture_timestamps = read_capture_timestamps(video)
        digest = hashlib.sha256()
        with video.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        self.source_id = digest.hexdigest()
        self.prefix = self.source_id[:16] + "_"
        source_path = root / "sources" / f"{self.source_id}.json"
        if source_path.exists():
            source = json.loads(source_path.read_text(encoding="utf-8"))
            self.match = source["match_id"]
            if match is not None and match != self.match:
                raise ValueError(f"This video is already assigned to match {self.match!r} in this dataset.")
            self.filename = source["video_filename"]
        else:
            self.match = match or f"match-{self.source_id[:16]}"
            self.filename = video.name
            source = {"match_id": self.match, "video_filename": self.filename}
            write_file(source_path, (json.dumps(source, indent=2) + "\n").encode("utf-8"))
        if not classes_path.exists():
            write_file(classes_path, ("\n".join(self.classes) + "\n").encode("utf-8"))

        self.saved_frames = {
            int(path.stem[len(self.prefix):])
            for path in (root / "images" / self.match).glob(f"{self.prefix}*.png")
            if path.stem[len(self.prefix):].isdigit()
        }

    def frame_path(self, index: int, folder: str, suffix: str) -> Path:
        return self.root / folder / self.match / f"{self.prefix}{index:08d}{suffix}"

    def load_boxes(self, index: int, width: int, height: int) -> list[Box]:
        path = self.frame_path(index, "labels", ".txt")
        if index not in self.saved_frames or not path.exists():
            return []
        boxes = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                label, *values = line.split()
                class_id = int(label)
                center_x, center_y, box_width, box_height = map(float, values)
                if not 0 <= class_id < len(self.classes) or not all(
                    math.isfinite(value) and 0 <= value <= 1
                    for value in (center_x, center_y, box_width, box_height)
                ):
                    raise ValueError
                left = round((center_x - box_width / 2) * width)
                top = round((center_y - box_height / 2) * height)
                right = round((center_x + box_width / 2) * width)
                bottom = round((center_y + box_height / 2) * height)
                if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                    raise ValueError
            except ValueError as error:
                raise ValueError(f"Invalid YOLO box in {path} at line {line_number}") from error
            boxes.append(Box(class_id, left, top, right, bottom))
        return boxes

    def save(self, index: int, image: np.ndarray, timestamp_ns: int, boxes: list[Box]) -> None:
        height, width = image.shape[:2]
        image_path = self.frame_path(index, "images", ".png")
        labels = "".join(box.yolo_row(width, height) + "\n" for box in boxes)
        png = None
        if not image_path.exists():
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise OSError(f"Could not encode {image_path}")
            png = encoded.tobytes()
            metadata = {
                "match_id": self.match,
                "video_filename": self.filename,
                "video_sha256": self.source_id,
                "frame_index": index,
                "video_timestamp_ns": timestamp_ns,
                "capture_monotonic_ns": self.capture_timestamps.get(index),
                "width": width,
                "height": height,
            }
            path = self.frame_path(index, "metadata", ".json")
            write_file(path, (json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
        write_file(self.frame_path(index, "labels", ".txt"), labels.encode("utf-8"))
        if png is not None:
            write_file(image_path, png)
        self.saved_frames.add(index)
