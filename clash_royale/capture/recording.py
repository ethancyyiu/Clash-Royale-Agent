import csv
from contextlib import contextmanager
from pathlib import Path

import cv2

from .screen import Frame


@contextmanager
def video_writer(path: Path, fps: float, width: int, height: int):
    # Some encoders truncate odd dimensions. Pad instead to preserve every pixel.
    right, bottom = width % 2, height % 2
    writer = cv2.VideoWriter()
    try:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        if not writer.open(str(path), codec, fps, (width + right, height + bottom)):
            raise OSError(f"Could not open the MP4 video writer for {path}")

        def write_image(image) -> None:
            if right or bottom:
                image = cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT)
            writer.write(image)

        yield write_image
    finally:
        writer.release()


@contextmanager
def record_video(path: Path, fps: float, width: int, height: int):
    timestamps = path.with_suffix(".timestamps.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Recording already exists: {path}")
    with timestamps.open("x", newline="", encoding="utf-8") as file:
        with video_writer(path, fps, width, height) as write_image:
            rows = csv.writer(file)
            rows.writerow(("frame_index", "capture_monotonic_ns"))
            frame_index = 0

            def write_frame(frame: Frame) -> None:
                nonlocal frame_index
                write_image(frame.bgr)
                rows.writerow((frame_index, frame.captured_ns))
                frame_index += 1

            print(f"Recording to {path} with timestamps in {timestamps}", flush=True)
            yield write_frame
