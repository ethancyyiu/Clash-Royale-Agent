import math
from contextlib import contextmanager
from pathlib import Path

import cv2


@contextmanager
def open_video(path: Path):
    video = cv2.VideoCapture()
    try:
        if path.suffix.lower() != ".mp4" or not video.open(str(path)):
            raise OSError(f"Could not open MP4 {path}")
        fps = video.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"MP4 did not report a valid frame rate: {path}")
        yield video, fps
    finally:
        video.release()


def read_frame(path: Path, index: int):
    with open_video(path) as (video, _):
        if index < 0 or (index and not video.set(cv2.CAP_PROP_POS_FRAMES, index)):
            raise ValueError(f"Cannot seek to frame {index}")
        ok, image = video.read()
        if not ok or video.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
            raise ValueError(f"Cannot decode frame {index} from {path}")
        return image
