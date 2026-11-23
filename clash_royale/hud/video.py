import math

import cv2

from ..capture.video import open_video, read_frame


def frame_timestamp(video, fps: float, index: int, capture_timestamps: dict[int, int]) -> tuple[int, str]:
    if index in capture_timestamps:
        return capture_timestamps[index], "capture_monotonic"
    milliseconds = video.get(cv2.CAP_PROP_POS_MSEC)
    ns = round(milliseconds * 1e6) if math.isfinite(milliseconds) and milliseconds > 0 else round(index * 1e9 / fps)
    return ns, "video_playback"
