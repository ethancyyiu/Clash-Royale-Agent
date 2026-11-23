import argparse
import math
import os
import sys
from pathlib import Path
from time import monotonic

import cv2

from detector import Detector, draw_detections

WINDOW = "Gameplay Detections"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview YOLO detections on an offline MP4.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, required=True, help="trained .pt checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or GPU index such as 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    args = parser.parse_args()
    if args.imgsz <= 0 or not 0 <= args.conf <= 1:
        parser.error("--imgsz must be positive and --conf must be in [0, 1]")

    video = cv2.VideoCapture()
    window_created = False
    try:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise OSError("Video preview needs a desktop display on Linux. Run with DISPLAY set.")
        if args.video.suffix.lower() != ".mp4" or not video.open(str(args.video)):
            raise OSError(f"Could not open MP4 video {args.video}")
        ok, frame = video.read()
        if not ok:
            raise OSError(f"Could not decode the first frame of {args.video}")
        detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf)
        fps = video.get(cv2.CAP_PROP_FPS)
        period = 1 / fps if math.isfinite(fps) and fps >= 1 else 0
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        window_created = True
        while ok:
            started_at = monotonic()
            preview = draw_detections(frame, detector.detect(frame))
            cv2.imshow(WINDOW, preview)
            delay = max(1, round((period - (monotonic() - started_at)) * 1000))
            if (cv2.waitKey(delay) & 0xFF) in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            ok, frame = video.read()
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        video.release()
        if window_created:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
