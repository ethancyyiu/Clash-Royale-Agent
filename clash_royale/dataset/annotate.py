import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from .replay_dataset import Box, Dataset

WINDOW = "Gameplay Replay Annotator"
TRACKBAR = "Frame"


def load_review_flags(path: Path, source_id: str, count: int) -> dict[int, list[str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    entries = report.get("flagged_frames") if isinstance(report, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Expected a review report from check-dataset or evaluate --holdout.")
    flags = {}
    for frame in entries:
        if frame.get("video_sha256") == source_id and 0 <= frame["frame_index"] < count:
            messages = flags.setdefault(frame["frame_index"], [])
            messages.extend(issue["message"] for issue in frame["issues"])
    if not flags:
        raise ValueError("No flagged frames for this video. Open a source MP4 listed in the report.")
    return flags


class Annotator:
    def __init__(
        self, video: cv2.VideoCapture, dataset: Dataset, count: int, fps: float,
        review_flags: dict[int, list[str]] | None = None, *, queue: list[int] | None = None,
    ):
        self.video = video
        self.dataset = dataset
        self.count = count
        self.fps = fps
        self.review_flags = review_flags
        self.queue = queue or []
        self.index = -1
        self.boxes: list[Box] = []
        self.active_class = 0
        self.selected: int | None = None
        self.redrawing = False
        self.drag: tuple[int, int] | None = None
        self.pending_box: Box | None = None
        self.dirty = False
        self.redraw = True
        self.next_frame_at: float | None = None
        self.message = ""
        self.label_error_count = 0

    def show_frame(self, index: int) -> None:
        index = max(0, min(self.count - 1, index))
        if self.dirty or self.drag is not None:
            self.message = "Finish drawing, then save (S) or revert (R) before navigating."
        elif self.index != index:
            if self.video.get(cv2.CAP_PROP_POS_FRAMES) != index:
                if not self.video.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise OSError(f"Could not seek to frame {index}")
            ok, image = self.video.read()
            if not ok or self.video.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
                raise OSError(f"Could not decode frame {index}")
            milliseconds = self.video.get(cv2.CAP_PROP_POS_MSEC)
            self.timestamp_ns = (
                round(milliseconds * 1_000_000)
                if math.isfinite(milliseconds) and milliseconds > 0
                else round(index * 1_000_000_000 / self.fps)
            )
            self.image = image
            self.height, self.width = image.shape[:2]
            scale = min(1, 960 / self.width, 640 / self.height)
            self.view_width = max(1, round(self.width * scale))
            self.view_height = max(1, round(self.height * scale))
            self.preview = cv2.resize(
                image, (self.view_width, self.view_height), interpolation=cv2.INTER_AREA
            ) if scale < 1 else image
            self.canvas_width = max(780, self.view_width)
            self.view_left = (self.canvas_width - self.view_width) // 2
            self.index = index
            self.reload_boxes()
            if self.review_flags and index in self.review_flags:
                print(f"Frame {index} flagged for review:", flush=True)
                for reason in self.review_flags[index]:
                    print(f"  {reason}", flush=True)
        cv2.setTrackbarPos(TRACKBAR, WINDOW, self.index)
        self.redraw = True

    def reload_boxes(self) -> None:
        self.boxes, errors = self.dataset.load_boxes(self.index, self.width, self.height)
        self.label_error_count = len(errors)
        for error in errors:
            print(f"Frame {self.index}: {error}", file=sys.stderr, flush=True)
        self.selected = None
        self.redrawing = False
        self.dirty = False
        self.message = ""

    def mouse(self, event: int, x: int, y: int, _flags: int, _data=None) -> None:
        x -= self.view_left
        inside = 0 <= x < self.view_width and 0 <= y < self.view_height
        point = (
            round(max(0, min(x, self.view_width)) * self.width / self.view_width),
            round(max(0, min(y, self.view_height)) * self.height / self.view_height),
        )
        if event == cv2.EVENT_RBUTTONDOWN and inside and self.drag is None:
            self.next_frame_at = None
            self.redrawing = False
            self.selected = next(
                (i for i in reversed(range(len(self.boxes)))
                 if self.boxes[i].left <= point[0] < self.boxes[i].right
                 and self.boxes[i].top <= point[1] < self.boxes[i].bottom),
                None,
            )
        elif event == cv2.EVENT_LBUTTONDOWN and inside:
            self.next_frame_at = None
            self.drag = point
        elif event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP) and self.drag is not None:
            left, right = sorted((self.drag[0], point[0]))
            top, bottom = sorted((self.drag[1], point[1]))
            class_id = self.boxes[self.selected].class_id if self.redrawing else self.active_class
            self.pending_box = Box(class_id, left, top, right, bottom)
            if event == cv2.EVENT_LBUTTONUP:
                if left < right and top < bottom:
                    if self.redrawing:
                        self.boxes[self.selected] = self.pending_box
                    else:
                        self.boxes.append(self.pending_box)
                        self.selected = len(self.boxes) - 1
                    self.dirty = True
                    self.message = ""
                    self.redrawing = False
                self.drag = None
                self.pending_box = None
        else:
            return
        self.redraw = True

    def handle_key(self, key: int) -> bool:
        if self.drag is not None:
            return True
        if key in (ord("q"), 27):
            if not self.dirty:
                return False
            self.message = "Save (S) or revert (R) before quitting."
        elif key == ord(" "):
            if self.next_frame_at is not None:
                self.next_frame_at = None
            elif not self.dirty and self.index < self.count - 1:
                self.next_frame_at = monotonic() + 1 / self.fps
            elif self.dirty:
                self.message = "Save (S) or revert (R) before playing."
        elif key in b"adjlpn,.":
            self.next_frame_at = None
            jump = max(1, round(self.fps))
            offsets = {ord("a"): -1, ord("d"): 1, ord("j"): -jump, ord("l"): jump}
            if key in offsets:
                self.show_frame(self.index + offsets[key])
            else:
                if key in b",.":
                    frames = (i for i in self.queue if i not in self.dataset.saved_frames)
                    kind = "unlabelled queue"
                elif self.review_flags is not None:
                    frames = self.review_flags
                    kind = "flagged"
                else:
                    frames = self.dataset.saved_frames
                    kind = "saved"
                if key in b"p,":
                    target = max((i for i in frames if i < self.index), default=None)
                else:
                    target = min((i for i in frames if i > self.index), default=None)
                if target is None:
                    self.message = f"No {kind} frame in that direction."
                else:
                    self.show_frame(target)
        elif key in b"[]":
            delta = 1 if key == ord("]") else -1
            self.active_class = (self.active_class + delta) % len(self.dataset.classes)
        elif key == ord("s"):
            self.next_frame_at = None
            self.dataset.save(self.index, self.image, self.timestamp_ns, self.boxes)
            self.dirty = False
            self.label_error_count = 0
            self.message = "Frame and labels saved."
        elif key == ord("r"):
            self.reload_boxes()
        elif key in b"ecx":
            self.next_frame_at = None
            if self.selected is None:
                self.message = "Right-click a box to select it."
            elif key == ord("e"):
                self.redrawing = True
                self.message = "Drag a replacement for the selected box."
            elif key == ord("c"):
                self.boxes[self.selected] = replace(self.boxes[self.selected], class_id=self.active_class)
                self.dirty = True
                self.message = ""
            else:
                self.boxes.pop(self.selected)
                self.selected = None
                self.redrawing = False
                self.dirty = True
                self.message = ""
        self.redraw = True
        return True

    def draw(self) -> None:
        canvas = np.full((self.view_height + 72, self.canvas_width, 3), 28, dtype=np.uint8)
        canvas[:self.view_height, self.view_left:self.view_left + self.view_width] = self.preview
        for i, box in enumerate(self.boxes):
            self.draw_box(canvas, box, (0, 220, 255) if i == self.selected else (80, 220, 80))
        if self.pending_box is not None:
            self.draw_box(canvas, self.pending_box, (255, 180, 75))

        state = "Playing" if self.next_frame_at is not None else "Paused"
        saved = "Saved" if self.index in self.dataset.saved_frames else "Unsaved frame"
        if self.dirty:
            saved = "Unsaved edits"
        active = self.dataset.classes[self.active_class]
        navigation = "P/N saved frames"
        if self.review_flags is not None:
            navigation = "P/N flagged frames"
            if self.index in self.review_flags:
                saved += " | Flagged (see terminal)"
        if self.queue:
            navigation += " | ,/. queue"
        help_text = f"Space play/pause | A/D step | {navigation} | S save | Q quit"
        if self.label_error_count:
            help_text = f"{self.label_error_count} invalid label rows skipped. Redraw as needed. S replaces labels."
        lines = [
            f"{state} | Frame {self.index}/{self.count - 1} | {self.timestamp_ns / 1e9:.3f} s | {saved}",
            f"Class {self.active_class}: {active}  ([ / ]) | {len(self.boxes)} boxes",
            self.message or help_text,
        ]
        for row, line in enumerate(lines):
            cv2.putText(canvas, line, (8, self.view_height + 19 + row * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)
        self.redraw = False

    def draw_box(self, canvas: np.ndarray, box: Box, color: tuple[int, int, int]) -> None:
        left = self.view_left + min(self.view_width - 1, round(box.left * self.view_width / self.width))
        right = self.view_left + min(self.view_width - 1, round(box.right * self.view_width / self.width))
        top = min(self.view_height - 1, round(box.top * self.view_height / self.height))
        bottom = min(self.view_height - 1, round(box.bottom * self.view_height / self.height))
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 1)
        cv2.putText(canvas, self.dataset.classes[box.class_id], (left, max(11, top - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def run(self, review: bool) -> None:
        cv2.createTrackbar(TRACKBAR, WINDOW, 0, max(1, self.count - 1), lambda _position: None)
        start = min(self.dataset.saved_frames, default=0) if review else 0
        if self.review_flags is not None:
            start = min(self.review_flags, default=start)
        elif self.queue:
            start = min((i for i in self.queue if i not in self.dataset.saved_frames), default=start)
        self.show_frame(start)
        cv2.setMouseCallback(WINDOW, self.mouse)
        while True:
            if self.redraw:
                self.draw()
            key = cv2.waitKey(10)
            try:
                visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                visible = False
            if not visible or (key >= 0 and not self.handle_key(key & 0xFF)):
                return
            target = cv2.getTrackbarPos(TRACKBAR, WINDOW)
            if target != self.index:
                self.next_frame_at = None
                self.show_frame(target)
            if self.next_frame_at is not None and monotonic() >= self.next_frame_at:
                self.show_frame(self.index + 1)
                self.next_frame_at += 1 / self.fps
                if self.index == self.count - 1:
                    self.next_frame_at = None


def load_queue(path: Path, source_id: str, count: int) -> list[int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read review queue {path}: {error}") from error
    frames = data.get("frames") if isinstance(data, dict) else None
    if (not isinstance(data, dict) or data.get("version") != 1 or
            data.get("video_sha256") != source_id or
            not isinstance(frames, list) or not frames or
            any(type(frame) is not int or not 0 <= frame < count for frame in frames) or
            len(set(frames)) != len(frames)):
        raise ValueError(f"Review queue {path} is invalid or belongs to another video.")
    return sorted(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Review an MP4 and export YOLO bounding boxes.")
    parser.add_argument("video", type=Path, help="source MP4")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="output folder (default: dataset)")
    parser.add_argument("--classes", type=Path, help="text file with one class name per line")
    parser.add_argument("--match", help="group multiple clips from the same match under this ID")
    review = parser.add_mutually_exclusive_group()
    review.add_argument("--review", action="store_true", help="start at the first saved frame")
    review.add_argument("--review-report", type=Path, help="start at the first flagged frame and navigate it with P/N")
    parser.add_argument("--queue", type=Path, help="start at a suggested frame and navigate with , and .")
    args = parser.parse_args()

    video = cv2.VideoCapture()
    window_created = False
    try:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise OSError("The annotator needs a desktop display on Linux. Run with DISPLAY set.")
        if args.video.suffix.lower() != ".mp4" or not video.open(str(args.video)):
            raise OSError(f"Could not open MP4 video {args.video}")
        fps, count = video.get(cv2.CAP_PROP_FPS), video.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError("The MP4 did not report a valid frame rate and frame count.")

        dataset = Dataset(args.dataset, args.video, args.classes, args.match)
        flags = load_review_flags(args.review_report, dataset.source_id, int(count)) if args.review_report else None
        queue = load_queue(args.queue, dataset.source_id, int(count)) if args.queue else None
        print(f"Match {dataset.match} | dataset {dataset.root}", flush=True)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        window_created = True
        Annotator(video, dataset, int(count), fps, review_flags=flags, queue=queue).run(args.review)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        video.release()
        if window_created:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
