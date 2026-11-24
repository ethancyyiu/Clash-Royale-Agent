"""Pick unlabelled, visually varied frames for the existing box annotator."""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .replay_dataset import write_file


@dataclass(frozen=True)
class Candidate:
    index: int
    pixels: np.ndarray


def video_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saved_indices(dataset: Path, source_id: str) -> set[int]:
    saved = set()
    for path in (dataset / "metadata").glob("*/*.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(metadata, dict) and metadata.get("video_sha256") == source_id and
                type(metadata.get("frame_index")) is int):
            saved.add(metadata["frame_index"])
    return saved


def sample_frames(video: cv2.VideoCapture, indices: range) -> list[Candidate]:
    candidates = []
    for index in indices:
        if not video.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise OSError(f"Could not seek to frame {index}")
        ok, frame = video.read()
        if not ok:
            raise OSError(f"Could not decode frame {index}")
        thumb = cv2.resize(frame, (24, 24), interpolation=cv2.INTER_AREA)
        candidates.append(Candidate(index, thumb.astype(np.float32).ravel() / 255))
    return candidates


def select_frames(candidates: list[Candidate], saved: set[int], limit: int,
                  min_gap_frames: int, span_frames: int, stride_frames: int) -> list[int]:
    available = [candidate for candidate in candidates
                 if all(abs(candidate.index - index) >= min_gap_frames for index in saved)]
    if not available:
        return []
    chosen: list[Candidate] = []
    anchors = set(saved)
    target = min(limit, len(available))
    while len(chosen) < target:
        def score(candidate: Candidate) -> tuple[float, int]:
            if anchors:
                gap = min(abs(candidate.index - index) for index in anchors)
                coverage = min(1.0, gap / max(stride_frames * 3, span_frames / max(limit, 1)))
            else:
                midpoint = (available[0].index + available[-1].index) / 2
                coverage = 1 - abs(candidate.index - midpoint) / max(
                    (available[-1].index - available[0].index) / 2, 1)
            novelty = (min(float(np.abs(candidate.pixels - other.pixels).mean()) for other in chosen)
                       if chosen else 0)
            return (0.65 * coverage + 0.35 * min(1.0, novelty / 0.15), -candidate.index)

        winner = max(available, key=score)
        chosen.append(winner)
        anchors.add(winner.index)
        available.remove(winner)
    return sorted(candidate.index for candidate in chosen)


def build_queue(video_path: Path, dataset: Path, output: Path, limit: int,
                stride_seconds: float, min_gap_seconds: float,
                start_seconds: float, end_seconds: float | None) -> dict:
    source_id = video_digest(video_path)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open MP4 {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frame_count) or frame_count < 1:
            raise ValueError("The MP4 did not report a valid frame rate and frame count.")
        count = int(frame_count)
        start = min(count - 1, round(start_seconds * fps))
        end = min(count, round(end_seconds * fps)) if end_seconds is not None else count
        if end <= start:
            raise ValueError("The selected time window contains no frames.")
        stride = max(1, round(stride_seconds * fps))
        candidates = sample_frames(capture, range(start, end, stride))
        saved = {index for index in saved_indices(dataset, source_id) if start <= index < end}
        frames = select_frames(candidates, saved, limit, max(1, round(min_gap_seconds * fps)),
                               end - start, stride)
    finally:
        capture.release()
    queue = {
        "version": 1,
        "video_filename": video_path.name,
        "video_sha256": source_id,
        "frame_count": count,
        "fps": fps,
        "stride_seconds": stride_seconds,
        "already_labelled": len(saved),
        "frames": frames,
    }
    write_file(output, (json.dumps(queue, indent=2) + "\n").encode("utf-8"))
    return queue


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a diverse review queue from an MP4.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, help="queue JSON; default: dataset/queues/<video>.json")
    parser.add_argument("--limit", type=int, default=30, help="maximum frames to suggest")
    parser.add_argument("--stride-seconds", type=float, default=2.0, help="candidate sampling interval")
    parser.add_argument("--min-gap-seconds", type=float, default=1.0, help="distance from labelled frames")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float)
    args = parser.parse_args()
    if (args.video.suffix.lower() != ".mp4" or args.limit < 1 or
            not all(math.isfinite(value) and value >= 0 for value in
                    (args.stride_seconds, args.min_gap_seconds, args.start_seconds)) or
            args.stride_seconds == 0 or
            (args.end_seconds is not None and
             (not math.isfinite(args.end_seconds) or args.end_seconds <= args.start_seconds))):
        parser.error("Use an MP4, a positive limit and stride, and a valid time window.")
    output = args.output or args.dataset / "queues" / f"{args.video.stem}.json"
    try:
        queue = build_queue(args.video, args.dataset, output, args.limit, args.stride_seconds,
                            args.min_gap_seconds, args.start_seconds, args.end_seconds)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(queue['frames'])} suggested frames to {output}")
    if queue["frames"]:
        print("Open with: python annotate.py", args.video, "--dataset", args.dataset, "--queue", output)
    else:
        print("No unlabelled candidates remain in the selected time window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
