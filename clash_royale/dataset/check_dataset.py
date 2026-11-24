import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from .prepare_dataset import pixel_digest, read_image, read_metadata, sample_keys
from .replay_dataset import TROOP_CLASSES, read_boxes, read_classes, training_class_ids, write_file


def flag(frame: dict, kind: str, message: str) -> None:
    frame["issues"].append({"kind": kind, "message": message})


def flag_pair(frame: dict, other: dict, kind: str, message: str) -> None:
    flag(frame, kind, f"{message} {other['key']}")
    flag(other, kind, f"{message} {frame['key']}")


def nearest_frame(thumbnail: np.ndarray, candidates: list) -> dict | None:
    best, best_difference = None, float("inf")
    for reference, frame in candidates:
        difference = np.abs(thumbnail - reference)
        average = float(difference.mean())
        if average <= 3 and average < best_difference and (difference.max(axis=2) > 12).mean() <= 0.05:
            best, best_difference = frame, average
    return best


def audit_dataset(root: Path, troop_only: bool) -> dict:
    classes = read_classes(root / "classes.txt")
    class_ids = training_class_ids(classes, troop_only)
    names = list(TROOP_CLASSES) if troop_only else classes
    frames = []
    sources, exact_pixels = {}, {}
    thumbnails = defaultdict(list)
    complete_frames = defaultdict(list)
    match_counts = defaultdict(Counter)
    for key in sample_keys(root):
        match = key.parts[0]
        frame = {"key": key.as_posix(), "match_id": match, "issues": []}
        frames.append(frame)
        counts = match_counts[match]
        if len(key.parts) != 2:
            flag(frame, "invalid_export", "Expected a file directly inside a match folder.")
            continue

        info = {}
        try:
            info, _ = read_metadata(root / "metadata" / key.with_suffix(".json"), match)
            identity = info["video_sha256"].lower(), info["frame_index"]
            frame.update(video_sha256=identity[0], frame_index=identity[1], video_filename=info.get("video_filename"))
            previous = sources.setdefault(identity, frame)
            if previous is not frame:
                flag_pair(frame, previous, "duplicate_source", "Same source video and frame index as")
        except (OSError, ValueError) as error:
            flag(frame, "invalid_export", f"Metadata: {error}")

        height, width = info.get("height"), info.get("width")
        try:
            image = read_image(root / "images" / key.with_suffix(".png"))
            if info and (height, width) != image.shape[:2]:
                flag(frame, "invalid_export", "Metadata dimensions disagree with the PNG.")
            height, width = image.shape[:2]
            previous = exact_pixels.setdefault(pixel_digest(image), frame)
            if previous is not frame:
                flag_pair(frame, previous, "duplicate_pixels", "Identical pixels to")
            else:
                thumbnail = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16)
                candidates = thumbnails[image.shape]
                near = nearest_frame(thumbnail, candidates)
                if near is not None:
                    flag_pair(frame, near, "near_duplicate", "Very similar to")
                candidates.append((thumbnail, frame))
        except (OSError, ValueError, cv2.error) as error:
            flag(frame, "invalid_export", f"Image: {error}")

        try:
            labels = (root / "labels" / key.with_suffix(".txt")).read_bytes()
        except OSError as error:
            flag(frame, "invalid_export", f"Labels: {error}")
            continue
        if not labels.strip():
            flag(frame, "empty_labels", "Annotation file is empty. Check whether this is a background frame.")
            continue
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            continue
        boxes, errors = read_boxes(labels, len(classes), width, height)
        for error in errors:
            flag(frame, "invalid_box", error)
        selected = Counter(box.class_id for box in boxes if box.class_id in class_ids)
        counts.update(selected)
        count = selected.total()
        if not errors and count:
            complete_frames[match].append((frame, count))
        elif not errors:
            flag(frame, "no_target_boxes", "No troop boxes. Check for unlabeled troops.")

    for entries in complete_frames.values():
        typical = median(count for _, count in entries)
        for frame, count in entries:
            if count <= typical / 2:
                flag(frame, "few_boxes", f"{count} target boxes. Median on nonempty frames in this match is {typical:g}.")

    def class_summary(counts: Counter) -> dict:
        boxes = {name: counts[class_id] for name, class_id in zip(names, class_ids)}
        largest = max(boxes.values())
        return {
            "boxes_per_class": boxes,
            "underrepresented_classes": [
                name for name, count in boxes.items() if count == 0 or count < largest / 4
            ],
        }

    totals = Counter()
    for counts in match_counts.values():
        totals.update(counts)
    flagged = [frame for frame in frames if frame["issues"]]
    return {
        "dataset_root": str(root),
        "troop_only": troop_only,
        "frames_checked": len(frames),
        "issue_counts": dict(Counter(issue["kind"] for frame in flagged for issue in frame["issues"])),
        **class_summary(totals),
        "by_match": {match: class_summary(counts) for match, counts in sorted(match_counts.items())},
        "flagged_frames": flagged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Flag exported frames and label problems for manual review.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="annotation dataset root")
    parser.add_argument("--troop-only", action="store_true", help="count only troop boxes for label coverage and balance")
    args = parser.parse_args()
    try:
        root = args.dataset.resolve()
        report = audit_dataset(root, args.troop_only)
        output = root / "quality.json"
        write_file(output, (json.dumps(report, indent=2) + "\n").encode("utf-8"))
        print(f"Flagged {len(report['flagged_frames'])} of {report['frames_checked']} frames. Report: {output}")
        for kind, count in sorted(report["issue_counts"].items()):
            print(f"  {kind}: {count}")
        for name, count in report["boxes_per_class"].items():
            print(f"  {name}: {count} boxes")
        underrepresented = report["underrepresented_classes"]
        if underrepresented:
            print("Underrepresented classes: " + ", ".join(underrepresented))
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
