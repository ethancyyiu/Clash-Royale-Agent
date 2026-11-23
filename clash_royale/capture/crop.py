import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from ..dataset.replay_dataset import read_capture_timestamps
from .recording import video_writer
from .region_file import load_region
from .video import open_video, read_frame


def select_crop(source: Path, index: int) -> list[int]:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise OSError("Crop selection needs a desktop display. Use --region-file to crop without a window.")
    frame = read_frame(source, index)
    height, width = frame.shape[:2]
    scale = min(1.0, 1100 / width, 760 / height)
    view_width, view_height = max(1, round(width * scale)), max(1, round(height * scale))
    preview = cv2.resize(frame, (view_width, view_height), interpolation=cv2.INTER_AREA)
    window = "Select game region"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        x, y, w, h = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(window)
    if w == 0 or h == 0:
        raise ValueError("Crop selection cancelled.")
    left, right = round(x * width / view_width), round((x + w) * width / view_width)
    top, bottom = round(y * height / view_height), round((y + h) * height / view_height)
    return [left, top, right - left, bottom - top]


def publish_files(files: list[tuple[Path, Path]]) -> None:
    published = []
    try:
        for staged, output in files:
            output.parent.mkdir(parents=True, exist_ok=True)
            with staged.open("rb") as source, output.open("xb") as target:
                published.append(output)
                shutil.copyfileobj(source, target)
    except (OSError, KeyboardInterrupt):
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def crop_video(source: Path, output: Path, region: list[int], save_region: Path | None) -> int:
    if output.suffix.lower() != ".mp4":
        raise ValueError("The output filename must end in .mp4.")
    if save_region is not None and save_region.suffix.lower() != ".json":
        raise ValueError("--save-region requires a .json filename.")
    sidecar = output.with_suffix(".timestamps.csv")
    for path in (output, sidecar, save_region):
        if path is not None and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"Output already exists: {path}")
    timestamps = read_capture_timestamps(source)
    has_sidecar = source.with_suffix(".timestamps.csv").exists()
    x, y, width, height = region
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        directory = Path(staging)
        staged_video = directory / "crop.mp4"
        count = 0
        with open_video(source) as (video, fps):
            expected = video.get(cv2.CAP_PROP_FRAME_COUNT)
            ok, image = video.read()
            if not ok:
                raise OSError(f"Could not decode the first frame of {source}")
            frame_height, frame_width = image.shape[:2]
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame_width or y + height > frame_height:
                raise ValueError("The crop must be nonempty and fit inside the source frame.")
            with video_writer(staged_video, fps, width, height) as write:
                while ok:
                    if image.shape[:2] != (frame_height, frame_width):
                        raise ValueError(f"Frame {count} changed dimensions while cropping.")
                    write(image[y:y + height, x:x + width])
                    count += 1
                    ok, image = video.read()
        if (expected > 0 and count != expected) or (timestamps and max(timestamps) >= count):
            raise ValueError("Decoded frame count disagrees with the source video or timestamp CSV.")
        with open_video(staged_video) as (video, output_fps):
            if video.get(cv2.CAP_PROP_FRAME_COUNT) != count or not math.isclose(output_fps, fps, abs_tol=0.001):
                raise OSError("The encoder could not preserve the source frame count and nominal FPS.")

        files = []
        if has_sidecar:
            staged_csv = directory / "timestamps.csv"
            with staged_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("frame_index", "capture_monotonic_ns"))
                writer.writerows(sorted(timestamps.items()))
            files.append((staged_csv, sidecar))
        if save_region is not None:
            staged_region = directory / "region.json"
            staged_region.write_text(json.dumps({"region": region}, indent=2) + "\n", encoding="utf-8")
            files.append((staged_region, save_region))
        files.append((staged_video, output))
        publish_files(files)
    print(f"Cropped {count} frames at {fps:g} FPS to {output}")
    print("Capture timestamps saved to " + str(sidecar) if has_sidecar else "No source capture timestamp CSV found.")
    if save_region is not None:
        print(f"Capture region saved to {save_region}")
    print("--region " + " ".join(map(str, region)))
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop an MP4 to a rectangle selected on a video frame.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="new cropped MP4")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frame", type=int, default=0, help="zero-based frame to show for selection")
    selection.add_argument("--region-file", type=Path, help="reuse a saved rectangle without opening a window")
    parser.add_argument("--save-region", type=Path, help="save the selected rectangle to a new JSON file")
    args = parser.parse_args()
    try:
        region = load_region(args.region_file) if args.region_file else select_crop(args.source, args.frame)
        crop_video(args.source, args.output, region, args.save_region)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
